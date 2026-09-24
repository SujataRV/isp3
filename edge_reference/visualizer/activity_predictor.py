"""
activity_predictor.py  —  real-time per-track activity classification
=====================================================================
Adapted from radar_gui_2person/predictor.py so it can be dropped straight
into the TI mmWave Industrial Visualizer.

It takes the exact `outputDict` produced by parseFrame.parseStandardFrame()
(pointCloud Nx7 with col 6 = track index, plus the separate 'trackIndexes'
TLV) and returns, every frame:

    {track_id: (label, confidence)}

Pipeline per track (unchanged from the trained pipeline):
  1. DBSCAN de-noising of the track's point cloud (adaptive eps, doppler
     weighted) — drops ghosts / multipath / cross-track contamination.
  2. Teleport guard      — centroid jump > 0.8 m resets that track's window.
  3. Min-points gate     — frames with < 3 points do not pollute the window.
  4. 30-frame sliding window -> 44-feature vector -> XGBoost.
  5. Confidence gate + temporal majority vote over the last 5 predictions.

Model files are looked up in ./model/ first, then next to this file, then in
../radar_gui_2person/.
"""
from __future__ import annotations

import csv
import json
import os
import warnings
from collections import deque, Counter
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

BASE = os.path.dirname(os.path.abspath(__file__))

# Search order for xgboost_model.json / label_encoder.pkl / model_config.json
MODEL_DIRS = [
    os.path.join(BASE, "model"),
    BASE,
    os.path.abspath(os.path.join(BASE, os.pardir, "radar_gui_2person")),
]

# ── Sliding-window / gating params (must match training) ──────────────────────
WINDOW             = 45   # overridden by model_config.json['window'] at load
CONF_THRESH        = 0.40
MIN_PTS            = 3
TRACK_TIMEOUT      = 60     # frames before a stale track is dropped
TELEPORT_THRESHOLD = 0.8    # metres
UNASSIGNED         = 255

# ── DBSCAN clustering params ─────────────────────────────────────────────────
DBSCAN_EPS_BASE    = 0.40
DBSCAN_EPS_SLOPE   = 0.04
DBSCAN_MIN_SAMPLES = 2
DBSCAN_DOP_WEIGHT  = 2.0

# ── Temporal voting ──────────────────────────────────────────────────────────
VOTE_WINDOW        = 5

# ── Terminal / logging behaviour (GUI friendly: quiet by default) ────────────
VERBOSE            = False   # True -> one line per track per frame
PRINT_EVERY        = 20      # when VERBOSE, only print every Nth frame
LOG_PREDICTIONS    = True    # write eval_logs/eval_<timestamp>.csv
LOG_DIR            = os.path.join(BASE, "eval_logs")

_LOG_COLUMNS = [
    "timestamp", "frame", "track_id", "n_points", "window_fill",
    "pred_label", "confidence",
]


def _find_model_file(name: str) -> str | None:
    for d in MODEL_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _open_log_file():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, "eval_{}.csv".format(ts))
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=_LOG_COLUMNS, extrasaction="ignore")
    w.writeheader()
    f.flush()
    print("[activity] Session log: {}".format(path), flush=True)
    return f, w, path


def _load_model():
    """Load model + label encoder + config. Returns (clf, enc, count_invariant)."""
    try:
        import joblib
        from sklearn.exceptions import InconsistentVersionWarning
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        from xgboost import XGBClassifier

        mpath = _find_model_file("xgboost_model.json")
        lpath = _find_model_file("label_encoder.pkl")
        cpath = _find_model_file("model_config.json")
        if mpath is None or lpath is None:
            print("[activity] ERROR: xgboost_model.json / label_encoder.pkl not found in "
                  + str(MODEL_DIRS))
            return None, None, False

        count_invariant = False
        if cpath:
            with open(cpath) as f:
                cfg = json.load(f)
            count_invariant = cfg.get("count_invariant", False)
            # Take the z-feature variant from the model's own config so the
            # extractor can never silently disagree with the trained model.
            global FEATURE_VARIANT
            FEATURE_VARIANT = cfg.get("variant", "abs_z")
            global WINDOW
            WINDOW = int(cfg.get("window", WINDOW))
            print("[activity] model_config: count_invariant={}, n_features={}, variant={}".format(
                count_invariant, cfg.get("n_features"), FEATURE_VARIANT))
        else:
            print("[activity] WARNING: model_config.json not found, assuming 36-feature model")

        clf = XGBClassifier()
        clf.load_model(mpath)
        enc = joblib.load(lpath)
        print("[activity] Model loaded from {} — classes: {}".format(
            mpath, [str(c) for c in enc.classes_]))
        return clf, enc, count_invariant
    except Exception as e:
        print("[activity] WARNING — model not loaded: {}".format(e))
        print("[activity] (activity labels will show 'model missing'; "
              "run: pip install xgboost scikit-learn joblib)")
        return None, None, False


def _dbscan_clean(pts: np.ndarray):
    """DBSCAN de-noise one track's cloud. Returns (cleaned_pts, n_clusters, n_noise)."""
    if len(pts) < DBSCAN_MIN_SAMPLES:
        return pts, 0, 0
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return pts, 0, 0

    y_mean = float(np.abs(pts[:, 1]).mean())
    eps = DBSCAN_EPS_BASE + DBSCAN_EPS_SLOPE * max(0.0, y_mean - 2.0)

    feat = np.column_stack([
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        pts[:, 3] * DBSCAN_DOP_WEIGHT,
    ]).astype(np.float32)

    labels = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES,
                    algorithm='ball_tree').fit_predict(feat)

    n_noise = int(np.sum(labels == -1))
    n_clusters = len(set(labels) - {-1})

    inlier_mask = labels != -1
    if inlier_mask.sum() == 0:
        return pts, 0, n_noise
    return pts[inlier_mask], n_clusters, n_noise


def compute_mdsg_features(points_array):
    """Micro-Doppler Spatial Gradient features -> (mean, p95, std)."""
    N = points_array.shape[0]
    if N < 2:
        return 0.0, 0.0, 0.0

    coords = points_array[:, :3]
    vels = points_array[:, 3]

    diff_coords = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff_coords, axis=2)
    diff_vels = np.abs(vels[:, np.newaxis] - vels[np.newaxis, :])

    gradient_matrix = diff_vels / (dist_matrix + 0.01)
    triu = np.triu_indices(N, k=1)
    g = gradient_matrix[triu]
    if len(g) == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(g)), float(np.percentile(g, 95)), float(np.std(g))


# Which z feature the loaded model expects. Set from model_config.json's
# "variant" at load time: "rel_z" (mounting-invariant, current model) or
# "abs_z" (legacy absolute z). Training and inference MUST agree here — a
# mismatch is exactly what skewed the old model toward "crouching".
FEATURE_VARIANT = "rel_z"


def _frame_row(pc, count_invariant: bool = False):
    """Per-frame statistics row. Computed once, when the frame enters the
    sliding window, and cached — recomputing the O(N^2) MDSG term for all 30
    frames on every radar frame is what makes the classifier expensive."""
    if pc is None or len(pc) == 0:
        return None
    pc = np.asarray(pc, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] < 4 or pc.shape[0] < MIN_PTS:
        return None
    x, y, z, dop = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
    snr = pc[:, 4] if pc.shape[1] > 4 else np.full(len(x), 15.0)

    mdsg_mean, mdsg_p95, mdsg_std = compute_mdsg_features(pc)

    # Body-relative z (centre of mass below the person's own highest point)
    # instead of absolute height, so re-mounting the sensor — or a recording
    # saved in world coordinates — cannot shift the feature.
    z_feat = float(z.mean()) if FEATURE_VARIANT == "abs_z" else float(z.mean() - z.max())

    row = [
        float(len(pc)),
        float(dop.mean()),
        float(dop.std()),
        float(np.abs(dop).max()),
        float(snr.mean()),
        float(x.mean()),
        float(y.mean()),
        z_feat,
        float(z.max() - z.min()),
        mdsg_mean,
        mdsg_p95,
        mdsg_std,
    ]
    if count_invariant:
        row = row[1:]
    return row


# ── Window-level temporal structure (MUST be identical in training and
# inference — see check_feature_parity.py) ───────────────────────────────────
# The mean/std/min/max aggregation throws away WHEN things happened, so
# "walked the whole time" and "walked then stopped" collapse to similar
# vectors. That was the single biggest confusion (walk_and_stop -> stationary).
# These six features put the temporal shape back.
MOTION_TH_LO = 0.15   # m/s — frame counts as moving
MOTION_TH_HI = 0.40   # m/s — frame counts as briskly moving


def _temporal_feats(rows, count_invariant: bool):
    """Per-frame speed proxy is |doppler|.max, whose column shifts by one when
    n_points is dropped for count invariance."""
    idx = 2 if count_invariant else 3
    spd = np.array([r[idx] for r in rows if r is not None], dtype=np.float32)
    if spd.size == 0:
        return [0.0] * 6
    mov = spd > MOTION_TH_LO
    longest = cur = trans = 0
    for i, m in enumerate(mov):
        cur = 0 if m else cur + 1
        longest = max(longest, cur)
        if i and mov[i] != mov[i - 1]:
            trans += 1
    n = float(spd.size)
    return [
        float(mov.mean()),                        # share of frames moving
        float((spd > MOTION_TH_HI).mean()),       # share moving briskly
        longest / n,                              # longest still stretch
        trans / n,                                # start/stop rate
        float(np.percentile(spd, 10)),            # slowest moments
        float(np.percentile(spd, 90)),            # fastest moments
    ]


def _features_from_rows(rows: list, count_invariant: bool = True):
    """Aggregate + temporal features for one window."""
    stats = [r for r in rows if r is not None]
    if len(stats) < max(3, WINDOW // 6):
        return None
    a = np.array(stats, dtype=np.float32)
    fvec = np.concatenate([a.mean(0), a.std(0, ddof=0), a.min(0), a.max(0),
                           np.array(_temporal_feats(stats, count_invariant), dtype=np.float32)])
    return None if (np.any(np.isnan(fvec)) or np.any(np.isinf(fvec))) else fvec.astype(np.float32)


def _extract_features(frames: list, count_invariant: bool = False):
    """48-feature (count_invariant=False) or 44-feature (True) vector."""
    return _features_from_rows([_frame_row(pc, count_invariant) for pc in frames], count_invariant)


class MultiPersonPredictor:
    """
    Per-track sliding-window activity predictor.
    Call push(outputDict) once per radar frame with the RAW parser output
    (before any tilt-rotation / sensor-height shift is applied).
    """

    def __init__(self):
        self._clf, self._enc, self._count_invariant = _load_model()
        self._windows: dict = {}
        self._last_seen: dict = {}
        self._labels: dict = {}
        self._centroids: dict = {}
        self._vote_history: dict = {}
        self._last_proba: dict = {}
        self._last_raw_pts: dict = {}
        self._last_clusters: dict = {}
        self._frame_count = 0
        self._prev_pc = None

        self._log_file = None
        self._log_writer = None
        self._log_path = None
        if LOG_PREDICTIONS:
            try:
                self._log_file, self._log_writer, self._log_path = _open_log_file()
            except Exception as e:
                print("[activity] could not open log file: {}".format(e))

    # ── public API ────────────────────────────────────────────────────────
    @property
    def model_ready(self) -> bool:
        return self._clf is not None

    @property
    def classes(self) -> list:
        return [str(c) for c in self._enc.classes_] if self._enc is not None else []

    @property
    def active_tracks(self) -> list:
        return list(self._windows.keys())

    def push(self, output_dict: dict) -> dict:
        """Feed one frame. Returns {track_id: (label, confidence)}."""
        self._update_windows(output_dict)
        self._drop_stale_tracks()
        results = self._predict_all()
        self._frame_count += 1
        return results

    def probabilities(self, tid: int) -> dict:
        """Last per-class probability dict for a track (may be empty)."""
        return self._last_proba.get(tid, {})

    def window_fill(self, tid: int) -> float:
        w = self._windows.get(tid)
        if not w:
            return 0.0
        return len([f for f in w if f is not None]) / float(WINDOW)

    def reset(self) -> None:
        """Forget all tracks (call on disconnect / restart)."""
        for d in (self._windows, self._last_seen, self._labels, self._centroids,
                  self._vote_history, self._last_proba,
                  self._last_raw_pts, self._last_clusters):
            d.clear()
        self._prev_pc = None
        self._frame_count = 0

    def close_log(self):
        if self._log_file and not self._log_file.closed:
            self._log_file.flush()
            self._log_file.close()
            print("[activity] Log closed: {}".format(self._log_path), flush=True)
        return self._log_path

    def __del__(self):
        try:
            self.close_log()
        except Exception:
            pass

    # ── internals ─────────────────────────────────────────────────────────
    def _update_windows(self, output_dict: dict) -> None:
        pc = output_dict.get("pointCloud")
        if pc is None:
            return
        arr = np.asarray(pc, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 7:
            return

        # The Target Index TLV is parsed into 'trackIndexes' and, on xWR6843
        # people tracking, labels the PREVIOUS frame's point cloud.
        ti = output_dict.get("trackIndexes")
        labeled = None
        if np.any(arr[:, 6] != UNASSIGNED):
            labeled = arr
        elif ti is not None:
            ti = np.asarray(ti, dtype=np.float32).ravel()
            if self._prev_pc is not None and len(ti) == self._prev_pc.shape[0]:
                labeled = self._prev_pc.copy()
                labeled[:, 6] = ti
            elif len(ti) == arr.shape[0]:
                labeled = arr.copy()
                labeled[:, 6] = ti
        self._prev_pc = arr
        if labeled is None:
            return
        arr = labeled

        for tid in np.unique(arr[:, 6]).astype(int):
            if tid >= 253:      # 253 weak SNR, 254 out of bounds, 255 noise
                continue

            person_pts = arr[arr[:, 6] == tid, :6]
            raw_count = person_pts.shape[0]

            if raw_count < MIN_PTS:
                if tid in self._windows:
                    self._windows[tid].append(None)
                    self._last_seen[tid] = self._frame_count
                continue

            cleaned_pts, n_clusters, _ = _dbscan_clean(person_pts)
            self._last_raw_pts[tid] = raw_count
            self._last_clusters[tid] = n_clusters

            centroid = cleaned_pts[:, :2].mean(axis=0)
            if tid in self._centroids:
                dist = float(np.linalg.norm(centroid - self._centroids[tid]))
                if dist > TELEPORT_THRESHOLD:
                    self._windows[tid] = deque(maxlen=WINDOW)
                    self._vote_history[tid] = deque(maxlen=VOTE_WINDOW)
                    self._labels[tid] = ("collecting...", 0.0)
            self._centroids[tid] = centroid

            if tid not in self._windows:
                self._windows[tid] = deque(maxlen=WINDOW)
                self._vote_history[tid] = deque(maxlen=VOTE_WINDOW)
                self._labels[tid] = ("collecting...", 0.0)
                print("[activity] New track {} — warming up".format(tid), flush=True)

            # cache the per-frame stats row next to the points
            self._windows[tid].append(
                (cleaned_pts, _frame_row(cleaned_pts, self._count_invariant)))
            self._last_seen[tid] = self._frame_count

    def _drop_stale_tracks(self) -> None:
        stale = [tid for tid, last in self._last_seen.items()
                 if self._frame_count - last > TRACK_TIMEOUT]
        for tid in stale:
            for d in (self._windows, self._last_seen, self._labels, self._centroids,
                      self._vote_history, self._last_proba,
                      self._last_raw_pts, self._last_clusters):
                d.pop(tid, None)

    def _predict_all(self) -> dict:
        if self._clf is None:
            return {tid: ("model missing", 0.0) for tid in self._windows}

        classes = self.classes

        for tid, window in self._windows.items():
            valid_frames = [f for f in window if f is not None]
            if len(window) < WINDOW or len(valid_frames) < max(3, WINDOW // 6):
                self._labels[tid] = ("collecting...", 0.0)
                continue

            fvec = _features_from_rows([e[1] for e in valid_frames], self._count_invariant)
            if fvec is None:
                self._labels[tid] = ("uncertain", 0.0)
                continue

            proba = self._clf.predict_proba(fvec.reshape(1, -1))[0]
            idx = int(np.argmax(proba))
            conf = float(proba[idx])

            raw_label = str(self._enc.classes_[idx]) if conf >= CONF_THRESH else "uncertain"

            if tid not in self._vote_history:
                self._vote_history[tid] = deque(maxlen=VOTE_WINDOW)
            if raw_label not in ("collecting...", "uncertain"):
                self._vote_history[tid].append(raw_label)

            voted = (Counter(self._vote_history[tid]).most_common(1)[0][0]
                     if self._vote_history[tid] else raw_label)
            label = voted if conf >= CONF_THRESH else "uncertain"
            self._labels[tid] = (label, conf)
            self._last_proba[tid] = {c: float(proba[i]) for i, c in enumerate(classes)}

            if self._log_writer is not None:
                self._log_writer.writerow({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "frame": self._frame_count,
                    "track_id": tid,
                    "n_points": int(np.mean([len(e[0]) for e in valid_frames])),
                    "window_fill": round(len(valid_frames) / float(WINDOW), 2),
                    "pred_label": label,
                    "confidence": round(conf, 4),
                })
                self._log_file.flush()

        if VERBOSE and self._frame_count % PRINT_EVERY == 0:
            for tid, (label, conf) in self._labels.items():
                print("[activity][f{:>6}] track {}: {:<14} {:5.1f}%".format(
                    self._frame_count, tid, label.upper(), conf * 100), flush=True)

        return dict(self._labels)
