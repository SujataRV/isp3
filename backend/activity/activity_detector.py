"""
activity_detector.py
---------------------
Per-device activity (HAR) classifier used by radar-api-main.py.

This used to run its own, simpler pipeline: whatever point cloud a request
carried was fed straight into a 36-feature extractor with no track filtering,
no denoising, and no temporal smoothing. That pipeline was self-consistent
(feature count matched the bundled model), but it diverged from the pipeline
this project actually validated for the same job — the 44-feature,
DBSCAN-denoised, count-invariant extractor with the Micro-Doppler Spatial
Gradient (MDSG) feature, trained and shipped from radar_gui_2person /
MMWAVE visualizer/activity_predictor.py. Two concrete problems this rewrite
fixes:

  1. `track_indexes` was accepted over the wire (ActivityPointCloudBatchFrame
     already had the field) but the API handler never passed it into
     `push()`, so the model was scored on the RAW, unfiltered point cloud —
     every static reflection, wall bounce, and (with more than one person in
     frame) every other person's points included as if they belonged to the
     tracked subject. The trained model expects one person's own points.
  2. The bundled model + feature extractor were an older generation (36
     features, no denoising, no MDSG, no confidence smoothing) than what the
     rest of the project has since moved to. Both endpoints now use the same
     model everywhere it's deployed.

Kept identical: the public `push()` contract (same keys in, same keys out),
so radar-api-main.py's call sites did not need any further changes beyond
passing `track_indexes` through.
"""
from __future__ import annotations
import os
import json
from collections import deque, Counter
from typing import Any
import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'xgboost_model.json')
LABEL_PATH  = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')
CONFIG_PATH = os.path.join(BASE_DIR, 'models', 'model_config.json')

WINDOW_SIZE   = 45     # overridden by model_config.json['window'] at load
CONF_THRESH   = 0.40
VOTE_WINDOW   = 5      # majority vote over the last N predictions
MIN_PTS       = 3      # frames with fewer clean points don't count toward the window
UNASSIGNED    = 255    # TI SDK: point not linked to any track (noise)
RESERVED_MIN  = 253    # 253 weak SNR, 254 out of bounds, 255 noise — never a real person

TELEPORT_THRESHOLD = 0.8   # metres — a bigger single-frame centroid jump resets the window
TRACK_TIMEOUT      = 60    # frames a person can be absent before their state is dropped

# ── DBSCAN denoising params (must match training) ─────────────────────────────
DBSCAN_EPS_BASE    = 0.40
DBSCAN_EPS_SLOPE   = 0.04
DBSCAN_MIN_SAMPLES = 2
DBSCAN_DOP_WEIGHT  = 2.0

DEFAULT_CLASSES = ['running', 'stationary', 'walking']


def _dbscan_clean(pts: np.ndarray):
    """Drop DBSCAN noise points from one person's raw cloud. Adaptive eps
    grows with distance from the radar since far points are naturally sparser."""
    if len(pts) < DBSCAN_MIN_SAMPLES:
        return pts
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return pts

    y_mean = float(np.abs(pts[:, 1]).mean())
    eps = DBSCAN_EPS_BASE + DBSCAN_EPS_SLOPE * max(0.0, y_mean - 2.0)
    feat = np.column_stack([
        pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3] * DBSCAN_DOP_WEIGHT,
    ]).astype(np.float32)
    labels = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES, algorithm='ball_tree').fit_predict(feat)
    inlier_mask = labels != -1
    if inlier_mask.sum() == 0:
        return pts   # everything was noise — fall back to the raw points
    return pts[inlier_mask]


def _compute_mdsg(points_array: np.ndarray):
    """Micro-Doppler Spatial Gradient: how differently nearby points move,
    normalized by distance. High = limbs moving independently (walk/run);
    low = the whole body moving as one block (crouch/stationary)."""
    N = points_array.shape[0]
    if N < 2:
        return 0.0, 0.0, 0.0
    coords = points_array[:, :3]
    vels = points_array[:, 3]
    diff_coords = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff_coords, axis=2)
    diff_vels = np.abs(vels[:, np.newaxis] - vels[np.newaxis, :])
    gradient = diff_vels / (dist_matrix + 0.01)
    triu = np.triu_indices(N, k=1)
    g = gradient[triu]
    if len(g) == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(g)), float(np.percentile(g, 95)), float(np.std(g))


# Must match the trained model — set from model_config.json's "variant".
FEATURE_VARIANT = "rel_z"


def _frame_row(pc: np.ndarray, count_invariant: bool) -> list[float] | None:
    if pc is None or pc.shape[0] < MIN_PTS:
        return None
    x, y, z, dop = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
    snr = pc[:, 4] if pc.shape[1] > 4 else np.full(len(x), 15.0, dtype=np.float32)
    mdsg_mean, mdsg_p95, mdsg_std = _compute_mdsg(pc)
    # Body-relative z: invariant to sensor mounting height (see
    # activity_predictor.py for the full rationale).
    z_feat = float(z.mean()) if FEATURE_VARIANT == "abs_z" else float(z.mean() - z.max())

    row = [
        float(len(pc)), float(dop.mean()), float(dop.std()), float(np.abs(dop).max()),
        float(snr.mean()), float(x.mean()), float(y.mean()), z_feat,
        float(z.max() - z.min()), mdsg_mean, mdsg_p95, mdsg_std,
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


def _features_from_rows(rows: list, count_invariant: bool = True) -> np.ndarray | None:
    stats = [r for r in rows if r is not None]
    if len(stats) < max(3, WINDOW_SIZE // 6):
        return None
    a = np.array(stats, dtype=np.float32)
    fvec = np.concatenate([a.mean(0), a.std(0, ddof=0), a.min(0), a.max(0),
                           np.array(_temporal_feats(stats, count_invariant), dtype=np.float32)])
    if np.any(np.isnan(fvec)) or np.any(np.isinf(fvec)):
        return None
    return fvec.astype(np.float32)


class ActivityDetector:
    """
    One instance is kept per device_id (see radar-api-main.py's
    `get_activity_detector`), so it's safe to hold state across calls:
    the sliding window, the previous frame (for track-index frame-lag
    pairing), and which track is currently "the" tracked person.
    """

    DEFAULT_CLASSES = DEFAULT_CLASSES

    def __init__(self, model_path: str = MODEL_PATH, label_path: str = LABEL_PATH,
                config_path: str = CONFIG_PATH):
        self.model_path = model_path
        self.label_path = label_path
        self._clf = None
        self._enc = None
        self._count_invariant = True
        self._classes = list(DEFAULT_CLASSES)
        # Per-person state: every tracked person gets their own sliding
        # window, vote history and centroid. Sharing one window across people
        # meant only one of them could ever be classified.
        self._tracks: dict[int, dict] = {}
        self._frame_count = 0
        self._primary_tid: int | None = None
        self._prev_pc: np.ndarray | None = None
        self._load_model(config_path)

    def _track_state(self, tid) -> dict:
        st = self._tracks.get(tid)
        if st is None:
            st = {
                'window':   deque(maxlen=WINDOW_SIZE),
                'votes':    deque(maxlen=VOTE_WINDOW),
                'centroid': None,
                'last_seen': self._frame_count,
                'label':    'warming_up',
                'conf':     0.0,
                'probs':    {c: 0.0 for c in self._classes},
            }
            self._tracks[tid] = st
            print(f'[ActivityDetector] New person track {tid}')
        st['last_seen'] = self._frame_count
        return st

    def _drop_stale_tracks(self) -> None:
        for tid in [t for t, st in self._tracks.items()
                    if self._frame_count - st['last_seen'] > TRACK_TIMEOUT]:
            self._tracks.pop(tid, None)

    def _load_model(self, config_path: str) -> None:
        try:
            import joblib
            from xgboost import XGBClassifier
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                self._count_invariant = cfg.get('count_invariant', True)
                global FEATURE_VARIANT, WINDOW_SIZE
                FEATURE_VARIANT = cfg.get('variant', 'abs_z')
                WINDOW_SIZE = int(cfg.get('window', WINDOW_SIZE))
            if os.path.exists(self.model_path) and os.path.exists(self.label_path):
                clf = XGBClassifier()
                clf.load_model(self.model_path)
                enc = joblib.load(self.label_path)
                self._clf = clf
                self._enc = enc
                self._classes = [str(c) for c in enc.classes_]
                print(f'[ActivityDetector] Loaded HAR model from {self.model_path} '
                      f'(count_invariant={self._count_invariant}, variant={FEATURE_VARIANT}, '
                      f'classes={self._classes})')
            else:
                print(f'[ActivityDetector] Model files missing at {self.model_path}')
        except Exception as e:
            print(f'[ActivityDetector] Failed to load model: {e}')

    @property
    def classes(self) -> list[str]:
        return self._classes

    def reset(self) -> None:
        self._tracks.clear()
        self._frame_count = 0
        self._primary_tid = None
        self._prev_pc = None

    # ── main entry point ─────────────────────────────────────────────────────
    def push(self, point_cloud: Any, track_indexes: Any = None) -> dict[str, Any]:
        self._frame_count += 1
        accepted_frame = False
        reason = 'accepted'
        det_id = id(self)

        by_track = self._split_tracks(point_cloud, track_indexes)

        # A frame with no track association comes back as {None: whole frame}.
        # That stand-in is only meaningful while no real person has been
        # tracked. Once real track IDs exist it must never coexist with them:
        # mixing everyone's points into a phantom person is wrong, and the
        # None key next to int keys made sorted() raise TypeError on the
        # second real-radar frame, which 500'd every activity batch.
        if any(t is not None for t in by_track) or any(t is not None for t in self._tracks):
            by_track.pop(None, None)
            self._tracks.pop(None, None)

        if not by_track:
            reason = 'no_valid_point_cloud'
        else:
            for tid, pts in by_track.items():
                if pts.shape[0] < MIN_PTS:
                    continue
                cleaned = _dbscan_clean(pts)
                if cleaned.shape[0] < MIN_PTS:
                    continue
                st = self._track_state(tid)
                centroid = cleaned[:, :2].mean(axis=0)
                if st['centroid'] is not None:
                    jump = float(np.linalg.norm(centroid - st['centroid']))
                    if jump > TELEPORT_THRESHOLD:
                        st['window'].clear()
                        st['votes'].clear()
                        reason = f'teleport_guard_reset (track {tid} jumped {jump:.2f}m)'
                st['centroid'] = centroid
                st['window'].append(_frame_row(cleaned, self._count_invariant))
                accepted_frame = True

        self._drop_stale_tracks()

        # ── classify every tracked person independently ──────────────────────
        people = []
        for tid in sorted(self._tracks, key=lambda t: (t is None, -1 if t is None else t)):
            people.append(self._classify_track(tid))

        # The "primary" person (most points most recently) fills the flat,
        # single-person fields so existing callers keep working unchanged.
        if people:
            ready = [p for p in people if p['status'] == 'ready']
            lead = max(ready, key=lambda p: p['confidence']) if ready else people[0]
        else:
            lead = {
                'track_id': None, 'status': 'warming_up', 'buffered': 0,
                'activity': 'warming_up', 'confidence': 0.0,
                'probs': {c: 0.0 for c in self._classes},
            }
        self._primary_tid = lead['track_id']

        if self._clf is None:
            lead = {**lead, 'status': 'model_not_loaded', 'activity': 'unknown'}

        return {
            'detector_id': det_id,
            'accepted_frame': accepted_frame,
            'reason': reason,
            'status': lead['status'],
            'buffered': lead['buffered'],
            'window_size': WINDOW_SIZE,
            'activity': lead['activity'],
            'confidence': lead['confidence'],
            'probs': lead['probs'],
            'track_id': lead['track_id'],
            # New: one entry per tracked person.
            'people': people,
            'people_count': len(people),
        }

    def _classify_track(self, tid: int) -> dict[str, Any]:
        """Run the model over one person's own sliding window."""
        st = self._tracks[tid]
        buffered = len(st['window'])
        base = {'track_id': tid, 'buffered': buffered, 'window_size': WINDOW_SIZE}

        if self._clf is None:
            return {**base, 'status': 'model_not_loaded', 'activity': 'unknown',
                    'confidence': 0.0, 'probs': {c: 0.0 for c in self._classes}}

        if buffered < WINDOW_SIZE:
            st['label'], st['conf'] = 'warming_up', 0.0
            return {**base, 'status': 'warming_up', 'activity': 'warming_up',
                    'confidence': 0.0, 'probs': {c: 0.0 for c in self._classes}}

        fvec = _features_from_rows(list(st['window']), self._count_invariant)
        if fvec is None:
            return {**base, 'status': 'uncertain', 'activity': 'uncertain',
                    'confidence': 0.0, 'probs': {c: 0.0 for c in self._classes}}

        proba = self._clf.predict_proba(fvec.reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        raw_conf = float(proba[idx])
        raw_label = str(self._enc.classes_[idx]) if raw_conf >= CONF_THRESH else 'uncertain'

        # Temporal majority vote so one noisy frame can't flip the label.
        if raw_label != 'uncertain':
            st['votes'].append(raw_label)
        label = (Counter(st['votes']).most_common(1)[0][0]
                 if st['votes'] else raw_label)
        if raw_conf < CONF_THRESH:
            label = 'uncertain'

        probs_dict = {str(c): round(float(p), 4) for c, p in zip(self._enc.classes_, proba)}
        st['label'], st['conf'], st['probs'] = label, round(raw_conf, 4), probs_dict
        return {**base, 'status': 'ready', 'activity': label,
                'confidence': round(raw_conf, 4), 'probs': probs_dict}

    # ── internals ─────────────────────────────────────────────────────────────
    def _split_tracks(self, point_cloud: Any, track_indexes: Any) -> dict[int, np.ndarray]:
        """
        Split a frame's points into {track_id: that person's points}.

        Returns every tracked person, not just one — this is what allows two
        or more people to be classified at the same time. Frames carrying no
        usable track information fall back to a single {None: all points}
        entry, matching the original whole-frame behaviour, so callers that
        never send track_indexes keep working.
        """
        if point_cloud is None:
            return {}
        arr = np.asarray(point_cloud, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 4:
            return {}

        ti = None
        if track_indexes is not None:
            ti = np.asarray(track_indexes, dtype=np.float32).ravel()

        # Column 6 (if present) is the TI parser's own embedded track index;
        # prefer it when it actually carries real values.
        has_embedded = arr.shape[1] > 6 and np.any(arr[:, 6] != UNASSIGNED)

        labeled = None
        if has_embedded:
            labeled = arr
        elif ti is not None and len(ti) > 0:
            # On the xWR6843 people-tracking firmware, the Target Index TLV
            # labels the PREVIOUS frame's point cloud, one frame later.
            if self._prev_pc is not None and len(ti) == self._prev_pc.shape[0]:
                labeled = np.zeros((self._prev_pc.shape[0], 7), dtype=np.float32)
                labeled[:, :self._prev_pc.shape[1]] = self._prev_pc
                labeled[:, 6] = ti
            elif len(ti) == arr.shape[0]:
                labeled = np.zeros((arr.shape[0], 7), dtype=np.float32)
                labeled[:, :arr.shape[1]] = arr
                labeled[:, 6] = ti
        self._prev_pc = arr

        if labeled is None:
            # No track information at all — treat the frame as one person.
            return {None: arr[:, :5] if arr.shape[1] >= 5 else arr}

        out: dict[int, np.ndarray] = {}
        for t in np.unique(labeled[:, 6]):
            if t >= RESERVED_MIN:      # 253 weak SNR, 254 out of bounds, 255 noise
                continue
            pts = labeled[labeled[:, 6] == t]
            out[int(t)] = pts[:, :min(5, pts.shape[1])]
        return out
