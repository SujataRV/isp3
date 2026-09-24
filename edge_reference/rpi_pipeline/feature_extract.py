"""
feature_extract.py
------------------
Extracts the same 20-dimensional feature vector used during training.
Mirrors the logic from the Google Colab notebook exactly.
"""

import numpy as np
from config import SNR_THRESHOLD, NUM_FEATURES, FRAME_DT


def extract_frame_features(point_cloud, track_data, height_data, prev_velocity=None):
    """
    Extract 20-dim feature vector from a single radar frame.

    Features:
      0-11: PointCloud (x, y, z, vx, vy, vz, ax, ay, az, n_points, spread_xy, height_range)
      12-17: TrackData (track_x, track_y, track_z, track_vx, track_vy, track_vz)
      18-19: HeightData (person_height, bottom_height)
    """
    # 1) Point Cloud Features (12)
    if not point_cloud or len(point_cloud) == 0:
        pc_feat = np.zeros(12, dtype=np.float32)
        current_velocity = np.zeros(3, dtype=np.float32)
    else:
        pts = np.array(point_cloud, dtype=np.float32)
        if pts.shape[1] > 4:
            snr_mask = pts[:, 4] >= SNR_THRESHOLD
            if snr_mask.sum() > 0:
                pts = pts[snr_mask]

        n_points = len(pts)
        if n_points == 0:
            pc_feat = np.zeros(12, dtype=np.float32)
            current_velocity = np.zeros(3, dtype=np.float32)
        else:
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            doppler = pts[:, 3] if pts.shape[1] > 3 else np.zeros(n_points)

            x_mean, y_mean, z_mean = x.mean(), y.mean(), z.mean()
            r = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2) + 1e-8
            vx_mean = doppler.mean() * (x_mean / r)
            vy_mean = doppler.mean() * (y_mean / r)
            vz_mean = doppler.mean() * (z_mean / r)
            current_velocity = np.array([vx_mean, vy_mean, vz_mean], dtype=np.float32)

            acc = (current_velocity - prev_velocity) / FRAME_DT if prev_velocity is not None else np.zeros(3, dtype=np.float32)

            spread_xy = float(np.sqrt(x.var() + y.var())) if n_points > 1 else 0.0
            height_range = float(z.max() - z.min()) if n_points > 1 else 0.0

            pc_feat = np.array([
                x_mean, y_mean, z_mean,
                vx_mean, vy_mean, vz_mean,
                acc[0], acc[1], acc[2],
                float(n_points), spread_xy, height_range
            ], dtype=np.float32)

    # 2) Track Features (6)
    track_feat = np.zeros(6, dtype=np.float32)
    if track_data and len(track_data) > 0:
        td = track_data[0]
        if len(td) >= 7:
            track_feat = np.array(td[1:7], dtype=np.float32)

    # 3) Height Features (2)
    height_feat = np.zeros(2, dtype=np.float32)
    if height_data and len(height_data) > 0:
        hd = height_data[0]
        if len(hd) >= 3:
            height_feat = np.array(hd[1:3], dtype=np.float32)

    # Combine into 20-dim feature vector and sanitize any inf/nan
    feat = np.concatenate([pc_feat, track_feat, height_feat])
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat, current_velocity


# ── Per-track (per-person) feature extraction ────────────────────────────────
#
# Why this exists
# ---------------
# `extract_frame_features` above collapses EVERY point in the frame into one
# 20-D vector: z_mean is the mean height of all points, n_points is the total,
# height_range is max(z)-min(z) across everyone. That is correct only while
# exactly one person is in view.
#
# With two people it silently breaks fall detection, because all three of the
# rule detector's votes are computed from those mixed statistics:
#   * z_mean       — a 1.0 m drop by one of two people moves the mean ~0.5 m,
#                    so the z_drop >= 0.50 vote stops firing.
#   * height_range — the person still standing keeps max(z) high while the
#                    fallen one lowers min(z), so the range GROWS instead of
#                    collapsing, and the "body flat" vote inverts.
#   * n_points     — the standing person's points keep the total well above
#                    the "few points" threshold.
# Measured on a real recording: a fall that fires 3 times alone fires once
# with one extra standing person, and drops to zero as the scene gets busier.
#
# It also took track_data[0] / height_data[0] — the FIRST track in the TLV,
# whose ordering is not stable — so those 8 features jittered between people.
#
# The fix is to compute one feature vector per tracked person, from that
# person's own points and their own track/height rows, and let the server run
# one detector per person.

def _associate_points_to_tracks(point_cloud, track_indexes, prev_point_cloud):
    """
    Return an (N, >=7) array whose column 6 holds each point's track id, or
    None when the frame carries no usable track association.

    Column 6 is used directly when the parser already filled it in. Otherwise
    the separate Target Index TLV is used — and on the xWR6843 people-tracking
    firmware that TLV labels the PREVIOUS frame's point cloud, arriving one
    frame later, which is why `prev_point_cloud` is needed.
    """
    pc = np.asarray(point_cloud, dtype=np.float32) if point_cloud is not None else None
    if pc is None or pc.size == 0 or pc.ndim != 2:
        return None

    if pc.shape[1] > 6 and np.any(pc[:, 6] != 255):
        return pc

    if track_indexes is None:
        return None
    ti = np.asarray(track_indexes, dtype=np.float32).ravel()
    if ti.size == 0:
        return None

    def _with_ids(base, ids):
        out = np.zeros((base.shape[0], 7), dtype=np.float32)
        out[:, :min(7, base.shape[1])] = base[:, :min(7, base.shape[1])]
        out[:, 6] = ids
        return out

    if prev_point_cloud is not None and prev_point_cloud.shape[0] == ti.size:
        return _with_ids(prev_point_cloud, ti)
    if pc.shape[0] == ti.size:
        return _with_ids(pc, ti)
    return None


def _row_for_tid(rows, tid):
    """Pick the track/height row belonging to `tid` (column 0 is the id)."""
    if not rows:
        return None
    for r in rows:
        try:
            if int(r[0]) == int(tid):
                return r
        except (TypeError, ValueError, IndexError):
            continue
    return None


def _pc_features(pts, prev_velocity):
    """The same 12 point-cloud features as extract_frame_features, over `pts`."""
    if pts is None or len(pts) == 0:
        return np.zeros(12, dtype=np.float32), np.zeros(3, dtype=np.float32)

    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[1] > 4:
        snr_mask = pts[:, 4] >= SNR_THRESHOLD
        if snr_mask.sum() > 0:
            pts = pts[snr_mask]

    n_points = len(pts)
    if n_points == 0:
        return np.zeros(12, dtype=np.float32), np.zeros(3, dtype=np.float32)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    doppler = pts[:, 3] if pts.shape[1] > 3 else np.zeros(n_points, dtype=np.float32)

    x_mean, y_mean, z_mean = x.mean(), y.mean(), z.mean()
    r = np.sqrt(x_mean ** 2 + y_mean ** 2 + z_mean ** 2) + 1e-8
    dop_mean = doppler.mean()
    current_velocity = np.array(
        [dop_mean * (x_mean / r), dop_mean * (y_mean / r), dop_mean * (z_mean / r)],
        dtype=np.float32,
    )
    acc = ((current_velocity - prev_velocity) / FRAME_DT
           if prev_velocity is not None else np.zeros(3, dtype=np.float32))

    spread_xy = float(np.sqrt(x.var() + y.var())) if n_points > 1 else 0.0
    height_range = float(z.max() - z.min()) if n_points > 1 else 0.0

    return np.array([
        x_mean, y_mean, z_mean,
        current_velocity[0], current_velocity[1], current_velocity[2],
        acc[0], acc[1], acc[2],
        float(n_points), spread_xy, height_range,
    ], dtype=np.float32), current_velocity


class PerTrackFeatureExtractor:
    """
    Stateful per-person feature extractor.

    Call `extract()` once per radar frame; it returns {track_id: feat_20},
    one 20-D vector per tracked person, each built only from that person's
    own points, track row, and height row. Velocity/acceleration history is
    kept per track so one person's motion never leaks into another's features.

    When a frame carries no usable track association it falls back to the
    original whole-frame behaviour and returns {None: feat_20}, so callers
    keep working on data recorded without the tracker running.
    """

    def __init__(self, stale_after_frames: int = 120):
        self._prev_velocity = {}     # tid -> np.array(3,)
        self._last_seen = {}         # tid -> frame counter
        self._prev_pc = None
        self._frame_i = 0
        self._stale_after = stale_after_frames

    def reset(self):
        self._prev_velocity.clear()
        self._last_seen.clear()
        self._prev_pc = None
        self._frame_i = 0

    def extract(self, point_cloud, track_data, height_data, track_indexes=None):
        self._frame_i += 1

        labeled = _associate_points_to_tracks(point_cloud, track_indexes, self._prev_pc)

        pc_now = np.asarray(point_cloud, dtype=np.float32) if point_cloud is not None else None
        if pc_now is not None and pc_now.size and pc_now.ndim == 2:
            self._prev_pc = pc_now

        # No track association available — behave exactly as before.
        if labeled is None:
            feat, vel = extract_frame_features(
                point_cloud, track_data, height_data, self._prev_velocity.get(None)
            )
            self._prev_velocity[None] = vel
            return {None: feat}

        # Every id the tracker reports this frame, plus any id that only shows
        # up in trackData (a person the tracker still holds but who returned no
        # points this frame — they must still be reported, with n_points = 0,
        # so the server can tell "no points" apart from "on the floor").
        tids = {int(t) for t in np.unique(labeled[:, 6]) if t < 253}
        for row in (track_data or []):
            try:
                tids.add(int(row[0]))
            except (TypeError, ValueError, IndexError):
                continue

        out = {}
        for tid in sorted(tids):
            pts = labeled[labeled[:, 6] == tid]
            pc_feat, vel = _pc_features(pts, self._prev_velocity.get(tid))
            self._prev_velocity[tid] = vel
            self._last_seen[tid] = self._frame_i

            track_feat = np.zeros(6, dtype=np.float32)
            td = _row_for_tid(track_data, tid)
            if td is not None and len(td) >= 7:
                track_feat = np.array(td[1:7], dtype=np.float32)

            height_feat = np.zeros(2, dtype=np.float32)
            hd = _row_for_tid(height_data, tid)
            if hd is not None and len(hd) >= 3:
                height_feat = np.array(hd[1:3], dtype=np.float32)

            feat = np.concatenate([pc_feat, track_feat, height_feat])
            out[tid] = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        # Drop velocity history for people who have been gone a while.
        for tid in [t for t, seen in self._last_seen.items()
                    if self._frame_i - seen > self._stale_after]:
            self._last_seen.pop(tid, None)
            self._prev_velocity.pop(tid, None)

        return out


def extract_recording_features(frames):
    """
    Process all frames in a recording -> (T, 20) feature array.
    """
    features = []
    prev_vel = None
    for frame_obj in frames:
        fd = frame_obj.get("frameData", frame_obj)
        pc = fd.get("pointCloud", [])
        td = fd.get("trackData", [])
        hd = fd.get("heightData", [])
        
        feat, prev_vel = extract_frame_features(pc, td, hd, prev_vel)
        features.append(feat)
    return np.array(features, dtype=np.float32)


def build_sliding_windows(feature_buffer, window_size, stride):
    """
    Build all complete sliding windows from a growing feature buffer.

    Args:
        feature_buffer: list of np.array(20,) frames accumulated so far
        window_size: int, frames per window
        stride: int, frames to slide

    Returns:
        list of np.array(window_size, 20) -- newly extractable windows
    """
    T = len(feature_buffer)
    windows = []
    if T < window_size:
        return windows
    for start in range(0, T - window_size + 1, stride):
        window = np.array(feature_buffer[start:start + window_size], dtype=np.float32)
        windows.append(window)
    return windows
