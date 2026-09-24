"""
aws_watcher.py
--------------
Watches a folder for new JSON files from the TI mmWave Visualizer and
streams per-frame features to the API immediately as files appear.

Sends BOTH:
  1. POST /frames/batch       — 20-D fall features (Project 1)
  2. POST /activity/pointcloud/batch — raw point clouds for HAR (Project 2)

Uses inotify (Linux) via the `watchdog` library for instant file detection
— no polling delay. Falls back to fast polling if watchdog is unavailable.

Usage:
    pip install watchdog requests numpy
    python aws_watcher.py /path/to/radar/json/folder

    # Also process files already in the folder:
    python aws_watcher.py /path/to/folder --process-existing

    # Override the API URL (local testing):
    python aws_watcher.py /path/to/folder --api-url http://127.0.0.1:8000/frame

Stop with Ctrl-C.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import gzip
import json
import time
import glob
import argparse
import threading
import queue
from datetime import datetime, timezone

import numpy as np
import requests

from config import CLOUD_API_URL, DEVICE_ID, CLOUD_TIMEOUT, WATCH_DIR
from feature_extract import extract_frame_features, PerTrackFeatureExtractor


# Uploading raw point clouds costs about 280 KB per second of radar, which is
# more than a typical home uplink can sustain (measured 0.2-3 Mbit/s on the
# test link). Both knobs below cut that down; the backend understands gzip
# request bodies and falls back cleanly if it does not.
GZIP_UPLOAD    = os.getenv("GZIP_UPLOAD", "true").strip().lower() in ("1", "true", "yes")
UPLOAD_ROUND_DP = int(os.getenv("UPLOAD_ROUND_DP", "3"))   # mm precision; 0 disables


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _round_points(pc):
    """Trim radar coordinates to millimetre precision to shrink the upload."""
    if not pc or UPLOAD_ROUND_DP <= 0:
        return pc or []
    dp = UPLOAD_ROUND_DP
    out = []
    for row in pc:
        try:
            out.append([round(float(v), dp) for v in row])
        except (TypeError, ValueError):
            out.append(row)
    return out


def post_json(session, url, payload, timeout):
    """POST JSON, gzip-compressed when enabled, plain if the server refuses."""
    if not GZIP_UPLOAD:
        return session.post(url, json=payload, timeout=timeout)
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    resp = session.post(url, data=gzip.compress(raw, 6), timeout=timeout,
                        headers={"Content-Type": "application/json",
                                 "Content-Encoding": "gzip"})
    if resp.status_code in (400, 411, 415, 501):
        # Backend too old to unpack gzip — resend uncompressed.
        return session.post(url, json=payload, timeout=timeout)
    return resp


def normalize_frame(fd):
    """
    Normalize a frameData dict into a consistent structure, preserving
    all fields needed by both the fall detector and the HAR detector.

    Returns a dict with guaranteed keys:
        frameNum, pointCloud, trackData, trackIndexes,
        heightData, numDetectedTracks, numDetectedPoints
    """
    return {
        "frameNum":          fd.get("frameNum", 0),
        "pointCloud":        fd.get("pointCloud", []),
        "trackData":         fd.get("trackData", []),
        "trackIndexes":      fd.get("trackIndexes", []),
        "heightData":        fd.get("heightData", []),
        "numDetectedTracks": int(fd.get("numDetectedTracks", 0)),
        "numDetectedPoints": int(fd.get("numDetectedPoints", 0)),
    }


def read_visualizer_json(json_path: str):
    """Read a TI visualizer JSON file and return a list of normalized frame dicts."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  [watcher] Skipping {os.path.basename(json_path)}: invalid JSON ({e})")
        return []
    except PermissionError:
        print(f"  [watcher] Skipping {os.path.basename(json_path)}: file locked")
        return []

    frames = []

    if isinstance(data, dict) and 'data' in data:
        for row in data['data']:
            fd = row.get("frameData", {})
            frames.append(normalize_frame(fd))
    elif isinstance(data, dict) and 'frameData' in data:
        frames.append(normalize_frame(data['frameData']))
    elif isinstance(data, dict) and 'pointCloud' in data:
        frames.append(normalize_frame(data))
    elif isinstance(data, list):
        for item in data:
            fd = item.get("frameData", item)
            frames.append(normalize_frame(fd))
    else:
        print(f"  [watcher] Unknown JSON format in {os.path.basename(json_path)}")
    return frames


def process_file(json_path, session, feature_extractor, counters, api_base_url):
    """
    Extract features from every frame in a JSON file and POST as two batches:
      1. /frames/batch              — 20-D fall features
      2. /activity/pointcloud/batch — raw point clouds for HAR
    """
    fname  = os.path.basename(json_path)
    frames = read_visualizer_json(json_path)
    if not frames:
        return

    counters['files'] += 1

    # ── Build fall-detection batch payload ────────────────────────────────────
    fall_batch_frames = []
    har_batch_frames  = []

    for frame_dict in frames:
        counters['frames'] += 1
        pc = frame_dict.get("pointCloud", [])
        td = frame_dict.get("trackData",  [])
        hd = frame_dict.get("heightData", [])
        ti = frame_dict.get("trackIndexes", [])
        hc = frame_dict.get("numDetectedTracks", 0)
        fn = frame_dict.get("frameNum", 0)

        # --- Fall features (20-D), one vector PER TRACKED PERSON ---
        # Mixing everyone into a single vector hid falls whenever more than
        # one person was in frame, so each person is now sent separately and
        # scored by their own detector server-side. Frames with no track
        # association yield a single {None: ...} entry — the old behaviour.
        per_track = feature_extractor.extract(pc, td, hd, ti)
        for tid, feat in per_track.items():
            fall_batch_frames.append({
                "features":    feat.tolist(),
                "timestamp":   now_iso(),
                "human_count": hc,
                "track_id":    None if tid is None else int(tid),
                "frame_num":   fn,
            })

        # --- HAR raw point cloud ---
        har_batch_frames.append({
            "frame_num":     fn,
            "point_cloud":   _round_points(pc),                # never send null
            "track_indexes": frame_dict.get("trackIndexes") or [],
        })

    # ── 1) Send fall-detection batch ─────────────────────────────────────────
    fall_batch_url = api_base_url.rstrip('/') + "/frames/batch"
    fall_payload   = {"device_id": DEVICE_ID, "frames": fall_batch_frames}

    t0 = time.perf_counter()
    try:
        resp = post_json(session, fall_batch_url, fall_payload, CLOUD_TIMEOUT * 10)
        counters['sent'] += len(fall_batch_frames)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  [fall]  {fname}  {len(har_batch_frames)} frames / "
              f"{len(fall_batch_frames)} person-rows → {fall_batch_url}  {ms:.0f}ms  "
              f"(total sent: {counters['sent']})")
    except requests.exceptions.RequestException as exc:
        counters['errors'] += 1
        print(f"  [fall]  batch send FAILED: {exc}")

    # ── 2) Send HAR point-cloud batch ────────────────────────────────────────
    har_batch_url = api_base_url.rstrip('/') + "/activity/pointcloud/batch"
    har_payload   = {"device_id": DEVICE_ID, "frames": har_batch_frames}

    t0 = time.perf_counter()
    try:
        resp = post_json(session, har_batch_url, har_payload, CLOUD_TIMEOUT * 10)
        ms = (time.perf_counter() - t0) * 1000
        rj = resp.json() if resp.status_code == 200 else {}
        if resp.status_code != 200:
            # Used to be silent (just printed '?'), which hid a total failure.
            print(f"  [har]   API REJECTED batch: HTTP {resp.status_code} {resp.text[:300]}")
        counters['har_sent'] += len(har_batch_frames)
        print(f"  [har]   {fname}  {len(har_batch_frames)} frames → {har_batch_url}  {ms:.0f}ms  "
              f"buffered={rj.get('buffered', '?')}/{rj.get('window_size', '?')} "
              f"activity={rj.get('activity', '?')} conf={rj.get('confidence', '?')}")
    except requests.exceptions.RequestException as exc:
        counters['har_errors'] += 1
        print(f"  [har]   batch send FAILED: {exc}")


def wait_for_stable_file(path, stable_ms=80, max_wait_s=15):
    """
    Wait until the file size stops changing — i.e. the writer has finished.
    Polls every `stable_ms` milliseconds. Returns True when stable, False on timeout.
    """
    last_size = -1
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
        except OSError:
            time.sleep(0.05)
            continue
        if size > 0 and size == last_size:
            return True   # size unchanged — file is fully written
        last_size = size
        time.sleep(stable_ms / 1000.0)
    return False  # timed out


def run_with_watchdog(watch_dir, session, feature_extractor, counters, seen_files, api_base_url):
    """Use watchdog (inotify on Linux) for zero-delay file detection."""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    file_queue = queue.Queue()

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory and event.src_path.endswith('.json'):
                file_queue.put(event.src_path)

        def on_moved(self, event):
            # Some apps write to .tmp then rename to .json
            if not event.is_directory and event.dest_path.endswith('.json'):
                file_queue.put(event.dest_path)

    observer = Observer()
    observer.schedule(Handler(), watch_dir, recursive=False)
    observer.start()
    print("[watcher] inotify active — zero-delay detection ✓")

    try:
        while True:
            try:
                path = file_queue.get(timeout=1.0)
                if path in seen_files:
                    continue
                seen_files.add(path)
                if wait_for_stable_file(path):
                    process_file(path, session, feature_extractor, counters, api_base_url)
                else:
                    print(f"  [watcher] Timed out waiting for {os.path.basename(path)} — skipping")
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ── Fallback fast-polling watcher ─────────────────────────────────────────────

def run_with_polling(watch_dir, session, feature_extractor, counters, seen_files,
                     poll_interval, api_base_url):
    """Poll every 100ms as a fallback when watchdog isn't available."""
    print(f"[watcher] polling every {poll_interval*1000:.0f}ms (install watchdog for instant detection)")
    try:
        while True:
            current = set(glob.glob(os.path.join(watch_dir, "*.json")))
            for path in sorted(current - seen_files):
                seen_files.add(path)
                time.sleep(0.02)
                process_file(path, session, feature_extractor, counters, api_base_url)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Watch a folder for new JSON files and stream features to the API."
    )
    parser.add_argument("watch_dir", nargs="?", default=WATCH_DIR,
                        help="Folder where TI visualizer saves JSON files "
                             f"(default: WATCH_DIR from config.py = {WATCH_DIR})")
    parser.add_argument("--process-existing", action="store_true",
                        help="Also process JSON files already in the folder on startup")
    parser.add_argument("--poll-interval", type=float, default=0.1,
                        help="Polling fallback interval in seconds (default 0.1)")
    parser.add_argument("--api-url", type=str, default=None,
                        help="Override API base URL (e.g. http://127.0.0.1:8000/frame). "
                             "Defaults to CLOUD_API_URL from config.py.")
    args = parser.parse_args()

    watch_dir = os.path.abspath(args.watch_dir)
    if not os.path.isdir(watch_dir):
        if watch_dir == os.path.abspath(WATCH_DIR):
            # This is the default shared folder with the mmWave Visualizer —
            # create it rather than failing, since it's normal for it not to
            # exist yet on a first run (the visualizer creates it too, but
            # whichever side starts first should not have to error out).
            os.makedirs(watch_dir, exist_ok=True)
            print(f"[watcher] Created watch folder: {watch_dir}")
        else:
            print(f"ERROR: Directory does not exist: {watch_dir}")
            sys.exit(1)

    # ── Derive API base URL ──────────────────────────────────────────────────
    raw_url = args.api_url or CLOUD_API_URL
    # Strip trailing endpoint paths to get the base
    api_base_url = raw_url
    for suffix in ["/frame", "/frames/batch", "/activity/pointcloud/batch",
                   "/activity/pointcloud", "/activity"]:
        if api_base_url.endswith(suffix):
            api_base_url = api_base_url[: -len(suffix)]
            break

    if "YOUR_EC2_IP" in api_base_url:
        print("ERROR: Set CLOUD_API_URL in config.py or use --api-url")
        sys.exit(1)

    print("=" * 60)
    print("  Real-Time Watcher → API (Fall + HAR)")
    print("=" * 60)
    print(f"  Watch dir  : {watch_dir}")
    print(f"  API base   : {api_base_url}")
    print(f"  Fall batch : {api_base_url}/frames/batch")
    print(f"  HAR batch  : {api_base_url}/activity/pointcloud/batch")
    print(f"  Device ID  : {DEVICE_ID}")
    print("=" * 60)

    session          = requests.Session()
    feature_extractor = PerTrackFeatureExtractor()   # holds per-track velocity history
    counters         = {
        'files': 0, 'frames': 0,
        'sent': 0, 'errors': 0,
        'har_sent': 0, 'har_errors': 0,
    }

    # Mark existing files as seen (skip them unless --process-existing)
    seen_files = set()
    existing = set(glob.glob(os.path.join(watch_dir, "*.json")))
    if args.process_existing:
        print(f"[watcher] Processing {len(existing)} existing file(s) first...")
        for path in sorted(existing):
            seen_files.add(path)
            process_file(path, session, feature_extractor, counters, api_base_url)
    else:
        seen_files = existing
        print(f"[watcher] Skipping {len(seen_files)} existing file(s). Waiting for new ones...")

    print("[watcher] Ready — waiting for new JSON files (Ctrl-C to stop)\n")

    # Try inotify first, fall back to polling
    try:
        import watchdog
        run_with_watchdog(watch_dir, session, feature_extractor, counters, seen_files, api_base_url)
    except ImportError:
        print("[watcher] watchdog not installed — using polling fallback")
        print("[watcher] For instant detection: pip install watchdog")
        run_with_polling(watch_dir, session, feature_extractor, counters, seen_files,
                         args.poll_interval, api_base_url)

    print("\n" + "=" * 60)
    print(f"  Files processed     : {counters['files']}")
    print(f"  Total frames        : {counters['frames']}")
    print(f"  Fall frames sent    : {counters['sent']}")
    print(f"  Fall errors         : {counters['errors']}")
    print(f"  HAR frames sent     : {counters['har_sent']}")
    print(f"  HAR errors          : {counters['har_errors']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
