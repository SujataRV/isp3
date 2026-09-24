"""
stream_writer.py — live JSON bridge from the mmWave Visualizer to isp3
=======================================================================
Writes radar frames to a fixed, well-known folder as small JSON files, once
per second, in the exact schema isp3's rpi_pipeline/aws_watcher.py already
expects (the same schema gui_parser.py's archival "Save Data to File"
feature writes, just flushed far more often and to a separate folder).

This is deliberately a *second*, independent writer from the archival one:
  - "Save Data to File" (gui_parser.py)  -> binData/<session timestamp>/
    one big, permanent, user-chosen archive. Keeps growing; the user manages
    it themselves, same as before.
  - "Stream to isp3" (this file)         -> radar_stream/ (fixed path)
    a rolling, disposable bridge folder that isp3's watcher tails in near
    real time. Meant to be started, stopped, and wiped freely — hence the
    stop-and-delete control wired up in gui_activity.py.

Design notes:
  - Flushing is wall-clock based (every ~1s), not frame-count based, so it
    holds regardless of the configured chirp/frame rate.
  - Each file is written to a ".tmp" name first, then atomically renamed to
    ".json" with os.replace(). aws_watcher.py's inotify handler explicitly
    watches for this exact pattern (its on_moved handler comment says
    "Some apps write to .tmp then rename to .json") — so this an intentional
    contract between the two projects, not an implementation detail.
  - The output folder is fixed (MMWAVE visualizer/radar_stream/) precisely so
    that swapping in a different isp3 checkout only ever requires repointing
    one path (rpi_pipeline/config.py's WATCH_DIR) — never touching this file.
"""
from __future__ import annotations

import os
import json
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STREAM_DIR = os.path.join(BASE_DIR, "radar_stream")

FLUSH_INTERVAL_S = 1.0     # target: one JSON file per second
MAX_FRAMES_PER_FILE = 200  # safety cap if frames arrive much faster than expected

# Only these keys matter to isp3's aws_watcher.py / feature_extract.py —
# skip anything else (e.g. 'occupancy', 'vitals') the parser may add.
_STREAM_FIELDS = (
    "frameNum", "pointCloud", "trackData", "trackIndexes",
    "heightData", "numDetectedTracks", "numDetectedPoints",
)


def _to_jsonable(value):
    """Recursively convert numpy arrays/scalars to plain Python types, and
    copy nested lists, so the result has no aliasing with live GUI state and
    no types json.dump() would choke on."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _snapshot_frame(frame_dict: dict) -> dict:
    """Take an immediate, fully-detached copy of the fields isp3 needs."""
    return {k: _to_jsonable(frame_dict.get(k)) for k in _STREAM_FIELDS if k in frame_dict}


class StreamWriter:
    """
    Call push(frame_dict) once per radar frame while active. Internally
    buffers frames and flushes a new JSON file roughly every second.
    """

    def __init__(self, out_dir: str = DEFAULT_STREAM_DIR):
        self.out_dir = out_dir
        self.active = False
        self.cfg: list[str] = []
        self.demo = ""
        self._buffer: list[dict] = []
        self._last_flush = 0.0
        self._seq = 0
        self.files_written = 0
        self.frames_written = 0

    def start(self, cfg: list[str] | None = None, demo: str = "") -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        self.cfg = cfg or []
        self.demo = demo
        self._buffer = []
        self._last_flush = time.time()
        self._seq = 0
        self.files_written = 0
        self.frames_written = 0
        self.active = True

    def stop(self) -> None:
        """Flush whatever is left in the buffer, then go inactive."""
        if self.active and self._buffer:
            self._flush()
        self.active = False

    def push(self, frame_dict: dict) -> None:
        """
        Buffer one frame; flush to disk if ~1s has elapsed.

        IMPORTANT: this snapshots frame_dict's arrays to plain Python lists
        immediately. gui_activity.py's updateGraph() rotates points for tilt
        and adds the sensor height IN PLACE on the same numpy arrays right
        after parsing, for on-screen display — if we merely kept a reference
        here, whatever we buffered would silently turn into the *rotated,
        height-shifted* coordinates by the time _flush() actually serializes
        it (JSON writing is deferred, array mutation is not). isp3's models
        were trained on raw, unrotated sensor-frame coordinates (the same
        thing gui_parser.py's own archival writer saves), so this must too.
        """
        if not self.active:
            return
        self._buffer.append({
            "frameData": _snapshot_frame(frame_dict),
            "timestamp": time.time() * 1000.0,
        })
        now = time.time()
        if (now - self._last_flush) >= FLUSH_INTERVAL_S or len(self._buffer) >= MAX_FRAMES_PER_FILE:
            self._flush()
            self._last_flush = now

    def _flush(self) -> None:
        if not self._buffer:
            return
        payload = {"cfg": self.cfg, "demo": self.demo, "data": self._buffer}
        self._seq += 1
        name = f"stream_{int(time.time() * 1000)}_{self._seq}.json"
        final_path = os.path.join(self.out_dir, name)
        tmp_path = final_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp_path, final_path)   # atomic — matches aws_watcher's expected write pattern
            self.files_written += 1
            self.frames_written += len(self._buffer)
        except Exception as e:
            print(f"[stream_writer] Failed to write {name}: {e}")
        finally:
            self._buffer = []

    def delete_all(self) -> int:
        """
        Delete every JSON file this stream has produced (only .json/.tmp
        files under out_dir — never used for anything else, so this is safe
        to call even while other files might theoretically be present).
        Returns the number of files removed.
        """
        if not os.path.isdir(self.out_dir):
            return 0
        removed = 0
        for name in os.listdir(self.out_dir):
            if name.endswith(".json") or name.endswith(".json.tmp"):
                try:
                    os.remove(os.path.join(self.out_dir, name))
                    removed += 1
                except OSError as e:
                    print(f"[stream_writer] Could not delete {name}: {e}")
        return removed

    def file_count(self) -> int:
        if not os.path.isdir(self.out_dir):
            return 0
        return sum(1 for n in os.listdir(self.out_dir) if n.endswith(".json"))
