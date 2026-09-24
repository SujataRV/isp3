import os
import gzip
import time
from datetime import datetime, timezone
from decimal import Decimal
from collections import deque

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
import threading

from activity import ActivityDetector

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
TABLE_NAME = os.getenv("RADAR_EVENTS_TABLE", "radar_events")

# ── DynamoDB is optional ─────────────────────────────────────────────────────
# Local/offline testing (this repo's whole point) must not require AWS
# credentials, a reachable EC2 instance, or even the boto3 package to be
# installed. Every AWS call in this file is already wrapped in try/except so
# a live server degrades gracefully if a request fails — but importing boto3
# and building the resource used to happen unconditionally at module load, so
# just running `uvicorn radar-api-main:app` locally without `pip install
# boto3` crashed before a single request could be served. Set
# USE_DYNAMODB=true (with real AWS credentials) to restore the original
# always-on behaviour when this actually deploys to EC2.
USE_DYNAMODB = os.getenv("USE_DYNAMODB", "false").strip().lower() in ("1", "true", "yes")

table = None
Key = None
if USE_DYNAMODB:
    try:
        import boto3
        from boto3.dynamodb.conditions import Key
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        table = dynamodb.Table(TABLE_NAME)
        print(f"[DynamoDB] Enabled — table={TABLE_NAME} region={AWS_REGION}")
    except Exception as e:
        print(f"[DynamoDB] USE_DYNAMODB=true but setup failed, falling back to local-only mode: {e}")
        USE_DYNAMODB = False
else:
    print("[DynamoDB] Disabled (USE_DYNAMODB=false) — fall events are kept in memory only. "
          "Set USE_DYNAMODB=true to persist to AWS.")


class FrameRequest(BaseModel):
    device_id: str
    timestamp: str | None = None
    features: list[float]
    human_count: int = 0
    track_id: int | None = None


class ActivityRequest(BaseModel):
    device_id: str
    activity: str
    confidence: float
    probs: dict[str, float] | None = None
    timestamp: str | None = None


class RawPointCloudFrameRequest(BaseModel):
    device_id: str
    point_cloud: list[list[float]] | None = None
    track_indexes: list[int] | None = None
    timestamp: str | None = None
    human_count: int = 0


class ActivityPointCloudBatchFrame(BaseModel):
    frame_num: int | None = None
    # Optional on purpose: a radar frame with no detections arrives as null.
    # When this field was required, one empty frame got the WHOLE batch
    # rejected with 422, so on real radar every activity batch failed and the
    # dashboard never showed a single activity.
    point_cloud: list[list[float]] | None = None
    track_indexes: list[int] | None = None


class ActivityPointCloudBatchRequest(BaseModel):
    device_id: str
    frames: list[ActivityPointCloudBatchFrame]


class WSManager:
    def __init__(self):
        self.clients = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)
        print(f"[WS MANAGER] Client connected. Total active clients: {len(self.clients)}")

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)
        print(f"[WS MANAGER] Client disconnected. Remaining clients: {len(self.clients)}")

    async def broadcast(self, payload: dict):
        dead = []
        count = len(self.clients)
        print(f"[WS BROADCAST] Sending payload to {count} connected WS client(s): activity={payload.get('activity')} conf={payload.get('activity_confidence')}")
        for ws in self.clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


# ── Rule-based fall detector ─────────────────────────────────────────────────
Z_DROP_THRESHOLD = 0.50
BODY_FLAT_THRESH = 0.50
LOW_POINTS_THRESH = 8
DETECTION_WINDOW = 20
MIN_VOTES = 2

# Optional stricter rule: require an actual observed height drop before any
# fall is declared, instead of letting "body flat" + "few points" alone do it.
# Those two votes both really measure "the point cloud got sparse", which is
# also what a person standing far away or partly occluded looks like — so on
# their own they call a fall with no height change ever observed.
#
# Measured over the 11 labelled recordings in Dataset1 (6 fall, 5 no-fall):
#     off (default, unchanged): falls found 6/6, false alarms 11
#     on:                       falls found 5/6, false alarms  2
# Left off by default so detection sensitivity is not silently reduced —
# set REQUIRE_Z_DROP=true to trade that one recording for ~5x fewer alarms.
REQUIRE_Z_DROP = os.getenv("REQUIRE_Z_DROP", "true").strip().lower() in ("1", "true", "yes")

# Minimum points a track must have before we are willing to call a fall on it.
# Live data showed why this is needed: on real radar the median tracked person
# survives the SNR filter with only a couple of points, and a 1-2 point cluster
# has a height range of ~0, so "body flat" and "few points" both vote yes and a
# fall is declared for a person who never moved. Measured on a 42 s live
# session: 25 falls before this gate, 5 after (with REQUIRE_Z_DROP).
MIN_POINTS_TO_JUDGE = int(os.getenv("MIN_POINTS_TO_JUDGE", "5"))

# A real body still has vertical extent, even lying down (~20-30 cm). A cluster
# flatter than this is a reflection or a ghost, not a person on the floor, so it
# must not satisfy the "body flat" vote.
MIN_BODY_EXTENT = float(os.getenv("MIN_BODY_EXTENT", "0.10"))

# Frames with no track association can only be scored by mixing every point in
# the room into one phantom person — the same averaging that used to hide real
# falls. Skip them instead of inventing someone.
FALL_REQUIRE_TRACK = os.getenv("FALL_REQUIRE_TRACK", "true").strip().lower() in ("1", "true", "yes")

# A fall means the person goes down AND STAYS down. Without this, a single
# noisy frame where the centroid dips is enough to fire, which is what made
# live radar report a fall every ~1.7 s: with per-track data the "few points"
# vote is almost always true (70% of real tracks have <=8 points), so it is a
# free vote and one bad z sample completes the pair.
# Measured on a 42 s live session: 25 falls originally, 7 with the votes fix,
# 4 requiring 5 sustained frames, 1 requiring 10 (~0.55 s at 18 fps).
# Lower this if real falls are being missed.
SUSTAIN_FRAMES = int(os.getenv("SUSTAIN_FRAMES", "10"))

# ── Fall method ──────────────────────────────────────────────────────────────
# "velocity" (default) replaces the height/flatness/point-count votes with the
# one signal that actually separates falls on this radar: how fast the person
# is moving DOWN, taken from TI's tracker (a Kalman-smoothed per-person
# velocity), not from our own sparse point clusters.
#
# Why the old votes failed on real radar:
#   * per-person clouds are tiny (median ~2 points after the SNR filter), so
#     z_mean and height_range are dominated by noise;
#   * TI's own height estimate barely moves during a fall (0.16-0.24 m drops
#     measured), likely because a person lying still is filtered out as
#     static clutter before the track follows them down.
#
# Measured on the 11 labelled Dataset1 recordings at 0.5 m/s:
#   velocity: falls caught 4/6, false alarms on sitting/standing 0
#   legacy:   falls caught 6/6 recordings but 11 false alarms
# Set FALL_METHOD=legacy to go back to the vote-based rule.
FALL_METHOD = os.getenv("FALL_METHOD", "velocity").strip().lower()
FALL_DESCENT_SPEED = float(os.getenv("FALL_DESCENT_SPEED", "0.5"))   # m/s downward
FALL_SMOOTH_FRAMES = int(os.getenv("FALL_SMOOTH_FRAMES", "3"))
# The sensor is tilted 15 deg down (sensorPosition 2 0 15), so the tracker's
# vz is in the sensor frame: someone walking TOWARD the radar at 1.2 m/s shows
# ~0.3 m/s of fake downward motion. Rotate back to world vertical exactly the
# way the visualizer's eulerRot does (azimuth tilt 0):
#     v_world_z = cos(tilt) * vz - sin(tilt) * vy
SENSOR_ELEV_TILT_DEG = float(os.getenv("SENSOR_ELEV_TILT_DEG", "15"))
_TILT_COS = float(np.cos(np.deg2rad(SENSOR_ELEV_TILT_DEG)))
_TILT_SIN = float(np.sin(np.deg2rad(SENSOR_ELEV_TILT_DEG)))
# One fall must produce one alert. At 30 frames (~1.7 s) the cooldown expired
# while the person was still on the floor, so a single live fall fired twice.
FALL_COOLDOWN_FRAMES = int(os.getenv("FALL_COOLDOWN_FRAMES", "90"))

# Reject "descents" that happen while the person is travelling across the room.
# Live session evidence (10 detections): the real fall fired at 0.10-0.39 m/s
# horizontal speed, every false one at 0.62-2.32 m/s. Running then stopping
# sharply knocks the tracker's forward and vertical velocity estimates out of
# step, and through the 15-degree tilt correction that appears as a drop.
# Set to 0 to disable.
FALL_MAX_HORIZONTAL_SPEED = float(os.getenv("FALL_MAX_HORIZONTAL_SPEED", "0.5"))
# Judge "travelling" over the last N frames, not just the current one. With
# only the current frame the gate merely DELAYED a running-stop false alarm:
# speed falls under the limit a couple of frames after stopping, while the
# phantom descent is still present, so it fired anyway.
#
# Measured (speed limit 0.5 m/s, cooldown 90):
#   lookback  3 frames: live false alarms 3, labelled falls 4/6, walking falls 3/3
#   lookback 10 frames: live false alarms 0, labelled falls 3/6, walking falls 2/3
# The dropped walking fall was moving at 0.72 m/s when it went down, which
# overlaps the running-stop false alarms, so no single setting keeps both.
# Default favours no false alarms. Use 3 to prioritise falls mid-stride.
FALL_TRAVEL_LOOKBACK_FRAMES = int(os.getenv("FALL_TRAVEL_LOOKBACK_FRAMES", "10"))

# Confirm before alerting. A descent opens a PENDING fall; it only becomes an
# alert if the person does not come back up within FALL_CONFIRM_FRAMES.
# Live evidence (21:03 on 16 Sep): a dip at -0.69 m/s went straight back up
# (+0.34 m/s) and fired an alert; its cooldown then SWALLOWED the real fall
# 2.5 s later (-1.87 m/s, track lost on the floor). A rejected dip must not
# start the cooldown, and a person the tracker loses while a fall is pending
# must still confirm, because someone lying still is dropped as static clutter.
FALL_CONFIRM_FRAMES = int(os.getenv("FALL_CONFIRM_FRAMES", "18"))      # ~1 s at 18 fps
FALL_REBOUND_SPEED  = float(os.getenv("FALL_REBOUND_SPEED", "0.25"))   # m/s upward
FALL_REBOUND_FRAMES = int(os.getenv("FALL_REBOUND_FRAMES", "2"))       # consecutive frames

# A fall must END with the person low, and must be FAST. Live 2-person test
# (21:30, 16 Sep), 16 alerts, 7 real:
#   * real falls took the head height (heightData maxZ, feature 18) to <=66% of
#     standing height; false alarms stayed at >=69% (two people crossing, bending)
#   * real falls peaked at >=0.76 m/s downward; the leftover false alarms <=0.60
#   * 3 false alarms came from a track created moments earlier (a new track
#     "descends" while the tracker settles on its height)
# Relative height, not absolute metres, so a different mounting height still works.
FALL_MIN_TRACK_AGE        = int(os.getenv("FALL_MIN_TRACK_AGE", "20"))          # frames
FALL_MAX_HEAD_RATIO       = float(os.getenv("FALL_MAX_HEAD_RATIO", "0.70"))     # lowest head / standing head
FALL_MIN_HEAD_DROP        = float(os.getenv("FALL_MIN_HEAD_DROP", "0.4"))       # m below pre-fall height
FALL_MIN_PEAK_SPEED       = float(os.getenv("FALL_MIN_PEAK_SPEED", "0.7"))      # m/s, fastest smoothed descent
FALL_HEAD_BASELINE_FRAMES = int(os.getenv("FALL_HEAD_BASELINE_FRAMES", "30"))
FALL_MAX_PENDING_FRAMES   = int(os.getenv("FALL_MAX_PENDING_FRAMES", "45"))     # wait for a slow collapse


class StreamingFallDetector:
    """
    Rule-based fall detector for ONE tracked person.

    One instance per (device, track) — see `get_fall_detector`. Feeding every
    person through a single shared instance is what used to make falls
    invisible in multi-person scenes: the z-history buffer mixed everyone's
    heights together, so one person going down barely moved it.
    """

    def __init__(self):
        self.z_buf = deque(maxlen=DETECTION_WINDOW + 5)
        self.v_buf = deque(maxlen=max(1, FALL_SMOOTH_FRAMES))
        self.h_buf = deque(maxlen=max(1, FALL_SMOOTH_FRAMES, FALL_TRAVEL_LOOKBACK_FRAMES))
        self.cooldown = 0
        self.pending = None          # open, unconfirmed descent
        self.up_run = 0              # consecutive upward frames while pending
        self.confirmed_feat = None   # features to report for the last confirmed fall
        self.age = 0                 # frames with valid tracker state
        self.head_buf = deque(maxlen=max(1, FALL_HEAD_BASELINE_FRAMES))

    def update(self, feat_20: np.ndarray):
        if FALL_METHOD == "velocity":
            return self._update_velocity(feat_20)
        return self._update_legacy(feat_20)

    def _update_velocity(self, feat_20: np.ndarray):
        # feat[12:18] = tracker x, y, z, vx, vy, vz for THIS person
        vx = float(feat_20[15])
        vy = float(feat_20[16])
        vz = float(feat_20[17])

        if self.cooldown > 0:
            self.cooldown -= 1

        # All-zero track state means the tracker had no row for this person this
        # frame: missing data, not "not moving". Treat it like an absent frame.
        if vy == 0.0 and vz == 0.0:
            return self._age_pending(present=False)

        self.age += 1
        head = float(feat_20[18])                      # this person's maxZ, m (0 = no row)

        v_down = _TILT_COS * vz - _TILT_SIN * vy      # world vertical velocity, m/s
        h_speed = float(np.hypot(vx, _TILT_COS * vy + _TILT_SIN * vz))
        self.v_buf.append(v_down)
        self.h_buf.append(h_speed)

        if len(self.v_buf) < self.v_buf.maxlen:
            return False, 0.0, {"method": "velocity", "status": "warming_up",
                                "buffered": len(self.v_buf)}

        v_mean = float(np.mean(self.v_buf))
        h_list = list(self.h_buf)
        h_mean = float(np.mean(h_list[-max(1, FALL_SMOOTH_FRAMES):]))
        h_peak = float(max(h_list[-max(1, FALL_TRAVEL_LOOKBACK_FRAMES):]))
        descending = v_mean <= -FALL_DESCENT_SPEED
        travelling = FALL_MAX_HORIZONTAL_SPEED > 0 and h_peak > FALL_MAX_HORIZONTAL_SPEED

        info = {
            "method":     "velocity",
            "v_down":     round(v_mean, 3),
            "h_speed":    round(h_mean, 3),
            "h_speed_recent_peak": round(h_peak, 3),
            "rejected_travelling": bool(descending and travelling),
            "threshold":  -FALL_DESCENT_SPEED,
            "vz_sensor":  round(vz, 3),
            "vy_sensor":  round(vy, 3),
            "npts":       int(feat_20[9]),
            "head_height": round(head, 3),
        }

        # A fall is already pending: wait for a rebound or for confirmation.
        if self.pending is not None:
            self.up_run = self.up_run + 1 if v_down >= FALL_REBOUND_SPEED else 0
            if self.up_run >= FALL_REBOUND_FRAMES:
                # Came back up: it was a dip, not a fall. No alert and NO cooldown.
                dip = self.pending
                self.pending = None
                self.up_run = 0
                info.update({"status": "dip_rejected", "dip_v_down": round(dip["v_peak"], 3)})
                return False, 0.0, info
            if head > 0:
                self.pending["head_min"] = min(self.pending["head_min"], head)
            if v_mean < self.pending["v_peak"]:
                self.pending.update(v_peak=v_mean, feat=feat_20.copy(), info=dict(info))
            return self._age_pending(present=True, info=info)

        # Open a pending fall.
        baseline = max(self.head_buf) if self.head_buf else 0.0
        if head > 0:
            self.head_buf.append(head)
        if descending and not travelling and self.cooldown == 0:
            if self.age < FALL_MIN_TRACK_AGE:
                info["status"] = "track_too_new"
                return False, 0.0, info
            self.pending = {"age": 0, "v_peak": v_mean, "feat": feat_20.copy(), "info": dict(info),
                            "head_base": max(baseline, head),
                            "head_min": head if head > 0 else float("inf")}
            self.up_run = 0
            info["status"] = "pending"
            return False, 0.0, info

        return False, 0.0, info

    def _age_pending(self, present: bool, info: dict | None = None):
        """Advance an open fall by one frame; confirm once the window passes."""
        if self.pending is None:
            return False, 0.0, info or {"method": "velocity", "status": "no_track_state"}
        self.pending["age"] += 1
        if self.pending["age"] >= FALL_CONFIRM_FRAMES:
            p = self.pending
            head_min, head_base = p["head_min"], p["head_base"]
            low_enough = (head_base > 0 and head_min <= FALL_MAX_HEAD_RATIO * head_base
                          and (head_base - head_min) >= FALL_MIN_HEAD_DROP)
            if -p["v_peak"] < FALL_MIN_PEAK_SPEED:
                # Too slow to be a fall (sitting down, bending, tracker jitter). No cooldown.
                self.pending = None
                self.up_run = 0
                base = dict(info) if info else {"method": "velocity"}
                base.update({"status": "too_slow", "v_down_peak": round(p["v_peak"], 3)})
                return False, 0.0, base
            if not low_enough:
                if p["age"] < FALL_MAX_PENDING_FRAMES:
                    base = dict(info) if info else {"method": "velocity"}
                    base.update({"status": "pending_height", "pending_age": p["age"]})
                    return False, 0.0, base
                # Never got low: bending, sitting, or two people crossing. No cooldown.
                self.pending = None
                self.up_run = 0
                base = dict(info) if info else {"method": "velocity"}
                base.update({"status": "no_height_drop",
                             "head_min": None if head_min == float("inf") else round(head_min, 3),
                             "head_base": round(head_base, 3)})
                return False, 0.0, base
            self.pending = None
            self.up_run = 0
            self.cooldown = FALL_COOLDOWN_FRAMES
            self.confirmed_feat = p["feat"]
            out = dict(p["info"])
            out.update({"status": "confirmed", "v_down_peak": round(p["v_peak"], 3),
                        "confirmed_with_track_present": present,
                        "head_min": round(head_min, 3), "head_base": round(head_base, 3)})
            conf = float(min(1.0, max(0.0, -p["v_peak"] / (2.0 * FALL_DESCENT_SPEED))))
            return True, conf, out
        base = dict(info) if info else {"method": "velocity"}
        base.update({"status": "pending", "pending_age": self.pending["age"]})
        return False, 0.0, base

    def tick_absent(self):
        """The tracker produced no row for this person in this frame."""
        if self.cooldown > 0:
            self.cooldown -= 1
        if FALL_METHOD != "velocity":
            return False, 0.0, {"status": "absent"}
        return self._age_pending(present=False)

    def _update_legacy(self, feat_20: np.ndarray):
        z    = float(feat_20[2])
        npts = float(feat_20[9])
        hrng = float(feat_20[11])

        if self.cooldown > 0:
            self.cooldown -= 1

        # A frame where this person returned no points is missing data, NOT a
        # person at floor level. The features come back all-zero, so appending
        # z = 0.0 here used to fabricate a ~1.2 m "drop" out of a tracking gap
        # and fire a fall on the next frame. Hold the buffer instead.
        if npts < max(1, MIN_POINTS_TO_JUDGE):
            return False, 0.0, {
                "status": "too_few_points",
                "buffered": len(self.z_buf),
                "npts": int(npts),
                "needed": MIN_POINTS_TO_JUDGE,
            }

        self.z_buf.append(z)

        if len(self.z_buf) < DETECTION_WINDOW:
            return False, 0.0, {"status": "warming_up", "buffered": len(self.z_buf)}

        window_z  = list(self.z_buf)[-DETECTION_WINDOW:]
        z_peak    = max(window_z)
        z_current = window_z[-1]
        z_drop    = z_peak - z_current

        height_dropped = z_drop >= Z_DROP_THRESHOLD
        # A body lying down is flat, but not infinitely flat — require real
        # vertical extent so noise specks cannot vote "flat".
        body_flat      = MIN_BODY_EXTENT <= hrng <= BODY_FLAT_THRESH
        few_points     = npts  <= LOW_POINTS_THRESH

        # Did the person STAY down, or did the centroid just blip for a frame?
        if SUSTAIN_FRAMES > 0 and len(window_z) >= SUSTAIN_FRAMES:
            floor = z_peak - Z_DROP_THRESHOLD * 0.7
            stayed_down = all(v <= floor for v in window_z[-SUSTAIN_FRAMES:])
        else:
            stayed_down = True

        votes   = int(height_dropped) + int(body_flat) + int(few_points)
        passes  = ((votes >= MIN_VOTES)
                   and (height_dropped or not REQUIRE_Z_DROP)
                   and stayed_down)
        is_fall = passes and (self.cooldown == 0)

        if is_fall:
            self.cooldown = 30

        confidence = min(1.0, votes / 3.0)
        info = {
            "stayed_down": bool(stayed_down),
            "z_drop":    round(z_drop, 3),
            "z_peak":    round(z_peak, 3),
            "z_current": round(z_current, 3),
            "hrng":      round(hrng, 3),
            "npts":      int(npts),
            "votes":     votes,
        }
        return is_fall, confidence, info


# Fall detectors are keyed by (device_id, track_id) so every tracked person
# gets their own z-history and cooldown. track_id is None for callers that
# send whole-frame features without track association (old behaviour).
detectors: dict[tuple, StreamingFallDetector] = {}
detector_last_seen: dict[tuple, float] = {}
DETECTOR_TTL_SECONDS = 120

activity_detectors: dict[str, ActivityDetector] = {}
latest_activity: dict[str, dict] = {}
latest_telemetry: dict[str, dict] = {}

# In-memory fall-event history, used whenever DynamoDB is disabled so
# /history still returns something useful during local testing instead of
# silently always answering []. Capped so it can't grow without bound.
LOCAL_HISTORY_MAX = 500
local_fall_history: dict[str, deque] = {}


def get_fall_detector(device_id: str, track_id=None) -> StreamingFallDetector:
    """One detector per tracked person, created on first sight."""
    key = (device_id, track_id)
    det = detectors.get(key)
    if det is None:
        det = StreamingFallDetector()
        detectors[key] = det
        if track_id is not None:
            print(f"[FALL INSTANCE] New detector for device={device_id} track={track_id}")
    detector_last_seen[key] = time.time()
    return det


def prune_fall_detectors() -> None:
    """Forget detectors for track ids the tracker has long since dropped."""
    cutoff = time.time() - DETECTOR_TTL_SECONDS
    for key in [k for k, seen in detector_last_seen.items() if seen < cutoff]:
        detectors.pop(key, None)
        detector_last_seen.pop(key, None)


activity_locks: dict[str, threading.Lock] = {}


def get_activity_lock(device_id: str) -> threading.Lock:
    dev_id = device_id.strip() if device_id else "rpi-1"
    lock = activity_locks.get(dev_id)
    if lock is None:
        lock = activity_locks.setdefault(dev_id, threading.Lock())
    return lock


def get_activity_detector(device_id: str) -> ActivityDetector:
    dev_id = device_id.strip() if device_id else "rpi-1"
    if dev_id not in activity_detectors:
        act = ActivityDetector()
        activity_detectors[dev_id] = act
        print(f"[HAR INSTANCE] Created persistent ActivityDetector instance id={id(act)} for device={dev_id}")
    return activity_detectors[dev_id]


ws_manager = WSManager()
app        = FastAPI(title="Radar Fall and Activity API", version="1.2.0")

class GzipRequestMiddleware:
    """
    Accept gzip-compressed request bodies.

    The Raspberry Pi uploads about 280 KB of point cloud per second of radar;
    on a home uplink that is most of the available bandwidth, and the uploads
    fall behind. aws_watcher.py compresses each batch (roughly 8x smaller);
    this unpacks it before FastAPI parses the JSON. Uncompressed requests pass
    straight through, so an older Pi still works.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.lower(): v for k, v in scope.get("headers") or []}
        if headers.get(b"content-encoding", b"").lower() != b"gzip":
            return await self.app(scope, receive, send)

        body, more = b"", True
        while more:
            msg = await receive()
            if msg["type"] != "http.request":
                break
            body += msg.get("body", b"")
            more = msg.get("more_body", False)
        try:
            body = gzip.decompress(body)
        except OSError:
            pass          # not actually gzip — let the route reject it

        scope = dict(scope, headers=[
            (k, v) for k, v in scope["headers"]
            if k.lower() not in (b"content-encoding", b"content-length")
        ] + [(b"content-length", str(len(body)).encode())])

        delivered = False

        async def receive_once():
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, receive_once, send)


app.add_middleware(GzipRequestMiddleware)

# ── CORS — must be added before routes ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def ddb_num(x, digits=4):
    return Decimal(str(round(float(x), digits)))


def _json_safe(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def save_fall_event(dev_id: str, ddb_item: dict, plain_event: dict) -> None:
    """
    Persist a FALL event. Writes to DynamoDB when USE_DYNAMODB=true (with the
    same best-effort try/except as before — a flaky AWS call must never break
    live detection), otherwise keeps it in the in-memory history ring buffer
    so /history has something to show during local testing.
    """
    if USE_DYNAMODB and table is not None:
        try:
            table.put_item(Item=ddb_item)
        except Exception as e:
            print(f"[DynamoDB] put_item failed, event kept in memory only: {e}")
            local_fall_history.setdefault(dev_id, deque(maxlen=LOCAL_HISTORY_MAX)).append(plain_event)
    else:
        local_fall_history.setdefault(dev_id, deque(maxlen=LOCAL_HISTORY_MAX)).append(plain_event)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Radar Fall Detection and HAR API is running",
        "docs": "/docs",
        "frontend_dashboard": "http://127.0.0.1:5173"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "aws" if USE_DYNAMODB else "local",
        "dynamodb_enabled": USE_DYNAMODB,
        "table": TABLE_NAME if USE_DYNAMODB else None,
        "region": AWS_REGION if USE_DYNAMODB else None,
        "activity_classes": ActivityDetector.DEFAULT_CLASSES,
    }


@app.post("/activity")
async def receive_activity(req: ActivityRequest):
    dev_id = req.device_id.strip() if req.device_id else "rpi-1"
    if not dev_id:
        raise HTTPException(status_code=422, detail="device_id cannot be empty")
    if not req.activity or not req.activity.strip():
        raise HTTPException(status_code=422, detail="activity cannot be empty")
    if not (0.0 <= req.confidence <= 1.0):
        raise HTTPException(status_code=422, detail="confidence must be between 0.0 and 1.0")

    ts = req.timestamp or now_iso()
    act_record = {
        "activity": req.activity.strip(),
        "confidence": round(float(req.confidence), 4),
        "probs": req.probs or {},
        "timestamp": ts,
    }
    latest_activity[dev_id] = act_record

    base = latest_telemetry.get(dev_id, {
        "device_id": dev_id,
        "class_id": 0,
        "class_name": "NO-FALL",
        "confidence": 0.95,
        "is_fall": False,
        "probs": [0.95, 0.05],
        "z_mean": 0.0,
        "height_range": 0.0,
        "n_points": 0,
        "x_mean": 0.0,
        "y_mean": 0.0,
        "frame_count": 0,
        "human_count": 1,
        "debug": {"status": "activity_update"},
    })

    event = {
        **base,
        "device_id": dev_id,
        "ts": ts,
        "timestamp": ts,
        "activity": act_record["activity"],
        "activity_confidence": act_record["confidence"],
        "activity_probs": act_record["probs"],
        "activity_people": act_record.get("people", []),
    }
    latest_telemetry[dev_id] = event

    await ws_manager.broadcast(event)
    print(f"[POST /activity] Broadcasted event to {len(ws_manager.clients)} WS clients: device={dev_id} activity={act_record['activity']} conf={act_record['confidence']}")
    return {"status": "ok", "device_id": dev_id, "data": act_record}


@app.get("/activity")
def get_activity(device_id: str = "rpi-1"):
    dev_id = device_id.strip() if device_id else "rpi-1"
    if dev_id in latest_activity:
        return {"status": "ok", "device_id": dev_id, "activity": latest_activity[dev_id]}
    return {"status": "no_data", "device_id": dev_id, "activity": None}


@app.post("/activity/pointcloud")
async def activity_pointcloud(req: RawPointCloudFrameRequest):
    dev_id = req.device_id.strip() if req.device_id else "rpi-1"
    if not dev_id:
        raise HTTPException(status_code=422, detail="device_id cannot be empty")
    ts = req.timestamp or now_iso()

    act_det = get_activity_detector(dev_id)
    def _run_one():
        with get_activity_lock(dev_id):
            return act_det.push(req.point_cloud, req.track_indexes)
    pred = await run_in_threadpool(_run_one)

    print(f"[HAR POST] dev={dev_id} det_id={pred.get('detector_id')} buffered={pred.get('buffered')}/{pred.get('window_size')} accepted={pred.get('accepted_frame')} reason={pred.get('reason')}")

    act_record = {
        "activity": pred.get("activity"),
        "confidence": pred.get("confidence", 0.0),
        "probs": pred.get("probs", {}),
        "status": pred.get("status"),
        "buffered": pred.get("buffered", 0),
        "people": pred.get("people", []),
        "people_count": pred.get("people_count", 0),
        "timestamp": ts,
    }
    latest_activity[dev_id] = act_record

    base = latest_telemetry.get(dev_id, {
        "device_id": dev_id,
        "class_id": 0,
        "class_name": "NO-FALL",
        "confidence": 0.95,
        "is_fall": False,
        "probs": [0.95, 0.05],
        "z_mean": 0.0,
        "height_range": 0.0,
        "n_points": len(req.point_cloud) if req.point_cloud else 0,
        "x_mean": 0.0,
        "y_mean": 0.0,
        "frame_count": 0,
        "human_count": req.human_count,
        "debug": {"status": "activity_pointcloud"},
    })

    event = {
        **base,
        "device_id": dev_id,
        "ts": ts,
        "timestamp": ts,
        "activity": act_record["activity"],
        "activity_confidence": act_record["confidence"],
        "activity_probs": act_record["probs"],
        "activity_people": act_record.get("people", []),
    }
    latest_telemetry[dev_id] = event

    await ws_manager.broadcast(event)
    return {"status": "ok", "device_id": dev_id, "prediction": pred}


@app.post("/activity/pointcloud/batch")
async def activity_pointcloud_batch(req: ActivityPointCloudBatchRequest):
    dev_id = req.device_id.strip() if req.device_id else "rpi-1"
    if not dev_id:
        raise HTTPException(status_code=422, detail="device_id cannot be empty")
    ts = now_iso()

    act_det = get_activity_detector(dev_id)

    frames_received = len(req.frames)
    frames_accepted = 0
    frames_rejected = 0
    last_frame_num = None
    last_pred = None

    # The detector is CPU-bound (DBSCAN + XGBoost per person per frame). Run it
    # in a worker thread so it can't stall the event loop: before this, a
    # slow or failing activity batch froze every other request, fall batches
    # included. One lock per device keeps that device's sliding windows
    # consistent if two batches ever overlap.
    def _run_batch():
        acc = rej = 0
        fn = None
        lp = None
        with get_activity_lock(dev_id):
            for item in req.frames:
                if item.frame_num is not None:
                    fn = item.frame_num
                try:
                    pred = act_det.push(item.point_cloud, item.track_indexes)
                except Exception as e:
                    # One bad frame must not take down the whole batch.
                    print(f"[HAR] frame {item.frame_num} skipped: {type(e).__name__}: {e}")
                    rej += 1
                    continue
                lp = pred
                if pred.get("accepted_frame"):
                    acc += 1
                else:
                    rej += 1
        return acc, rej, fn, lp

    frames_accepted, frames_rejected, last_frame_num, last_pred = await run_in_threadpool(_run_batch)

    if last_pred is None:
        last_pred = {
            "detector_id": id(act_det),
            "status": "warming_up",
            "buffered": len(act_det._window),
            "window_size": 30,
            "activity": "warming_up",
            "confidence": 0.0,
            "probs": {},
        }

    act_record = {
        "activity": last_pred.get("activity"),
        "confidence": last_pred.get("confidence", 0.0),
        "probs": last_pred.get("probs", {}),
        "status": last_pred.get("status"),
        "buffered": last_pred.get("buffered", 0),
        "people": last_pred.get("people", []),
        "people_count": last_pred.get("people_count", 0),
        "timestamp": ts,
    }
    latest_activity[dev_id] = act_record

    base = latest_telemetry.get(dev_id, {
        "device_id": dev_id,
        "class_id": 0,
        "class_name": "NO-FALL",
        "confidence": 0.95,
        "is_fall": False,
        "probs": [0.95, 0.05],
        "z_mean": 0.0,
        "height_range": 0.0,
        "n_points": 0,
        "x_mean": 0.0,
        "y_mean": 0.0,
        "frame_count": 0,
        "human_count": 1,
        "debug": {"status": "activity_batch_pointcloud"},
    })

    event = {
        **base,
        "device_id": dev_id,
        "ts": ts,
        "timestamp": ts,
        "activity": act_record["activity"],
        "activity_confidence": act_record["confidence"],
        "activity_probs": act_record["probs"],
        "activity_people": act_record.get("people", []),
    }
    latest_telemetry[dev_id] = event

    # Perform exactly ONE WebSocket broadcast at the end of the batch
    await ws_manager.broadcast(event)

    print(f"[HAR BATCH POST] dev={dev_id} received={frames_received} accepted={frames_accepted} rejected={frames_rejected} buffered={last_pred.get('buffered')} status={last_pred.get('status')} activity={last_pred.get('activity')}")

    return {
        "device_id": dev_id,
        "frames_received": frames_received,
        "frames_accepted": frames_accepted,
        "frames_rejected": frames_rejected,
        "last_frame_num": last_frame_num,
        "detector_id": id(act_det),
        "status": last_pred.get("status"),
        "buffered": last_pred.get("buffered", 0),
        "window_size": last_pred.get("window_size", 30),
        "activity": last_pred.get("activity"),
        "confidence": last_pred.get("confidence", 0.0),
        "probs": last_pred.get("probs", {}),
        "people": last_pred.get("people", []),
        "people_count": last_pred.get("people_count", 0),
    }


@app.post("/frame")
async def frame(req: FrameRequest):
    if len(req.features) != 20:
        raise HTTPException(status_code=422, detail="features must be length 20")

    feat      = np.array(req.features, dtype=np.float32)
    dev_id    = req.device_id.strip() if req.device_id else "rpi-1"
    ts        = req.timestamp or now_iso()

    det = get_fall_detector(dev_id, getattr(req, "track_id", None))

    is_fall, conf, info = det.update(feat)

    p_fall   = conf if is_fall else 0.05
    p_nofall = 1.0 - p_fall

    act_data = latest_activity.get(dev_id)
    act_name = act_data.get("activity") if act_data else None
    act_conf = act_data.get("confidence") if act_data else None
    act_probs = act_data.get("probs") if act_data else None
    act_people = act_data.get("people", []) if act_data else []

    event = {
        "device_id":   dev_id,
        "ts":          ts,
        "timestamp":   ts,
        "class_id":    1 if is_fall else 0,
        "class_name":  "FALL" if is_fall else "NO-FALL",
        "confidence":  float(conf if is_fall else p_nofall),
        "is_fall":     bool(is_fall),
        "probs":       [round(p_nofall, 3), round(p_fall, 3)],
        "z_mean":      float(feat[2]),
        "height_range": float(feat[11]),
        "n_points":    int(feat[9]),
        "x_mean":      float(feat[0]),
        "y_mean":      float(feat[1]),
        "frame_count": 0,
        "human_count": req.human_count,
        "debug":       info,
        # Extended activity recognition fields
        "activity":            act_name,
        "activity_confidence": act_conf,
        "activity_probs":      act_probs,
        "activity_people":     act_people,
    }
    latest_telemetry[dev_id] = event

    # Broadcast ALL events over WebSocket (live feed - no filter)
    await ws_manager.broadcast(event)

    # Save ONLY FALL events (to DynamoDB if enabled, else in-memory)
    if is_fall:
        ttl = int(time.time()) + 86400  # 24h TTL
        item = {
            "device_id":   dev_id,
            "ts":          ts,
            "pk":          "all",
            "class_id":    1,
            "class_name":  "FALL",
            "confidence":  ddb_num(conf),
            "is_fall":     True,
            "z_mean":      ddb_num(feat[2]),
            "x_mean":      ddb_num(feat[0]),
            "y_mean":      ddb_num(feat[1]),
            "height_range": ddb_num(feat[11]),
            "n_points":    int(feat[9]),
            "expire_at":   ttl,
        }
        save_fall_event(dev_id, item, _json_safe(item))

    return event


# ── Batch endpoint: send many frames in one HTTP round-trip ──────────────────

class FrameItem(BaseModel):
    features:    list[float]
    timestamp:   str | None = None
    human_count: int = 0
    # Per-person fields. The watcher now sends one item per tracked person per
    # frame, tagged with that person's track id and the frame they came from.
    # Both stay optional so anything still sending one whole-frame vector per
    # frame keeps working exactly as before.
    track_id:    int | None = None
    frame_num:   int | None = None

class BatchFrameRequest(BaseModel):
    device_id: str
    frames:    list[FrameItem]


def _group_by_frame(items: list[FrameItem]) -> list[list[FrameItem]]:
    """
    Group the flat item list into per-frame buckets.

    With per-person items a single radar frame arrives as several items that
    share a frame_num — they must be judged together so the frame produces one
    verdict, not one per person. Items without a frame_num each stand alone,
    which is exactly the old one-item-per-frame behaviour.
    """
    groups: list[list[FrameItem]] = []
    for item in items:
        if (item.frame_num is not None and groups
                and groups[-1][0].frame_num == item.frame_num):
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def evaluate_fall_group(dev_id: str, group: list) -> list[dict]:
    """
    Run one radar frame -- every person row in it -- through the per-person fall
    detectors, including people the tracker LOST this frame, so a pending fall can
    still confirm after a fallen person drops out of tracking. Used by
    /frames/batch and by offline replay tests, so both run the same logic.
    """
    per_track: list[dict] = []
    present = set()
    frame_ts = group[0].timestamp if group else None
    group_humans = max([it.human_count for it in group] + [0])

    for item in group:
        if len(item.features) != 20:
            continue
        if FALL_REQUIRE_TRACK and item.track_id is None:
            continue          # no track association -> no attributable person
        feat = np.array(item.features, dtype=np.float32)
        det = get_fall_detector(dev_id, item.track_id)
        is_fall, conf, info = det.update(feat)
        present.add(item.track_id)
        if is_fall and det.confirmed_feat is not None:
            feat = det.confirmed_feat      # report the moment of the descent
        per_track.append({"track_id": item.track_id, "feat": feat, "is_fall": is_fall,
                          "conf": conf, "info": info, "human_count": item.human_count,
                          "timestamp": item.timestamp})

    # Absence only means something when rows are grouped by real frame numbers.
    if group and all(it.frame_num is not None for it in group):
        for (d_dev, d_tid), det in list(detectors.items()):
            if d_dev != dev_id or d_tid is None or d_tid in present:
                continue
            is_fall, conf, info = det.tick_absent()
            if is_fall and det.confirmed_feat is not None:
                per_track.append({"track_id": d_tid, "feat": det.confirmed_feat, "is_fall": True,
                                  "conf": conf, "info": info, "human_count": group_humans,
                                  "timestamp": frame_ts})
    return per_track


@app.post("/frames/batch")
async def frames_batch(req: BatchFrameRequest):
    dev_id = req.device_id.strip() if req.device_id else "rpi-1"

    act_data = latest_activity.get(dev_id)
    act_name = act_data.get("activity") if act_data else None
    act_conf = act_data.get("confidence") if act_data else None
    act_probs = act_data.get("probs") if act_data else None
    act_people = act_data.get("people", []) if act_data else []

    results = []
    for group in _group_by_frame(req.frames):
        per_track = evaluate_fall_group(dev_id, group)
        if not per_track:
            continue

        # The frame is a FALL if ANY person fell. That person becomes the
        # subject of the event; otherwise report whoever has the most points,
        # which is the best single summary of the scene.
        fallen = [r for r in per_track if r["is_fall"]]
        if fallen:
            lead = max(fallen, key=lambda r: r["conf"])
        else:
            lead = max(per_track, key=lambda r: float(r["feat"][9]))

        is_fall = bool(lead["is_fall"])
        conf    = lead["conf"]
        feat    = lead["feat"]
        info    = dict(lead["info"])
        ts      = lead["timestamp"] or now_iso()

        tracked_ids = [r["track_id"] for r in per_track if r["track_id"] is not None]
        if tracked_ids:
            info["tracks_evaluated"] = len(per_track)
            info["fell"] = [r["track_id"] for r in fallen]
        human_count = max([r["human_count"] for r in per_track] + [0])
        if tracked_ids and not human_count:
            human_count = len(set(tracked_ids))

        p_fall   = conf if is_fall else 0.05
        p_nofall = 1.0 - p_fall

        event = {
            "device_id":    dev_id,
            "ts":           ts,
            "timestamp":    ts,
            "class_id":     1 if is_fall else 0,
            "class_name":   "FALL" if is_fall else "NO-FALL",
            "confidence":   float(conf if is_fall else p_nofall),
            "is_fall":      is_fall,
            "probs":        [round(p_nofall, 3), round(p_fall, 3)],
            "z_mean":       float(feat[2]),
            "height_range": float(feat[11]),
            "n_points":     int(feat[9]),
            "x_mean":       float(feat[0]),
            "y_mean":       float(feat[1]),
            "frame_count":  0,
            "human_count":  human_count,
            "debug":        info,
            "track_id":     lead["track_id"],
            "activity":            act_name,
            "activity_confidence": act_conf,
            "activity_probs":      act_probs,
            "activity_people":     act_people,
        }
        latest_telemetry[dev_id] = event

        await ws_manager.broadcast(event)

        if is_fall:
            ttl = int(time.time()) + 86400
            item_db = {
                "device_id":    dev_id,
                "ts":           ts,
                "pk":           "all",
                "class_id":     1,
                "class_name":   "FALL",
                "confidence":   ddb_num(conf),
                "is_fall":      True,
                "z_mean":       ddb_num(feat[2]),
                "x_mean":       ddb_num(feat[0]),
                "y_mean":       ddb_num(feat[1]),
                "height_range": ddb_num(feat[11]),
                "n_points":     int(feat[9]),
                "expire_at":    ttl,
            }
            if lead["track_id"] is not None:
                item_db["track_id"] = int(lead["track_id"])
            save_fall_event(dev_id, item_db, _json_safe(item_db))

        results.append(event)

    prune_fall_detectors()
    return {"processed": len(results), "results": results}


@app.get("/history")
def history(device_id: str, limit: int = 60):
    if USE_DYNAMODB and table is not None:
        try:
            resp = table.query(
                KeyConditionExpression=Key("device_id").eq(device_id),
                ScanIndexForward=False,
                Limit=limit,
            )
            items = resp.get("Items", [])
            return [_json_safe(x) for x in items]
        except Exception as e:
            print(f"[DynamoDB] query failed: {e}")
            return []
    # Local mode: serve the in-memory ring buffer, most recent first
    items = list(local_fall_history.get(device_id, []))[-limit:]
    items.reverse()
    return items


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
