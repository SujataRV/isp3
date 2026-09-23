from boto3.dynamodb.conditions import Key
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from collections import deque

import numpy as np
import boto3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
TABLE_NAME = os.getenv("RADAR_EVENTS_TABLE", "radar_events")

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(TABLE_NAME)


class FrameRequest(BaseModel):
    device_id: str
    timestamp: str | None = None
    features: list[float]
    human_count: int = 0


class WSManager:
    def __init__(self):
        self.clients = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, payload: dict):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


# ── Rule-based fall detector ─────────────────────────────────────────────────
# Tuned for demo: ~1.65 m subjects, standing-to-ground falls only.
# Root-cause of every-8s false positive: one noisy high-z frame keeps z_peak
# elevated for 20 frames → z_drop stays above threshold → re-fires on cooldown reset.
# Fix: use 90th-percentile z_peak (spike-resistant) + smooth z_current.

Z_DROP_THRESHOLD      = 0.90   # z must drop 90 cm from its robust peak
BODY_FLAT_THRESH      = 0.40   # height_range < 0.40 m → body is horizontal
LOW_POINTS_THRESH     = 8      # upper bound: few radar points on ground
MIN_POINTS_VALID      = 3      # lower bound: < 3 pts → z_mean unreliable, skip frame
DETECTION_WINDOW      = 20     # rolling look-back window (frames)
SUSTAINED_DROP_FRAMES = 3      # drop must persist for 3 consecutive frames
COOLDOWN_FRAMES       = 150    # ~8 s at 18 fps — no re-trigger while on ground


class StreamingFallDetector:
    def __init__(self):
        self.z_buf       = deque(maxlen=DETECTION_WINDOW + 5)
        self.drop_streak = 0
        self.cooldown    = 0

    def clear(self):
        """Reset the z buffer and streak — called when no human is in frame."""
        self.z_buf.clear()
        self.drop_streak = 0
        # Keep cooldown as-is so an in-progress cooldown is not skipped

    def update(self, feat_20: np.ndarray, human_count: int = 1):
        # ── Guard: no human in frame — block detection and flush stale buffer ──
        # Without this, old standing z values remain in the buffer after a person
        # leaves, and radar noise can satisfy the drop conditions on an empty room.
        if human_count == 0:
            self.clear()
            return False, 0.0, {"status": "no_human", "human_count": 0}

        z    = float(feat_20[2])
        npts = float(feat_20[9])
        hrng = float(feat_20[11])

        # Only buffer z when enough points exist — low-point frames have
        # unstable z_mean that can spike the window max by 1+ metre
        if npts >= MIN_POINTS_VALID:
            self.z_buf.append(z)

        if self.cooldown > 0:
            self.cooldown -= 1

        if len(self.z_buf) < DETECTION_WINDOW:
            return False, 0.0, {"status": "warming_up", "buffered": len(self.z_buf)}

        window_z = list(self.z_buf)[-DETECTION_WINDOW:]

        # 90th-percentile peak: a single stray spike frame (1 out of 20)
        # no longer keeps z_peak elevated for the whole next window
        z_peak    = float(np.percentile(window_z, 90))

        # Smoothed current: mean of last 3 frames — one noisy frame can't
        # suddenly drag z_current down by a metre
        z_current = float(np.mean(window_z[-3:]))

        z_drop = z_peak - z_current

        # Sustained drop streak — resets the moment z recovers
        if z_drop >= Z_DROP_THRESHOLD:
            self.drop_streak += 1
        else:
            self.drop_streak = 0

        # ── Three conditions — ALL must be true simultaneously ────────────────
        height_dropped = self.drop_streak >= SUSTAINED_DROP_FRAMES

        body_flat      = hrng <= BODY_FLAT_THRESH

        # Bounded range: npts must be 3–8 (too few = noisy; too many = standing)
        few_points     = MIN_POINTS_VALID <= npts <= LOW_POINTS_THRESH

        is_fall = height_dropped and body_flat and few_points and self.cooldown == 0

        if is_fall:
            self.cooldown    = COOLDOWN_FRAMES
            self.drop_streak = 0

        votes      = int(height_dropped) + int(body_flat) + int(few_points)
        confidence = min(1.0, votes / 3.0)
        info = {
            "z_drop":      round(z_drop, 3),
            "z_peak_p90":  round(z_peak, 3),
            "z_current":   round(z_current, 3),
            "drop_streak": self.drop_streak,
            "hrng":        round(hrng, 3),
            "npts":        int(npts),
            "votes":       votes,
            "initial_trigger": is_fall,
        }
        return is_fall, confidence, info


detectors  = {}
ws_manager = WSManager()
app        = FastAPI(title="Radar Rule-Based API", version="1.0.0")

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


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "table": TABLE_NAME, "region": AWS_REGION}


@app.post("/frame")
async def frame(req: FrameRequest):
    if len(req.features) != 20:
        raise HTTPException(status_code=422, detail="features must be length 20")

    feat      = np.array(req.features, dtype=np.float32)
    device_id = req.device_id
    ts        = req.timestamp or now_iso()

    det = detectors.get(device_id)
    if det is None:
        det = StreamingFallDetector()
        detectors[device_id] = det

    is_fall, conf, info = det.update(feat, human_count=req.human_count)

    p_fall   = conf if is_fall else 0.05
    p_nofall = 1.0 - p_fall

    event = {
        "device_id":   device_id,
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
    }

    # Broadcast ALL events over WebSocket (live feed — no filter)
    await ws_manager.broadcast(event)

    # Save ONLY FALL events to DynamoDB
    if is_fall:
        ttl = int(time.time()) + 86400  # 24h TTL
        item = {
            "device_id":   device_id,
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
        table.put_item(Item=item)

    return event


# ── Batch endpoint: send many frames in one HTTP round-trip ──────────────────

class FrameItem(BaseModel):
    features:    list[float]
    timestamp:   str | None = None
    human_count: int = 0

class BatchFrameRequest(BaseModel):
    device_id: str
    frames:    list[FrameItem]

@app.post("/frames/batch")
async def frames_batch(req: BatchFrameRequest):
    device_id = req.device_id
    det = detectors.get(device_id)
    if det is None:
        det = StreamingFallDetector()
        detectors[device_id] = det

    results = []
    for item in req.frames:
        if len(item.features) != 20:
            continue

        feat = np.array(item.features, dtype=np.float32)
        ts   = item.timestamp or now_iso()

        is_fall, conf, info = det.update(feat)

        p_fall   = conf if is_fall else 0.05
        p_nofall = 1.0 - p_fall

        event = {
            "device_id":    device_id,
            "ts":           ts,
            "timestamp":    ts,
            "class_id":     1 if is_fall else 0,
            "class_name":   "FALL" if is_fall else "NO-FALL",
            "confidence":   float(conf if is_fall else p_nofall),
            "is_fall":      bool(is_fall),
            "probs":        [round(p_nofall, 3), round(p_fall, 3)],
            "z_mean":       float(feat[2]),
            "height_range": float(feat[11]),
            "n_points":     int(feat[9]),
            "x_mean":       float(feat[0]),
            "y_mean":       float(feat[1]),
            "frame_count":  0,
            "human_count":  item.human_count,
            "debug":        info,
        }

        await ws_manager.broadcast(event)

        if is_fall:
            ttl = int(time.time()) + 86400
            item_db = {
                "device_id":    device_id,
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
            table.put_item(Item=item_db)

        results.append(event)

    return {"processed": len(results), "results": results}


@app.get("/history")
def history(device_id: str, limit: int = 60):
    resp = table.query(
        KeyConditionExpression=Key("device_id").eq(device_id),
        ScanIndexForward=False,
        Limit=limit,
    )
    items = resp.get("Items", [])
    return [_json_safe(x) for x in items]


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
