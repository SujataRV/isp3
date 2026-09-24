# How long does a JSON file take to reach AWS?

Measured on 23 Sep on the actual Raspberry Pi 5, using a real 1-second radar
file (18 frames, 2 people, 281 KB on disk). The Pi posted to the backend
running on the laptop over Wi-Fi; the internet portion is estimated from the
bandwidth and round-trip time measured from the Pi.

## Per stage

| Stage | Time | Measured how |
|---|---|---|
| Frame captured → written into the current JSON file | **0–1000 ms** (avg ~500) | The visualizer flushes one file per second, so a frame waits for the next flush. |
| File appears → watcher notices it | **0.1 ms** | inotify (watchdog) on the Pi. |
| Read file + extract per-person features | **36 ms** | Pi CPU, 18 frames, 2 people. |
| Compress + upload both batches, backend processes them | **184 ms** over Wi-Fi to the laptop | Measured. Includes fall detection and activity inference on the server. |
| Same upload to EC2 over the internet | **~250–600 ms** (estimated) | 26 KB at the measured uplink (0.2–3 Mbit/s) plus ~100–200 ms round trip. |

**A JSON file reaches AWS and is fully processed in roughly 0.3–0.7 s after it
is written**, and each frame inside it is 0–1 s old by then because of the
one-second batching.

## Fall alert, end to end

| Step | Time |
|---|---|
| Person falls | 0 |
| Detector waits to confirm they stayed down (`FALL_CONFIRM_FRAMES=18`) | ~1.0 s |
| Those frames wait for the next file flush | 0–1.0 s |
| Upload + detection + WebSocket push to the dashboard | ~0.3–0.7 s |
| **Alert visible on the dashboard** | **≈ 1.5–2.7 s after the fall** |

The confirm window is deliberate: it is what stops someone sitting down or
bending over from raising an alert.

## Bandwidth

Raw point clouds are the bulk of the traffic. The watcher gzips each batch and
trims coordinates to millimetres:

| | Per second of radar | Sustained |
|---|---|---|
| Raw JSON (old behaviour) | 284 KB | 2.2 Mbit/s |
| Rounded only | 140 KB | 1.1 Mbit/s |
| **Rounded + gzip (current)** | **26 KB** | **0.2 Mbit/s** |

This matters: the uplink measured from the Pi ranged from 0.2 to 3 Mbit/s, so
the old 2.2 Mbit/s stream would have fallen steadily further behind, while
0.2 Mbit/s fits even on the worst of it. Roughly 90 MB per hour of radar, or
2.2 GB per day of continuous streaming.

Controlled by `GZIP_UPLOAD` and `UPLOAD_ROUND_DP` in the Pi's `edge.env`. The
backend accepts both compressed and plain uploads, and the Pi automatically
retries uncompressed if the backend does not understand gzip.

## What would make it faster

1. **Shorter flush interval.** `FLUSH_INTERVAL_S` in `stream_writer.py` — 0.5 s
   halves the batching wait but doubles the number of requests.
2. **Same AWS region as the radar** (`ap-south-1` for India) — the round trip
   to `us-east-1` was more than 3× the Mumbai one.
3. **Shorter fall confirm window.** `FALL_CONFIRM_FRAMES=12` saves ~0.3 s, but
   it caused missed falls in testing, so it is not recommended.

## Caveats

The internet figures are an estimate: the old EC2 box (43.205.167.81) was
unreachable during the test, so nothing was timed against a live AWS
deployment. Re-run `smoke_test.sh` once the backend is deployed, and compare
the `[fall]` / `[har]` millisecond figures the watcher prints on the Pi with
the 184 ms measured over the LAN.
