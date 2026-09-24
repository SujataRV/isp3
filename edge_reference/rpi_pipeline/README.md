# RPi Pipeline

Streams radar frame features from the IWR6843 sensor (or JSON files) to the
fall + activity API — **local by default** (`CLOUD_API_URL` in `config.py`
already points at `http://127.0.0.1:8000/frame`), pointable at a real AWS
EC2 deployment when needed. See `../LOCAL_TESTING.md` for the full local
three-process setup (visualizer + this watcher + the API/frontend).

## Files

| File | Purpose |
|---|---|
| `config.py` | All settings — API URL, watch folder, serial ports, window size |
| `feature_extract.py` | Extracts 20-dim feature vector from each radar frame |
| `radar_capture.py` | Reads serial data from IWR6843 and yields frame dicts |
| `model.py` | TransformerCNNLSTM architecture (used by the API server) |
| `aws_watcher.py` | **Watches a folder for JSON files → streams features to the API** |
| `cloud_stream_sender.py` | **Reads live serial data → streams features to the API** |
| `cloud_stream_simulator.py` | Replays a single JSON file → streams to the API (for testing) |
| `multi_class_model_best.pth` | Trained model weights |
| `multi_class_scaler.pkl` | StandardScaler for feature normalisation |
| `watch_test/` | Sample JSON files for testing |

## Quick Start

### Option A — Radar connected via USB (live streaming)
```bash
pip install -r requirements.txt
python cloud_stream_sender.py
```

### Option B — No radar connected (JSON file watcher)
```bash
pip install -r requirements.txt
# No folder argument needed — defaults to WATCH_DIR in config.py, which is
# the mmWave Visualizer's radar_stream/ folder (its "Stream to isp3" checkbox):
python aws_watcher.py --process-existing

# Or point it at any other folder of TI-visualizer-format JSON files:
python aws_watcher.py ./watch_test --process-existing
```

### Option C — Test with a single JSON file
```bash
python cloud_stream_simulator.py watch_test/replay_test_1.json --loop
```

## Configuration

Edit `config.py` or set environment variables:

```bash
# API endpoint — defaults to the local backend, no AWS required
export CLOUD_API_URL="http://127.0.0.1:8000/frame"

# Folder to watch for JSON files — defaults to the sibling
# "MMWAVE visualizer/radar_stream" folder; only needs overriding if that
# project lives somewhere else, or you're pointing at a different isp3 checkout
export WATCH_DIR="/path/to/radar_stream"

# Serial ports (Linux defaults for IWR6843)
export SERIAL_PORT_DATA="/dev/ttyACM0"
export SERIAL_PORT_CFG="/dev/ttyACM1"

# Device identifier sent with each frame
export DEVICE_ID="rpi-1"
```

To point at a real AWS EC2 deployment instead:
```bash
export CLOUD_API_URL="http://<your-ec2-ip>/frame"
```

## Monitor on EC2
```bash
docker logs -f radar-api
```
