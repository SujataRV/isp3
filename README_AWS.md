# Radar fall + activity system — AWS side

This zip is the **cloud half** of the project. The radar itself and the two
programs that talk to it run on a Raspberry Pi and are **not** deployed from
here — `edge_reference/` is included only so you can read that code.

```
 ── EDGE (Raspberry Pi, already set up) ──────    ── CLOUD (this zip) ────────────
 radar ──USB──> visualizer  ──1 JSON file/sec──>  watcher ──HTTP──> FastAPI backend
                (3D view, people count,           (feature                │
                 activity labels)                  extraction)            ├─> fall detection
                                                                          ├─> activity (HAR)
                                                                          ├─> WebSocket /ws
                                                                          └─> DynamoDB (optional)
                                                                                  │
                                                                     React dashboard (nginx)
```

## What runs where — exactly

**On AWS (deploy these):**

| File | Role |
|---|---|
| `backend/radar-api-main.py` | **The only server program.** FastAPI app: fall detection, HAR, history, WebSocket. |
| `backend/activity/activity_detector.py` | Activity (HAR) classifier used by the backend. |
| `backend/activity/models/` | Trained XGBoost model: `xgboost_model.json`, `label_encoder.pkl`, `model_config.json`. |
| `backend/requirements.txt` | Python dependencies for the backend. |
| `backend/requirements-aws.txt` | Extra dependency (`boto3`) — only with `USE_DYNAMODB=true`. |
| `backend/.env.example` | Configuration; copy to `.env`. |
| `backend/radar-api.service` | systemd unit that keeps the API running. |
| `backend/nginx-radar.conf` | nginx: serves the dashboard on port 80, proxies API + WebSocket. |
| `frontend/` | React dashboard. Built with `npm run build`; nginx serves `dist/`. |
| `deploy_ec2.sh` | Does all of the above on a fresh Ubuntu EC2 instance. |
| `smoke_test.sh` | Checks a deployment responds. |
| `LATENCY.md` | Measured timing and bandwidth of the whole path. |

**On the Raspberry Pi (do NOT deploy to AWS — already installed at `~/radar_edge`):**

| File | Role |
|---|---|
| `edge_reference/visualizer/gui_activity.py` | Radar GUI; writes one JSON file per second. |
| `edge_reference/visualizer/*` (parsers, model) | Support code for the GUI. |
| `edge_reference/rpi_pipeline/aws_watcher.py` | Reads those JSON files and POSTs to this backend. |
| `edge_reference/rpi_pipeline/feature_extract.py` | Turns a radar frame into per-person features. |
| `edge_reference/edge.env` | Where the Pi is told the backend's address. |

The backend does **not** read files or talk to the radar. Its only input is
HTTP from the Pi.

## Deploy (fresh EC2)

Ubuntu 22.04+, t3.small or larger. Security group: inbound **80** (dashboard +
Pi uploads) and **22** (ssh). The Pi only needs port 80 outbound.

```bash
scp -i key.pem -r AWS_DEPLOY ubuntu@<ec2-ip>:~/
ssh -i key.pem ubuntu@<ec2-ip>
cd ~/AWS_DEPLOY && ./deploy_ec2.sh <ec2-ip>
```

Then:

```bash
./smoke_test.sh <ec2-ip>      # should print health, activity, history, HTTP 200
```

Open `http://<ec2-ip>/` for the dashboard.

## Update an existing deployment

Copy the zip over and run the same script — it replaces the code, rebuilds the
dashboard and restarts the service:

```bash
cd ~/AWS_DEPLOY && ./deploy_ec2.sh <ec2-ip>
sudo journalctl -u radar-api -f     # watch the logs
```

To change only a setting, edit `/home/ubuntu/radar-backend/.env` and
`sudo systemctl restart radar-api`.

## API endpoints

| Method | Path | Used by | Purpose |
|---|---|---|---|
| POST | `/frames/batch` | Pi | Per-person fall features, one batch per JSON file. |
| POST | `/activity/pointcloud/batch` | Pi | Raw point clouds for activity recognition. |
| GET | `/activity?device_id=rpi-1` | dashboard | Current activity per person. |
| GET | `/history?device_id=rpi-1` | dashboard | Recent fall alerts. |
| GET | `/health` | both | Liveness check. |
| WS | `/ws` | dashboard | Live fall alerts pushed as they happen. |
| POST | `/frame`, `/activity`, `/activity/pointcloud` | — | Single-item versions, useful for testing. |

## Uploads are compressed

The Pi gzips every batch, so the backend must be able to unpack it — that is
built into `radar-api-main.py` in this zip (`GzipRequestMiddleware`). It also
still accepts plain JSON, and the Pi retries uncompressed if it gets a 4xx, so
a mismatch degrades rather than breaks. This takes the stream from 284 KB/s
down to 26 KB/s; see `LATENCY.md`.

## Storage

`USE_DYNAMODB=false` (default) keeps fall events in memory: they survive while
the service runs and are lost on restart. That is enough for a demo.

For persistence set `USE_DYNAMODB=true`, install `requirements-aws.txt`, create
a DynamoDB table named `radar_events` (partition key `pk` string, sort key `ts`
string) in `ap-south-1`, and attach an IAM role to the instance with
`dynamodb:PutItem` and `dynamodb:Query` on that table. No access keys in files.

## Configuration worth knowing

The fall-detection thresholds in `.env.example` were tuned against real
recordings on 16 Sep (7 real falls caught, 0 false alarms with two people in
the room). `SENSOR_ELEV_TILT_DEG=15` must match how the radar is physically
tilted; the rest is best left alone.

## Troubleshooting

| Symptom | Check |
|---|---|
| Dashboard shows "disconnected" | `sudo systemctl status radar-api`; is port 80 open in the security group? |
| No activity shown | Is the Pi's watcher running and printing sent counts? `curl "http://<ec2-ip>/activity?device_id=rpi-1"` |
| Pi prints `API REJECTED` | Version mismatch: redeploy the backend from this zip. |
| Dashboard loads, no data | `VITE_API_URL`/`VITE_WS_URL` in `frontend/.env` must point at the EC2 address; rebuild. |
| Falls never fire | `SENSOR_ELEV_TILT_DEG` wrong, or the radar config on the Pi is not `AOP_6m_default.cfg`. |
