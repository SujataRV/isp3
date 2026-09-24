# Radar edge unit (Raspberry Pi)

This Pi does two jobs and nothing else:

1. **`visualizer/gui_activity.py`** — talks to the mmWave radar over USB, shows
   the 3D point cloud, people count and per-person activity, and writes one
   small JSON file per second into `visualizer/radar_stream/`.
2. **`rpi_pipeline/aws_watcher.py`** — notices each new JSON file (inotify, so
   within milliseconds), turns it into per-person features, and POSTs them to
   the backend on AWS.

Everything else — fall detection, the activity model that feeds the dashboard,
the database and the web dashboard — runs on AWS. See `AWS_DEPLOY/README_AWS.md`
in the zip on the laptop.

```
radar  --USB-->  visualizer  --JSON file/sec-->  watcher  --HTTPS-->  AWS API  -->  dashboard
                                        (this Pi)                       (cloud)
```

## One-time setup

```bash
cd ~/radar_edge
./setup_edge.sh
```

Creates `venv/` (reusing the Pi's system PyQt5/numpy/pyserial/PyOpenGL) and
installs pyqtgraph, xgboost, scikit-learn, joblib, watchdog and requests.

## Point it at the cloud

Edit `edge.env` — it is the only file to change:

```bash
export CLOUD_API_URL="http://<ec2-public-ip>/frame"   # backend address
export DEVICE_ID="rpi-1"                              # name of this radar unit
```

## Running

Plug the radar in first. It shows up as two USB serial ports, normally
`/dev/ttyACM0` (data) and `/dev/ttyACM1` (config/CLI).

Terminal 1 — the GUI (needs a screen: the Pi's desktop, VNC, or `ssh -X`):

```bash
cd ~/radar_edge && ./run_visualizer.sh
```

In the GUI: pick the two COM ports, press **Connect**, then **Start**, then
**Stream to isp3** so it starts writing JSON files.

Terminal 2 — the uploader:

```bash
cd ~/radar_edge && ./run_watcher.sh
```

It prints one line per file with how many frames were sent. `--process-existing`
also uploads files already in the folder.

## Bandwidth

Each second of radar uploads about **26 KB** (roughly 90 MB per hour): batches
are gzipped and coordinates trimmed to millimetres, down from 284 KB raw. Turn
that off with `GZIP_UPLOAD=false` in `edge.env` only if you are talking to an
old backend that cannot unpack it.

## Housekeeping

`visualizer/radar_stream/` is a scratch folder. The GUI's **Stop & delete**
button empties it; deleting it by hand is fine too.

```bash
rm -f ~/radar_edge/visualizer/radar_stream/*.json
```

## Checking it works

```bash
ls -l ~/radar_edge/visualizer/radar_stream | tail    # new file about every second
curl http://<ec2-public-ip>/health                   # backend reachable
curl "http://<ec2-public-ip>/activity?device_id=rpi-1"
```

If the watcher prints `API REJECTED` or connection errors, the backend address
in `edge.env` is wrong or the EC2 security group is not allowing port 80.

## Layout

```
radar_edge/
  edge.env              <- the only file you normally edit
  setup_edge.sh         <- one-time install
  run_visualizer.sh     <- job 1 (GUI + JSON writer)
  run_watcher.sh        <- job 2 (uploader)
  visualizer/           <- TI visualizer + activity overlay
    gui_activity.py, gui_parser.py, gui_threads.py, gui_common.py,
    parseFrame.py, parseTLVs.py, graphUtilities.py, gl_classes.py,
    activity_predictor.py, stream_writer.py,
    AOP_6m_default.cfg  <- radar chirp config, auto-loaded
    model/              <- activity model for the on-screen labels
    radar_stream/       <- JSON bridge folder (created at run time)
  rpi_pipeline/
    aws_watcher.py, feature_extract.py, config.py
```

Note: the GUI's on-screen activity label uses the local copy of the model in
`visualizer/model/`. The dashboard's activity panel is computed separately on
AWS from the uploaded point clouds. Both use the same model file, so keep them
in sync if you retrain.
