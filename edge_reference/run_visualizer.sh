#!/usr/bin/env bash
# Radar GUI: reads the radar over USB, shows people + activity, and writes one
# JSON file per second into visualizer/radar_stream/ for the watcher.
# Needs a screen (run from the Pi's desktop, VNC, or `ssh -X`).
set -e
cd "$(dirname "$0")"
source ./edge.env
exec ./venv/bin/python visualizer/gui_activity.py "$@"
