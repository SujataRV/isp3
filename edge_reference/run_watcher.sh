#!/usr/bin/env bash
# Uploader: tails the JSON folder and posts features to the AWS backend.
set -e
cd "$(dirname "$0")"
source ./edge.env
mkdir -p "$WATCH_DIR"
exec ./venv/bin/python rpi_pipeline/aws_watcher.py "$WATCH_DIR" "$@"
