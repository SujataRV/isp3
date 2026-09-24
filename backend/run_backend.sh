#!/usr/bin/env bash
# Start the API in the foreground (development). For production use systemd:
#   sudo systemctl start radar-api
set -e
cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a
exec ./venv/bin/uvicorn radar-api-main:app --host 0.0.0.0 --port 8000
