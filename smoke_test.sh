#!/usr/bin/env bash
# Verify a deployment answers correctly. Run from anywhere:
#   ./smoke_test.sh <ec2-public-ip>
set -e
H="${1:?usage: ./smoke_test.sh <ec2-ip>}"
echo "health:"; curl -fsS "http://$H/health"; echo
echo "activity:"; curl -fsS "http://$H/activity?device_id=rpi-1"; echo
echo "history:"; curl -fsS "http://$H/history?device_id=rpi-1&limit=3"; echo
echo "dashboard:"; curl -fsS -o /dev/null -w "HTTP %{http_code}\n" "http://$H/"
