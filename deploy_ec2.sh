#!/usr/bin/env bash
# One-shot deploy/update on an Ubuntu EC2 instance.
#
#   scp -i key.pem -r AWS_DEPLOY ubuntu@<ec2-ip>:~/
#   ssh -i key.pem ubuntu@<ec2-ip>
#   cd ~/AWS_DEPLOY && ./deploy_ec2.sh <ec2-public-ip>
#
# Re-running it is safe: it updates the code, rebuilds the dashboard and
# restarts the service.
set -e
EC2_IP="${1:-}"
if [ -z "$EC2_IP" ]; then
  echo "usage: ./deploy_ec2.sh <ec2-public-ip-or-hostname>"; exit 1
fi
APP=/home/ubuntu/radar-backend
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== system packages =="
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-pip nginx
if ! command -v node >/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi

echo "== backend =="
mkdir -p "$APP"
cp -r "$HERE/backend/." "$APP/"
cd "$APP"
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install --upgrade pip -q
./venv/bin/pip install -r requirements.txt -q
# Uncomment for DynamoDB:  ./venv/bin/pip install -r requirements-aws.txt -q
[ -f .env ] || cp .env.example .env

echo "== service =="
sudo cp "$APP/radar-api.service" /etc/systemd/system/radar-api.service
sudo systemctl daemon-reload
sudo systemctl enable radar-api
sudo systemctl restart radar-api

echo "== dashboard =="
cd "$HERE/frontend"
cat > .env <<ENV
VITE_API_URL=http://$EC2_IP
VITE_WS_URL=ws://$EC2_IP/ws
VITE_DEVICE_ID=rpi-1
ENV
npm ci --silent || npm install --silent
npm run build
sudo mkdir -p /var/www/radar
sudo rm -rf /var/www/radar/*
sudo cp -r dist/* /var/www/radar/

echo "== nginx =="
sudo cp "$APP/nginx-radar.conf" /etc/nginx/sites-available/radar
sudo ln -sf /etc/nginx/sites-available/radar /etc/nginx/sites-enabled/radar
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx

echo
echo "Deployed."
echo "  dashboard : http://$EC2_IP/"
echo "  API health: http://$EC2_IP/health"
echo "  logs      : sudo journalctl -u radar-api -f"
echo
echo "On the Raspberry Pi set in ~/radar_edge/edge.env:"
echo "  export CLOUD_API_URL=\"http://$EC2_IP/frame\""
