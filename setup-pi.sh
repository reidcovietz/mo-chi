#!/bin/bash
# Run this on the Raspberry Pi to install and enable mo-chi as a boot service.
# Usage: bash setup-pi.sh

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Creating Python venv..."
python3 -m venv "$REPO_DIR/venv"
"$REPO_DIR/venv/bin/pip" install --upgrade pip -q
"$REPO_DIR/venv/bin/pip" install -r "$REPO_DIR/requirements.txt" -q

echo "==> Checking .env..."
if [ ! -f "$REPO_DIR/.env" ]; then
  cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
  echo "    Created .env from example — edit it and add your API keys before starting."
fi

echo "==> Installing systemd service..."
# Patch the service file with the actual user and path
sed \
  -e "s|User=pi|User=$(whoami)|g" \
  -e "s|/home/pi/mo-chi|$REPO_DIR|g" \
  "$REPO_DIR/mo-chi.service" \
  | sudo tee /etc/systemd/system/mo-chi.service > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable mo-chi
sudo systemctl restart mo-chi

echo ""
echo "==> Done! mo-chi is running and will start on every boot."
echo "    Check status : sudo systemctl status mo-chi"
echo "    View logs    : sudo journalctl -u mo-chi -f"
echo "    Open in browser: http://$(hostname -I | awk '{print $1}'):8000"
