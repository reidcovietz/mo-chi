#!/bin/bash
# Polls GitHub for new commits and auto-deploys if behind.
# Run via cron: * * * * * /home/reid/mo-chi/auto-deploy.sh >> /home/reid/mo-chi/auto-deploy.log 2>&1

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$REPO_DIR"
git fetch origin main -q

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" != "$REMOTE" ]; then
  echo "[$(date)] New commits detected — pulling and restarting..."
  git pull origin main -q
  sudo systemctl restart mo-chi
  echo "[$(date)] Done."
fi
