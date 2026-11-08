#!/usr/bin/env bash
# Start the FastAPI backend for local development.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && source .env; set +a
export PYTHONPATH="python${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m uvicorn attraction_engine.api.main:app \
    --host "${AE_HOST:-0.0.0.0}" --port "${AE_PORT:-8000}" \
    --proxy-headers --forwarded-allow-ips='*' "$@"
