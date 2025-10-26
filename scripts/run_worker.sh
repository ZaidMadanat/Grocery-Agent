#!/usr/bin/env bash
set -euo pipefail

if [ ! -d .venv-worker ]; then
  echo "[run_worker] Missing .venv-worker. Run scripts/setup_worker.sh first." >&2
  exit 1
fi

source .venv-worker/bin/activate
exec python Cooking-Companion/cooking_companion.py
