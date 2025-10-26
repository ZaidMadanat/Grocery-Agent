#!/usr/bin/env bash
set -euo pipefail

# Ensure backend venv exists
if [ ! -d .venv ]; then
  echo "[run_backend] Missing .venv. Run scripts/setup_backend.sh first." >&2
  exit 1
fi

# Activate venv
source .venv/bin/activate

# Run from the backend project root so 'agents' is importable
cd CalHacks-Agents

# Ensure repo root is on PYTHONPATH (for imports like 'database', 'auth')
export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH:-}"

echo "[run_backend] Starting FastAPI at http://localhost:8000 (docs at /docs)"
exec uvicorn main:app --host 0.0.0.0 --port 8000
