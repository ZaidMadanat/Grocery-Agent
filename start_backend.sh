#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/CalHacks-Agents"

if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  # Reuse the repo-managed virtualenv when available
  source "$ROOT_DIR/.venv/bin/activate"
fi

export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
echo "🚀 Starting Backend API on http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
