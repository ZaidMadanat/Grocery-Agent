#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.venv-worker/bin/activate" ]; then
  source "$ROOT_DIR/.venv-worker/bin/activate"
fi

echo "🎙️  Starting Voice Worker (LiveKit + Claude)"
python Cooking-Companion/cooking_companion.py dev
