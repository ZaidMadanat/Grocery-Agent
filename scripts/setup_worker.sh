#!/usr/bin/env bash
set -euo pipefail

echo "[setup_worker] Creating Python virtual environment .venv-worker"
python -m venv .venv-worker
source .venv-worker/bin/activate

echo "[setup_worker] Upgrading pip toolchain"
python -m pip install --upgrade pip setuptools wheel

echo "[setup_worker] Installing worker requirements"
pip install -r Cooking-Companion/requirements.txt

echo "[setup_worker] Done. Activate with: source .venv-worker/bin/activate"
