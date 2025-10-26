#!/usr/bin/env bash
set -euo pipefail

echo "[setup_backend] Creating Python virtual environment .venv"
python -m venv .venv
source .venv/bin/activate

echo "[setup_backend] Upgrading pip toolchain"
python -m pip install --upgrade pip setuptools wheel

echo "[setup_backend] Installing backend requirements"
pip install -r CalHacks-Agents/requirements.txt

echo "[setup_backend] Done. Activate with: source .venv/bin/activate"
