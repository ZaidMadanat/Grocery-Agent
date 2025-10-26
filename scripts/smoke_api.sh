#!/usr/bin/env bash
set -euo pipefail

TOKEN="${TOKEN:-}"
BASE_URL="${BASE_URL:-http://localhost:8000}"

if [ -z "$TOKEN" ]; then
  echo "[smoke_api] Set TOKEN env var with a valid Bearer token." >&2
  exit 1
fi

echo "[smoke_api] Checking health"
curl -fsSL "$BASE_URL/health" >/dev/null || { echo "Health check failed"; exit 1; }

echo "[smoke_api] Generating daily meals for Monday"
curl -fsSL -X POST "$BASE_URL/daily-meals/generate-by-day?day=Monday" \
  -H "Authorization: Bearer $TOKEN" >/dev/null || { echo "Daily meals failed"; exit 1; }

echo "[smoke_api] Minting LiveKit token"
curl -fsSL -X POST "$BASE_URL/session/create" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"room":"cooking-demo","ttl_seconds":3600}' >/dev/null || { echo "Token mint failed"; exit 1; }

echo "[smoke_api] OK"
