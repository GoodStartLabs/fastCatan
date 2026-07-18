#!/usr/bin/env bash
# One-command PPO wiring smoke (~2 min): runs the shipped MaskablePPO trainer,
# asserts loss finite + throughput floor, and records a W&B run
# (project=goodsettler, name=wiring-smoke). Phase-0 "one command trains" proof.
#
# Needs WANDB_API_KEY (from the goodSettler .env). Usage:
#   WANDB_API_KEY=... bin/train_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
: "${WANDB_API_KEY:?set WANDB_API_KEY (from the goodSettler .env) before running}"
export PYTHONHASHSEED=0
export WANDB_PROJECT=goodsettler
export SMOKE_STEPS="${SMOKE_STEPS:-300000}"
export FPS_FLOOR="${FPS_FLOOR:-800}"
exec python -m bin.train_smoke
