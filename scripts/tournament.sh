#!/usr/bin/env bash
# One-command reproducible tournament (spec 0.3 §C). Pure-fastcatan, built on the
# frozen eval_seats.play_one driver: paired boards + full 4-cyclic seat rotation,
# Wilson lower bound + trades-on/off delta, appends a row to results/ladder.parquet.
# The catanatron BRIDGE tournament is reserved for the final G2/G3 gate.
#
#   scripts/tournament.sh [--games N] [--seed S] [--newest CKPT] [--ladder-version V]
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
export PYTHONHASHSEED=0
export PYTHONPATH="EVAL:${PYTHONPATH:-}"
exec python -m bin.tournament "$@"
