#!/usr/bin/env bash
# Spec 3.2b one-command gate.  This lives on the editable research surface;
# substrate-v1 freezes every bin/ path, including additions.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHONHASHSEED=0
if tmux has-session -t repro31-r3-stage3 2>/dev/null; then
  DEFAULT_THREADS=2
else
  DEFAULT_THREADS="$(nproc)"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$DEFAULT_THREADS}"
if tmux has-session -t repro31-r3-stage3 2>/dev/null \
    && (( OMP_NUM_THREADS > 2 )); then
  echo "repro31-r3-stage3 is active; OMP_NUM_THREADS must be <=2" >&2
  exit 2
fi
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SMOKE_SECONDS="${BATCHED_SMOKE_SECONDS:-1800}"
SMOKE_ENVS="${BATCHED_SMOKE_ENVS:-4000}"
DEFAULT_ENV_WORKERS="$(( (OMP_NUM_THREADS + 1) / 2 ))"
ENV_WORKERS="${BATCHED_ENV_WORKERS:-$DEFAULT_ENV_WORKERS}"
THROUGHPUT_FLOOR="${BATCHED_THROUGHPUT_FLOOR:-250000}"
LEAK_GAMES="${BATCHED_LEAK_GAMES:-10000}"
LEAK_WORKERS="${BATCHED_LEAK_WORKERS:-$OMP_NUM_THREADS}"
PROMOTION_GAMES="${BATCHED_PROMOTION_GAMES:-256}"
OUT_DIR="${BATCHED_SMOKE_OUT:-models/checkpoints/batched_smoke}"
mkdir -p "$OUT_DIR"

python -m pytest -q tests/test_batched_trainer.py tests/test_wiring.py
PYTHONPATH=EVAL python -m bridge.leakage_referee \
  --games "$LEAK_GAMES" --workers "$LEAK_WORKERS" --seed 0 \
  --sample-every 12 --out "$OUT_DIR/leakage.json"

if [[ -f "$HOME/goodSettler/wandb.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOME/goodSettler/wandb.env"
  set +a
fi

python -m models.batched_trainer.train \
  --num-envs "$SMOKE_ENVS" \
  --env-workers "$ENV_WORKERS" \
  --split-env-affinity \
  --duration-seconds "$SMOKE_SECONDS" \
  --opponents random \
  --rollout-decisions 262144 \
  --batch-size 65536 \
  --update-epochs 1 \
  --assert-floor "$THROUGHPUT_FLOOR" \
  --promotion-games "$PROMOTION_GAMES" \
  --assert-promotion-moves \
  --wandb-mode online \
  --wandb-entity good-start-labs \
  --wandb-project goodsettler-rl \
  --checkpoint-out "$OUT_DIR/final.pt" \
  --json-out "$OUT_DIR/summary.json"
