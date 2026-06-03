#!/usr/bin/env bash
# "Give the larger nets more time" — long self-play run for 512,512,256.
# Replaces the 6-round sweep cell with a 24-round run on the most stable config
# (ent0.01 + LINEAR lr decay: the constant-lr cells regressed hardest, linear is
# the recipe's built-in late-round stabiliser). Single config, not a sweep, so
# the extra rounds are affordable. Warm-starts from the same 50M vs-random seed.
# Chained: waits for the running 512 6-round sweep to finish first (no two
# self-play runs at once -> oversubscription).
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0
PY=/home/sinan/anaconda3/bin/python
DONE_512=models/checkpoints/arch_sweep_xl/sweep_results.md
RUN=arch_long/512-512-256_long
LOG="models/checkpoints/$RUN/run.log"
mkdir -p "models/checkpoints/$RUN"

echo "[long512] waiting for 6-round 512 sweep to finish: $DONE_512"
while [ ! -f "$DONE_512" ]; do sleep 60; done
echo "[long512] launching 24-round long run $(date '+%F %T')"

echo "=== 512 LONG START $(date '+%F %T') ===" > "$LOG"
$PY -u -m models.selfplay.train_selfplay \
  --init-from models/checkpoints/seeds/512-512-256/ppo_final.zip --seed-pool \
  --net-arch 512,512,256 \
  --lr 3e-4 --ent-coef 0.01 --lr-schedule linear \
  --steps-per-round 1000000 --num-rounds 24 \
  --num-envs 8 --gate-lag 2 --gate-games 200 --seed 42 \
  --run-name "$RUN" >> "$LOG" 2>&1
echo "=== 512 LONG DONE $(date '+%F %T') ===" >> "$LOG"
