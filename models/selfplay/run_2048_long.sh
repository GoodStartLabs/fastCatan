#!/usr/bin/env bash
# "Give the larger nets more time" — long self-play run for 2048,2048,1024 (17.3M).
# 16 rounds (3x the original 6; bounded because the 17M net's CPU opponent
# inference is ~3x slower than 512 -> ~14-18h). Same stable config as the 512
# long run (ent0.01 + linear lr). Warm-starts from the 50M vs-random 2048 seed.
# Waits for the 512 long run to finish so the two never overlap. Launch this
# AFTER reviewing the 512-long result (if more-time doesn't help 512, an even
# bigger net almost certainly won't either -> save ~16h).
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0
PY=/home/sinan/anaconda3/bin/python
DONE_512_LONG=models/checkpoints/arch_long/512-512-256_long/summary.json
RUN=arch_long/2048-2048-1024_long
LOG="models/checkpoints/$RUN/run.log"
mkdir -p "models/checkpoints/$RUN"

echo "[long2048] waiting for 512 long run to finish: $DONE_512_LONG"
while [ ! -f "$DONE_512_LONG" ]; do sleep 60; done
echo "[long2048] launching 16-round long run $(date '+%F %T')"

echo "=== 2048 LONG START $(date '+%F %T') ===" > "$LOG"
$PY -u -m models.selfplay.train_selfplay \
  --init-from models/checkpoints/seeds/2048-2048-1024/ppo_final.zip --seed-pool \
  --net-arch 2048,2048,1024 \
  --lr 3e-4 --ent-coef 0.01 --lr-schedule linear \
  --steps-per-round 1000000 --num-rounds 16 \
  --num-envs 8 --gate-lag 2 --gate-games 200 --seed 42 \
  --run-name "$RUN" >> "$LOG" 2>&1
echo "=== 2048 LONG DONE $(date '+%F %T') ===" >> "$LOG"
