#!/usr/bin/env bash
# Chained 2048,2048,1024 arch-sweep point (extends the M3 capacity curve past
# 512,512,256). Self-sequences so it can be launched now and left overnight:
#   1. wait for the 2048,2048,1024 vs-random seed (trained separately) to land,
#   2. wait for the running 512,512,256 sweep to finish (its sweep_results.md
#      appears) so two self-play sweeps never run concurrently (load>24 -> both
#      crawl; one self-play sweep already pulls ~13 cores via torch CPU matmul),
#   3. run the 4-cell sweep (ent {0,0.01} x sched {const,linear}) with the SAME
#      protocol as arch_sweep_xl for an apples-to-apples curve point.
# Signals are file-existence (PID-reuse proof). Idempotent-ish: re-running before
# the seed/512-sweep finish just re-waits.
set -uo pipefail
cd "$(dirname "$0")/../.."                       # -> repo root
export PYTHONHASHSEED=0
PY=/home/sinan/anaconda3/bin/python

SEED=models/checkpoints/seeds/2048-2048-1024/ppo_final.zip
DONE_512=models/checkpoints/arch_sweep_xl/sweep_results.md
OUT=models/checkpoints/arch_sweep_xxl
LOG="$OUT/sweep.log"
mkdir -p "$OUT"

echo "[chain] waiting for 2048 seed: $SEED"
while [ ! -f "$SEED" ]; do sleep 60; done
echo "[chain] seed ready $(date '+%F %T')"

echo "[chain] waiting for 512 sweep to finish: $DONE_512"
while [ ! -f "$DONE_512" ]; do sleep 60; done
echo "[chain] 512 sweep done $(date '+%F %T') -> launching 2048,2048,1024 sweep"

echo "=== arch_sweep_xxl START $(date '+%F %T') ===" > "$LOG"
$PY -u -m models.selfplay.sweep \
  --init-dir models/checkpoints/seeds --seed-pool \
  --lr 3e-4 --ent-coef 0.0 0.01 \
  --steps-per-round 1000000 \
  --net-arch 2048,2048,1024 \
  --lr-schedule constant linear --target-kl none \
  --num-rounds 6 --num-envs 8 --gate-lag 2 --gate-games 200 --seed 42 \
  --out-dir "$OUT" >> "$LOG" 2>&1
echo "=== arch_sweep_xxl DONE $(date '+%F %T') ===" >> "$LOG"
