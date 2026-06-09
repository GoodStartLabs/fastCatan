#!/usr/bin/env bash
# Local shard driver for the SELF-CONTAINED mixed-table arms:
# {2v2, 3v1} x {trades-ON, trades-OFF}, MCTS seats with learned leaf value
# (--leaf-eval net) + learned in-tree opponent (--model-opp net) — no
# ab_value/ab_decide at inference (the M4 two-tier self-containment rule).
#
# Mirrors scripts/hpc/eval_array.sbatch seeding (shard i seed = BASE+i*GPS)
# so local and HPC shards pool interchangeably via merge_results.py. The
# same BASE window across arms is deliberate: ON and OFF arms of the same
# table replay identical board/dice seeds — a paired contrast on the
# trading delta.
#
#   bash scripts/run_mixed_arms_local.sh
#   SHARDS=12 GPS=84 SIMS=512 bash scripts/run_mixed_arms_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${CKPT:-models/checkpoints/il_ab_d2_640k_vpm_ep10/il_final.pt}
SIMS=${SIMS:-512}
SHARDS=${SHARDS:-12}
GPS=${GPS:-84}                    # games per shard (12*84 = 1008/arm)
BASE=${BASE:-42}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT=${OUT_ROOT:-EVAL/AB/results/arms_selfcontained_$STAMP}
PY=.venv/bin/python

run_arm () {            # $1 = n_agents, $2 = on|off
  local n=$1 trade=$2 extra=""
  [[ $trade == off ]] && extra="--no-trades"
  local out="$OUT_ROOT/${n}v$((4-n))_${trade}"
  mkdir -p "$out" DEBUG/logs
  echo "=== arm ${n}v$((4-n)) trades-$trade  shards=$SHARDS x $GPS -> $out"
  for i in $(seq 0 $((SHARDS-1))); do
    PYTHONHASHSEED=0 PYTHONPATH=EVAL OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    nice -n 10 $PY -m AB.mixed_tournament \
      --n-agents "$n" --policy mcts --ckpt "$CKPT" \
      --leaf-eval net --model-opp net --mcts-sims "$SIMS" \
      --ab-depth 2 --ab-prune $extra \
      --games "$GPS" --seed $((BASE + i*GPS)) --out "$out" \
      > "DEBUG/logs/arm_${n}v$((4-n))_${trade}_shard${i}_$STAMP.log" 2>&1 &
  done
  wait
  $PY scripts/hpc/merge_results.py "$out/*.json" | tee "$out/MERGED.txt"
}

# Cheap OFF baselines first (early numbers), then the long ON arms.
run_arm 2 off
run_arm 3 off
run_arm 2 on
run_arm 3 on
echo "ALL ARMS DONE -> $OUT_ROOT"
