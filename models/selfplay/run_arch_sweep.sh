#!/usr/bin/env bash
# M3 arch sweep on the 1084/286 interface. Two stages:
#   1. per-arch vs-random M2 seeds (so every arch warm-starts from a base of its
#      OWN architecture — a single --init-from would mismatch non-default arches
#      and cold-start them, which sparse +-1 reward can't recover from).
#   2. warm-started self-play sweep over lr x ent x sched x ARCH, with p2p trades
#      ON. Stall control (validated 2026-05-29): the C++ MAX_TURNS=2000 length cap
#      (state.hpp) is the decidability fix; the per-seat trade-compose LIVENESS cap
#      (CAP, forces turns to end so turn_count advances) + -2 tie reward are the
#      supporting pieces. The old step-based length cap (5000) guillotined
#      trade-heavy games (which need 7k-61k steps) -> 50-100% no-winner; moving the
#      length authority to turns fixed it (smoke: 0% no-winner on the worst cells).
#
# Run detached:  tmux new -s sweep -c <repo> 'bash models/selfplay/run_arch_sweep.sh'
# Idempotent: existing seeds are reused, so a re-run resumes at the sweep.
set -euo pipefail
cd "$(dirname "$0")/../.."                       # -> repo root
export PYTHONHASHSEED=0
PY=/home/sinan/anaconda3/bin/python

# --- knobs (edit to trim cost) ------------------------------------------
SEED_DIR=models/checkpoints/seeds
SEED_STEPS=50000000        # match the existing 64,64 seed (ppo_1084_50m) => fair
SEED_ENVS=768
ARCHES=("64,64" "128,128" "256,256")   # drop 256,256 to ~halve the sweep
CAP=50                     # per-seat trade-compose LIVENESS cap (C++ MAX_TURNS is the length authority)
OUT=models/checkpoints/arch_sweep
mkdir -p "$OUT"

# --- 1. per-arch seeds (64,64 reuses the existing 50M run via symlink) ---
mkdir -p "$SEED_DIR/64-64"
ln -sf "$PWD/models/checkpoints/ppo_1084_50m/ppo_final.zip" "$SEED_DIR/64-64/ppo_final.zip"
for arch in "${ARCHES[@]}"; do
  tag=${arch//,/-}
  if [[ -f "$SEED_DIR/$tag/ppo_final.zip" ]]; then
    echo "[seeds] $tag: exists -> reuse"; continue
  fi
  echo "[seeds] training $tag ($SEED_STEPS steps vs random, $SEED_ENVS envs)..."
  $PY -m models.train_ppo --net-arch "$arch" --total-steps "$SEED_STEPS" \
      --num-envs "$SEED_ENVS" --seed 42 \
      --save-dir "$SEED_DIR" --run-name "$tag" 2>&1 | tee "$OUT/m2seed_${tag}.log"
done

# --- 2. warm-started self-play sweep, trades ON, per-arch seeds ----------
$PY -m models.selfplay.sweep \
  --init-dir "$SEED_DIR" --seed-pool --trade-compose-cap "$CAP" \
  --lr 3e-4 --ent-coef 0.0 0.01 \
  --steps-per-round 1000000 \
  --net-arch "${ARCHES[@]}" \
  --lr-schedule constant linear --target-kl none \
  --num-rounds 6 --num-envs 8 --gate-lag 2 --gate-games 200 --seed 42 \
  --out-dir "$OUT" 2>&1 | tee "$OUT/sweep.log"

echo "[done] results -> $OUT/sweep_results.md"
