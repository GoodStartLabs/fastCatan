#!/usr/bin/env bash
#
# Reactive-RL baseline benchmark: PPO vs A2C vs DQN, each trained N steps vs
# random with NO p2p trades, then evaluated over GAMES games vs random.
# Writes one JSON per algo + a summary.md / summary.csv table. Run once, reuse
# the results forever.
#
#   bash scripts/run_baseline_benchmark.sh                 # full 50M run (~5h)
#   STEPS=2000000 GAMES=200 bash scripts/run_baseline_benchmark.sh   # quick test
#
# Overridable via env: PY STEPS GAMES SEED NET PPO_ENVS A2C_ENVS DQN_ENVS RESULTS
#
# Parity: same env (1084 obs / 286 act), same uniform-random opponents, same
# sparse +1/-1/-2 reward, gamma 0.999, two 256-wide hidden layers, trades OFF
# (p2p suppressed, maritime on), same step budget. Eval = 1000 games, sampling
# actions (the M2 convention), seed 12345, Wilson 95% CI.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
STEPS=${STEPS:-50000000}
GAMES=${GAMES:-1000}
SEED=${SEED:-42}
NET=${NET:-256,256}
PPO_ENVS=${PPO_ENVS:-32}
A2C_ENVS=${A2C_ENVS:-8}
DQN_ENVS=${DQN_ENVS:-16}
RESULTS=${RESULTS:-models/benchmarks/baselines_vs_random}
NAME=${NAME:-baseline}      # checkpoint-dir prefix (override for smoke runs)
CKPT=models/checkpoints
TAG="${STEPS} steps vs random, no-trades"

mkdir -p "$RESULTS"
export PYTHONUNBUFFERED=1
exec > >(tee -a "$RESULTS/run.log") 2>&1

echo "######## baseline benchmark | steps=$STEPS games=$GAMES net=$NET seed=$SEED"
echo "######## $(date -u +%FT%TZ) | python=$PY | results=$RESULTS"

eval_algo () {  # algo ckpt envs
  local algo=$1 ckpt=$2 envs=$3 secs=$4
  echo "======== $algo : EVAL ($GAMES games vs random, no-trades) ========"
  $PY -m models.eval --algo "$algo" --ckpt "$ckpt" --games "$GAMES" \
    --no-trades --seed 12345 --tag "$TAG" \
    --train-steps "$STEPS" --train-seconds "$secs" \
    --train-net-arch "$NET" --train-num-envs "$envs" \
    --out "$RESULTS/$algo.json"
}

# ---------------- PPO (MaskablePPO, GPU) ----------------
echo "======== ppo : TRAIN ($STEPS steps) ========"
t0=$(date +%s)
$PY -m models.train_ppo --num-envs "$PPO_ENVS" --total-steps "$STEPS" \
  --net-arch "$NET" --no-trades --seed "$SEED" \
  --run-name "${NAME}_ppo_nt" --save-freq 10000000
eval_algo ppo "$CKPT/${NAME}_ppo_nt/ppo_final.zip" "$PPO_ENVS" "$(( $(date +%s) - t0 ))"

# ---------------- A2C (custom, CPU) ----------------
echo "======== a2c : TRAIN ($STEPS steps) ========"
t0=$(date +%s)
$PY -m models.train_a2c --num-envs "$A2C_ENVS" --total-steps "$STEPS" \
  --no-trades --seed "$SEED" --save-dir "$CKPT/${NAME}_a2c_nt"
eval_algo a2c "$CKPT/${NAME}_a2c_nt/a2c_final.pt" "$A2C_ENVS" "$(( $(date +%s) - t0 ))"

# ---------------- DQN (custom, multi-env GPU) ----------------
echo "======== dqn : TRAIN ($STEPS steps) ========"
t0=$(date +%s)
$PY -m models.train_dqn --num-envs "$DQN_ENVS" --total-steps "$STEPS" \
  --no-trades --seed "$SEED" --save-dir "$CKPT/${NAME}_dqn_nt" \
  --save-freq 10000000
eval_algo dqn "$CKPT/${NAME}_dqn_nt/dqn_final.pt" "$DQN_ENVS" "$(( $(date +%s) - t0 ))"

# ---------------- summary table ----------------
echo "======== SUMMARY ========"
$PY models/benchmarks/summarize.py "$RESULTS" \
  --title "Reactive RL baselines vs random ($STEPS steps, no-trades)"
echo "######## DONE $(date -u +%FT%TZ) — see $RESULTS/summary.md"
