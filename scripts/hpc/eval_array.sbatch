#!/usr/bin/env bash
#SBATCH --job-name=fc-eval
#SBATCH --partition=standard          # TODO [verify]: sinfo for the CPU partition name
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00               # >=3x the expected shard time
#SBATCH --output=slurm-%A_%a.out
#
# Sharded bridge evaluation. Each array task runs GAMES_PER_SHARD games with
# a disjoint seed range (per-game seed = args.seed + g, so shard i starts at
# BASE_SEED + i*GAMES_PER_SHARD), writing its JSON into a per-shard out dir.
# Pool afterwards with scripts/hpc/merge_results.py.
#
# Submit (see scripts/hpc/README.md for full examples):
#   sbatch --array=0-19 --export=ALL,GAMES_PER_SHARD=50,BASE_SEED=42,\
#     DRIVER=AB.mixed_tournament,EXTRA_ARGS="--n-agents 2 --policy mcts ..." \
#     scripts/hpc/eval_array.sbatch
set -euo pipefail

REPO=${REPO:-$HOME/fastCatan}
GAMES_PER_SHARD=${GAMES_PER_SHARD:-50}
BASE_SEED=${BASE_SEED:-42}
DRIVER=${DRIVER:-AB.mixed_tournament}     # AB.mixed_tournament | AB.tournament
EXTRA_ARGS=${EXTRA_ARGS:-}

# Env activation: pass VENV=/path/to/venv for a python-3.12 venv, else the
# conda env `catan` is activated (CONDA_SH = conda's profile.d script).
VENV=${VENV:-}
CONDA_SH=${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}
if [[ -n "$VENV" ]]; then
  source "$VENV/bin/activate"
else
  source "$CONDA_SH"
  conda activate catan
fi
cd "$REPO"

SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID * GAMES_PER_SHARD))
OUTDIR="EVAL/AB/results/hpc_${SLURM_ARRAY_JOB_ID}/shard_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTDIR"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[shard $SLURM_ARRAY_TASK_ID] driver=$DRIVER games=$GAMES_PER_SHARD seed=$SEED"
PYTHONPATH=EVAL python -m "$DRIVER" \
    --games "$GAMES_PER_SHARD" --seed "$SEED" --out "$OUTDIR" $EXTRA_ARGS
