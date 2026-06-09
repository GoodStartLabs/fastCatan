#!/usr/bin/env bash
#SBATCH --job-name=fc-train
#SBATCH --partition=gpu               # TODO [verify]: sinfo for the GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00               # TODO [verify]: partition walltime limit
#SBATCH --output=slurm-%j.out
#
# Single-node training template — pass the full command via CMD:
#   sbatch --export=ALL,CMD="python -m models.train_ppo --total-steps 50000000 ..." \
#       scripts/hpc/train.sbatch
#
# Runs longer than the walltime: chain with checkpoint-resume —
#   jid=$(sbatch --parsable --export=ALL,CMD="..." scripts/hpc/train.sbatch)
#   sbatch --dependency=afterok:$jid --export=ALL,CMD="... --resume ..." scripts/hpc/train.sbatch
set -euo pipefail

REPO=${REPO:-$HOME/fastCatan}
: "${CMD:?pass the training command via --export=ALL,CMD=\"...\"}"

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

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[train] $CMD"
eval "$CMD"
