#!/usr/bin/env bash
#SBATCH --job-name=fc-selfplay
#SBATCH --partition=gpu               # TODO [verify]: sinfo for the GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00               # TODO [verify]: partition walltime limit
#SBATCH --output=slurm-selfplay-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL    # email on start / finish / crash
#SBATCH --mail-user=bove@campus.tu-berlin.de
#
# AlphaZero SELF-PLAY training (GPU-batched, models/alphazero/batched_selfplay).
# The az-scratch-vpm arm: pure self-play, dense vp_margin values, NO AlphaBeta
# data anywhere — tests whether self-play alone reaches the learned-judge band.
#
# Resume-safe: if SAVE_DIR already holds checkpoints, the newest az_g*.pt is
# picked up via --init-from automatically, so chained jobs just continue:
#
#   jid=$(sbatch --parsable scripts/hpc/selfplay.sbatch)
#   sbatch --dependency=afterany:$jid scripts/hpc/selfplay.sbatch   # chain more walltime
#
# Tune via --export (all optional):
#   sbatch --export=ALL,GAMES=60000,SIMS=128,NUM_GAMES=256,HIDDEN=512,512,256 \
#       scripts/hpc/selfplay.sbatch
set -euo pipefail

# Repo: default to the directory you ran `sbatch` from (works no matter where the
# repo lives), NOT a hardcoded $HOME/fastCatan. Override with REPO=... if needed.
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO" || { echo "[FATAL] cannot cd to REPO=$REPO"; exit 1; }
echo "[env] host=$(hostname) REPO=$REPO"

# Optional environment modules (clusters WITHOUT conda): pass e.g.
#   MODULES="python/3.12 gcc/14"   (also needed at runtime for editable.rebuild)
MODULES=${MODULES:-}
if [[ -n "$MODULES" ]]; then
  command -v module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null || true
  module load $MODULES || { echo "[FATAL] module load '$MODULES' failed"; exit 1; }
  echo "[env] modules: $MODULES"
fi

# Env activation, in order of preference: explicit VENV -> explicit CONDA_SH ->
# auto-find a conda in the usual spots -> `conda` already on PATH. Fail loudly.
if [[ -n "${VENV:-}" ]]; then
  source "$VENV/bin/activate"
else
  CONDA_SH=${CONDA_SH:-}
  if [[ -z "$CONDA_SH" ]]; then
    for c in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
      [[ -f "$c/etc/profile.d/conda.sh" ]] && CONDA_SH="$c/etc/profile.d/conda.sh" && break
    done
  fi
  if [[ -n "$CONDA_SH" && -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "[FATAL] no conda found — pass CONDA_SH=/path/to/etc/profile.d/conda.sh or VENV=/path/to/venv"; exit 1
  fi
  conda activate "${CONDA_ENV:-catan}"
fi

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Fail fast with a clear message if the native build / torch is not importable on
# THIS compute node (e.g. an -march=native .so built on a different-CPU login node).
echo "[env] python=$(command -v python) ($(python --version 2>&1))"
python -c "import fastcatan, torch; print('[env] fastcatan', fastcatan.OBS_SIZE, fastcatan.NUM_ACTIONS, '| cuda', torch.cuda.is_available())" \
  || { echo '[FATAL] fastcatan/torch import failed on this node — rebuild on a compute node (scripts/hpc/README.md §2, -march=native caveat)'; exit 1; }

GAMES=${GAMES:-60000}                  # games THIS job plays (chain jobs to add more)
NUM_GAMES=${NUM_GAMES:-256}            # concurrent games per lockstep batch
SIMS=${SIMS:-128}                      # MCTS simulations per decision
HIDDEN=${HIDDEN:-512,512,256}
SEED=${SEED:-20260611}
SAVE_DIR=${SAVE_DIR:-models/checkpoints/az_scratch_vpm_hpc}
EVAL_GAMES=${EVAL_GAMES:-200}          # final native AB-d2 ladder size
ALLOW_TRADES=${ALLOW_TRADES:-0}        # 1 = p2p trades ON in self-play (maritime always on)

mkdir -p "$SAVE_DIR"

TRADES_FLAG=""
[[ "$ALLOW_TRADES" == "1" ]] && TRADES_FLAG="--allow-trades"

# Resume: newest checkpoint in SAVE_DIR (az_g<NUM>.pt or az_final.pt) wins.
INIT=""
latest=$(ls -t "$SAVE_DIR"/az_*.pt 2>/dev/null | head -1 || true)
if [[ -n "$latest" ]]; then
  INIT="--init-from $latest"
  echo "[resume] continuing from $latest"
fi

echo "[selfplay] games=$GAMES G=$NUM_GAMES sims=$SIMS hidden=$HIDDEN trades=$([[ -n $TRADES_FLAG ]] && echo on || echo off) -> $SAVE_DIR"
python -m models.alphazero.batched_selfplay \
    --total-games "$GAMES" \
    --num-games "$NUM_GAMES" \
    --sims "$SIMS" \
    --hidden "$HIDDEN" \
    --value-mode vp_margin \
    --checkpoint-every 2000 \
    --eval-every 2000 \
    --seed "$SEED" \
    --device cuda \
    --save-dir "$SAVE_DIR" \
    $TRADES_FLAG $INIT

# Headline number: native AB-d2 dev ladder on the final net (CPU-bound but
# fine to run on the GPU node — it is one process, ~hours at 512 sims).
# Band to beat (2026-06): self-contained learned-judge cells 16.5-20.0%,
# parity 25%, hybrid reference 29.5%.
echo "[ladder] ${EVAL_GAMES}g vs native AB-d2 @512 sims (trades match training)"
python -m models.alphazero.evaluate \
    --ckpt "$SAVE_DIR/az_final.pt" \
    --opponent alphabeta --ab-depth 2 --sims 512 \
    --games "$EVAL_GAMES" --device cpu $TRADES_FLAG \
    | tee "$SAVE_DIR/ladder_abd2.txt"
