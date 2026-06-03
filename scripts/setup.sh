#!/usr/bin/env bash
# One-command, idempotent setup for fastCatan on ANY device (macOS / Linux).
# Canonical env: conda env `catan` = CPython 3.12 + numpy 1.26.4 (see requirements.txt).
#
#   bash scripts/setup.sh                          # CPU / macOS (MPS) torch
#   FASTCATAN_CUDA=cu124 bash scripts/setup.sh     # Linux CUDA box: force cu124 torch wheel
#   FASTCATAN_ENV=other  bash scripts/setup.sh     # use a different conda env name
#
# Re-run any time: the conda env is updated in place, the native ext rebuilt, the
# env verified against scripts/check_env.py. Safe — it never touches `base`.
set -euo pipefail

ENV_NAME="${FASTCATAN_ENV:-catan}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 1. conda env with the canonical interpreter (idempotent).
if conda env list | grep -qE "^[[:space:]]*${ENV_NAME}[[:space:]]"; then
  echo "[setup] conda env '${ENV_NAME}' exists — ensuring python=3.12…"
  conda install -n "${ENV_NAME}" -y --override-channels -c conda-forge python=3.12 >/dev/null
else
  echo "[setup] creating conda env '${ENV_NAME}' (python=3.12, conda-forge only)…"
  # Explicit --override-channels -c conda-forge (NOT `conda env create -f environment.yml`):
  # the global condarc forces `defaults` (pkgs/main, pkgs/r), whose commercial ToS blocks
  # non-interactive creates. Overriding to conda-forge-only sidesteps it and stays free/reproducible.
  conda create -n "${ENV_NAME}" -y --override-channels -c conda-forge python=3.12 pip
fi

# Run everything else inside the env without needing `conda activate` in a script.
RUN() { conda run --no-capture-output -n "${ENV_NAME}" "$@"; }

RUN python -m pip install -U pip

# 2. CUDA torch override (Linux GPU box) — must precede requirements.txt so the
#    cu124 wheel wins over the bare `torch==2.5.1` pin.
if [[ -n "${FASTCATAN_CUDA:-}" ]]; then
  echo "[setup] installing torch 2.5.1+${FASTCATAN_CUDA}…"
  RUN python -m pip install "torch==2.5.1" --index-url "https://download.pytorch.org/whl/${FASTCATAN_CUDA}"
fi

# 3. Pinned deps (numpy 1.26.4, SB3, gymnasium, catanatron@git, …).
RUN python -m pip install -r requirements.txt

# 4. Native extension — editable, auto-rebuilds on C++ edits. Builds as cp312.
RUN python -m pip install -e . --no-build-isolation --config-settings=editable.rebuild=true

# 5. Verify the env matches spec on THIS device — fail loud on any drift.
RUN python scripts/check_env.py

echo
echo "[setup] done. Activate with:  conda activate ${ENV_NAME}"
