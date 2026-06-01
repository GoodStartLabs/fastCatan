# M4 pipeline validation on the current (1084/286) interface

Date: 2026-05-27. Validates that the AB/ harness + `bridge/` work end-to-end on
the **committed source interface** (obs 1084, actions 286), independent of any
trained model. Run in an **isolated** git worktree + venv during the interface
rebuild, before the 1084 build became the live anaconda env.

## Why this was needed

The committed source is **obs 1084 / actions 286** (`include/obs.hpp`,
`include/mask.hpp`), and `bridge/obs_encoder.py` is on the matching 1084 layout.
At the time of this validation no model had been trained on 1084/286 yet, so a
real bridge-vs-AlphaBeta run could not use any checkpoint — this validation
proves the *plumbing* is correct and ready for the 1084 model. (That model,
`ppo_1084_50m`, has since been trained and run through the harness.)

## Setup (isolated)

```
git worktree add -d /home/sinan/Desktop/msc/fastcatan-ab1084 HEAD   # f622d10
python3 -m venv .venv-ab            # python 3.12.4
pip install cmake ninja nanobind scikit-build-core numpy pytest
pip install -e . --no-build-isolation     # -> fastcatan OBS 1084 ACT 286
pip install -e /home/sinan/Desktop/msc/catanatron   # 3.3.0 @ 38207ca
```

## Results

| check | result |
|---|---|
| build dims | `OBS 1084  ACT 286  MASK_WORDS 5` ✓ |
| `test_obs_identity` (encoder ↔ C++ `write_obs`, bit-for-bit) | **5 passed** ✓ |
| pipeline: uniform bridge vs `ValueFunctionPlayer`, 6 games | all decided, no crash, 0.26 s/game ✓ |
| pipeline: uniform bridge vs `AlphaBetaPlayer` (depth 2, no prune), 2 games | all decided, AB path OK, ~6.4 s/game ✓ |

Uniform bridge policy wins 0 (expected: it is random-over-legal). The point is
that full games complete and winners are assigned on the 1084/286 interface,
exercising obs-encode, action-decode, trade-compose, and robber sub-policies.

## Timing note for the real 1000-game run

AlphaBeta unpruned depth-2 ≈ **6.4 s/game** (×3 seats dominate; the bridge NN
forward pass is negligible). 1000 games ≈ **1.8 h** unpruned → use `--ab-prune`
to cut substantially. The real PPO policy does not change this (AB is the cost).

## Update (2026-05-27, later)

The isolated worktree + venv-ab were **removed** (`git worktree remove`) once the
shared **anaconda** env was rebuilt to 1084/286 — that is now the live thesis env
(`REPRODUCIBILITY.md` §5). So the steps below run directly in anaconda; no
worktree needed.

## To produce the actual thesis number

1. Train a model on the **1084/286** interface (anaconda env, 768 envs):
   `…/anaconda3/bin/python -m models.train_ppo --num-envs 768 --total-steps 50_000_000 --run-name ppo_1084_50m`
2. fastcatan is already 1084/286 in anaconda (`pip install -e .` from current
   source).
3. `PYTHONHASHSEED=0 …/anaconda3/bin/python -m AB.tournament --ckpt models/checkpoints/ppo_1084_50m/ppo_final.zip \
      --opponent alphabeta --ab-prune --games 1000 --no-trades`
4. Gate: 95% Wilson CI lower bound > 0.25.
