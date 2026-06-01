#!/usr/bin/env python3
"""bench/throughput.py — BatchedEnv hot-path throughput + per-component breakdown.

Measures the actual RL stepping path (the 5e7 steps/sec target in PLAN.md):
N envs stepped in lockstep with a vectorized random-legal policy and auto-reset.
Reports the steps/sec headline plus a per-component µs/step breakdown so the
bottleneck is named before training (PLAN.md M1 dashboard deliverable).

Components timed (per batched call, amortized to per-env-step):
  write_masks   : copy the maintained 320-bit mask for all envs
  policy        : vectorized random-legal action pick (numpy, not C++)
  step          : step_one across the batch (+ incremental mask + auto-reset)
  write_obs     : encode the 1084-float obs for all envs

Also reports a "step-only" ceiling (fixed actions, no mask/policy/obs) — the
pure C++ batched step rate, the number to compare against the C++ target.

Usage:
  python bench/throughput.py --n-envs 4096 --steps 20000 --warmup 2000
  OMP_NUM_THREADS=8 python bench/throughput.py --n-envs 8192 --steps 20000
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

import fastcatan as fc

NUM_ACTIONS = fc.NUM_ACTIONS
MASK_WORDS = fc.MASK_WORDS
OBS_SIZE = fc.OBS_SIZE


def _legal_bool(masks: np.ndarray) -> np.ndarray:
    """(N, MASK_WORDS) uint64 -> (N, NUM_ACTIONS) bool of legal actions."""
    bits = np.unpackbits(masks.view(np.uint8), axis=1, bitorder="little")
    return bits[:, :NUM_ACTIONS].astype(bool)


def _random_legal(masks: np.ndarray, rng: np.random.Generator,
                  out: np.ndarray) -> None:
    """Fill `out` (N,) uint32 with a uniformly random legal action per env."""
    legal = _legal_bool(masks)
    # argmax over random values masked to legal slots = random legal action.
    r = rng.random((masks.shape[0], NUM_ACTIONS), dtype=np.float32)
    r *= legal
    out[:] = np.argmax(r, axis=1).astype(np.uint32)


def _fmt(n: float) -> str:
    for unit, div in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def run(n_envs: int, steps: int, warmup: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    env = fc.BatchedEnv(num_envs=n_envs, seed=seed)
    env.reset()

    masks = np.zeros((n_envs, MASK_WORDS), dtype=np.uint64)
    actions = np.zeros(n_envs, dtype=np.uint32)
    rewards = np.zeros(n_envs, dtype=np.float32)
    dones = np.zeros(n_envs, dtype=np.uint8)
    obs = np.zeros((n_envs, OBS_SIZE), dtype=np.float32)

    # Warmup (also gets envs past initial placement into steady state).
    for _ in range(warmup):
        env.write_masks(masks)
        _random_legal(masks, rng, actions)
        env.step(actions, rewards, dones)

    # ---- Full loop: mask + policy + step + obs ----
    t_mask = t_pol = t_step = t_obs = 0.0
    t0 = time.perf_counter()
    for _ in range(steps):
        a = time.perf_counter(); env.write_masks(masks)
        b = time.perf_counter(); _random_legal(masks, rng, actions)
        c = time.perf_counter(); env.step(actions, rewards, dones)
        d = time.perf_counter(); env.write_obs(obs)
        e = time.perf_counter()
        t_mask += b - a; t_pol += c - b; t_step += d - c; t_obs += e - d
    wall = time.perf_counter() - t0
    env_steps = n_envs * steps

    # ---- Step-only ceiling: reuse last actions, no mask/policy/obs ----
    env.write_masks(masks); _random_legal(masks, rng, actions)
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(actions, rewards, dones)
    wall_step_only = time.perf_counter() - t0

    omp = os.environ.get("OMP_NUM_THREADS", "unset")
    print(f"\n{'='*60}")
    print(f"BatchedEnv throughput  | n_envs={n_envs} steps={steps} "
          f"OMP_NUM_THREADS={omp}")
    print(f"{'='*60}")
    print(f"  full loop (mask+policy+step+obs): {_fmt(env_steps/wall):>9} steps/s "
          f"({wall*1e9/env_steps:.1f} ns/env-step)")
    print(f"  step-only ceiling               : {_fmt(env_steps/wall_step_only):>9} steps/s "
          f"({wall_step_only*1e9/env_steps:.1f} ns/env-step)")
    print(f"\n  per-component (ns / env-step, amortized over batch):")
    tot = t_mask + t_pol + t_step + t_obs
    for label, t in (("write_masks", t_mask), ("policy (numpy)", t_pol),
                     ("step", t_step), ("write_obs", t_obs)):
        print(f"    {label:<18} {t*1e9/env_steps:>8.1f} ns  ({t/tot*100:>4.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.n_envs, args.steps, args.warmup, args.seed)


if __name__ == "__main__":
    main()
