"""Sweep num_envs x (DummyVec, SubprocVec) for MaskablePPO rollout throughput.

Run:
    python -m models.bench_envs --steps-per-config 100000

Prints a sorted SPS table so you can pick the best num_envs/mode for the
full training run.
"""
from __future__ import annotations

import argparse
import time
from contextlib import redirect_stdout
from io import StringIO

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from models.train_ppo import _build_vec_env


def bench_one(num_envs: int, subproc: bool, steps: int, base_seed: int) -> float:
    env = _build_vec_env(num_envs, base_seed, use_subproc=subproc)
    # Match train_ppo defaults so numbers transfer.
    n_steps = 512
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        n_steps=n_steps,
        batch_size=2048,
        n_epochs=4,
        learning_rate=3e-4,
        verbose=0,
    )

    # Warmup: one rollout + one update so JIT / forks / cache settle.
    warmup_total = n_steps * num_envs
    buf = StringIO()
    with redirect_stdout(buf):
        model.learn(total_timesteps=warmup_total, progress_bar=False)

    # Round steps up to whole rollouts so we measure complete cycles.
    rollout = n_steps * num_envs
    rounded = max(rollout, ((steps + rollout - 1) // rollout) * rollout)

    t0 = time.perf_counter()
    with redirect_stdout(buf):
        model.learn(total_timesteps=rounded, progress_bar=False, reset_num_timesteps=False)
    elapsed = time.perf_counter() - t0

    env.close()
    return rounded / elapsed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps-per-config", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dummy-envs", type=int, nargs="*",
                   default=[8, 16, 32, 48])
    p.add_argument("--subproc-envs", type=int, nargs="*",
                   default=[8, 16, 24, 32, 48])
    args = p.parse_args()

    results: list[tuple[str, int, float]] = []

    print(f"Benching ~{args.steps_per_config} steps/config (rounded to whole rollouts).")
    print(f"{'mode':10s} {'num_envs':>9s} {'SPS':>12s}")
    print("-" * 35)

    for n in args.dummy_envs:
        sps = bench_one(n, subproc=False, steps=args.steps_per_config, base_seed=args.seed)
        results.append(("dummy", n, sps))
        print(f"{'dummy':10s} {n:>9d} {sps:>12,.0f}", flush=True)

    for n in args.subproc_envs:
        sps = bench_one(n, subproc=True, steps=args.steps_per_config, base_seed=args.seed)
        results.append(("subproc", n, sps))
        print(f"{'subproc':10s} {n:>9d} {sps:>12,.0f}", flush=True)

    print()
    print("Ranked:")
    for mode, n, sps in sorted(results, key=lambda r: -r[2]):
        print(f"  {mode:8s} num_envs={n:<4d} {sps:>10,.0f} SPS")

    best = max(results, key=lambda r: r[2])
    print()
    print(f"Best: {best[0]} num_envs={best[1]} -> {best[2]:,.0f} SPS")
    secs_10m = 10_000_000 / best[2]
    print(f"10M steps at best config: ~{secs_10m / 60:.1f} min")


if __name__ == "__main__":
    main()
