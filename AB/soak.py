"""M4 stability soak: sustained fastcatan stepping over a long horizon.

The thesis bottleneck is simulator throughput, so the simulator must also be
*stable* over the step counts modern RL consumes. This drives `FastCatanEnv`
for `--steps` environment steps across many episodes and, every step, asserts:

  * the C++ sim raised no exception,
  * the observation is all-finite (no NaN/Inf creeping in),
  * the chosen action is in the legal mask (incremental-mask integrity),
  * the legal mask is never empty.

It samples process RSS to catch leaks. Exit 0 iff stable.

    python -m AB.soak --steps 100000000 --seed 7      # full 10^8 soak (~hours)
    python -m AB.soak --steps 10000      --seed 7      # smoke (~sub-second)

Pure fastcatan — no catanatron. A `step()` is one learner action plus the
random opponents' replies (seats 1-3 stepped inside the env), so the C++
`step_one` is exercised several times per counted step. Random policy by
default for maximum state coverage; `--ckpt` soaks a trained PPO policy.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np

import fastcatan
from models.env import FastCatanEnv, LEARNER_SEAT, WIN_VP


_PAGE = os.sysconf("SC_PAGE_SIZE")


def _rss_mb() -> float:
    """Current resident set size in MiB (Linux /proc)."""
    with open("/proc/self/statm") as f:
        resident_pages = int(f.read().split()[1])
    return resident_pages * _PAGE / (1024 * 1024)


def _build_pick(ckpt: Path | None, rng: random.Random):
    """Return pick(obs, bool_mask) -> int. Random over legal, or PPO."""
    if ckpt is None:
        def pick(obs, bool_mask):
            return int(rng.choice(np.flatnonzero(bool_mask)))
        return pick

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(ckpt))

    def pick(obs, bool_mask):
        action, _ = model.predict(obs, action_masks=bool_mask, deterministic=False)
        return int(action)

    return pick


def _winner(env) -> int:
    for p in range(fastcatan.NUM_PLAYERS):
        if env._env.player_vp(p) >= WIN_VP:
            return p
    return -1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100_000_000,
                   help="total env.step() calls (default 10^8)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ckpt", type=str, default=None,
                   help="soak a PPO checkpoint instead of random-legal")
    p.add_argument("--report-every", type=int, default=1_000_000)
    p.add_argument("--max-rss-growth", type=float, default=1.5,
                   help="FAIL if final RSS > this x post-warmup RSS (leak guard)")
    args = p.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else None
    if ckpt and not ckpt.exists():
        raise FileNotFoundError(ckpt)

    rng = random.Random(args.seed)
    pick = _build_pick(ckpt, rng)
    env = FastCatanEnv(seed=args.seed)

    obs, _ = env.reset()
    if not np.isfinite(obs).all():
        raise AssertionError("non-finite obs at reset")

    steps = 0
    episodes = 0
    ep_steps = 0
    seat_wins = [0] * fastcatan.NUM_PLAYERS
    no_winner = 0
    rss_warmup = None
    rss_max = _rss_mb()
    t0 = time.perf_counter()

    try:
        while steps < args.steps:
            mask = env.action_masks()
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                raise AssertionError(
                    f"empty legal mask @ ep {episodes} step {ep_steps}")

            action = pick(obs, mask)
            if not mask[action]:
                raise AssertionError(
                    f"illegal action {action} (not in mask) @ ep {episodes} "
                    f"step {ep_steps}")

            obs, reward, done, trunc, _info = env.step(action)
            steps += 1
            ep_steps += 1

            if not np.isfinite(obs).all():
                raise AssertionError(
                    f"non-finite obs @ ep {episodes} step {ep_steps} "
                    f"(action {action})")

            if done:
                w = _winner(env)
                if w < 0:
                    no_winner += 1
                else:
                    seat_wins[w] += 1
                episodes += 1
                ep_steps = 0
                obs, _ = env.reset()
                if not np.isfinite(obs).all():
                    raise AssertionError(f"non-finite obs at reset ep {episodes}")

            if steps % args.report_every == 0:
                rss = _rss_mb()
                rss_max = max(rss_max, rss)
                if rss_warmup is None:
                    rss_warmup = rss  # baseline after first reporting window
                rate = steps / (time.perf_counter() - t0)
                print(f"[{steps:,}/{args.steps:,}] eps={episodes} "
                      f"rss={rss:.1f}MiB (max {rss_max:.1f}) "
                      f"{rate:,.0f} steps/s")

    except Exception as e:  # noqa: BLE001 — soak must report the failing context
        elapsed = time.perf_counter() - t0
        print(f"\n*** SOAK FAILED after {steps:,} steps / {episodes} episodes "
              f"in {elapsed:.1f}s ***")
        print(f"seed={args.seed}  ckpt={ckpt}")
        print(f"error: {type(e).__name__}: {e}")
        raise SystemExit(1)

    elapsed = time.perf_counter() - t0
    rss_final = _rss_mb()
    rss_max = max(rss_max, rss_final)
    base = rss_warmup if rss_warmup is not None else rss_final
    growth = rss_final / base if base else 1.0
    leak = growth > args.max_rss_growth

    print(f"\n=== M4 soak: {'random' if ckpt is None else ckpt.name} ===")
    print(f"steps:       {steps:,}")
    print(f"episodes:    {episodes}  (no-winner/stall: {no_winner})")
    print(f"seat wins:   {seat_wins}")
    print(f"throughput:  {steps / elapsed:,.0f} steps/s  ({elapsed:.1f}s total)")
    print(f"RSS:         base {base:.1f}MiB -> final {rss_final:.1f}MiB "
          f"(max {rss_max:.1f}MiB, growth {growth:.2f}x)")
    print(f"leak guard (<{args.max_rss_growth}x): {'FAIL' if leak else 'OK'}")
    print(f"STABILITY: {'FAIL (leak)' if leak else 'PASS'}")
    raise SystemExit(1 if leak else 0)


if __name__ == "__main__":
    main()
