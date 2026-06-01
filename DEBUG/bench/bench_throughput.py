#!/usr/bin/env python3
"""
bench/bench_throughput.py — throughput + bottleneck analyzer (M1 gate).

Answers the two questions the M1 PLAN gate asks:

  1. WHERE does the time go?  Per-component ns/µs breakdown of (a) the
     single-env baseline loop (what the Python bots + Catanatron bridge use)
     and (b) the batched RL hot path.  Each section prints an explicit
     BOTTLENECK line.

  2. HOW does fastcatan compare to Catanatron on EQUAL FOOTING?  Cross-sim
     games/s and turns/s — both independent of action granularity.  steps/s
     is reported too but flagged NOT-comparable, because fastcatan emits
     extra sub-phase micro-actions (compositional trades, discards) that
     Catanatron never enumerates, so a "step" is not the same unit of work
     in the two simulators.

Measurement notes
-----------------
  * Sub-100ns components can't be timed with one perf_counter pair per call —
    the timer's own ~40ns overhead dominates.  The read-only kernels
    (action_mask / write_obs / write_masks) are looped on a fixed state
    (state never mutates) and divided by iteration count.  env.step() mutates,
    so it's isolated by replaying a recorded game's action stream.
  * step_one fuses rules + incremental mask-update + RNG in one C++ call;
    they are not separable from Python (would need rules.cpp instrumentation).
    Their combined cost is the reported `env.step` figure minus dispatch.
  * nanobind dispatch is isolated from a linear fit of write_masks() wall-time
    vs num_envs: t(N) = dispatch + N * per_env_write; the intercept is the
    fixed per-call boundary cost.

Usage
-----
  python bench/bench_throughput.py                 # full run, ~30s
  python bench/bench_throughput.py --quick         # smaller, ~8s
  python bench/bench_throughput.py --games 1000    # gate-grade cross-sim N
  python bench/bench_throughput.py --skip-catanatron
  python bench/bench_throughput.py --out-dir bench/results/throughput

Outputs (CSV + terminal)
-------------------------
  single_env_components.csv   per-component ns/µs, single-env baseline loop
  batched_scaling.csv         per-kernel ns/env + steps/s vs num_envs
  equal_footing.csv           fastcatan vs catanatron, fair + flagged metrics
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent       # DEBUG/bench
_ROOT = _HERE.parents[1]                        # repo root
sys.path.insert(0, str(_HERE))               # for bench_comprehensive
sys.path.insert(0, str(_ROOT / "EVAL"))      # for bridge.* (moved under EVAL/)
sys.path.insert(0, str(_ROOT / "examples"))  # for player_base / random_player

import numpy as np

import fastcatan
from player_base import build_p2p_trade_filter, legal_actions

MASK_WORDS = fastcatan.MASK_WORDS
OBS_SIZE = fastcatan.OBS_SIZE
PHASE_MAIN = 2
PHASE_ENDED = 3


# ==================================================================
# timing primitives
# ==================================================================

def _loop_ns(fn, iters: int) -> float:
    """Mean ns/call of an idempotent fn, timed over one tight loop.

    Warmup first to settle the JIT-free interpreter caches / branch
    predictors, then one perf_counter pair around `iters` calls so the
    timer overhead is paid once, not per call.
    """
    for _ in range(max(1, iters // 20)):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        fn()
    return (time.perf_counter_ns() - t0) / iters


def _noop_floor(iters: int) -> float:
    """ns/iter of an empty Python loop body — the measurement noise floor."""
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        pass
    return (time.perf_counter_ns() - t0) / iters


# ==================================================================
# single-env: record a game's action stream (for step isolation)
# ==================================================================

def _record_game(seed: int, forbid: np.ndarray, cap: int = 4000) -> list[int]:
    """Play one random game; return the action id stream (no timing)."""
    import random
    env = fastcatan.Env()
    env.reset(seed)
    mask = np.zeros(MASK_WORDS, dtype=np.uint64)
    rng = random.Random(seed)
    actions: list[int] = []
    for _ in range(cap):
        env.action_mask(mask)
        legals = legal_actions(mask & ~forbid)
        if not legals:
            break
        a = rng.choice(legals)
        actions.append(int(a))
        _, done = env.step(a)
        if done:
            break
    return actions


def _advance_to_midgame(env, mask, forbid, rng, steps: int = 60) -> bool:
    """Step a fresh env into a representative MAIN-phase state.

    Returns True if a non-terminal MAIN state was reached.
    """
    for _ in range(steps):
        if env.phase == PHASE_MAIN:
            return True
        env.action_mask(mask)
        legals = legal_actions(mask & ~forbid)
        if not legals:
            return False
        _, done = env.step(rng.choice(legals))
        if done:
            return False
    return env.phase == PHASE_MAIN and env.phase != PHASE_ENDED


# ==================================================================
# single-env component breakdown
# ==================================================================

def measure_single_env(seed: int, iters: int) -> tuple[list[dict], dict]:
    """Per-component ns of the single-env baseline loop. Names the bottleneck."""
    import random

    forbid = build_p2p_trade_filter()
    env = fastcatan.Env()
    env.reset(seed)
    mask = np.zeros(MASK_WORDS, dtype=np.uint64)
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    rng = random.Random(seed)

    # Reach a representative mid-game state for the idempotent probes.
    tries = 0
    while not _advance_to_midgame(env, mask, forbid, rng) and tries < 8:
        env.reset(seed + 1000 + tries)
        tries += 1
    env.action_mask(mask)                  # freeze a real mid-game mask
    legals = legal_actions(mask)

    noop = _noop_floor(iters)

    # --- idempotent kernels (state never mutates → safe to tight-loop) ---
    t_dispatch = _loop_ns(lambda: env.current_player, iters)         # bare boundary
    t_mask = _loop_ns(lambda: env.action_mask(mask), iters)          # dispatch + memcpy
    t_obs = _loop_ns(lambda: env.write_obs(0, obs), iters)           # dispatch + encode
    t_scan = _loop_ns(lambda: legal_actions(mask), iters)            # pure-Python bit-scan
    t_policy = _loop_ns(lambda: rng.choice(legals), iters)           # pure-Python choice

    # --- env.step(): mutates, so isolate via recorded-game replay ---
    stream = _record_game(seed, forbid)
    env2 = fastcatan.Env()
    env2.reset(seed)
    t0 = time.perf_counter_ns()
    for a in stream:
        env2.step(a)
    t_step = (time.perf_counter_ns() - t0) / len(stream)             # dispatch + C++ step_one

    # --- full realistic baseline loop (sanity + true steps/s) ---
    # Run it exactly as a normal bot does: raw mask, no forbid filter (forbid is
    # a Catanatron-parity artifact used only in the cross-sim section). Keeping
    # it raw lets the component sum reconcile against the measured wall.
    env3 = fastcatan.Env()
    env3.reset(seed)
    g = 0
    nsteps = 0
    cap = max(20000, 4 * len(stream))
    t0 = time.perf_counter_ns()
    while nsteps < cap:
        env3.action_mask(mask)
        legals_l = legal_actions(mask)
        if not legals_l:
            g += 1
            env3.reset(seed + g)
            continue
        _, done = env3.step(rng.choice(legals_l))
        nsteps += 1
        if done:
            g += 1
            env3.reset(seed + g)
    t_total = (time.perf_counter_ns() - t0) / nsteps                 # real per-step wall

    # The four kernels above are the explicit per-step work; the remainder of
    # the measured wall is interpreter glue (attribute lookups, branching, list
    # build in rng.choice, the occasional reset). Surfacing it as a residual
    # makes the breakdown reconcile to 100% of the measured per-step time.
    on_path_kernels = [
        ("env.step  (dispatch + C++ step_one: rules+mask+rng)", t_step),
        ("mask bit-scan  legal_actions() [Python]", t_scan),
        ("env.action_mask  (dispatch + memcpy)", t_mask),
        ("policy  rng.choice() [Python]", t_policy),
    ]
    residual = max(0.0, t_total - sum(v for _, v in on_path_kernels))

    # The baseline loop does NOT call write_obs (bots read VP directly); obs is
    # the RL-path add-on, reported separately (off-path).
    components = [
        (on_path_kernels[0][0], t_step, True),
        (on_path_kernels[1][0], t_scan, True),
        (on_path_kernels[2][0], t_mask, True),
        (on_path_kernels[3][0], t_policy, True),
        ("interpreter glue  (loop/branch/alloc, residual)", residual, True),
        ("env.write_obs  (dispatch + encode) [RL-path only]", t_obs, False),
        ("nanobind dispatch floor  (env.current_player)", t_dispatch, False),
        ("Python loop noise floor", noop, False),
    ]

    # Bottleneck named among the explicit kernels (not the catch-all residual).
    bn_name, bn_val = max(on_path_kernels, key=lambda kv: kv[1])

    rows = []
    for name, val, counts in components:
        rows.append({
            "component": name,
            "ns_per_call": round(val, 1),
            "us_per_call": round(val / 1e3, 4),
            "on_baseline_path": counts,
            "pct_of_per_step": (round(100.0 * val / t_total, 1) if counts else ""),
        })

    info = {
        "measured_per_step_ns": round(t_total, 1),
        "steps_per_sec": round(1e9 / t_total, 0),
        "bottleneck": bn_name,
        "bottleneck_ns": round(bn_val, 1),
        "bottleneck_pct": round(100.0 * bn_val / t_total, 1),
        "iters": iters,
        "replay_game_len": len(stream),
    }
    return rows, info


# ==================================================================
# batched hot-path scaling
# ==================================================================

def _ctz64(x: np.ndarray) -> np.ndarray:
    """Trailing-zero count of nonzero uint64 lanes (vectorized)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # unsigned wraparound is intended
        low = x & (np.uint64(0) - x)             # isolate lowest set bit (two's-comp)
    return np.rint(np.log2(low.astype(np.float64))).astype(np.uint32)


def _pick_legal(masks: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Lowest legal action id per env. Correctness over speed; timed separately."""
    n = masks.shape[0]
    out.fill(0)
    remaining = np.ones(n, dtype=bool)
    for w in range(masks.shape[1]):
        word = masks[:, w]
        active = remaining & (word != 0)
        if not active.any():
            continue
        idx = np.where(active)[0]
        out[idx] = np.uint32(w) * np.uint32(64) + _ctz64(word[active])
        remaining[idx] = False
    return out


def measure_batched(n_list: list[int], passes: int, probe_iters: int
                    ) -> tuple[list[dict], dict]:
    """Per-kernel ns/env + steps/s across batch sizes. Isolates dispatch."""
    rows = []
    masks_call_by_n: dict[int, float] = {}

    for n in n_list:
        env = fastcatan.BatchedEnv(num_envs=n, seed=42)
        env.reset()
        actions = np.zeros(n, dtype=np.uint32)
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.zeros(n, dtype=np.uint8)
        masks = np.zeros((n, MASK_WORDS), dtype=np.uint64)
        obs = np.zeros((n, OBS_SIZE), dtype=np.float32)

        # Warm the batch into mid-game so masks/obs reflect real states.
        for _ in range(40):
            env.write_masks(masks)
            _pick_legal(masks, actions)
            env.step(actions, rewards, dones)

        # Read-only kernels: idempotent → tight loop, divide out timer overhead.
        env.write_masks(masks)
        t_masks_call = _loop_ns(lambda: env.write_masks(masks), probe_iters)
        t_obs_call = _loop_ns(lambda: env.write_obs(obs), probe_iters)
        masks_call_by_n[n] = t_masks_call

        # Realistic step loop: masks -> pick -> step. Isolate step by subtraction.
        env.write_masks(masks)
        t0 = time.perf_counter_ns()
        for _ in range(passes):
            _pick_legal(masks, actions)
            env.step(actions, rewards, dones)
            env.write_masks(masks)
        loop_call = (time.perf_counter_ns() - t0) / passes
        t_pick_call = _loop_ns(lambda: _pick_legal(masks, actions), max(50, probe_iters // 4))
        t_step_call = max(0.0, loop_call - t_pick_call - t_masks_call)

        per_env_step = t_step_call / n
        per_env_masks = t_masks_call / n
        per_env_obs = t_obs_call / n
        rows.append({
            "num_envs": n,
            "step_us_per_call": round(t_step_call / 1e3, 3),
            "step_ns_per_env": round(per_env_step, 2),
            "write_masks_us_per_call": round(t_masks_call / 1e3, 3),
            "write_masks_ns_per_env": round(per_env_masks, 2),
            "write_obs_us_per_call": round(t_obs_call / 1e3, 3),
            "write_obs_ns_per_env": round(per_env_obs, 2),
            "pick_us_per_call": round(t_pick_call / 1e3, 3),
            "step_steps_per_sec": round(n / (t_step_call / 1e9), 0) if t_step_call > 0 else "inf",
            "hotpath_steps_per_sec": round(n / (loop_call / 1e9), 0),
        })

    # Isolate nanobind dispatch: linear fit of write_masks call-time vs N.
    ns = np.array(sorted(masks_call_by_n), dtype=np.float64)
    ts = np.array([masks_call_by_n[int(k)] for k in ns], dtype=np.float64)
    slope, intercept = np.polyfit(ns, ts, 1)        # t = slope*N + intercept

    # Batched bottleneck = heaviest C++ kernel at the largest batch.
    big = rows[-1]
    kern = {
        "env.write_obs (C++ obs encode, %d floats/env)" % OBS_SIZE: big["write_obs_ns_per_env"],
        "env.step (C++ step_one)": big["step_ns_per_env"],
        "env.write_masks (C++ + memcpy)": big["write_masks_ns_per_env"],
    }
    bn_name = max(kern, key=kern.get)

    info = {
        "dispatch_floor_us": round(intercept / 1e3, 4),
        "per_env_write_ns_slope": round(slope, 3),
        "largest_n": big["num_envs"],
        "batched_bottleneck": bn_name,
        "batched_bottleneck_ns_per_env": round(kern[bn_name], 2),
        "passes": passes,
    }
    return rows, info


# ==================================================================
# equal-footing cross-sim comparison
# ==================================================================

def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def measure_equal_footing(games_fast: int, games_cat: int, seed: int
                          ) -> tuple[list[dict], dict]:
    """fastcatan vs catanatron, 4 random players. Fair (games/s, turns/s) +
    flagged (steps/s) metrics from one shared schema."""
    from bench_comprehensive import run_fast, run_cat   # reuse the runners

    print("  [fastcatan] running %d random4 games ..." % games_fast)
    recs_fast, _, _ = run_fast(games_fast, seed, n_timing_games=0)
    print("  [catanatron] running %d random4 games ..." % games_cat)
    recs_cat = run_cat(games_cat, seed)

    rows = []
    for label, recs in [("fastcatan", recs_fast), ("catanatron", recs_cat)]:
        wall_s = [r.wall_ms / 1e3 for r in recs]
        steps = [r.steps for r in recs]
        turns = [r.turns for r in recs]
        total_wall = sum(wall_s)
        n = len(recs)
        rows.append({
            "simulator": label,
            "n_games": n,
            "ms_per_game": round(_mean(wall_s) * 1e3, 3),
            "games_per_sec": round(n / total_wall, 2) if total_wall else "inf",
            "turns_per_game": round(_mean(turns), 1),
            "turns_per_sec": round(sum(turns) / total_wall, 0) if total_wall else "inf",
            # NOT cross-comparable: different action granularity. Shown for context.
            "steps_per_game_NONCOMP": round(_mean(steps), 1),
            "steps_per_sec_NONCOMP": round(sum(steps) / total_wall, 0) if total_wall else "inf",
        })

    fast, cat = rows[0], rows[1]
    info = {
        "games_per_sec_speedup": (round(fast["games_per_sec"] / cat["games_per_sec"], 1)
                                  if cat["games_per_sec"] else "inf"),
        "turns_per_sec_speedup": (round(fast["turns_per_sec"] / cat["turns_per_sec"], 1)
                                  if cat["turns_per_sec"] else "inf"),
        "ms_per_game_ratio": (round(cat["ms_per_game"] / fast["ms_per_game"], 1)
                              if fast["ms_per_game"] else "inf"),
    }
    return rows, info


# ==================================================================
# CSV + terminal
# ==================================================================

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _rule(width: int = 74) -> str:
    return "-" * width


def main() -> None:
    ap = argparse.ArgumentParser(
        description="fastcatan throughput + bottleneck analyzer (M1 gate).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--games", type=int, default=200,
                    help="cross-sim games per simulator (default 200; gate uses 1000)")
    ap.add_argument("--games-cat", type=int, default=None,
                    help="override catanatron game count (it's the slow one)")
    ap.add_argument("--iters", type=int, default=100_000,
                    help="tight-loop iters for single-env component probes")
    ap.add_argument("--passes", type=int, default=2000,
                    help="batched step passes per N")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="smaller everything for a fast smoke run")
    ap.add_argument("--skip-catanatron", action="store_true")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.quick:
        args.iters = 20_000
        args.passes = 500
        args.games = min(args.games, 50)

    n_list = [1, 64, 256, 1024, 4096] if not args.quick else [1, 64, 1024]
    games_cat = args.games_cat if args.games_cat is not None else args.games

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else _HERE / "results" / ("throughput_" + ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("fastcatan throughput + bottleneck analyzer")
    print("=" * 74)
    print("Output -> %s\n" % out_dir)

    # ---- 1. single-env component breakdown ----
    print("[1/3] single-env baseline loop — per-component breakdown")
    se_rows, se_info = measure_single_env(args.seed, args.iters)
    _write_csv(out_dir / "single_env_components.csv", se_rows)

    print("  %-52s %10s %8s" % ("component", "ns/call", "% step"))
    print("  " + _rule(72))
    for r in se_rows:
        pct = ("%5s" % r["pct_of_per_step"]) if r["pct_of_per_step"] != "" else "   --"
        tag = "" if r["on_baseline_path"] else "  (off-path)"
        print("  %-52s %10s %7s%s" % (r["component"][:52], r["ns_per_call"], pct, tag))
    print("  " + _rule(72))
    print("  measured per-step: %.1f ns  (on-path rows sum to 100%%)"
          % se_info["measured_per_step_ns"])
    print("  single-env throughput: %s steps/s" % f"{int(se_info['steps_per_sec']):,}")
    print("  >> BOTTLENECK (single-env baseline): %s" % se_info["bottleneck"])
    print("     %.1f ns/call = %.1f%% of the per-step budget\n"
          % (se_info["bottleneck_ns"], se_info["bottleneck_pct"]))

    # ---- 2. batched scaling ----
    print("[2/3] batched hot-path — kernel scaling + dispatch isolation")
    b_rows, b_info = measure_batched(n_list, args.passes,
                                     probe_iters=2000 if not args.quick else 400)
    _write_csv(out_dir / "batched_scaling.csv", b_rows)

    print("  %7s %12s %12s %12s %16s" %
          ("N", "step ns/env", "masks ns/env", "obs ns/env", "step steps/s"))
    print("  " + _rule(72))
    for r in b_rows:
        sps = r["step_steps_per_sec"]
        sps_s = f"{int(sps):,}" if sps != "inf" else "inf"
        print("  %7d %12s %12s %12s %16s" %
              (r["num_envs"], r["step_ns_per_env"], r["write_masks_ns_per_env"],
               r["write_obs_ns_per_env"], sps_s))
    print("  " + _rule(72))
    print("  nanobind dispatch floor (write_masks fit intercept): %.3f µs/call"
          % b_info["dispatch_floor_us"])
    print("  >> BOTTLENECK (batched hot path, N=%d): %s"
          % (b_info["largest_n"], b_info["batched_bottleneck"]))
    print("     %.2f ns/env\n" % b_info["batched_bottleneck_ns_per_env"])

    # ---- 3. equal-footing cross-sim ----
    ef_info = {}
    if not args.skip_catanatron:
        print("[3/3] equal footing — fastcatan vs catanatron (4 random players)")
        ef_rows, ef_info = measure_equal_footing(args.games, games_cat, args.seed)
        _write_csv(out_dir / "equal_footing.csv", ef_rows)

        print("  %-11s %9s %11s %11s %12s %14s" %
              ("simulator", "ms/game", "games/s", "turns/s", "turns/game", "steps/s(*)"))
        print("  " + _rule(72))
        for r in ef_rows:
            print("  %-11s %9s %11s %11s %12s %14s" %
                  (r["simulator"], r["ms_per_game"], r["games_per_sec"],
                   r["turns_per_sec"], r["turns_per_game"], r["steps_per_sec_NONCOMP"]))
        print("  " + _rule(72))
        print("  (*) steps/s is NOT cross-comparable — fastcatan emits extra")
        print("      sub-phase micro-actions Catanatron never enumerates.")
        print("  >> EQUAL-FOOTING SPEEDUP (fastcatan / catanatron):")
        print("     %sx games/s   |   %sx turns/s   |   %sx faster per game"
              % (ef_info["games_per_sec_speedup"], ef_info["turns_per_sec_speedup"],
                 ef_info["ms_per_game_ratio"]))
    else:
        print("[3/3] equal footing — SKIPPED (--skip-catanatron)")

    print("\n" + "=" * 74)
    print("CSVs written to %s" % out_dir)
    print("=" * 74)


if __name__ == "__main__":
    main()
