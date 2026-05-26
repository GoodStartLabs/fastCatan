#!/usr/bin/env python3
"""
bench/bench_comprehensive.py — Thesis benchmark data collector.

Compares random-policy game distributions between fastcatan and catanatron.
Identical matchup (4 random players, same seed range) in both simulators.
Use summary.csv to cross-check win rates per seat, game lengths, VP at end.

Trading note: catanatron RandomPlayer never proposes p2p trades, so fastcatan
RandomPlayer is run with a forbid mask that blocks p2p trade actions. Bank
and port trades remain enabled in both sims.

Usage
-----
  python bench/bench_comprehensive.py [options]

  --games       N   games per simulator (default 1000, applies to both)
  --games-fast  N   override games in fastcatan
  --games-cat   N   override games in catanatron
  --seed        S   base RNG seed       (default 42)
  --skip-timing     skip per-step ns timing collection
  --out-dir     DIR output dir (default bench/results/<timestamp>/)

Outputs
-------
  games_fast_random4.csv    one row per game (fastcatan)
  games_cat_random4.csv     one row per game (catanatron)
  timing_breakdown.csv      per-component latency (ns / µs)
  summary.csv               side-by-side aggregate stats
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import NamedTuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "examples"))

import numpy as np

import fastcatan
from random_player import RandomPlayer as FastRandom
from player_base import build_p2p_trade_filter

from catanatron import Color
from catanatron.game import Game
from catanatron.models.player import RandomPlayer as CatRandom
from catanatron.models.enums import ActionType
from catanatron.state_functions import player_key

from bridge.obs_encoder import encode_obs


# ==================================================================
# Statistics
# ==================================================================

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z2n = z * z / n
    center = (p + z2n / 2) / (1 + z2n)
    spread = z * math.sqrt(p * (1 - p) / n + z2n / (4 * n)) / (1 + z2n)
    return max(0.0, center - spread), min(1.0, center + spread)


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list) -> float:
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _pct(xs: list, p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


# ==================================================================
# Game record — unified schema for both simulators
# ==================================================================

class GameRecord(NamedTuple):
    matchup:     str
    game_id:     int
    seed:        int
    winner_seat: int    # -1 = timeout
    vp0:         int
    vp1:         int
    vp2:         int
    vp3:         int
    steps:       int    # total actions taken
    turns:       int    # END_TURN count
    wall_ms:     float


_GAME_FIELDS = list(GameRecord._fields)
_MAX_FAST_STEPS = 200_000


# ==================================================================
# fastcatan runner
# ==================================================================

def _fast_game(
    game_id: int,
    seed: int,
    players: list,
    step_ns_out: list[int] | None,
    mask_ns_out: list[int] | None,
) -> GameRecord:
    env = fastcatan.Env()
    env.reset(seed)
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)

    t_game = time.perf_counter()
    steps = 0

    for _ in range(_MAX_FAST_STEPS):
        if mask_ns_out is not None:
            t0 = time.perf_counter_ns()
            env.action_mask(mask)
            mask_ns_out.append(time.perf_counter_ns() - t0)
        else:
            env.action_mask(mask)

        action = players[env.current_player].act(env, mask)

        if step_ns_out is not None:
            t0 = time.perf_counter_ns()
            _, done = env.step(action)
            step_ns_out.append(time.perf_counter_ns() - t0)
        else:
            _, done = env.step(action)

        steps += 1
        if done:
            break

    wall_ms = (time.perf_counter() - t_game) * 1e3
    vps = tuple(env.player_vp(p) for p in range(4))
    winner = next((p for p, v in enumerate(vps) if v >= 10), -1)

    return GameRecord("fast_random4", game_id, seed, winner,
                      vps[0], vps[1], vps[2], vps[3],
                      steps, env.turn_count, wall_ms)


def run_fast(n_games: int, base_seed: int, n_timing_games: int = 20
             ) -> tuple[list[GameRecord], list[int], list[int]]:
    records: list[GameRecord] = []
    step_ns: list[int] = []
    mask_ns: list[int] = []

    forbid = build_p2p_trade_filter()  # match catanatron: random never trades p2p

    for g in range(n_games):
        seed = base_seed + g
        timed = g < n_timing_games
        players = [FastRandom(seed=seed + s * 997, forbid=forbid) for s in range(4)]
        rec = _fast_game(g, seed, players,
                         step_ns if timed else None,
                         mask_ns if timed else None)
        records.append(rec)
        _progress("fast_random4", g + 1, n_games)

    return records, step_ns, mask_ns


# ==================================================================
# Catanatron runner
# ==================================================================

_CAT_COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


def _cat_game(game_id: int, seed: int) -> GameRecord:
    players = [CatRandom(c) for c in _CAT_COLORS]
    t_game = time.perf_counter()
    game = Game(players, seed=seed)
    winner_color = game.play()
    wall_ms = (time.perf_counter() - t_game) * 1e3

    state = game.state
    vps = []
    for c in _CAT_COLORS:
        k = player_key(state, c)
        vps.append(int(state.player_state.get(f"{k}_ACTUAL_VICTORY_POINTS", 0)))

    # Index by fixed color (_CAT_COLORS order), NOT state.color_to_index:
    # catanatron shuffles turn order internally, so color_to_index is a per-game
    # permutation. vps[] above is built in _CAT_COLORS order, so the winner's VP
    # must be looked up the same way (else winner VP reads a random opponent).
    # This also makes "seat s" mean a fixed color in both sims (turn order varies
    # via shuffle here / random start_player in fastcatan), so the per-seat win
    # rates are measuring the same thing.
    winner_seat = _CAT_COLORS.index(winner_color) if winner_color else -1
    steps = len(state.action_records)
    turns = sum(1 for r in state.action_records
                if r.action.action_type == ActionType.END_TURN)

    return GameRecord("cat_random4", game_id, seed, winner_seat,
                      vps[0], vps[1], vps[2], vps[3],
                      steps, turns, wall_ms)


def run_cat(n_games: int, base_seed: int) -> list[GameRecord]:
    records: list[GameRecord] = []
    for g in range(n_games):
        records.append(_cat_game(g, base_seed + g))
        _progress("cat_random4", g + 1, n_games)
    return records


# ==================================================================
# Obs-encode micro-benchmark
# ==================================================================

def bench_encode_obs(n_calls: int = 2000) -> list[int]:
    """Time encode_obs() on a completed catanatron game final state."""
    game = Game([CatRandom(c) for c in _CAT_COLORS], seed=0)
    game.play()
    samples: list[int] = []
    for _ in range(n_calls):
        t0 = time.perf_counter_ns()
        encode_obs(game, Color.RED)
        samples.append(time.perf_counter_ns() - t0)
    return samples


# ==================================================================
# Summarize
# ==================================================================

def summarize(matchup: str, records: list[GameRecord]) -> dict:
    n = len(records)
    if n == 0:
        return {"matchup": matchup, "n_games": 0}

    win_counts = [sum(1 for r in records if r.winner_seat == s) for s in range(4)]
    n_timeout  = sum(1 for r in records if r.winner_seat < 0)

    steps_list = [r.steps for r in records]
    turns_list = [r.turns for r in records]
    wall_list  = [r.wall_ms for r in records]

    winner_vps = [getattr(r, f"vp{r.winner_seat}")
                  for r in records if r.winner_seat >= 0]
    loser_vps  = [getattr(r, f"vp{s}")
                  for r in records if r.winner_seat >= 0
                  for s in range(4) if s != r.winner_seat]

    total_steps  = sum(steps_list)
    total_wall_s = sum(wall_list) / 1e3

    d: dict = {"matchup": matchup, "n_games": n, "n_timeout": n_timeout}

    for s in range(4):
        k = win_counts[s]
        lo, hi = wilson_ci(k, n)
        d[f"wins_s{s}"]        = k
        d[f"win_rate_s{s}"]    = round(k / n, 4)
        d[f"win_ci95_lo_s{s}"] = round(lo, 4)
        d[f"win_ci95_hi_s{s}"] = round(hi, 4)

    d["avg_steps"] = round(_mean(steps_list), 1)
    d["std_steps"] = round(_std(steps_list),  1)
    d["p25_steps"] = round(_pct(steps_list,  25), 0)
    d["p50_steps"] = round(_pct(steps_list,  50), 0)
    d["p75_steps"] = round(_pct(steps_list,  75), 0)
    d["p95_steps"] = round(_pct(steps_list,  95), 0)

    d["avg_turns"] = round(_mean(turns_list), 1)
    d["std_turns"] = round(_std(turns_list),  1)
    d["p50_turns"] = round(_pct(turns_list,  50), 0)
    d["p95_turns"] = round(_pct(turns_list,  95), 0)

    d["avg_wall_ms_per_game"] = round(_mean(wall_list), 3)
    d["avg_winner_vp"] = round(_mean(winner_vps), 2) if winner_vps else "nan"
    d["avg_loser_vp"]  = round(_mean(loser_vps),  2) if loser_vps  else "nan"

    if total_wall_s > 0:
        d["throughput_games_s"] = round(n / total_wall_s, 3)
        d["throughput_steps_s"] = round(total_steps / total_wall_s, 0)
    else:
        d["throughput_games_s"] = "nan"
        d["throughput_steps_s"] = "nan"

    d["total_wall_s"] = round(total_wall_s, 2)
    return d


def _timing_row(label: str, samples_ns: list[int]) -> dict:
    if not samples_ns:
        return {"component": label, "n_samples": 0,
                "mean_ns": "nan", "p50_ns": "nan",
                "p95_ns": "nan", "p99_ns": "nan", "mean_us": "nan"}
    return {
        "component": label,
        "n_samples": len(samples_ns),
        "mean_ns":   round(_mean(samples_ns), 1),
        "p50_ns":    round(_pct(samples_ns, 50), 1),
        "p95_ns":    round(_pct(samples_ns, 95), 1),
        "p99_ns":    round(_pct(samples_ns, 99), 1),
        "mean_us":   round(_mean(samples_ns) / 1e3, 3),
    }


# ==================================================================
# CSV output
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


def _write_games(path: Path, records: list[GameRecord]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_GAME_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow(r._asdict())


def _progress(tag: str, done: int, total: int) -> None:
    step = max(1, total // 10)
    if done % step == 0 or done == total:
        print(f"  [{tag}] {done}/{total} ({done/total*100:.0f}%)")


# ==================================================================
# Main
# ==================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Thesis benchmark — random4 in fastcatan vs catanatron.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--games",       type=int, default=1000,
                    help="games per simulator (default 1000)")
    ap.add_argument("--games-fast",  type=int, default=None,
                    help="override fastcatan game count")
    ap.add_argument("--games-cat",   type=int, default=None,
                    help="override catanatron game count")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--out-dir",     type=str, default="")
    args = ap.parse_args()

    games_fast = args.games_fast if args.games_fast is not None else args.games
    games_cat  = args.games_cat  if args.games_cat  is not None else args.games

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else _HERE / "results" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_dir}\n")

    timing_rows: list[dict] = []
    n_timing = 0 if args.skip_timing else 20

    # ---- fastcatan ----
    print("=== fastcatan  random4 ===")
    recs_fast, step_ns, mask_ns = run_fast(games_fast, args.seed, n_timing)
    _write_games(out_dir / "games_fast_random4.csv", recs_fast)
    sum_fast = summarize("fast_random4", recs_fast)

    if step_ns:
        timing_rows.append(_timing_row("fastcatan  env.step()", step_ns))
        timing_rows.append(_timing_row("fastcatan  env.action_mask()", mask_ns))

    # ---- obs encode ----
    if not args.skip_timing:
        print("\n=== obs encode micro-bench ===")
        obs_ns = bench_encode_obs(n_calls=2000)
        timing_rows.append(_timing_row("bridge     encode_obs()", obs_ns))

    # ---- catanatron ----
    print("\n=== catanatron random4 ===")
    recs_cat = run_cat(games_cat, args.seed)
    _write_games(out_dir / "games_cat_random4.csv", recs_cat)
    sum_cat = summarize("cat_random4", recs_cat)

    # ---- write aggregates ----
    _write_csv(out_dir / "summary.csv", [sum_fast, sum_cat])
    if timing_rows:
        _write_csv(out_dir / "timing_breakdown.csv", timing_rows)

    # ---- console summary ----
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"Results → {out_dir}")
    print(sep)
    print(f"  {'matchup':<22}  {'s0':>7}  {'s1':>7}  {'s2':>7}  {'s3':>7}  "
          f"{'steps/g':>8}  {'turns/g':>8}  {'games/s':>9}")
    print("  " + "-" * 85)
    for s in [sum_fast, sum_cat]:
        wr = [f"{float(s.get(f'win_rate_s{i}', 0)):.1%}" for i in range(4)]
        print(f"  {s['matchup']:<22}  "
              f"{'  '.join(f'{w:>5}' for w in wr)}  "
              f"{str(s.get('avg_steps', '?')):>8}  "
              f"{str(s.get('avg_turns', '?')):>8}  "
              f"{str(s.get('throughput_games_s', '?')):>9}")

    if timing_rows:
        print(f"\n  {'component':<30}  {'p50 ns':>9}  {'p95 ns':>9}  {'mean µs':>9}")
        print("  " + "-" * 62)
        for r in timing_rows:
            print(f"  {r['component']:<30}  "
                  f"{str(r['p50_ns']):>9}  "
                  f"{str(r['p95_ns']):>9}  "
                  f"{str(r['mean_us']):>9}")


if __name__ == "__main__":
    main()
