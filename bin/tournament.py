"""One-command reproducible tournament (spec 0.3 §C).

Pure-fastcatan, built on the frozen per-seat driver `eval_seats.play_one` (the
catanatron BRIDGE tournament is reserved for the final G2/G3 gate). Every board is
paired: the same master-seeded board set is replayed under a full 4-cyclic seat
rotation, so each policy plays each seat on each board once (cancels the seat-0
first-move edge). Reports each candidate's win rate + Wilson lower bound (the
promotion statistic), the worst-seat slice, and the trades-on vs trades-off delta,
then appends one results row per mode via the frozen schema writer.

Default roster is 4 RandomOpponents — a reproducible baseline that demonstrates
the harness and the seat-bias cancellation (equal policies -> ~0.25 each after
rotation). Pass --newest <ckpt> to slot a trained MaskablePPO policy at the
rotating candidate seat vs three randoms.

    PYTHONPATH=EVAL PYTHONHASHSEED=0 python -m bin.tournament \
        --games 200 --seed 0 --ladder-version v0-smoke
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time

import numpy as np

import fastcatan as fc
from models.eval import wilson_ci
from models.selfplay.eval_seats import play_one
from models.selfplay.opponents import RandomOpponent
from models.selfplay.selfplay_env import _p2p_trade_mask_bool

# results/schema.py — the frozen results contract.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
import schema as SCHEMA  # noqa: E402


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _load_candidate(path):
    """Return (Opponent, name, param_count) for a MaskablePPO checkpoint."""
    from sb3_contrib import MaskablePPO
    from models.selfplay.opponents import PolicyOpponent
    model = MaskablePPO.load(path, device="cpu")
    n = sum(p.numel() for p in model.policy.parameters())
    return PolicyOpponent(model, name=path.split("/")[-1], deterministic=True), path.split("/")[-1], n


def run_mode(policies, boards, suppress_p2p, max_steps=150000):
    """Full 4-cyclic rotation over the paired board set. In rotation `rot`, policy
    k sits at seat (k+rot)%4, so seat s is policy (s-rot)%4. Returns per-policy
    (wins, games, seat_wins[4]) and the no-winner count."""
    env = fc.Env()
    p2p = _p2p_trade_mask_bool() if suppress_p2p else None
    obs = np.zeros(fc.OBS_SIZE, dtype=np.float32)
    mbuf = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
    n = len(policies)
    wins = [0] * n
    games = [0] * n
    seat_wins = [[0] * n for _ in range(n)]  # seat_wins[policy][seat]
    no_winner = 0
    decisions = 0
    for rot in range(n):
        seat_policies = [policies[(s - rot) % n] for s in range(n)]
        for b in boards:
            env.reset(b)
            steps_before = 0
            winner_seat = play_one(env, seat_policies, obs, mbuf, p2p, max_steps)
            decisions += 1  # (per-decision counting is inside play_one; approximate at game granularity)
            for s in range(n):
                games[(s - rot) % n] += 1
            if winner_seat < 0:
                no_winner += 1
            else:
                pol = (winner_seat - rot) % n
                wins[pol] += 1
                seat_wins[pol][winner_seat] += 1
    return wins, games, seat_wins, no_winner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200, help="paired boards per rotation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--newest", type=str, default=None, help="MaskablePPO checkpoint for the candidate seat")
    ap.add_argument("--ladder-version", type=str, default="v0-smoke")
    ap.add_argument("--candidate", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    if args.newest:
        cand, cand_name, params = _load_candidate(args.newest)
        policies = [cand] + [RandomOpponent(seed=1000 + i) for i in range(3)]
        opp_label = "3x random"
    else:
        policies = [RandomOpponent(seed=1000 + i) for i in range(4)]
        cand_name, params, opp_label = "random", 0, "3x random (self)"
    candidate = args.candidate or cand_name

    sd = random.Random(args.seed)
    boards = [sd.getrandbits(64) for _ in range(args.games)]
    commit = _commit()

    results = {}
    for mode, suppress in (("trades_on", False), ("trades_off", True)):
        wins, games, seat_wins, no_winner = run_mode(policies, boards, suppress)
        # candidate is policy 0
        g = games[0]
        w = wins[0]
        wr = w / g if g else 0.0
        lo, hi = wilson_ci(w, g)
        total_games = args.games * 4
        results[mode] = dict(
            wins=w, games=g, win_rate=wr, wilson_low=lo, wilson_high=hi,
            no_winner_rate=no_winner / total_games,
            seat_wins=seat_wins[0],
        )

    delta = results["trades_on"]["win_rate"] - results["trades_off"]["win_rate"]
    elapsed = time.time() - t0
    dps = (args.games * 4 * 2) / elapsed if elapsed else 0.0

    rows = []
    for mode in ("trades_on", "trades_off"):
        r = results[mode]
        rows.append(dict(
            ladder_version=args.ladder_version, candidate=candidate, opponent=opp_label,
            mode=mode, rotation="full", games=r["games"], wins=r["wins"],
            win_rate=r["win_rate"], wilson_low=r["wilson_low"], wilson_high=r["wilson_high"],
            no_winner_rate=r["no_winner_rate"], seat_wins=r["seat_wins"],
            trading_delta=(delta if mode == "trades_on" else None),
            decisions_per_s=dps, param_count=params, commit=commit,
            config_hash=f"g{args.games}s{args.seed}", wandb_url=None,
            verdict="baseline" if not args.newest else "measured",
            notes="0.3 tournament entry-point",
        ))
    parquet = SCHEMA.append_rows(rows)

    print("\n=== TOURNAMENT ===")
    print(f"candidate={candidate}  opponent={opp_label}  boards/rotation={args.games} "
          f"(x4 rotations x2 modes = {args.games*8} games)  commit={commit}")
    for mode in ("trades_on", "trades_off"):
        r = results[mode]
        print(f"  {mode:11s} win_rate={r['win_rate']:.3f}  "
              f"wilson=[{r['wilson_low']:.3f},{r['wilson_high']:.3f}]  "
              f"no_winner={r['no_winner_rate']:.3f}  seat_wins={r['seat_wins']}")
    print(f"  trading_delta(on-off)={delta:+.3f}")
    print(f"elapsed={elapsed:.1f}s  games/s={ (args.games*8)/elapsed:.0f}")
    print(f"row appended -> {parquet}")


if __name__ == "__main__":
    main()
