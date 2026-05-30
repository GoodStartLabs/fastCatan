"""Per-seat win rate: one model at seat 0 vs three (older) models at seats 1-3.

gate.py reports only seat-0's win rate; this prints the full per-seat
distribution by driving ALL four seats with explicit policies on each seat's
POV obs. Lets you see the newest model's win share AND the seat-0 first-move
advantage confound — even four equal policies do NOT split 25/25/25/25; seat 0
(first to place/act) wins more, so read seat 0's number against that baseline,
not against 0.25. Run --equal-baseline to measure that baseline on one model.

    python -m models.selfplay.eval_seats \
        --newest models/checkpoints/sp_smoke_5m/snap_5017600.zip \
        --seats  models/checkpoints/sp_smoke_5m/snap_4014080.zip \
                 models/checkpoints/sp_smoke_5m/snap_3010560.zip \
                 models/checkpoints/sp_smoke_5m/snap_2007040.zip \
        --games 200 --no-p2p-trade
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

import fastcatan

from models.env import ComposeCapper, MAX_TRADE_COMPOSE_PER_TURN, _unpack_mask
from models.eval import wilson_ci
from models.selfplay.opponents import Opponent, PolicyOpponent
from models.selfplay.selfplay_env import _p2p_trade_mask_bool


def play_one(
    env,
    seat_policies: list[Opponent],
    obs_buf: np.ndarray,
    mask_buf: np.ndarray,
    p2p: np.ndarray | None,
    max_steps: int,
    capper: "ComposeCapper | None" = None,
) -> int:
    """Drive one already-reset game; seat s acts with seat_policies[s] on seat s's
    POV obs. Returns the winning seat, or -1 if no one reached 10 VP by `max_steps`
    (stall). Shared by `eval_seats` (fixed seats) and `gate.play_2v2` (rotating).

    `capper` (optional, reset once per game) applies the per-seat trade-compose
    cap so p2p-trade games terminate instead of stalling in the ADD/CANCEL churn.
    Pass one when trades are enabled; None is fine when suppress_p2p removes them."""
    done = False
    steps = 0
    while not done and steps < max_steps:
        seat = env.current_player
        env.action_mask(mask_buf)
        mask = _unpack_mask(mask_buf)
        if p2p is not None:
            filtered = mask & ~p2p
            mask = filtered if filtered.any() else mask
        if capper is not None:
            mask = capper.filter(seat, mask)
        if not mask.any():
            break
        env.write_obs(seat, obs_buf)
        action = int(seat_policies[seat].act(obs_buf.copy(), mask))
        if capper is not None:
            capper.update(seat, action)
        _, done = env.step(action)
        steps += 1
    for p in range(fastcatan.NUM_PLAYERS):
        if env.player_vp(p) >= 10:
            return p
    return -1


def eval_seats(
    seat_policies: list[Opponent],
    games: int,
    seed: int = 0,
    max_steps: int = 150000,  # should-never-fire backstop; C++ MAX_TURNS is the real length cap
    suppress_p2p: bool = False,
    trade_compose_cap: int = MAX_TRADE_COMPOSE_PER_TURN,
) -> tuple[list[int], int]:
    """Play `games`; seat s acts with seat_policies[s]. Returns (wins[4], no_winner)."""
    env = fastcatan.Env()
    seed_seq = random.Random(seed)
    p2p = _p2p_trade_mask_bool() if suppress_p2p else None
    capper = ComposeCapper(trade_compose_cap)
    obs_buf = np.zeros(fastcatan.OBS_SIZE, dtype=np.float32)
    mask_buf = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)

    wins = [0, 0, 0, 0]
    no_winner = 0
    for _ in range(games):
        env.reset(seed_seq.getrandbits(64))
        capper.reset()
        winner = play_one(env, seat_policies, obs_buf, mask_buf, p2p, max_steps, capper)
        if winner < 0:
            no_winner += 1
        else:
            wins[winner] += 1
    return wins, no_winner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--newest", required=True, help="model at seat 0")
    p.add_argument("--seats", nargs=3, required=True,
                   help="three models for seats 1, 2, 3 (one each)")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-steps", type=int, default=150000,
                   help="All-seat step backstop (C++ MAX_TURNS is the real length "
                        "cap; this only guards a hypothetical frozen turn_count).")
    p.add_argument("--no-p2p-trade", action="store_true",
                   help="Match training: forbid p2p trades so games terminate.")
    p.add_argument("--trade-compose-cap", type=int,
                   default=MAX_TRADE_COMPOSE_PER_TURN,
                   help="Per-seat trade-compose actions/turn before the compose "
                        "block is masked (kills the ADD/CANCEL stall with trades "
                        "ON). Match training's value.")
    p.add_argument("--equal-baseline", action="store_true",
                   help="Put --newest at ALL four seats (ignore --seats) to "
                        "measure the pure seat-0 first-move baseline.")
    args = p.parse_args()

    newest = PolicyOpponent.load(Path(args.newest), name=Path(args.newest).stem)
    if args.equal_baseline:
        policies = [newest, newest, newest, newest]
        labels = [Path(args.newest).stem] * 4
    else:
        others = [PolicyOpponent.load(Path(s), name=Path(s).stem) for s in args.seats]
        policies = [newest, *others]
        labels = [Path(args.newest).stem] + [Path(s).stem for s in args.seats]

    wins, no_winner = eval_seats(
        policies, args.games, seed=args.seed,
        max_steps=args.max_steps, suppress_p2p=args.no_p2p_trade,
        trade_compose_cap=args.trade_compose_cap,
    )
    decided = args.games - no_winner

    print(f"\n=== per-seat win rate (N={args.games}, decided={decided}, "
          f"no-winner={no_winner}) ===")
    for s in range(4):
        rate = wins[s] / decided if decided else 0.0
        role = "newest" if s == 0 else f"older#{s}"
        extra = ""
        if s == 0 and decided:
            lo, hi = wilson_ci(wins[s], decided)
            extra = f"  95% CI [{lo:.3f}, {hi:.3f}]"
        print(f"  seat {s} ({role:7s} {labels[s]}): "
              f"wins={wins[s]:4d}  rate={rate:.3f}{extra}")
    if not args.equal_baseline:
        print("note: seat 0 also carries the first-move advantage; run "
              "--equal-baseline to see that baseline alone.")


if __name__ == "__main__":
    main()
