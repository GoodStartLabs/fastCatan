"""M4 thesis tournament: a trained fastcatan policy vs Catanatron baselines.

The trained agent plays seat RED via `CatanatronBridge` INSIDE Catanatron's
reference engine; the other three seats are a baseline bot (default:
`AlphaBetaPlayer`, depth 2). Reports the bridge win rate with a 95% Wilson CI
and evaluates the thesis gate.

    # full thesis run (slow — AlphaBeta is depth-N minimax x3 seats):
    python -m AB.tournament --games 1000 --opponent alphabeta --ab-depth 2 --ab-prune

    # smoke:
    python -m AB.tournament --games 20 --opponent random

THESIS GATE: win rate vs AlphaBeta > 25% with 95% confidence, i.e. the Wilson
CI lower bound > 0.25 (0.25 = the 4-player chance baseline).

Requires catanatron 3.3.0 (git 38207ca...). See AB/REPRODUCIBILITY.md.

Reproducibility: pass a fixed --seed AND launch with PYTHONHASHSEED set, e.g.
    PYTHONHASHSEED=0 python -m AB.tournament ...
Catanatron's RandomPlayer and set-iteration order depend on the hash seed
(see bridge/PLAN.md); without it the game stream is not bit-reproducible.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from catanatron import Color
from catanatron.game import Game
from catanatron.models.player import RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer

from bridge.catanatron_bridge import CatanatronBridge
from models.eval import wilson_ci  # single source of truth for the CI math
from AB.policy import build_policy


COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
BRIDGE_COLOR = Color.RED
DEFAULT_CKPT = "models/checkpoints/ppo_1084_20m/ppo_final.zip"
GATE_BASELINE = 0.25  # 4-player chance baseline / thesis threshold


def make_opponent(name: str, color: Color, ab_depth: int, ab_prune: bool):
    if name == "random":
        return RandomPlayer(color)
    if name == "alphabeta":
        return AlphaBetaPlayer(color, depth=ab_depth, prunning=ab_prune)
    if name == "value":
        return ValueFunctionPlayer(color)
    raise ValueError(f"unknown opponent: {name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                   help="checkpoint for the bridge policy (default: capped PPO)")
    p.add_argument("--algo", default="ppo", choices=["ppo"],
                   help="policy family (only ppo wired today; see AB/policy.py)")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--opponent", choices=["alphabeta", "value", "random"],
                   default="alphabeta", help="bot for the other 3 seats")
    p.add_argument("--ab-depth", type=int, default=2,
                   help="AlphaBetaPlayer search depth (default 2)")
    p.add_argument("--ab-prune", action="store_true",
                   help="enable AlphaBeta action pruning (much faster; "
                        "recommended for the 1000-game run)")
    p.add_argument("--deterministic", action="store_true",
                   help="argmax policy (default: sample; see AB/policy.py)")
    p.add_argument("--no-trades", action="store_true",
                   help="disable the bridge OFFER_TRADE compose loop")
    p.add_argument("--out", type=str, default="AB/results",
                   help="directory for the results JSON")
    p.add_argument("--progress-every", type=int, default=25)
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    # Deterministic-as-possible run (see module docstring re PYTHONHASHSEED).
    random.seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    hashseed = os.environ.get("PYTHONHASHSEED")
    if hashseed is None:
        print("[warn] PYTHONHASHSEED unset — run is NOT bit-reproducible "
              "(catanatron RNG + set order depend on it).")

    policy = build_policy(args.algo, ckpt, deterministic=args.deterministic)
    enable_trades = not args.no_trades

    wins = {c: 0 for c in COLORS}
    no_winner = 0
    t0 = time.perf_counter()

    for g in range(args.games):
        seed = args.seed + g
        players = [
            CatanatronBridge(BRIDGE_COLOR, policy=policy, seed=seed,
                             enable_trades=enable_trades),
            make_opponent(args.opponent, Color.BLUE, args.ab_depth, args.ab_prune),
            make_opponent(args.opponent, Color.ORANGE, args.ab_depth, args.ab_prune),
            make_opponent(args.opponent, Color.WHITE, args.ab_depth, args.ab_prune),
        ]
        game = Game(players, seed=seed)
        winner = game.play()
        if winner is None:
            no_winner += 1
        else:
            wins[winner] += 1

        if (g + 1) % args.progress_every == 0:
            bw = wins[BRIDGE_COLOR]
            dec = (g + 1) - no_winner
            rate = bw / dec if dec else 0.0
            el = time.perf_counter() - t0
            print(f"[{g+1}/{args.games}] bridge {bw} wins "
                  f"({rate:.3f} of {dec} decided)  "
                  f"{el / (g + 1):.2f}s/game")

    elapsed = time.perf_counter() - t0
    bridge_wins = wins[BRIDGE_COLOR]
    decided = args.games - no_winner
    lo, hi = wilson_ci(bridge_wins, decided)
    rate = bridge_wins / decided if decided else 0.0
    gate_pass = lo > GATE_BASELINE

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ckpt": str(ckpt),
        "algo": args.algo,
        "deterministic": args.deterministic,
        "enable_trades": enable_trades,
        "opponent": args.opponent,
        "ab_depth": args.ab_depth if args.opponent == "alphabeta" else None,
        "ab_prune": args.ab_prune if args.opponent == "alphabeta" else None,
        "seed": args.seed,
        "pythonhashseed": hashseed,
        "games": args.games,
        "decided": decided,
        "no_winner": no_winner,
        "bridge_wins": bridge_wins,
        "win_rate": rate,
        "ci95_low": lo,
        "ci95_high": hi,
        "gate_baseline": GATE_BASELINE,
        "gate_pass": gate_pass,
        "seat_wins": {c.name: wins[c] for c in COLORS},
        "elapsed_s": elapsed,
        "s_per_game": elapsed / args.games if args.games else 0.0,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"tournament_{args.algo}_{args.opponent}_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\n=== M4 tournament: {args.algo} vs {args.opponent} ===")
    print(f"ckpt:        {ckpt}")
    print(f"games:       {decided}/{args.games} decided (no-winner: {no_winner})")
    print(f"bridge wins: {bridge_wins}")
    print(f"win rate:    {rate:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"seat wins:   {result['seat_wins']}")
    print(f"time:        {elapsed:.1f}s  ({result['s_per_game']:.2f}s/game)")
    print(f"THESIS GATE (CI-low > {GATE_BASELINE}): "
          f"{'PASS' if gate_pass else 'FAIL'}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
