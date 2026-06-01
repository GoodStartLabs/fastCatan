"""Eval driver: CatanatronBridge (random stub for now) vs catanatron bots.

Smoke test. Confirms:
  - Bridge instantiates as a Player subclass.
  - Catanatron's Game accepts it.
  - Game completes; we can read winner + VP from state.

Once a trained NN policy is plugged into CatanatronBridge, the same
driver gives thesis-grade numbers (win rate, avg VP) directly comparable
to Catanatron paper baselines.
"""

from __future__ import annotations

import argparse
import time

from catanatron import Color
from catanatron.game import Game
from catanatron.models.player import RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer

from bridge.catanatron_bridge import CatanatronBridge


COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


def make_opponent(name: str, color: Color):
    if name == "random":
        return RandomPlayer(color)
    if name == "alphabeta":
        return AlphaBetaPlayer(color)
    if name == "value":
        return ValueFunctionPlayer(color)
    raise ValueError(f"unknown opponent: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opponent", choices=["random", "alphabeta", "value"],
                        default="random",
                        help="bot used for the other 3 seats")
    args = parser.parse_args()

    wins = {c: 0 for c in COLORS}
    no_winner = 0

    t0 = time.perf_counter()
    for g in range(args.games):
        seed = args.seed + g
        players = [
            CatanatronBridge(Color.RED, seed=seed),
            make_opponent(args.opponent, Color.BLUE),
            make_opponent(args.opponent, Color.ORANGE),
            make_opponent(args.opponent, Color.WHITE),
        ]
        game = Game(players, seed=seed)
        winner = game.play()
        if winner is None:
            no_winner += 1
        else:
            wins[winner] += 1
    elapsed = time.perf_counter() - t0

    n = args.games
    bridge_wins = wins[Color.RED]
    print(f"opponent:        {args.opponent}")
    print(f"games:           {n}  (no-winner: {no_winner})")
    print(f"bridge (RED):    {bridge_wins} wins  ({bridge_wins / n * 100:.1f}%)")
    for c in [Color.BLUE, Color.ORANGE, Color.WHITE]:
        print(f"{args.opponent:8s} ({c.name:6s}): {wins[c]} wins  ({wins[c] / n * 100:.1f}%)")
    print(f"total time:      {elapsed:.2f}s")
    print(f"time/game:       {elapsed / n * 1000:.1f} ms")
    print(f"throughput:      {n / elapsed:.2f} games/s")


if __name__ == "__main__":
    main()
