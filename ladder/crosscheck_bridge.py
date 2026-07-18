"""One-time native-port versus real-catanatron bridge-band cross-check."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from models.eval import wilson_ci

from .config import MASTER_SEED
from .match import run_matchup
from .registry import build_agent


BOTS = {
    "weighted-random": "catanatron-weighted-random",
    "value": "catanatron-value",
    "alphabeta-d1": "catanatron-alphabeta-d1",
}


def intervals_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def _native(bot: str, games: int, master_seed: int) -> dict:
    from examples.player_base import Player, legal_actions

    class UniformNoTrade(Player):
        name = "catanatron-random"

        def bind_seat(self, seat: int) -> None:
            self.seat = seat

        def act(self, env, mask):
            return self.rng.choice(legal_actions(mask))

    registered = BOTS[bot]
    blocks = run_matchup(
        candidate_name=registered,
        opponent_name="catanatron-random",
        candidate_factory=lambda seed: build_agent(registered, seed),
        opponent_factory=lambda seed: UniformNoTrade(seed),
        games=games,
        master_seed=master_seed,
        suppress_p2p=True,
    )
    wins = sum(block.candidate_wins for block in blocks)
    no_winner = sum(block.no_winner for block in blocks)
    interval = wilson_ci(wins, games)
    return {
        "wins": wins,
        "games": games,
        "no_winner": no_winner,
        "rate": wins / games,
        "ci95": list(interval),
        "decisions": sum(block.decisions for block in blocks),
        "wall_seconds": sum(block.wall_seconds for block in blocks),
    }


def _real_catanatron(bot: str, games: int, seed: int) -> dict:
    from catanatron import Color
    from catanatron.game import Game
    from catanatron.models.player import RandomPlayer
    from catanatron.players.minimax import AlphaBetaPlayer
    from catanatron.players.value import ValueFunctionPlayer
    from catanatron.players.weighted_random import WeightedRandomPlayer

    colors = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
    wins = 0
    no_winner = 0
    for game_index in range(games):
        game_seed = seed + game_index
        random.seed(game_seed)
        np.random.seed(game_seed & 0xFFFFFFFF)
        if bot == "weighted-random":
            candidate = WeightedRandomPlayer(Color.RED)
        elif bot == "value":
            candidate = ValueFunctionPlayer(Color.RED)
        elif bot == "alphabeta-d1":
            candidate = AlphaBetaPlayer(Color.RED, depth=1, prunning=False)
        else:  # pragma: no cover - argparse/constant guard
            raise ValueError(bot)
        players = [candidate] + [RandomPlayer(color) for color in colors[1:]]
        game = Game(players, seed=game_seed)
        winner = game.play()
        if winner is None:
            no_winner += 1
        elif winner == Color.RED:
            wins += 1
    interval = wilson_ci(wins, games)
    return {
        "wins": wins,
        "games": games,
        "no_winner": no_winner,
        "rate": wins / games,
        "ci95": list(interval),
    }


def run_crosscheck(bots: list[str], *, games: int, seed: int) -> dict:
    if games < 200:
        raise ValueError("bridge-band cross-check requires at least 200 games")
    if games % 4:
        raise ValueError("games must be divisible by four for native rotation blocks")
    results = {}
    for bot in bots:
        native = _native(bot, games, MASTER_SEED ^ seed)
        real = _real_catanatron(bot, games, seed)
        overlap = intervals_overlap(tuple(native["ci95"]), tuple(real["ci95"]))
        results[bot] = {"native": native, "real_catanatron": real, "ci_overlap": overlap}
        print(
            f"{bot}: native={native['rate']:.3f} {native['ci95']} "
            f"real={real['rate']:.3f} {real['ci95']} overlap={overlap}",
            flush=True,
        )
    return {
        "games_per_bot": games,
        "seed": seed,
        "protocol": "one bot vs three uniform random; no p2p trades",
        "results": results,
        "all_ci_overlap": all(result["ci_overlap"] for result in results.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=24_601)
    parser.add_argument("--bots", nargs="+", choices=sorted(BOTS), default=sorted(BOTS))
    parser.add_argument("--out", type=Path, default=Path("results/bridge_crosscheck_v1.json"))
    args = parser.parse_args()
    result = run_crosscheck(args.bots, games=args.games, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["all_ci_overlap"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
