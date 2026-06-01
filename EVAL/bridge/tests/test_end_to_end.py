"""End-to-end smoke: CatanatronBridge (uniform-over-mask policy) inside
catanatron's engine vs 3 RandomPlayers. Verifies the pipeline
(encode_obs + reverse-mapped mask + decide) completes games without
raising and produces winners with sensible VP."""

from __future__ import annotations

import pytest

from catanatron import Color
from catanatron.game import Game
from catanatron.models.player import RandomPlayer

from bridge.catanatron_bridge import CatanatronBridge


COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_bridge_completes_game(seed):
    players = [
        CatanatronBridge(Color.RED, seed=seed),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE),
        RandomPlayer(Color.WHITE),
    ]
    game = Game(players, seed=seed)
    winner = game.play()
    # Game must finish (winner or stuck at turn cap).
    if winner is not None:
        from catanatron.state_functions import get_actual_victory_points
        wvp = get_actual_victory_points(game.state, winner)
        assert wvp >= 10, f"winner has {wvp} VP, expected >= 10"


def test_bridge_batch_smoke():
    """20-game batch — checks the pipeline holds up across many random
    seeds without raising. Win rate not asserted (uniform policy is weak)."""
    wins = 0
    completed = 0
    for seed in range(20):
        players = [
            CatanatronBridge(Color.RED, seed=seed),
            RandomPlayer(Color.BLUE),
            RandomPlayer(Color.ORANGE),
            RandomPlayer(Color.WHITE),
        ]
        game = Game(players, seed=seed)
        winner = game.play()
        completed += 1
        if winner == Color.RED:
            wins += 1
    assert completed == 20
    # No tight bound on wins — just confirm games ran.
