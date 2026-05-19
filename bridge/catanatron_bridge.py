"""CatanatronBridge: wraps a Python policy as a catanatron Player.

Thesis eval path:
    - Train agent in fastcatan (fast C++ sim).
    - Plug agent's policy fn into this bridge.
    - Run inside catanatron's reference engine vs catanatron baselines
      (AlphaBetaPlayer, ValueFunctionPlayer, etc.).
    - Numbers comparable to Catanatron paper.

Policy signature (current — random stub):
    policy(game, playable_actions, rng) -> Action

Future NN policy will replace `random_policy` with:
    1. encode catanatron Game -> obs vector matching training input
    2. run NN forward
    3. decode NN output (action ID over flat space) -> Catanatron Action
       from `playable_actions`
"""

from __future__ import annotations

import random
from typing import Callable

from catanatron import Color
from catanatron.game import Game
from catanatron.models.enums import Action
from catanatron.models.player import Player


PolicyFn = Callable[[Game, "list[Action]", random.Random], Action]


def random_policy(game: Game, playable_actions, rng: random.Random) -> Action:
    """Uniform random over the legal action list."""
    return rng.choice(playable_actions)


class CatanatronBridge(Player):
    """Wraps a Python policy as a Catanatron Player.

    `policy` defaults to `random_policy` so the bridge plugs in
    immediately for smoke tests / pipeline checks. Swap in a NN-backed
    policy once the agent is trained.
    """

    def __init__(self, color: Color, policy: PolicyFn | None = None,
                 seed: int = 0):
        super().__init__(color)
        self._rng = random.Random(seed)
        self._policy = policy if policy is not None else random_policy

    def decide(self, game: Game, playable_actions):
        return self._policy(game, playable_actions, self._rng)
