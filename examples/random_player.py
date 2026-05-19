"""Uniform random legal-action policy."""
import numpy as np
from player_base import Player, legal_actions

class RandomPlayer(Player):
    name = "random"

    def act(self, env, mask: np.ndarray) -> int:
        if self.forbid is not None:
            mask = mask & ~self.forbid
        legals = legal_actions(mask)
        if not legals:
            raise RuntimeError("no legal actions in mask")
        return self.rng.choice(legals)
