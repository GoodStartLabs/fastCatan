"""Alpha-beta search player (port target: Catanatron's AlphaBetaPlayer).

Stub. Planned components:
  - depth-limited minimax with alpha-beta pruning
  - heuristic eval at leaves (VP + longest road + dev cards + hand value)
  - chance node handling for dice rolls (expectimax average)

Catanatron reference:
  catanatron/players/minimax.py
"""
import numpy as np
from player_base import Player

class AlphaBetaPlayer(Player):
    name = "alphabeta"

    def __init__(self, seed: int = 0, forbid: np.ndarray | None = None,
                 depth: int = 2):
        super().__init__(seed=seed, forbid=forbid)
        self.depth = depth

    def act(self, env, mask: np.ndarray) -> int:
        raise NotImplementedError("alpha-beta port pending")

    def alphabeta(self, env, depth: int, alpha: float, beta: float) -> tuple[float, int]:
        """(best_value, best_action) — TODO port from Catanatron."""
        raise NotImplementedError

    def value_fn(self, env, pov: int) -> float:
        """Heuristic eval of state from `pov`'s perspective. TODO."""
        raise NotImplementedError
