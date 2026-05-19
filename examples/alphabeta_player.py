"""Minimal alpha-beta player. Internal baseline (NOT the thesis baseline).

Thesis comparison uses Catanatron's AlphaBetaPlayer inside Catanatron's
engine via bridge/. This file is for quick sanity-checking the RL
pipeline against a non-trivial opponent inside fastcatan.

Algorithm:
    paranoid minimax (pov vs combined opponents) + alpha-beta pruning
    on the deterministic post-step state (no chance-node forking).
"""

from __future__ import annotations

import numpy as np

import fastcatan
from player_base import Player, legal_actions


ALPHABETA_DEFAULT_DEPTH = 2

# Heuristic weights — subset of Catanatron's DEFAULT_WEIGHTS that maps
# onto fastcatan-exposed primitives. VP dominates so positions one move
# from a win always beat anything else.
DEFAULT_WEIGHTS = {
    "public_vps":      3e14,
    "longest_road":    10.0,
    "hand_resources":  1.0,
    "discard_penalty": -5.0,
    "army_size":       10.1,
}


def value_fn(env, pov: int, params=DEFAULT_WEIGHTS) -> float:
    """Paranoid eval: pov score minus summed opponents (1/3 weight each)."""
    def score(p: int) -> float:
        hand = sum(env.player_resource(p, r) for r in range(5))
        return (
            env.player_vp_public(p)      * params["public_vps"]
            + env.player_road_length(p)  * params["longest_road"]
            + hand                       * params["hand_resources"]
            + (params["discard_penalty"] if hand > 7 else 0.0)
            + env.player_knights_played(p) * params["army_size"]
        )

    own = score(pov)
    opp = sum(score(p) for p in range(4) if p != pov) / 3.0
    return own - opp


class AlphaBetaPlayer(Player):
    name = "alphabeta"

    def __init__(
        self,
        seed: int = 0,
        forbid: np.ndarray | None = None,
        depth: int = ALPHABETA_DEFAULT_DEPTH,
        params=DEFAULT_WEIGHTS,
    ):
        super().__init__(seed=seed, forbid=forbid)
        self.depth = int(depth)
        self.params = params
        self.color: int | None = None
        self._mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)

    def act(self, env, mask: np.ndarray) -> int:
        if self.color is None:
            self.color = env.current_player

        actions = self._legal(mask)
        if len(actions) == 1:
            return actions[0]

        snap = env.snapshot()
        best_action = actions[0]
        best_value = float("-inf")
        for action in actions:
            env.step(action)
            v = self._alphabeta(env, self.depth - 1, float("-inf"), float("inf"))
            env.load_snapshot(snap)
            if v > best_value:
                best_value = v
                best_action = action
        return best_action

    def _alphabeta(self, env, depth: int, alpha: float, beta: float) -> float:
        if depth == 0 or self._terminal(env):
            return value_fn(env, self.color, self.params)

        actions = self._legal_from_env(env)
        if not actions:
            return value_fn(env, self.color, self.params)

        maximizing = env.current_player == self.color
        snap = env.snapshot()

        if maximizing:
            value = float("-inf")
            for action in actions:
                env.step(action)
                v = self._alphabeta(env, depth - 1, alpha, beta)
                env.load_snapshot(snap)
                value = max(value, v)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float("inf")
            for action in actions:
                env.step(action)
                v = self._alphabeta(env, depth - 1, alpha, beta)
                env.load_snapshot(snap)
                value = min(value, v)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def _terminal(self, env) -> bool:
        for p in range(4):
            if env.player_vp(p) >= 10:
                return True
        return False

    def _legal(self, mask: np.ndarray) -> list[int]:
        if self.forbid is not None:
            mask = mask & ~self.forbid
        return legal_actions(mask)

    def _legal_from_env(self, env) -> list[int]:
        env.action_mask(self._mask)
        return self._legal(self._mask)

    def __repr__(self) -> str:
        return f"AlphaBetaPlayer(depth={self.depth})"
