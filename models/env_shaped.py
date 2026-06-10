"""VP-only reward shaping over FastCatanEnv.

The base env (models/env.py) gives a *sparse* terminal reward only: +1 win /
-1 loss / -2 no-winner, 0.0 every intermediate step. Over a ~200-action horizon
that is almost no gradient for strategic play. This wrapper adds a dense
per-step signal from the learner's own victory points.

Potential-based shaping (Ng, Harada & Russell 1999):
    phi(s) = coef * own_VP(s)
    F      = gamma * phi(s') - phi(s)        (added to the step reward)

Potential-based shaping is policy-invariant in the discounted-return sense, so
it speeds credit assignment WITHOUT changing which policy is optimal -- the
win/loss terminal still defines the objective. We apply F on non-terminal steps
only; the terminal step keeps the bare +1/-1/-2 so the win signal stays crisp
and un-rescaled (a common, near-invariant practical choice).

VP-only by request: phi reads ONLY the learner's victory points
(env.player_vp(LEARNER_SEAT)) -- no opponent or board features. Opponents cannot
change the learner's VP, so phi measured after the opponents move equals phi
right after the learner's own action.

Usage -- swap FastCatanEnv for VPShapedEnv in a training factory:
    env = VPShapedEnv(seed=seed, shaping_coef=0.1, gamma=0.999)
Use the SAME gamma the PPO learner uses (default 0.999) so the shaping term's
discounting matches the agent's return.

Self-play note: SelfPlayEnv also subclasses FastCatanEnv (its own step/
_step_opponents). This wrapper covers the single-env (vs-random / vs-strong)
path; applying the identical phi delta inside SelfPlayEnv.step would shape the
self-play path too.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from models.env import FastCatanEnv, LEARNER_SEAT


class VPShapedEnv(FastCatanEnv):
    """FastCatanEnv + potential-based shaping on the learner's own VP."""

    def __init__(
        self,
        seed: int = 0,
        shaping_coef: float = 0.1,
        gamma: float = 0.999,
        opponent: str = "random",
        ab_depth: int = 2,
        ab_prune: bool = False,
        suppress_p2p_trade: bool = False,
    ):
        super().__init__(
            seed=seed, opponent=opponent, ab_depth=ab_depth, ab_prune=ab_prune,
            suppress_p2p_trade=suppress_p2p_trade,
        )
        self._coef = float(shaping_coef)
        self._gamma = float(gamma)
        self._prev_phi = 0.0

    def _phi(self) -> float:
        """Potential = coef * learner's victory points."""
        return self._coef * self._env.player_vp(LEARNER_SEAT)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        # Baseline potential at the learner's first decision (post opening
        # phase); the first step's delta is measured from here.
        self._prev_phi = self._phi()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            # Terminal dominated by the +1/-1/-2 win/loss; no shaping term.
            return obs, reward, terminated, truncated, info
        phi = self._phi()
        shaped = self._gamma * phi - self._prev_phi
        self._prev_phi = phi
        info = {**info, "vp_shaping": shaped}
        return obs, reward + shaped, terminated, truncated, info


def make_shaped_env(seed: int = 0, shaping_coef: float = 0.1, gamma: float = 0.999):
    """Factory for SB3 make_vec_env / DummyVecEnv (mirrors models.env.make_env)."""

    def _thunk():
        return VPShapedEnv(seed=seed, shaping_coef=shaping_coef, gamma=gamma)

    return _thunk
