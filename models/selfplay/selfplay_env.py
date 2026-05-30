"""Self-play env: FastCatanEnv with seats 1-3 driven by frozen snapshots.

Reuses every FastCatanEnv mechanic (seat-0 learner, terminal reward, stall cap,
the `action_masks()` MaskablePPO hook). The only override is `_step_opponents`:
instead of uniform-random, each non-learner seat acts with the opponent the pool
assigned it for this episode, evaluated on that seat's perspective-flipped obs.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import fastcatan

from models.env import (
    ComposeCapper,
    FastCatanEnv,
    LEARNER_SEAT,
    MAX_TRADE_COMPOSE_PER_TURN,
    NUM_ACTIONS,
    _unpack_mask,
)
from models.selfplay.opponents import Opponent, OpponentPool


def _p2p_trade_mask_bool() -> np.ndarray:
    """bool[NUM_ACTIONS] marking every player-to-player trade action.

    AND-NOT this off a legal mask to forbid p2p trading (bank/port trades stay).
    Mirrors examples.player_base.build_p2p_trade_filter, inlined to keep models/
    self-contained. See SelfPlayEnv(suppress_p2p_trade=...)."""
    a = fastcatan.action
    ids = (
        list(range(a.TRADE_ADD_GIVE_BASE, a.TRADE_ADD_GIVE_BASE + 5))
        + list(range(a.TRADE_ADD_WANT_BASE, a.TRADE_ADD_WANT_BASE + 5))
        + [a.TRADE_OPEN, a.TRADE_ACCEPT, a.TRADE_DECLINE]
        + list(range(a.TRADE_CONFIRM_BASE, a.TRADE_CONFIRM_BASE + 4))
        + [a.TRADE_CANCEL]
    )
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    for i in ids:
        if i < NUM_ACTIONS:
            m[i] = True
    return m


class SelfPlayEnv(FastCatanEnv):
    """FastCatanEnv whose seats 1-3 are pool snapshots.

    `suppress_p2p_trade`: AND-NOT the player-to-player trade actions out of every
    seat's mask (learner AND opponents, so the trade sub-phase is never entered).
    This kills the TRADE_OPEN/CANCEL stall (root PLAN.md) at the Python mask layer
    — without it, self-play between strong policies stalls to the step cap and the
    gate is undecidable (no winner). Apply it consistently in train AND gate, or
    not at all: a policy trained with it on faces a different game than one without.
    The clean long-term fix is the C++ mask cap on TRADE_OPEN re-opens (open M2 item)."""

    def __init__(
        self, pool: OpponentPool, seed: int = 0, suppress_p2p_trade: bool = False,
        trade_compose_cap: int = MAX_TRADE_COMPOSE_PER_TURN,
    ):
        super().__init__(seed=seed, trade_compose_cap=trade_compose_cap)
        self._pool = pool
        self._seat_opp: dict[int, Opponent] = {}
        self._p2p_bool = _p2p_trade_mask_bool() if suppress_p2p_trade else None
        # Per-seat cap for the opponent seats (the learner is capped by the base
        # FastCatanEnv). Without it, a frozen trade-happy opponent stalls the game.
        self._capper = ComposeCapper(trade_compose_cap)

    def _apply_filter(self, mask_bool: np.ndarray) -> np.ndarray:
        """Forbid p2p trade if configured; never strand a seat with no legal move."""
        if self._p2p_bool is None:
            return mask_bool
        filtered = mask_bool & ~self._p2p_bool
        return filtered if filtered.any() else mask_bool

    def action_masks(self) -> np.ndarray:
        return self._apply_filter(super().action_masks())

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Re-sample the seat->opponent map before the base reset steps opponents.
        self._seat_opp = self._pool.sample()
        self._capper.reset()
        return super().reset(seed=seed, options=options)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """As FastCatanEnv.step, but on episode end credit this episode's
        opponents to the pool: each league opponent that was at the table gets a
        game, and a win iff seat 0 won (terminal reward > 0). No-op for the window
        pool. `self._seat_opp` is still this episode's map (reset re-samples it)."""
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            self._pool.record_result(
                self._seat_opp.values(), learner_won=reward > 0.0)
        return obs, reward, terminated, truncated, info

    def _step_opponents(self) -> tuple[bool, float]:
        """Advance until current_player == LEARNER_SEAT or terminal.

        Each opponent seat acts via its assigned policy on its own POV obs.
        """
        while self._env.current_player != LEARNER_SEAT:
            seat = self._env.current_player
            self._env.action_mask(self._mask_buf)
            mask_bool = self._apply_filter(_unpack_mask(self._mask_buf))
            mask_bool = self._capper.filter(seat, mask_bool)
            if not mask_bool.any():
                # No legal actions — should not happen; treat as terminal.
                return True, self._terminal_reward()
            self._env.write_obs(seat, self._obs_buf)
            opp = self._seat_opp.get(seat, self._pool.random_opponent)
            action = int(opp.act(self._obs_buf.copy(), mask_bool))
            self._capper.update(seat, action)
            _, done = self._env.step(action)
            if done:
                return True, self._terminal_reward()
        return False, 0.0
