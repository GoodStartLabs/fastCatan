"""Gymnasium env wrapping a single fastcatan.Env.

Learner controls seat 0; seats 1-3 act with a uniform-random legal policy.
Use SB3's SubprocVecEnv/DummyVecEnv to parallelize across processes/threads.

Note: PLAN.md mentions a VecEnv-direct adapter over BatchedEnv. For the M2
first cut we use the simpler single-env Gym + SB3 vectorization path — it
is the industry-standard SB3 layout and avoids per-env-skip plumbing. The
BatchedEnv-backed VecEnv is deferred to M3 if throughput becomes the bottleneck.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import fastcatan


OBS_SIZE = fastcatan.OBS_SIZE
NUM_ACTIONS = fastcatan.NUM_ACTIONS
MASK_WORDS = fastcatan.MASK_WORDS

LEARNER_SEAT = 0
WIN_VP = 10

# --- Stall control -------------------------------------------------------
# The dominant stall is a within-turn trade-compose loop: ADD_WANT is legal
# whenever trade_want[r] < 19 and CANCEL whenever the scratch is non-empty
# (rules.cpp:1491-1504), so ADD_WANT -> CANCEL -> ADD_WANT churns forever
# WITHOUT ever opening a trade or ending the turn. turn_count only advances on
# END_TURN (rules.cpp:540), so it stays frozen and a turn_count cap never fires.
#
# Primary fix: bound trade-compose actions per turn at the mask level
# (action_masks). After MAX_TRADE_COMPOSE_PER_TURN of them the compose bits are
# masked off, forcing build / bank-trade / END_TURN. Compose actions are the
# contiguous block ADD_GIVE_BASE..TRADE_OPEN. CANCEL/ACCEPT/DECLINE/CONFIRM stay
# legal (CANCEL is the always-available escape from a pending trade, so the mask
# can never be emptied; and without re-composing the loop can't restart). Bank
# trades (TRADE_BASE) are excluded — they consume resources and self-limit.
# 20 is well above any legitimate single-turn trade composition (a real offer is
# a handful of ADD_* + OPEN) but far below the thousands seen in the stall.
_A = fastcatan.action
TRADE_COMPOSE_IDS = np.arange(_A.TRADE_ADD_GIVE_BASE, _A.TRADE_OPEN + 1, dtype=np.intp)
MAX_TRADE_COMPOSE_PER_TURN = 20

# Backstop only: episode cap counted in *learner steps* (calls to step()), NOT
# turn_count (frozen during the loop above). With the trade-compose cap in place
# this should rarely fire; it guards any residual non-terminating policy. A
# random-policy learner finishes in <=~2100 steps (40-game sample, max 2088).
MAX_EPISODE_STEPS = 3000


def _unpack_mask(mask_words: np.ndarray) -> np.ndarray:
    """uint64[MASK_WORDS] -> bool[NUM_ACTIONS] action mask."""
    out = np.zeros(NUM_ACTIONS, dtype=bool)
    for w_idx, word in enumerate(mask_words):
        w = int(word)
        base = w_idx * 64
        while w:
            bit = (w & -w).bit_length() - 1
            aid = base + bit
            if aid < NUM_ACTIONS:
                out[aid] = True
            w &= w - 1
    return out


def _legal_action_ids(mask_words: np.ndarray) -> list[int]:
    out: list[int] = []
    for w_idx, word in enumerate(mask_words):
        w = int(word)
        base = w_idx * 64
        while w:
            bit = (w & -w).bit_length() - 1
            aid = base + bit
            if aid < NUM_ACTIONS:
                out.append(aid)
            w &= w - 1
    return out


class FastCatanEnv(gym.Env):
    """Single-agent Catan env, learner = seat 0, opponents = random."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0):
        super().__init__()
        self._env = fastcatan.Env()
        self._seed_seq = random.Random(seed)
        self._rng = random.Random(seed ^ 0xC0FFEE)
        self._obs_buf = np.zeros(OBS_SIZE, dtype=np.float32)
        self._mask_buf = np.zeros(MASK_WORDS, dtype=np.uint64)
        self._ep_steps = 0
        self._compose_count = 0  # trade-compose actions in the current turn

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    # --- internals ---

    def _read_obs(self) -> np.ndarray:
        self._env.write_obs(LEARNER_SEAT, self._obs_buf)
        return self._obs_buf.copy()

    def _read_mask(self) -> np.ndarray:
        self._env.action_mask(self._mask_buf)
        return self._mask_buf

    def _terminal_reward(self) -> float:
        for p in range(fastcatan.NUM_PLAYERS):
            if self._env.player_vp(p) >= WIN_VP:
                return 1.0 if p == LEARNER_SEAT else -1.0
        # No winner (tie / no-winner terminal): treat as loss.
        return -1.0

    def _step_opponents(self) -> tuple[bool, float]:
        """Advance the sim until current_player == LEARNER_SEAT or terminal.

        Returns (done, terminal_reward).
        """
        while self._env.current_player != LEARNER_SEAT:
            mask = self._read_mask()
            legal = _legal_action_ids(mask)
            if not legal:
                # No legal actions — should not happen; treat as terminal.
                return True, self._terminal_reward()
            action = self._rng.choice(legal)
            _, done = self._env.step(action)
            if done:
                return True, self._terminal_reward()
        return False, 0.0

    # --- Gymnasium API ---

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._seed_seq = random.Random(seed)
            self._rng = random.Random(seed ^ 0xC0FFEE)
        super().reset(seed=seed)

        game_seed = self._seed_seq.getrandbits(64)
        self._env.reset(game_seed)
        self._ep_steps = 0
        self._compose_count = 0

        done, term_r = self._step_opponents()
        if done:
            # Game ended before learner ever moved (very unlikely). Re-reset.
            return self.reset()

        return self._read_obs(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        self._ep_steps += 1
        # Reset the per-turn compose budget at each turn boundary (ROLL starts
        # the learner's action phase; END_TURN closes it). Count compose actions
        # so action_masks() can gate them once the budget is spent.
        if action == _A.ROLL_DICE or action == _A.END_TURN:
            self._compose_count = 0
        elif _A.TRADE_ADD_GIVE_BASE <= action <= _A.TRADE_OPEN:
            self._compose_count += 1
        _, done = self._env.step(action)
        if done:
            return self._read_obs(), self._terminal_reward(), True, False, {}

        done, term_r = self._step_opponents()
        if done:
            return self._read_obs(), term_r, True, False, {}

        if self._ep_steps >= MAX_EPISODE_STEPS:
            # Stalled game (no winner): treat as terminal loss, no bootstrap.
            return self._read_obs(), -1.0, True, False, {}

        return self._read_obs(), 0.0, False, False, {}

    # --- MaskablePPO hook ---

    def action_masks(self) -> np.ndarray:
        mask = _unpack_mask(self._read_mask())
        if self._compose_count >= MAX_TRADE_COMPOSE_PER_TURN:
            gated = mask.copy()
            gated[TRADE_COMPOSE_IDS] = False
            # Only apply if a non-trade alternative remains, so we never hand
            # MaskablePPO an all-False mask (e.g. a forced trade-resolution state).
            if gated.any():
                mask = gated
        return mask


def make_env(seed: int = 0):
    """Factory for SB3 make_vec_env / SubprocVecEnv."""

    def _thunk():
        return FastCatanEnv(seed=seed)

    return _thunk
