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

# Terminal reward for a no-winner game (stall or tie): -2, strictly worse than a
# loss (-1), so the learner prefers losing to stalling and is pushed to close games
# out. win=+1, loss=-1, no-winner=-2. This is the training-signal half of the stall
# fix; the liveness half is the C++ compose cap (MAX_TRADE_COMPOSE_PER_TURN,
# state.hpp), which guarantees turns end.
TIE_REWARD = -2.0

# Backstop episode length, counted in learner step() calls. The C++ MAX_TURNS cap
# (state.hpp) is the single length authority and always terminates first (worst
# case ~30k learner steps < 40000); this only guards a hypothetical frozen
# turn_count. No-winner here still costs TIE_REWARD. (Random learner ends <=~2100.)
MAX_EPISODE_STEPS = 40000


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
        # No winner (tie / no-winner terminal): penalize harder than a loss (-2 vs
        # -1) so the learner treats stalling as strictly worse than losing and
        # learns to close games out. See TIE_REWARD note above.
        return TIE_REWARD

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
        _, done = self._env.step(action)
        if done:
            return self._read_obs(), self._terminal_reward(), True, False, {}

        done, term_r = self._step_opponents()
        if done:
            return self._read_obs(), term_r, True, False, {}

        if self._ep_steps >= MAX_EPISODE_STEPS:
            # Stalled game (no winner): terminal, no bootstrap. -2 (TIE_REWARD),
            # strictly worse than a loss, to push the learner to close out.
            return self._read_obs(), TIE_REWARD, True, False, {}

        return self._read_obs(), 0.0, False, False, {}

    # --- MaskablePPO hook ---

    def action_masks(self) -> np.ndarray:
        # Straight mask read; the compose cap is baked into the C++ mask (state.hpp).
        return _unpack_mask(self._read_mask())


def make_env(seed: int = 0):
    """Factory for SB3 make_vec_env / SubprocVecEnv."""

    def _thunk():
        return FastCatanEnv(seed=seed)

    return _thunk
