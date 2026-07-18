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

# Phase-2 terminal categorical reward: win=+1, loss=-1, no-winner=-1.  The
# no-winner case is separately reported as a truncation; it is not shaped beyond
# the clipped terminal category.  Liveness remains the frozen C++ compose cap.
TIE_REWARD = -1.0

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


def _p2p_trade_mask_bool() -> np.ndarray:
    """bool[NUM_ACTIONS] marking every player-to-player (p2p) trade action.

    AND-NOT this off a legal mask to forbid p2p trading; bank/port *maritime*
    trades (TRADE_BASE) stay legal. This is the canonical thesis ``--no-trades``
    ablation — AlphaBeta has no p2p trade model, so the M4 gate has always run
    p2p-off (see the trades memories). Mirrors models/selfplay/selfplay_env.py
    so models/ stays self-contained."""
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


class FastCatanEnv(gym.Env):
    """Single-agent Catan env, learner = seat 0; opponents pluggable.

    ``opponent`` selects the policy driving seats 1-3:
      - ``"random"``    — uniform random legal action (default; fastest).
      - ``"alphabeta"`` — the native expectimax alpha-beta (a faithful
        Catanatron AlphaBetaPlayer port, ``Env.ab_decide``), so the learner
        trains directly against the M4 eval opponent instead of random play.
        ``ab_depth`` (Catanatron default 2) and ``ab_prune`` tune it; depth 1
        is ~3x faster and equals Catanatron's ValueFunctionPlayer.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int = 0,
        opponent: str = "random",
        ab_depth: int = 2,
        ab_prune: bool = False,
        suppress_p2p_trade: bool = False,
    ):
        super().__init__()
        self._env = fastcatan.Env()
        self._seed_seq = random.Random(seed)
        self._rng = random.Random(seed ^ 0xC0FFEE)
        self._obs_buf = np.zeros(OBS_SIZE, dtype=np.float32)
        self._mask_buf = np.zeros(MASK_WORDS, dtype=np.uint64)
        self._ep_steps = 0
        self._opponent = str(opponent)
        self._ab_depth = int(ab_depth)
        self._ab_prune = bool(ab_prune)
        # --no-trades ablation: AND-NOT p2p trade actions out of every seat's mask
        # (learner AND random opponents) so the p2p trade sub-phase is never
        # entered; maritime bank/port trades stay. Default off = full game.
        self._p2p_bool = _p2p_trade_mask_bool() if suppress_p2p_trade else None
        self._banned_ids = (
            frozenset(int(i) for i in np.nonzero(self._p2p_bool)[0])
            if self._p2p_bool is not None else frozenset()
        )

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
        # No winner is the clipped terminal loss category and is reported
        # separately by Phase-2 wrappers/evaluators.
        return TIE_REWARD

    def _opponent_action(self, legal: list[int]) -> int:
        """Pick the acting opponent seat's action per the configured policy."""
        if self._opponent == "alphabeta":
            a = self._env.ab_decide(
                self._env.current_player, self._ab_depth, self._ab_prune
            )
            # ab_decide recomputes the live mask, so its pick is normally in
            # `legal`. It returns 0xFFFFFFFF if it sees no legal action; and in a
            # cross-seat forced sub-phase (e.g. a discard the learner owes on an
            # opponent's 7) the search's seat-relative pick can fall outside the
            # acting seat's set. Fall back to random so the sim always advances.
            if a != 0xFFFFFFFF and a in legal:
                return a
        return self._rng.choice(legal)

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
            if self._banned_ids:
                # Opponents obey --no-trades too; never strand a seat.
                filtered = [a for a in legal if a not in self._banned_ids]
                if filtered:
                    legal = filtered
            action = self._opponent_action(legal)
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
            # Stalled game (no winner): clipped terminal loss category.
            return self._read_obs(), TIE_REWARD, True, False, {}

        return self._read_obs(), 0.0, False, False, {}

    # --- MaskablePPO hook ---

    def action_masks(self) -> np.ndarray:
        # Straight mask read; the compose cap is baked into the C++ mask (state.hpp).
        m = _unpack_mask(self._read_mask())
        if self._p2p_bool is not None:
            # Forbid p2p trades (maritime stays); never strand the learner.
            filtered = m & ~self._p2p_bool
            if filtered.any():
                return filtered
        return m


def make_env(
    seed: int = 0,
    opponent: str = "random",
    ab_depth: int = 2,
    ab_prune: bool = False,
    suppress_p2p_trade: bool = False,
):
    """Factory for SB3 make_vec_env / SubprocVecEnv."""

    def _thunk():
        return FastCatanEnv(
            seed=seed, opponent=opponent, ab_depth=ab_depth, ab_prune=ab_prune,
            suppress_p2p_trade=suppress_p2p_trade,
        )

    return _thunk
