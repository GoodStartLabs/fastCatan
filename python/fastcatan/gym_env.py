"""Gymnasium wrapper for fastCatan single-env play.

Compatible with Stable-Baselines3 / sb3-contrib MaskablePPO:
    - obs is a flat float32 vector of shape (OBS_SIZE,)
    - action_space is Discrete(NUM_ACTIONS)
    - info["action_mask"] is a bool[NUM_ACTIONS] array (packed bits unpacked)


Wraps a `BatchedEnv(num_envs=1)` underneath. Single-agent perspective: the
"learner" plays one seat (default 0); the other 3 seats are driven by
`opponent_fn(obs, mask) -> action_id`.

Usage:
    import fastcatan
    from fastcatan.gym_env import GymEnv

    env = GymEnv(
        seat=0,
        seed=42,
        opponent_fn=my_random_policy,  # callable for the 3 frozen opponents
    )
    obs, info = env.reset(seed=42)
    for _ in range(1000):
        action = my_agent(obs, info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
"""
from __future__ import annotations
from typing import Any, Callable, Optional

import numpy as np

import fastcatan as fc


def random_legal_policy(rng: np.random.Generator) -> Callable:
    """Picks uniformly random from set bits in the action mask."""
    def policy(obs: np.ndarray, mask: np.ndarray) -> int:
        bits = []
        for w in range(fc.MASK_WORDS):
            v = int(mask[w])
            while v:
                lsb = v & (-v)
                bits.append(w * 64 + lsb.bit_length() - 1)
                v ^= lsb
        if not bits:
            return 0
        return int(rng.choice(bits))
    return policy


def lowest_legal_policy(obs: np.ndarray, mask: np.ndarray) -> int:
    """Always pick the action with the lowest ID. Deterministic."""
    for w in range(fc.MASK_WORDS):
        v = int(mask[w])
        if v:
            return w * 64 + (v & -v).bit_length() - 1
    return 0


def unpack_mask(packed: np.ndarray) -> np.ndarray:
    """Unpack a uint64[MASK_WORDS] mask into bool[NUM_ACTIONS] for SB3 consumers."""
    out = np.zeros(fc.NUM_ACTIONS, dtype=bool)
    for w in range(fc.MASK_WORDS):
        v = int(packed[w])
        base = w * 64
        while v:
            lsb = v & (-v)
            idx = base + (lsb.bit_length() - 1)
            if idx < fc.NUM_ACTIONS:
                out[idx] = True
            v ^= lsb
    return out


# Lazy-import gymnasium to keep it a soft dependency.
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False


if GYM_AVAILABLE:

    class GymEnv(gym.Env):
        """Single-agent Gymnasium wrapper.

        Observation: float32 vector of shape (OBS_SIZE,).
        Action: Discrete(NUM_ACTIONS); illegal actions are no-ops in the C++
        core but should be masked by the agent. `info["action_mask"]` returns
        the legal bitmask as a uint64 array of length MASK_WORDS.

        Reward: +1 if learner wins, -1 if any opponent wins, 0 otherwise.
        Terminated: True when the game ends (Phase::ENDED).
        Truncated: True if `max_steps` is reached without termination.
        """

        metadata = {"render_modes": []}

        observation_space = spaces.Box(
            low=0.0, high=255.0, shape=(fc.OBS_SIZE,), dtype=np.float32
        )
        action_space = spaces.Discrete(fc.NUM_ACTIONS)

        def __init__(
            self,
            seat: int = 0,
            seed: int = 42,
            opponent_fn: Optional[Callable] = None,
            max_steps: int = 5000,
        ) -> None:
            super().__init__()
            self.seat = int(seat)
            self.max_steps = int(max_steps)
            self._initial_seed = int(seed)
            self._rng = np.random.default_rng(seed)
            self._opponent_fn = opponent_fn or random_legal_policy(self._rng)
            self._env = fc.BatchedEnv(num_envs=1, seed=self._initial_seed)
            self._actions = np.zeros(1, dtype=np.uint32)
            self._rewards = np.zeros(1, dtype=np.float32)
            self._dones   = np.zeros(1, dtype=np.uint8)
            self._mask    = np.zeros((1, fc.MASK_WORDS), dtype=np.uint64)
            self._obs_buf = np.zeros((1, fc.OBS_SIZE), dtype=np.float32)
            self._steps   = 0
            self._reset_seed = self._initial_seed

        # --- gym.Env interface ---
        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            if seed is not None:
                self._reset_seed = int(seed)
                self._rng = np.random.default_rng(seed)
                self._opponent_fn = random_legal_policy(self._rng) \
                    if not isinstance(self._opponent_fn, Callable) else self._opponent_fn
            self._env = fc.BatchedEnv(num_envs=1, seed=self._reset_seed)
            self._env.reset()
            self._steps = 0
            # Auto-advance until it's the learner's turn (initial placement starts
            # at start_player which may not be us).
            self._advance_to_learner()
            return self._make_obs(), self._make_info()

        def step(self, action: int):
            # Apply learner's action.
            self._actions[0] = int(action)
            self._env.step(self._actions, self._rewards, self._dones)
            self._steps += 1
            reward = float(self._rewards[0])
            terminated = bool(self._dones[0])

            # If game continues, let opponents act until learner's turn or terminal.
            if not terminated:
                terminated, opp_reward = self._advance_to_learner()
                # If an opponent's action ended the game, the learner sees -1.
                if terminated and reward == 0.0:
                    reward = opp_reward

            truncated = (not terminated) and (self._steps >= self.max_steps)
            obs = self._make_obs()
            info = self._make_info()
            return obs, reward, terminated, truncated, info

        # --- internals ---
        def _make_obs(self) -> np.ndarray:
            self._env.write_obs(self._obs_buf)
            return self._obs_buf[0].copy()

        def _make_mask(self) -> np.ndarray:
            self._env.write_masks(self._mask)
            return self._mask[0].copy()

        def _make_info(self) -> dict[str, Any]:
            packed = self._make_mask()
            return {
                "action_mask": unpack_mask(packed),     # bool[NUM_ACTIONS] for SB3
                "action_mask_packed": packed,            # uint64[MASK_WORDS] raw
                "current_player": self._env.current_player(0),
                "phase": self._env.phase(0),
                "steps": self._steps,
                "is_learner_turn": (self._env.current_player(0) == self.seat),
            }

        def _advance_to_learner(self) -> tuple[bool, float]:
            """Step opponents until learner's turn or game ends. Returns
            (terminated, reward_for_learner_due_to_opponents).
            Reward is -1 if the active opponent's action ended the game by
            them winning, else 0.
            """
            opp_reward = 0.0
            safety = self.max_steps * 4
            while safety > 0:
                safety -= 1
                if bool(self._dones[0]):
                    return True, opp_reward
                cp = self._env.current_player(0)
                if cp == self.seat:
                    return False, 0.0
                # Opponent turn — get their obs from THEIR POV would require
                # POV-flipped obs. For simplicity, expose mask only.
                self._env.write_obs(self._obs_buf)
                obs_view = self._obs_buf[0]
                mask = self._make_mask()
                a = self._opponent_fn(obs_view, mask)
                self._actions[0] = int(a)
                self._env.step(self._actions, self._rewards, self._dones)
                self._steps += 1
                if bool(self._dones[0]):
                    # Reward from learner's POV: they didn't act, so r is whatever
                    # step_one wrote (which is from `actor` POV = opponent who won).
                    # If it's positive (opponent won), learner gets -1.
                    if self._rewards[0] > 0:
                        opp_reward = -1.0
                    return True, opp_reward
            return True, opp_reward  # safety break treated as termination

else:
    # gym not installed; provide a stub that explains.
    class GymEnv:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GymEnv requires `gymnasium`. Install with `pip install gymnasium`."
            )
