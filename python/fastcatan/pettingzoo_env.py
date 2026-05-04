"""PettingZoo Agent-Environment-Cycle (AEC) wrapper for fastCatan.

Each of the 4 players is a separate agent. The active agent (whoever's turn
it is in the underlying env) is exposed via `agent_selection`. Per-agent
observations are POV-flipped — agent `player_k`'s seat-0 slot in the obs
is always its own private state.

Used by the trading-net training setup (PLAN.md M3): different policies per
sub-decision class (build vs trade) require AEC's fine-grained per-step
control.

Usage:
    from fastcatan.pettingzoo_env import CatanAECEnv

    env = CatanAECEnv(seed=42)
    env.reset()
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        action = my_policy[agent](obs, info["action_mask"])
        env.step(action)
"""
from __future__ import annotations
from typing import Any, Optional
import numpy as np

import fastcatan as fc

try:
    from pettingzoo import AECEnv
    from pettingzoo.utils.agent_selector import agent_selector
    from gymnasium import spaces
    PZ_AVAILABLE = True
except ImportError:
    PZ_AVAILABLE = False


if PZ_AVAILABLE:

    class CatanAECEnv(AECEnv):
        metadata = {"name": "fastcatan_aec_v0", "is_parallelizable": False}

        def __init__(self, seed: int = 42, max_steps: int = 5000) -> None:
            super().__init__()
            self._seed = int(seed)
            self._max_steps = int(max_steps)
            self.possible_agents = [f"player_{i}" for i in range(fc.NUM_PLAYERS)]
            self.observation_spaces = {
                a: spaces.Dict({
                    "observation": spaces.Box(
                        low=0.0, high=255.0, shape=(fc.OBS_SIZE,), dtype=np.float32
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=2**64 - 1, shape=(fc.MASK_WORDS,), dtype=np.uint64
                    ),
                })
                for a in self.possible_agents
            }
            self.action_spaces = {
                a: spaces.Discrete(fc.NUM_ACTIONS) for a in self.possible_agents
            }
            self._reset_buffers()

        def _reset_buffers(self):
            self._env = fc.BatchedEnv(num_envs=1, seed=self._seed)
            self._actions = np.zeros(1, dtype=np.uint32)
            self._rewards = np.zeros(1, dtype=np.float32)
            self._dones   = np.zeros(1, dtype=np.uint8)
            self._mask    = np.zeros((1, fc.MASK_WORDS), dtype=np.uint64)
            self._obs_buf = np.zeros(fc.OBS_SIZE, dtype=np.float32)
            self._steps   = 0

        # --- AECEnv interface ---
        def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
            if seed is not None:
                self._seed = int(seed)
            self._reset_buffers()
            self._env.reset()
            self.agents = list(self.possible_agents)
            self.rewards = {a: 0.0 for a in self.agents}
            self._cumulative_rewards = {a: 0.0 for a in self.agents}
            self.terminations = {a: False for a in self.agents}
            self.truncations  = {a: False for a in self.agents}
            self.infos = {a: {} for a in self.agents}
            self._update_agent_selection()
            self._refresh_info_for_current()

        def observe(self, agent: str) -> dict:
            seat = self._seat_of(agent)
            self._env.write_obs_pov(0, seat, self._obs_buf)
            self._env.write_masks(self._mask)
            return {
                "observation": self._obs_buf.copy(),
                "action_mask": self._mask[0].copy(),
            }

        def step(self, action: int):
            agent = self.agent_selection
            if self.terminations[agent] or self.truncations[agent]:
                # Required to call _was_dead_step in AEC API.
                self._was_dead_step(action)
                return

            self._actions[0] = int(action)
            self._env.step(self._actions, self._rewards, self._dones)
            self._steps += 1

            # The acting agent gets the reward written by step_one (its POV).
            r = float(self._rewards[0])
            self.rewards = {a: 0.0 for a in self.agents}
            self.rewards[agent] = r
            for a in self.agents:
                self._cumulative_rewards[a] += self.rewards[a]

            if bool(self._dones[0]):
                # last_winner was captured BEFORE the auto-reset wiped state.
                winner_seat = self._env.last_winner(0)
                if winner_seat < fc.NUM_PLAYERS:
                    winner = self.possible_agents[winner_seat]
                    for a in self.agents:
                        self.rewards[a] = 1.0 if a == winner else -1.0
                    self._cumulative_rewards = {
                        a: self._cumulative_rewards[a] + self.rewards[a]
                        for a in self.agents
                    }
                for a in self.agents:
                    self.terminations[a] = True
            elif self._steps >= self._max_steps:
                for a in self.agents:
                    self.truncations[a] = True

            self._update_agent_selection()
            self._refresh_info_for_current()

        def last(self, observe: bool = True):
            # PettingZoo convention: last() returns CUMULATIVE reward since
            # last reset for the active agent. Per-step reward is in self.rewards.
            agent = self.agent_selection
            obs = self.observe(agent) if observe else None
            return (
                obs,
                self._cumulative_rewards.get(agent, 0.0),
                self.terminations.get(agent, False),
                self.truncations.get(agent, False),
                self.infos.get(agent, {}),
            )

        # --- helpers ---
        def _seat_of(self, agent: str) -> int:
            return int(agent.split("_")[1])

        def _update_agent_selection(self):
            cp = self._env.current_player(0)
            self.agent_selection = self.possible_agents[cp]

        def _refresh_info_for_current(self):
            self._env.write_masks(self._mask)
            agent = self.agent_selection
            self.infos[agent] = {
                "action_mask": self._mask[0].copy(),
                "phase": self._env.phase(0),
                "current_player": self._env.current_player(0),
                "steps": self._steps,
            }

        def render(self):  # no-op
            pass

        def close(self):
            pass


else:
    class CatanAECEnv:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "CatanAECEnv requires `pettingzoo` and `gymnasium`. "
                "Install with `pip install pettingzoo gymnasium`."
            )
