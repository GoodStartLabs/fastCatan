"""Random-seat, persona-pool environment for Phase-2 MaskablePPO.

All opponent choices and the learner seat are sampled once per episode.  The
offer-range curriculum is a mask intersection above the frozen engine: at
stage 0, compose actions cannot raise either side of a p2p offer above two
resource units; stage 1 lifts that cap and shifts mass toward stronger bots.
"""
from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import fastcatan
from examples.player_base import Player, legal_actions
from ladder.registry import build_agent
from models.baseline.policy import (
    ACTOR_OBS_SIZE,
    FULL_OBS_SIZE,
    POOL_SIZE,
    TRAIN_OBS_SIZE,
)


POOL_NAMES = (
    "x3-plain",
    "weighted-random",
    "builder-basic",
    "builder-strong",
    "trade-happy",
    "trade-averse",
    "balanced-strong",
    "catanatron-value",
)
POOL_INDEX = {name: idx for idx, name in enumerate(POOL_NAMES)}

# Initial weak/basic mass = 0.60.  Strong-anchor mass
# {balanced-strong,catanatron-value} = 0.20.
INITIAL_WEIGHTS = (0.20, 0.20, 0.20, 0.08, 0.07, 0.05, 0.10, 0.10)
ADVANCED_WEIGHTS = (0.08, 0.08, 0.14, 0.14, 0.12, 0.10, 0.17, 0.17)
MID_TIER = frozenset({
    "builder-basic", "builder-strong", "trade-happy", "trade-averse",
})
STRONG_TIER = frozenset({"balanced-strong", "catanatron-value"})

WIN_VP = 10
TIE_REWARD = -1.0
MAX_LEARNER_STEPS = 40_000
INITIAL_OFFER_CAP = 2


@dataclass
class CurriculumState:
    """Process-local curriculum knob; a shared proxy may supply ``value``."""

    value: int = 0


import os


X3_CKPT_DEFAULT = "/home/ubuntu/goodSettler/x3_plain.zip"


def _load_checkpoint_opponent_model():
    """Load the frozen x3_plain platform once per env (CPU; opponents are cheap)."""
    from sb3_contrib import MaskablePPO

    path = os.environ.get("X3_CKPT", X3_CKPT_DEFAULT)
    return MaskablePPO.load(path, device="cpu")


class CheckpointPlayer:
    """A frozen neural checkpoint acting as a pool opponent (Player interface).

    Reads only its OWN seat's legal observation and samples a masked action, so
    it introduces trade pressure without any path into the learner's actor input.
    """

    def __init__(self, model) -> None:
        self._model = model
        self._seat = 0
        self._obs = np.zeros(ACTOR_OBS_SIZE, dtype=np.float32)

    def bind_seat(self, seat: int) -> None:
        self._seat = int(seat)

    def set_trading_mode(self, on: bool) -> None:  # its own policy decides trades
        return None

    def act(self, env, mask_words: np.ndarray) -> int:
        env.write_obs(self._seat, self._obs)
        masks = _unpack_mask(mask_words)
        action, _ = self._model.predict(
            self._obs, action_masks=masks, deterministic=False,
        )
        return int(action)


class OpponentPoolSampler:
    def __init__(self, seed: int, curriculum: CurriculumState | Any | None = None):
        self.rng = random.Random(seed ^ 0xA11CE)
        self.curriculum = curriculum or CurriculumState()

    @property
    def stage(self) -> int:
        return int(self.curriculum.value)

    @property
    def weights(self) -> tuple[float, ...]:
        return ADVANCED_WEIGHTS if self.stage else INITIAL_WEIGHTS

    @property
    def offer_cap(self) -> int | None:
        return None if self.stage else INITIAL_OFFER_CAP

    def sample_names(self, n: int = 3) -> tuple[str, ...]:
        # random.choices performs independent draws with replacement.
        return tuple(self.rng.choices(POOL_NAMES, weights=self.weights, k=n))

    def advance(self) -> bool:
        if self.stage:
            return False
        self.curriculum.value = 1
        return True


def _unpack_mask(mask_words: np.ndarray) -> np.ndarray:
    mask = np.zeros(fastcatan.NUM_ACTIONS, dtype=bool)
    for word_idx, word in enumerate(mask_words):
        bits = int(word)
        base = word_idx * 64
        while bits:
            bit = (bits & -bits).bit_length() - 1
            action = base + bit
            if action < fastcatan.NUM_ACTIONS:
                mask[action] = True
            bits &= bits - 1
    return mask


def _clear_action(mask_words: np.ndarray, action: int) -> None:
    mask_words[action >> 6] &= ~(np.uint64(1) << np.uint64(action & 63))


def intersect_offer_cap(
    mask_words: np.ndarray,
    *,
    give_total: int,
    want_total: int,
    cap: int | None,
) -> np.ndarray:
    """Intersect a legal word-mask with the p2p compose-range curriculum."""
    result = mask_words.copy()
    if cap is None:
        return result
    action = fastcatan.action
    if give_total >= cap:
        for aid in range(action.TRADE_ADD_GIVE_BASE, action.TRADE_ADD_GIVE_BASE + 5):
            _clear_action(result, aid)
    if want_total >= cap:
        for aid in range(action.TRADE_ADD_WANT_BASE, action.TRADE_ADD_WANT_BASE + 5):
            _clear_action(result, aid)
    return result


class Phase2CatanEnv(gym.Env):
    """Single learner at a freshly randomized seat versus three pool bots."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int = 0,
        curriculum: CurriculumState | Any | None = None,
        rolling_window: int = 200,
        curriculum_min_games: int = 50,
    ) -> None:
        super().__init__()
        self._env = fastcatan.Env()
        self._seed_seq = random.Random(seed)
        self._rng = random.Random(seed ^ 0xC0FFEE)
        self._sampler = OpponentPoolSampler(seed, curriculum)
        self._x3_model = (
            _load_checkpoint_opponent_model() if "x3-plain" in POOL_NAMES else None
        )
        self._rolling_mid: deque[int] = deque(maxlen=rolling_window)
        self._curriculum_min_games = int(curriculum_min_games)
        self._obs = np.zeros(ACTOR_OBS_SIZE, dtype=np.float32)
        self._full_obs = np.zeros(FULL_OBS_SIZE, dtype=np.float32)
        self._train_obs = np.zeros(TRAIN_OBS_SIZE, dtype=np.float32)
        self._mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
        self._routing_obs = np.zeros(ACTOR_OBS_SIZE, dtype=np.float32)
        self._learner_seat = 0
        self._seat_opponents: dict[int, Player] = {}
        self._seat_names: dict[int, str] = {}
        self._discarding_seat: int | None = None
        self._learner_steps = 0
        self._mask_violations = 0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(TRAIN_OBS_SIZE,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(fastcatan.NUM_ACTIONS)

    @property
    def learner_seat(self) -> int:
        return self._learner_seat

    @property
    def opponent_lineup(self) -> tuple[str, str, str]:
        return tuple(
            self._seat_names[(self._learner_seat + rel) & 3]
            for rel in range(1, 4)
        )

    def _read_obs(self) -> np.ndarray:
        # Explicitly call write_obs for the actor.  Do not source its prefix
        # from the privileged full observation even though the bytes match.
        self._env.write_obs(self._learner_seat, self._obs)
        self._env.write_obs_full(self._learner_seat, self._full_obs)
        self._train_obs[:ACTOR_OBS_SIZE] = self._obs
        tail = ACTOR_OBS_SIZE
        self._train_obs[tail:tail + FULL_OBS_SIZE] = self._full_obs
        onehots = self._train_obs[tail + FULL_OBS_SIZE:]
        onehots.fill(0.0)
        for rel, name in enumerate(self.opponent_lineup):
            onehots[rel * POOL_SIZE + POOL_INDEX[name]] = 1.0
        return self._train_obs.copy()

    def _acting_seat(self) -> int:
        """Route discard sub-phases to the public active discarder."""
        turn_owner = int(self._env.current_player)
        if int(self._env.flag) != 1:
            self._discarding_seat = None
            return turn_owner

        self._env.write_obs(turn_owner, self._routing_obs)
        remaining = [0, 0, 0, 0]
        for relative in range(4):
            absolute = (turn_owner + relative) & 3
            remaining[absolute] = int(
                round(float(self._routing_obs[relative * 16 + 14]) * 10)
            )
        if self._discarding_seat is None:
            self._discarding_seat = next(
                (seat for seat in range(4) if remaining[seat] > 0), None,
            )
        elif remaining[self._discarding_seat] == 0:
            self._discarding_seat = next(
                ((turn_owner + offset) & 3 for offset in range(1, 5)
                 if remaining[(turn_owner + offset) & 3] > 0),
                None,
            )
        if self._discarding_seat is None:
            raise RuntimeError("discard flag has no public active discarder")
        return self._discarding_seat

    def _capped_mask_words(self) -> np.ndarray:
        self._env.action_mask(self._mask)
        return intersect_offer_cap(
            self._mask,
            give_total=sum(int(self._env.trade_give(r)) for r in range(5)),
            want_total=sum(int(self._env.trade_want(r)) for r in range(5)),
            cap=self._sampler.offer_cap,
        )

    def action_masks(self) -> np.ndarray:
        if self._acting_seat() != self._learner_seat:
            raise RuntimeError("learner mask requested while an opponent acts")
        return _unpack_mask(self._capped_mask_words())

    def _terminal(self) -> tuple[float, bool]:
        winner = next(
            (seat for seat in range(4) if self._env.player_vp(seat) >= WIN_VP),
            None,
        )
        return (1.0 if winner == self._learner_seat else -1.0), winner is None

    def _episode_info(self, *, no_winner: bool, curriculum_changed: bool = False) -> dict:
        return {
            "learner_seat": self._learner_seat,
            "opponent_lineup": json.dumps(self.opponent_lineup),
            "pool_stage": self._sampler.stage,
            "offer_cap": self._sampler.offer_cap,
            "no_winner": bool(no_winner),
            "truncation": bool(no_winner),
            "mask_violations": self._mask_violations,
            "curriculum_changed": bool(curriculum_changed),
        }

    def _record_curriculum_result(self, learner_won: bool) -> bool:
        if self._sampler.stage == 0 and all(
            name in MID_TIER for name in self.opponent_lineup
        ):
            self._rolling_mid.append(int(learner_won))
        if (
            self._sampler.stage == 0
            and len(self._rolling_mid) >= self._curriculum_min_games
            and sum(self._rolling_mid) / len(self._rolling_mid) > 0.25
        ):
            changed = self._sampler.advance()
            if changed:
                print(
                    "[curriculum] stage=1 mid_rolling_win_rate="
                    f"{sum(self._rolling_mid)/len(self._rolling_mid):.4f} "
                    f"games={len(self._rolling_mid)} offer_cap=lifted",
                    flush=True,
                )
            return changed
        return False

    def _advance_until_learner(self) -> tuple[bool, float, bool]:
        while self._acting_seat() != self._learner_seat:
            seat = self._acting_seat()
            mask = self._capped_mask_words()
            legal = legal_actions(mask)
            if not legal:
                raise RuntimeError(f"empty opponent mask seat={seat}")
            action = int(self._seat_opponents[seat].act(self._env, mask.copy()))
            if action not in legal:
                self._mask_violations += 1
                raise ValueError(
                    f"{self._seat_names[seat]} returned illegal action {action}"
                )
            _, done = self._env.step(action)
            if done:
                reward, no_winner = self._terminal()
                return True, reward, no_winner
        return False, 0.0, False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._seed_seq = random.Random(seed)
            self._rng = random.Random(seed ^ 0xC0FFEE)
        super().reset(seed=seed)
        self._learner_seat = self._rng.randrange(4)
        names = self._sampler.sample_names(3)
        self._seat_opponents.clear()
        self._seat_names.clear()
        for rel, name in enumerate(names, start=1):
            seat = (self._learner_seat + rel) & 3
            if name == "x3-plain":
                opponent = CheckpointPlayer(self._x3_model)
            else:
                opponent = build_agent(name, self._seed_seq.getrandbits(64))
            bind = getattr(opponent, "bind_seat", None)
            if bind is not None:
                bind(seat)
            set_trading = getattr(opponent, "set_trading_mode", None)
            if set_trading is not None:
                set_trading(True)
            self._seat_opponents[seat] = opponent
            self._seat_names[seat] = name
        self._env.reset(self._seed_seq.getrandbits(64))
        self._discarding_seat = None
        self._learner_steps = 0
        self._mask_violations = 0
        done, _reward, _no_winner = self._advance_until_learner()
        if done:
            return self.reset()
        return self._read_obs(), {
            "learner_seat": self._learner_seat,
            "opponent_lineup": json.dumps(self.opponent_lineup),
            "pool_stage": self._sampler.stage,
            "offer_cap": self._sampler.offer_cap,
        }

    def step(self, action: int):
        action = int(action)
        legal = np.flatnonzero(self.action_masks())
        if action not in legal:
            self._mask_violations += 1
            raise ValueError(f"learner returned masked action {action}")
        self._learner_steps += 1
        _, done = self._env.step(action)
        no_winner = False
        reward = 0.0
        if done:
            reward, no_winner = self._terminal()
        else:
            done, reward, no_winner = self._advance_until_learner()
        if not done and self._learner_steps >= MAX_LEARNER_STEPS:
            done, reward, no_winner = True, TIE_REWARD, True
        if done:
            changed = self._record_curriculum_result(reward > 0.0)
            return (
                self._read_obs(), reward, True, False,
                self._episode_info(no_winner=no_winner, curriculum_changed=changed),
            )
        return self._read_obs(), 0.0, False, False, {}


def make_env(seed: int, curriculum: CurriculumState | Any | None = None):
    def _thunk():
        return Phase2CatanEnv(seed=seed, curriculum=curriculum)
    return _thunk
