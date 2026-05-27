"""Opponent policies and the frozen-snapshot pool for M3 self-play.

An *opponent* maps `(obs, legal_mask) -> action_id` for a non-learner seat. The
learner trains at seat 0; opponents fill seats 1-3. Every snapshot plays from
its OWN seat POV: `Env.write_obs(seat, buf)` emits a perspective-flipped
observation in the exact format every policy was trained on (seat 0), so a
seat-0-trained net plays any seat unchanged (root PLAN.md: "perspective flip
for self-play"). No simulator change is needed for self-play.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Opponent(Protocol):
    name: str

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """Pick an action id. `mask` is a bool[NUM_ACTIONS] legal-action mask."""
        ...


class RandomOpponent:
    """Uniform over legal actions. The M2 baseline opponent; pool fallback."""

    def __init__(self, seed: int = 0):
        self.name = "random"
        self._rng = random.Random(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        return int(legal[self._rng.randrange(len(legal))])


class PolicyOpponent:
    """A frozen MaskablePPO snapshot. No grad; samples by default.

    Argmax (deterministic=True) trips the TRADE_OPEN/CANCEL stall (root PLAN.md),
    so opponents sample like the eval harness.
    """

    def __init__(self, model, name: str, deterministic: bool = False):
        self._model = model
        self.name = name
        self._deterministic = deterministic

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        action, _ = self._model.predict(
            obs, action_masks=mask, deterministic=self._deterministic
        )
        return int(action)

    @classmethod
    def load(
        cls,
        path: str | Path,
        name: str | None = None,
        deterministic: bool = False,
        device: str = "cpu",
    ) -> "PolicyOpponent":
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(str(path), device=device)
        return cls(model, name or Path(path).stem, deterministic)


class OpponentPool:
    """Holds frozen snapshots; samples a {seat: opponent} map per episode.

    Sampling (per episode, each opponent seat independently):
      - prob `p_random`               -> RandomOpponent (anti-collapse),
      - else uniform over last `window` snapshots (recency-weighted self-play).
    With no snapshots yet, every seat gets the random opponent (M2 regime).

    The trainer mutates one pool across rounds (`add`); since each env `reset`
    re-samples, a growing pool feeds harder opponents without rebuilding envs.
    """

    def __init__(
        self,
        seats: list[int],
        seed: int = 0,
        p_random: float = 0.2,
        window: int = 5,
    ):
        self.seats = list(seats)
        self.p_random = p_random
        self.window = window
        self._snaps: list[Opponent] = []
        self.random_opponent = RandomOpponent(seed)
        self._rng = random.Random(seed ^ 0x5EA17)

    def add(self, opp: Opponent) -> None:
        self._snaps.append(opp)

    def __len__(self) -> int:
        return len(self._snaps)

    @property
    def snapshots(self) -> list[Opponent]:
        return self._snaps

    @property
    def latest(self) -> Opponent:
        return self._snaps[-1] if self._snaps else self.random_opponent

    def sample(self) -> dict[int, Opponent]:
        out: dict[int, Opponent] = {}
        for seat in self.seats:
            if not self._snaps or self._rng.random() < self.p_random:
                out[seat] = self.random_opponent
            else:
                recent = self._snaps[-self.window:]
                out[seat] = self._rng.choice(recent)
        return out
