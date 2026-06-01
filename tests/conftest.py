"""Shared helpers for simulator tests."""
from __future__ import annotations

import random
import numpy as np

import fastcatan


MAX_STEPS = 100_000


def legal_actions(mask: np.ndarray) -> list[int]:
    """Return action IDs whose bit is set in the uint64 bitmask buffer."""
    out: list[int] = []
    for word_idx, word in enumerate(mask):
        w = int(word)
        base = word_idx * 64
        while w:
            bit = (w & -w).bit_length() - 1
            out.append(base + bit)
            w &= w - 1
    return out


def play_random_game(seed: int, max_steps: int = MAX_STEPS, on_step=None):
    """Play one full random-policy game. Optional `on_step(env, mask, action, step_idx)` hook."""
    rng = random.Random(seed)
    env = fastcatan.Env()
    env.reset(seed)
    mask_buf = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)

    for step_idx in range(max_steps):
        env.action_mask(mask_buf)
        legals = legal_actions(mask_buf)
        assert legals, f"no legal actions at step {step_idx}, phase={env.phase} flag={env.flag}"
        action = rng.choice(legals)
        if on_step is not None:
            on_step(env, mask_buf, action, step_idx)
        _, done = env.step(action)
        if done:
            return env, step_idx + 1
    raise AssertionError(f"game did not terminate in {max_steps} steps (seed={seed})")
