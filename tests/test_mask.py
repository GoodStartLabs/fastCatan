"""Legal-action mask semantics.

  - mask is never all-zero in a live game
  - legal action advances state (changes something observable)
  - sampling only from the mask reaches terminal state across seeds
  - illegal action is a no-op (state hash unchanged) — uses snapshot equality
"""
from __future__ import annotations

import random
import numpy as np
import pytest

import fastcatan
from tests.conftest import legal_actions, play_random_game


def test_mask_nonempty_throughout_game():
    def hook(env, mask, action, step_idx):
        assert int(mask.sum()) != 0, f"empty mask at step {step_idx}"
    play_random_game(seed=13, on_step=hook)


def test_mask_bits_within_action_range():
    env = fastcatan.Env()
    env.reset(0)
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
    env.action_mask(mask)
    for aid in legal_actions(mask):
        assert 0 <= aid < fastcatan.NUM_ACTIONS, f"out-of-range action ID in mask: {aid}"


def test_illegal_action_is_noop():
    """Pick an action NOT in the mask; snapshot must be byte-identical after step."""
    rng = random.Random(99)
    env = fastcatan.Env()
    env.reset(99)
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)

    # Advance ~30 steps so we're past initial placement / interesting state
    for _ in range(30):
        env.action_mask(mask)
        legals = legal_actions(mask)
        env.step(rng.choice(legals))
        if env.phase == 3:
            pytest.skip("game ended too early; rerun with later state")

    env.action_mask(mask)
    legals = set(legal_actions(mask))
    illegals = [a for a in range(fastcatan.NUM_ACTIONS) if a not in legals]
    assert illegals, "every action legal — cannot test illegal no-op"

    snap_before = env.snapshot()
    reward, done = env.step(illegals[0])
    snap_after = env.snapshot()
    assert snap_before == snap_after, "illegal action mutated state (expected no-op)"
    assert reward == 0.0
    assert int(done) == 0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_policy_only_uses_mask(seed):
    """If we *only* sample inside the mask, we should never see no-op state stalls."""
    last_snap = None
    stalls = 0

    def hook(env, mask, action, step_idx):
        nonlocal last_snap, stalls
        cur = env.snapshot()
        if last_snap is not None and cur == last_snap:
            stalls += 1
        last_snap = cur

    play_random_game(seed, on_step=hook)
    # Some stalls technically possible (e.g. successive TRADE_DECLINE producing
    # identical snapshots is impossible since trade scratch updates). Be strict.
    assert stalls == 0, f"{stalls} no-op steps under mask-only sampling"
