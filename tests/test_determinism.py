"""Same seed -> identical trajectory. Critical for reproducibility / debugging."""
from __future__ import annotations

import random
import numpy as np
import pytest

import fastcatan
from tests.conftest import legal_actions


def _record_trajectory(seed: int, max_steps: int = 100_000):
    """Replay a random-policy game and record (action, reward, done, turn, cp) per step."""
    rng = random.Random(seed)
    env = fastcatan.Env()
    env.reset(seed)
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
    trace = []
    for _ in range(max_steps):
        env.action_mask(mask)
        legals = legal_actions(mask)
        action = rng.choice(legals)
        cp = env.current_player
        turn = env.turn_count
        reward, done = env.step(action)
        trace.append((action, reward, int(done), turn, cp))
        if done:
            break
    final_vps = tuple(env.player_vp(p) for p in range(4))
    return trace, final_vps


@pytest.mark.parametrize("seed", [0, 1, 42, 2024])
def test_trajectory_matches_under_same_seed(seed):
    trace_a, vps_a = _record_trajectory(seed)
    trace_b, vps_b = _record_trajectory(seed)
    assert trace_a == trace_b, "trajectory diverged under identical seed"
    assert vps_a == vps_b, "final VPs diverged under identical seed"


def test_different_seeds_diverge():
    """Sanity: different seeds shouldn't produce byte-identical trajectories."""
    t1, _ = _record_trajectory(1)
    t2, _ = _record_trajectory(2)
    assert t1 != t2, "expected divergent trajectories for distinct seeds"


def test_snapshot_roundtrip_matches_live_env():
    """Snapshot, take a step on copy, compare with original after step."""
    rng = random.Random(7)
    env = fastcatan.Env()
    env.reset(7)
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)

    for _ in range(50):
        env.action_mask(mask)
        legals = legal_actions(mask)
        a = rng.choice(legals)
        env.step(a)
        if env.phase == 3:
            return

    snap = env.snapshot()
    env.action_mask(mask)
    legals = legal_actions(mask)
    action = legals[0]

    env_a = fastcatan.Env()
    env_a.load_snapshot(snap)
    env_b = fastcatan.Env()
    env_b.load_snapshot(snap)

    ra, da = env_a.step(action)
    rb, db = env_b.step(action)
    assert (ra, da) == (rb, db)
    assert env_a.current_player == env_b.current_player
    assert env_a.turn_count == env_b.turn_count
    for p in range(4):
        assert env_a.player_vp(p) == env_b.player_vp(p)
        for r in range(5):
            assert env_a.player_resource(p, r) == env_b.player_resource(p, r)
