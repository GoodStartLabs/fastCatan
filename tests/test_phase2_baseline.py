"""Phase-2 wiring tests in the Phase-0 silent-failure style."""
from __future__ import annotations

import numpy as np
import torch

import fastcatan
from models.baseline.env import (
    INITIAL_WEIGHTS,
    POOL_NAMES,
    STRONG_TIER,
    CurriculumState,
    OpponentPoolSampler,
    Phase2CatanEnv,
    intersect_offer_cap,
)
from models.env import TIE_REWARD as DEFAULT_TIE_REWARD
from models.baseline.policy import (
    ACTOR_OBS_SIZE,
    FULL_OBS_SIZE,
    POOL_ID_SIZE,
    TRAIN_OBS_SIZE,
    SplitMlpExtractor,
)


def _all_actions_mask() -> np.ndarray:
    words = np.full(fastcatan.MASK_WORDS, np.uint64(0xFFFFFFFFFFFFFFFF))
    return words


def _set(mask: np.ndarray, action: int) -> bool:
    return bool((int(mask[action >> 6]) >> (action & 63)) & 1)


def test_actor_is_invariant_to_critic_only_tail():
    torch.manual_seed(7)
    split = SplitMlpExtractor().eval()
    obs = torch.randn(5, TRAIN_OBS_SIZE)
    changed = obs.clone()
    changed[:, ACTOR_OBS_SIZE:] = torch.randn_like(changed[:, ACTOR_OBS_SIZE:]) * 100
    with torch.no_grad():
        before = split.forward_actor(obs)
        after = split.forward_actor(changed)
        critic_before = split.forward_critic(obs)
        critic_after = split.forward_critic(changed)
    assert torch.equal(before, after), "critic-only information reached the actor"
    assert not torch.equal(critic_before, critic_after), "critic tail is disconnected"


def test_terminal_reward_default_is_clipped():
    assert DEFAULT_TIE_REWARD == -1.0


def test_parameter_budget_exact():
    split = SplitMlpExtractor()
    actor = sum(p.numel() for p in split.actor.parameters())
    actor += (512 * fastcatan.NUM_ACTIONS + fastcatan.NUM_ACTIONS)
    critic = sum(p.numel() for p in split.critic.parameters()) + 512 + 1
    assert actor == 4_993_942
    assert critic == 1_710_081
    assert actor + critic == 6_704_023
    assert actor + critic <= 8_000_000


def test_offer_cap_intersects_only_compose_additions():
    action = fastcatan.action
    original = _all_actions_mask()
    capped = intersect_offer_cap(
        original, give_total=2, want_total=1, cap=2,
    )
    assert all(
        not _set(capped, aid)
        for aid in range(action.TRADE_ADD_GIVE_BASE, action.TRADE_ADD_GIVE_BASE + 5)
    )
    assert all(
        _set(capped, aid)
        for aid in range(action.TRADE_ADD_WANT_BASE, action.TRADE_ADD_WANT_BASE + 5)
    )
    assert _set(capped, action.TRADE_OPEN)
    assert _set(capped, action.TRADE_CANCEL)
    lifted = intersect_offer_cap(
        original, give_total=99, want_total=99, cap=None,
    )
    assert np.array_equal(lifted, original)


def test_pool_initial_weights_and_iid_sampling():
    assert abs(sum(INITIAL_WEIGHTS[:3]) - 0.60) < 1e-12
    strong_mass = sum(
        weight for name, weight in zip(POOL_NAMES, INITIAL_WEIGHTS)
        if name in STRONG_TIER
    )
    assert strong_mass <= 0.20 + 1e-12
    sampler = OpponentPoolSampler(123)
    samples = [sampler.sample_names() for _ in range(4000)]
    assert all(len(lineup) == 3 for lineup in samples)
    # Replacement is required: repeated identities within one lineup must occur.
    assert any(len(set(lineup)) < 3 for lineup in samples)
    flat = [name for lineup in samples for name in lineup]
    observed_strong = sum(name in STRONG_TIER for name in flat) / len(flat)
    assert 0.17 < observed_strong < 0.23


def test_single_curriculum_knob_lifts_offer_cap():
    state = CurriculumState()
    sampler = OpponentPoolSampler(5, state)
    assert sampler.stage == 0
    assert sampler.offer_cap == 2
    assert sampler.advance()
    assert sampler.stage == 1
    assert sampler.offer_cap is None
    assert not sampler.advance()


def test_randomized_seat_obs_and_pool_ids_are_aligned():
    env = Phase2CatanEnv(seed=987, curriculum_min_games=10_000)
    seats = set()
    for _ in range(24):
        obs, info = env.reset()
        seats.add(info["learner_seat"])
        assert obs.shape == (TRAIN_OBS_SIZE,)
        ref = np.zeros(ACTOR_OBS_SIZE, dtype=np.float32)
        full = np.zeros(FULL_OBS_SIZE, dtype=np.float32)
        env._env.write_obs(env.learner_seat, ref)
        env._env.write_obs_full(env.learner_seat, full)
        assert np.array_equal(obs[:ACTOR_OBS_SIZE], ref)
        assert np.array_equal(
            obs[ACTOR_OBS_SIZE:ACTOR_OBS_SIZE + FULL_OBS_SIZE], full,
        )
        onehots = obs[-POOL_ID_SIZE:].reshape(3, len(POOL_NAMES))
        assert np.array_equal(onehots.sum(axis=1), np.ones(3))
        mask = env.action_masks()
        assert mask.any()
    assert seats == {0, 1, 2, 3}


def test_env_advances_with_zero_mask_violations():
    env = Phase2CatanEnv(seed=321, curriculum_min_games=10_000)
    obs, _ = env.reset()
    rng = np.random.default_rng(321)
    for _ in range(300):
        legal = np.flatnonzero(env.action_masks())
        assert len(legal)
        obs, _reward, done, _truncated, info = env.step(int(rng.choice(legal)))
        assert obs.shape == (TRAIN_OBS_SIZE,)
        if done:
            assert info["mask_violations"] == 0
            obs, _ = env.reset()
