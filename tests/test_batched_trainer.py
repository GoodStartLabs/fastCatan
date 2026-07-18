"""Spec-3.2 wiring gates ported onto the new batched rollout path."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import fastcatan as fc
from models.batched_trainer.losses import DistillationBatch, DistillationLoss
from models.batched_trainer.trainer import (
    BatchedTrainer,
    SeatAssignments,
    TrainerConfig,
    compute_gae,
    terminal_rewards,
)


def _first_legal(words: np.ndarray) -> np.ndarray:
    legal = np.unpackbits(words.view(np.uint8), axis=1,
                          bitorder="little")[:, :fc.NUM_ACTIONS]
    return legal.argmax(axis=1).astype(np.uint32)


def test_terminal_reward_tracks_permuted_learner_seat_and_no_winner():
    winners = np.array([0, 1, 2, 3, 255], dtype=np.uint8)
    learner = np.array([0, 2, 2, 0, 1], dtype=np.uint8)
    assert np.array_equal(
        terminal_rewards(winners, learner),
        np.array([1, -1, 1, -1, -1], dtype=np.float32),
    )


def test_gae_propagates_terminal_reward_without_cross_episode_bleed():
    # env 0 episode A: transition 0 -> terminal transition 2.  Episode B is
    # transition 3 and must not bleed backward across transition 2's done.
    adv = compute_gae(
        env_ids=np.array([0, 1, 0, 0], dtype=np.int32),
        rewards=np.array([0, 0, 1, -1], dtype=np.float32),
        dones=np.array([0, 1, 1, 1], dtype=np.uint8),
        values=np.zeros(4, dtype=np.float32),
        next_values=np.zeros(4, dtype=np.float32),
        num_envs=2,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert np.array_equal(adv, np.array([1, 0, 1, -1], dtype=np.float32))


def test_actor_pov_batch_is_byte_aligned_with_single_env():
    n = 8
    trainer = BatchedTrainer(TrainerConfig(
        num_envs=n, device="cpu", hidden=(16,),
        rollout_decisions=64, batch_size=16,
    ))
    for _ in range(25):
        trainer.env.write_masks(trainer.masks)
        trainer.actions[:] = _first_legal(trainer.masks)
        trainer.env.step(trainer.actions, trainer.rewards, trainer.dones)
    players, _policies, _legal = trainer._encode()
    snapshots = np.empty((n, fc.SNAPSHOT_BYTES), dtype=np.uint8)
    trainer.env.save_snapshots(snapshots)
    single = fc.Env()
    expected = np.empty(fc.OBS_SIZE, dtype=np.float32)
    for i in range(n):
        single.load_snapshot(snapshots[i].tobytes())
        single.write_obs(int(players[i]), expected)
        assert np.array_equal(trainer.obs[i], expected)


def test_policy_slot_fidelity_for_every_current_seat():
    rng = np.random.default_rng(7)
    assignments = SeatAssignments(512, rng)
    for seat in range(4):
        current = np.full(512, seat, dtype=np.uint8)
        got = assignments.current_policies(current)
        assert np.array_equal(got, assignments.seat_to_policy[:, seat])
    assert np.array_equal(np.sort(assignments.seat_to_policy, axis=1),
                          np.tile(np.arange(4, dtype=np.uint8), (512, 1)))


def test_reset_reshuffles_only_finished_env_assignments():
    rng = np.random.default_rng(11)
    assignments = SeatAssignments(128, rng)
    before = assignments.seat_to_policy.copy()
    finished = np.arange(0, 128, 2)
    assignments.reshuffle(finished)
    assert np.array_equal(assignments.seat_to_policy[1::2], before[1::2])
    assert np.any(np.any(assignments.seat_to_policy[finished]
                         != before[finished], axis=1))
    seats = []
    for _ in range(64):
        assignments.reshuffle(np.arange(128))
        seats.append(assignments.learner_seats.copy())
    counts = np.bincount(np.concatenate(seats), minlength=4)
    assert np.all(np.abs(counts - counts.mean()) < counts.mean() * 0.08)


def test_actor_path_is_legal_width_not_hidden_appendix():
    trainer = BatchedTrainer(TrainerConfig(
        num_envs=4, device="cpu", hidden=(16,),
        rollout_decisions=16, batch_size=4,
    ))
    trainer._encode()
    assert trainer.obs.shape == (4, fc.OBS_SIZE)
    assert trainer.net.trunk[0].in_features == fc.OBS_SIZE
    assert fc.OBS_FULL_SIZE > trainer.net.trunk[0].in_features


def test_distillation_loss_seam_accepts_actions_and_dense_search_policy():
    trainer = BatchedTrainer(TrainerConfig(
        num_envs=4, device="cpu", hidden=(16,),
        rollout_decisions=16, batch_size=4,
    ))
    trainer._encode()
    obs = torch.from_numpy(trainer.obs.copy())
    legal_np = np.unpackbits(trainer.masks.view(np.uint8), axis=1,
                             bitorder="little")[:, :fc.NUM_ACTIONS].astype(bool)
    legal = torch.from_numpy(legal_np)
    actions = torch.from_numpy(legal_np.argmax(axis=1))
    values = torch.zeros(4)
    loss_fn = DistillationLoss()
    loss, _ = loss_fn(trainer.net, DistillationBatch(
        obs, legal, actions, values,
    ))
    assert torch.isfinite(loss)
    dense = legal.float() / legal.sum(dim=1, keepdim=True)
    loss, _ = loss_fn(trainer.net, DistillationBatch(
        obs, legal, dense, values,
    ))
    assert torch.isfinite(loss)


def test_short_ppo_update_has_finite_losses():
    trainer = BatchedTrainer(TrainerConfig(
        num_envs=32, device="cpu", hidden=(32,),
        rollout_decisions=32, batch_size=16, update_epochs=1,
    ))
    for _ in range(1000):
        trainer.collect_step()
        if trainer._completed >= 32:
            break
    else:
        pytest.fail("learner transitions did not complete")
    metrics = trainer.update_ppo()
    assert metrics
    assert all(np.isfinite(value) for value in metrics.values())
