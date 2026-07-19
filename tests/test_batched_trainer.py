"""Spec-3.2 wiring gates ported onto the new batched rollout path."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import fastcatan as fc
from models.batched_trainer.losses import DistillationBatch, DistillationLoss
from models.batched_trainer.multiprocess_env import ProcessBatchedEnv
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


def test_multiprocess_shards_are_per_seat_obs_byte_exact():
    n = 9
    pool = ProcessBatchedEnv(n, seed=1234, worker_count=3)
    try:
        snapshots = np.empty((n, fc.SNAPSHOT_BYTES), dtype=np.uint8)
        pool.save_snapshots(snapshots)
        got = np.empty((n, fc.OBS_SIZE), dtype=np.float32)
        expected = np.empty(fc.OBS_SIZE, dtype=np.float32)
        single = fc.Env()
        for pov in range(fc.NUM_PLAYERS):
            povs = np.full(n, pov, dtype=np.uint8)
            pool.write_obs_pov_batch(povs, got)
            for row in range(n):
                single.load_snapshot(snapshots[row].tobytes())
                single.write_obs(pov, expected)
                assert np.array_equal(got[row], expected), (row, pov)
    finally:
        pool.close()


def test_multiprocess_actions_and_successors_stay_in_their_shards():
    n = 10
    pool = ProcessBatchedEnv(n, seed=99, worker_count=4)
    try:
        before = np.empty((n, fc.SNAPSHOT_BYTES), dtype=np.uint8)
        after = np.empty_like(before)
        masks = np.empty((n, fc.MASK_WORDS), dtype=np.uint64)
        actions = np.empty(n, dtype=np.uint32)
        rewards = np.empty(n, dtype=np.float32)
        dones = np.empty(n, dtype=np.uint8)
        pool.save_snapshots(before)
        pool.write_masks(masks)
        actions[:] = _first_legal(masks)
        pool.step(actions, rewards, dones)
        pool.save_snapshots(after)

        single = fc.Env()
        for row in range(n):
            single.load_snapshot(before[row].tobytes())
            reward, done = single.step(int(actions[row]))
            assert not done
            assert reward == rewards[row]
            assert single.snapshot() == after[row].tobytes(), row
    finally:
        pool.close()


def test_multiprocess_trainer_collects_global_rollout_rows():
    trainer = BatchedTrainer(TrainerConfig(
        num_envs=16, env_workers=3, device="cpu", hidden=(16,),
        rollout_decisions=32, batch_size=16,
    ))
    try:
        count = trainer.collect_step()
        assert 0 <= count <= 16
        assert trainer.obs.shape == (16, fc.OBS_SIZE)
        assert trainer.masks.shape == (16, fc.MASK_WORDS)
        assert trainer.total_decisions == 16
    finally:
        trainer.close()


def test_multiprocess_short_ppo_update_has_finite_losses():
    trainer = BatchedTrainer(TrainerConfig(
        num_envs=32, env_workers=2, device="cpu", hidden=(32,),
        rollout_decisions=32, batch_size=16, update_epochs=1,
    ))
    try:
        for _ in range(1000):
            trainer.collect_step()
            if trainer._completed >= 32:
                break
        else:
            pytest.fail("multiprocess learner transitions did not complete")
        metrics = trainer.update_ppo()
        assert metrics
        assert all(np.isfinite(value) for value in metrics.values())
    finally:
        trainer.close()


def test_terminal_series_tracks_rolling_win_rate_and_reward():
    trainer = BatchedTrainer(TrainerConfig(
        num_envs=4, device="cpu", hidden=(16,),
        win_rate_window_episodes=3,
    ))
    trainer._record_terminal_rewards(
        np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    )
    metrics = trainer._terminal_summary()
    assert metrics["terminal_episodes"] == 4
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["mean_reward"] == pytest.approx(0.0)
    assert metrics["rolling_episodes"] == 3
    assert metrics["rolling_win_rate"] == pytest.approx(2 / 3)
    assert metrics["rolling_reward"] == pytest.approx(1 / 3)


def test_resume_preserves_optimizer_counters_rng_and_anneal(tmp_path):
    checkpoint = tmp_path / "resume.pt"
    source = BatchedTrainer(TrainerConfig(
        num_envs=8, device="cpu", hidden=(16,),
        total_learner_decisions=1000,
    ))
    parameter = next(source.net.parameters())
    parameter.sum().backward()
    source.optimizer.step()
    source.optimizer.zero_grad(set_to_none=True)
    source.learner_decisions = 250
    source.total_decisions = 1000
    source.updates = 7
    source.episodes = 11
    source.rng.random(5)
    torch.rand(5)
    source._record_terminal_rewards(
        np.array([1.0, -1.0, 1.0], dtype=np.float32)
    )
    source.save(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)

    resumed = BatchedTrainer(TrainerConfig(
        num_envs=8, device="cpu", hidden=(16,), resume=str(checkpoint),
        anchor_ref=str(checkpoint), anchor_coef=0.5,
        anchor_coef_final=0.1, total_learner_decisions=1000,
    ))
    try:
        assert resumed.learner_decisions == 250
        assert resumed.total_decisions == 1000
        assert resumed.updates == 7
        assert resumed.episodes == 11
        assert resumed._anchor_beta() == pytest.approx(0.4)
        assert resumed.rng.bit_generator.state == saved["np_rng_state"]
        assert torch.equal(torch.get_rng_state(), saved["torch_rng_state"])
        saved_step = next(iter(saved["optimizer_state"]["state"].values()))["step"]
        resumed_step = next(iter(resumed.optimizer.state.values()))["step"]
        assert torch.equal(resumed_step.cpu(), saved_step.cpu())
        assert resumed._terminal_summary()["terminal_episodes"] == 3

        summary = resumed.run(
            max_learner_decisions=251,
            benchmark_only=True,
            warmup_steps=0,
        )
        assert summary["learner_decisions"] > 250
        assert summary["measurement_learner_decisions"] > 0
    finally:
        resumed.close()
