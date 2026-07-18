"""Training-loop wiring tests — the failures that do NOT throw, they just train
on garbage (Gin Rummy lost eight runs to exactly these). Phase-0 exit guard.

Vehicles are the shipped surfaces: the single-agent gym env (`models/env.py`,
learner = seat 0) that the PPO trainer drives, the `BatchedEnv` obs path, and the
per-seat self-play driver (`models/selfplay/eval_seats.play_one`).

Documented conventions asserted here (see `docs/reports/0.3-freeze.md`):
  * terminal reward is from the learner seat's view: win=+1, loss=-1, and a
    no-winner terminal (stall / MAX_TURNS backstop) = TIE_REWARD (-2), NOT 0/None.
  * mid-episode reward is exactly 0.0.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

import fastcatan as fc
from models.env import FastCatanEnv, LEARNER_SEAT, TIE_REWARD, WIN_VP
from models.selfplay.eval_seats import play_one
from models.selfplay.opponents import Opponent


# ---------------------------------------------------------------------------
# 1. terminal reward lands on the correct seat; no-winner convention documented
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_terminal_reward_convention(seed):
    env = FastCatanEnv(seed=seed, opponent="random")
    env.reset(seed=seed)
    rng = random.Random(seed ^ 0x5EED)
    saw_terminal = False
    for _ in range(300_000):
        legal = np.nonzero(env.action_masks())[0]
        assert len(legal) > 0, "empty learner mask mid-game"
        _, reward, term, trunc, _ = env.step(int(rng.choice(legal)))
        if not (term or trunc):
            assert reward == 0.0, f"mid-episode reward must be 0, got {reward}"
            continue
        # Terminal: reward must follow the documented seat-0 convention exactly.
        vps = [env._env.player_vp(p) for p in range(fc.NUM_PLAYERS)]
        winner = next((p for p in range(fc.NUM_PLAYERS) if vps[p] >= WIN_VP), None)
        if winner is None:
            assert reward == TIE_REWARD, f"no-winner terminal must be {TIE_REWARD}, got {reward}"
        elif winner == LEARNER_SEAT:
            assert reward == 1.0, f"learner win must be +1, got {reward}"
        else:
            assert reward == -1.0, f"learner loss must be -1, got {reward}"
        saw_terminal = True
        break
    assert saw_terminal, "game never terminated within the step budget"


# ---------------------------------------------------------------------------
# 2. batched obs rows are POV-correct and per-env independent
# ---------------------------------------------------------------------------


def _first_legal(mask_word_row):
    for w in range(fc.MASK_WORDS):
        bits = int(mask_word_row[w])
        if bits:
            return w * 64 + (bits & -bits).bit_length() - 1
    return int(fc.SKIP_ACTION)


def test_batched_obs_rows_pov_correct_and_independent():
    from bridge import state_mirror as M

    N = 8
    be = fc.BatchedEnv(N, 12345)
    masks = np.zeros((N, fc.MASK_WORDS), dtype=np.uint64)
    rewards = np.zeros(N, dtype=np.float32)
    dones = np.zeros(N, dtype=np.uint8)
    for _ in range(40):  # warm the batch so states differ across envs
        be.write_masks(masks)
        acts = np.array([_first_legal(masks[i]) for i in range(N)], dtype=np.uint32)
        be.step(acts, rewards, dones)

    buf = np.zeros((N, fc.SNAPSHOT_BYTES), dtype=np.uint8)
    be.save_snapshots(buf)
    povs = (np.arange(N, dtype=np.uint8) % 4)
    base = np.zeros((N, fc.OBS_SIZE), dtype=np.float32)
    be.write_obs_pov_batch(povs, base)

    # POV-correct: each batched row equals a single-Env write_obs at that pov.
    env = fc.Env()
    ref = np.zeros(fc.OBS_SIZE, dtype=np.float32)
    for i in range(N):
        env.load_snapshot(buf[i].tobytes())
        env.write_obs(int(povs[i]), ref)
        assert np.array_equal(base[i], ref), f"batched pov row {i} != single-Env write_obs"

    # Independent: perturb ONE env's hidden+public self state, reload the batch,
    # and require ONLY that env's obs row to move.
    j = 3
    snap = M.parse_snapshot(buf[j].tobytes())
    pov_j = int(povs[j])
    snap.gs.player_resources[pov_j][0] += 1
    snap.gs.player_handsize[pov_j] += 1
    snap.gs.bank[0] -= 1  # keep resource conservation intact
    newbuf = buf.copy()
    newbuf[j] = np.frombuffer(M.to_bytes(snap), dtype=np.uint8)
    be.load_snapshots(newbuf)
    after = np.zeros((N, fc.OBS_SIZE), dtype=np.float32)
    be.write_obs_pov_batch(povs, after)
    for i in range(N):
        if i == j:
            assert not np.array_equal(after[i], base[i]), "perturbed env row did not change"
        else:
            assert np.array_equal(after[i], base[i]), (
                f"row {i} changed though only env {j} was perturbed (cross-env bleed)")


# ---------------------------------------------------------------------------
# 3. opponent-slot loading: policy[s] decides seat s, on seat s's POV obs
# ---------------------------------------------------------------------------


class _Sentinel(Opponent):
    """Records (own list-index, acting seat, obs-matches-write_obs(seat)) every
    call. Closes over the live env so it can read the seat and recompute the POV
    obs the driver *should* have handed it (the env has not stepped yet)."""

    def __init__(self, idx, env, log):
        self.idx = idx
        self.env = env
        self.log = log

    def act(self, obs, mask):
        seat = self.env.current_player
        ref = np.zeros(fc.OBS_SIZE, dtype=np.float32)
        self.env.write_obs(seat, ref)
        self.log.append((self.idx, seat, bool(np.array_equal(obs, ref))))
        legal = np.nonzero(mask)[0]
        return int(legal[0])


def _run_sentinels(seed=7):
    """Drive one game with a sentinel at each slot k (its idx == k). play_one maps
    slot k -> seat k, so a correct driver only ever asks sentinel k on seat k."""
    env = fc.Env()
    env.reset(seed)
    log = []
    policies = [_Sentinel(k, env, log) for k in range(4)]
    obs_buf = np.zeros(fc.OBS_SIZE, dtype=np.float32)
    mask_buf = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
    play_one(env, policies, obs_buf, mask_buf, None, max_steps=150000)
    return log


def test_opponent_slot_indexing_and_pov():
    log = _run_sentinels(seed=7)
    assert log, "no decisions were made"
    for idx, seat, aligned in log:
        assert idx == seat, f"policy at slot {idx} was asked to play seat {seat}"
        assert aligned, f"policy for seat {seat} got the wrong POV obs"


# ---------------------------------------------------------------------------
# 4. seat/board reshuffle between episodes
# ---------------------------------------------------------------------------


def test_board_varies_between_episodes():
    """A fused loop that reset to the same board every episode trains on one
    board. reset() must draw a fresh game each episode."""
    env = FastCatanEnv(seed=99, opponent="random")
    o1, _ = env.reset(seed=99)
    o2, _ = env.reset()  # next episode, no explicit seed -> advances the seed seq
    o3, _ = env.reset()
    assert not np.array_equal(o1, o2), "episode 2 reused episode 1's board"
    assert not np.array_equal(o2, o3), "episode 3 reused episode 2's board"


def test_seat_assignment_follows_policy_slot():
    """The per-seat driver places policy list-slot k at seat k — the mechanism a
    training loop uses to randomize the learner's seat across episodes (put the
    learner at a different slot). The slot->seat invariant must hold on every
    board."""
    for seed in (100, 101, 202, 303):
        log = _run_sentinels(seed=seed)
        assert log, f"no decisions on seed {seed}"
        assert all(idx == seat for idx, seat, _ in log), (
            f"slot->seat mapping broke on board seed {seed}")
