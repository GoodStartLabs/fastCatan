#!/usr/bin/env python3
"""Smoke tests for the Gymnasium single-env wrapper."""
from __future__ import annotations
import sys
import numpy as np

import fastcatan as fc


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


def test_reset():
    env = fc.GymEnv(seed=42)
    obs, info = env.reset(seed=42)
    fails = 0
    fails += fail(obs.shape == (fc.OBS_SIZE,), f"obs shape {obs.shape}")
    fails += fail(obs.dtype == np.float32, f"obs dtype {obs.dtype}")
    fails += fail("action_mask" in info, "no action_mask in info")
    fails += fail(info["action_mask"].shape == (fc.MASK_WORDS,),
                  f"mask shape {info['action_mask'].shape}")
    fails += fail("current_player" in info, "no current_player in info")
    fails += fail("phase" in info, "no phase in info")
    return fails


def test_spaces():
    env = fc.GymEnv()
    fails = 0
    fails += fail(env.action_space.n == fc.NUM_ACTIONS,
                  f"action_space.n={env.action_space.n}")
    fails += fail(env.observation_space.shape == (fc.OBS_SIZE,),
                  f"obs_space shape {env.observation_space.shape}")
    return fails


def test_step_random_play():
    """Random legal play with both seats — game should terminate."""
    rng = np.random.default_rng(7)
    opponent = fc.random_legal_policy(rng)
    env = fc.GymEnv(seat=0, seed=42, opponent_fn=opponent, max_steps=10000)
    obs, info = env.reset(seed=42)

    fails = 0
    n_steps = 0
    terminated = False
    truncated = False
    total_reward = 0.0
    for _ in range(10000):
        if not info["is_learner_turn"]:
            fails += fail(False, "after _advance_to_learner, should be learner turn")
            break
        # Pick random legal action for learner.
        a = opponent(obs, info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(a)
        n_steps += 1
        total_reward += reward
        if terminated or truncated:
            break

    fails += fail(terminated, f"game didn't terminate (truncated={truncated} steps={n_steps})")
    fails += fail(reward in (1.0, -1.0, 0.0), f"final reward = {reward}")
    print(f"  steps={n_steps} terminated={terminated} reward={reward}")
    return fails


def test_lowest_legal_deterministic():
    env = fc.GymEnv(seat=0, seed=42, opponent_fn=fc.lowest_legal_policy, max_steps=10000)
    seq1 = []
    obs, info = env.reset(seed=42)
    for _ in range(50):
        a = fc.lowest_legal_policy(obs, info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(a)
        seq1.append(int(a))
        if terminated or truncated:
            break

    env2 = fc.GymEnv(seat=0, seed=42, opponent_fn=fc.lowest_legal_policy, max_steps=10000)
    seq2 = []
    obs, info = env2.reset(seed=42)
    for _ in range(50):
        a = fc.lowest_legal_policy(obs, info["action_mask"])
        obs, reward, terminated, truncated, info = env2.step(a)
        seq2.append(int(a))
        if terminated or truncated:
            break

    return fail(seq1 == seq2, f"deterministic Gym replay diverged: {seq1[:10]} vs {seq2[:10]}")


def test_seat_choice():
    """Different seats see different obs (learner vs opponent indicator differs)."""
    rng = np.random.default_rng(0)
    opp = fc.random_legal_policy(rng)
    e0 = fc.GymEnv(seat=0, seed=42, opponent_fn=opp)
    e1 = fc.GymEnv(seat=1, seed=42, opponent_fn=opp)
    obs0, _ = e0.reset(seed=42)
    obs1, _ = e1.reset(seed=42)
    return fail(not np.array_equal(obs0, obs1),
                "obs identical for seat 0 and seat 1 (POV-relative encoding)")


def main():
    total = 0
    print("== test_reset ==");                       total += test_reset()
    print("== test_spaces ==");                      total += test_spaces()
    print("== test_step_random_play ==");            total += test_step_random_play()
    print("== test_lowest_legal_deterministic ==");  total += test_lowest_legal_deterministic()
    print("== test_seat_choice ==");                 total += test_seat_choice()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} failures")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
