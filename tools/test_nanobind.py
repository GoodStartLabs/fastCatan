#!/usr/bin/env python3
"""Smoke test for the nanobind-based fastcatan package.

Run after `uv pip install -e .`.
"""

from __future__ import annotations
import sys
import numpy as np

import fastcatan as fc

NUM_ACTIONS = fc.NUM_ACTIONS


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


def test_module_constants():
    fails = 0
    fails += fail(fc.OBS_SIZE == 724, f"OBS_SIZE = {fc.OBS_SIZE}")
    fails += fail(fc.MASK_WORDS == 5, f"MASK_WORDS = {fc.MASK_WORDS}")
    fails += fail(fc.NUM_ACTIONS == 296, f"NUM_ACTIONS = {fc.NUM_ACTIONS}")
    fails += fail(fc.NUM_PLAYERS == 4, f"NUM_PLAYERS = {fc.NUM_PLAYERS}")
    fails += fail(fc.action.ROLL_DICE == 180, f"ROLL_DICE = {fc.action.ROLL_DICE}")
    fails += fail(fc.action.END_TURN == 181, f"END_TURN = {fc.action.END_TURN}")
    return fails


def test_single_env():
    env = fc.Env()
    env.reset(42)
    fails = 0
    fails += fail(env.phase == 0, f"phase = {env.phase}")
    fails += fail(env.dice_roll == 0, f"dice_roll = {env.dice_roll}")
    fails += fail(0 <= env.current_player < 4, f"current_player = {env.current_player}")
    return fails


def test_batched_create_reset():
    env = fc.BatchedEnv(num_envs=16, seed=42)
    env.reset()
    fails = 0
    fails += fail(env.num_envs == 16, f"num_envs = {env.num_envs}")
    for i in range(16):
        fails += fail(env.phase(i) == 0, f"env {i} phase = {env.phase(i)}")
    return fails


def test_batched_zero_copy_step():
    n = 32
    env = fc.BatchedEnv(num_envs=n, seed=42)
    env.reset()

    actions = np.zeros(n, dtype=np.uint32)
    rewards = np.zeros(n, dtype=np.float32)
    dones   = np.zeros(n, dtype=np.uint8)
    masks   = np.zeros((n, fc.MASK_WORDS), dtype=np.uint64)

    env.write_masks(masks)
    fails = 0
    # In initial-settle phase, every env should have all 54 nodes legal.
    for i in range(n):
        bits = sum(bin(int(masks[i, w])).count('1') for w in range(fc.MASK_WORDS))
        fails += fail(bits == 54, f"env {i} mask bits = {bits}")

    # Pick first legal action for every env.
    for i in range(n):
        for w in range(fc.MASK_WORDS):
            v = int(masks[i, w])
            if v:
                actions[i] = w * 64 + (v & -v).bit_length() - 1
                break
    env.step(actions, rewards, dones)
    return fails


def test_obs_shape():
    n = 8
    env = fc.BatchedEnv(num_envs=n, seed=42)
    env.reset()
    obs = np.zeros((n, fc.OBS_SIZE), dtype=np.float32)
    env.write_obs(obs)
    fails = fail(obs.shape == (n, fc.OBS_SIZE), f"obs shape = {obs.shape}")
    fails += fail(obs.min() >= 0.0, "obs has negative values")
    fails += fail(np.isfinite(obs).all(), "obs has non-finite values")
    return fails


def test_buffer_shape_validation():
    """Pass mismatched buffer; should raise."""
    n = 8
    env = fc.BatchedEnv(num_envs=n, seed=42)
    env.reset()
    actions = np.zeros(n + 1, dtype=np.uint32)  # wrong shape
    rewards = np.zeros(n, dtype=np.float32)
    dones   = np.zeros(n, dtype=np.uint8)
    fails = 0
    try:
        env.step(actions, rewards, dones)
        fails += fail(False, "should have raised on shape mismatch")
    except RuntimeError:
        pass
    except Exception as ex:
        fails += fail(False, f"raised unexpected exception: {type(ex).__name__}")
    return fails


def main():
    total = 0
    print("== test_module_constants ==");        total += test_module_constants()
    print("== test_single_env ==");              total += test_single_env()
    print("== test_batched_create_reset ==");    total += test_batched_create_reset()
    print("== test_batched_zero_copy_step =="); total += test_batched_zero_copy_step()
    print("== test_obs_shape ==");               total += test_obs_shape()
    print("== test_buffer_shape_validation =="); total += test_buffer_shape_validation()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} failures")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
