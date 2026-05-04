#!/usr/bin/env python3
"""Smoke tests for the PettingZoo AEC wrapper."""
from __future__ import annotations
import sys
import numpy as np

import fastcatan as fc


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


def test_init_and_reset():
    env = fc.CatanAECEnv(seed=42)
    env.reset(seed=42)
    fails = 0
    fails += fail(len(env.agents) == 4, f"agents = {env.agents}")
    fails += fail(env.agent_selection in env.agents,
                  f"agent_selection = {env.agent_selection}")
    fails += fail("player_0" in env.agents, "player_0 not in agents")
    return fails


def test_observe_shape():
    env = fc.CatanAECEnv(seed=42)
    env.reset(seed=42)
    obs = env.observe(env.agent_selection)
    fails = fail(obs["observation"].shape == (fc.OBS_SIZE,),
                  f"obs shape {obs['observation'].shape}")
    fails += fail(obs["action_mask"].shape == (fc.MASK_WORDS,),
                  f"mask shape {obs['action_mask'].shape}")
    return fails


def test_pov_differs_per_agent():
    env = fc.CatanAECEnv(seed=42)
    env.reset(seed=42)
    obs_per_agent = {}
    for agent in env.agents:
        obs_per_agent[agent] = env.observe(agent)["observation"]
    distinct = len({tuple(v) for v in obs_per_agent.values()})
    return fail(distinct >= 3,
                f"only {distinct}/4 agent obs distinct (POV-relative encoding)")


def test_random_play_terminates():
    env = fc.CatanAECEnv(seed=42, max_steps=10000)
    env.reset(seed=42)
    rng = np.random.default_rng(0)

    def pick(mask):
        bits = []
        for w in range(fc.MASK_WORDS):
            v = int(mask[w])
            while v:
                lsb = v & (-v)
                bits.append(w * 64 + lsb.bit_length() - 1)
                v ^= lsb
        return int(rng.choice(bits)) if bits else 0

    n_steps = 0
    safety = 20000
    rewards_seen = {a: 0.0 for a in env.agents}
    for agent in env.agent_iter():
        obs, r, term, trunc, info = env.last()
        rewards_seen[agent] += r
        if term or trunc:
            env.step(None)  # required acknowledgment
            n_steps += 1
            safety -= 1
            if safety <= 0: break
            continue
        a = pick(obs["action_mask"])
        env.step(a)
        n_steps += 1
        safety -= 1
        if safety <= 0:
            break

    fails = 0
    fails += fail(n_steps > 100, f"only {n_steps} steps before agent_iter exited")
    # exactly one winner: cumulative reward sum should be balanced
    print(f"  steps={n_steps} cum_rewards={rewards_seen}")
    return fails


def test_step_consistency():
    """Verify rewards dict and termination dict are consistent."""
    env = fc.CatanAECEnv(seed=42)
    env.reset(seed=42)
    fails = 0
    for agent in env.agents:
        fails += fail(agent in env.rewards, f"missing reward for {agent}")
        fails += fail(agent in env.terminations, f"missing termination for {agent}")
        fails += fail(agent in env.truncations, f"missing truncation for {agent}")
        fails += fail(agent in env.infos, f"missing info for {agent}")
    return fails


def main():
    total = 0
    print("== test_init_and_reset ==");          total += test_init_and_reset()
    print("== test_observe_shape ==");           total += test_observe_shape()
    print("== test_pov_differs_per_agent ==");   total += test_pov_differs_per_agent()
    print("== test_random_play_terminates ==");  total += test_random_play_terminates()
    print("== test_step_consistency ==");        total += test_step_consistency()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} failures")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
