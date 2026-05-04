#!/usr/bin/env python3
"""End-to-end smoke trainer: MaskablePPO vs random opponent on fastCatan.

Verifies the full RL stack works locally:
  - PyTorch (MPS on Mac, CUDA on HPC)
  - sb3-contrib MaskablePPO
  - fastcatan.GymEnv with bool action mask
  - Random opponent for the 3 non-learner seats

Usage:
    source .venv/bin/activate
    pip install -e . --no-build-isolation       # if not installed
    python3 tools/train_smoke.py [--total-timesteps 5000] [--device auto]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import fastcatan as fc
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mask_fn(env) -> np.ndarray:
    """sb3-contrib's ActionMasker wrapper expects this signature."""
    return env.unwrapped._make_info()["action_mask"]


def make_env(seed: int) -> "ActionMasker":  # type: ignore[name-defined]
    from sb3_contrib.common.wrappers import ActionMasker
    rng = np.random.default_rng(seed)
    opponent = fc.random_legal_policy(rng)
    base = fc.GymEnv(seat=0, seed=seed, opponent_fn=opponent, max_steps=8000)
    return ActionMasker(base, mask_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-timesteps", type=int, default=1024,
                    help="rough number of env steps for the smoke run")
    ap.add_argument("--device", default="auto", help="cpu / mps / cuda / auto")
    ap.add_argument("--save", default=None, help="optional checkpoint path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-episodes", type=int, default=20,
                    help="evaluation episodes after training; 0 to skip")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device:     {device}")
    print(f"torch:      {torch.__version__}")
    print(f"OBS_SIZE:   {fc.OBS_SIZE}")
    print(f"NUM_ACTIONS:{fc.NUM_ACTIONS}")
    print()

    env = make_env(args.seed)

    # Tiny network so the smoke run is fast on any device.
    policy_kwargs = dict(net_arch=dict(pi=[128, 64], vf=[128, 64]))

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=2,
        device=device,
        verbose=1,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
    )

    print(f"training for {args.total_timesteps} timesteps...", flush=True)
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    print("training complete.", flush=True)

    if args.eval_episodes <= 0:
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            model.save(args.save)
            print(f"saved checkpoint to {args.save}")
        return

    eval_env = make_env(args.seed + 1)
    wins = losses = ties = 0
    for ep in range(args.eval_episodes):
        obs, info = eval_env.reset(seed=args.seed + 100 + ep)
        terminated = truncated = False
        ep_reward = 0.0
        while not (terminated or truncated):
            mask = info["action_mask"]
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            ep_reward += float(reward)
        if ep_reward > 0.5:
            wins += 1
        elif ep_reward < -0.5:
            losses += 1
        else:
            ties += 1

    print()
    print("=== eval (deterministic policy vs 3 random opponents, 50 episodes) ===")
    print(f"  wins:   {wins}/50  ({100*wins/50:.0f}%)")
    print(f"  losses: {losses}/50")
    print(f"  ties/truncated: {ties}/50")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.save)
        print(f"saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
