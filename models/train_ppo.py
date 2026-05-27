"""MaskablePPO training entrypoint.

Run:
    python -m models.train_ppo --num-envs 16 --total-steps 5_000_000

M2 gate: >90% win rate vs random over 1000 games (see models/eval.py).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from models.env import FastCatanEnv


CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


def _mask_fn(env):
    return env.action_masks()


def _make_env(seed: int):
    def _thunk():
        e = FastCatanEnv(seed=seed)
        e = ActionMasker(e, _mask_fn)
        # Monitor records episode reward/length so SB3 logs rollout/ep_rew_mean
        # and ep_len_mean — without it you cannot see whether the agent learns
        # (the prior 5k-step run logged nothing and looked identical to random).
        return Monitor(e)

    return _thunk


def _build_vec_env(num_envs: int, base_seed: int, use_subproc: bool):
    fns = [_make_env(base_seed + i) for i in range(num_envs)]
    if use_subproc and num_envs > 1:
        return SubprocVecEnv(fns)
    return DummyVecEnv(fns)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--net-arch", type=str, default="64,64",
                   help="Hidden layer sizes for the pi and vf nets, comma-separated. "
                        "Default '64,64' = the SB3 MlpPolicy default (small for a "
                        "1084-dim obs: fine vs random, likely a ceiling vs Alpha-Beta). "
                        "Scale up for M3 self-play / M4, e.g. '256,256' or '512,256' — "
                        "see models/PLAN.md §4. Bigger net = slower fps (the C++ sim is "
                        "cheap, so the policy net becomes the throughput bottleneck).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default=str(CKPT_DIR))
    p.add_argument("--save-freq", type=int, default=500_000)
    p.add_argument("--run-name", type=str, default="ppo_random")
    p.add_argument("--subproc", action="store_true",
                   help="Use SubprocVecEnv (multi-process). Default is DummyVecEnv: "
                        "the C++ sim is cheap enough (~50 ns/step) that per-step "
                        "cross-process obs pickling usually costs more than the step.")
    args = p.parse_args()

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = save_dir / "tb"

    # net_arch applies to both the policy and value heads (SB3 reads a flat list
    # as pi=vf=list). Default "64,64" reproduces the implicit SB3 default, so
    # existing checkpoints stay architecturally identical.
    net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]

    env = _build_vec_env(args.num_envs, args.seed, use_subproc=args.subproc)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        policy_kwargs=dict(net_arch=net_arch),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        seed=args.seed,
        tensorboard_log=str(tb_dir),
        verbose=1,
    )
    print(f"[train] run={args.run_name} net_arch={net_arch} "
          f"num_envs={args.num_envs} total_steps={args.total_steps} lr={args.lr}")

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.save_freq // args.num_envs),
        save_path=str(save_dir),
        name_prefix="ppo",
    )

    model.learn(
        total_timesteps=args.total_steps,
        callback=ckpt_cb,
        progress_bar=True,
    )

    final = save_dir / "ppo_final.zip"
    model.save(str(final))
    print(f"[train] saved final model -> {final}")


if __name__ == "__main__":
    main()
