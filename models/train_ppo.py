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
from models.env_shaped import VPShapedEnv
from models.ckpt import write_stamp


CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


def _mask_fn(env):
    return env.action_masks()


def _make_env(
    seed: int, shaped: bool, shaping_coef: float, gamma: float,
    opponent: str, ab_depth: int, ab_prune: bool, suppress_p2p_trade: bool,
):
    def _thunk():
        # VP-only potential shaping (models/env_shaped.py) when --shaped; its
        # gamma must match the PPO gamma so the shaping discount matches the
        # learner's return. Else the bare sparse-terminal env.
        opp = dict(opponent=opponent, ab_depth=ab_depth, ab_prune=ab_prune,
                   suppress_p2p_trade=suppress_p2p_trade)
        if shaped:
            e = VPShapedEnv(seed=seed, shaping_coef=shaping_coef, gamma=gamma, **opp)
        else:
            e = FastCatanEnv(seed=seed, **opp)
        e = ActionMasker(e, _mask_fn)
        # Monitor records episode reward/length so SB3 logs rollout/ep_rew_mean
        # and ep_len_mean — without it you cannot see whether the agent learns
        # (the prior 5k-step run logged nothing and looked identical to random).
        return Monitor(e)

    return _thunk


def _build_vec_env(
    num_envs: int, base_seed: int, use_subproc: bool,
    shaped: bool, shaping_coef: float, gamma: float,
    opponent: str, ab_depth: int, ab_prune: bool, suppress_p2p_trade: bool,
):
    fns = [_make_env(base_seed + i, shaped, shaping_coef, gamma,
                     opponent, ab_depth, ab_prune, suppress_p2p_trade)
           for i in range(num_envs)]
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
    p.add_argument("--opponent", type=str, default="random",
                   choices=["random", "alphabeta"],
                   help="Policy for seats 1-3. 'random' (default) or 'alphabeta' "
                        "(native Catanatron-AlphaBeta port via Env.ab_decide) to "
                        "train directly against the M4 eval opponent. AlphaBeta is "
                        "far slower per step than random — lower --num-envs and/or "
                        "use --ab-depth 1; see models/PLAN.md.")
    p.add_argument("--ab-depth", type=int, default=2,
                   help="AlphaBeta search depth when --opponent alphabeta "
                        "(Catanatron default 2; 1 = ValueFunctionPlayer, ~3x faster).")
    p.add_argument("--ab-prune", action="store_true",
                   help="Enable AlphaBeta action pruning (most-impactful robber + "
                        "1-tile initial settlements) for a faster, narrower search.")
    p.add_argument("--no-trades", action="store_true",
                   help="Suppress player-to-player trades for the learner AND the "
                        "opponents (maritime bank/port trades stay) — the canonical "
                        "thesis ablation. Train and eval must match.")
    p.add_argument("--shaped", action="store_true",
                   help="Use VPShapedEnv (models/env_shaped.py): VP-only potential "
                        "shaping on top of the sparse +1/-1/-1 terminal. gamma is "
                        "shared with the learner.")
    p.add_argument("--shaping-coef", type=float, default=0.1,
                   help="Potential coefficient phi=coef*own_VP (only with --shaped).")
    p.add_argument("--init-from", type=str, default=None,
                   help="Warm-start weights from a checkpoint (set_parameters, weights "
                        "only — keeps the CLI lr/ent_coef). --net-arch MUST match the "
                        "checkpoint or set_parameters fails -> falls back to scratch "
                        "(logged). E.g. the 50M base is [512,512,256].")
    args = p.parse_args()

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = save_dir / "tb"

    # net_arch applies to both the policy and value heads (SB3 reads a flat list
    # as pi=vf=list). Default "64,64" reproduces the implicit SB3 default, so
    # existing checkpoints stay architecturally identical.
    net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]

    env = _build_vec_env(
        args.num_envs, args.seed, use_subproc=args.subproc,
        shaped=args.shaped, shaping_coef=args.shaping_coef, gamma=args.gamma,
        opponent=args.opponent, ab_depth=args.ab_depth, ab_prune=args.ab_prune,
        suppress_p2p_trade=args.no_trades,
    )

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
    if args.init_from:
        # Weights only — keeps the CLI lr/ent_coef. Arch must match the ckpt;
        # a mismatch raises -> log and fall through to scratch (mirrors the
        # self-play warm-start in models/selfplay/train_selfplay.py).
        try:
            model.set_parameters(args.init_from, device=model.device)
            print(f"[train] warm-started weights from {args.init_from}")
        except (ValueError, RuntimeError, KeyError) as e:
            print(f"[train] WARM-START SKIPPED (arch mismatch with "
                  f"{args.init_from}? net_arch={net_arch}): "
                  f"{type(e).__name__}: {str(e)[:140]} -> training from scratch")

    opp_desc = args.opponent + (
        f"(depth={args.ab_depth},prune={args.ab_prune})"
        if args.opponent == "alphabeta" else "")
    print(f"[train] run={args.run_name} net_arch={net_arch} opponent={opp_desc} "
          f"num_envs={args.num_envs} total_steps={args.total_steps} lr={args.lr} "
          f"trades={'off' if args.no_trades else 'on'} "
          f"shaped={args.shaped} coef={args.shaping_coef if args.shaped else None}")

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
    write_stamp(final)
    print(f"[train] saved final model -> {final}")


if __name__ == "__main__":
    main()
