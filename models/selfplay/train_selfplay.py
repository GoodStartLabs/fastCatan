"""M3 iterative self-play with frozen-snapshot rotation.

One MaskablePPO model + one DummyVecEnv of SelfPlayEnv; the opponent pool is
shared and mutated across rounds. Each round:
  1. train the learner for `--steps-per-round` (= the snapshot interval),
  2. freeze the current policy to `snap_<timesteps>.zip` and add it to the pool,
  3. inline-gate the latest snapshot vs the snapshot from `--gate-lag` rounds ago.

Warm-start from an M2 checkpoint loads *weights only* (`set_parameters`) so the
swept hyperparams (lr, ent_coef) still apply. DummyVecEnv only: the pool holds
loaded torch models (not cheaply picklable) and shares across envs in-process.

Run (real schedule, warm-started from the M2 10M checkpoint):
    python -m models.selfplay.train_selfplay \
        --init-from models/checkpoints/ppo_random_10m/ppo_final.zip \
        --seed-pool --num-envs 8 --steps-per-round 1_000_000 --num-rounds 12 \
        --run-name sp_main
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from models.selfplay.gate import gate_result, play_2v2
from models.selfplay.opponents import OpponentPool, PolicyOpponent
from models.selfplay.selfplay_env import SelfPlayEnv


CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def _mask_fn(env):
    return env.action_masks()


def _tb_dir_if_available(tb_dir: Path) -> str | None:
    """SB3 hard-errors if tensorboard_log is set but tensorboard is missing.
    Return the path only when tensorboard imports; else None (SB3 still logs
    to stdout). Avoids coupling self-play runs to an optional dep."""
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print("[selfplay] tensorboard not installed -> stdout logging only")
        return None
    return str(tb_dir)


def _build_vec_env(
    pool: OpponentPool, num_envs: int, base_seed: int, suppress_p2p_trade: bool
) -> DummyVecEnv:
    def _make(seed: int):
        def _thunk():
            e = SelfPlayEnv(pool, seed=seed, suppress_p2p_trade=suppress_p2p_trade)
            e = ActionMasker(e, _mask_fn)
            return Monitor(e)

        return _thunk

    return DummyVecEnv([_make(base_seed + i) for i in range(num_envs)])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # self-play schedule
    p.add_argument("--init-from", type=str, default=None,
                   help="M2 checkpoint to warm-start weights from (set_parameters).")
    p.add_argument("--seed-pool", action="store_true",
                   help="Add the init checkpoint to the pool so round 0 already "
                        "faces a real opponent (else round 0 is vs random).")
    p.add_argument("--num-rounds", type=int, default=10)
    p.add_argument("--steps-per-round", type=int, default=1_000_000,
                   help="Train steps between snapshots == the snapshot interval.")
    p.add_argument("--num-envs", type=int, default=8)
    # pool sampling
    p.add_argument("--p-random", type=float, default=0.2)
    p.add_argument("--pool-window", type=int, default=5)
    p.add_argument("--opponent-device", type=str, default="cpu",
                   help="Device for frozen opponents. CPU is faster for the "
                        "single-obs forward passes (no GPU launch/transfer "
                        "overhead per call); the learner keeps its own device.")
    p.add_argument("--no-p2p-trade", action="store_true",
                   help="Forbid player-to-player trades in train AND gate. Kills "
                        "the trade-loop stall so self-play games terminate; without "
                        "it strong-vs-strong games stall to the cap and the gate is "
                        "inconclusive. See SelfPlayEnv / models/selfplay/PLAN.md.")
    # PPO hyperparams (mirror models/train_ppo.py defaults)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--lr-schedule", choices=["constant", "linear"],
                   default="constant",
                   help="'linear' decays lr from --lr toward 0 across the whole "
                        "run (per-round stepwise: lr*(1-round/num_rounds)). Avoids "
                        "the late-round oscillation that flattens self-play gains.")
    p.add_argument("--target-kl", type=float, default=None,
                   help="PPO early-stops an update once policy KL exceeds this "
                        "(e.g. 0.03). Keeps steps conservative while the opponent "
                        "pool shifts each round; default None = off.")
    p.add_argument("--net-arch", type=str, default=None,
                   help="Hidden layer sizes for the pi & vf MLPs, comma-separated "
                        "(e.g. '256,256'). Default: SB3 default [64,64]. NOTE: an "
                        "arch differing from --init-from's checkpoint cannot "
                        "warm-start (shape mismatch) -> that run trains from "
                        "scratch (logged). Seed-pool opponents keep their own arch.")
    # gate
    p.add_argument("--gate-lag", type=int, default=2,
                   help="Compare latest snapshot vs the one this many rounds ago.")
    p.add_argument("--gate-games", type=int, default=200)
    p.add_argument("--gate-threshold", type=float, default=0.55)
    # bookkeeping
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default=str(CKPT_DIR))
    p.add_argument("--run-name", type=str, default="selfplay")
    p.add_argument("--progress", action="store_true", help="SB3 progress bar.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = save_dir / "tb"

    pool = OpponentPool(
        seats=[1, 2, 3], seed=args.seed,
        p_random=args.p_random, window=args.pool_window,
    )
    env = _build_vec_env(pool, args.num_envs, args.seed, args.no_p2p_trade)

    policy_kwargs = None
    if args.net_arch:
        hidden = [int(x) for x in args.net_arch.split(",") if x.strip()]
        policy_kwargs = {"net_arch": hidden}
        print(f"[selfplay] net_arch = {hidden}")

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        target_kl=args.target_kl,
        seed=args.seed,
        tensorboard_log=_tb_dir_if_available(tb_dir),
        verbose=1,
    )

    if args.init_from:
        # Weights only — keeps our swept lr/ent_coef. Arch must match the ckpt;
        # a swept --net-arch that differs will mismatch -> fall back to scratch.
        try:
            model.set_parameters(args.init_from, device=model.device)
            print(f"[selfplay] warm-started weights from {args.init_from}")
        except (ValueError, RuntimeError, KeyError) as e:
            print(f"[selfplay] WARM-START SKIPPED (arch mismatch with "
                  f"{args.init_from}?): {type(e).__name__}: {str(e)[:140]} "
                  f"-> training from scratch")
        if args.seed_pool:
            pool.add(PolicyOpponent.load(
                args.init_from, name="init", device=args.opponent_device))
            print("[selfplay] seeded pool with init checkpoint")

    snap_paths: list[Path] = []
    gate_log: list[dict] = []

    for rnd in range(args.num_rounds):
        if args.lr_schedule == "linear":
            # Decay across the WHOLE run, stepwise per round. We call learn() once
            # per round, so SB3's built-in progress_remaining resets to 1.0 each
            # round (would sawtooth lr); overriding lr_schedule here decays it
            # globally: round 0 -> --lr, last round -> ~--lr/num_rounds.
            round_lr = args.lr * (1.0 - rnd / args.num_rounds)
            model.lr_schedule = lambda _progress, _lr=round_lr: _lr
            print(f"[selfplay] round {rnd}: lr -> {round_lr:.2e}")
        model.learn(
            total_timesteps=args.steps_per_round,
            reset_num_timesteps=(rnd == 0),
            progress_bar=args.progress,
        )

        snap = save_dir / f"snap_{model.num_timesteps}.zip"
        model.save(snap)
        snap_paths.append(snap)
        pool.add(PolicyOpponent.load(
            snap, name=f"snap{rnd}", device=args.opponent_device))
        print(f"[selfplay] round {rnd}: saved {snap.name}, pool={len(pool)}")

        # Inline gate: latest vs N-rounds-ago.
        if len(snap_paths) > args.gate_lag:
            latest = pool.snapshots[-1]
            nago = pool.snapshots[-1 - args.gate_lag]
            latest_wins, decided, no_winner = play_2v2(
                latest, nago, args.gate_games, seed=args.seed + rnd,
                suppress_p2p=args.no_p2p_trade)
            r = gate_result(latest_wins, decided, no_winner, args.gate_threshold)
            entry = {
                "round": rnd, "timesteps": model.num_timesteps,
                "latest": snap_paths[-1].name,
                "nago": snap_paths[-1 - args.gate_lag].name, **r,
            }
            gate_log.append(entry)
            tag = "PASS" if r["pass"] else ("inconclusive" if not r["conclusive"] else "fail")
            print(f"[selfplay] gate r{rnd}: latest vs -{args.gate_lag} "
                  f"= {r['win_rate']:.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}] "
                  f"(no-winner {r['no_winner_rate']:.0%}) {tag}")

    final = save_dir / "selfplay_final.zip"
    model.save(final)

    last_gate = gate_log[-1] if gate_log else None
    summary = {
        "run_name": args.run_name,
        "config": {
            "lr": args.lr, "lr_schedule": args.lr_schedule,
            "ent_coef": args.ent_coef, "target_kl": args.target_kl,
            "steps_per_round": args.steps_per_round,
            "num_rounds": args.num_rounds, "num_envs": args.num_envs,
            "p_random": args.p_random, "pool_window": args.pool_window,
            "net_arch": args.net_arch,
            "gate_lag": args.gate_lag, "init_from": args.init_from,
        },
        "final_ckpt": str(final),
        "gate_log": gate_log,
        "final_gate_rate": last_gate["win_rate"] if last_gate else None,
        "final_gate_pass": last_gate["pass"] if last_gate else None,
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[selfplay] done -> {final}")
    print(f"[selfplay] summary -> {save_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
