"""Twin-ready MaskablePPO trainer for the Phase-2 boring baseline."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import signal
import subprocess
import time
from pathlib import Path

from sb3_contrib import MaskablePPO

from models.baseline.kl_ppo import (
    KLRegularizedMaskablePPO,
    ReferenceActor,
    warmstart_policy,
)
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from models.baseline.env import POOL_NAMES, Phase2CatanEnv
from models.baseline.evaluate import evaluate_checkpoint
from models.baseline.policy import Phase2Policy, load_il_weights, parameter_counts
from models.ckpt import write_stamp
from results.schema import append_rows


def _wandb_url(run) -> str:
    return getattr(run, "url", None) or ""


def _env_factory(seed: int, curriculum):
    def build():
        env = Phase2CatanEnv(seed=seed, curriculum=curriculum)
        env = ActionMasker(env, lambda wrapped: wrapped.action_masks())
        return Monitor(env)
    return build


def _save_model(model: MaskablePPO, path: Path) -> Path:
    model.save(str(path))
    actual = path if path.suffix == ".zip" else Path(str(path) + ".zip")
    write_stamp(actual)
    return actual


def _run_frozen_tournament(
    root: Path,
    checkpoint: Path,
    candidate: str,
    step: int,
    games: int,
) -> str:
    command = [
        str(root / "scripts" / "tournament.sh"),
        "--games", str(games),
        "--seed", str(step),
        "--newest", str(checkpoint.resolve()),
        "--ladder-version", "1.0-v1",
        "--candidate", candidate,
    ]
    proc = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
        check=True,
    )
    return proc.stdout


class BaselineCallback(BaseCallback):
    def __init__(
        self,
        *,
        run,
        save_dir: Path,
        run_name: str,
        first_eval: int,
        eval_every: int,
        eval_games: int,
        tournament_games: int,
        tournament_root: Path | None,
    ) -> None:
        super().__init__()
        self.run = run
        self.save_dir = save_dir
        self.run_name = run_name
        self.eval_every = int(eval_every)
        self.eval_games = int(eval_games)
        self.tournament_games = int(tournament_games)
        self.tournament_root = tournament_root
        self.next_eval = int(first_eval or eval_every)
        self.lineup_path = save_dir / "lineups.jsonl"
        self.lineup_file = None
        self.episodes = 0
        self.last_loss: float | None = None
        self.nan_detected = False

    def _on_training_start(self) -> None:
        self.lineup_file = self.lineup_path.open("a", encoding="utf-8")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "opponent_lineup" not in info or "episode" not in info:
                continue
            self.episodes += 1
            row = {
                "step": int(self.num_timesteps),
                "learner_seat": int(info["learner_seat"]),
                "opponents": json.loads(info["opponent_lineup"]),
                "pool_stage": int(info["pool_stage"]),
                "offer_cap": info["offer_cap"],
                "reward": float(info["episode"]["r"]),
                "length": int(info["episode"]["l"]),
                "no_winner": bool(info["no_winner"]),
                "mask_violations": int(info["mask_violations"]),
                "curriculum_changed": bool(info["curriculum_changed"]),
            }
            assert self.lineup_file is not None
            self.lineup_file.write(json.dumps(row, sort_keys=True) + "\n")
            self.lineup_file.flush()
            if row["curriculum_changed"]:
                self.run.log({
                    "curriculum/stage": row["pool_stage"],
                    "curriculum/offer_cap_lifted": 1,
                }, step=int(self.num_timesteps))

        if self.num_timesteps >= self.next_eval:
            self.run_checkpoint_eval(int(self.num_timesteps))
            if self.next_eval < self.eval_every:
                # One early launch-verification checkpoint, then the fixed
                # ~2M-decision cadence from the spec.
                self.next_eval = self.eval_every
            else:
                while self.next_eval <= self.num_timesteps:
                    self.next_eval += self.eval_every
        return not self.nan_detected

    def _on_rollout_end(self) -> None:
        values = self.model.logger.name_to_value
        metrics = {
            key: float(value)
            for key, value in values.items()
            if key.startswith(("rollout/", "train/", "time/"))
            and isinstance(value, (int, float))
        }
        metrics["phase2/episodes"] = self.episodes
        if "train/loss" in metrics:
            self.last_loss = metrics["train/loss"]
            if not math.isfinite(self.last_loss):
                self.nan_detected = True
        if metrics:
            self.run.log(metrics, step=int(self.num_timesteps))

    def run_checkpoint_eval(self, step: int) -> dict:
        checkpoint = _save_model(
            self.model, self.save_dir / f"ppo_{step}_steps.zip",
        )
        candidate = f"{self.run_name}@{step}"
        frozen_output = None
        if self.tournament_root is not None:
            frozen_output = _run_frozen_tournament(
                self.tournament_root,
                checkpoint,
                candidate,
                step,
                self.tournament_games,
            )
            (self.save_dir / f"tournament_{step}.log").write_text(
                frozen_output, encoding="utf-8",
            )
        summary = evaluate_checkpoint(
            checkpoint,
            candidate=candidate,
            results_dir=self.save_dir / "eval_results",
            games_per_opponent=self.eval_games,
            wandb_url=_wandb_url(self.run),
            step=step,
        )
        self.run.log(
            {f"eval/{key}": value for key, value in summary.items()},
            step=step,
        )
        (self.save_dir / f"eval_{step}.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[eval] step={step} rows={summary['rows_appended']} "
            f"promotion_mean={summary['promotion_mean']:.4f} "
            f"wall={summary['wall_seconds']:.1f}s",
            flush=True,
        )
        return summary

    def _on_training_end(self) -> None:
        if self.lineup_file is not None:
            self.lineup_file.close()


def _record_lifecycle(
    results_dir: Path,
    *,
    run_name: str,
    status: str,
    steps: int,
    wandb_url: str,
    param_count: int,
) -> None:
    common = {
        "ladder_version": "1.0-v1",
        "candidate": run_name,
        "opponent": "phase2-lifecycle",
        "rotation": "full",
        "games": 0,
        "wins": 0,
        "win_rate": 0.0,
        "wilson_low": 0.0,
        "wilson_high": 0.0,
        "no_winner_rate": 0.0,
        "seat_wins": [0, 0, 0, 0],
        "trading_delta": None,
        "decisions_per_s": 0.0,
        "param_count": param_count,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip(),
        "config_hash": run_name,
        "wandb_url": wandb_url,
        "verdict": status,
        "notes": f"learner_decisions={steps}",
    }
    append_rows(
        [common | {"mode": mode} for mode in ("trades_on", "trades_off")],
        results_dir=results_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--save-root", default="models/checkpoints/phase2_sprints")
    parser.add_argument("--total-steps", type=int, default=50_000_000)
    parser.add_argument("--num-envs", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1536)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-from-il", default="")
    parser.add_argument("--kl-ref", default="")
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--kl-coef-final", type=float, default=None)
    parser.add_argument("--kl-form", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--eval-every", type=int, default=2_000_000)
    parser.add_argument("--first-eval", type=int, default=100_000)
    parser.add_argument("--eval-games", type=int, default=32)
    parser.add_argument("--tournament-games", type=int, default=16)
    parser.add_argument("--tournament-root", default="")
    parser.add_argument("--wandb-group", default="phase2-baseline-twins-20260718")
    parser.add_argument("--no-initial-eval", action="store_true")
    parser.add_argument("--dummy-vec", action="store_true")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()

    if args.wandb_mode != "online":
        os.environ["WANDB_MODE"] = args.wandb_mode
    save_dir = Path(args.save_root) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    manager = mp.Manager()
    curriculum = manager.Value("i", 0)
    factories = [
        _env_factory(args.seed + rank * 1009, curriculum)
        for rank in range(args.num_envs)
    ]
    if args.dummy_vec or args.num_envs == 1:
        env = DummyVecEnv(factories)
    else:
        env = SubprocVecEnv(factories, start_method="spawn")

    model = KLRegularizedMaskablePPO(
        Phase2Policy,
        env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        seed=args.seed,
        device=args.device,
        # W&B is logged directly by BaselineCallback; tensorboard is not a
        # dependency of the pinned Box environment.
        tensorboard_log=None,
        verbose=1,
    )
    il_state = None
    if args.init_from_il:
        il_state = warmstart_policy(model.policy, args.init_from_il)
    model.kl_coef0 = float(args.kl_coef)
    model.kl_coef_final = float(
        args.kl_coef if args.kl_coef_final is None else args.kl_coef_final
    )
    model.kl_form = args.kl_form
    model.kl_ref = None
    if args.kl_ref:
        model.kl_ref = ReferenceActor.from_checkpoint(
            args.kl_ref, model.device, int(model.action_space.n)
        )
    counts = parameter_counts(model.policy)
    if counts["total"] > 8_000_000:
        raise ValueError(f"parameter budget exceeded: {counts}")

    config = vars(args) | {
        "actor_params": counts["actor"],
        "critic_params": counts["critic"],
        "total_params": counts["total"],
        "il_val_top1": None if il_state is None else il_state.get("val_top1"),
        "reward": "terminal:+1_win,-1_loss,-1_no_winner",
        "pool_initial_weights": [0.20, 0.20, 0.20, 0.08, 0.07, 0.05, 0.10, 0.10],
        "pool_names": list(POOL_NAMES),
        "offer_cap_initial": 2,
    }
    import wandb
    run = wandb.init(
        project="goodsettler-rl",
        name=args.run_name,
        group=args.wandb_group,
        job_type="phase2-ppo",
        config=config,
        reinit=True,
        sync_tensorboard=False,
    )
    run_url = _wandb_url(run)
    print(f"[wandb] {run_url or 'disabled'}", flush=True)
    (save_dir / "wandb_url.txt").write_text(run_url + "\n", encoding="utf-8")

    callback = BaselineCallback(
        run=run,
        save_dir=save_dir,
        run_name=args.run_name,
        first_eval=args.first_eval,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        tournament_games=args.tournament_games,
        tournament_root=Path(args.tournament_root) if args.tournament_root else None,
    )
    status = "completed_budget"
    interrupted = False
    started = time.time()
    try:
        if not args.no_initial_eval:
            initial = _save_model(model, save_dir / "ppo_0_steps.zip")
            if callback.tournament_root is not None:
                output = _run_frozen_tournament(
                    callback.tournament_root,
                    initial,
                    f"{args.run_name}@0",
                    0,
                    args.tournament_games,
                )
                (save_dir / "tournament_0.log").write_text(output, encoding="utf-8")
            summary = evaluate_checkpoint(
                initial,
                candidate=f"{args.run_name}@0",
                results_dir=save_dir / "eval_results",
                games_per_opponent=args.eval_games,
                wandb_url=run_url,
                step=0,
            )
            run.log(
                {f"eval/{key}": value for key, value in summary.items()}, step=0,
            )
            (save_dir / "eval_0.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"[eval] initial rows={summary['rows_appended']}", flush=True)
        model.learn(
            total_timesteps=args.total_steps,
            callback=callback,
            progress_bar=False,
        )
        if callback.nan_detected:
            status = "killed_nan"
    except KeyboardInterrupt:
        interrupted = True
        status = "killed_wallclock_or_interrupt"
        print("[train] interrupt received; saving resumable checkpoint", flush=True)
    finally:
        final = _save_model(model, save_dir / "ppo_final.zip")
        _record_lifecycle(
            save_dir / "eval_results",
            run_name=args.run_name,
            status=status,
            steps=int(model.num_timesteps),
            wandb_url=run_url,
            param_count=counts["total"],
        )
        run.summary["status"] = status
        run.summary["final_steps"] = int(model.num_timesteps)
        run.summary["wall_seconds"] = time.time() - started
        run.summary["final_checkpoint"] = str(final)
        run.summary["last_loss"] = callback.last_loss
        run.finish()
        env.close()
        manager.shutdown()
    if status == "killed_nan":
        raise SystemExit(3)
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
