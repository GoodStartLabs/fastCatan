"""CLI for PPO training or a single-count throughput smoke."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
from dataclasses import asdict
from pathlib import Path

import torch

from models.alphazero.batched_eval import eval_vs_random_raw
from models.batched_trainer.trainer import BatchedTrainer, TrainerConfig


def _hidden(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def _opponents(value: str) -> tuple[str, ...]:
    specs = tuple(part.strip() for part in value.split(",") if part.strip())
    return specs * 3 if len(specs) == 1 else specs


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=4000)
    p.add_argument("--env-workers", type=int, default=0,
                   help="shared-memory BatchedEnv processes; 0 keeps the "
                        "single-process compatibility path")
    p.add_argument("--split-env-affinity", action="store_true",
                   help="reserve the highest offered CPU for the learner and "
                        "restrict env workers to the remaining CPUs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--hidden", default="512,512,256")
    p.add_argument("--opponents", default="random",
                   help="comma-separated vectorized pool: random, self, ab1, "
                        "ab2, or checkpoint:/path (one spec repeats x3)")
    p.add_argument("--init-from", default="")
    p.add_argument("--resume", default="",
                   help="checkpoint to resume net+optimizer+decisions+rng from")
    p.add_argument("--duration-seconds", type=float, default=1800.0)
    p.add_argument("--max-learner-decisions", type=int, default=0)
    p.add_argument("--rollout-decisions", type=int, default=262144)
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--update-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--anchor-ref", default="")
    p.add_argument("--anchor-coef", type=float, default=0.0)
    p.add_argument("--anchor-coef-final", type=float, default=0.0)
    p.add_argument("--total-learner-decisions", type=int, default=100_000_000)
    p.add_argument("--win-rate-window-episodes", type=int, default=10_000)
    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--assert-floor", type=float, default=0.0)
    p.add_argument("--entropy-min", type=float, default=0.01)
    p.add_argument("--entropy-max", type=float, default=5.66)
    p.add_argument("--promotion-games", type=int, default=0)
    p.add_argument("--assert-promotion-moves", action="store_true")
    p.add_argument("--checkpoint-out", default="")
    p.add_argument("--json-out", default="")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                   default="disabled")
    p.add_argument("--wandb-entity", default="good-start-labs")
    p.add_argument("--wandb-project", default="goodsettler-rl")
    p.add_argument("--wandb-name", default="batched-ppo-smoke")
    p.add_argument("--wandb-group", default="")
    p.add_argument("--wandb-notes", default="")
    p.add_argument("--tripwire-entropy-max", type=float, default=0.0)
    p.add_argument("--tripwire-min-throughput", type=float, default=0.0)
    p.add_argument("--tripwire-low-logs", type=int, default=6)
    p.add_argument("--log-interval", type=float, default=10.0)
    return p


def main() -> None:
    args = parser().parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("spec 3.2 requires PYTHONHASHSEED=0")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)

    cfg = TrainerConfig(
        num_envs=args.num_envs,
        env_workers=args.env_workers,
        split_env_affinity=args.split_env_affinity,
        seed=args.seed,
        device=args.device,
        hidden=_hidden(args.hidden),
        opponents=_opponents(args.opponents),  # type: ignore[arg-type]
        init_from=args.init_from,
        resume=args.resume,
        rollout_decisions=args.rollout_decisions,
        batch_size=args.batch_size,
        update_epochs=args.update_epochs,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        amp=not args.no_amp,
        anchor_ref=args.anchor_ref,
        anchor_coef=args.anchor_coef,
        anchor_coef_final=args.anchor_coef_final,
        total_learner_decisions=args.total_learner_decisions,
        win_rate_window_episodes=args.win_rate_window_episodes,
    )
    trainer = BatchedTrainer(cfg)

    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name,
            group=args.wandb_group or None,
            notes=args.wandb_notes or None,
            mode=args.wandb_mode,
            config=asdict(cfg),
            tags=[
                "batched-trainer",
                "spec-3.2b" if args.env_workers else "spec-3.2",
            ],
        )
        run.define_metric("learner_decisions")
        run.define_metric("rollout/*", step_metric="learner_decisions")
        run.define_metric("train/*", step_metric="learner_decisions")
        run.define_metric("throughput/*", step_metric="learner_decisions")

    promotion_before = None
    if args.promotion_games:
        promotion_before, before_no_winner = eval_vs_random_raw(
            trainer.net, args.device, games=args.promotion_games,
            seed=args.seed ^ 0xE7A1,
        )
        if run is not None:
            run.log({"promotion/win_rate": promotion_before,
                     "promotion/no_winner": before_no_winner,
                     "learner_decisions": trainer.learner_decisions})

    low_throughput_logs = 0
    high_entropy_logs = 0

    def log_callback(values):
        nonlocal low_throughput_logs, high_entropy_logs
        print(json.dumps({"progress": values}, sort_keys=True), flush=True)
        if run is not None:
            payload = {
                "learner_decisions": values["learner_decisions"],
                "throughput/learner_decisions_per_s": values["learner_decisions_per_s"],
                "throughput/total_decisions_per_s": values["total_decisions_per_s"],
                "rollout/entropy": values["entropy"],
                "rollout/episodes": values["episodes"],
                "rollout/terminal_episodes": values["terminal_episodes"],
                "rollout/win_rate": values["win_rate"],
                "rollout/mean_reward": values["mean_reward"],
                "rollout/rolling_win_rate": values["rolling_win_rate"],
                "rollout/rolling_reward": values["rolling_reward"],
                "rollout/rolling_episodes": values["rolling_episodes"],
                "gpu/memory_allocated_mb": values.get("gpu_mem_allocated_mb", 0.0),
                "gpu/util_percent": values.get("gpu_util_percent_sample", 0.0),
                "gpu/memory_used_mb": values.get("gpu_memory_used_mb_sample", 0.0),
            }
            for key, value in values["time_ms_per_step"].items():
                payload[f"time_ms_per_step/{key}"] = value
            for key, value in values.get("last_update", {}).items():
                payload[f"train/{key}"] = value
            run.log(payload)

        tripwire = None
        monitored = [values["entropy"], values["learner_decisions_per_s"]]
        monitored.extend(values.get("last_update", {}).values())
        if not all(math.isfinite(float(value)) for value in monitored):
            tripwire = "non-finite training metric"
        if (args.tripwire_entropy_max
                and values["entropy"] > args.tripwire_entropy_max):
            high_entropy_logs += 1
        else:
            high_entropy_logs = 0
        if not tripwire and high_entropy_logs >= 2:
            tripwire = (
                f"entropy {values['entropy']:.6f} > "
                f"{args.tripwire_entropy_max:.6f} for "
                f"{high_entropy_logs} consecutive logs"
            )
        if (args.tripwire_min_throughput
                and values["learner_decisions_per_s"]
                < args.tripwire_min_throughput):
            low_throughput_logs += 1
        else:
            low_throughput_logs = 0
        if (not tripwire and args.tripwire_min_throughput
                and low_throughput_logs >= args.tripwire_low_logs):
            tripwire = (
                f"learner throughput below {args.tripwire_min_throughput:.0f} "
                f"decisions/s for {low_throughput_logs} consecutive logs"
            )
        if tripwire:
            print(f"TRIPWIRE={tripwire}", flush=True)
            if run is not None:
                run.summary["tripwire"] = tripwire
                run.log({"tripwire/triggered": 1})
            os.kill(os.getpid(), signal.SIGINT)

    summary = trainer.run(
        duration_seconds=args.duration_seconds,
        max_learner_decisions=args.max_learner_decisions,
        benchmark_only=args.benchmark_only,
        warmup_steps=args.warmup_steps,
        log_interval_seconds=args.log_interval,
        log_callback=log_callback,
    )

    promotion_after = None
    if args.promotion_games:
        promotion_after, after_no_winner = eval_vs_random_raw(
            trainer.net, args.device, games=args.promotion_games,
            seed=args.seed ^ 0xE7A1,
        )
        summary["promotion"] = {
            "before": promotion_before,
            "after": promotion_after,
            "delta": promotion_after - promotion_before,
            "no_winner_after": after_no_winner,
        }
        if run is not None:
            run.log({"promotion/win_rate": promotion_after,
                     "promotion/no_winner": after_no_winner,
                     "promotion/delta": promotion_after - promotion_before,
                     "learner_decisions": summary["learner_decisions"]})

    finite_metrics = summary.get("last_update", {})
    summary["finite_losses"] = bool(finite_metrics) and all(
        math.isfinite(float(value)) for value in finite_metrics.values()
    )
    summary["entropy_sane"] = (
        args.entropy_min <= summary["entropy"] <= args.entropy_max
    )
    summary["floor_pass"] = (
        not args.assert_floor
        or summary["learner_decisions_per_s"] >= args.assert_floor
    )
    if args.checkpoint_out and not args.benchmark_only:
        trainer.save(args.checkpoint_out, extra={"summary": summary})
    if args.json_out:
        trainer.write_summary(args.json_out, summary)
    if run is not None:
        run.summary.update(summary)
        summary["wandb_url"] = run.url
        run.finish()
    trainer.close()

    print("BATCHED_TRAINER_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
    if args.assert_floor and not summary["floor_pass"]:
        raise SystemExit(
            f"throughput floor failed: {summary['learner_decisions_per_s']:.0f} "
            f"< {args.assert_floor:.0f} learner decisions/s"
        )
    if not args.benchmark_only:
        if not summary["finite_losses"]:
            raise SystemExit("finite-loss gate failed (no update or non-finite metric)")
        if not summary["entropy_sane"]:
            raise SystemExit(
                f"entropy gate failed: {summary['entropy']:.4f} not in "
                f"[{args.entropy_min}, {args.entropy_max}]"
            )
        if (args.assert_promotion_moves and promotion_after is not None
                and promotion_after == promotion_before):
            raise SystemExit("promotion gate failed: evaluation did not move")


if __name__ == "__main__":
    main()
