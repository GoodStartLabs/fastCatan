"""Measure the frozen BatchedEnv + GPU actor at several environment counts."""
from __future__ import annotations

import argparse
import json
import os

import torch

from models.batched_trainer.trainer import BatchedTrainer, TrainerConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-counts", default="1000,2000,4000")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--floor", type=float, default=100000.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", default="512,512,256")
    p.add_argument("--opponents", default="random")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--json-out", default="")
    args = p.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("spec 3.2 requires PYTHONHASHSEED=0")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    counts = [int(value) for value in args.env_counts.split(",")]
    hidden = tuple(int(value) for value in args.hidden.split(","))
    opponent_values = tuple(x.strip() for x in args.opponents.split(",") if x.strip())
    if len(opponent_values) == 1:
        opponent_values *= 3
    results = []
    for i, count in enumerate(counts):
        trainer = BatchedTrainer(TrainerConfig(
            num_envs=count,
            seed=args.seed + i * 1009,
            device=args.device,
            hidden=hidden,
            opponents=opponent_values,  # type: ignore[arg-type]
        ))
        result = trainer.run(
            duration_seconds=args.seconds,
            benchmark_only=True,
            warmup_steps=args.warmup_steps,
        )
        result["floor"] = args.floor
        result["floor_pass"] = result["learner_decisions_per_s"] >= args.floor
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    payload = {"results": results, "all_pass": all(r["floor_pass"] for r in results)}
    if args.json_out:
        BatchedTrainer.write_summary(args.json_out, payload)
    print("BATCHED_BENCHMARK_RESULT=" + json.dumps(payload, sort_keys=True))
    if not payload["all_pass"]:
        raise SystemExit("one or more environment counts missed the throughput floor")


if __name__ == "__main__":
    main()
