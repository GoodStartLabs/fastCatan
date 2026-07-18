"""Paired-board, both-mode round-robin calibration for the frozen roster."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd

import fastcatan

from .config import (
    FULL_GAMES_PER_OPPONENT_MODE,
    LADDER_VERSION,
    MASTER_SEED,
    SMOKE_GAMES_PER_OPPONENT_MODE,
)
from .match import play_rotation_block
from .registry import REGISTRY, build_agent
from .results import (
    append_parquet,
    block_row,
    git_commit,
    mirror_wandb,
    publish_results,
    utc_now,
    write_markdown_summary,
)
from .stats import aggregate


@dataclass(frozen=True)
class Task:
    candidate: str
    opponent: str
    master_seed: int
    block_index: int
    suppress_p2p: bool
    max_steps: int


def _execute(task: Task):
    # Avoid nested BLAS oversubscription when many MCTS blocks run in parallel.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    return play_rotation_block(
        env=fastcatan.Env(),
        candidate_name=task.candidate,
        opponent_name=task.opponent,
        candidate_factory=lambda seed: build_agent(task.candidate, seed),
        opponent_factory=lambda seed: build_agent(task.opponent, seed),
        master_seed=task.master_seed,
        block_index=task.block_index,
        suppress_p2p=task.suppress_p2p,
        max_steps=task.max_steps,
    )


def _agent_names(tier: str, requested: list[str] | None) -> list[str]:
    if requested:
        unknown = sorted(set(requested) - set(REGISTRY))
        if unknown:
            raise ValueError(f"unknown calibration agents: {unknown}")
        return requested
    if tier == "smoke":
        return [spec.name for spec in REGISTRY.values() if spec.smoke]
    return list(REGISTRY)


def _matrix_and_ranking(rows: list[dict], agents: list[str]) -> dict:
    grouped = aggregate(rows, "candidate", "opponent", "mode")
    matrices: dict[str, dict[str, dict[str, float | None]]] = {}
    rankings: dict[str, list[dict]] = {}
    for mode in ("trading_on", "trading_off"):
        matrix: dict[str, dict[str, float | None]] = {}
        for candidate in agents:
            matrix[candidate] = {}
            for opponent in agents:
                value = grouped.get((candidate, opponent, mode))
                matrix[candidate][opponent] = None if value is None else value.rate
        matrices[mode] = matrix
        rating_rows = []
        for candidate in agents:
            values = [
                value for opponent, value in matrix[candidate].items()
                if opponent != candidate and value is not None
            ]
            rating_rows.append({
                "agent": candidate,
                "mean_win_share": sum(values) / len(values) if values else 0.0,
                "opponents": len(values),
            })
        rankings[mode] = sorted(
            rating_rows,
            key=lambda item: (-item["mean_win_share"], item["agent"]),
        )
    return {"matrices": matrices, "rankings": rankings}


def _inversions(summary: dict, agents: list[str]) -> list[str]:
    on_ranking = {
        row["agent"]: row["mean_win_share"]
        for row in summary["rankings"]["trading_on"]
    }
    inversions: list[str] = []
    expected_pairs = (
        ("random-legal", "weighted-random"),
        ("builder-basic", "builder-strong"),
        ("oracle-mcts-abvalue-256", "oracle-mcts-abvalue-1024"),
    )
    for weaker, stronger in expected_pairs:
        if weaker in on_ranking and stronger in on_ranking and on_ranking[stronger] < on_ranking[weaker]:
            inversions.append(
                f"{stronger} ({on_ranking[stronger]:.4f}) below "
                f"{weaker} ({on_ranking[weaker]:.4f})"
            )
    oracle_names = [name for name in agents if REGISTRY[name].band == "oracle"]
    non_oracles = [name for name in agents if REGISTRY[name].band != "oracle"]
    if oracle_names and non_oracles:
        oracle_floor = min(on_ranking[name] for name in oracle_names)
        non_oracle_ceiling = max(on_ranking[name] for name in non_oracles)
        if oracle_floor < non_oracle_ceiling:
            inversions.append(
                f"oracle floor {oracle_floor:.4f} below non-oracle ceiling "
                f"{non_oracle_ceiling:.4f}"
            )
    return inversions


def _write_calibration_markdown(result: dict, path: Path) -> None:
    agents = result["agents"]
    lines = [
        f"# {LADDER_VERSION} {result['tier']} calibration\n\n",
        f"Run `{result['run_id']}`; {result['games_per_opponent_mode']} games per "
        f"directed opponent/mode; wall {result['wall_seconds']:.1f}s; "
        f"{result['decisions_per_second']:.0f} decisions/s.\n\n",
    ]
    for mode in ("trading_on", "trading_off"):
        lines.extend([
            f"## {mode}\n\n",
            "| Candidate | " + " | ".join(agents) + " |\n",
            "|---|" + "---:|" * len(agents) + "\n",
        ])
        matrix = result["matrices"][mode]
        for candidate in agents:
            values = [
                "—" if matrix[candidate][opponent] is None
                else f"{matrix[candidate][opponent]:.3f}"
                for opponent in agents
            ]
            lines.append(f"| {candidate} | " + " | ".join(values) + " |\n")
        lines.extend(["\n| Rank | Agent | Mean win share |\n", "|---:|---|---:|\n"])
        for rank, row in enumerate(result["rankings"][mode], 1):
            lines.append(f"| {rank} | {row['agent']} | {row['mean_win_share']:.4f} |\n")
        lines.append("\n")
    lines.append("## Sanity verdict\n\n")
    if result["inversions"]:
        lines.extend(f"- {item}\n" for item in result["inversions"])
    else:
        lines.append("No configured tier inversion detected.\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    tier = p.add_mutually_exclusive_group(required=True)
    tier.add_argument("--smoke", action="store_true")
    tier.add_argument("--full", action="store_true")
    p.add_argument("--games", type=int)
    p.add_argument("--agents", nargs="+")
    p.add_argument("--master-seed", type=lambda value: int(value, 0), default=MASTER_SEED)
    p.add_argument("--max-steps", type=int, default=150_000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--run-id")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-publish", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    tier = "smoke" if args.smoke else "full"
    games = args.games or (
        SMOKE_GAMES_PER_OPPONENT_MODE if args.smoke
        else FULL_GAMES_PER_OPPONENT_MODE
    )
    if games <= 0 or games % 4:
        raise SystemExit("--games must be a positive multiple of four")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    agents = _agent_names(tier, args.agents)
    if len(agents) < 2:
        raise SystemExit("calibration requires at least two agents")

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = (repo_root / args.results_dir).resolve()
    parquet_path = results_dir / "ladder.parquet"
    markdown_path = results_dir / "ladder.md"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{LADDER_VERSION}-roundrobin-{tier}-{stamp}"
    timestamp = utc_now()
    commit = git_commit(repo_root)
    completed: set[str] = set()
    if args.resume and parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        completed = set(existing.loc[existing.run_id == run_id, "candidate"].unique())
        print(f"[{run_id}] resume: completed={sorted(completed)}", flush=True)

    started = perf_counter()
    for candidate in agents:
        if candidate in completed:
            continue
        opponent_names = [name for name in agents if name != candidate]
        tasks = [
            Task(candidate, opponent, args.master_seed, block, suppress, args.max_steps)
            for opponent in opponent_names
            for suppress in (False, True)
            for block in range(games // 4)
        ]
        print(
            f"[{run_id}] candidate={candidate} tasks={len(tasks)} workers={args.workers}",
            flush=True,
        )
        if args.workers == 1:
            blocks = [_execute(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                blocks = list(executor.map(_execute, tasks, chunksize=1))
        rows: list[dict] = []
        for task, block in zip(tasks, blocks, strict=True):
            opponent = REGISTRY[task.opponent]
            rows.append(block_row(
                block,
                repo_root=repo_root,
                run_id=run_id,
                timestamp_utc=timestamp,
                tier=tier,
                candidate=candidate,
                candidate_band=REGISTRY[candidate].band,
                opponent=opponent.name,
                opponent_band=opponent.band,
                promotion_eligible=opponent.promotion_eligible,
                anchor=opponent.anchor,
                incumbent=opponent.incumbent,
                mode="trading_off" if task.suppress_p2p else "trading_on",
                master_seed=args.master_seed,
                commit=commit,
            ))
        append_parquet(rows, parquet_path)
        write_markdown_summary(rows, markdown_path)
        if not args.no_wandb:
            mirror_wandb(rows)
        print(
            f"[{run_id}] candidate={candidate} complete "
            f"games={sum(row['games'] for row in rows)}",
            flush=True,
        )

    elapsed = perf_counter() - started
    frame = pd.read_parquet(parquet_path)
    selected = frame.loc[frame.run_id == run_id]
    all_rows = selected.to_dict(orient="records")
    expected_rows = len(agents) * (len(agents) - 1) * 2 * (games // 4)
    if len(all_rows) != expected_rows:
        raise RuntimeError(
            f"calibration incomplete: rows={len(all_rows)} expected={expected_rows}"
        )
    summary = _matrix_and_ranking(all_rows, agents)
    decisions = sum(int(row["decisions"]) for row in all_rows)
    no_winner = sum(int(row["no_winner"]) for row in all_rows)
    result = {
        "ladder_version": LADDER_VERSION,
        "run_id": run_id,
        "tier": tier,
        "agents": agents,
        "games_per_opponent_mode": games,
        "master_seed": f"0x{args.master_seed:016x}",
        "no_winner": no_winner,
        "no_winner_policy": "loss_for_all_seats",
        "wall_seconds": elapsed,
        "decisions": decisions,
        "decisions_per_second": decisions / elapsed if elapsed else 0.0,
        **summary,
    }
    result["inversions"] = _inversions(result, agents)
    json_path = results_dir / f"calibration_{LADDER_VERSION}_{tier}_{run_id}.json"
    calibration_md = results_dir / f"calibration_{LADDER_VERSION}_{tier}_{run_id}.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_calibration_markdown(result, calibration_md)
    print(
        f"[{run_id}] complete wall={elapsed:.1f}s dps={result['decisions_per_second']:.0f} "
        f"no_winner={no_winner} inversions={len(result['inversions'])}",
        flush=True,
    )
    if not args.no_publish:
        publish_results(
            repo_root,
            [parquet_path, markdown_path, json_path, calibration_md],
            run_id,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
