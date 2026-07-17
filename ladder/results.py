"""Append-only parquet results, Markdown summaries, W&B, and git publishing."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .config import LADDER_VERSION, SCHEMA_VERSION, WANDB_PROJECT
from .stats import aggregate, promotion_metric, seat_rates


RESULT_COLUMNS = [
    "schema_version", "ladder_version", "run_id", "timestamp_utc", "tier",
    "candidate", "candidate_band", "opponent", "opponent_band",
    "promotion_eligible", "anchor", "incumbent", "mode", "rotation_block",
    "master_seed", "board_seed", "games", "candidate_wins", "opponent_wins",
    "no_winner", "no_winner_policy", "candidate_seat0_games",
    "candidate_seat0_wins", "candidate_seat1_games", "candidate_seat1_wins",
    "candidate_seat2_games", "candidate_seat2_wins", "candidate_seat3_games",
    "candidate_seat3_wins", "winner_seats_json", "decisions", "wall_seconds",
    "decisions_per_second", "git_commit", "hostname", "python_version",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
        check=True, capture_output=True,
    )
    return proc.stdout.strip()


def block_row(
    block,
    *,
    repo_root: Path,
    run_id: str,
    timestamp_utc: str,
    tier: str,
    candidate: str,
    candidate_band: str,
    opponent: str,
    opponent_band: str,
    promotion_eligible: bool,
    anchor: bool,
    incumbent: bool,
    mode: str,
    master_seed: int,
) -> dict:
    row = {
        "schema_version": SCHEMA_VERSION,
        "ladder_version": LADDER_VERSION,
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "tier": tier,
        "candidate": candidate,
        "candidate_band": candidate_band,
        "opponent": opponent,
        "opponent_band": opponent_band,
        "promotion_eligible": bool(promotion_eligible),
        "anchor": bool(anchor),
        "incumbent": bool(incumbent),
        "mode": mode,
        "rotation_block": int(block.block_index),
        "master_seed": f"0x{master_seed:016x}",
        "board_seed": f"0x{block.board_seed:016x}",
        "games": int(block.games),
        "candidate_wins": int(block.candidate_wins),
        "opponent_wins": int(block.opponent_wins),
        "no_winner": int(block.no_winner),
        "no_winner_policy": block.no_winner_policy,
        "winner_seats_json": json.dumps(list(block.winner_seats)),
        "decisions": int(block.decisions),
        "wall_seconds": float(block.wall_seconds),
        "decisions_per_second": float(block.decisions_per_second),
        "git_commit": git_commit(repo_root),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
    }
    for seat in range(4):
        row[f"candidate_seat{seat}_games"] = int(block.candidate_seat_games[seat])
        row[f"candidate_seat{seat}_wins"] = int(block.candidate_seat_wins[seat])
    return {column: row[column] for column in RESULT_COLUMNS}


def append_parquet(rows: Sequence[Mapping], path: Path) -> None:
    """Append rows by atomically replacing one schema-stable parquet file."""
    if not rows:
        return
    incoming = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if path.exists():
        prior = pd.read_parquet(path)
        if list(prior.columns) != RESULT_COLUMNS:
            raise ValueError(f"existing result schema does not match {path}")
        incoming = pd.concat([prior, incoming], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    incoming.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def _fmt_rate(value) -> str:
    return f"{value.rate:.3f} [{value.ci95_low:.3f}, {value.ci95_high:.3f}]"


def write_markdown_summary(rows: Sequence[Mapping], path: Path) -> None:
    """Append a compact, human-readable section for one completed run."""
    if not rows:
        return
    run_id = str(rows[0]["run_id"])
    candidate = str(rows[0]["candidate"])
    by_opp_mode = aggregate(rows, "opponent", "mode")
    promotion = promotion_metric(rows)
    seats = seat_rates(rows)
    worst_seat, worst_seat_rate = min(seats.items(), key=lambda item: item[1].rate)
    total_decisions = sum(int(row["decisions"]) for row in rows)
    total_seconds = sum(float(row["wall_seconds"]) for row in rows)
    total_no_winner = sum(int(row["no_winner"]) for row in rows)

    lines = [
        f"\n## {run_id} — {candidate}\n",
        "| Opponent | Mode | Win share (95% Wilson CI) | No winner |\n",
        "|---|---|---:|---:|\n",
    ]
    for (opponent, mode), value in sorted(by_opp_mode.items()):
        lines.append(
            f"| {opponent} | {mode} | {_fmt_rate(value)} | "
            f"{value.no_winner}/{value.games} |\n"
        )
    lines.extend([
        "\n| Opponent | Trading-on | Trading-off | Delta (on - off) |\n",
        "|---|---:|---:|---:|\n",
    ])
    opponents = sorted({opponent for opponent, _mode in by_opp_mode})
    for opponent in opponents:
        on = by_opp_mode[(opponent, "trading_on")]
        off = by_opp_mode[(opponent, "trading_off")]
        lines.append(
            f"| {opponent} | {on.rate:.3f} | {off.rate:.3f} | "
            f"{on.rate - off.rate:+.3f} |\n"
        )
    lines.extend([
        "\n",
        f"Promotion metric (trading-on legal-info pool): "
        f"**{promotion.ci95_low:.4f} lower Wilson bound** "
        f"({promotion.wins}/{promotion.games}, mean={promotion.rate:.4f}).  ",
        f"No-winner/truncation: {total_no_winner}; scored as a loss for every seat.  ",
        f"Worst seat: {worst_seat} at {worst_seat_rate.rate:.4f}.  ",
        f"Throughput: {total_decisions / total_seconds if total_seconds else 0.0:,.0f} "
        f"decisions/s over {total_seconds:.2f}s.\n",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Ladder results\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("".join(lines))


def mirror_wandb(rows: Sequence[Mapping], *, project: str = WANDB_PROJECT) -> str | None:
    """Mirror a completed run to W&B when credentials are available."""
    if not rows or os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("wandb is required unless WANDB_MODE=disabled") from exc

    run_id = str(rows[0]["run_id"])
    run = wandb.init(
        project=project,
        name=run_id,
        job_type="ladder",
        config={
            "ladder_version": rows[0]["ladder_version"],
            "tier": rows[0]["tier"],
            "candidate": rows[0]["candidate"],
            "master_seed": rows[0]["master_seed"],
            "no_winner_policy": rows[0]["no_winner_policy"],
        },
    )
    table = wandb.Table(columns=RESULT_COLUMNS)
    for row in rows:
        table.add_data(*(row[column] for column in RESULT_COLUMNS))
    metric = promotion_metric(rows)
    run.log({
        "ladder/blocks": table,
        "ladder/promotion_mean": metric.rate,
        "ladder/promotion_ci95_low": metric.ci95_low,
        "ladder/no_winner": sum(int(row["no_winner"]) for row in rows),
        "ladder/decisions_per_second": (
            sum(int(row["decisions"]) for row in rows)
            / sum(float(row["wall_seconds"]) for row in rows)
        ),
    })
    url = run.url
    run.finish()
    return url


def publish_results(repo_root: Path, paths: Iterable[Path], run_id: str) -> None:
    """Commit only result artifacts and push the current non-default branch."""
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo_root, text=True,
        check=True, capture_output=True,
    ).stdout.strip()
    if branch in {"main", "master", "phase0-audit", ""}:
        raise RuntimeError(f"refusing to publish ladder results from {branch!r}")
    relative = [str(path.resolve().relative_to(repo_root.resolve())) for path in paths]
    subprocess.run(["git", "add", "--", *relative], cwd=repo_root, check=True)
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=repo_root,
    ).returncode != 0
    if not changed:
        return
    subprocess.run(
        ["git", "commit", "-m", f"ladder results: {run_id}"],
        cwd=repo_root, check=True,
    )
    subprocess.run(
        ["git", "push", "-u", "origin", branch], cwd=repo_root, check=True,
    )
