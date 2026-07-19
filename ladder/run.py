"""CLI entry point for a versioned candidate ladder run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    FULL_GAMES_PER_OPPONENT_MODE,
    LADDER_VERSION,
    MASTER_SEED,
    SMOKE_GAMES_PER_OPPONENT_MODE,
)
from .match import run_matchup
from .results import (
    append_parquet,
    block_row,
    mirror_wandb,
    publish_results,
    utc_now,
    write_markdown_summary,
)
from .stats import aggregate, promotion_metric, seat_rates


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True, help="registered roster/candidate name")
    tier = p.add_mutually_exclusive_group(required=True)
    tier.add_argument("--smoke", action="store_true", help="256 games/opponent-mode")
    tier.add_argument("--full", action="store_true", help="1024 games/opponent-mode")
    p.add_argument("--games", type=int, help="override games/opponent-mode; multiple of 4")
    p.add_argument("--master-seed", type=lambda value: int(value, 0), default=MASTER_SEED)
    p.add_argument("--max-steps", type=int, default=150_000)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-publish", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    # Imported lazily so the Part-A harness remains independently testable.
    from .registry import build_agent, candidate_spec, opponent_specs

    repo_root = Path(__file__).resolve().parents[1]
    tier = "smoke" if args.smoke else "full"
    games = args.games or (
        SMOKE_GAMES_PER_OPPONENT_MODE if args.smoke
        else FULL_GAMES_PER_OPPONENT_MODE
    )
    if games <= 0 or games % 4:
        raise SystemExit("--games must be a positive multiple of four")

    candidate = candidate_spec(args.candidate)
    opponents = opponent_specs(tier=tier, exclude=args.candidate)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{LADDER_VERSION}-{args.candidate}-{tier}-{stamp}"
    timestamp = utc_now()
    rows: list[dict] = []

    for opponent in opponents:
        for suppress_p2p, mode in ((False, "trading_on"), (True, "trading_off")):
            print(
                f"[{run_id}] opponent={opponent.name} mode={mode} games={games}",
                flush=True,
            )
            blocks = run_matchup(
                candidate_name=candidate.name,
                opponent_name=opponent.name,
                candidate_factory=lambda seed, name=candidate.name: build_agent(name, seed),
                opponent_factory=lambda seed, name=opponent.name: build_agent(name, seed),
                games=games,
                master_seed=args.master_seed,
                suppress_p2p=suppress_p2p,
                max_steps=args.max_steps,
            )
            rows.extend(
                block_row(
                    block,
                    repo_root=repo_root,
                    run_id=run_id,
                    timestamp_utc=timestamp,
                    tier=tier,
                    candidate=candidate.name,
                    candidate_band=candidate.band,
                    opponent=opponent.name,
                    opponent_band=opponent.band,
                    promotion_eligible=opponent.promotion_eligible,
                    anchor=opponent.anchor,
                    incumbent=opponent.incumbent,
                    mode=mode,
                    master_seed=args.master_seed,
                )
                for block in blocks
            )

    parquet_path = (repo_root / args.results_dir / "ladder.parquet").resolve()
    markdown_path = (repo_root / args.results_dir / "ladder.md").resolve()
    append_parquet(rows, parquet_path)
    write_markdown_summary(rows, markdown_path)
    wandb_url = None
    if not args.no_wandb:
        wandb_url = mirror_wandb(rows)

    promotion = promotion_metric(rows)
    by_opponent = aggregate(rows, "opponent", "mode")
    on_rates = {
        opponent: value for (opponent, mode), value in by_opponent.items()
        if mode == "trading_on"
    }
    worst_opponent, worst_rate = min(on_rates.items(), key=lambda item: item[1].rate)
    seats = seat_rates(rows)
    worst_seat, worst_seat_rate = min(seats.items(), key=lambda item: item[1].rate)
    incumbent = next((spec.name for spec in opponents if spec.incumbent), None)
    incumbent_rate = on_rates.get(incumbent) if incumbent is not None else None
    print(
        f"promotion_ci95_low={promotion.ci95_low:.4f} "
        f"mean={promotion.rate:.4f} worst={worst_opponent}:{worst_rate.rate:.4f} "
        f"worst_seat={worst_seat}:{worst_seat_rate.rate:.4f} "
        f"incumbent={incumbent}:{incumbent_rate.rate if incumbent_rate else float('nan'):.4f} "
        f"wandb={wandb_url or 'disabled'}",
        flush=True,
    )
    if not args.no_publish:
        publish_results(repo_root, [parquet_path, markdown_path], run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
