"""Aggregation and promotion statistics for ladder result rows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from models.eval import wilson_ci


@dataclass(frozen=True)
class Rate:
    wins: int
    games: int
    no_winner: int
    rate: float
    ci95_low: float
    ci95_high: float


def rate(wins: int, games: int, no_winner: int = 0) -> Rate:
    """Wilson rate with no-winner games retained as losses in ``games``."""
    low, high = wilson_ci(wins, games)
    return Rate(
        wins=wins,
        games=games,
        no_winner=no_winner,
        rate=wins / games if games else 0.0,
        ci95_low=low,
        ci95_high=high,
    )


def aggregate(rows: Iterable[Mapping], *keys: str) -> dict[tuple, Rate]:
    buckets: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        bucket = tuple(row[key] for key in keys)
        buckets[bucket][0] += int(row["candidate_wins"])
        buckets[bucket][1] += int(row["games"])
        buckets[bucket][2] += int(row["no_winner"])
    return {
        bucket: rate(wins, games, no_winner)
        for bucket, (wins, games, no_winner) in buckets.items()
    }


def promotion_metric(rows: Iterable[Mapping]) -> Rate:
    """Pooled Wilson bound across equal-sized legal-information matchups."""
    eligible = [
        row for row in rows
        if bool(row["promotion_eligible"]) and row["mode"] == "trading_on"
    ]
    return rate(
        sum(int(row["candidate_wins"]) for row in eligible),
        sum(int(row["games"]) for row in eligible),
        sum(int(row["no_winner"]) for row in eligible),
    )


def seat_rates(rows: Iterable[Mapping]) -> dict[int, Rate]:
    materialized = list(rows)
    return {
        seat: rate(
            sum(int(row[f"candidate_seat{seat}_wins"]) for row in materialized),
            sum(int(row[f"candidate_seat{seat}_games"]) for row in materialized),
            0,
        )
        for seat in range(4)
    }
