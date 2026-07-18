"""Results schema v0 (frozen at substrate-v1; spec 0.3 §E).

The single append-only results contract every tournament / checkpoint / kill
writes. One row per (ladder_version, candidate, opponent, mode, rotation-block).
Rows land in `results/ladder.parquet` (canonical) + a human `results/ladder.md`
line. See `results/SCHEMA.md` for the field-by-field contract.

Frozen: adding/renaming/removing a column is a schema change — bump
`SCHEMA_VERSION`, update `SCHEMA.md`, and (since it is eval-visible) bump the
ladder version. `check_frozen.sh` guards this file and `SCHEMA.md`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

SCHEMA_VERSION = "v0"

# Ordered column contract. Anything a writer omits is filled with None so the
# parquet stays rectangular across schema-compatible writers.
FIELDS = [
    "schema_version",   # this contract's version
    "ts",               # unix seconds (float)
    "ladder_version",   # frozen roster id the row was measured against
    "candidate",        # agent under test (checkpoint id / persona name)
    "opponent",         # opponent set label (e.g. "3x random", "AB-d2")
    "mode",             # "trades_on" | "trades_off" (program requires both logged)
    "rotation",         # "full" (seat-balanced block) | seat index for a slice
    "games",            # games in this row
    "wins",             # candidate wins
    "win_rate",         # wins / games
    "wilson_low",       # Wilson lower bound (z=1.96) — the promotion statistic
    "wilson_high",
    "no_winner_rate",   # fraction ending with no winner (MAX_TURNS backstop / stall)
    "seat_wins",        # json [w0,w1,w2,w3] — worst-seat slice
    "trading_delta",    # win_rate(trades_on) - win_rate(trades_off); None off-row
    "decisions_per_s",  # throughput
    "param_count",      # candidate parameter count (0 for scripted/random)
    "commit",           # engine commit the row was produced on
    "config_hash",      # hash of the run config
    "wandb_url",        # W&B run, if any
    "verdict",          # "pass" | "fail" | "baseline" | free text
    "notes",
]


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return p.parent.parent


def _normalize(row: dict) -> dict:
    out = {k: row.get(k) for k in FIELDS}
    out["schema_version"] = SCHEMA_VERSION
    if out["ts"] is None:
        out["ts"] = time.time()
    if isinstance(out["seat_wins"], (list, tuple)):
        out["seat_wins"] = json.dumps(list(out["seat_wins"]))
    return out


def append_rows(rows: list[dict], results_dir: Path | None = None) -> Path:
    """Append normalized rows to results/ladder.parquet (+ ladder.md). Returns the
    parquet path. Creates the store on first write."""
    results_dir = results_dir or (_repo_root() / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    parquet = results_dir / "ladder.parquet"
    md = results_dir / "ladder.md"

    norm = [_normalize(r) for r in rows]

    import pandas as pd

    new = pd.DataFrame(norm, columns=FIELDS)
    if parquet.exists():
        old = pd.read_parquet(parquet)
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df.to_parquet(parquet, index=False)

    if not md.exists():
        md.write_text(
            "# Ladder results (schema " + SCHEMA_VERSION + ")\n\n"
            "Append-only; one row per (ladder_version, candidate, opponent, mode, "
            "rotation). Canonical store is `ladder.parquet`; this table mirrors it.\n\n"
            "| ts | ladder | candidate | opponent | mode | games | win_rate | "
            "wilson_low | no_winner | verdict |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n"
        )
    with md.open("a") as f:
        for r in norm:
            f.write(
                f"| {int(r['ts'])} | {r['ladder_version']} | {r['candidate']} | "
                f"{r['opponent']} | {r['mode']} | {r['games']} | "
                f"{_fmt(r['win_rate'])} | {_fmt(r['wilson_low'])} | "
                f"{_fmt(r['no_winner_rate'])} | {r['verdict']} |\n"
            )
    return parquet


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)
