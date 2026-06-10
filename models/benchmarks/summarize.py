"""Aggregate per-algo eval JSONs (written by `models/eval.py --out`) into a
thesis-ready table.

    python models/benchmarks/summarize.py models/benchmarks/baselines_vs_random

Scans the directory for `*.json` results (anything with `algo` + `win_rate`),
sorts them in the canonical PPO -> A2C -> DQN order, and writes:

  - <dir>/summary.md   Markdown table (drop straight into the thesis)
  - <dir>/summary.csv  same data, machine-readable

and prints the table to stdout. Re-run any time you add/replace a result —
it is idempotent and never touches the source JSONs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

# Canonical display order; anything else is appended alphabetically after these.
ALGO_ORDER = ["ppo", "a2c", "dqn", "muzero"]

COLUMNS = [
    ("algo", "Algo"),
    ("train_steps", "Train steps"),
    ("trades", "Trades"),
    ("games_scored", "Games"),
    ("win_rate_pct", "Win% vs random"),
    ("ci95", "95% CI"),
    ("m2_gate", "M2 gate"),
    ("seat_wins", "Seat wins"),
    ("train_time", "Train time"),
    ("git_sha", "Git SHA"),
]


def _fmt_steps(n) -> str:
    if not n:
        return "?"
    n = int(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}k"
    return str(n)


def _fmt_time(sec) -> str:
    if not sec:
        return "?"
    sec = float(sec)
    if sec >= 3600:
        return f"{sec/3600:.1f}h"
    if sec >= 60:
        return f"{sec/60:.1f}m"
    return f"{sec:.0f}s"


def _row(d: dict) -> dict:
    train = d.get("train", {}) or {}
    rate = d.get("win_rate", 0.0)
    lo, hi = d.get("ci95_low", 0.0), d.get("ci95_high", 0.0)
    return {
        "algo": d.get("algo", "?").upper(),
        "train_steps": _fmt_steps(train.get("total_steps")),
        "trades": d.get("trades", "?"),
        "games_scored": f"{d.get('games_scored', 0)}/{d.get('games_requested', 0)}",
        "win_rate_pct": f"{rate*100:.1f}",
        "ci95": f"[{lo*100:.1f}, {hi*100:.1f}]",
        "m2_gate": "PASS" if d.get("m2_gate_pass") else "FAIL",
        "seat_wins": str(d.get("seat_wins", [])),
        "train_time": _fmt_time(train.get("train_seconds")),
        "git_sha": (d.get("env", {}) or {}).get("git_sha", "?"),
        # sort keys (not displayed)
        "_order": ALGO_ORDER.index(d["algo"]) if d.get("algo") in ALGO_ORDER else 99,
        "_rate": rate,
    }


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for fp in sorted(glob.glob(str(results_dir / "*.json"))):
        try:
            d = json.loads(Path(fp).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(d, dict) or "algo" not in d or "win_rate" not in d:
            continue  # skip non-result JSONs (configs, etc.)
        rows.append(_row(d))
    rows.sort(key=lambda r: (r["_order"], -r["_rate"]))
    return rows


def to_markdown(rows: list[dict], title: str) -> str:
    keys = [k for k, _ in COLUMNS]
    head = [h for _, h in COLUMNS]
    out = [f"# {title}", ""]
    out.append(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
               f"· opponent: random · {len(rows)} agent(s)_")
    out.append("")
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join(["---"] * len(head)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
    out.append("")
    return "\n".join(out)


def to_csv(rows: list[dict], path: Path) -> None:
    keys = [k for k, _ in COLUMNS]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([h for _, h in COLUMNS])
        for r in rows:
            w.writerow([r[k] for k in keys])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", help="Directory of *.json eval results.")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise NotADirectoryError(results_dir)

    rows = load_results(results_dir)
    if not rows:
        print(f"[summarize] no result JSONs found in {results_dir}")
        return

    title = args.title or f"Baselines vs random — {results_dir.name}"
    md = to_markdown(rows, title)
    (results_dir / "summary.md").write_text(md + "\n")
    to_csv(rows, results_dir / "summary.csv")

    print(md)
    print(f"[summarize] wrote {results_dir/'summary.md'} and "
          f"{results_dir/'summary.csv'}")


if __name__ == "__main__":
    main()
