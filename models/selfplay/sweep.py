"""M3 hyperparam sweep: lr × entropy × snapshot-interval × arch × lr-schedule × target-kl.

Shells out one `train_selfplay` run per grid cell (process isolation), each
writing `summary.json`. Aggregates the final gate rate per cell into a sorted
markdown table + a CSV. Each axis takes >=1 value; an axis left at its default
single value drops out of the grid, so you expand only the knobs you care about.

The lr-schedule and target-kl axes target self-play's diminishing returns
(gains shrink each round): 'linear' lr decay settles late rounds instead of
oscillating; a KL target keeps updates conservative as the opponent pool shifts.

    python -m models.selfplay.sweep \
        --init-from models/checkpoints/ppo_random_10m/ppo_final.zip \
        --seed-pool --no-p2p-trade \
        --lr 3e-4 --ent-coef 0.0 0.01 \
        --lr-schedule constant linear --target-kl none 0.03 \
        --steps-per-round 1000000 --num-rounds 6 --num-envs 8 \
        --out-dir models/checkpoints/sweep
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # grid axes (each takes >=1 value)
    p.add_argument("--lr", type=float, nargs="+", default=[3e-4])
    p.add_argument("--ent-coef", type=float, nargs="+", default=[0.0, 0.01])
    p.add_argument("--steps-per-round", type=int, nargs="+",
                   default=[500_000, 1_000_000])
    p.add_argument("--net-arch", type=str, nargs="+", default=["64,64"],
                   help="Hidden-layer specs to sweep, each comma-separated, e.g. "
                        "--net-arch 64,64 256,256. An arch != the --init-from "
                        "checkpoint's trains from scratch (train_selfplay logs it).")
    p.add_argument("--lr-schedule", choices=["constant", "linear"], nargs="+",
                   default=["constant"], help="lr decay schedules to sweep.")
    p.add_argument("--target-kl", type=str, nargs="+", default=["none"],
                   help="KL targets to sweep; 'none' = off, else a float like 0.03.")
    # shared run config (forwarded to every cell)
    p.add_argument("--init-from", type=str, default=None)
    p.add_argument("--init-dir", type=str, default=None,
                   help="Per-arch warm-start seeds: each cell loads "
                        "{init_dir}/{arch_tag}/ppo_final.zip (arch_tag = arch with "
                        "commas->dashes, e.g. 256-256). Use when sweeping --net-arch "
                        "so each arch warm-starts from its OWN matching-arch seed "
                        "(a single --init-from mismatches every non-default arch and "
                        "cold-starts it). Falls back to --init-from if a seed is "
                        "missing.")
    p.add_argument("--seed-pool", action="store_true")
    p.add_argument("--no-p2p-trade", action="store_true",
                   help="Forbid p2p trades (train+gate) so gates are conclusive; "
                        "without it self-play games stall and gates are undecidable.")
    p.add_argument("--trade-compose-cap", type=int, default=None,
                   help="Per-seat trade-compose cap/turn, forwarded to every cell. "
                        "Lets trades stay ON without the stall (alternative to "
                        "--no-p2p-trade). Omit = train_selfplay default (20).")
    p.add_argument("--num-rounds", type=int, default=8)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--gate-lag", type=int, default=2)
    p.add_argument("--gate-games", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str,
                   default="models/checkpoints/sweep")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands without running them.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(
        args.lr, args.ent_coef, args.steps_per_round, args.net_arch,
        args.lr_schedule, args.target_kl))
    print(f"[sweep] {len(grid)} cells: lr×ent×interval×arch×sched×kl = "
          f"{len(args.lr)}×{len(args.ent_coef)}×{len(args.steps_per_round)}"
          f"×{len(args.net_arch)}×{len(args.lr_schedule)}×{len(args.target_kl)}")

    rows: list[dict] = []
    for i, (lr, ent, interval, arch, lrsched, kl) in enumerate(grid):
        arch_tag = arch.replace(",", "-").replace(" ", "")
        run_name = (f"sweep/lr{lr:g}_ent{ent:g}_int{interval}_arch{arch_tag}"
                    f"_{lrsched}_kl{kl}")
        cmd = [
            sys.executable, "-m", "models.selfplay.train_selfplay",
            "--lr", str(lr),
            "--ent-coef", str(ent),
            "--steps-per-round", str(interval),
            "--net-arch", arch,
            "--lr-schedule", lrsched,
            "--num-rounds", str(args.num_rounds),
            "--num-envs", str(args.num_envs),
            "--gate-lag", str(args.gate_lag),
            "--gate-games", str(args.gate_games),
            "--seed", str(args.seed),
            "--run-name", run_name,
        ]
        if kl != "none":
            cmd += ["--target-kl", kl]
        cell_init = args.init_from
        if args.init_dir:
            cand = Path(args.init_dir) / arch_tag / "ppo_final.zip"
            if cand.exists():
                cell_init = str(cand)
            else:
                fallback = f"--init-from {args.init_from}" if args.init_from else "cold-start"
                print(f"  [warn] no per-arch seed {cand} for arch {arch} -> {fallback}")
        if cell_init:
            cmd += ["--init-from", cell_init]
        if args.seed_pool:
            cmd += ["--seed-pool"]
        if args.no_p2p_trade:
            cmd += ["--no-p2p-trade"]
        if args.trade_compose_cap is not None:
            cmd += ["--trade-compose-cap", str(args.trade_compose_cap)]

        print(f"\n[sweep] cell {i + 1}/{len(grid)}: {run_name}")
        if args.dry_run:
            print("  " + " ".join(cmd))
            continue

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            # Don't let one bad cell abort an unattended overnight sweep: log it,
            # record an empty row, and carry on to the next cell.
            print(f"  [error] cell FAILED (exit {exc.returncode}): {run_name}")
            rows.append({
                "lr": lr, "ent_coef": ent, "steps_per_round": interval,
                "net_arch": arch, "lr_schedule": lrsched, "target_kl": kl,
                "final_gate_rate": None, "pass": None, "run_name": run_name,
            })
            continue

        summary_path = (Path("models/checkpoints") / run_name / "summary.json")
        rate = pass_ = None
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            rate, pass_ = s.get("final_gate_rate"), s.get("final_gate_pass")
        rows.append({
            "lr": lr, "ent_coef": ent, "steps_per_round": interval,
            "net_arch": arch, "lr_schedule": lrsched, "target_kl": kl,
            "final_gate_rate": rate, "pass": pass_, "run_name": run_name,
        })

    if args.dry_run or not rows:
        return

    # CSV
    csv_path = out_dir / "sweep_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Markdown table, best gate rate first.
    rows.sort(key=lambda r: (r["final_gate_rate"] is None, -(r["final_gate_rate"] or 0)))
    lines = ["| lr | ent | interval | arch | sched | kl | gate rate | pass |",
             "|----|-----|----------|------|-------|----|-----------|------|"]
    for r in rows:
        rate = f"{r['final_gate_rate']:.3f}" if r["final_gate_rate"] is not None else "—"
        lines.append(f"| {r['lr']:g} | {r['ent_coef']:g} | {r['steps_per_round']} "
                     f"| {r['net_arch']} | {r['lr_schedule']} | {r['target_kl']} "
                     f"| {rate} | {r['pass']} |")
    table = "\n".join(lines)
    (out_dir / "sweep_results.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\n[sweep] -> {csv_path}\n[sweep] -> {out_dir / 'sweep_results.md'}")


if __name__ == "__main__":
    main()
