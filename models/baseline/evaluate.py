"""Seat-balanced pool smoke for Phase-2 checkpoints.

This uses the Phase-1 paired-board four-seat rotation primitives, appends rows
through the frozen v0 schema into a run-specific results directory, and returns
compact metrics to the training run's W&B callback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

import fastcatan
from examples.player_base import Player
from ladder.config import LADDER_VERSION, MASTER_SEED
from ladder.match import run_matchup
from ladder.registry import build_agent
from models.baseline.env import POOL_NAMES
from models.eval import wilson_ci
from results.schema import append_rows


class CheckpointPlayer(Player):
    name = "phase2-checkpoint"

    def __init__(self, model, seed: int = 0, deterministic: bool = True):
        super().__init__(seed=seed)
        self.model = model
        self.deterministic = deterministic
        self.seat = 0
        self.obs = np.zeros(fastcatan.OBS_SIZE, dtype=np.float32)

    def bind_seat(self, seat: int) -> None:
        self.seat = int(seat)

    def act(self, env, mask: np.ndarray) -> int:
        env.write_obs(self.seat, self.obs)
        legal = np.zeros(fastcatan.NUM_ACTIONS, dtype=bool)
        for word_idx, word in enumerate(mask):
            bits = int(word)
            while bits:
                bit = (bits & -bits).bit_length() - 1
                action = word_idx * 64 + bit
                if action < fastcatan.NUM_ACTIONS:
                    legal[action] = True
                bits &= bits - 1
        action, _ = self.model.predict(
            self.obs, action_masks=legal, deterministic=self.deterministic,
        )
        return int(action)


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True,
    ).strip()


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    candidate: str,
    results_dir: str | Path,
    opponents: tuple[str, ...] = POOL_NAMES,
    games_per_opponent: int = 32,
    seed: int = MASTER_SEED,
    wandb_url: str | None = None,
    step: int = 0,
) -> dict:
    if games_per_opponent <= 0 or games_per_opponent % 4:
        raise ValueError("games_per_opponent must be a positive multiple of four")
    model = MaskablePPO.load(
        str(checkpoint), device="cuda" if torch.cuda.is_available() else "cpu",
    )
    param_count = sum(p.numel() for p in model.policy.parameters())
    config_blob = json.dumps({
        "checkpoint": str(checkpoint),
        "opponents": opponents,
        "games_per_opponent": games_per_opponent,
        "seed": seed,
        "step": step,
    }, sort_keys=True)
    config_hash = hashlib.sha256(config_blob.encode()).hexdigest()[:16]
    commit = _commit()
    started = time.time()
    rows: list[dict] = []
    summary: dict[str, float] = {}

    def candidate_factory(policy_seed: int):
        return CheckpointPlayer(model, seed=policy_seed, deterministic=True)

    for opponent in opponents:
        for mode, suppress in (("trades_on", False), ("trades_off", True)):
            blocks = run_matchup(
                candidate_name=candidate,
                opponent_name=opponent,
                candidate_factory=candidate_factory,
                opponent_factory=lambda policy_seed, name=opponent: build_agent(
                    name, policy_seed,
                ),
                games=games_per_opponent,
                master_seed=seed,
                suppress_p2p=suppress,
            )
            games = sum(block.games for block in blocks)
            wins = sum(block.candidate_wins for block in blocks)
            no_winner = sum(block.no_winner for block in blocks)
            seat_wins = [
                sum(block.candidate_seat_wins[seat] for block in blocks)
                for seat in range(4)
            ]
            decisions = sum(block.decisions for block in blocks)
            wall = sum(block.wall_seconds for block in blocks)
            low, high = wilson_ci(wins, games)
            win_rate = wins / games
            summary[f"{opponent}/{mode}/win_rate"] = win_rate
            summary[f"{opponent}/{mode}/wilson_low"] = low
            rows.append({
                "ladder_version": LADDER_VERSION,
                "candidate": candidate,
                "opponent": opponent,
                "mode": mode,
                "rotation": "full",
                "games": games,
                "wins": wins,
                "win_rate": win_rate,
                "wilson_low": low,
                "wilson_high": high,
                "no_winner_rate": no_winner / games,
                "seat_wins": seat_wins,
                "trading_delta": None,
                "decisions_per_s": decisions / wall if wall else 0.0,
                "param_count": param_count,
                "commit": commit,
                "config_hash": config_hash,
                "wandb_url": wandb_url,
                "verdict": "checkpoint_smoke",
                "notes": f"step={step} checkpoint={checkpoint}",
            })

    by_opponent = {name: {} for name in opponents}
    for row in rows:
        by_opponent[row["opponent"]][row["mode"]] = row
    for name, modes in by_opponent.items():
        modes["trades_on"]["trading_delta"] = (
            modes["trades_on"]["win_rate"] - modes["trades_off"]["win_rate"]
        )
    append_rows(rows, results_dir=Path(results_dir))
    on_rows = [row for row in rows if row["mode"] == "trades_on"]
    summary["promotion_mean"] = sum(row["wins"] for row in on_rows) / sum(
        row["games"] for row in on_rows
    )
    summary["no_winner_rate"] = sum(
        row["no_winner_rate"] * row["games"] for row in rows
    ) / sum(row["games"] for row in rows)
    summary["wall_seconds"] = time.time() - started
    summary["rows_appended"] = len(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--opponents", default=",".join(POOL_NAMES))
    parser.add_argument("--games-per-opponent", type=int, default=32)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=MASTER_SEED)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--wandb-url", default=None)
    parser.add_argument("--gate-builder-basic", action="store_true")
    args = parser.parse_args()
    opponents = tuple(name for name in args.opponents.split(",") if name)
    summary = evaluate_checkpoint(
        args.checkpoint,
        candidate=args.candidate,
        results_dir=args.results_dir,
        opponents=opponents,
        games_per_opponent=args.games_per_opponent,
        seed=args.seed,
        wandb_url=args.wandb_url,
        step=args.step,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.gate_builder_basic:
        rate = summary["builder-basic/trades_on/win_rate"]
        passed = rate > 0.25
        print(
            f"[gate] builder-basic trading_on win_rate={rate:.6f} "
            f"threshold=>0.25 pass={passed}",
            flush=True,
        )
        if not passed:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
