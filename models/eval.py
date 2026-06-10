"""Evaluate a trained checkpoint vs random opponents.

Supports all four algos in this dir:

    python -m models.eval --algo ppo    --ckpt models/checkpoints/ppo_random/ppo_final.zip --games 1000
    python -m models.eval --algo dqn    --ckpt models/checkpoints/dqn/dqn_final.pt        --games 1000
    python -m models.eval --algo a2c    --ckpt models/checkpoints/a2c/a2c_final.pt        --games 1000
    python -m models.eval --algo muzero --ckpt models/checkpoints/muzero/muzero_final.pt  --games 200 --sims 16

Each algo gets wrapped behind a single `pick_action(obs, mask) -> int` interface;
the rollout loop is identical across algos.

M2 gate: win rate Wilson 95% CI lower bound > 0.90.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from models.env import FastCatanEnv, NUM_ACTIONS
from models.ckpt import verify_stamp, _fingerprint


PickAction = Callable[[np.ndarray, np.ndarray], int]


# -------------------- statistics --------------------

def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return center - half, center + half


# -------------------- algo loaders --------------------

def load_ppo(ckpt: Path, deterministic: bool) -> PickAction:
    from sb3_contrib import MaskablePPO
    verify_stamp(ckpt)
    model = MaskablePPO.load(str(ckpt))

    def pick(obs: np.ndarray, mask: np.ndarray) -> int:
        action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
        return int(action)

    return pick


def load_dqn(ckpt: Path, deterministic: bool) -> PickAction:
    from models.train_dqn import QNet
    verify_stamp(ckpt, strict=False)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    q = QNet()
    q.load_state_dict(state["q_state"])
    q.eval()

    def pick(obs: np.ndarray, mask: np.ndarray) -> int:
        with torch.no_grad():
            q_vals = q(torch.from_numpy(obs).float().unsqueeze(0))[0].numpy()
        masked = np.where(mask, q_vals, -np.inf)
        return int(np.argmax(masked))

    return pick


def load_a2c(ckpt: Path, deterministic: bool) -> PickAction:
    from models.train_a2c import ActorCritic, masked_categorical
    verify_stamp(ckpt, strict=False)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    net = ActorCritic()
    net.load_state_dict(state["net_state"])
    net.eval()

    def pick(obs: np.ndarray, mask: np.ndarray) -> int:
        with torch.no_grad():
            logits, _ = net(torch.from_numpy(obs).float().unsqueeze(0))
            mask_t = torch.from_numpy(mask).unsqueeze(0)
            if deterministic:
                neg_inf = torch.full_like(logits, float("-inf"))
                masked = torch.where(mask_t, logits, neg_inf)
                return int(masked.argmax(dim=-1).item())
            dist = masked_categorical(logits, mask_t)
            return int(dist.sample().item())

    return pick


def load_muzero(ckpt: Path, deterministic: bool, sims: int) -> PickAction:
    from models.train_muzero import MuZeroNets, run_mcts, visit_count_policy
    verify_stamp(ckpt, strict=False)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    nets = MuZeroNets()
    nets.load_state_dict(state["nets_state"])
    nets.eval()
    gamma = state.get("args", {}).get("gamma", 0.997)

    def pick(obs: np.ndarray, mask: np.ndarray) -> int:
        with torch.no_grad():
            h = nets.repr(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0)
        root = run_mcts(h, mask, nets, sims, gamma)
        pi = visit_count_policy(root, temperature=0.0 if deterministic else 1.0)
        pi_legal = pi * mask
        if pi_legal.sum() == 0:
            legal_ids = np.where(mask)[0]
            return int(np.random.choice(legal_ids))
        pi_legal /= pi_legal.sum()
        if deterministic:
            return int(np.argmax(pi_legal))
        return int(np.random.choice(NUM_ACTIONS, p=pi_legal))

    return pick


def build_agent(algo: str, ckpt: Path, deterministic: bool, sims: int) -> PickAction:
    if algo == "ppo":
        return load_ppo(ckpt, deterministic)
    if algo == "dqn":
        return load_dqn(ckpt, deterministic)
    if algo == "a2c":
        return load_a2c(ckpt, deterministic)
    if algo == "muzero":
        return load_muzero(ckpt, deterministic, sims)
    raise ValueError(f"unknown algo: {algo}")


# -------------------- rollout loop --------------------

def play_game(env: FastCatanEnv, pick: PickAction, max_steps: int = 5000) -> int:
    """Play one game. Returns winning seat, or -1 if no terminal within max_steps
    (guard against deterministic-argmax policies that get stuck in legal no-op
    loops, e.g. TRADE_OPEN/CANCEL ping-pong)."""
    obs, _ = env.reset()
    done = False
    for _ in range(max_steps):
        if done:
            break
        mask = env.action_masks()
        action = pick(obs, mask)
        obs, _r, done, _trunc, _info = env.step(action)
    for p in range(4):
        if env._env.player_vp(p) >= 10:
            return p
    return -1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["ppo", "dqn", "a2c", "muzero"])
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--deterministic", action="store_true",
                   help="Argmax / greedy action selection (default: sample).")
    p.add_argument("--sims", type=int, default=16,
                   help="MuZero MCTS simulations per move.")
    p.add_argument("--max-steps", type=int, default=5000,
                   help="Per-game step cap (game counted as no-winner past cap).")
    p.add_argument("--no-trades", action="store_true",
                   help="Suppress player-to-player trades for the learner AND the "
                        "random opponents (maritime bank/port trades stay). The "
                        "canonical thesis ablation; match it to how the agent was "
                        "trained.")
    p.add_argument("--out", type=str, default=None,
                   help="Write a rich JSON result (win rate, CI, seat dist, full "
                        "metadata) to this path so the run never has to be repeated. "
                        "Aggregate many with models/benchmarks/summarize.py.")
    p.add_argument("--tag", type=str, default=None,
                   help="Free-text label stored in the JSON (e.g. '50M vs random').")
    # Optional training provenance (the orchestrator passes these so the summary
    # table can show train steps / time / net arch). Purely descriptive.
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--train-seconds", type=float, default=None)
    p.add_argument("--train-net-arch", type=str, default=None)
    p.add_argument("--train-num-envs", type=int, default=None)
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    pick = build_agent(args.algo, ckpt, args.deterministic, args.sims)
    env = FastCatanEnv(seed=args.seed, suppress_p2p_trade=args.no_trades)

    wins = 0
    seat_wins = [0, 0, 0, 0]
    no_winner = 0
    for g in range(args.games):
        w = play_game(env, pick, max_steps=args.max_steps)
        if w < 0:
            no_winner += 1
            continue
        seat_wins[w] += 1
        if w == 0:
            wins += 1
        if (g + 1) % 50 == 0:
            print(f"[{g+1}/{args.games}] learner wins {wins} "
                  f"({wins / (g + 1):.3f})")

    n = args.games - no_winner
    lo, hi = wilson_ci(wins, n)
    rate = wins / n if n else 0.0
    gate_pass = lo > 0.90
    trades = "off (p2p suppressed, maritime on)" if args.no_trades else "on"

    print(f"\n=== Eval results ({args.algo}) ===")
    print(f"ckpt: {ckpt}")
    print(f"opponent: random | trades: {trades} | "
          f"action={'argmax' if args.deterministic else 'sample'}")
    print(f"games (winnered): {n}/{args.games} (no-winner: {no_winner})")
    print(f"learner win rate: {rate:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"seat distribution: {seat_wins}")
    print(f"M2 gate (>0.90 CI low): {'PASS' if gate_pass else 'FAIL'}")

    if args.out:
        result = {
            "algo": args.algo,
            "tag": args.tag,
            "ckpt": str(ckpt),
            "opponent": "random",
            "trades": "off" if args.no_trades else "on",
            "no_trades": bool(args.no_trades),
            "deterministic": bool(args.deterministic),
            "games_requested": args.games,
            "games_scored": n,
            "no_winner": no_winner,
            "wins": wins,
            "win_rate": rate,
            "ci95_low": lo,
            "ci95_high": hi,
            "m2_gate_pass": gate_pass,
            "seat_wins": seat_wins,
            "eval_seed": args.seed,
            "max_steps": args.max_steps,
            "sims": args.sims if args.algo == "muzero" else None,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "env": _fingerprint(),
        }
        if any(v is not None for v in (args.train_steps, args.train_seconds,
                                       args.train_net_arch, args.train_num_envs)):
            result["train"] = {
                "total_steps": args.train_steps,
                "train_seconds": args.train_seconds,
                "net_arch": args.train_net_arch,
                "num_envs": args.train_num_envs,
            }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[eval] saved result -> {out}")


if __name__ == "__main__":
    main()
