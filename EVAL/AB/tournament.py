"""M4 thesis tournament: a trained fastcatan policy vs Catanatron baselines.

The trained agent plays seat RED via `CatanatronBridge` INSIDE Catanatron's
reference engine; the other three seats are a baseline bot (default:
`AlphaBetaPlayer`, depth 2). Reports the bridge win rate with a 95% Wilson CI
and evaluates the thesis gate.

    # full thesis run (slow — AlphaBeta is depth-N minimax x3 seats):
    python -m AB.tournament --games 1000 --opponent alphabeta --ab-depth 2 --ab-prune

    # smoke:
    python -m AB.tournament --games 20 --opponent random

THESIS GATE: win rate vs AlphaBeta > 25% with 95% confidence, i.e. the Wilson
CI lower bound > 0.25 (0.25 = the 4-player chance baseline).

Requires catanatron 3.3.0 (git 38207ca...). See AB/REPRODUCIBILITY.md.

Reproducibility: pass a fixed --seed AND launch with PYTHONHASHSEED set, e.g.
    PYTHONHASHSEED=0 python -m AB.tournament ...
Catanatron's RandomPlayer and set-iteration order depend on the hash seed
(see bridge/PLAN.md); without it the game stream is not bit-reproducible.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from catanatron import Color
from catanatron.game import Game
from catanatron.models.enums import ActionType
from catanatron.models.player import RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer

from bridge.catanatron_bridge import CatanatronBridge
from models.eval import wilson_ci  # single source of truth for the CI math
from AB.policy import build_policy


COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
BRIDGE_COLOR = Color.RED
DEFAULT_CKPT = "models/checkpoints/ppo_1084_20m/ppo_final.zip"
GATE_BASELINE = 0.25  # 4-player chance baseline / thesis threshold


class TradeSafeAlphaBetaPlayer(AlphaBetaPlayer):
    """AlphaBetaPlayer that can sit at a table with P2P trades enabled.

    Catanatron's AlphaBeta crashes on domestic trades: its minimax steps every
    candidate action through tree_search_utils.expand_spectrum, whose
    execute_spectrum raises RuntimeError on the domestic-trade action types
    (ACCEPT/REJECT/OFFER/CONFIRM/CANCEL_TRADE) — only MARITIME_TRADE is in
    DETERMINISTIC_ACTIONS. The only domestic-trade action AB is ever handed is
    a DECIDE_TRADE response to the bridge's OFFER_TRADE: the engine never lists
    OFFER_TRADE during PLAY_TURN (catanatron/models/actions.py), and
    DECIDE_ACCEPTEES (CONFIRM/CANCEL) goes to the proposer — the bridge, not
    AB. AB has no trade model, so reject the offer directly, skipping minimax.
    Every other decision still runs the real depth-N search, so this is
    behaviourally identical to AlphaBetaPlayer whenever no offer is on the
    table (e.g. every prior --no-trades run)."""

    def decide(self, game, playable_actions):
        if playable_actions and all(
            a.action_type in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE)
            for a in playable_actions
        ):
            for a in playable_actions:
                if a.action_type == ActionType.REJECT_TRADE:
                    return a
            return playable_actions[0]
        return super().decide(game, playable_actions)


def make_opponent(name: str, color: Color, ab_depth: int, ab_prune: bool):
    if name == "random":
        return RandomPlayer(color)
    if name == "alphabeta":
        return TradeSafeAlphaBetaPlayer(color, depth=ab_depth, prunning=ab_prune)
    if name == "value":
        return ValueFunctionPlayer(color)
    raise ValueError(f"unknown opponent: {name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                   help="checkpoint for the bridge policy (default: capped PPO)")
    p.add_argument("--algo", default="ppo", choices=["ppo"],
                   help="policy family (only ppo wired today; see AB/policy.py)")
    p.add_argument("--policy", default="ppo", choices=["ppo", "mcts"],
                   help="'mcts' = state-aware hybrid search (AB/mcts_policy: "
                        "inject the live game into fastcatan, MCTSvsFixed "
                        "with --leaf-eval leaves; --ckpt is the IL .pt).")
    p.add_argument("--mcts-sims", type=int, default=512)
    p.add_argument("--leaf-eval", choices=["net", "ab_value"],
                   default="ab_value")
    p.add_argument("--ab-value-scale", type=float, default=86e6)
    p.add_argument("--model-ab-depth", type=int, default=1,
                   help="depth of the IN-TREE opponent model (native AB); "
                        "independent of the actual table opponents.")
    p.add_argument("--rotate-seats", action="store_true",
                   help="rotate the agent through all 4 seats (game g -> "
                        "seat g%%4) instead of always RED/seat-0.")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--opponent", choices=["alphabeta", "value", "random"],
                   default="alphabeta", help="bot for the other 3 seats")
    p.add_argument("--ab-depth", type=int, default=2,
                   help="AlphaBetaPlayer search depth (default 2)")
    p.add_argument("--ab-prune", action="store_true",
                   help="enable AlphaBeta action pruning (much faster; "
                        "recommended for the 1000-game run)")
    p.add_argument("--deterministic", action="store_true",
                   help="argmax policy (default: sample; see AB/policy.py)")
    p.add_argument("--no-trades", action="store_true",
                   help="disable the bridge OFFER_TRADE compose loop")
    p.add_argument("--max-offers-per-turn", type=int, default=50,
                   help="bridge cap on OFFER_TRADE emissions per turn "
                        "(anti-stall; catanatron has no within-turn offer "
                        "limit). Only matters with trades enabled.")
    p.add_argument("--out", type=str, default="AB/results",
                   help="directory for the results JSON")
    p.add_argument("--progress-every", type=int, default=25)
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    # Deterministic-as-possible run (see module docstring re PYTHONHASHSEED).
    random.seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    hashseed = os.environ.get("PYTHONHASHSEED")
    if hashseed is None:
        print("[warn] PYTHONHASHSEED unset — run is NOT bit-reproducible "
              "(catanatron RNG + set order depend on it).")

    mcts_net = None
    if args.policy == "mcts":
        import torch
        from AB.mcts_policy import MctsStatePolicy
        from models.alphazero.net import load_policy_value_net
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        mcts_net = load_policy_value_net(state, "cpu")
        policy = None                       # built per game (seat-aware)
    else:
        policy = build_policy(args.algo, ckpt,
                              deterministic=args.deterministic)
    enable_trades = not args.no_trades

    wins = {c: 0 for c in COLORS}
    bridge_wins = 0
    mcts_fallbacks = mcts_decisions = 0
    no_winner = 0
    t0 = time.perf_counter()

    for g in range(args.games):
        seed = args.seed + g
        seat = (g % 4) if args.rotate_seats else 0
        bridge_color = COLORS[seat]

        if args.policy == "mcts":
            game_policy = MctsStatePolicy(
                mcts_net, seat=seat, sims=args.mcts_sims,
                leaf_eval=args.leaf_eval,
                ab_value_scale=args.ab_value_scale,
                model_ab_depth=args.model_ab_depth, seed=seed)
        else:
            game_policy = policy

        bridge = CatanatronBridge(bridge_color, policy=game_policy,
                                  seed=seed, enable_trades=enable_trades,
                                  max_offers_per_turn=args.max_offers_per_turn)
        if args.policy == "mcts":
            game_policy.bridge = bridge      # state-aware wiring

        players = []
        for i, color in enumerate(COLORS):
            if i == seat:
                players.append(bridge)
            else:
                players.append(make_opponent(args.opponent, color,
                                             args.ab_depth, args.ab_prune))
        game = Game(players, seed=seed)
        winner = game.play()
        if winner is None:
            no_winner += 1
        else:
            wins[winner] += 1
            if winner == bridge_color:
                bridge_wins += 1
        if args.policy == "mcts":
            mcts_fallbacks += game_policy.fallbacks
            mcts_decisions += game_policy.decisions

        if (g + 1) % args.progress_every == 0:
            dec = (g + 1) - no_winner
            rate = bridge_wins / dec if dec else 0.0
            el = time.perf_counter() - t0
            print(f"[{g+1}/{args.games}] bridge {bridge_wins} wins "
                  f"({rate:.3f} of {dec} decided)  "
                  f"{el / (g + 1):.2f}s/game", flush=True)

    elapsed = time.perf_counter() - t0
    decided = args.games - no_winner
    lo, hi = wilson_ci(bridge_wins, decided)
    rate = bridge_wins / decided if decided else 0.0
    gate_pass = lo > GATE_BASELINE

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ckpt": str(ckpt),
        "algo": args.algo,
        "policy": args.policy,
        "mcts_sims": args.mcts_sims if args.policy == "mcts" else None,
        "leaf_eval": args.leaf_eval if args.policy == "mcts" else None,
        "ab_value_scale": (args.ab_value_scale
                           if args.policy == "mcts" else None),
        "model_ab_depth": (args.model_ab_depth
                           if args.policy == "mcts" else None),
        "rotate_seats": args.rotate_seats,
        "mcts_fallbacks": mcts_fallbacks if args.policy == "mcts" else None,
        "mcts_decisions": mcts_decisions if args.policy == "mcts" else None,
        "deterministic": args.deterministic,
        "enable_trades": enable_trades,
        "max_offers_per_turn": args.max_offers_per_turn,
        "opponent": args.opponent,
        "ab_depth": args.ab_depth if args.opponent == "alphabeta" else None,
        "ab_prune": args.ab_prune if args.opponent == "alphabeta" else None,
        "seed": args.seed,
        "pythonhashseed": hashseed,
        "games": args.games,
        "decided": decided,
        "no_winner": no_winner,
        "bridge_wins": bridge_wins,
        "win_rate": rate,
        "ci95_low": lo,
        "ci95_high": hi,
        "gate_baseline": GATE_BASELINE,
        "gate_pass": gate_pass,
        "seat_wins": {c.name: wins[c] for c in COLORS},
        "elapsed_s": elapsed,
        "s_per_game": elapsed / args.games if args.games else 0.0,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.policy if args.policy != "ppo" else args.algo
    out_path = out_dir / f"tournament_{tag}_{args.opponent}_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\n=== M4 tournament: {args.algo} vs {args.opponent} ===")
    print(f"ckpt:        {ckpt}")
    print(f"games:       {decided}/{args.games} decided (no-winner: {no_winner})")
    print(f"bridge wins: {bridge_wins}")
    print(f"win rate:    {rate:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"seat wins:   {result['seat_wins']}")
    print(f"time:        {elapsed:.1f}s  ({result['s_per_game']:.2f}s/game)")
    print(f"THESIS GATE (CI-low > {GATE_BASELINE}): "
          f"{'PASS' if gate_pass else 'FAIL'}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
