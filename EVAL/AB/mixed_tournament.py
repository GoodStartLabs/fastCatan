"""Mixed-table experiment: N trade-capable trained seats vs (4-N) Alpha-Beta.

The standard thesis run (`AB.tournament`) seats ONE trained agent against THREE
AlphaBeta bots and asks "can the agent beat AlphaBeta?". This driver flips the
ratio: `--n-agents` copies of a trained fastcatan policy (PPO bridge or the
hybrid MCTS search agent) share a table with the remaining AlphaBeta seats —
3v1 (`--n-agents 3`) or 2v2 (`--n-agents 2`). Trades are ON by default: the
trained seats can deal with EACH OTHER — the one capability AlphaBeta
structurally lacks (it never offers; it can only respond). The question:
does a trading table suppress AlphaBeta below its fair seat share, and how
much of that is the trading itself (`--no-trades` ablation, same table)?

Policies:
  --policy ppo  (default) — reactive SB3 checkpoint via AB/policy.build_policy;
                all trained seats share one loaded model (serial queries, safe).
                Sampling mode makes the otherwise identical seats diverge.
  --policy mcts — the hybrid search agent (AB/mcts_policy.MctsStatePolicy):
                IL prior + ab_value leaves + in-tree AB opponent model. One
                net is shared; each trained seat gets its OWN policy instance
                (per-seat envs + TRUE-seat sync). With trades enabled the
                search composes offers natively in-tree (learner_trades).

Seat fairness: the AlphaBeta block is rotated across colors over the series
(`--rotate`, default on); catanatron additionally shuffles turn order per
game, so neither color nor turn position biases the aggregate.

Metrics: AlphaBeta block win rate vs its fair share ((4-N)/4 of decided
games), per-trained-seat rate vs 0.25, full p2p/maritime trade tallies.

Usage (from repo root, conda env `catan` — see AB/REPRODUCIBILITY.md):

    # 3v1, PPO traders (the 2026-06-01 experiment):
    PYTHONHASHSEED=0 PYTHONPATH=EVAL python -m AB.mixed_tournament \
        --games 200 --ab-depth 2 --ab-prune --seed 42

    # 2v2, hybrid search seats, gate-config search (slow: 2 MCTS seats/game):
    PYTHONHASHSEED=0 PYTHONPATH=EVAL python -m AB.mixed_tournament \
        --n-agents 2 --policy mcts --mcts-sims 512 --model-ab-depth 2 \
        --model-ab-prune --ab-depth 2 --ab-prune --games 200 --seed 42

    # trades-off ablation of either: add --no-trades
    # smoke:
    PYTHONHASHSEED=0 PYTHONPATH=EVAL python -m AB.mixed_tournament --games 5

Reproducibility: pass a fixed --seed AND set PYTHONHASHSEED (catanatron's
RandomPlayer / set-iteration order depend on it). See AB/tournament.py docstring.
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
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import get_value_fn

from bridge.catanatron_bridge import CatanatronBridge
from models.eval import wilson_ci  # single source of truth for the CI math
from AB.policy import build_policy


COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]

# Domestic-trade resolution action types. Catanatron's AlphaBeta tree search
# (tree_search_utils.execute_spectrum) does NOT know how to expand these — it
# raises "Unknown ActionType" — which is why the thesis gate runs --no-trades.
# To let AlphaBeta sit at a table where the trained agents trade, we intercept
# these prompts (see TradeAwareAlphaBeta) and answer them with a depth-0 value
# lookup instead of the unsupported tree expansion.
_TRADE_RESOLVE_TYPES = frozenset({
    ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE,
    ActionType.CONFIRM_TRADE, ActionType.CANCEL_TRADE,
})
_DECLINE_TYPES = (ActionType.REJECT_TRADE, ActionType.CANCEL_TRADE)


class TradeAwareAlphaBeta(AlphaBetaPlayer):
    """AlphaBetaPlayer that can survive a trading table.

    Vanilla AlphaBeta crashes the instant it must respond to a domestic trade:
    `decide` runs minimax, and `execute_spectrum` has no case for
    ACCEPT/REJECT/CONFIRM/CANCEL_TRADE. This subclass detects a trade-resolution
    prompt and answers it greedily with the SAME heuristic AlphaBeta uses at its
    search leaves (one-ply value comparison of each response), bypassing the
    tree search entirely. Every non-trade decision is delegated unchanged to the
    real AlphaBeta search, so its playing strength on the board is intact.

    AlphaBeta never *offers* trades (the bridge composes OFFER_TRADE; it is not
    in catanatron's `playable_actions`), so its own search turns never generate
    trade nodes — only the responder path needs this guard.
    """

    def decide(self, game: Game, playable_actions):
        if any(a.action_type in _TRADE_RESOLVE_TYPES for a in playable_actions):
            return self._decide_trade(game, playable_actions)
        return super().decide(game, playable_actions)

    def _decide_trade(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        value_fn = get_value_fn(self.value_fn_builder_name, self.params, None)
        best, best_v = None, float("-inf")
        for a in playable_actions:
            try:
                g2 = game.copy()
                g2.execute(a, validate_action=False)
                v = value_fn(g2, self.color)
            except Exception:
                # Unscored response = treat as worst; keep AB robust over a long
                # run rather than crash on an engine edge case.
                v = float("-inf")
            if v > best_v:
                best, best_v = a, v
        if best is not None:
            return best
        # All responses unscored — fall back to declining the trade.
        for a in playable_actions:
            if a.action_type in _DECLINE_TYPES:
                return a
        return playable_actions[0]

# 200M self-play league checkpoint (1084/286) — the strongest trade-trained
# reactive agent in the repo. For --policy mcts the default is the gate-run
# IL prior (the 32.5% hybrid's net).
DEFAULT_PPO_CKPT = "models/checkpoints/sp_league_200m_512/selfplay_final.zip"
DEFAULT_MCTS_CKPT = "models/checkpoints/il_ab_d2_160k_vpm/il_final.pt"
FAIR_SHARE = 0.25  # per-seat chance baseline (1 of 4 seats)


def build_players(ab_colors, args, policy, mcts_net, game_seed: int,
                  judge_net=None):
    """AlphaBeta at every color in `ab_colors`; the other seats are trained
    bridges. PPO seats share `policy`; MCTS seats each get their own
    state-aware policy instance (TRUE seat is re-synced live every decision).
    Distinct per-seat RNG seeds keep fallback / compose tie-breaks from
    lock-stepping. Returns (players, mcts_policies)."""
    enable_trades = not args.no_trades
    players, mcts_policies = [], []
    for i, c in enumerate(COLORS):
        if c in ab_colors:
            players.append(TradeAwareAlphaBeta(
                c, depth=args.ab_depth, prunning=args.ab_prune))
            continue
        if args.policy == "mcts":
            from AB.mcts_policy import MctsStatePolicy
            game_policy = MctsStatePolicy(
                mcts_net, seat=i, sims=args.mcts_sims,
                leaf_eval=args.leaf_eval,
                ab_value_scale=args.ab_value_scale,
                model_ab_depth=args.model_ab_depth,
                model_ab_prune=args.model_ab_prune,
                model_catanatron_chance=args.model_catanatron_chance,
                opp_model=args.model_opp,
                enable_trades=enable_trades,
                trade_add_cap=args.trade_add_cap,
                seed=game_seed * 4 + i,
                judge=judge_net)
            mcts_policies.append(game_policy)
        else:
            game_policy = policy
        bridge = CatanatronBridge(
            c, policy=game_policy, seed=game_seed * 4 + i,
            enable_trades=enable_trades,
            max_offers_per_turn=args.max_offers_per_turn)
        if args.policy == "mcts":
            game_policy.bridge = bridge      # state-aware wiring
        players.append(bridge)
    return players, mcts_policies


# Trade action types tallied from the post-game action log. OFFER_TRADE =
# a domestic offer attempt; CONFIRM_TRADE = a completed player-to-player trade
# (proposer confirms an accepter); MARITIME_TRADE = a bank/port (4:1/3:1/2:1)
# trade — not player-to-player.
_TRADE_COUNT_TYPES = {
    ActionType.OFFER_TRADE: "offers",
    ActionType.CONFIRM_TRADE: "confirms",
    ActionType.ACCEPT_TRADE: "accepts",
    ActionType.REJECT_TRADE: "rejects",
    ActionType.CANCEL_TRADE: "cancels",
    ActionType.MARITIME_TRADE: "maritime",
}
_TRADE_KEYS = (list(dict.fromkeys(_TRADE_COUNT_TYPES.values()))
               + ["ab_accepts", "maritime_ab"])


def tally_trades(game, ab_colors) -> dict:
    """Count trade actions in a finished game's action log. Offers/confirms are
    inherently all-trained-seats (AlphaBeta never offers); `maritime_ab` splits
    out the AB block's bank trades. `ab_accepts` = times any AB accepted a
    domestic offer."""
    out = {k: 0 for k in _TRADE_KEYS}
    for rec in game.state.action_records:
        a = rec.action
        key = _TRADE_COUNT_TYPES.get(a.action_type)
        if key is None:
            continue
        out[key] += 1
        if a.color in ab_colors:
            if a.action_type == ActionType.ACCEPT_TRADE:
                out["ab_accepts"] += 1
            elif a.action_type == ActionType.MARITIME_TRADE:
                out["maritime_ab"] += 1
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None,
                   help="checkpoint for the trained seats (default: "
                        f"{DEFAULT_PPO_CKPT} for ppo, "
                        f"{DEFAULT_MCTS_CKPT} for mcts)")
    p.add_argument("--policy", default="ppo", choices=["ppo", "mcts"],
                   help="'mcts' = state-aware hybrid search per trained seat "
                        "(AB/mcts_policy; --ckpt is the IL .pt)")
    p.add_argument("--algo", default="ppo", choices=["ppo"])
    p.add_argument("--n-agents", type=int, default=3, choices=[1, 2, 3],
                   help="trained seats at the table; the other 4-N are "
                        "AlphaBeta (3 = 3v1, 2 = 2v2)")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ab-depth", type=int, default=2)
    p.add_argument("--ab-prune", action="store_true",
                   help="enable AlphaBeta action pruning (recommended)")
    # MCTS knobs — mirror AB/tournament.py exactly.
    p.add_argument("--mcts-sims", type=int, default=512)
    p.add_argument("--leaf-eval", choices=["net", "ab_value"],
                   default="ab_value")
    p.add_argument("--ab-value-scale", type=float, default=86e6)
    p.add_argument("--model-ab-depth", type=int, default=1,
                   help="depth of the IN-TREE opponent model (native AB); "
                        "independent of the actual table opponents.")
    p.add_argument("--model-ab-prune", action="store_true",
                   help="prune the IN-TREE opponent model's action set — "
                        "match this to the table's --ab-prune.")
    p.add_argument("--model-catanatron-chance", action="store_true",
                   help="in-tree model uses Catanatron's chance blur + "
                        "first-enemy robber pruner (see AB/model_divergence).")
    p.add_argument("--judge-ckpt", type=str, default="",
                   help="FULL-OBS JUDGE checkpoint: owns the leaf value + "
                        "trade pricing inside each MCTS seat (self-contained "
                        "replacement for ab_value's information set). --ckpt "
                        "keeps the prior + in-tree opponent.")
    p.add_argument("--model-opp", choices=["alphabeta", "net"],
                   default="alphabeta",
                   help="IN-TREE opponent model: 'net' = the clone's own "
                        "argmax + value-head trade responses (stage-2 "
                        "de-cat; with --leaf-eval net the seats are FULLY "
                        "SELF-CONTAINED — no ab_value/ab_decide at "
                        "inference).")
    p.add_argument("--trade-add-cap", type=int, default=3,
                   help="max cards per side of an in-tree composed offer "
                        "(bounds compose-churn arms; mcts only)")
    p.add_argument("--deterministic", action="store_true",
                   help="argmax ppo policy (default: sample; mcts is always "
                        "argmax-by-visits)")
    p.add_argument("--no-trades", action="store_true",
                   help="disable p2p trading at the trained seats — the "
                        "ablation arm (default: trades ON, the point of "
                        "this experiment)")
    p.add_argument("--max-offers-per-turn", type=int, default=5,
                   help="bridge cap on OFFER_TRADE emissions per turn per "
                        "seat (anti-spam; the search tries a trade whenever "
                        "it beats END_TURN, so keep this small)")
    p.add_argument("--no-rotate", action="store_true",
                   help="pin the AlphaBeta block to the first colors instead "
                        "of rotating it across seats")
    p.add_argument("--out", type=str, default="EVAL/AB/results")
    p.add_argument("--progress-every", type=int, default=10)
    args = p.parse_args()

    if args.ckpt is None:
        args.ckpt = (DEFAULT_MCTS_CKPT if args.policy == "mcts"
                     else DEFAULT_PPO_CKPT)
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    random.seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass
    hashseed = os.environ.get("PYTHONHASHSEED")
    if hashseed is None:
        print("[warn] PYTHONHASHSEED unset — run is NOT bit-reproducible "
              "(catanatron RNG + set order depend on it).")

    n = args.n_agents
    k = 4 - n
    policy = mcts_net = judge_net = None
    if args.policy == "mcts":
        import torch
        from models.alphazero.net import load_policy_value_net
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        mcts_net = load_policy_value_net(state, "cpu")
        if args.judge_ckpt:
            jstate = torch.load(args.judge_ckpt, map_location="cpu",
                                weights_only=False)
            judge_net = load_policy_value_net(jstate, "cpu")
    else:
        policy = build_policy(args.algo, ckpt,
                              deterministic=args.deterministic)
    enable_trades = not args.no_trades
    rotate = not args.no_rotate
    ab_fair = k / 4.0

    wins = {c: 0 for c in COLORS}
    ab_wins = 0          # games won by the AlphaBeta block (any of its seats)
    agent_wins = 0       # games won by one of the trained seats
    no_winner = 0
    ab_seat_wins = {c: 0 for c in COLORS}   # AB wins broken out by seat color
    ab_seat_games = {c: 0 for c in COLORS}
    trade_tot = {k_: 0 for k_ in _TRADE_KEYS}
    mcts_fallbacks = mcts_decisions = 0

    t0 = time.perf_counter()
    for g in range(args.games):
        game_seed = args.seed + g
        if rotate:
            ab_colors = {COLORS[(g + j) % 4] for j in range(k)}
        else:
            ab_colors = set(COLORS[:k])
        for c in ab_colors:
            ab_seat_games[c] += 1
        players, game_mcts = build_players(ab_colors, args, policy,
                                           mcts_net, game_seed,
                                           judge_net=judge_net)
        game = Game(players, seed=game_seed)
        winner = game.play()

        for key, v in tally_trades(game, ab_colors).items():
            trade_tot[key] += v
        for mp in game_mcts:
            mcts_fallbacks += mp.fallbacks
            mcts_decisions += mp.decisions

        if winner is None:
            no_winner += 1
        else:
            wins[winner] += 1
            if winner in ab_colors:
                ab_wins += 1
                ab_seat_wins[winner] += 1
            else:
                agent_wins += 1

        if (g + 1) % args.progress_every == 0:
            dec = (g + 1) - no_winner
            ab_rate = ab_wins / dec if dec else 0.0
            el = time.perf_counter() - t0
            print(f"[{g+1}/{args.games}] AB {ab_wins} wins "
                  f"({ab_rate:.3f} of {dec} decided, fair {ab_fair}), "
                  f"agents {agent_wins}  "
                  f"{el / (g + 1):.2f}s/game", flush=True)

    elapsed = time.perf_counter() - t0
    decided = args.games - no_winner

    ab_lo, ab_hi = wilson_ci(ab_wins, decided)
    ab_rate = ab_wins / decided if decided else 0.0
    ag_lo, ag_hi = wilson_ci(agent_wins, decided)
    ag_rate = agent_wins / decided if decided else 0.0
    # Per-trained-seat share = collective / N (N symmetric trained seats).
    per_agent = ag_rate / n

    result = {
        "experiment": f"{n}x{args.policy.upper()}_vs_{k}xAlphaBeta",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ckpt": str(ckpt),
        "policy": args.policy,
        "algo": args.algo if args.policy == "ppo" else None,
        "n_agents": n,
        "n_ab": k,
        "mcts_sims": args.mcts_sims if args.policy == "mcts" else None,
        "leaf_eval": args.leaf_eval if args.policy == "mcts" else None,
        "ab_value_scale": (args.ab_value_scale
                           if args.policy == "mcts" else None),
        "model_ab_depth": (args.model_ab_depth
                           if args.policy == "mcts" else None),
        "model_ab_prune": (args.model_ab_prune
                           if args.policy == "mcts" else None),
        "model_catanatron_chance": (args.model_catanatron_chance
                                    if args.policy == "mcts" else None),
        "model_opp": args.model_opp if args.policy == "mcts" else None,
        "judge_ckpt": (args.judge_ckpt or None) if args.policy == "mcts" else None,
        "trade_add_cap": (args.trade_add_cap
                          if args.policy == "mcts" else None),
        "mcts_fallbacks": mcts_fallbacks if args.policy == "mcts" else None,
        "mcts_decisions": mcts_decisions if args.policy == "mcts" else None,
        "deterministic": args.deterministic,
        "enable_trades": enable_trades,
        "max_offers_per_turn": args.max_offers_per_turn,
        "rotate_ab_seat": rotate,
        "ab_depth": args.ab_depth,
        "ab_prune": args.ab_prune,
        "seed": args.seed,
        "pythonhashseed": hashseed,
        "games": args.games,
        "decided": decided,
        "no_winner": no_winner,
        "ab_wins": ab_wins,
        "ab_win_rate": ab_rate,
        "ab_ci95": [ab_lo, ab_hi],
        "ab_fair_share": ab_fair,
        "agent_wins": agent_wins,
        "agent_win_rate": ag_rate,
        "agent_ci95": [ag_lo, ag_hi],
        "agent_per_seat_rate": per_agent,
        "per_seat_fair_share": FAIR_SHARE,
        "seat_wins": {c.name: wins[c] for c in COLORS},
        "ab_seat_wins": {c.name: ab_seat_wins[c] for c in COLORS},
        "ab_seat_games": {c.name: ab_seat_games[c] for c in COLORS},
        "trades_total": trade_tot,
        "trades_per_game": {k_: trade_tot[k_] / args.games if args.games
                            else 0.0 for k_ in _TRADE_KEYS},
        "elapsed_s": elapsed,
        "s_per_game": elapsed / args.games if args.games else 0.0,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"mixed_{n}{args.policy}_{k}ab_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\n=== mixed table: {n}x {args.policy.upper()} "
          f"(trades={'on' if enable_trades else 'off'}) "
          f"vs {k}x AlphaBeta(d{args.ab_depth}"
          f"{',prune' if args.ab_prune else ''}) ===")
    print(f"ckpt:            {ckpt}")
    print(f"games:           {decided}/{args.games} decided "
          f"(no-winner: {no_winner})")
    print(f"AlphaBeta block: {ab_wins} wins   rate {ab_rate:.4f}  "
          f"95% CI [{ab_lo:.4f}, {ab_hi:.4f}]   (fair share = {ab_fair})")
    print(f"agents (all {n}):  {agent_wins} wins   rate {ag_rate:.4f}  "
          f"95% CI [{ag_lo:.4f}, {ag_hi:.4f}]")
    print(f"agent per seat:  {per_agent:.4f}   (vs fair {FAIR_SHARE})")
    print(f"seat wins:       {result['seat_wins']}")
    if rotate:
        print("AB by seat:      "
              + ", ".join(f"{c.name}:{ab_seat_wins[c]}/{ab_seat_games[c]}"
                          for c in COLORS))
    if args.policy == "mcts" and mcts_decisions:
        print(f"mcts fallbacks:  {mcts_fallbacks}/{mcts_decisions} "
              f"({mcts_fallbacks / mcts_decisions:.4%})")
    g = args.games or 1
    confirms = trade_tot["confirms"]
    offers = trade_tot["offers"]
    acc_rate = confirms / offers if offers else 0.0
    print(f"--- trades (domestic p2p) ---")
    print(f"offers:          {offers}  ({offers / g:.2f}/game)")
    print(f"confirmed p2p:   {confirms}  ({confirms / g:.2f}/game)  "
          f"accept rate {acc_rate:.1%}")
    print(f"rejects:         {trade_tot['rejects']}   "
          f"AB-accepts: {trade_tot['ab_accepts']}   "
          f"cancels: {trade_tot['cancels']}")
    mar_agents = trade_tot["maritime"] - trade_tot["maritime_ab"]
    print(f"maritime(bank):  {trade_tot['maritime']} total  "
          f"agents {mar_agents} ({mar_agents / g:.2f}/game over {n} seats = "
          f"{mar_agents / g / n:.2f}/agent)  AB {trade_tot['maritime_ab']} "
          f"({trade_tot['maritime_ab'] / g:.2f}/game)")
    verdict = (f"AlphaBeta ABOVE fair share ({ab_fair}) — still strong "
               f"outnumbered" if ab_lo > ab_fair else
               f"AlphaBeta BELOW fair share ({ab_fair}) — trained table "
               f"suppresses it" if ab_hi < ab_fair else
               f"AlphaBeta ~ fair share (CI straddles {ab_fair})")
    print(f"verdict:         {verdict}")
    print(f"time:            {elapsed:.1f}s  ({result['s_per_game']:.2f}s/game)")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
