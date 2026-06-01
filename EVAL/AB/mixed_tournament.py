"""Mixed-table experiment: 3 trade-capable PPO bridges vs 1 Alpha-Beta.

The standard thesis run (`AB.tournament`) seats ONE trained agent against THREE
AlphaBeta bots and asks "can the agent beat AlphaBeta?". This driver inverts the
table: THREE copies of a trained fastcatan policy (default: the 200M self-play
league checkpoint, trade-compose enabled so the agents can deal with each other)
share a table with a SINGLE `AlphaBetaPlayer`. The question becomes "outnumbered
3-to-1, how does AlphaBeta fare against a table of trained traders, and how does
that move the win rate off the 0.25 fair-share baseline?".

The three PPO seats all run the *same* loaded model object (one
`build_policy(...)` call, shared closure). Catanatron drives the game
single-threaded so the shared model is queried serially — safe. Sampling
(`deterministic=False`, the validated eval mode) makes the three otherwise
identical agents diverge in play.

Seat fairness: the lone AlphaBeta is rotated across all four colors over the
game series (`--rotate`, default on) so AlphaBeta and the PPO agents each occupy
every seat colour an equal number of times. Catanatron additionally shuffles
turn order per game, so neither colour nor turn position biases the aggregate.

Usage (from repo root, anaconda env — see AB/REPRODUCIBILITY.md):

    PYTHONHASHSEED=0 PYTHONPATH=EVAL python -m AB.mixed_tournament \
        --games 200 --ab-depth 2 --ab-prune --seed 42

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

# 200M self-play league checkpoint (1084/286). The "3 trained models" are 3
# instances of this one agent — it is the only 200M checkpoint in the repo.
DEFAULT_CKPT = "models/checkpoints/sp_league_200m_512/selfplay_final.zip"
FAIR_SHARE = 0.25  # 4-player chance baseline (1 of 4 seats)


def build_players(ab_color: Color, policy, enable_trades: bool,
                  ab_depth: int, ab_prune: bool, game_seed: int):
    """One AlphaBetaPlayer at `ab_color`; the other three seats are PPO bridges
    sharing `policy`. Each bridge gets a distinct RNG seed so their fallback /
    compose tie-breaks do not lock-step."""
    players = []
    for i, c in enumerate(COLORS):
        if c == ab_color:
            players.append(TradeAwareAlphaBeta(c, depth=ab_depth, prunning=ab_prune))
        else:
            players.append(CatanatronBridge(
                c, policy=policy, seed=game_seed * 4 + i,
                enable_trades=enable_trades))
    return players


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


def tally_trades(game, ab_color) -> dict:
    """Count trade actions in a finished game's action log. Offers/confirms are
    inherently all-PPO (AlphaBeta never offers); `maritime_ab` splits out the
    lone AB's bank trades so the rest is the PPO seats'. `ab_accepts` = times AB
    accepted a domestic offer."""
    out = {k: 0 for k in _TRADE_KEYS}
    for rec in game.state.action_records:
        a = rec.action
        key = _TRADE_COUNT_TYPES.get(a.action_type)
        if key is None:
            continue
        out[key] += 1
        if a.color == ab_color:
            if a.action_type == ActionType.ACCEPT_TRADE:
                out["ab_accepts"] += 1
            elif a.action_type == ActionType.MARITIME_TRADE:
                out["maritime_ab"] += 1
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                   help="checkpoint for the 3 PPO bridge seats")
    p.add_argument("--algo", default="ppo", choices=["ppo"])
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ab-depth", type=int, default=2)
    p.add_argument("--ab-prune", action="store_true",
                   help="enable AlphaBeta action pruning (recommended)")
    p.add_argument("--deterministic", action="store_true",
                   help="argmax policy (default: sample; see AB/policy.py)")
    p.add_argument("--no-trades", action="store_true",
                   help="disable the bridge OFFER_TRADE compose loop "
                        "(default: trades ON — the point of this experiment)")
    p.add_argument("--no-rotate", action="store_true",
                   help="pin AlphaBeta to RED instead of rotating seats")
    p.add_argument("--out", type=str, default="EVAL/AB/results")
    p.add_argument("--progress-every", type=int, default=10)
    args = p.parse_args()

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

    policy = build_policy(args.algo, ckpt, deterministic=args.deterministic)
    enable_trades = not args.no_trades
    rotate = not args.no_rotate

    wins = {c: 0 for c in COLORS}
    ab_wins = 0          # games won by the lone AlphaBeta (any rotated seat)
    ppo_wins = 0         # games won by one of the 3 PPO seats
    no_winner = 0
    ab_seat_wins = {c: 0 for c in COLORS}   # AB win rate broken out by its seat
    ab_seat_games = {c: 0 for c in COLORS}
    trade_tot = {k: 0 for k in _TRADE_KEYS}

    t0 = time.perf_counter()
    for g in range(args.games):
        game_seed = args.seed + g
        ab_color = COLORS[g % 4] if rotate else Color.RED
        ab_seat_games[ab_color] += 1
        players = build_players(ab_color, policy, enable_trades,
                                args.ab_depth, args.ab_prune, game_seed)
        game = Game(players, seed=game_seed)
        winner = game.play()

        for k, v in tally_trades(game, ab_color).items():
            trade_tot[k] += v

        if winner is None:
            no_winner += 1
        else:
            wins[winner] += 1
            if winner == ab_color:
                ab_wins += 1
                ab_seat_wins[ab_color] += 1
            else:
                ppo_wins += 1

        if (g + 1) % args.progress_every == 0:
            dec = (g + 1) - no_winner
            ab_rate = ab_wins / dec if dec else 0.0
            el = time.perf_counter() - t0
            print(f"[{g+1}/{args.games}] AB {ab_wins} wins "
                  f"({ab_rate:.3f} of {dec} decided), PPO {ppo_wins}  "
                  f"{el / (g + 1):.2f}s/game")

    elapsed = time.perf_counter() - t0
    decided = args.games - no_winner

    ab_lo, ab_hi = wilson_ci(ab_wins, decided)
    ab_rate = ab_wins / decided if decided else 0.0
    ppo_lo, ppo_hi = wilson_ci(ppo_wins, decided)
    ppo_rate = ppo_wins / decided if decided else 0.0
    # Per-PPO-seat share = PPO collective / 3 (3 symmetric trained seats).
    ppo_per_agent = ppo_rate / 3.0

    result = {
        "experiment": "3xPPO_vs_1xAlphaBeta",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ckpt": str(ckpt),
        "algo": args.algo,
        "deterministic": args.deterministic,
        "enable_trades": enable_trades,
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
        "ppo_wins": ppo_wins,
        "ppo_win_rate": ppo_rate,
        "ppo_ci95": [ppo_lo, ppo_hi],
        "ppo_per_agent_rate": ppo_per_agent,
        "fair_share": FAIR_SHARE,
        "seat_wins": {c.name: wins[c] for c in COLORS},
        "ab_seat_wins": {c.name: ab_seat_wins[c] for c in COLORS},
        "ab_seat_games": {c.name: ab_seat_games[c] for c in COLORS},
        "trades_total": trade_tot,
        "trades_per_game": {k: trade_tot[k] / args.games if args.games else 0.0
                            for k in _TRADE_KEYS},
        "elapsed_s": elapsed,
        "s_per_game": elapsed / args.games if args.games else 0.0,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"mixed_3ppo_1ab_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\n=== mixed table: 3x PPO (trades={'on' if enable_trades else 'off'}) "
          f"vs 1x AlphaBeta(d{args.ab_depth}"
          f"{',prune' if args.ab_prune else ''}) ===")
    print(f"ckpt:           {ckpt}")
    print(f"games:          {decided}/{args.games} decided (no-winner: {no_winner})")
    print(f"AlphaBeta:      {ab_wins} wins   rate {ab_rate:.4f}  "
          f"95% CI [{ab_lo:.4f}, {ab_hi:.4f}]   (fair share = {FAIR_SHARE})")
    print(f"PPO (all 3):    {ppo_wins} wins   rate {ppo_rate:.4f}  "
          f"95% CI [{ppo_lo:.4f}, {ppo_hi:.4f}]")
    print(f"PPO per agent:  {ppo_per_agent:.4f}   (vs fair {FAIR_SHARE})")
    print(f"seat wins:      {result['seat_wins']}")
    if rotate:
        print(f"AB by seat:     "
              + ", ".join(f"{c.name}:{ab_seat_wins[c]}/{ab_seat_games[c]}"
                          for c in COLORS))
    g = args.games or 1
    confirms = trade_tot["confirms"]
    offers = trade_tot["offers"]
    acc_rate = confirms / offers if offers else 0.0
    print(f"--- trades (domestic p2p) ---")
    print(f"offers:         {offers}  ({offers / g:.2f}/game)")
    print(f"confirmed p2p:  {confirms}  ({confirms / g:.2f}/game)  "
          f"accept rate {acc_rate:.1%}")
    print(f"rejects:        {trade_tot['rejects']}   "
          f"AB-accepts: {trade_tot['ab_accepts']}   "
          f"cancels: {trade_tot['cancels']}")
    mar_ppo = trade_tot["maritime"] - trade_tot["maritime_ab"]
    print(f"maritime(bank): {trade_tot['maritime']} total  "
          f"PPO {mar_ppo} ({mar_ppo / g:.2f}/game over 3 seats = "
          f"{mar_ppo / g / 3:.2f}/agent)  AB {trade_tot['maritime_ab']} "
          f"({trade_tot['maritime_ab'] / g:.2f}/game)")
    verdict = ("AlphaBeta ABOVE fair share — still strong outnumbered"
               if ab_lo > FAIR_SHARE else
               "AlphaBeta BELOW fair share — trained table suppresses it"
               if ab_hi < FAIR_SHARE else
               "AlphaBeta ~ fair share (CI straddles 0.25)")
    print(f"verdict:        {verdict}")
    print(f"time:           {elapsed:.1f}s  ({result['s_per_game']:.2f}s/game)")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
