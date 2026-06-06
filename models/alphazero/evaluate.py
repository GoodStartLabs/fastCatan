"""Evaluate an AlphaZero checkpoint vs random opponents (pure fastcatan).

Seat 0 = AlphaZero (MCTS, greedy, no root noise); seats 1-3 = uniform-random legal.
This is the AZ analogue of models/eval.py's vs-random gate. It can't reuse that
file's pick(obs, mask) loop because MCTS needs the live env state, not just the obs.

vs-AlphaBeta goes through EVAL/bridge (catanatron engine) and needs the MCTS to run on
a mirrored fastcatan state via bridge/state_mirror.py — that's the M4 follow-up, not
wired here.

    python -m models.alphazero.evaluate --ckpt models/checkpoints/alphazero/az_final.pt \
        --games 200 --sims 100 --device cpu
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import fastcatan

from models.alphazero.net import PolicyValueNet, load_policy_value_net
from models.ckpt import verify_stamp
from models.alphazero.mcts import (
    MCTS, _unpack, filter_p2p, p2p_trade_mask, p2p_banned_words, MASK_WORDS,
)
from models.eval import wilson_ci

WIN_VP = 10
LEARNER_SEAT = 0


_NO_ACTION = 0xFFFFFFFF


def play_game(game_env, mcts: MCTS, rng, p2p_bool, opp_pick=None,
              sims_temp: float = 0.0, max_moves: int = 4000) -> int:
    """Seat 0 = AZ (MCTS). Opponent seats 1-3 use ``opp_pick`` (default random)."""
    game_env.reset(rng.getrandbits(64))
    mask_buf = np.zeros(MASK_WORDS, dtype=np.uint64)
    for _ in range(max_moves):
        game_env.action_mask(mask_buf)
        mask, legal = _unpack(mask_buf)
        if p2p_bool is not None:
            _, legal = filter_p2p(mask, p2p_bool)
        if not legal:
            break
        if game_env.current_player == LEARNER_SEAT and len(legal) > 1:
            action, _pi, _m = mcts.choose(
                game_env.snapshot(), temperature=sims_temp, add_root_noise=False)
        elif game_env.current_player == LEARNER_SEAT:
            action = legal[0]
        elif opp_pick is not None:
            action = opp_pick(game_env, game_env.current_player, legal)
        else:
            action = rng.choice(legal)
        _r, done = game_env.step(action)
        if done:
            break
    for p in range(fastcatan.NUM_PLAYERS):
        if game_env.player_vp(p) >= WIN_VP:
            return p
    return -1


def make_alphabeta_pick(rng, depth: int, prune: bool, banned=None):
    """Opponent picker using the native C++ AlphaBeta (Env.ab_decide).

    ``banned`` (uint64[MASK_WORDS], e.g. mcts.p2p_banned_words() when p2p
    trades are suppressed) keeps the WHOLE search inside the filtered action
    space, so the pick lands in our ``legal`` set and the random fallback
    below is a never-strand safety net instead of a hole the learner can farm
    (PPO trained vs the holey AB scored 0/500 on the bridge, 2026-06-03).

    Without ``banned`` (legacy): ab_decide sees the C++ full mask incl.
    trades; it returns 0xFFFFFFFF if it sees no move, in a cross-seat forced
    sub-phase its pick can fall outside the acting seat's set, and a trade id
    won't be in trade-filtered ``legal`` — all of which fell back to a
    uniform-random move (mirrors models/env.py._opponent_action)."""
    def pick(game_env, cp, legal):
        a = (game_env.ab_decide(cp, depth, prune, banned)
             if banned is not None else game_env.ab_decide(cp, depth, prune))
        return a if (a != _NO_ACTION and a in legal) else rng.choice(legal)
    return pick


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--allow-trades", action="store_true",
                   help="Allow p2p trades (default: suppressed, matching training "
                        "and the --no-trades AlphaBeta eval).")
    p.add_argument("--opponent", choices=["random", "alphabeta"], default="random",
                   help="Seats 1-3 policy. 'alphabeta' = native C++ Catanatron-AB "
                        "port (Env.ab_decide) — the decisive metric vs PPO's 0/200.")
    p.add_argument("--ab-depth", type=int, default=2,
                   help="AlphaBeta search depth (Catanatron default 2; 1 is ~free).")
    p.add_argument("--ab-prune", action="store_true",
                   help="Use AlphaBeta action pruning (default off = full legal set, "
                        "matching Catanatron AlphaBetaPlayer).")
    p.add_argument("--leaf-eval", choices=["net", "ab_value"], default="net",
                   help="MCTSvsFixed leaf evaluation: 'ab_value' = hybrid "
                        "(net prior + deterministic native heuristic leaves; "
                        "attacks leaf-noise saturation).")
    p.add_argument("--ab-value-scale", type=float, default=30.0)
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    verify_stamp(ckpt, strict=False)
    state = torch.load(str(ckpt), map_location=args.device, weights_only=False)
    net = load_policy_value_net(state, args.device)

    suppress = not args.allow_trades
    p2p_bool = p2p_trade_mask() if suppress else None
    if args.opponent == "alphabeta":
        # AB-in-tree search (lazy import: mcts_vs_fixed imports this module).
        from models.alphazero.mcts_vs_fixed import MCTSvsFixed
        mcts = MCTSvsFixed(net, device=args.device, sims=args.sims, c_puct=args.c_puct,
                           dirichlet_frac=0.0, seed=args.seed, suppress_p2p=suppress,
                           ab_depth=args.ab_depth, ab_prune=args.ab_prune,
                           leaf_eval=args.leaf_eval,
                           ab_value_scale=args.ab_value_scale)
    else:
        mcts = MCTS(net, device=args.device, sims=args.sims, c_puct=args.c_puct,
                    dirichlet_frac=0.0, seed=args.seed, suppress_p2p=suppress)
    game_env = fastcatan.Env()
    import random
    rng = random.Random(args.seed)

    opp_pick = None
    if args.opponent == "alphabeta":
        opp_pick = make_alphabeta_pick(
            rng, args.ab_depth, args.ab_prune,
            banned=p2p_banned_words() if suppress else None)

    wins = 0
    seat_wins = [0, 0, 0, 0]
    no_winner = 0
    for g in range(args.games):
        w = play_game(game_env, mcts, rng, p2p_bool, opp_pick=opp_pick)
        if w < 0:
            no_winner += 1
            continue
        seat_wins[w] += 1
        if w == LEARNER_SEAT:
            wins += 1
        if (g + 1) % 20 == 0:
            print(f"[{g+1}/{args.games}] wins {wins} ({wins/(g+1):.3f})", flush=True)

    n = args.games - no_winner
    lo, hi = wilson_ci(wins, n)
    rate = wins / n if n else 0.0
    opp_desc = (f"AlphaBeta(d={args.ab_depth},prune={args.ab_prune})"
                if args.opponent == "alphabeta" else "random")
    print(f"\n=== AlphaZero vs {opp_desc} ===")
    print(f"ckpt: {ckpt}  sims: {args.sims}")
    print(f"games (winnered): {n}/{args.games} (no-winner: {no_winner})")
    print(f"win rate: {rate:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"seat distribution: {seat_wins}")


if __name__ == "__main__":
    main()
