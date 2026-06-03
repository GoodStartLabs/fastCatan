"""Parallel AlphaZero trainer (generational self-play) for fastCatan.

Self-play is the bottleneck and is embarrassingly parallel, so this runs W worker
processes that each play games with a snapshot of the current net (CPU, 1 torch
thread each to avoid oversubscription), then the main process trains on GPU and
broadcasts updated weights for the next generation.

This is the "validate learning" entry point: it periodically evaluates vs random so
you can watch the win-rate curve climb (the question is whether AZ learns on this env
at all before spending real compute). vs-AlphaBeta eval is a separate follow-up.

    python -m models.alphazero.train --total-games 2048 --workers 8 \
        --games-per-worker 4 --sims 64 --device cuda
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from models.ckpt import write_stamp, verify_stamp

CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def _worker(payload: dict) -> dict:
    """Play payload['n_games'] self-play games with the given net weights (CPU)."""
    import torch
    torch.set_num_threads(1)
    import numpy as np
    import random
    import fastcatan
    from models.alphazero.net import PolicyValueNet
    from models.alphazero.mcts import MCTS, p2p_trade_mask
    from models.alphazero.selfplay import play_one_game
    from models.alphazero.evaluate import make_alphabeta_pick

    net = PolicyValueNet()
    net.load_state_dict(payload["weights"])
    net.eval()
    suppress = payload["suppress"]
    value_mode = payload["value_mode"]
    mode = payload["opponent"]

    game_env = fastcatan.Env()
    p2p = p2p_trade_mask() if suppress else None
    seed_seq = random.Random(payload["seed"] ^ 0x5EED)
    opp_rng = random.Random(payload["seed"] ^ 0xABABAB)

    if mode == "self":
        mcts = MCTS(net, device="cpu", sims=payload["sims"], c_puct=payload["c_puct"],
                    seed=payload["seed"], suppress_p2p=suppress)
    else:  # alphabeta / league -> single-agent search vs fixed or per-seat opponents
        from models.alphazero.mcts_vs_fixed import MCTSvsFixed
        mcts = MCTSvsFixed(net, device="cpu", sims=payload["sims"],
                           c_puct=payload["c_puct"], seed=payload["seed"],
                           suppress_p2p=suppress, value_mode=value_mode,
                           ab_depth=payload.get("ab_depth", 1),
                           ab_prune=payload.get("ab_prune", False))

    members = weights_pfsp = fixed_opp = None
    if mode == "league":
        from models.alphazero.league import build_members, sample_seat_assignment
        members = build_members(payload, opp_rng)
        weights_pfsp = payload["weights_pfsp"]
    elif mode == "alphabeta":
        fixed_opp = make_alphabeta_pick(opp_rng, payload["ab_depth"], payload["ab_prune"])

    obs_l, pi_l, mask_l, z_l = [], [], [], []
    winners, decisions, tables = [], [], []
    for _ in range(payload["n_games"]):
        if mode == "league":
            dispatch, table = sample_seat_assignment(members, weights_pfsp, opp_rng)
            mcts.opp = dispatch          # tree models each seat's actual opponent
            opp_pick = dispatch
        else:
            opp_pick, table = fixed_opp, None
        recs, winner, dmoves = play_one_game(
            game_env, mcts, seed_seq.getrandbits(64), payload["temp_moves"], p2p,
            opp_pick=opp_pick, value_mode=value_mode)
        winners.append(winner)
        decisions.append(dmoves)
        if table is not None:
            tables.append((tuple(sorted(table)), winner == 0))
        for s in recs:
            obs_l.append(s.obs); pi_l.append(s.pi); mask_l.append(s.mask); z_l.append(s.z)

    if not obs_l:
        return {"obs": None, "winners": winners, "decisions": decisions, "tables": tables}
    return {
        "obs": np.stack(obs_l), "pi": np.stack(pi_l),
        "mask": np.stack(mask_l), "z": np.asarray(z_l, dtype=np.float32),
        "winners": winners, "decisions": decisions, "tables": tables,
    }


def _evaluate(net, sims, c_puct, games, seed, suppress, opponent="random",
              ab_depth=1, ab_prune=False, device="cpu") -> float:
    """Win rate for the learner at seat 0 (greedy, no root noise) vs `opponent`."""
    import fastcatan
    from models.alphazero.mcts import MCTS, p2p_trade_mask
    from models.alphazero.evaluate import play_game, make_alphabeta_pick

    if opponent == "alphabeta":
        from models.alphazero.mcts_vs_fixed import MCTSvsFixed
        mcts = MCTSvsFixed(net, device=device, sims=sims, c_puct=c_puct,
                           dirichlet_frac=0.0, seed=seed, suppress_p2p=suppress,
                           ab_depth=ab_depth, ab_prune=ab_prune)
    else:
        mcts = MCTS(net, device=device, sims=sims, c_puct=c_puct,
                    dirichlet_frac=0.0, seed=seed, suppress_p2p=suppress)
    game_env = fastcatan.Env()
    p2p = p2p_trade_mask() if suppress else None
    rng = random.Random(seed)
    opp_pick = (make_alphabeta_pick(rng, ab_depth, ab_prune)
                if opponent == "alphabeta" else None)
    wins = winnered = 0
    for _ in range(games):
        w = play_game(game_env, mcts, rng, p2p, opp_pick=opp_pick)
        if w < 0:
            continue
        winnered += 1
        if w == 0:
            wins += 1
    return wins / winnered if winnered else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-games", type=int, default=2048)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--games-per-worker", type=int, default=4)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--temp-moves", type=int, default=20)
    p.add_argument("--buffer-size", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-steps-per-iter", type=int, default=40)
    p.add_argument("--min-buffer", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--value-coef", type=float, default=1.0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default=str(CKPT_DIR / "alphazero"))
    p.add_argument("--eval-every", type=int, default=8, help="iterations between evals")
    p.add_argument("--eval-games", type=int, default=60)
    p.add_argument("--eval-sims", type=int, default=64)
    p.add_argument("--eval-ab-games", type=int, default=0,
                   help="In self-play mode, also eval vs AlphaBeta(d1) each eval point "
                        "(N games) to track whether self-play strength transfers.")
    p.add_argument("--allow-trades", action="store_true")
    p.add_argument("--opponent", choices=["self", "alphabeta", "league"], default="self",
                   help="Non-learner seats: 'self' (self-play, record all seats), "
                        "'alphabeta' (seats 1-3 = native AB), or 'league' (PFSP pool of "
                        "random + AB(d1,d2) + past-self snapshots — the no-win-collapse fix).")
    p.add_argument("--ab-depth", type=int, default=1,
                   help="AlphaBeta opponent depth for --opponent alphabeta.")
    p.add_argument("--ab-prune", action="store_true")
    p.add_argument("--snapshot-every", type=int, default=4,
                   help="League: snapshot the current net into the pool every N gens.")
    p.add_argument("--max-snapshots", type=int, default=6,
                   help="League: cap on past-self snapshot members (drop oldest).")
    p.add_argument("--value-mode", choices=["sparse", "vp_margin"], default="sparse",
                   help="Terminal value target. vp_margin is dense (recommended vs a "
                        "dominant AB opponent, where sparse +-1 saturates).")
    p.add_argument("--init-from", type=str, default="",
                   help="Warm-start net weights from this checkpoint (.pt).")
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    from models.alphazero.net import PolicyValueNet, masked_log_softmax
    import torch.nn.functional as F

    suppress = not args.allow_trades
    net = PolicyValueNet().to(args.device)
    if args.init_from:
        verify_stamp(args.init_from, strict=False)
        state = torch.load(args.init_from, map_location=args.device, weights_only=False)
        net.load_state_dict(state["net_state"])
        print(f"[init] warm-started from {args.init_from}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    buffer: deque = deque(maxlen=args.buffer_size)

    games_per_iter = args.workers * args.games_per_worker
    iterations = math.ceil(args.total_games / games_per_iter)
    ctx = mp.get_context("spawn")

    # League state (only used when --opponent league).
    league_stats = snapshots = None
    if args.opponent == "league":
        from models.alphazero.league import LeagueStats, pfsp_weights
        league_stats = LeagueStats()
        snapshots = []  # list of (name, cpu_state_dict)

    opp_label = {"alphabeta": f"alphabeta(d={args.ab_depth})",
                 "league": "league(rand+ab1+ab2+self)"}.get(args.opponent, "self")
    print(f"[cfg] {iterations} iters x {games_per_iter} games "
          f"({args.workers}w x {args.games_per_worker}) sims={args.sims} "
          f"device={args.device} trades={'on' if not suppress else 'off'} "
          f"opp={opp_label} value={args.value_mode}", flush=True)

    games_done = 0
    t0 = time.time()
    with ctx.Pool(args.workers) as pool:
        for it in range(1, iterations + 1):
            weights = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            extra = {}
            if args.opponent == "league":
                names = ["random", "ab_d1", "ab_d2"] + [n for n, _ in snapshots]
                extra = {
                    "weights_pfsp": pfsp_weights(league_stats.winrates(names)),
                    "snapshots": snapshots, "ab_depths": [1, 2],
                    "include_random": True,
                }
            payloads = [{
                "weights": weights, "sims": args.sims, "c_puct": args.c_puct,
                "temp_moves": args.temp_moves, "suppress": suppress,
                "n_games": args.games_per_worker,
                "opponent": args.opponent, "ab_depth": args.ab_depth,
                "ab_prune": args.ab_prune, "value_mode": args.value_mode,
                "seed": (args.seed * 1_000_003 + it * 9176 + w),
                **extra,
            } for w in range(args.workers)]

            tg = time.time()
            results = pool.map(_worker, payloads)
            gen_s = time.time() - tg

            winners, decisions = [], []
            for r in results:
                winners += r["winners"]
                decisions += r["decisions"]
                if r["obs"] is None:
                    continue
                for i in range(r["obs"].shape[0]):
                    buffer.append((r["obs"][i], r["pi"][i], r["mask"][i], float(r["z"][i])))
            games_done += games_per_iter

            if args.opponent == "league":
                for r in results:
                    for table_names, won in r.get("tables", []):
                        league_stats.update(table_names, won)
                if it % args.snapshot_every == 0:
                    snap_sd = {k: v.detach().cpu().clone()
                               for k, v in net.state_dict().items()}
                    snapshots.append((f"snap_g{it}", snap_sd))
                    if len(snapshots) > args.max_snapshots:
                        snapshots.pop(0)

            # train
            stats = {"loss": float("nan"), "policy": float("nan"), "value": float("nan")}
            if len(buffer) >= args.min_buffer:
                net.train()
                for _ in range(args.train_steps_per_iter):
                    batch = random.sample(buffer, min(args.batch_size, len(buffer)))
                    obs = torch.from_numpy(np.stack([b[0] for b in batch])).to(args.device)
                    pi = torch.from_numpy(np.stack([b[1] for b in batch])).to(args.device)
                    mask = torch.from_numpy(np.stack([b[2] for b in batch])).to(args.device)
                    z = torch.tensor([b[3] for b in batch], dtype=torch.float32,
                                     device=args.device)
                    logits, value = net(obs)
                    logp = masked_log_softmax(logits, mask)
                    policy_loss = -(pi * logp).sum(dim=1).mean()
                    value_loss = F.mse_loss(value, z)
                    loss = policy_loss + args.value_coef * value_loss
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                    stats = {"loss": float(loss), "policy": float(policy_loss),
                             "value": float(value_loss)}

            n_win = sum(1 for w in winners if w >= 0)
            fps = games_done / (time.time() - t0)
            print(f"[it {it:>3d}/{iterations}] games={games_done} buf={len(buffer)} "
                  f"gen={gen_s:.1f}s won={n_win}/{len(winners)} "
                  f"dec~{int(np.mean(decisions))} loss={stats['loss']:.3f} "
                  f"p={stats['policy']:.3f} v={stats['value']:.3f} "
                  f"({fps:.2f} g/s)", flush=True)

            if it % args.eval_every == 0 or it == iterations:
                net.eval()
                eval_net = PolicyValueNet()
                eval_net.load_state_dict({k: v.detach().cpu()
                                          for k, v in net.state_dict().items()})
                eval_net.eval()
                ab_games = args.eval_ab_games or args.eval_games

                if args.opponent == "league":
                    wr_rand = _evaluate(eval_net, args.eval_sims, args.c_puct,
                                        args.eval_games, args.seed + 777, suppress,
                                        opponent="random")
                    wr = _evaluate(eval_net, args.eval_sims, args.c_puct, ab_games,
                                   args.seed + 778, suppress, opponent="alphabeta",
                                   ab_depth=1)
                    wr2 = _evaluate(eval_net, args.eval_sims, args.c_puct, ab_games,
                                    args.seed + 779, suppress, opponent="alphabeta",
                                    ab_depth=2)
                    print(f"   [eval] vs-random {wr_rand:.3f} | vs-AB(d1) {wr:.3f} | "
                          f"vs-AB(d2) {wr2:.3f}  (sims={args.eval_sims})", flush=True)
                    names = ["random", "ab_d1", "ab_d2"] + [n for n, _ in snapshots]
                    wrs = league_stats.winrates(names)
                    print("   [pool] " + " ".join(f"{n}={wrs[n]:.2f}" for n in names),
                          flush=True)
                else:
                    wr = _evaluate(eval_net, args.eval_sims, args.c_puct,
                                   args.eval_games, args.seed + 777, suppress,
                                   opponent=args.opponent, ab_depth=args.ab_depth,
                                   ab_prune=args.ab_prune)
                    print(f"   [eval] vs-{opp_label} win rate {wr:.3f} "
                          f"({args.eval_games} games, sims={args.eval_sims})", flush=True)
                    if args.opponent != "alphabeta" and args.eval_ab_games > 0:
                        wr_ab = _evaluate(eval_net, args.eval_sims, args.c_puct,
                                          args.eval_ab_games, args.seed + 778, suppress,
                                          opponent="alphabeta", ab_depth=1)
                        print(f"   [eval] vs-alphabeta(d=1) win rate {wr_ab:.3f} "
                              f"({args.eval_ab_games} games, sims={args.eval_sims})",
                              flush=True)
                ck = save_dir / f"az_it{it}.pt"
                torch.save({"net_state": net.state_dict(), "args": vars(args),
                            "iter": it, "vs_ab_d1": wr}, str(ck))
                write_stamp(ck)

    final = save_dir / "az_final.pt"
    torch.save({"net_state": net.state_dict(), "args": vars(args)}, str(final))
    write_stamp(final)
    print(f"[done] {games_done} games in {time.time()-t0:.0f}s -> {final}", flush=True)


if __name__ == "__main__":
    main()
