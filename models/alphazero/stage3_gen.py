"""Stage-3 STRONG-PLAY generator: record the hybrid agent's own games + its
MCTS root search-value as the value target (de-cat [[learned-judge-distillation]]).

Stages 1+2 left the fully self-contained agent saturated at ~17% vs AB-d2: 4x
data lifted the clone (top-1 0.84) but NOT wins, because the value head's POV
target — the OUTCOME of *AB-vs-AB* play — is the binding constraint, not the
prior. Stage 3 attacks the target directly: generate games with our STRONGEST
agent (seat 0 = the 29.5% hybrid: net prior + native ab_value leaves + 512-sim
search, in-tree AB-d2 opponent) against real AB-d2, and record for each seat-0
decision the search's backed-up ROOT VALUE (mean over all sims of the leaf
values that flowed back to the root). That root value is a denoised,
lookahead-aggregated estimate — the canonical AlphaZero value target, strictly
richer than a single sparse outcome — and it is recorded under POV-purity (the
target uses ab_value's full state at TRAIN time only; the head still reads the
POV obs at inference). The hybrid's chosen move is kept as a strong policy label.

    obs   float16 (OBS_SIZE,)  seat-0 POV at the decision
    act   uint16               the hybrid's chosen action (visit-argmax)
    mask  uint8  (MASK_BYTES,)  packed legal mask (np.packbits, p2p-filtered)
    z     float16              sparse +-1 outcome for seat 0 (fallback target)
    vps   uint8  (4,)          final VPs (lets the trainer recompute vp_margin)
    seat  uint8               always 0 (only the hybrid seat is recorded)
    rootv float32             MCTS root search-value in [-1,1] (the stage-3 target)

Shards are compressed .npz, ~250 games each. Workers are spawn processes with
the net on CPU (single-thread) + their own Env; everything is nice'd so a
concurrent GPU run keeps priority. Consume with
`il_pretrain --value-target search_value --init-from <640k ckpt>`.

Run:
    python -m models.alphazero.stage3_gen --games 15000 --workers 20 \
        --sims 512 --ab-depth 2 --ckpt models/checkpoints/il_ab_d2_640k_vpm_ep10/il_final.pt \
        --out-dir models/datasets/stage3_hybrid_s512
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

MASK_BYTES = 36  # ceil(286/8)
WIN_VP = 10


def _play_games_worker(payload: dict) -> dict:
    if payload.get("nice", 0):
        os.nice(payload["nice"])
    import random

    import torch
    torch.set_num_threads(1)
    import fastcatan as fc
    from models.alphazero.net import load_policy_value_net
    from models.alphazero.mcts_vs_fixed import MCTSvsFixed
    from models.alphazero.mcts import (
        _unpack, filter_p2p, p2p_trade_mask, p2p_banned_words, MASK_WORDS,
    )
    from models.alphazero.evaluate import make_alphabeta_pick

    depth = payload["ab_depth"]
    sims = payload["sims"]
    scale = payload["ab_value_scale"]
    state = torch.load(payload["ckpt"], map_location="cpu", weights_only=False)
    net = load_policy_value_net(state, "cpu")

    rng = random.Random(payload["seed"])
    seed_seq = random.Random(payload["seed"] ^ 0x5EED)
    p2p = p2p_trade_mask()
    banned = p2p_banned_words()

    mcts = MCTSvsFixed(net, device="cpu", sims=sims, c_puct=1.5,
                       dirichlet_frac=0.0, seed=payload["seed"] ^ 0xA11CE,
                       suppress_p2p=True, ab_depth=depth, ab_prune=False,
                       leaf_eval="ab_value", ab_value_scale=scale,
                       opp_model="alphabeta")
    opp = make_alphabeta_pick(rng, depth, False, banned=banned)

    env = fc.Env()
    mask_buf = np.zeros(MASK_WORDS, dtype=np.uint64)
    obs_buf = np.zeros(fc.OBS_SIZE, dtype=np.float32)

    obs_l, act_l, mask_l, z_l, vps_l, seat_l, rootv_l = ([] for _ in range(7))
    decisions = 0
    winners = []

    # Optional per-game wall-clock watchdog. Some seat-0 MCTS searches enter a
    # non-terminating loop on rare game states (normal games finish in seconds);
    # such a game pins a worker at 100% CPU forever and stalls the whole shard.
    # When STAGE3_GAME_TIMEOUT_S>0 we abandon any game that exceeds it, roll back
    # its partial seat-0 records, and continue. Default (unset/0) = original path.
    import signal as _signal
    _game_timeout = int(os.environ.get("STAGE3_GAME_TIMEOUT_S", "0"))

    class _GameTimeout(Exception):
        pass

    def _on_alarm(_signum, _frame):
        raise _GameTimeout()

    if _game_timeout > 0:
        _signal.signal(_signal.SIGALRM, _on_alarm)
    _skipped = 0

    for _gi in range(payload["n_games"]):
        env.reset(seed_seq.getrandbits(64))
        recs: list[int] = []   # sample indices for this game (all seat 0)
        _pre = (len(obs_l), len(act_l), len(mask_l), len(rootv_l))
        if _game_timeout > 0:
            _signal.alarm(_game_timeout)
        try:
            for _ply in range(40000):
                env.action_mask(mask_buf)
                mask, legal = _unpack(mask_buf)
                mask, legal = filter_p2p(mask, p2p)
                if not legal:
                    break
                cp = env.current_player
                if cp != 0:
                    a = opp(env, cp, legal)
                    _r, done = env.step(a)
                    if done:
                        break
                    continue
                if len(legal) == 1:
                    _r, done = env.step(legal[0])
                    if done:
                        break
                    continue
                # seat-0 hybrid decision: search, record obs + chosen move + root value.
                env.write_obs(0, obs_buf)
                action, _pi, _m = mcts.choose(env.snapshot(), temperature=0.0,
                                              add_root_noise=False)
                obs_l.append(obs_buf.astype(np.float16))
                act_l.append(np.uint16(action))
                mask_l.append(np.packbits(mask))
                rootv_l.append(np.float32(mcts.last_root_value))
                recs.append(len(obs_l) - 1)
                decisions += 1
                _r, done = env.step(int(action))
                if done:
                    break
        except _GameTimeout:
            del obs_l[_pre[0]:]
            del act_l[_pre[1]:]
            del mask_l[_pre[2]:]
            del rootv_l[_pre[3]:]
            decisions -= len(recs)
            _skipped += 1
            # Rebuild the search from clean state: the interrupt unwound out of
            # mcts.choose mid-simulation, so discard any partial tree to keep
            # subsequent games' recorded root values trustworthy. (net is reused.)
            mcts = MCTSvsFixed(net, device="cpu", sims=sims, c_puct=1.5,
                               dirichlet_frac=0.0, seed=payload["seed"] ^ 0xA11CE,
                               suppress_p2p=True, ab_depth=depth, ab_prune=False,
                               leaf_eval="ab_value", ab_value_scale=scale,
                               opp_model="alphabeta")
            print(f"[stage3-watchdog] shard {payload['shard_id']} game {_gi} "
                  f"exceeded {_game_timeout}s (non-terminating); abandoned, "
                  f"{_skipped} skipped so far", flush=True)
            continue
        finally:
            if _game_timeout > 0:
                _signal.alarm(0)

        vps = np.array([env.player_vp(p) for p in range(4)], dtype=np.uint8)
        winner = next((p for p in range(4) if vps[p] >= WIN_VP), -1)
        winners.append(winner)
        for _idx in recs:
            z_l.append(np.float16(1.0 if winner == 0 else -1.0))
            vps_l.append(vps)
            seat_l.append(np.uint8(0))

    shard = Path(payload["out_dir"]) / f"shard_{payload['shard_id']:05d}.npz"
    np.savez_compressed(
        shard,
        obs=np.stack(obs_l) if obs_l else np.zeros((0, fc.OBS_SIZE), np.float16),
        act=np.asarray(act_l, dtype=np.uint16),
        mask=np.stack(mask_l) if mask_l else np.zeros((0, MASK_BYTES), np.uint8),
        z=np.asarray(z_l, dtype=np.float16),
        vps=np.stack(vps_l) if vps_l else np.zeros((0, 4), np.uint8),
        seat=np.asarray(seat_l, dtype=np.uint8),
        rootv=np.asarray(rootv_l, dtype=np.float32),
    )
    n_won = sum(1 for w in winners if w == 0)
    return {"shard": str(shard), "games": len(winners),
            "decisions": decisions, "seat0_won": n_won}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=15000)
    p.add_argument("--games-per-shard", type=int, default=250)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--sims", type=int, default=512)
    p.add_argument("--ab-depth", type=int, default=2,
                   help="in-tree AND table AlphaBeta depth (the real "
                        "opponents are AB-d2; in-tree d2 = faithful model).")
    p.add_argument("--ckpt", type=str, required=True,
                   help="net providing the hybrid's prior (e.g. the 640k clone).")
    p.add_argument("--ab-value-scale", type=float, default=86e6)
    p.add_argument("--out-dir", type=str,
                   default="models/datasets/stage3_hybrid_s512")
    p.add_argument("--seed", type=int, default=20260608)
    p.add_argument("--nice", type=int, default=10)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_shards = (args.games + args.games_per_shard - 1) // args.games_per_shard
    payloads = [{
        "n_games": min(args.games_per_shard,
                       args.games - i * args.games_per_shard),
        "shard_id": i, "out_dir": str(out), "sims": args.sims,
        "ab_depth": args.ab_depth, "ckpt": args.ckpt,
        "ab_value_scale": args.ab_value_scale,
        "seed": args.seed * 1_000_003 + i * 7919, "nice": args.nice,
    } for i in range(n_shards)]

    print(f"[cfg] {args.games} games -> {n_shards} shards, {args.workers} "
          f"workers, hybrid seat0 (sims={args.sims}, leaf=ab_value, in-tree "
          f"d{args.ab_depth}) vs AB-d{args.ab_depth}, ckpt={args.ckpt}",
          flush=True)

    t0 = time.time()
    dg = dd = dw = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_play_games_worker, payloads):
            dg += r["games"]; dd += r["decisions"]; dw += r["seat0_won"]
            el = time.time() - t0
            print(f"[{dg:>6d}/{args.games}] dec={dd} seat0_won={dw} "
                  f"({100*dw/max(dg,1):.1f}%) ({dg/el:.2f} g/s, "
                  f"eta {(args.games-dg)/max(dg/el,1e-9)/60:.0f}m)", flush=True)

    print(f"[done] {dg} games, {dd} decisions, seat0 won {dw} "
          f"({100*dw/max(dg,1):.1f}%), {time.time()-t0:.0f}s -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
