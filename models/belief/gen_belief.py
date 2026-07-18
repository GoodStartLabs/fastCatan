"""AB-vs-AB generator that ALSO records the legal belief-feature vector per
decision (models.belief.belief_tracker), for the P3-X1 `legal` IL experiment.

Mirrors models/alphazero/il_dataset.py (teacher mode, p2p banned, full-obs) but
runs a BeliefTracker over each game's public event stream and stores, per
recorded decision, the POV-relative belief features. Shards add a `belief`
array; obs stays OBS_FULL_SIZE so the same shards drive plain / oracle48 / legal.

    python -m models.belief.gen_belief --games 18000 --workers 6 --ab-depth 2 \
        --out-dir models/datasets/p3x1_belief_d2_train --seed 20260718
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

MASK_BYTES = 36
WIN_VP = 10
_NO_ACTION = 0xFFFFFFFF


def _discarder(env, obs_buf, turn_owner, prev):
    """Public active discarder during a DISCARD flag (mirrors match.play_one)."""
    import fastcatan as fc
    env.write_obs(turn_owner, obs_buf)
    remaining = [0, 0, 0, 0]
    for rel in range(4):
        ab = (turn_owner + rel) & 3
        remaining[ab] = int(round(float(obs_buf[rel * 16 + 14]) * 10))
    if prev is None:
        return next((p for p in range(4) if remaining[p] > 0), turn_owner)
    if remaining[prev] == 0:
        return next(((turn_owner + o) & 3 for o in range(1, 5)
                     if remaining[(turn_owner + o) & 3] > 0), turn_owner)
    return prev


def _play_games_worker(payload):
    if payload.get("nice", 0):
        os.nice(payload["nice"])
    import fastcatan as fc
    from models.alphazero.mcts import _unpack, filter_p2p, p2p_trade_mask, p2p_banned_words
    from models.belief.belief_tracker import BeliefTracker, N_FEATURES

    depth = payload["ab_depth"]
    prune = payload["ab_prune"]
    chance_mode = payload.get("chance_mode", 0)
    import random
    rng = random.Random(payload["seed"])
    seed_seq = random.Random(payload["seed"] ^ 0x5EED)
    p2p = p2p_trade_mask()
    banned = p2p_banned_words()

    env = fc.Env()
    mask_buf = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
    obs_w = fc.OBS_FULL_SIZE
    obs_buf = np.zeros(obs_w, dtype=np.float32)
    route_buf = np.zeros(fc.OBS_SIZE, dtype=np.float32)
    tr = BeliefTracker()

    obs_l, act_l, mask_l, z_l, vps_l, seat_l, bel_l = [], [], [], [], [], [], []
    fallbacks = decisions = 0
    winners = []

    for _ in range(payload["n_games"]):
        env.reset(seed_seq.getrandbits(64))
        tr.reset(env)
        recs = []
        discarding = None
        for _ply in range(40000):
            env.action_mask(mask_buf)
            deployment_mask, deployment_legal = _unpack(mask_buf)
            mask, legal = filter_p2p(deployment_mask, p2p)
            if not legal:
                break
            turn_owner = int(env.current_player)
            phase_before = int(env.phase)
            if int(env.flag) == 1:
                discarding = _discarder(env, route_buf, turn_owner, discarding)
                actor = discarding
            else:
                discarding = None
                actor = turn_owner
            if len(deployment_legal) == 1:
                a = legal[0]
                _r, done = env.step(a)
                tr.after_step(env, actor, int(a), phase_before)
                if done:
                    break
                continue

            teacher_a = env.ab_decide(actor, depth, prune, banned, chance_mode)
            if teacher_a == _NO_ACTION or teacher_a not in legal:
                teacher_a = rng.choice(legal)
                fallbacks += 1
            env.write_obs(actor, obs_buf)
            obs_l.append(obs_buf.astype(np.float16))
            act_l.append(np.uint16(teacher_a))
            mask_l.append(np.packbits(deployment_mask))
            bel_l.append(tr.features(actor).astype(np.float16))
            recs.append((len(obs_l) - 1, actor))
            decisions += 1
            _r, done = env.step(int(teacher_a))
            tr.after_step(env, actor, int(teacher_a), phase_before)
            if done:
                break

        vps = np.array([env.player_vp(p) for p in range(4)], dtype=np.uint8)
        winner = next((p for p in range(4) if vps[p] >= WIN_VP), -1)
        winners.append(winner)
        for idx, seat in recs:
            z_l.append(np.float16(1.0 if seat == winner else -1.0))
            vps_l.append(vps)
            seat_l.append(np.uint8(seat))

    shard = Path(payload["out_dir"]) / f"shard_{payload['shard_id']:05d}.npz"
    np.savez_compressed(
        shard,
        obs=np.stack(obs_l) if obs_l else np.zeros((0, obs_w), np.float16),
        act=np.asarray(act_l, dtype=np.uint16),
        mask=np.stack(mask_l) if mask_l else np.zeros((0, MASK_BYTES), np.uint8),
        z=np.asarray(z_l, dtype=np.float16),
        vps=np.stack(vps_l) if vps_l else np.zeros((0, 4), np.uint8),
        seat=np.asarray(seat_l, dtype=np.uint8),
        belief=np.stack(bel_l) if bel_l else np.zeros((0, N_FEATURES), np.float16),
    )
    return {"games": len(winners), "decisions": decisions, "fallbacks": fallbacks,
            "won": sum(1 for w in winners if w >= 0)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=18000)
    p.add_argument("--games-per-shard", type=int, default=250)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--ab-depth", type=int, default=2)
    p.add_argument("--ab-prune", action="store_true")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=20260718)
    p.add_argument("--nice", type=int, default=15)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_shards = (args.games + args.games_per_shard - 1) // args.games_per_shard
    payloads = [{
        "n_games": min(args.games_per_shard, args.games - i * args.games_per_shard),
        "shard_id": i, "out_dir": str(out),
        "ab_depth": args.ab_depth, "ab_prune": args.ab_prune, "chance_mode": 0,
        "seed": args.seed * 1_000_003 + i * 7919, "nice": args.nice,
    } for i in range(n_shards)]
    print(f"[cfg] {args.games} games -> {n_shards} shards, {args.workers} workers, "
          f"AB d={args.ab_depth} + belief", flush=True)
    t0 = time.time()
    dg = dd = df = dw = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_play_games_worker, payloads):
            dg += r["games"]; dd += r["decisions"]; df += r["fallbacks"]; dw += r["won"]
            el = time.time() - t0
            print(f"[{dg:>6d}/{args.games}] dec={dd} won={dw} fb={df} "
                  f"({dg/el:.1f} g/s)", flush=True)
    (out / "manifest.json").write_text(json.dumps({
        "games": dg, "decisions": dd, "fallbacks": df, "won": dw,
        "ab_depth": args.ab_depth, "belief": True}, indent=2))
    print(f"[done] {dg} games {dd} decisions -> {out}", flush=True)


if __name__ == "__main__":
    main()
