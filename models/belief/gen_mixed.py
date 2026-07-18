"""Mixed-expert teacher generator for P3-X2 (trade-competent distillation).

Plays trades-ON self-play where each decision is labelled by the expert that
owns its domain:
  * robber (flag MOVE_ROBBER / ROBBER_STEAL) and ALL p2p-trade decisions
    (compose / open / accept / decline / confirm / cancel) -> the trade-competent
    persona `balanced-strong` (leader-blocking robber + coherent targeted trading);
  * main-phase build/dev/bank/end, discard, YoP/Mono/place-road, initial
    placement -> native AlphaBeta (ab_decide with p2p banned) — same build teacher
    as the incumbent il_best (AB-d1), so any win-rate delta isolates trading.

In main phase the persona is consulted first: if it wants to trade (its action is
a p2p action) that is taken; otherwise AB's build/end action is used. All four
seats play this mixed policy. Records obs(1132), teacher action, full trades-on
legal mask, sparse outcome, and the legal belief-feature vector per decision.

    python -m models.belief.gen_mixed --games 16000 --workers 6 --ab-depth 1 \
        --out-dir models/datasets/p3x2_mixed_train --seed 20260718
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
    import fastcatan as fc  # noqa: F401
    env.write_obs(turn_owner, obs_buf)
    rem = [0, 0, 0, 0]
    for rel in range(4):
        ab = (turn_owner + rel) & 3
        rem[ab] = int(round(float(obs_buf[rel * 16 + 14]) * 10))
    if prev is None:
        return next((p for p in range(4) if rem[p] > 0), turn_owner)
    if rem[prev] == 0:
        return next(((turn_owner + o) & 3 for o in range(1, 5)
                     if rem[(turn_owner + o) & 3] > 0), turn_owner)
    return prev


def _play_games_worker(payload):
    if payload.get("nice", 0):
        os.nice(payload["nice"])
    import fastcatan as fc
    from models.alphazero.mcts import _unpack, p2p_banned_words
    from models.belief.belief_tracker import BeliefTracker, N_FEATURES
    from ladder.registry import build_agent

    depth = payload["ab_depth"]
    prune = payload["ab_prune"]
    import random
    rng = random.Random(payload["seed"])
    seed_seq = random.Random(payload["seed"] ^ 0x5EED)
    banned = p2p_banned_words()
    P2P_LO = int(fc.action.TRADE_ADD_GIVE_BASE)  # 268: aid >= this ⇒ p2p

    env = fc.Env()
    mask_buf = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
    obs_buf = np.zeros(fc.OBS_FULL_SIZE, dtype=np.float32)
    route_buf = np.zeros(fc.OBS_SIZE, dtype=np.float32)
    tr = BeliefTracker()

    obs_l, act_l, mask_l, z_l, vps_l, seat_l, bel_l = [], [], [], [], [], [], []
    src_persona = 0
    decisions = 0
    winners = []

    for _ in range(payload["n_games"]):
        env.reset(seed_seq.getrandbits(64))
        tr.reset(env)
        personas = []
        for s in range(4):
            pa = build_agent("balanced-strong", seed_seq.getrandbits(64))
            b = getattr(pa, "bind_seat", None)
            if b is not None:
                b(s)
            personas.append(pa)
        recs = []
        discarding = None
        for _ply in range(40000):
            env.action_mask(mask_buf)
            deployment_mask, deployment_legal = _unpack(mask_buf)  # full, trades ON
            if not deployment_legal:
                break
            turn_owner = int(env.current_player)
            flag = int(env.flag)
            phase_before = int(env.phase)
            if flag == 1:  # discard
                actor = discarding = _discarder(env, route_buf, turn_owner, discarding)
            else:
                discarding = None
                actor = turn_owner if flag not in (2, 3) else turn_owner
                # robber move/steal is taken by the turn owner (knight or 7)
                actor = turn_owner

            if len(deployment_legal) == 1:
                a = deployment_legal[0]
                _r, done = env.step(int(a))
                tr.after_step(env, actor, int(a), phase_before)
                if done:
                    break
                continue

            # --- mixed teacher selection ---
            use_persona = flag in (2, 3, 7)  # robber move/steal, trade-pending
            a_persona = None
            if flag in (0, 2, 3, 7):
                a_persona = int(personas[actor].act(env, mask_buf.copy()))
            if flag == 0 and a_persona is not None and a_persona >= P2P_LO:
                use_persona = True
            if use_persona and a_persona is not None and a_persona in deployment_legal:
                chosen = a_persona
                src = 1
            else:
                a_ab = env.ab_decide(actor, depth, prune, banned, 0)
                if a_ab == _NO_ACTION or a_ab not in deployment_legal:
                    a_ab = rng.choice(deployment_legal)
                chosen = int(a_ab)
                src = 0

            env.write_obs_full(actor, obs_buf)
            obs_l.append(obs_buf.astype(np.float16))
            act_l.append(np.uint16(chosen))
            mask_l.append(np.packbits(deployment_mask))
            bel_l.append(tr.features(actor).astype(np.float16))
            recs.append((len(obs_l) - 1, actor))
            src_persona += src
            decisions += 1
            _r, done = env.step(int(chosen))
            tr.after_step(env, actor, int(chosen), phase_before)
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
        obs=np.stack(obs_l) if obs_l else np.zeros((0, fc.OBS_FULL_SIZE), np.float16),
        act=np.asarray(act_l, dtype=np.uint16),
        mask=np.stack(mask_l) if mask_l else np.zeros((0, MASK_BYTES), np.uint8),
        z=np.asarray(z_l, dtype=np.float16),
        vps=np.stack(vps_l) if vps_l else np.zeros((0, 4), np.uint8),
        seat=np.asarray(seat_l, dtype=np.uint8),
        belief=np.stack(bel_l) if bel_l else np.zeros((0, N_FEATURES), np.float16),
    )
    return {"games": len(winners), "decisions": decisions,
            "persona_share": src_persona, "won": sum(1 for w in winners if w >= 0)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=16000)
    p.add_argument("--games-per-shard", type=int, default=250)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--ab-depth", type=int, default=1)
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
        "shard_id": i, "out_dir": str(out), "ab_depth": args.ab_depth,
        "ab_prune": args.ab_prune, "seed": args.seed * 1_000_003 + i * 7919,
        "nice": args.nice,
    } for i in range(n_shards)]
    print(f"[cfg] {args.games} games -> {n_shards} shards, {args.workers} workers, "
          f"mixed(AB d={args.ab_depth} build + balanced-strong trade/robber)", flush=True)
    t0 = time.time()
    dg = dd = dp = dw = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_play_games_worker, payloads):
            dg += r["games"]; dd += r["decisions"]; dp += r["persona_share"]; dw += r["won"]
            el = time.time() - t0
            print(f"[{dg:>6d}/{args.games}] dec={dd} persona_frac={dp/max(dd,1):.3f} "
                  f"won={dw} ({dg/el:.1f} g/s)", flush=True)
    (out / "manifest.json").write_text(json.dumps({
        "games": dg, "decisions": dd, "persona_decisions": dp, "won": dw,
        "ab_depth": args.ab_depth, "mixed": True, "belief": True}, indent=2))
    print(f"[done] {dg} games {dd} decisions persona_frac={dp/max(dd,1):.3f} -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
