"""P3-X3 guarded relabel generator: mixed teacher + trade guards.

Same mixed-expert scheme as gen_mixed (AB-d1 build/dev, balanced-strong robber +
p2p-trade), but two CONDITIONAL guards fix the two measured X2 loss mechanisms:

  * CONVERSION guard — when the persona would TRADE_OPEN, compute the offer's
    base-value EV = v·want - v·give; if it is net value-LOSING (EV <= EV_THRESH),
    override to AB's best non-trade action. Suppresses the low-conversion
    proposal spam (X2 confirm_rate 0.122).
  * PARTNER guard (CONDITIONAL, not blanket) — when the proposer would
    TRADE_CONFIRM the current VP leader, relabel to TRADE_CANCEL ONLY IF the
    leader is within striking distance (public VP >= LEAD_VP). Trades with a
    non-leading, or a leading-but-far, player are kept — the RL stage should not
    have to unlearn a blanket "never trade the leader" heuristic. We log how many
    leader-confirms SURVIVED the guard.

The guarded teacher both plays and labels; the net (warm-started from x2_plain)
learns the improved policy. Records obs(1132), guarded action, full mask, z, belief.

    python -m models.belief.gen_guarded --games 10000 --workers 6 --out-dir ...
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
RES_VALUE = np.array([1.0, 0.9, 0.6, 0.95, 1.0])  # brick,lumber,wool,grain,ore
EV_THRESH = -0.5      # suppress opens that lose more than this in base value
LEAD_VP = 8           # "striking distance": guard confirms to leaders at/above


def _discarder(env, obs_buf, turn_owner, prev):
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
    A = fc.action
    P2P_LO = int(A.TRADE_ADD_GIVE_BASE)
    OPEN = int(A.TRADE_OPEN)
    CANCEL = int(A.TRADE_CANCEL)
    CONFIRM = int(A.TRADE_CONFIRM_BASE)

    env = fc.Env()
    mask_buf = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
    obs_buf = np.zeros(fc.OBS_FULL_SIZE, dtype=np.float32)
    route_buf = np.zeros(fc.OBS_SIZE, dtype=np.float32)
    tr = BeliefTracker()

    obs_l, act_l, mask_l, z_l, vps_l, seat_l, bel_l = [], [], [], [], [], [], []
    decisions = persona_share = 0
    opens_suppressed = leader_confirm_suppressed = leader_confirm_survived = 0
    winners = []

    def ab_fallback(actor, legal):
        a = env.ab_decide(actor, depth, prune, banned, 0)
        if a == _NO_ACTION or a not in legal:
            a = rng.choice(legal)
        return int(a)

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
            deployment_mask, deployment_legal = _unpack(mask_buf)
            if not deployment_legal:
                break
            turn_owner = int(env.current_player)
            flag = int(env.flag)
            phase_before = int(env.phase)
            actor = _discarder(env, route_buf, turn_owner, discarding) if flag == 1 else turn_owner
            discarding = actor if flag == 1 else None

            if len(deployment_legal) == 1:
                a = deployment_legal[0]
                _r, done = env.step(int(a))
                tr.after_step(env, actor, int(a), phase_before)
                if done:
                    break
                continue

            use_persona = flag in (2, 3, 7)
            a_persona = int(personas[actor].act(env, mask_buf.copy())) if flag in (0, 2, 3, 7) else None
            if flag == 0 and a_persona is not None and a_persona >= P2P_LO:
                use_persona = True

            src = 0
            if use_persona and a_persona is not None and a_persona in deployment_legal:
                chosen = a_persona
                src = 1
                # --- CONVERSION guard ---
                if chosen == OPEN:
                    give = np.array([env.trade_give(r) for r in range(5)])
                    want = np.array([env.trade_want(r) for r in range(5)])
                    ev = float(RES_VALUE @ want - RES_VALUE @ give)
                    if ev <= EV_THRESH:
                        chosen = ab_fallback(actor, deployment_legal)
                        src = 0
                        opens_suppressed += 1
                # --- PARTNER guard (conditional) ---
                elif CONFIRM <= chosen < CONFIRM + 4:
                    partner = chosen - CONFIRM
                    vps = [int(env.player_vp_public(p)) for p in range(4)]
                    leader = int(np.argmax(vps))
                    if partner == leader:
                        if vps[leader] >= LEAD_VP and CANCEL in deployment_legal:
                            chosen = CANCEL
                            leader_confirm_suppressed += 1
                        else:
                            leader_confirm_survived += 1
            else:
                chosen = ab_fallback(actor, deployment_legal)

            env.write_obs_full(actor, obs_buf)
            obs_l.append(obs_buf.astype(np.float16))
            act_l.append(np.uint16(chosen))
            mask_l.append(np.packbits(deployment_mask))
            bel_l.append(tr.features(actor).astype(np.float16))
            recs.append((len(obs_l) - 1, actor))
            persona_share += src
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
    return {"games": len(winners), "decisions": decisions, "persona_share": persona_share,
            "opens_suppressed": opens_suppressed,
            "leader_confirm_suppressed": leader_confirm_suppressed,
            "leader_confirm_survived": leader_confirm_survived,
            "won": sum(1 for w in winners if w >= 0)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=10000)
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
    print(f"[cfg] {args.games} games -> {n_shards} shards, guarded mixed teacher "
          f"(EV_THRESH={EV_THRESH}, LEAD_VP={LEAD_VP})", flush=True)
    t0 = time.time()
    tot = {"games": 0, "decisions": 0, "persona_share": 0, "opens_suppressed": 0,
           "leader_confirm_suppressed": 0, "leader_confirm_survived": 0, "won": 0}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_play_games_worker, payloads):
            for k in tot:
                tot[k] += r[k]
            el = time.time() - t0
            print(f"[{tot['games']:>6d}/{args.games}] dec={tot['decisions']} "
                  f"opens_supp={tot['opens_suppressed']} "
                  f"leadconf_supp={tot['leader_confirm_suppressed']} "
                  f"leadconf_surv={tot['leader_confirm_survived']} "
                  f"({tot['games']/el:.1f} g/s)", flush=True)
    lc_tot = tot["leader_confirm_suppressed"] + tot["leader_confirm_survived"]
    tot["leader_trade_survival_frac"] = (tot["leader_confirm_survived"] / lc_tot) if lc_tot else None
    (out / "manifest.json").write_text(json.dumps({**tot, "guarded": True}, indent=2))
    print(f"[done] {tot['games']} games; opens_suppressed={tot['opens_suppressed']}; "
          f"leader_trade_survival_frac={tot['leader_trade_survival_frac']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
