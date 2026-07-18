"""P3-X4 TRUE DAgger generator for trade CONVERSION.

The x3 net imitates the mixed teacher at ~0.81 top-1; its DIVERGENCE is
concentrated in offer composition (confirm_rate 0.121 vs persona 0.217 — it
composes offers partners decline). X2/X3 could not fix this because the teacher
both PLAYED and LABELLED, so the net never saw its OWN bad-offer states with a
correction.

TRUE DAgger: the STUDENT (x3 net) plays the learner seat — generating its own
(often awkward) offer-composition states — while the guarded mixed teacher
(AB-d1 build + balanced-strong robber/trade + the X3 conditional partner guard)
LABELS what to do from those states. Opponents are the same guarded mixed teacher.
Fine-tuning x3 on (student-state, teacher-label) should pull the net's offer
composition toward the persona's, raising confirm_rate.

Records obs(1132), teacher label, full mask, z, belief for the learner's
multi-legal decisions only; the learner seat rotates for balance.

    python -m models.belief.gen_dagger --games 6000 --workers 6 \
        --student models/checkpoints/p3_creat_x3/x3_plain.zip --out-dir ...
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
RES_VALUE = np.array([1.0, 0.9, 0.6, 0.95, 1.0])
EV_THRESH = -0.5
LEAD_VP = 8


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
    from sb3_contrib import MaskablePPO

    depth = payload["ab_depth"]
    prune = payload["ab_prune"]
    import random
    rng = random.Random(payload["seed"])
    seed_seq = random.Random(payload["seed"] ^ 0x5EED)
    banned = p2p_banned_words()
    A = fc.action
    P2P_LO = int(A.TRADE_ADD_GIVE_BASE)
    OPEN, CANCEL, CONFIRM = int(A.TRADE_OPEN), int(A.TRADE_CANCEL), int(A.TRADE_CONFIRM_BASE)

    net = MaskablePPO.load(payload["student"], device="cpu")

    env = fc.Env()
    mask_buf = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
    obs_full = np.zeros(fc.OBS_FULL_SIZE, dtype=np.float32)
    obs_pov = np.zeros(fc.OBS_SIZE, dtype=np.float32)
    route_buf = np.zeros(fc.OBS_SIZE, dtype=np.float32)
    tr = BeliefTracker()

    obs_l, act_l, mask_l, z_l, vps_l, seat_l, bel_l = [], [], [], [], [], [], []
    decisions = opens_suppressed = lc_supp = lc_surv = 0
    winners = []

    def ab_fallback(actor, legal):
        a = env.ab_decide(actor, depth, prune, banned, 0)
        if a == _NO_ACTION or a not in legal:
            a = rng.choice(legal)
        return int(a)

    def teacher_label(env, actor, personas, deployment_legal, flag):
        nonlocal opens_suppressed, lc_supp, lc_surv
        a_persona = int(personas[actor].act(env, mask_buf.copy())) if flag in (0, 2, 3, 7) else None
        use_persona = flag in (2, 3, 7)
        if flag == 0 and a_persona is not None and a_persona >= P2P_LO:
            use_persona = True
        if use_persona and a_persona is not None and a_persona in deployment_legal:
            chosen = a_persona
            if chosen == OPEN:
                give = np.array([env.trade_give(r) for r in range(5)])
                want = np.array([env.trade_want(r) for r in range(5)])
                if float(RES_VALUE @ want - RES_VALUE @ give) <= EV_THRESH:
                    opens_suppressed += 1
                    return ab_fallback(actor, deployment_legal)
            elif CONFIRM <= chosen < CONFIRM + 4:
                partner = chosen - CONFIRM
                vps = [int(env.player_vp_public(p)) for p in range(4)]
                leader = int(np.argmax(vps))
                if partner == leader:
                    if vps[leader] >= LEAD_VP and CANCEL in deployment_legal:
                        lc_supp += 1
                        return CANCEL
                    lc_surv += 1
            return chosen
        return ab_fallback(actor, deployment_legal)

    def net_action(actor, deployment_mask, deployment_legal):
        env.write_obs(actor, obs_pov)
        legal_bool = np.zeros(fc.NUM_ACTIONS, dtype=bool)
        legal_bool[deployment_legal] = True
        a, _ = net.predict(obs_pov, action_masks=legal_bool, deterministic=False)
        a = int(a)
        return a if a in deployment_legal else deployment_legal[0]

    for g in range(payload["n_games"]):
        env.reset(seed_seq.getrandbits(64))
        tr.reset(env)
        learner_seat = g & 3
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

            if actor == learner_seat:
                # student ACTS (its own state distribution); teacher LABELS.
                label = teacher_label(env, actor, personas, deployment_legal, flag)
                env.write_obs_full(actor, obs_full)
                obs_l.append(obs_full.astype(np.float16))
                act_l.append(np.uint16(label))
                mask_l.append(np.packbits(deployment_mask))
                bel_l.append(tr.features(actor).astype(np.float16))
                recs.append((len(obs_l) - 1, actor))
                decisions += 1
                play = net_action(actor, deployment_mask, deployment_legal)
            else:
                play = teacher_label(env, actor, personas, deployment_legal, flag)
            _r, done = env.step(int(play))
            tr.after_step(env, actor, int(play), phase_before)
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
            "opens_suppressed": opens_suppressed, "lc_supp": lc_supp, "lc_surv": lc_surv,
            "won": sum(1 for w in winners if w >= 0)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=6000)
    p.add_argument("--games-per-shard", type=int, default=200)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--ab-depth", type=int, default=1)
    p.add_argument("--ab-prune", action="store_true")
    p.add_argument("--student", required=True)
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
        "ab_prune": args.ab_prune, "student": args.student,
        "seed": args.seed * 1_000_003 + i * 7919, "nice": args.nice,
    } for i in range(n_shards)]
    print(f"[cfg] DAgger {args.games} games, student={args.student}", flush=True)
    t0 = time.time()
    tot = {"games": 0, "decisions": 0, "opens_suppressed": 0, "lc_supp": 0, "lc_surv": 0, "won": 0}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_play_games_worker, payloads):
            for k in tot:
                tot[k] += r[k]
            el = time.time() - t0
            print(f"[{tot['games']:>6d}/{args.games}] rec={tot['decisions']} "
                  f"({tot['games']/el:.1f} g/s)", flush=True)
    lc = tot["lc_supp"] + tot["lc_surv"]
    tot["leader_trade_survival_frac"] = (tot["lc_surv"] / lc) if lc else None
    (out / "manifest.json").write_text(json.dumps({**tot, "dagger": True}, indent=2))
    print(f"[done] {tot['games']} games rec={tot['decisions']} "
          f"leader_trade_survival={tot['leader_trade_survival_frac']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
