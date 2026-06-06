"""AB-vs-AB game generator for imitation warm-starting (the OOD pivot).

Why: five net rungs (vs-random 0.43->0.73) and every sims level (64->512)
left vs-AB flat at 2-4% — the self-play distribution never contains AB's
tight value-greedy lines, so the net's prior/value are out-of-distribution
from the initial placement onward, and neither in-distribution improvement
nor deeper search over a wrong evaluation moves the needle. The untried
lever (and the one that dodges the never-wins gradient collapse that killed
direct-vs-AB training) is imitation: put the net ON AB's distribution AT
AB's level, then fine-tune with search from parity.

This script plays games where ALL FOUR seats are the native fixed-hole
AlphaBeta (Env.ab_decide with the p2p banned mask — no random-fallback hole
to clone) and records every multi-legal decision:

    obs   float16 (OBS_SIZE,)   current player's POV
    act   uint16                the teacher's action id
    mask  uint8  (MASK_BYTES,)  packed legal mask (np.packbits, p2p-filtered)
    z     float16               sparse +-1 outcome for the recording seat
    vps   uint8  (4,)           final VPs (lets the pretrainer recompute
                                vp_margin targets without replay)

Shards are compressed .npz, ~250 games each, written to --out-dir. Workers
are separate processes (spawn) with their own Env; everything is CPU and
nice'd so a concurrent GPU training run keeps priority.

Run:
    python -m models.alphazero.il_dataset --games 40000 --workers 8 \
        --ab-depth 1 --out-dir models/datasets/il_ab_d1
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
_NO_ACTION = 0xFFFFFFFF


def _play_games_worker(payload: dict) -> dict:
    """Play n_games games; write one shard; return stats.

    Teacher mode (student_ckpt=None): all 4 seats are AB; every multi-legal
    decision is recorded with the teacher's action (which is also played).

    DAgger mode (student_ckpt set): seat 0 is the STUDENT (argmax masked
    policy — the deployment policy); seats 1-3 are AB. Every seat-0 decision
    records the TEACHER's label for the student-visited state, but the
    STUDENT's action is played — classic DAgger: labels on the learner's own
    state distribution, fixing the compounding covariate shift that one-step
    cloning leaves behind. z is the real outcome of student-vs-AB play, i.e.
    the deployment value target.
    """
    if payload.get("nice", 0):
        os.nice(payload["nice"])
    import random

    import fastcatan as fc
    from models.alphazero.mcts import (
        _unpack, filter_p2p, p2p_trade_mask, p2p_banned_words,
    )

    depth = payload["ab_depth"]
    prune = payload["ab_prune"]
    chance_mode = payload.get("chance_mode", 0)
    rng = random.Random(payload["seed"])
    seed_seq = random.Random(payload["seed"] ^ 0x5EED)
    p2p = p2p_trade_mask()
    banned = p2p_banned_words()

    student = None
    if payload.get("student_ckpt"):
        import torch
        torch.set_num_threads(1)
        from models.alphazero.net import load_policy_value_net
        state = torch.load(payload["student_ckpt"], map_location="cpu",
                           weights_only=False)
        student = (torch, load_policy_value_net(state, "cpu"))

    env = fc.Env()
    mask_buf = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
    obs_buf = np.zeros(fc.OBS_SIZE, dtype=np.float32)

    obs_l, act_l, mask_l, z_l, vps_l, seat_l = [], [], [], [], [], []
    fallbacks = 0
    decisions = 0
    winners = []

    for _ in range(payload["n_games"]):
        env.reset(seed_seq.getrandbits(64))
        recs: list[tuple[int, int]] = []   # (sample_index, seat)
        for _ply in range(40000):
            env.action_mask(mask_buf)
            mask, legal = _unpack(mask_buf)
            mask, legal = filter_p2p(mask, p2p)
            if not legal:
                break
            cp = env.current_player
            if len(legal) == 1:
                _r, done = env.step(legal[0])
                if done:
                    break
                continue

            if student is not None and cp != 0:
                # DAgger: opponent seats are plain AB, not recorded.
                a = env.ab_decide(cp, depth, prune, banned, chance_mode)
                if a == _NO_ACTION or a not in legal:
                    a = rng.choice(legal)
                    fallbacks += 1
                _r, done = env.step(int(a))
                if done:
                    break
                continue

            teacher_a = env.ab_decide(cp, depth, prune, banned, chance_mode)
            if teacher_a == _NO_ACTION or teacher_a not in legal:
                teacher_a = rng.choice(legal)  # safety net; ~never (counted)
                fallbacks += 1

            env.write_obs(cp, obs_buf)
            obs_l.append(obs_buf.astype(np.float16))
            act_l.append(np.uint16(teacher_a))
            mask_l.append(np.packbits(mask))
            recs.append((len(obs_l) - 1, cp))
            decisions += 1

            if student is None:
                play_a = int(teacher_a)
            else:                              # student acts, teacher labels
                torch_mod, net = student
                with torch_mod.no_grad():
                    logits, _v = net(torch_mod.from_numpy(obs_buf).unsqueeze(0))
                row = logits[0].numpy()
                row[~mask] = -np.inf
                play_a = int(row.argmax())
            _r, done = env.step(play_a)
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
        obs=np.stack(obs_l) if obs_l else np.zeros((0, 1084), np.float16),
        act=np.asarray(act_l, dtype=np.uint16),
        mask=np.stack(mask_l) if mask_l else np.zeros((0, MASK_BYTES), np.uint8),
        z=np.asarray(z_l, dtype=np.float16),
        vps=np.stack(vps_l) if vps_l else np.zeros((0, 4), np.uint8),
        seat=np.asarray(seat_l, dtype=np.uint8),
    )
    n_won = sum(1 for w in winners if w >= 0)
    return {"shard": str(shard), "games": len(winners), "decisions": decisions,
            "fallbacks": fallbacks, "won": n_won}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=40000)
    p.add_argument("--games-per-shard", type=int, default=250)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ab-depth", type=int, default=1)
    p.add_argument("--ab-prune", action="store_true")
    p.add_argument("--catanatron-chance", action="store_true",
                   help="teacher uses Catanatron's chance blur + first-enemy "
                        "robber pruner (rules.hpp CHANCE_CATANATRON) — clone "
                        "the BRIDGE table's actual lines, not the native "
                        "port's (their games differ; model_divergence.py).")
    p.add_argument("--out-dir", type=str, default="models/datasets/il_ab_d1")
    p.add_argument("--student-ckpt", type=str, default="",
                   help="DAgger mode: seat-0 acts with this net's argmax "
                        "policy while the teacher labels its states; seats "
                        "1-3 are AB. Empty = teacher mode (AB plays + labels "
                        "all seats).")
    p.add_argument("--seed", type=int, default=20260605)
    p.add_argument("--nice", type=int, default=10,
                   help="os.nice for workers so GPU training keeps priority.")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_shards = (args.games + args.games_per_shard - 1) // args.games_per_shard
    payloads = [{
        "n_games": min(args.games_per_shard,
                       args.games - i * args.games_per_shard),
        "shard_id": i, "out_dir": str(out),
        "ab_depth": args.ab_depth, "ab_prune": args.ab_prune,
        "student_ckpt": args.student_ckpt or None,
        "chance_mode": 1 if args.catanatron_chance else 0,
        "seed": args.seed * 1_000_003 + i * 7919, "nice": args.nice,
    } for i in range(n_shards)]

    mode = (f"DAgger(student={args.student_ckpt})" if args.student_ckpt
            else "teacher")
    print(f"[cfg] {args.games} games -> {n_shards} shards, "
          f"{args.workers} workers, AB d={args.ab_depth} "
          f"prune={args.ab_prune} (fixed-hole, no-trades), mode={mode}",
          flush=True)

    t0 = time.time()
    done_games = done_dec = done_fb = done_won = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_play_games_worker, payloads):
            done_games += r["games"]
            done_dec += r["decisions"]
            done_fb += r["fallbacks"]
            done_won += r["won"]
            el = time.time() - t0
            print(f"[{done_games:>6d}/{args.games}] dec={done_dec} "
                  f"won={done_won} fallbacks={done_fb} "
                  f"({done_games/el:.1f} g/s)", flush=True)

    print(f"[done] {done_games} games, {done_dec} decisions, "
          f"{done_fb} fallbacks ({100*done_fb/max(done_dec,1):.3f}%), "
          f"{time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
