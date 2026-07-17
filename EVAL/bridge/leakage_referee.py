"""Leakage-test harness v0 — the frozen information-boundary gate (spec 0.2 §C,
decision DR-001).

Claim under test: the 1084-float perspective obs (`write_obs(pov)`) is a function
of ONLY the public game state plus `pov`'s own private state. It must draw
*nothing* from opponents' hidden state — their exact resource composition, their
dev cards by type, their hidden VP cards, their pending buys.

Rather than re-derive the obs from the public event stream (which would just
re-implement the encoder), this referee proves the equivalent, stronger property
by *perturbation invariance*: take a real game state, mutate an opponent's HIDDEN
state in a way that leaves every PUBLIC quantity identical (each seat's hand
size, dev count, played-knight count, public VP; the per-resource bank; the dev
deck), and require `write_obs(pov)` to be byte-identical before and after. If any
such perturbation moves a single obs float, the obs read hidden state — a leak,
reported with the exact obs indices.

This is engine-agnostic: it manipulates *snapshots* (state_mirror) and treats
`write_obs` as a black box, so any future obs function — including the
belief-augmented research obs of DR-001 — is validated against the same referee.
A legal belief feature depends on public history; it must still be invariant to
which hidden card an opponent actually holds, so it passes this gate, while an
obs that peeks at a resolved steal/discard/dev-draw fails it.

Perturbations (all public-preserving, applied between two opponents of `pov`):
  R  resource swap    — i:+r2/-r1, j:+r1/-r2. Hand sizes and per-resource bank
                        fixed; only hidden composition moves. Covers hidden
                        resource composition, steals, and discard mixes.
  D  non-VP dev swap  — trade one hidden non-VP dev type each way. total_dev,
                        knights-played, public VP and the deck are fixed; only
                        hidden dev-by-type moves.
  V  hidden-VP swap   — move a hidden VP card i->j and a non-VP dev j->i,
                        adjusting total VP. total_dev, PUBLIC VP and the deck are
                        fixed; the hidden VP card changes owner and each seat's
                        *total* VP. Fails iff the obs shows opponents' total
                        (dev-inclusive) VP instead of public VP.

Run (box, 14 cores):
  PYTHONPATH=EVAL PYTHONHASHSEED=0 python -m bridge.leakage_referee \
      --games 10000 --seed 0 --workers 14 --sample-every 12 --out ~/leak.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import fastcatan as fc
from bridge import state_mirror as M

OBS = fc.OBS_SIZE
DEV_VP = 1  # dev_deck / player_dev index order: knight,VP,road,yop,mono


def _mask_arr():
    return np.zeros(fc.MASK_WORDS, dtype=np.uint64)


def _legal(m):
    out = []
    for w in range(fc.MASK_WORDS):
        bits = int(m[w])
        while bits:
            b = bits & -bits
            out.append(w * 64 + b.bit_length() - 1)
            bits ^= b
    return out


def _obs(env, pov):
    o = np.zeros(OBS, dtype=np.float32)
    env.write_obs(pov, o)
    return o


def _resource_swap(gs, a, b):
    for r1 in range(5):
        if gs.player_resources[a][r1] <= 0:
            continue
        for r2 in range(5):
            if r1 == r2 or gs.player_resources[b][r2] <= 0:
                continue
            gs.player_resources[a][r1] -= 1
            gs.player_resources[a][r2] += 1
            gs.player_resources[b][r2] -= 1
            gs.player_resources[b][r1] += 1
            return True
    return False


def _dev_swap_nonvp(gs, a, b):
    nonvp = [d for d in range(5) if d != DEV_VP]
    for d1 in nonvp:
        if gs.player_dev[a][d1] <= 0:
            continue
        for d2 in nonvp:
            if d1 == d2 or gs.player_dev[b][d2] <= 0:
                continue
            gs.player_dev[a][d1] -= 1
            gs.player_dev[a][d2] += 1
            gs.player_dev[b][d2] -= 1
            gs.player_dev[b][d1] += 1
            return True
    return False


def _dev_swap_vp(gs, a, b):
    # move a hidden VP card a->b, a non-VP dev b->a; keep total_dev per seat and
    # public VP fixed, adjust each seat's total VP so the state stays consistent.
    if gs.player_dev[a][DEV_VP] <= 0:
        return False
    nonvp = [d for d in range(5) if d != DEV_VP and gs.player_dev[b][d] > 0]
    if not nonvp:
        return False
    d2 = nonvp[0]
    gs.player_dev[a][DEV_VP] -= 1
    gs.player_dev[a][d2] += 1
    gs.player_dev[b][d2] -= 1
    gs.player_dev[b][DEV_VP] += 1
    # a lost a hidden VP (total vp -1), b gained one (+1); public vp unchanged.
    gs.player_vp[a] = max(0, gs.player_vp[a] - 1)
    gs.player_vp[b] = gs.player_vp[b] + 1
    return True


PERTURBATIONS = {"R": _resource_swap, "D": _dev_swap_nonvp, "V": _dev_swap_vp}


def _check_state(env, base_bytes, cov):
    """Return list of leak findings for one game state, over all pov + pairs."""
    findings = []
    for pov in range(4):
        env.load_snapshot(base_bytes)
        base = _obs(env, pov)
        opponents = [s for s in range(4) if s != pov]
        for k in range(len(opponents)):
            for l in range(len(opponents)):
                if k == l:
                    continue
                a, b = opponents[k], opponents[l]
                for tag, fn in PERTURBATIONS.items():
                    snap = M.parse_snapshot(base_bytes)
                    if not fn(snap.gs, a, b):
                        continue
                    cov[f"perturb_{tag}"] += 1
                    env.load_snapshot(M.to_bytes(snap))
                    o = _obs(env, pov)
                    if not np.array_equal(o, base):
                        idx = np.nonzero(o != base)[0]
                        findings.append({
                            "perturb": tag, "pov": pov, "opp_a": a, "opp_b": b,
                            "obs_idx": idx.tolist()[:16],
                            "deltas": [(int(i), float(base[i]), float(o[i]))
                                       for i in idx[:16]],
                        })
    return findings


def run_games(seed0, n, sample_every):
    rng = np.random.default_rng(seed0 ^ 0xABCDEF)
    env = fc.Env()
    m = _mask_arr()
    cov = Counter()
    findings = []
    states_checked = 0
    for g in range(n):
        seed = seed0 + g
        env.reset(seed)
        step = 0
        while True:
            env.action_mask(m)
            legals = _legal(m)
            if not legals or env.phase == 3:
                break
            # sample a state for the referee before stepping
            if step % sample_every == 0:
                fl = int(env.flag)
                if fl == 1:
                    cov["state_discard"] += 1
                elif fl == 3:
                    cov["state_robber_steal"] += 1
                elif fl == 7:
                    cov["state_trade_pending"] += 1
                base_bytes = env.snapshot()
                fnd = _check_state(env, base_bytes, cov)
                states_checked += 1
                cov["states_checked"] += 1
                if fnd:
                    findings.extend(fnd)
                # snapshot round-trip may leave mask stale; restore & re-mask
                env.load_snapshot(base_bytes)
                env.recompute_mask()
                env.action_mask(m)
                legals = _legal(m)
                if not legals:
                    break
            env.step(int(legals[rng.integers(len(legals))]))
            step += 1
            if step > 6000:
                break
    return {"cov": dict(cov), "findings": findings[:50],
            "n_findings": len(findings), "states_checked": states_checked}


def _worker(args):
    seed0, n, sample_every = args
    return run_games(seed0, n, sample_every)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--sample-every", type=int, default=12)
    ap.add_argument("--out", type=str, default="/tmp/leak.json")
    args = ap.parse_args()

    t0 = time.time()
    per = args.games // args.workers
    chunks = []
    s = args.seed
    for w in range(args.workers):
        n = per + (1 if w < args.games % args.workers else 0)
        chunks.append((s, n, args.sample_every))
        s += n

    agg = Counter()
    findings = []
    states = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_worker, chunks):
            for k, v in res["cov"].items():
                agg[k] += v
            findings.extend(res["findings"])
            states += res["states_checked"]

    elapsed = time.time() - t0
    total_perturb = sum(agg.get(f"perturb_{t}", 0) for t in PERTURBATIONS)
    summary = {
        "games": args.games,
        "states_checked": states,
        "total_perturbations": total_perturb,
        "perturbations_by_type": {t: agg.get(f"perturb_{t}", 0) for t in PERTURBATIONS},
        "states_with_discard": agg.get("state_discard", 0),
        "states_with_robber_steal": agg.get("state_robber_steal", 0),
        "states_with_trade_pending": agg.get("state_trade_pending", 0),
        "n_leak_findings": len(findings),
        "leak_findings": findings[:50],
        "elapsed_s": elapsed,
        "games_per_s": args.games / elapsed if elapsed else 0,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LEAKAGE REFEREE SUMMARY ===")
    print(f"games:                    {args.games}")
    print(f"states refereed:          {states}")
    print(f"perturbations applied:    {total_perturb} "
          f"(R={agg.get('perturb_R',0)} D={agg.get('perturb_D',0)} V={agg.get('perturb_V',0)})")
    print(f"  states w/ discard:      {agg.get('state_discard',0)}")
    print(f"  states w/ robber-steal: {agg.get('state_robber_steal',0)}")
    print(f"  states w/ trade-pending:{agg.get('state_trade_pending',0)}")
    print(f"LEAK findings:            {len(findings)}   (MUST be 0)")
    print(f"elapsed:                  {elapsed:.1f}s ({summary['games_per_s']:.0f} games/s)")
    if findings:
        print("\nFIRST LEAK:\n", json.dumps(findings[0], indent=2)[:1200])
    else:
        print("\nNo leak: write_obs(pov) is byte-invariant to every hidden-state "
              "perturbation. The 1084 obs draws nothing from opponents' hidden "
              "resources, dev-by-type, or hidden VP.")


if __name__ == "__main__":
    main()
