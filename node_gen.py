"""Thin node-side generation launcher: reproduce the box's stage-3 shards for an
explicit set of shard IDs by reusing stage3_gen._play_games_worker with the EXACT
per-shard payload the box uses (no edit to stage3_gen.py). Each shard's content is
a pure function of its id via seed = BASE_SEED*1_000_003 + i*7919, so generating a
disjoint id subset yields shards byte-compatible with the box's numbering.

Honors STAGE3_GAME_TIMEOUT_S (read inside the worker). Tracks per-shard abandoned
games (n_games - completed) and enforces a node-aggregate abandon guard.
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# Box repro defaults (stage3_gen argparse defaults) — must match exactly.
BASE_SEED = 20260608
GAMES_PER_SHARD = 250
SIMS = 512
AB_DEPTH = 2
AB_VALUE_SCALE = 86e6
NICE = 10


def parse_ids(spec: str) -> list[int]:
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-ids", required=True, help='e.g. "32,37,40-49"')
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=0, help="0 => one per shard")
    ap.add_argument("--abort-frac", type=float, default=0.02)
    a = ap.parse_args()

    ids = parse_ids(a.shard_ids)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    from models.alphazero.stage3_gen import _play_games_worker

    payloads = [{
        "n_games": GAMES_PER_SHARD, "shard_id": i, "out_dir": a.out_dir,
        "sims": SIMS, "ab_depth": AB_DEPTH, "ckpt": a.ckpt,
        "ab_value_scale": AB_VALUE_SCALE,
        "seed": BASE_SEED * 1_000_003 + i * 7919, "nice": NICE,
    } for i in ids]
    workers = a.workers or len(ids)
    to = os.environ.get("STAGE3_GAME_TIMEOUT_S", "0")
    print(f"[node-gen] shards={ids} n={len(ids)} workers={workers} "
          f"STAGE3_GAME_TIMEOUT_S={to} ckpt={a.ckpt} out={a.out_dir}", flush=True)

    t0 = time.time()
    done = tot_games = tot_abandon = 0
    flagged = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for r in pool.imap_unordered(_play_games_worker, payloads):
            done += 1
            comp = r["games"]
            ab = GAMES_PER_SHARD - comp
            tot_games += comp
            tot_abandon += ab
            frac = ab / GAMES_PER_SHARD
            el = time.time() - t0
            print(f"[shard-done] {r['shard']} completed={comp}/{GAMES_PER_SHARD} "
                  f"abandoned={ab} ({100*frac:.1f}%) decisions={r['decisions']} "
                  f"seat0_won={r['seat0_won']} | {done}/{len(ids)} "
                  f"agg={tot_games/el:.3f} g/s", flush=True)
            if frac > a.abort_frac:
                flagged.append((r["shard"], ab))
                print(f"[ABORT-GUARD] {r['shard']} abandoned {ab}/{GAMES_PER_SHARD} "
                      f"> {100*a.abort_frac:.0f}%", flush=True)

    agg_frac = tot_abandon / max(1, tot_games + tot_abandon)
    print(f"[node-gen-done] shards={len(ids)} completed_games={tot_games} "
          f"abandoned={tot_abandon} agg_abandon_frac={100*agg_frac:.2f}% "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
    if agg_frac > a.abort_frac or flagged:
        print(f"[ABORT-GUARD-TRIP] agg={100*agg_frac:.2f}% flagged={flagged}",
              flush=True)
        sys.exit(3)
    print("NODE_GEN_OK", flush=True)


if __name__ == "__main__":
    main()
