"""On- vs off-distribution check of leaf evaluators.

Correlates each net's value with the two-scale ab_value squash (the hybrid
leaf, treated here as the reference evaluator) on two state families:
  - AB-line states: the il_ab_d2_160k_full val slice (training distribution)
  - off-dist states: random-playout states (proxy for search-visited states
    far from AB lines)

If the full-obs judge's correlation degrades off-distribution much more than
the POV net's, the appendix features are brittle exactly where the search
needs them (H3 for a weak Cell J).

    python -m models.alphazero.probe_judge_offdist \
        --judge models/checkpoints/il_ab_d2_160k_full_vpm_ep10/il_final.pt \
        --pov models/checkpoints/il_ab_d2_160k_vpm/il_final.pt \
        --data-dir models/datasets/il_ab_d2_160k_full
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

import fastcatan

from models.alphazero.net import load_policy_value_net
from models.alphazero.il_pretrain import build_cache
from models.alphazero.il_dataset import _abv_label
from models.alphazero.mcts import _unpack, filter_p2p, p2p_trade_mask

OBS = fastcatan.OBS_SIZE
FULL = fastcatan.OBS_FULL_SIZE


@torch.no_grad()
def values(net, rows: np.ndarray, device: str) -> np.ndarray:
    out = []
    for s in range(0, len(rows), 8192):
        _lg, v = net(torch.from_numpy(rows[s:s + 8192]).to(device))
        out.append(v.float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--judge", required=True)
    p.add_argument("--pov", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n", type=int, default=50_000)
    p.add_argument("--offdist-games", type=int, default=150)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    judge = load_policy_value_net(
        torch.load(args.judge, map_location=args.device, weights_only=False),
        args.device)
    pov = load_policy_value_net(
        torch.load(args.pov, map_location=args.device, weights_only=False),
        args.device)

    # ---- AB-line slice ----
    data = build_cache([Path(args.data_dir)])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(data["n"])
    idx = np.sort(perm[: args.n])
    full_rows = np.asarray(data["obs"][idx], dtype=np.float32)
    abv = np.asarray(data["abv"][idx], dtype=np.float64)
    jv = values(judge, full_rows, args.device)
    pv = values(pov, full_rows[:, :OBS], args.device)
    print(f"[AB-line n={len(idx)}] corr(judge, abv)={np.corrcoef(jv, abv)[0,1]:.4f}  "
          f"corr(pov, abv)={np.corrcoef(pv, abv)[0,1]:.4f}")

    # ---- off-dist: random-playout states ----
    env = fastcatan.Env()
    prng = random.Random(args.seed ^ 0xDEAD)
    mask_buf = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
    p2p = p2p_trade_mask()
    fo, ab = [], []
    buf = np.zeros(FULL, dtype=np.float32)
    for g in range(args.offdist_games):
        env.reset(prng.getrandbits(64))
        for step in range(4000):
            env.action_mask(mask_buf)
            mask, legal = _unpack(mask_buf)
            mask, legal = filter_p2p(mask, p2p)
            if not legal:
                break
            cp = env.current_player
            if len(legal) > 1 and step % 7 == 0:
                env.write_obs_full(cp, buf)
                fo.append(buf.copy())
                ab.append(_abv_label(env, cp, 86e6)[0])
            _r, done = env.step(prng.choice(legal))
            if done:
                break
    fr = np.stack(fo)
    abv2 = np.asarray(ab, dtype=np.float64)
    jv2 = values(judge, fr, args.device)
    pv2 = values(pov, fr[:, :OBS], args.device)
    print(f"[off-dist n={len(fr)}] corr(judge, abv)={np.corrcoef(jv2, abv2)[0,1]:.4f}  "
          f"corr(pov, abv)={np.corrcoef(pv2, abv2)[0,1]:.4f}")


if __name__ == "__main__":
    main()
