"""Does the trained full-obs judge actually READ the hidden-enemy appendix?

Loads the judge + the full-obs cache, computes value MSE on a held-out slice
twice: (a) real appendix, (b) appendix zeroed. A meaningful (b)-(a) gap =
the value head exploits hidden enemy state; ~0 gap = it learned to ignore
it (and Cell J would be expected to reproduce the POV band).

    python -m models.alphazero.probe_judge_appendix \
        --ckpt models/checkpoints/il_ab_d2_160k_full_vpm_ep10/il_final.pt \
        --data-dir models/datasets/il_ab_d2_160k_full
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import fastcatan

from models.alphazero.net import load_policy_value_net
from models.alphazero.il_pretrain import build_cache

OBS_SIZE = fastcatan.OBS_SIZE


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    st = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    net = load_policy_value_net(st, args.device)
    data = build_cache([Path(args.data_dir)])
    n_total = data["n"]
    # same val convention as il_pretrain: seed-0 permutation tail is TRAIN,
    # head is VAL — sample from the head region to stay held-out.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    val_pool = perm[: max(int(n_total * 0.02), 4096)]
    idx = np.sort(rng.choice(val_pool, size=min(args.n, len(val_pool)),
                             replace=False))

    vps = np.asarray(data["vps"][idx], dtype=np.float32)
    seat = np.asarray(data["seat"][idx], dtype=np.int64)
    rows = np.arange(len(seat))
    own = vps[rows, seat]
    vo = vps.copy()
    vo[rows, seat] = -1.0
    z = np.clip((own - vo.max(axis=1)) / 10.0, -1.0, 1.0).astype(np.float32)

    def mse(zero_appendix: bool) -> float:
        se = 0.0
        with torch.no_grad():
            for s in range(0, len(idx), args.batch_size):
                sl = idx[s:s + args.batch_size]
                obs = np.asarray(data["obs"][sl], dtype=np.float32)
                if zero_appendix:
                    obs[:, OBS_SIZE:] = 0.0
                _lg, v = net(torch.from_numpy(obs).to(args.device))
                se += float(((v.float().cpu().numpy()
                              - z[s:s + args.batch_size]) ** 2).sum())
        return se / len(idx)

    real = mse(False)
    zeroed = mse(True)
    print(f"n={len(idx)}  value-mse real-appendix={real:.5f}  "
          f"zeroed-appendix={zeroed:.5f}  gap={zeroed - real:+.5f} "
          f"({100*(zeroed-real)/max(real,1e-9):+.1f}%)")


if __name__ == "__main__":
    main()
