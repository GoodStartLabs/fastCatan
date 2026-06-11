"""Supervised imitation pretrain on AB-vs-AB games (il_dataset shards).

Policy head: masked cross-entropy against the teacher's action (illegal
logits forced to -inf, exactly the masking the net sees at play time).
Value head: MSE against the recording seat's sparse +-1 outcome — this is
the value-calibration-on-AB-lines that every previous vs-AB experiment
lacked, and what the hybrid learned-value-AB evaluator would feed on.

First run builds uncompressed memmap caches (obs/act/mask/z) next to the
shards for fast random access; subsequent runs reuse them. Validation =
held-out tail shards; reports teacher-action top-1 accuracy and value MSE.

The checkpoint is az-loader compatible ({net_state, args} + stamp), so
evaluate.py / mcts_vs_fixed / batched_selfplay --init-from work unchanged.

Run:
    python -m models.alphazero.il_pretrain --data-dir models/datasets/il_ab_d1 \
        --hidden 1024,1024,512 --epochs 3 --device cuda
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import fastcatan

from models.ckpt import write_stamp
from models.alphazero.net import PolicyValueNet

OBS_SIZE = fastcatan.OBS_SIZE
NUM_ACTIONS = fastcatan.NUM_ACTIONS
MASK_BYTES = 36
VP_W = 3e14  # catanatron lexicographic VP weight — lockstep with il_dataset

CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def build_cache(data_dirs: list[Path]) -> dict:
    """Concatenate shards (from one or more dirs) into uncompressed memmaps.

    Multiple dirs = DAgger aggregation: train on the union of the teacher
    set and the student-state relabel sets. The union cache lives under the
    first dir, keyed by the extra dir names."""
    if len(data_dirs) == 1:
        cache = data_dirs[0] / "cache"
    else:
        key = "_".join(d.name for d in data_dirs[1:])
        cache = data_dirs[0] / f"cache_union_{key}"
    meta = cache / "meta.npz"
    if meta.exists():
        m = np.load(meta)
        n = int(m["n"])
        out = {
            "n": n,
            "obs": np.lib.format.open_memmap(cache / "obs.npy", mode="r"),
            "act": np.lib.format.open_memmap(cache / "act.npy", mode="r"),
            "mask": np.lib.format.open_memmap(cache / "mask.npy", mode="r"),
            "z": np.lib.format.open_memmap(cache / "z.npy", mode="r"),
            "vps": np.lib.format.open_memmap(cache / "vps.npy", mode="r"),
            "seat": np.lib.format.open_memmap(cache / "seat.npy", mode="r"),
        }
        if (cache / "abv.npy").exists():   # v3 shards (learned-judge labels)
            out["abv"] = np.lib.format.open_memmap(cache / "abv.npy", mode="r")
        if (cache / "abm.npy").exists():   # v4 shards (raw ab_value margin)
            out["abm"] = np.lib.format.open_memmap(cache / "abm.npy", mode="r")
        if (cache / "rootv.npy").exists():  # stage-3 shards (MCTS root value)
            out["rootv"] = np.lib.format.open_memmap(cache / "rootv.npy", mode="r")
        return out
    shards = sorted(s for d in data_dirs
                    for s in glob.glob(str(d / "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"no shards under {data_dirs}")
    counts = []
    has_abv = has_abm = has_rootv = True
    obs_w = None
    for s in shards:
        with np.load(s) as d:
            counts.append(d["act"].shape[0])
            has_abv = has_abv and ("abv" in d.files)
            has_abm = has_abm and ("abm" in d.files)
            has_rootv = has_rootv and ("rootv" in d.files)
            w = d["obs"].shape[1]
            if obs_w is None:
                obs_w = w
            elif obs_w != w:
                raise ValueError(f"obs width mismatch across shards: {obs_w} "
                                 f"vs {w} in {s} — don't mix POV and full-obs "
                                 f"datasets in one cache")
    n = int(sum(counts))
    cache.mkdir(exist_ok=True)
    obs = np.lib.format.open_memmap(cache / "obs.npy", mode="w+",
                                    dtype=np.float16, shape=(n, obs_w))
    act = np.lib.format.open_memmap(cache / "act.npy", mode="w+",
                                    dtype=np.uint16, shape=(n,))
    mask = np.lib.format.open_memmap(cache / "mask.npy", mode="w+",
                                     dtype=np.uint8, shape=(n, MASK_BYTES))
    z = np.lib.format.open_memmap(cache / "z.npy", mode="w+",
                                  dtype=np.float16, shape=(n,))
    vps = np.lib.format.open_memmap(cache / "vps.npy", mode="w+",
                                    dtype=np.uint8, shape=(n, 4))
    seat = np.lib.format.open_memmap(cache / "seat.npy", mode="w+",
                                     dtype=np.uint8, shape=(n,))
    abv = (np.lib.format.open_memmap(cache / "abv.npy", mode="w+",
                                     dtype=np.float32, shape=(n,))
           if has_abv else None)
    abm = (np.lib.format.open_memmap(cache / "abm.npy", mode="w+",
                                     dtype=np.float64, shape=(n,))
           if has_abm else None)
    rootv = (np.lib.format.open_memmap(cache / "rootv.npy", mode="w+",
                                       dtype=np.float32, shape=(n,))
             if has_rootv else None)
    o = 0
    for s, c in zip(shards, counts):
        with np.load(s) as d:
            obs[o:o + c] = d["obs"]
            act[o:o + c] = d["act"]
            mask[o:o + c] = d["mask"]
            z[o:o + c] = d["z"]
            vps[o:o + c] = d["vps"]
            seat[o:o + c] = d["seat"]
            if abv is not None:
                abv[o:o + c] = d["abv"]
            if abm is not None:
                abm[o:o + c] = d["abm"]
            if rootv is not None:
                rootv[o:o + c] = d["rootv"]
        o += c
        print(f"[cache] {o}/{n}", flush=True)
    obs.flush(); act.flush(); mask.flush(); z.flush()
    vps.flush(); seat.flush()
    if abv is not None:
        abv.flush()
    if abm is not None:
        abm.flush()
    if rootv is not None:
        rootv.flush()
    np.savez(meta, n=n)
    print(f"[cache] built: {n} samples", flush=True)
    return build_cache(data_dirs)


def _batch(data, idx, device, value_target="sparse", abv_scale=86e6,
           obs_width=0):
    obs_np = np.asarray(data["obs"][idx], dtype=np.float32)
    if obs_width:
        obs_np = obs_np[:, :obs_width]      # POV-control on full-obs shards
    obs = torch.from_numpy(obs_np).to(device)
    act = torch.from_numpy(np.asarray(data["act"][idx], dtype=np.int64)).to(device)
    mask_bits = np.unpackbits(np.asarray(data["mask"][idx]), axis=1)[:, :NUM_ACTIONS]
    mask = torch.from_numpy(mask_bits.astype(bool)).to(device)
    if value_target == "vp_margin":
        # Dense final-VP margin from the recording seat's POV — finer leaf
        # resolution than sparse +-1 (the search-SNR hypothesis: MCTS can't
        # resolve 1-3%-win-prob move differences through a noisy +-1 head).
        vps = np.asarray(data["vps"][idx], dtype=np.float32)
        seat = np.asarray(data["seat"][idx], dtype=np.int64)
        rows = np.arange(len(seat))
        own = vps[rows, seat]
        vps_other = vps.copy()
        vps_other[rows, seat] = -1.0
        zt = np.clip((own - vps_other.max(axis=1)) / 10.0, -1.0, 1.0)
        z = torch.from_numpy(zt.astype(np.float32)).to(device)
    elif value_target == "ab_value":
        # Learned-judge distillation: regress the recorded hybrid leaf value
        # (two-scale ab_value squash, il_dataset abv). DETERMINISTIC function
        # of the state — pure function approximation, no outcome noise.
        # MEASURED 2026-06-07: scores 9-11% vs native AB-d2 — WORSE than the
        # vp_margin head (20%). The combined-scalar MSE weights fine-channel
        # errors at 0.25^2 = 1/16, so the optimizer never fits the fine
        # discrimination that makes ab_value strong. Use ab_two_scale.
        if "abv" not in data:
            raise KeyError("value-target ab_value needs v3 shards with the "
                           "'abv' field (regenerate via il_dataset)")
        z = torch.from_numpy(np.asarray(data["abv"][idx],
                                        dtype=np.float32)).to(device)
    elif value_target in ("ab_two_scale", "ab_mixed"):
        # Learned judge v2: per-channel regression targets — tanh(vp_part/3)
        # and tanh(fine_part/abv_scale) recomputed from the RAW margin. Each
        # channel spans its own [-1,1] and gets FULL loss weight (train
        # against net.forward_channels); the deploy-time net recombines
        # 0.75/0.25 inside forward(), matching the hybrid leaf.
        # ab_mixed adds the vp_margin OUTCOME channel (decorrelated with the
        # heuristic-mimicry channels, which are info-capped by hidden enemy
        # state): forward() blends 0.5*two_scale + 0.5*outcome.
        if "abm" not in data:
            raise KeyError("value-target ab_two_scale/ab_mixed needs v4 "
                           "shards with the 'abm' field (regenerate via "
                           "il_dataset)")
        m = np.asarray(data["abm"][idx], dtype=np.float64)
        vp = np.round(m / VP_W)
        fine = m - vp * VP_W
        cols = [np.tanh(vp / 3.0), np.tanh(fine / abv_scale)]
        if value_target == "ab_mixed":
            vps = np.asarray(data["vps"][idx], dtype=np.float32)
            seat = np.asarray(data["seat"][idx], dtype=np.int64)
            rows = np.arange(len(seat))
            own = vps[rows, seat]
            vps_other = vps.copy()
            vps_other[rows, seat] = -1.0
            cols.append(np.clip((own - vps_other.max(axis=1)) / 10.0,
                                -1.0, 1.0))
        zt = np.stack(cols, axis=1)
        z = torch.from_numpy(zt.astype(np.float32)).to(device)
    elif value_target == "search_value":
        # Stage-3 LEARNED-from-strong-play target: the hybrid's MCTS root
        # value (stage3_gen rootv), already in [-1,1]. Denoised, lookahead-
        # aggregated — the AlphaZero value target, vs the sparse outcome.
        if "rootv" not in data:
            raise KeyError("value-target search_value needs stage-3 shards "
                           "with the 'rootv' field (generate via stage3_gen)")
        z = torch.from_numpy(np.asarray(data["rootv"][idx],
                                        dtype=np.float32)).to(device)
    else:
        z = torch.from_numpy(np.asarray(data["z"][idx], dtype=np.float32)).to(device)
    return obs, act, mask, z


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="models/datasets/il_ab_d1",
                   help="shard dir, or comma-separated dirs (DAgger union).")
    p.add_argument("--hidden", type=str, default="1024,1024,512")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-coef", type=float, default=1.0)
    p.add_argument("--value-target",
                   choices=["sparse", "vp_margin", "ab_value", "ab_two_scale",
                            "ab_mixed", "search_value"],
                   default="sparse",
                   help="vp_margin = dense final-VP margin (needs v2 shards "
                        "with seat field); tests the search-SNR hypothesis. "
                        "ab_value = naive single-scalar distill of the hybrid "
                        "leaf squash (v3 shards; measured WORSE than "
                        "vp_margin — 1/16 fine-channel loss weight). "
                        "ab_two_scale = the LEARNED JUDGE: per-channel "
                        "two-scale distill from the raw margin (v4 shards "
                        "with abm), value_channels=2 net, recombined in "
                        "forward() — deploy with --leaf-eval net.")
    p.add_argument("--abv-scale", type=float, default=86e6,
                   help="fine-part tanh scale for ab_two_scale targets; MUST "
                        "match the search's --ab-value-scale (gate: 86e6).")
    p.add_argument("--block-shuffle", type=int, default=0,
                   help="approximate-shuffle block size in SAMPLES (e.g. "
                        "2000000). 0 = exact full-random sampling. Use for "
                        "caches BIGGER THAN RAM: full-random batches degrade "
                        "to NVMe random reads (640k-union cache 245G ran at "
                        "7.8k smp/s = 46x slower); block mode streams one "
                        "contiguous block into RAM, shuffles within, and "
                        "permutes block order per epoch.")
    p.add_argument("--obs-width", type=int, default=0,
                   help="slice obs columns to this width (e.g. 1084 = train "
                        "a POV control on full-obs shards). 0 = native width.")
    p.add_argument("--value-hidden", type=int, default=128,
                   help="value-head hidden width (128 was the underfit "
                        "bottleneck for the two-scale fine channel).")
    p.add_argument("--value-skip-obs", action="store_true",
                   help="value head reads [trunk, raw obs] — direct input "
                        "access for value-only features the policy-shared "
                        "trunk discards.")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", type=str, default=str(CKPT_DIR / "il_ab_d1"))
    p.add_argument("--init-from", type=str, default="",
                   help="warm-start: load net_state from this checkpoint "
                        "before training (arch must match --hidden / value "
                        "head). Stage-3 fine-tunes the 640k clone on "
                        "strong-play data; use a lower --lr (e.g. 3e-4).")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = build_cache([Path(x) for x in args.data_dir.split(",") if x])
    n = data["n"]
    n_val = max(int(n * args.val_frac), args.batch_size)
    perm = rng.permutation(n)
    val_idx = np.sort(perm[:n_val])
    train_idx = perm[n_val:]
    print(f"[cfg] {n} samples ({len(train_idx)} train / {n_val} val) "
          f"hidden={args.hidden} epochs={args.epochs} device={args.device}",
          flush=True)

    hidden = tuple(int(x) for x in args.hidden.split(",") if x)
    n_channels = {"ab_two_scale": 2, "ab_mixed": 3}.get(args.value_target, 1)
    multi_v = n_channels > 1
    obs_dim = int(args.obs_width or data["obs"].shape[1])  # 1084 POV / 1132 full
    net = PolicyValueNet(obs_dim=obs_dim,
                         hidden=hidden,
                         value_channels=n_channels,
                         value_hidden=args.value_hidden,
                         value_skip_obs=args.value_skip_obs).to(args.device)
    if args.init_from:
        st = torch.load(args.init_from, map_location=args.device,
                        weights_only=False)
        missing, unexpected = net.load_state_dict(st["net_state"], strict=False)
        print(f"[init] warm-started from {args.init_from} "
              f"(missing={list(missing)}, unexpected={list(unexpected)})",
              flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = len(train_idx) // args.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch, eta_min=args.lr * 0.05)

    def evaluate_val() -> tuple[float, list[float]]:
        """Returns (top1, per-channel value MSEs) — one entry for scalar
        heads, [vp, fine] for ab_two_scale, [vp, fine, outcome] for
        ab_mixed."""
        net.eval()
        correct = tot = 0
        vsum = None
        with torch.no_grad():
            for s in range(0, n_val, args.batch_size):
                idx = val_idx[s:s + args.batch_size]
                obs, act, mask, z = _batch(data, idx, args.device,
                                           args.value_target, args.abv_scale,
                                           args.obs_width)
                logits, value = (net.forward_channels(obs) if multi_v
                                 else net(obs))
                logits = logits.masked_fill(~mask, float("-inf"))
                correct += int((logits.argmax(dim=1) == act).sum())
                tot += len(idx)
                se = (value - z) ** 2
                if se.dim() == 1:
                    se = se.unsqueeze(1)
                bsum = se.sum(dim=0)
                vsum = bsum if vsum is None else vsum + bsum
        net.train()
        return correct / tot, [float(x) / tot for x in vsum]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    arr_keys = [k for k in ("obs", "act", "mask", "z", "vps", "seat",
                            "abv", "abm") if k in data]
    train_sorted = np.sort(train_idx)

    def batch_iter(ep_seed: int):
        """Yields (data_source, idx) batches. Block mode trades exact
        shuffling for sequential IO — mandatory for bigger-than-RAM caches."""
        brng = np.random.default_rng(ep_seed)
        if args.block_shuffle > 0:
            B = args.block_shuffle
            n_blocks = (len(train_sorted) + B - 1) // B
            for b in brng.permutation(n_blocks):
                blk = train_sorted[b * B:(b + 1) * B]
                sub = {k: np.asarray(data[k][blk]) for k in arr_keys}
                local = brng.permutation(len(blk))
                for s in range(0, len(local) - args.batch_size + 1,
                               args.batch_size):
                    yield sub, local[s:s + args.batch_size]
        else:
            order = brng.permutation(train_idx)
            for s in range(0, len(order) - args.batch_size + 1,
                           args.batch_size):
                yield data, np.sort(order[s:s + args.batch_size])

    t0 = time.time()
    step = 0
    net.train()
    for ep in range(1, args.epochs + 1):
        for src, idx in batch_iter(args.seed * 1000 + ep):
            obs, act, mask, z = _batch(src, idx, args.device,
                                       args.value_target, args.abv_scale,
                                       args.obs_width)
            logits, value = (net.forward_channels(obs) if multi_v
                             else net(obs))
            logits = logits.masked_fill(~mask, float("-inf"))
            policy_loss = F.cross_entropy(logits, act)
            value_loss = F.mse_loss(value, z)
            loss = policy_loss + args.value_coef * value_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"[ep {ep} step {step}] loss={float(loss):.4f} "
                      f"p={float(policy_loss):.4f} v={float(value_loss):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"({step*args.batch_size/(time.time()-t0):.0f} smp/s)",
                      flush=True)
        acc, vmses = evaluate_val()
        ch_names = ["value-mse", "value-mse-fine", "value-mse-out"]
        msg = " ".join(f"{ch_names[i]}={m:.4f}" for i, m in enumerate(vmses))
        print(f"[ep {ep}] VAL teacher-top1={acc:.4f} {msg}", flush=True)
        ck = save_dir / f"il_ep{ep}.pt"
        torch.save({"net_state": net.state_dict(), "args": vars(args),
                    "val_top1": acc, "val_vmse": vmses}, str(ck))
        write_stamp(ck)

    final = save_dir / "il_final.pt"
    torch.save({"net_state": net.state_dict(), "args": vars(args)}, str(final))
    write_stamp(final)
    print(f"[done] {step} steps in {time.time()-t0:.0f}s -> {final}", flush=True)


if __name__ == "__main__":
    main()
