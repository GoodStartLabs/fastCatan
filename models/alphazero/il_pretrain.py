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
        return out
    shards = sorted(s for d in data_dirs
                    for s in glob.glob(str(d / "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"no shards under {data_dirs}")
    counts = []
    has_abv = True
    for s in shards:
        with np.load(s) as d:
            counts.append(d["act"].shape[0])
            has_abv = has_abv and ("abv" in d.files)
    n = int(sum(counts))
    cache.mkdir(exist_ok=True)
    obs = np.lib.format.open_memmap(cache / "obs.npy", mode="w+",
                                    dtype=np.float16, shape=(n, OBS_SIZE))
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
        o += c
        print(f"[cache] {o}/{n}", flush=True)
    obs.flush(); act.flush(); mask.flush(); z.flush()
    vps.flush(); seat.flush()
    if abv is not None:
        abv.flush()
    np.savez(meta, n=n)
    print(f"[cache] built: {n} samples", flush=True)
    return build_cache(data_dirs)


def _batch(data, idx, device, value_target="sparse"):
    obs = torch.from_numpy(np.asarray(data["obs"][idx], dtype=np.float32)).to(device)
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
        if "abv" not in data:
            raise KeyError("value-target ab_value needs v3 shards with the "
                           "'abv' field (regenerate via il_dataset)")
        z = torch.from_numpy(np.asarray(data["abv"][idx],
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
    p.add_argument("--value-target", choices=["sparse", "vp_margin", "ab_value"],
                   default="sparse",
                   help="vp_margin = dense final-VP margin (needs v2 shards "
                        "with seat field); tests the search-SNR hypothesis. "
                        "ab_value = distill the hybrid leaf squash (needs v3 "
                        "shards with abv field) — the LEARNED JUDGE: replaces "
                        "the symbolic ab_value leaves at search time with the "
                        "value head (--leaf-eval net).")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", type=str, default=str(CKPT_DIR / "il_ab_d1"))
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
    net = PolicyValueNet(hidden=hidden).to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = len(train_idx) // args.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch, eta_min=args.lr * 0.05)

    def evaluate_val() -> tuple[float, float]:
        net.eval()
        correct = tot = 0
        vmse = 0.0
        with torch.no_grad():
            for s in range(0, n_val, args.batch_size):
                idx = val_idx[s:s + args.batch_size]
                obs, act, mask, z = _batch(data, idx, args.device,
                                           args.value_target)
                logits, value = net(obs)
                logits = logits.masked_fill(~mask, float("-inf"))
                correct += int((logits.argmax(dim=1) == act).sum())
                tot += len(idx)
                vmse += float(F.mse_loss(value, z, reduction="sum"))
        net.train()
        return correct / tot, vmse / tot

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    step = 0
    net.train()
    for ep in range(1, args.epochs + 1):
        order = rng.permutation(train_idx)
        for s in range(0, len(order) - args.batch_size + 1, args.batch_size):
            idx = np.sort(order[s:s + args.batch_size])
            obs, act, mask, z = _batch(data, idx, args.device,
                                       args.value_target)
            logits, value = net(obs)
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
        acc, vmse = evaluate_val()
        print(f"[ep {ep}] VAL teacher-top1={acc:.4f} value-mse={vmse:.4f}",
              flush=True)
        ck = save_dir / f"il_ep{ep}.pt"
        torch.save({"net_state": net.state_dict(), "args": vars(args),
                    "val_top1": acc, "val_vmse": vmse}, str(ck))
        write_stamp(ck)

    final = save_dir / "il_final.pt"
    torch.save({"net_state": net.state_dict(), "args": vars(args)}, str(final))
    write_stamp(final)
    print(f"[done] {step} steps in {time.time()-t0:.0f}s -> {final}", flush=True)


if __name__ == "__main__":
    main()
