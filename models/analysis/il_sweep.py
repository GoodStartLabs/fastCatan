"""IL architecture / parameter-budget sweep on the 640k AB-d2 teacher set.

Measurement harness for the program's open "parameter budget" question:
holds the imitation objective, data, optimizer budget and batch order fixed
and varies only the trunk width/depth, so held-out teacher top-1 (and value
MSE) can be read as a function of parameter count. This ranks *priors* for
future search-generation passes; it is NOT a playing-strength eval.

Reuses il_pretrain's `_batch` (identical masked-CE loss + illegal-logit
masking + ab_two_scale value channels) and net.PolicyValueNet verbatim —
only the training loop is wrapped in a config loop with a fixed step budget.

The 640k cache (obs float16 240G) is bigger than RAM, so we do NOT stream the
whole thing: a fixed contiguous TRAIN pool and a disjoint held-out EVAL pool
are read once into RAM (shards are concatenated in game order, so disjoint
index ranges = disjoint games = no leakage). Every config then trains on the
identical in-RAM batches.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.alphazero.net import PolicyValueNet
from models.alphazero.il_pretrain import _batch
from models.ckpt import write_stamp

DEFAULT_CACHE = ("/home/ubuntu/goodSettler/wt-repro/models/datasets/"
                 "repro_3_1_il_ab_d2_640k/cache")
CACHE = Path(DEFAULT_CACHE)  # overridden by --cache in main()

# (name, hidden tuple, lr). Base sweep spans ~1M..20M params at 3 layers, plus
# a deeper 4-layer variant at ~5M and ~10M. The LR variant on the best size is
# appended programmatically after the base sweep.
CONFIGS = [
    ("d3_w448_1M",   (448, 448, 448),          1e-3),
    ("d3_w768_2p5M", (768, 768, 768),          1e-3),
    ("d3_w1280_5M",  (1280, 1280, 1280),       1e-3),
    ("d3_w1920_10M", (1920, 1920, 1920),       1e-3),
    ("d3_w2816_20M", (2816, 2816, 2816),       1e-3),
    ("d4_w1024_5M",  (1024, 1024, 1024, 1024), 1e-3),
    ("d4_w1536_10M", (1536, 1536, 1536, 1536), 1e-3),
]


def load_pool(keys, start, n, label):
    t = time.time()
    out = {}
    for k in keys:
        a = np.lib.format.open_memmap(CACHE / f"{k}.npy", mode="r")
        out[k] = np.ascontiguousarray(a[start:start + n])
    print(f"[pool:{label}] {n} rows from {start} in {time.time()-t:.0f}s "
          f"(obs {out['obs'].nbytes/1e9:.1f}G)", flush=True)
    return out


def evaluate(net, data, device, batch_size, multi_v):
    net.eval()
    n = data["obs"].shape[0]
    correct = tot = 0
    vsum = None
    with torch.no_grad():
        for s in range(0, n, batch_size):
            idx = np.arange(s, min(s + batch_size, n))
            obs, act, mask, z = _batch(data, idx, device, "ab_two_scale")
            logits, value = net(obs)   # forward() recombines channels -> scalar
            logits = logits.masked_fill(~mask, float("-inf"))
            correct += int((logits.argmax(dim=1) == act).sum())
            tot += len(idx)
            # combined-scalar value MSE against recombined two-scale target
            _, vchan = net.forward_channels(obs)
            w = net.TWO_SCALE_W
            ztgt = w[0] * z[:, 0] + w[1] * z[:, 1]
            se = (value - ztgt) ** 2
            vsum = float(se.sum()) if vsum is None else vsum + float(se.sum())
    net.train()
    return correct / tot, vsum / tot


def train_config(name, hidden, lr, train, val, args, device):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = PolicyValueNet(obs_dim=train["obs"].shape[1], hidden=hidden,
                         value_channels=2, value_hidden=args.value_hidden).to(device)
    nparams = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=lr * 0.05)

    run = None
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(project="goodsettler-il", entity=os.environ.get("WANDB_ENTITY"),
                             group="il-sweep-640k", name=name, reinit=True,
                             config=dict(hidden=hidden, lr=lr, params=nparams,
                                         steps=args.steps, batch_size=args.batch_size,
                                         value_target="ab_two_scale",
                                         train_pool=train["obs"].shape[0]))
        except Exception as e:  # W&B must never abort a training config
            print(f"[{name}] wandb init failed: {e}", flush=True)
            run = None

    def wlog(d):
        if run is not None:
            try:
                run.log(d)
            except Exception:
                pass

    npool = train["obs"].shape[0]
    print(f"[{name}] params={nparams/1e6:.2f}M hidden={hidden} lr={lr:g}", flush=True)
    t0 = time.time()
    net.train()
    step = 0
    order = rng.permutation(npool)
    pos = 0
    while step < args.steps:
        if pos + args.batch_size > npool:
            order = rng.permutation(npool)
            pos = 0
        idx = np.sort(order[pos:pos + args.batch_size])
        pos += args.batch_size
        obs, act, mask, z = _batch(train, idx, device, "ab_two_scale")
        logits, value = net.forward_channels(obs)
        logits = logits.masked_fill(~mask, float("-inf"))
        policy_loss = F.cross_entropy(logits, act)
        value_loss = F.mse_loss(value, z)
        loss = policy_loss + args.value_coef * value_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step()
        step += 1
        if step % 200 == 0:
            sps = step * args.batch_size / (time.time() - t0)
            print(f"[{name} step {step}/{args.steps}] loss={float(loss):.4f} "
                  f"p={float(policy_loss):.4f} v={float(value_loss):.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} ({sps:.0f} smp/s)", flush=True)
            wlog({"step": step, "loss": float(loss),
                  "policy_loss": float(policy_loss),
                  "value_loss": float(value_loss), "smp_s": sps})
        if step % args.eval_every == 0 or step == args.steps:
            acc, vmse = evaluate(net, val, device, args.batch_size, True)
            print(f"[{name} step {step}] VAL top1={acc:.4f} vmse={vmse:.4f}", flush=True)
            wlog({"step": step, "val_top1": acc, "val_vmse": vmse})

    acc, vmse = evaluate(net, val, device, args.batch_size, True)
    dur = time.time() - t0
    print(f"[{name}] DONE params={nparams/1e6:.3f}M top1={acc:.4f} "
          f"vmse={vmse:.4f} in {dur:.0f}s", flush=True)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    ck = save_dir / f"{name}.pt"
    torch.save({"net_state": net.state_dict(),
                "args": dict(hidden=hidden, lr=lr, value_channels=2,
                             value_hidden=args.value_hidden,
                             value_target="ab_two_scale", steps=args.steps),
                "val_top1": acc, "val_vmse": vmse}, str(ck))
    write_stamp(ck)
    if run is not None:
        try:
            run.summary["val_top1"] = acc
            run.summary["val_vmse"] = vmse
            run.summary["params"] = nparams
            run.finish()
        except Exception:
            pass
    return dict(name=name, params=nparams, hidden=hidden, lr=lr,
                top1=acc, vmse=vmse, seconds=dur, ckpt=str(ck))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=9000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--train-pool", type=int, default=8_000_000)
    p.add_argument("--eval-pool", type=int, default=400_000)
    p.add_argument("--eval-start", type=int, default=100_000_000)
    p.add_argument("--eval-every", type=int, default=3000)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-coef", type=float, default=1.0)
    p.add_argument("--value-hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", type=str, default="models/checkpoints/il_sweep_640k")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--cache", type=str, default=DEFAULT_CACHE)
    p.add_argument("--smoke", action="store_true", help="tiny run to measure throughput")
    args = p.parse_args()

    global CACHE
    CACHE = Path(args.cache)
    torch.set_num_threads(2)  # keep CPU footprint off the stage3 job's cores

    device = "cuda" if torch.cuda.is_available() else "cpu"
    keys = ["obs", "act", "mask", "abm"]
    train = load_pool(keys, 0, args.train_pool, "train")
    val = load_pool(keys, args.eval_start, args.eval_pool, "val")

    configs = list(CONFIGS)
    if args.smoke:
        args.steps = 60
        args.eval_every = 60
        configs = [("smoke_w1280", (1280, 1280, 1280), 1e-3),
                   ("smoke_w2816", (2816, 2816, 2816), 1e-3)]

    results = []
    for name, hidden, lr in configs:
        results.append(train_config(name, hidden, lr, train, val, args, device))

    # LR variant on the best (highest top1) base size.
    if not args.smoke:
        best = max(results, key=lambda r: r["top1"])
        alt_lr = best["lr"] * 3
        r = train_config(f"lrx3_{best['name']}", best["hidden"], alt_lr,
                         train, val, args, device)
        results.append(r)

    print("\n===== SWEEP SUMMARY (sorted by params) =====", flush=True)
    print(f"{'name':18s} {'params':>10s} {'top1':>8s} {'vmse':>8s} {'sec':>6s}", flush=True)
    for r in sorted(results, key=lambda r: r["params"]):
        print(f"{r['name']:18s} {r['params']/1e6:9.3f}M {r['top1']:8.4f} "
              f"{r['vmse']:8.4f} {r['seconds']:6.0f}", flush=True)
    import json
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.save_dir) / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] summary -> {args.save_dir}/summary.json", flush=True)


if __name__ == "__main__":
    main()
