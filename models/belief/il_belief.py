"""Belief-augmented IL: imitate native AB into an actor whose input can include
hidden / belief features, to test whether the ~17% ceiling is an actor-encoding
limit (DR-001, P3-X1).

Actor-input modes (critic is always the baseline privileged 1156 = 1132 + 24
zero pool IDs, unchanged):
  * ``plain``    -> 1084  (reproduces the incumbent il_best control)
  * ``oracle48`` -> 1132  (1084 POV prefix + the TRUE 48-float hidden appendix)
                   -- a NON-DEPLOYABLE diagnostic: it reads hidden enemy state,
                   so it FAILS the leakage referee by construction. It is the
                   UPPER BOUND on what any legal belief tracker could recover,
                   and it decides cheaply whether the direction is alive.
  * ``legal``    -> 1084 + K legal belief features supplied as a sidecar memmap
                   (built by models/belief/belief_features.py; leakage-gated).

Reuses the baseline data + build_cache; identical optimizer/schedule so the only
difference across runs is the actor input. Run per mode, compare val_top1.

    python -m models.belief.il_belief --mode oracle48 \
        --train-dir <abs>/phase2_ab_d1_train --val-dir <abs>/phase2_ab_d1_val \
        --epochs 3 --run-name p3x1-oracle48
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import fastcatan
from models.alphazero.il_pretrain import build_cache

ACTOR_OBS = int(fastcatan.OBS_SIZE)        # 1084
FULL_OBS = int(fastcatan.OBS_FULL_SIZE)    # 1132
APPENDIX = FULL_OBS - ACTOR_OBS            # 48
POOL_ID = 8 * (int(fastcatan.NUM_PLAYERS) - 1)  # 24
CRITIC_OBS = FULL_OBS + POOL_ID            # 1156
ACTOR_HIDDEN = (2048, 1024, 512)
CRITIC_HIDDEN = (1024, 512)


class BeliefActorCritic(nn.Module):
    def __init__(self, actor_in: int) -> None:
        super().__init__()
        self.actor_in = actor_in
        self.actor = nn.Sequential(
            nn.LayerNorm(actor_in),
            nn.Linear(actor_in, ACTOR_HIDDEN[0]), nn.GELU(),
            nn.Linear(ACTOR_HIDDEN[0], ACTOR_HIDDEN[1]), nn.GELU(),
            nn.Linear(ACTOR_HIDDEN[1], ACTOR_HIDDEN[2]), nn.GELU(),
        )
        self.actor_head = nn.Linear(ACTOR_HIDDEN[-1], int(fastcatan.NUM_ACTIONS))
        self.critic = nn.Sequential(
            nn.Linear(CRITIC_OBS, CRITIC_HIDDEN[0]), nn.GELU(),
            nn.Linear(CRITIC_HIDDEN[0], CRITIC_HIDDEN[1]), nn.GELU(),
        )
        self.critic_head = nn.Linear(CRITIC_HIDDEN[-1], 1)

    def forward(self, actor_obs, critic_obs):
        pi = self.actor_head(self.actor(actor_obs))
        vf = self.critic_head(self.critic(critic_obs)).squeeze(-1)
        return pi, vf


def _make_batcher(mode: str, belief_train, belief_val):
    def _batch(data, idx, device, belief_side):
        raw = np.asarray(data["obs"][idx], dtype=np.float32)  # (B, 1132)
        if raw.shape[1] != FULL_OBS:
            raise ValueError(f"need full-obs shards ({FULL_OBS}), got {raw.shape[1]}")
        if mode == "plain":
            actor_np = raw[:, :ACTOR_OBS]
        elif mode == "oracle48":
            actor_np = raw[:, :FULL_OBS]
        elif mode == "legal":
            bf = np.asarray(belief_side[idx], dtype=np.float32)
            actor_np = np.concatenate([raw[:, :ACTOR_OBS], bf], axis=1)
        else:
            raise ValueError(mode)
        critic_np = np.zeros((len(idx), CRITIC_OBS), dtype=np.float32)
        critic_np[:, :FULL_OBS] = raw  # pool IDs stay zero (teacher AB, no pool id)
        actor_t = torch.from_numpy(actor_np).to(device)
        critic_t = torch.from_numpy(critic_np).to(device)
        act = torch.from_numpy(np.asarray(data["act"][idx], dtype=np.int64)).to(device)
        mask = np.unpackbits(np.asarray(data["mask"][idx]), axis=1)[:, :int(fastcatan.NUM_ACTIONS)]
        mask_t = torch.from_numpy(mask.astype(bool)).to(device)
        z = torch.from_numpy(np.asarray(data["z"][idx], dtype=np.float32)).to(device)
        return actor_t, critic_t, act, mask_t, z
    return _batch


@torch.no_grad()
def _evaluate(net, data, batch, device, batcher, belief_side):
    net.eval()
    correct = total = 0
    ce_sum = mse_sum = 0.0
    for start in range(0, int(data["n"]), batch):
        idx = np.arange(start, min(start + batch, int(data["n"])))
        ao, co, act, mask, z = batcher(data, idx, device, belief_side)
        logits, value = net(ao, co)
        logits = logits.masked_fill(~mask, float("-inf"))
        correct += int((logits.argmax(1) == act).sum())
        total += len(idx)
        ce_sum += float(F.cross_entropy(logits, act, reduction="sum"))
        mse_sum += float(F.mse_loss(value, z, reduction="sum"))
    net.train()
    return correct / total, ce_sum / total, mse_sum / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plain", "oracle48", "legal"], required=True)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--belief-train", default="", help="legal mode: sidecar .npy (N,K) aligned to train cache")
    ap.add_argument("--belief-val", default="")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-coef", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260717)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default="models/checkpoints/p3_creat_x1")
    ap.add_argument("--run-name", default="p3x1")
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.wandb_mode != "online":
        os.environ["WANDB_MODE"] = args.wandb_mode

    train = build_cache([Path(args.train_dir)])
    val = build_cache([Path(args.val_dir)])
    belief_train = belief_val = None
    if args.mode == "legal":
        belief_train = np.lib.format.open_memmap(args.belief_train, mode="r")
        belief_val = np.lib.format.open_memmap(args.belief_val, mode="r")
        assert belief_train.shape[0] == int(train["n"]), "belief/train misalignment"
        assert belief_val.shape[0] == int(val["n"]), "belief/val misalignment"
        actor_in = ACTOR_OBS + int(belief_train.shape[1])
    else:
        actor_in = ACTOR_OBS if args.mode == "plain" else FULL_OBS

    batcher = _make_batcher(args.mode, belief_train, belief_val)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    param_config = {
        "mode": args.mode, "actor_in": actor_in, "critic_in": CRITIC_OBS,
        "train_samples": int(train["n"]), "val_samples": int(val["n"]),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
    }
    run = None
    if args.wandb_mode == "online":
        try:
            import wandb
            run = wandb.init(project="goodsettler-il", entity="good-start-labs",
                             group="p3-creat-x1", name=args.run_name,
                             config=param_config, reinit=True)
            print(f"[wandb] {run.url}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[wandb] disabled: {ex!r}", flush=True)

    net = BeliefActorCritic(actor_in).to(args.device)
    n_params = sum(p.numel() for p in net.parameters())
    actor_params = sum(p.numel() for p in net.actor.parameters()) + sum(
        p.numel() for p in net.actor_head.parameters())
    print(f"[params] total={n_params} actor={actor_params} actor_in={actor_in}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, int(train["n"]) // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch, eta_min=args.lr * 0.05)
    rng = np.random.default_rng(args.seed)
    best = -math.inf
    best_metrics = {}
    started = time.time()
    gstep = 0
    net.train()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(int(train["n"]))
        for start in range(0, len(order) - args.batch_size + 1, args.batch_size):
            idx = np.sort(order[start:start + args.batch_size])
            ao, co, act, mask, z = batcher(train, idx, args.device, belief_train)
            opt.zero_grad(set_to_none=True)
            amp = args.device.startswith("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits, value = net(ao, co)
                logits = logits.masked_fill(~mask, float("-inf"))
                loss = F.cross_entropy(logits, act) + args.value_coef * F.mse_loss(value, z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            gstep += 1
        top1, ce, mse = _evaluate(net, val, args.batch_size, args.device, batcher, belief_val)
        dt = time.time() - started
        print(f"[{args.mode} epoch {epoch}] val_top1={top1:.6f} val_ce={ce:.6f} "
              f"val_value_mse={mse:.6f} elapsed={dt:.0f}s", flush=True)
        if run is not None:
            run.log({"il/val_top1": top1, "il/val_ce": ce, "il/val_value_mse": mse,
                     "il/epoch": epoch}, step=gstep)
        if top1 > best:
            best = top1
            best_metrics = {"mode": args.mode, "epoch": epoch, "val_top1": top1,
                            "val_ce": ce, "val_value_mse": mse, "actor_in": actor_in}
            torch.save({"actor_body": net.actor.state_dict(),
                        "actor_head": net.actor_head.state_dict(),
                        "critic_body": net.critic.state_dict(),
                        "critic_head": net.critic_head.state_dict(),
                        "meta": best_metrics}, save_dir / f"belief_{args.mode}_best.pt")
    result = {**param_config, "best_val_top1": best, "best": best_metrics,
              "wall_seconds": time.time() - started}
    (save_dir / f"result_{args.mode}.json").write_text(json.dumps(result, indent=2))
    print(f"[done] mode={args.mode} best_val_top1={best:.6f} "
          f"wall={result['wall_seconds']:.0f}s", flush=True)
    if run is not None:
        run.summary.update({"best_val_top1": best})
        run.finish()


if __name__ == "__main__":
    main()
