"""Behavior-clone native AB into the fixed Phase-2 actor and critic.

Training and validation directories are generated independently so held-out
metrics are board-disjoint.  Actor CE reads only the first 1,084 columns of
``write_obs_full`` shards.  The critic reads all 1,132 columns plus all-zero
pool IDs: AB-d1 is intentionally not assigned a pool-v1 identity before the
sprint gate.
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
from models.baseline.policy import (
    ACTOR_OBS_SIZE,
    CRITIC_OBS_SIZE,
    FULL_OBS_SIZE,
    POOL_ID_SIZE,
    SplitMlpExtractor,
)
from models.ckpt import write_stamp


class ILActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.split = SplitMlpExtractor()
        self.actor_head = nn.Linear(self.split.latent_dim_pi, fastcatan.NUM_ACTIONS)
        self.critic_head = nn.Linear(self.split.latent_dim_vf, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pi, vf = self.split(obs)
        return self.actor_head(pi), self.critic_head(vf).squeeze(-1)


def _batch(data: dict, idx: np.ndarray, device: str):
    raw = np.asarray(data["obs"][idx], dtype=np.float32)
    if raw.shape[1] != FULL_OBS_SIZE:
        raise ValueError(
            f"Phase-2 critic needs --full-obs shards ({FULL_OBS_SIZE}), got {raw.shape[1]}"
        )
    combined = np.zeros(
        (len(idx), ACTOR_OBS_SIZE + CRITIC_OBS_SIZE), dtype=np.float32,
    )
    combined[:, :ACTOR_OBS_SIZE] = raw[:, :ACTOR_OBS_SIZE]
    combined[:, ACTOR_OBS_SIZE:ACTOR_OBS_SIZE + FULL_OBS_SIZE] = raw
    # The final POOL_ID_SIZE columns stay zero: teacher AB is outside pool v1.
    assert combined.shape[1] == ACTOR_OBS_SIZE + FULL_OBS_SIZE + POOL_ID_SIZE
    obs = torch.from_numpy(combined).to(device)
    act = torch.from_numpy(
        np.asarray(data["act"][idx], dtype=np.int64)
    ).to(device)
    mask = np.unpackbits(
        np.asarray(data["mask"][idx]), axis=1,
    )[:, :fastcatan.NUM_ACTIONS]
    mask_t = torch.from_numpy(mask.astype(bool)).to(device)
    outcome = torch.from_numpy(
        np.asarray(data["z"][idx], dtype=np.float32)
    ).to(device)
    return obs, act, mask_t, outcome


def _evaluate(
    net: ILActorCritic,
    data: dict,
    batch_size: int,
    device: str,
) -> tuple[float, float, float]:
    net.eval()
    correct = total = 0
    ce_sum = mse_sum = 0.0
    with torch.no_grad():
        for start in range(0, int(data["n"]), batch_size):
            idx = np.arange(start, min(start + batch_size, int(data["n"])))
            obs, act, mask, z = _batch(data, idx, device)
            logits, value = net(obs)
            logits = logits.masked_fill(~mask, float("-inf"))
            correct += int((logits.argmax(1) == act).sum())
            total += len(idx)
            ce_sum += float(F.cross_entropy(logits, act, reduction="sum"))
            mse_sum += float(F.mse_loss(value, z, reduction="sum"))
    net.train()
    return correct / total, ce_sum / total, mse_sum / total


def _save(net: ILActorCritic, path: Path, metadata: dict) -> None:
    torch.save({
        "actor_body": net.split.actor.state_dict(),
        "actor_head": net.actor_head.state_dict(),
        "critic_body": net.split.critic.state_dict(),
        "critic_head": net.critic_head.state_dict(),
        **metadata,
    }, path)
    write_stamp(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="models/checkpoints/phase2_il")
    parser.add_argument("--run-name", default="phase2-il-ab-d1")
    parser.add_argument("--gate-top1", type=float, default=0.85)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.wandb_mode != "online":
        os.environ["WANDB_MODE"] = args.wandb_mode

    train_data = build_cache([Path(args.train_dir)])
    val_data = build_cache([Path(args.val_dir)])
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = json.loads((Path(args.train_dir) / "manifest.json").read_text())
    val_manifest = json.loads((Path(args.val_dir) / "manifest.json").read_text())
    config = vars(args) | {
        "train_games": train_manifest["games"],
        "val_games": val_manifest["games"],
        "train_samples": int(train_data["n"]),
        "val_samples": int(val_data["n"]),
        "actor_input": ACTOR_OBS_SIZE,
        "critic_input": CRITIC_OBS_SIZE,
    }

    import wandb
    run = wandb.init(project="goodsettler", name=args.run_name, config=config, reinit=True)
    print(f"[wandb] {run.url}", flush=True)

    net = ILActorCritic().to(args.device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, int(train_data["n"]) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * steps_per_epoch,
        eta_min=args.lr * 0.05,
    )
    rng = np.random.default_rng(args.seed)
    best_top1 = -math.inf
    best_metrics: dict[str, float] = {}
    started = time.time()
    global_step = 0
    net.train()

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(int(train_data["n"]))
        for start in range(0, len(order) - args.batch_size + 1, args.batch_size):
            idx = np.sort(order[start:start + args.batch_size])
            obs, act, mask, z = _batch(train_data, idx, args.device)
            optimizer.zero_grad(set_to_none=True)
            amp = args.device.startswith("cuda")
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=amp,
            ):
                logits, value = net(obs)
                logits = logits.masked_fill(~mask, float("-inf"))
                policy_loss = F.cross_entropy(logits, act)
                value_loss = F.mse_loss(value, z)
                loss = policy_loss + args.value_coef * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

        top1, val_ce, val_mse = _evaluate(
            net, val_data, args.batch_size, args.device,
        )
        metrics = {
            "il/val_top1": top1,
            "il/val_ce": val_ce,
            "il/val_value_mse": val_mse,
            "il/epoch": epoch,
            "il/lr": scheduler.get_last_lr()[0],
            "il/samples_per_s": (
                epoch * int(train_data["n"]) / max(time.time() - started, 1e-9)
            ),
        }
        run.log(metrics, step=global_step)
        print(
            f"[epoch {epoch}] val_top1={top1:.6f} val_ce={val_ce:.6f} "
            f"val_value_mse={val_mse:.6f}",
            flush=True,
        )
        metadata = {
            "config": config,
            "epoch": epoch,
            "val_top1": top1,
            "val_ce": val_ce,
            "val_value_mse": val_mse,
            "wandb_url": run.url,
        }
        _save(net, save_dir / f"il_ep{epoch}.pt", metadata)
        if top1 > best_top1:
            best_top1 = top1
            best_metrics = metadata
            _save(net, save_dir / "il_best.pt", metadata)

    _save(net, save_dir / "il_final.pt", best_metrics)
    gate_pass = best_top1 >= args.gate_top1
    run.summary["best_val_top1"] = best_top1
    run.summary["gate_top1"] = args.gate_top1
    run.summary["top1_gate_pass"] = gate_pass
    run.summary["wall_seconds"] = time.time() - started
    print(
        f"[gate] heldout_top1={best_top1:.6f} threshold={args.gate_top1:.2f} "
        f"pass={gate_pass} wandb={run.url}",
        flush=True,
    )
    run.finish()
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
