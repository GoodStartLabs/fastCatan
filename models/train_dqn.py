"""Minimal DQN (Q-Learning with neural function approximator) for Catan.

Reference: Mnih et al. 2015 ("Human-level control through deep RL").

Pieces:
  - Q-network: MLP(obs) -> Q-values for every action.
  - Target network: periodically-copied snapshot used to compute TD targets
    (stabilizes training; without it the target moves with the gradient).
  - Replay buffer: ring of past transitions, sampled uniformly so updates
    are decorrelated.
  - Epsilon-greedy exploration: with prob eps pick a uniform legal action,
    else pick the legal action with max Q. Eps decays linearly.
  - Action masking: illegal action Q-values are set to -inf before argmax
    so they can never be picked or contribute to the target.

Run:
    python -m models.train_dqn --total-steps 1_000_000

This is the simplest of the four algos here. Single env, no parallelism.
"""
from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.env import FastCatanEnv, NUM_ACTIONS, OBS_SIZE
from models.ckpt import write_stamp


CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


class QNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_SIZE, n_actions: int = NUM_ACTIONS,
                 hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ReplayBuffer:
    """Fixed-size ring buffer of (obs, action, reward, next_obs, done, next_mask)."""

    def __init__(self, capacity: int):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done, next_mask):
        self.buf.append((obs, action, reward, next_obs, done, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nobs, done, nmask = zip(*batch)
        return (
            torch.from_numpy(np.stack(obs)).float(),
            torch.tensor(act, dtype=torch.long),
            torch.tensor(rew, dtype=torch.float32),
            torch.from_numpy(np.stack(nobs)).float(),
            torch.tensor(done, dtype=torch.float32),
            torch.from_numpy(np.stack(nmask)).bool(),
        )

    def __len__(self) -> int:
        return len(self.buf)


def epsilon_greedy(q_values: np.ndarray, mask: np.ndarray, eps: float,
                   rng: random.Random) -> int:
    legal = np.where(mask)[0]
    if len(legal) == 0:
        return 0
    if rng.random() < eps:
        return int(rng.choice(legal.tolist()))
    masked_q = np.where(mask, q_values, -np.inf)
    return int(np.argmax(masked_q))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=5_000)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--target-update", type=int, default=2_000)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default=str(CKPT_DIR / "dqn"))
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    env = FastCatanEnv(seed=args.seed)
    q = QNet()
    q_target = QNet()
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    buf = ReplayBuffer(args.buffer_size)

    obs, _ = env.reset()
    mask = env.action_masks()
    ep_return = 0.0
    ep_len = 0
    ep_rewards: deque = deque(maxlen=100)

    for step in range(1, args.total_steps + 1):
        frac = min(1.0, step / args.eps_decay_steps)
        eps = args.eps_start + (args.eps_end - args.eps_start) * frac

        with torch.no_grad():
            q_vals = q(torch.from_numpy(obs).float().unsqueeze(0))[0].numpy()
        action = epsilon_greedy(q_vals, mask, eps, rng)

        next_obs, reward, done, _trunc, _info = env.step(action)
        next_mask = env.action_masks()
        buf.push(obs, action, reward, next_obs, float(done), next_mask)

        obs = next_obs
        mask = next_mask
        ep_return += reward
        ep_len += 1

        if done:
            ep_rewards.append(ep_return)
            ep_return = 0.0
            ep_len = 0
            obs, _ = env.reset()
            mask = env.action_masks()

        # Training step
        if len(buf) >= max(args.warmup, args.batch_size):
            b_obs, b_act, b_rew, b_nobs, b_done, b_nmask = buf.sample(args.batch_size)
            with torch.no_grad():
                next_q = q_target(b_nobs)
                next_q = next_q.masked_fill(~b_nmask, float("-inf"))
                # If a state has no legal actions (terminal), use 0.
                best_next = next_q.max(dim=1).values
                best_next = torch.where(torch.isfinite(best_next),
                                        best_next, torch.zeros_like(best_next))
                target = b_rew + args.gamma * (1.0 - b_done) * best_next

            pred = q(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()

            if step % args.target_update == 0:
                q_target.load_state_dict(q.state_dict())

        if step % 5_000 == 0:
            mean_r = (sum(ep_rewards) / len(ep_rewards)) if ep_rewards else 0.0
            print(f"[step {step:>8d}] eps={eps:.3f} buf={len(buf)} "
                  f"mean_ep_return(100)={mean_r:+.3f}")

    final = save_dir / "dqn_final.pt"
    torch.save({"q_state": q.state_dict(), "args": vars(args)}, str(final))
    write_stamp(final)
    print(f"[train] saved -> {final}")


if __name__ == "__main__":
    main()
