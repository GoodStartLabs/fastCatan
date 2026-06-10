"""DQN (Q-Learning with a neural function approximator) for Catan — multi-env.

Reference: Mnih et al. 2015 ("Human-level control through deep RL").

Pieces:
  - Q-network: MLP(obs) -> Q-values for every action.
  - Target network: periodically-copied snapshot used to compute TD targets
    (stabilizes training; without it the target moves with the gradient).
  - Replay buffer: numpy ring of past transitions, sampled uniformly so updates
    are decorrelated.
  - Epsilon-greedy exploration: with prob eps pick a uniform legal action,
    else the legal action with max Q. Eps decays linearly over env steps.
  - Action masking: illegal-action Q-values set to -inf before argmax / in the
    target, so they can never be picked or contribute to the bootstrap.

MULTI-ENV (vs the original single-env reference): `--num-envs` FastCatanEnvs are
stepped together each tick (synchronous, single process — the C++ sim is cheap),
their transitions pushed to one shared buffer, and the Q-net runs on the GPU.
This is what makes a 50M-step run finish in hours instead of ~40h: the per-tick
batched action selection amortizes the Python/GPU overhead, and the replay ratio
drops to gradient_steps/(num_envs) instead of 1 update per env step.

Run (50M steps vs random, no p2p trades):
    python -m models.train_dqn --num-envs 16 --total-steps 50_000_000 --no-trades

`--total-steps` counts TOTAL env steps across all envs (so it is comparable to
the PPO/A2C budgets).
"""
from __future__ import annotations

import argparse
import time
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
    """Fixed-size numpy ring buffer. O(1) push, O(batch) uniform sample —
    avoids the deque+random.sample O(n) sampling of the single-env version."""

    def __init__(self, capacity: int, obs_dim: int, n_actions: int):
        self.cap = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.nobs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.nmask = np.zeros((capacity, n_actions), dtype=bool)
        self.pos = 0
        self.size = 0

    def push_batch(self, obs, act, rew, nobs, done, nmask) -> None:
        b = len(act)
        idx = (self.pos + np.arange(b)) % self.cap
        self.obs[idx] = obs
        self.nobs[idx] = nobs
        self.act[idx] = act
        self.rew[idx] = rew
        self.done[idx] = done
        self.nmask[idx] = nmask
        self.pos = (self.pos + b) % self.cap
        self.size = min(self.size + b, self.cap)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.nobs[idx], self.done[idx], self.nmask[idx])

    def __len__(self) -> int:
        return self.size


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=16,
                   help="Parallel envs stepped per tick (synchronous, 1 process).")
    p.add_argument("--total-steps", type=int, default=50_000_000,
                   help="Total env steps across ALL envs (comparable to PPO/A2C).")
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=10_000,
                   help="Env steps of pure collection before training starts.")
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gradient-steps", type=int, default=1,
                   help="Q-net updates per tick (one tick = num_envs env steps). "
                        "Replay ratio = gradient_steps / num_envs.")
    p.add_argument("--target-update", type=int, default=2_000,
                   help="Copy online->target every N gradient updates.")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=1_000_000,
                   help="Env steps over which epsilon decays start->end.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default=str(CKPT_DIR / "dqn"))
    p.add_argument("--save-freq", type=int, default=5_000_000,
                   help="Checkpoint every N env steps (0 = only final).")
    p.add_argument("--no-trades", action="store_true",
                   help="Suppress player-to-player trades (learner AND opponents; "
                        "maritime stays) — the canonical thesis ablation.")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"])
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    n = args.num_envs
    envs = [FastCatanEnv(seed=args.seed + i, suppress_p2p_trade=args.no_trades)
            for i in range(n)]
    cur_obs = [e.reset()[0] for e in envs]
    cur_masks = [e.action_masks() for e in envs]

    q = QNet().to(device)
    q_target = QNet().to(device)
    q_target.load_state_dict(q.state_dict())
    q_target.eval()
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    buf = ReplayBuffer(args.buffer_size, OBS_SIZE, NUM_ACTIONS)

    ep_return = [0.0] * n
    recent: deque = deque(maxlen=200)
    grad_updates = 0
    total_env_steps = 0
    next_save = args.save_freq
    t0 = time.time()
    last_log = 0

    print(f"[train] dqn num_envs={n} total_steps={args.total_steps} "
          f"device={device} lr={args.lr} grad_steps/tick={args.gradient_steps} "
          f"replay_ratio={args.gradient_steps / n:.3f} "
          f"trades={'off' if args.no_trades else 'on'}")

    def save(path: Path) -> None:
        cpu_state = {k: v.cpu() for k, v in q.state_dict().items()}
        torch.save({"q_state": cpu_state, "args": vars(args)}, str(path))
        write_stamp(path)

    while total_env_steps < args.total_steps:
        # ---- Vectorized epsilon-greedy action selection ----
        frac = min(1.0, total_env_steps / args.eps_decay_steps)
        eps = args.eps_start + (args.eps_end - args.eps_start) * frac

        obs_arr = np.stack(cur_obs)
        mask_arr = np.stack(cur_masks)
        with torch.no_grad():
            qv = q(torch.from_numpy(obs_arr).to(device))
            mt = torch.from_numpy(mask_arr).to(device)
            qv = qv.masked_fill(~mt, float("-inf"))
            greedy = qv.argmax(dim=1).cpu().numpy()

        explore = rng.random(n) < eps
        acts = np.empty(n, dtype=np.int64)
        for i in range(n):
            if explore[i]:
                legal = np.nonzero(mask_arr[i])[0]
                acts[i] = legal[rng.integers(len(legal))] if len(legal) else 0
            else:
                acts[i] = greedy[i]

        # ---- Step all envs, collect transitions ----
        b_obs = obs_arr  # next_obs/mask filled below
        b_nobs = np.empty_like(obs_arr)
        b_nmask = np.empty_like(mask_arr)
        b_rew = np.empty(n, dtype=np.float32)
        b_done = np.empty(n, dtype=np.float32)
        for i in range(n):
            a = int(acts[i])
            nobs, r, done, _trunc, _info = envs[i].step(a)
            nmask = envs[i].action_masks()  # terminal mask if done (done zeroes it)
            b_nobs[i] = nobs
            b_nmask[i] = nmask
            b_rew[i] = r
            b_done[i] = float(done)
            ep_return[i] += r
            if done:
                recent.append(ep_return[i])
                ep_return[i] = 0.0
                nobs, _ = envs[i].reset()
                nmask = envs[i].action_masks()
            cur_obs[i] = nobs
            cur_masks[i] = nmask

        buf.push_batch(b_obs, acts, b_rew, b_nobs, b_done, b_nmask)
        total_env_steps += n

        # ---- Training ----
        if len(buf) >= max(args.warmup, args.batch_size):
            for _ in range(args.gradient_steps):
                o, a, r, no, d, nm = buf.sample(args.batch_size)
                o = torch.from_numpy(o).to(device)
                no = torch.from_numpy(no).to(device)
                a = torch.from_numpy(a).to(device)
                r = torch.from_numpy(r).to(device)
                d = torch.from_numpy(d).to(device)
                nm = torch.from_numpy(nm).to(device)
                with torch.no_grad():
                    nq = q_target(no).masked_fill(~nm, float("-inf"))
                    best = nq.max(dim=1).values
                    best = torch.where(torch.isfinite(best), best,
                                       torch.zeros_like(best))
                    target = r + args.gamma * (1.0 - d) * best
                pred = q(o).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(pred, target)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                grad_updates += 1
                if grad_updates % args.target_update == 0:
                    q_target.load_state_dict(q.state_dict())

        # ---- Logging ----
        if total_env_steps - last_log >= 50_000:
            last_log = total_env_steps
            sps = total_env_steps / (time.time() - t0)
            mean_r = (sum(recent) / len(recent)) if recent else 0.0
            print(f"[step {total_env_steps:>9d}] eps={eps:.3f} buf={len(buf)} "
                  f"updates={grad_updates} sps={sps:.0f} "
                  f"mean_ep_return(200)={mean_r:+.3f}")

        # ---- Periodic checkpoint ----
        if args.save_freq and total_env_steps >= next_save:
            ckpt = save_dir / f"dqn_{total_env_steps}.pt"
            save(ckpt)
            print(f"[train] checkpoint -> {ckpt}")
            next_save += args.save_freq

    final = save_dir / "dqn_final.pt"
    save(final)
    print(f"[train] saved -> {final}")


if __name__ == "__main__":
    main()
