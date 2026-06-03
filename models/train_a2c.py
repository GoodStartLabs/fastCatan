"""Minimal synchronous Advantage Actor-Critic (A2C) for Catan.

Reference: Mnih et al. 2016 ("Asynchronous Methods for Deep RL", sync variant).

Pieces:
  - Shared MLP trunk feeding two heads:
      * Policy head -> logits over actions (softmax = policy pi(a|s))
      * Value head  -> scalar V(s)
  - N parallel envs collect rollouts of length T (synchronous, single process here).
  - After T steps compute n-step advantage A = R - V(s) where R is the
    bootstrapped n-step return.
  - Loss = policy_loss + value_coef * value_loss - ent_coef * entropy
      policy_loss = -mean( log pi(a|s) * A.detach() )
      value_loss  = mean( (R - V(s))^2 )
      entropy     = mean( H(pi) )  (encourages exploration)
  - Action masking: logits[~mask] = -inf before softmax -> illegal actions
    have zero probability and don't appear in log_prob/entropy.

Run:
    python -m models.train_a2c --num-envs 8 --total-steps 2_000_000

Conceptually the simplest policy-gradient method that uses a baseline.
PPO is A2C + clipped surrogate objective + multiple epochs per rollout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.env import FastCatanEnv, NUM_ACTIONS, OBS_SIZE
from models.ckpt import write_stamp


CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_SIZE, n_actions: int = NUM_ACTIONS,
                 hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor):
    """Categorical over logits with -inf where mask is False."""
    neg_inf = torch.full_like(logits, float("-inf"))
    masked = torch.where(mask, logits, neg_inf)
    return torch.distributions.Categorical(logits=masked)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--rollout-len", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default=str(CKPT_DIR / "a2c"))
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    envs = [FastCatanEnv(seed=args.seed + i) for i in range(args.num_envs)]
    obs = np.stack([e.reset()[0] for e in envs])
    masks = np.stack([e.action_masks() for e in envs])

    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    ep_returns = [0.0] * args.num_envs
    recent_returns: list[float] = []

    total_env_steps = 0
    update_idx = 0
    while total_env_steps < args.total_steps:
        # ---- Rollout collection ----
        T = args.rollout_len
        buf_obs = np.zeros((T, args.num_envs, OBS_SIZE), dtype=np.float32)
        buf_mask = np.zeros((T, args.num_envs, NUM_ACTIONS), dtype=bool)
        buf_act = np.zeros((T, args.num_envs), dtype=np.int64)
        buf_logp = torch.zeros(T, args.num_envs)
        buf_val = torch.zeros(T, args.num_envs)
        buf_rew = np.zeros((T, args.num_envs), dtype=np.float32)
        buf_done = np.zeros((T, args.num_envs), dtype=np.float32)

        for t in range(T):
            obs_t = torch.from_numpy(obs).float()
            mask_t = torch.from_numpy(masks)
            logits, value = net(obs_t)
            dist = masked_categorical(logits, mask_t)
            action = dist.sample()
            logp = dist.log_prob(action)

            buf_obs[t] = obs
            buf_mask[t] = masks
            buf_act[t] = action.numpy()
            buf_logp[t] = logp
            buf_val[t] = value

            for i, e in enumerate(envs):
                no, r, done, _trunc, _info = e.step(int(action[i].item()))
                ep_returns[i] += r
                buf_rew[t, i] = r
                buf_done[t, i] = float(done)
                if done:
                    recent_returns.append(ep_returns[i])
                    if len(recent_returns) > 100:
                        recent_returns.pop(0)
                    ep_returns[i] = 0.0
                    no, _ = e.reset()
                obs[i] = no
                masks[i] = e.action_masks()

            total_env_steps += args.num_envs

        # ---- Bootstrap value of state after rollout ----
        with torch.no_grad():
            _, last_val = net(torch.from_numpy(obs).float())

        # ---- Compute n-step returns (backward) ----
        returns = torch.zeros(T, args.num_envs)
        R = last_val
        for t in reversed(range(T)):
            done_t = torch.from_numpy(buf_done[t])
            r_t = torch.from_numpy(buf_rew[t])
            R = r_t + args.gamma * (1.0 - done_t) * R
            returns[t] = R

        advantages = returns - buf_val.detach()

        # ---- Recompute logits/values on collected obs for grad ----
        flat_obs = torch.from_numpy(buf_obs.reshape(-1, OBS_SIZE)).float()
        flat_mask = torch.from_numpy(buf_mask.reshape(-1, NUM_ACTIONS))
        flat_act = torch.from_numpy(buf_act.reshape(-1))
        flat_ret = returns.reshape(-1)
        flat_adv = advantages.reshape(-1)

        logits, values = net(flat_obs)
        dist = masked_categorical(logits, flat_mask)
        logp = dist.log_prob(flat_act)
        entropy = dist.entropy().mean()

        policy_loss = -(logp * flat_adv.detach()).mean()
        value_loss = F.mse_loss(values, flat_ret)
        loss = policy_loss + args.value_coef * value_loss - args.ent_coef * entropy

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
        opt.step()

        update_idx += 1
        if update_idx % 20 == 0:
            mean_r = (sum(recent_returns) / len(recent_returns)) if recent_returns else 0.0
            print(f"[upd {update_idx:>5d} | env_steps {total_env_steps:>8d}] "
                  f"loss={loss.item():.3f} pi={policy_loss.item():+.3f} "
                  f"v={value_loss.item():.3f} H={entropy.item():.3f} "
                  f"mean_ep_return(100)={mean_r:+.3f}")

    final = save_dir / "a2c_final.pt"
    torch.save({"net_state": net.state_dict(), "args": vars(args)}, str(final))
    write_stamp(final)
    print(f"[train] saved -> {final}")


if __name__ == "__main__":
    main()
