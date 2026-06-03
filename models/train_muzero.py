"""Minimal MuZero scaffold for Catan.

Reference: Schrittwieser et al. 2020 ("Mastering Atari, Go, Chess and Shogi
by Planning with a Learned Model"). DeepMind.

MuZero learns three networks and plans with MCTS on the learned model:

  1. Representation:  h_0 = f_repr(obs)
  2. Dynamics:        (h_{k+1}, r_k) = f_dyn(h_k, a_k)
  3. Prediction:      (p_k, v_k)     = f_pred(h_k)
       p_k = policy prior over actions, v_k = value estimate.

The dynamics net is NOT trained to reconstruct the true environment state —
only to predict reward + future policy/value targets correctly. The hidden
state is whatever shape lets that work.

Action selection: run N simulations of MCTS rooted at h_0. Each sim walks
the tree using PUCT to pick actions, then expands a leaf by querying f_dyn
and f_pred. Visit counts give the improved policy target.

Training: sample trajectories from replay. For each sampled position, unroll
the dynamics net K steps using the actions actually taken. Train so that
predicted (policy, value, reward) match the MCTS-improved policy, the
n-step return, and the observed reward.

THIS FILE IS A REFERENCE-QUALITY SKELETON, NOT A TUNED IMPLEMENTATION:
  - small networks, small MCTS (16 sims), no reanalyze, no priorities
  - won't beat the random baseline without significant tuning + compute
  - exists to show the algorithm structure on this env

For a production MuZero see github.com/werner-duvaud/muzero-general.

Run:
    python -m models.train_muzero --total-games 200 --sims 16
"""
from __future__ import annotations

import argparse
import math
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

HIDDEN_DIM = 128


# -------------------- Networks --------------------

class Representation(nn.Module):
    def __init__(self, obs_dim: int = OBS_SIZE, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Dynamics(nn.Module):
    """(h, a_onehot) -> (h_next, reward_scalar)."""

    def __init__(self, hidden: int = HIDDEN_DIM, n_actions: int = NUM_ACTIONS):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(hidden + n_actions, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.h_head = nn.Sequential(nn.Linear(256, hidden), nn.Tanh())
        self.r_head = nn.Linear(256, 1)

    def forward(self, h: torch.Tensor, a_onehot: torch.Tensor):
        z = self.trunk(torch.cat([h, a_onehot], dim=-1))
        return self.h_head(z), self.r_head(z).squeeze(-1)


class Prediction(nn.Module):
    """h -> (policy_logits, value_scalar)."""

    def __init__(self, hidden: int = HIDDEN_DIM, n_actions: int = NUM_ACTIONS):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU())
        self.policy = nn.Linear(256, n_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, h: torch.Tensor):
        z = self.trunk(h)
        return self.policy(z), self.value(z).squeeze(-1)


class MuZeroNets(nn.Module):
    def __init__(self):
        super().__init__()
        self.repr = Representation()
        self.dyn = Dynamics()
        self.pred = Prediction()


# -------------------- MCTS --------------------

class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "hidden", "reward")

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, Node] = {}
        self.hidden: torch.Tensor | None = None
        self.reward: float = 0.0

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


def _ucb_score(parent: Node, child: Node, c_puct: float = 1.25) -> float:
    pb_c = c_puct * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    return child.value() + pb_c * child.prior


def run_mcts(root_hidden: torch.Tensor, root_mask: np.ndarray, nets: MuZeroNets,
             num_simulations: int, gamma: float) -> Node:
    nets.eval()
    with torch.no_grad():
        logits, _ = nets.pred(root_hidden.unsqueeze(0))
        logits = logits.squeeze(0).numpy()
        logits = np.where(root_mask, logits, -np.inf)
        priors = _softmax(logits)

        root = Node(prior=0.0)
        root.hidden = root_hidden
        for a, pr in enumerate(priors):
            if root_mask[a]:
                root.children[a] = Node(prior=float(pr))

        for _ in range(num_simulations):
            node = root
            path = [node]
            actions: list[int] = []
            # Selection: walk down using UCB until we hit an unexpanded node.
            while node.children:
                a, node = max(node.children.items(),
                              key=lambda kv: _ucb_score(path[-1], kv[1]))
                actions.append(a)
                path.append(node)

            # Expansion: query dynamics from parent for the chosen action.
            parent = path[-2]
            a = actions[-1]
            a_oh = F.one_hot(torch.tensor([a]), NUM_ACTIONS).float()
            h_next, r_pred = nets.dyn(parent.hidden.unsqueeze(0), a_oh)
            node.hidden = h_next.squeeze(0)
            node.reward = float(r_pred.item())

            child_logits, value = nets.pred(node.hidden.unsqueeze(0))
            child_priors = _softmax(child_logits.squeeze(0).numpy())
            # No legal-mask info at simulated depth; allow all actions.
            for ca, cpr in enumerate(child_priors):
                node.children[ca] = Node(prior=float(cpr))

            # Backup
            g = float(value.item())
            for n in reversed(path):
                n.value_sum += g
                n.visit_count += 1
                g = n.reward + gamma * g

    return root


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def visit_count_policy(root: Node, temperature: float = 1.0) -> np.ndarray:
    counts = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for a, c in root.children.items():
        counts[a] = c.visit_count
    if temperature == 0:
        out = np.zeros_like(counts)
        out[int(np.argmax(counts))] = 1.0
        return out
    counts = counts ** (1.0 / temperature)
    s = counts.sum()
    return counts / s if s > 0 else counts


# -------------------- Self-play + replay --------------------

class GameRecord:
    def __init__(self):
        self.obs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.policies: list[np.ndarray] = []
        self.values: list[float] = []


def play_one_game(env: FastCatanEnv, nets: MuZeroNets, sims: int,
                  gamma: float) -> GameRecord:
    rec = GameRecord()
    obs, _ = env.reset()
    done = False
    while not done:
        mask = env.action_masks()
        with torch.no_grad():
            h = nets.repr(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0)
        root = run_mcts(h, mask, nets, sims, gamma)
        pi = visit_count_policy(root, temperature=1.0)
        # Renormalize over legal only (sims at root only expanded legal kids).
        pi_legal = pi * mask
        if pi_legal.sum() == 0:
            legal_ids = np.where(mask)[0]
            action = int(np.random.choice(legal_ids))
        else:
            pi_legal /= pi_legal.sum()
            action = int(np.random.choice(NUM_ACTIONS, p=pi_legal))

        rec.obs.append(obs.copy())
        rec.actions.append(action)
        rec.policies.append(pi_legal)
        rec.values.append(root.value())

        obs, reward, done, _trunc, _info = env.step(action)
        rec.rewards.append(float(reward))
    return rec


def n_step_return(rec: GameRecord, t: int, n: int, gamma: float) -> float:
    R = 0.0
    for k in range(n):
        if t + k >= len(rec.rewards):
            break
        R += (gamma ** k) * rec.rewards[t + k]
    bootstrap_t = t + n
    if bootstrap_t < len(rec.values):
        R += (gamma ** n) * rec.values[bootstrap_t]
    return R


def sample_batch(buffer: deque, batch_size: int, unroll_k: int, td_n: int,
                 gamma: float):
    """Sample (obs, actions[K], target_pi[K+1], target_v[K+1], target_r[K])."""
    obs_b, act_b, tpi_b, tv_b, tr_b = [], [], [], [], []
    for _ in range(batch_size):
        rec = random.choice(buffer)
        T = len(rec.actions)
        t = random.randint(0, T - 1)
        obs_b.append(rec.obs[t])

        acts = []
        pis = [rec.policies[t]]
        vals = [n_step_return(rec, t, td_n, gamma)]
        rews = []
        for k in range(unroll_k):
            tk = t + k
            if tk < T:
                acts.append(rec.actions[tk])
                rews.append(rec.rewards[tk])
            else:
                acts.append(0)
                rews.append(0.0)
            tk1 = t + k + 1
            if tk1 < T:
                pis.append(rec.policies[tk1])
                vals.append(n_step_return(rec, tk1, td_n, gamma))
            else:
                pis.append(np.zeros(NUM_ACTIONS, dtype=np.float32))
                vals.append(0.0)

        act_b.append(acts)
        tpi_b.append(pis)
        tv_b.append(vals)
        tr_b.append(rews)

    return (
        torch.from_numpy(np.stack(obs_b)).float(),
        torch.tensor(act_b, dtype=torch.long),
        torch.from_numpy(np.array(tpi_b, dtype=np.float32)),
        torch.tensor(tv_b, dtype=torch.float32),
        torch.tensor(tr_b, dtype=torch.float32),
    )


# -------------------- Training --------------------

def train_step(nets: MuZeroNets, opt: torch.optim.Optimizer,
               batch, unroll_k: int) -> dict:
    obs, actions, tgt_pi, tgt_v, tgt_r = batch
    B = obs.shape[0]

    h = nets.repr(obs)
    logits, value = nets.pred(h)

    losses_pi = [F.kl_div(F.log_softmax(logits, dim=-1), tgt_pi[:, 0],
                          reduction="batchmean")]
    losses_v = [F.mse_loss(value, tgt_v[:, 0])]
    losses_r: list[torch.Tensor] = []

    for k in range(unroll_k):
        a_oh = F.one_hot(actions[:, k], NUM_ACTIONS).float()
        h, r_pred = nets.dyn(h, a_oh)
        h = h * 0.5 + h.detach() * 0.5  # MuZero "half-gradient" trick
        logits, value = nets.pred(h)
        losses_pi.append(F.kl_div(F.log_softmax(logits, dim=-1), tgt_pi[:, k + 1],
                                  reduction="batchmean"))
        losses_v.append(F.mse_loss(value, tgt_v[:, k + 1]))
        losses_r.append(F.mse_loss(r_pred, tgt_r[:, k]))

    pi_loss = sum(losses_pi) / len(losses_pi)
    v_loss = sum(losses_v) / len(losses_v)
    r_loss = sum(losses_r) / max(1, len(losses_r))
    loss = pi_loss + v_loss + r_loss

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(nets.parameters(), 5.0)
    opt.step()
    return {"loss": loss.item(), "pi": pi_loss.item(),
            "v": v_loss.item(), "r": r_loss.item()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-games", type=int, default=200)
    p.add_argument("--sims", type=int, default=16)
    p.add_argument("--buffer-size", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--unroll-k", type=int, default=5)
    p.add_argument("--td-n", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-steps-per-game", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default=str(CKPT_DIR / "muzero"))
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = FastCatanEnv(seed=args.seed)
    nets = MuZeroNets()
    opt = torch.optim.Adam(nets.parameters(), lr=args.lr)
    buffer: deque = deque(maxlen=args.buffer_size)

    for game in range(1, args.total_games + 1):
        rec = play_one_game(env, nets, args.sims, args.gamma)
        buffer.append(rec)
        game_return = sum(rec.rewards)
        print(f"[game {game:>4d}] len={len(rec.actions)} return={game_return:+.2f} "
              f"buf={len(buffer)}")

        if len(buffer) >= 4:
            for _ in range(args.train_steps_per_game):
                batch = sample_batch(buffer, args.batch_size,
                                     args.unroll_k, args.td_n, args.gamma)
                stats = train_step(nets, opt, batch, args.unroll_k)
            print(f"   train: loss={stats['loss']:.3f} pi={stats['pi']:.3f} "
                  f"v={stats['v']:.3f} r={stats['r']:.3f}")

    final = save_dir / "muzero_final.pt"
    torch.save({"nets_state": nets.state_dict(), "args": vars(args)}, str(final))
    write_stamp(final)
    print(f"[train] saved -> {final}")


if __name__ == "__main__":
    main()
