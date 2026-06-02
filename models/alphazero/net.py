"""Policy + value network for AlphaZero Catan.

Shared MLP trunk (matches the 512x512x256 width that worked for PPO) feeding two
heads:
  - policy: logits over all NUM_ACTIONS (masked to legal at use time)
  - value:  scalar in [-1, 1] via tanh = predicted final outcome (+1 win / -1 not)
            for the seat whose POV the observation is encoded from.

The value is POV-relative on purpose: the obs encoder already emits each seat's
view with "self" at slot 0, so one scalar head serves all four seats. MCTS reads
the value from each seat's POV to build its length-4 backup vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastcatan

OBS_SIZE = fastcatan.OBS_SIZE
NUM_ACTIONS = fastcatan.NUM_ACTIONS


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_SIZE,
        n_actions: int = NUM_ACTIONS,
        hidden: tuple[int, ...] = (512, 512, 256),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(d, n_actions)
        self.value_head = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh()
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: (B, OBS_SIZE) -> (logits (B, NUM_ACTIONS), value (B,))."""
        z = self.trunk(obs)
        return self.policy_head(z), self.value_head(z).squeeze(-1)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log-softmax over legal actions only. mask: bool (B, NUM_ACTIONS).

    Illegal logits are set to -inf before the softmax so they get zero probability
    and contribute nothing to the normalizer. Used for the policy loss (target pi is
    already zero on illegal actions, so the cross-entropy only sums over legal).
    """
    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.where(mask, logits, torch.full_like(logits, neg_inf))
    return F.log_softmax(masked, dim=-1)
