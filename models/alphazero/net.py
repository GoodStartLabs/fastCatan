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
    # Two-scale recombination weights, fixed to the hybrid leaf formula
    # (mcts_vs_fixed leaf_eval='ab_value'): 0.75*vp_channel + 0.25*fine_channel.
    TWO_SCALE_W = (0.75, 0.25)

    def __init__(
        self,
        obs_dim: int = OBS_SIZE,
        n_actions: int = NUM_ACTIONS,
        hidden: tuple[int, ...] = (512, 512, 256),
        value_channels: int = 1,
        value_hidden: int = 128,
        value_skip_obs: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(d, n_actions)
        self.value_channels = value_channels
        self.value_skip_obs = value_skip_obs
        # value_skip_obs: the head reads [trunk features, RAW obs] — the
        # two-scale fine target is a function of board features the
        # policy-shared trunk has no reason to preserve; the skip hands the
        # head direct access instead of forcing a trunk detour (ep10
        # underfit: train==val fine-MSE plateau at 0.026).
        vin = d + (obs_dim if value_skip_obs else 0)
        self.value_head = nn.Sequential(
            nn.Linear(vin, value_hidden), nn.ReLU(),
            nn.Linear(value_hidden, value_channels), nn.Tanh()
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: (B, OBS_SIZE) -> (logits (B, NUM_ACTIONS), value (B,)).

        With value_channels=2 (the LEARNED JUDGE: per-channel two-scale
        distillation of the ab_value margin) the channels are recombined
        here with the fixed hybrid weights, so every consumer — MCTS leaf
        eval, evaluate.py, the bridge — still sees one scalar in [-1, 1]."""
        z = self.trunk(obs)
        v = self.value_head(self._value_in(z, obs))
        if self.value_channels == 2:
            w = self.TWO_SCALE_W
            value = w[0] * v[..., 0] + w[1] * v[..., 1]
        elif self.value_channels == 3:
            # MIXED-FAMILY judge: two-scale heuristic mimicry (channels 0,1 —
            # information-capped at fine-MSE ~0.026 by hidden enemy state) +
            # outcome prediction (channel 2, vp_margin) — decorrelated error
            # sources, equal blend.
            w = self.TWO_SCALE_W
            value = 0.5 * (w[0] * v[..., 0] + w[1] * v[..., 1]) \
                + 0.5 * v[..., 2]
        else:
            value = v.squeeze(-1)
        return self.policy_head(z), value

    def _value_in(self, z: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat([z, obs], dim=-1) if self.value_skip_obs else z

    def forward_channels(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training-time access to the RAW value channels (B, value_channels)
        — per-channel regression targets need full loss weight per channel
        (combined-scalar MSE down-weights the 0.25 fine channel by 1/16,
        which is exactly what buried the fine signal in the naive distill)."""
        z = self.trunk(obs)
        return self.policy_head(z), self.value_head(self._value_in(z, obs))


def infer_hidden(net_state: dict) -> tuple[int, ...]:
    """Trunk widths from a PolicyValueNet state_dict (trunk.N.weight shapes).

    Lets loaders reconstruct the right architecture from any checkpoint
    instead of assuming the default — the batched trainer's --hidden made
    net size a per-run choice."""
    idx_w = sorted(
        (int(k.split(".")[1]), v.shape[0])
        for k, v in net_state.items()
        if k.startswith("trunk.") and k.endswith(".weight")
    )
    return tuple(w for _i, w in idx_w)


def load_policy_value_net(ckpt_state: dict, device: str = "cpu") -> PolicyValueNet:
    """Build + load a PolicyValueNet with the architecture the checkpoint
    actually has (trunk widths, value channels/width/obs-skip — all inferred
    from weight shapes). ``ckpt_state`` is the torch.load()'d dict holding
    'net_state'."""
    sd = ckpt_state["net_state"]
    hidden = infer_hidden(sd)
    v0 = sd["value_head.0.weight"]          # (value_hidden, trunk_d [+ obs])
    net = PolicyValueNet(obs_dim=sd["trunk.0.weight"].shape[1],
                         hidden=hidden,
                         value_channels=sd["value_head.2.weight"].shape[0],
                         value_hidden=v0.shape[0],
                         value_skip_obs=v0.shape[1] != hidden[-1],
                         ).to(device)
    net.load_state_dict(sd)
    net.eval()
    return net


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log-softmax over legal actions only. mask: bool (B, NUM_ACTIONS).

    Illegal logits are set to -inf before the softmax so they get zero probability
    and contribute nothing to the normalizer. Used for the policy loss (target pi is
    already zero on illegal actions, so the cross-entropy only sums over legal).
    """
    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.where(mask, logits, torch.full_like(logits, neg_inf))
    return F.log_softmax(masked, dim=-1)
