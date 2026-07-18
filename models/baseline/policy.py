"""Split actor/critic policy for the Phase-2 boring baseline.

The training observation is a flat concatenation of:

* the frozen legal-information ``write_obs`` actor input (1,084 floats),
* the critic-only ``write_obs_full`` input (1,132 floats), and
* one 8-way pool-ID one-hot for each of the three opponent seats (24 floats).

The actor slice is structural, rather than a training convention: no actor
module is connected to the critic-only tail.  At inference, the frozen
tournament hands MaskablePPO a 1,084-vector.  ``obs_to_tensor`` pads only the
unused critic tail, so the same checkpoint remains deployable there.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import fastcatan
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


ACTOR_OBS_SIZE = int(fastcatan.OBS_SIZE)
FULL_OBS_SIZE = int(fastcatan.OBS_FULL_SIZE)
POOL_SIZE = 8
OPPONENT_SLOTS = int(fastcatan.NUM_PLAYERS) - 1
POOL_ID_SIZE = POOL_SIZE * OPPONENT_SLOTS
CRITIC_OBS_SIZE = FULL_OBS_SIZE + POOL_ID_SIZE
TRAIN_OBS_SIZE = ACTOR_OBS_SIZE + CRITIC_OBS_SIZE

ACTOR_HIDDEN = (2048, 1024, 512)
CRITIC_HIDDEN = (1024, 512)


class SplitMlpExtractor(nn.Module):
    """Two disjoint GELU MLPs with the fixed Phase-2 input boundary."""

    latent_dim_pi = ACTOR_HIDDEN[-1]
    latent_dim_vf = CRITIC_HIDDEN[-1]

    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.LayerNorm(ACTOR_OBS_SIZE),
            nn.Linear(ACTOR_OBS_SIZE, ACTOR_HIDDEN[0]), nn.GELU(),
            nn.Linear(ACTOR_HIDDEN[0], ACTOR_HIDDEN[1]), nn.GELU(),
            nn.Linear(ACTOR_HIDDEN[1], ACTOR_HIDDEN[2]), nn.GELU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(CRITIC_OBS_SIZE, CRITIC_HIDDEN[0]), nn.GELU(),
            nn.Linear(CRITIC_HIDDEN[0], CRITIC_HIDDEN[1]), nn.GELU(),
        )

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor(features[..., :ACTOR_OBS_SIZE])

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic(features[..., ACTOR_OBS_SIZE:])

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class Phase2Policy(MaskableActorCriticPolicy):
    """MaskablePPO policy with a legal-info actor and privileged critic."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["net_arch"] = []
        kwargs["ortho_init"] = False
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SplitMlpExtractor().to(self.device)

    def obs_to_tensor(self, observation):
        """Accept the 1,084-wide inference observation used by tournament.sh."""
        if isinstance(observation, np.ndarray) and observation.shape[-1] == ACTOR_OBS_SIZE:
            padded = np.zeros((*observation.shape[:-1], TRAIN_OBS_SIZE), dtype=np.float32)
            padded[..., :ACTOR_OBS_SIZE] = observation
            observation = padded
        return super().obs_to_tensor(observation)

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
        action_masks=None,
    ):
        # The repository's PolicyOpponent convention samples masked softmaxes:
        # greedy TRADE_OPEN/CANCEL choices can legally repeat until the engine's
        # liveness cap.  Training observations remain TRAIN_OBS_SIZE wide; the
        # 1,084-wide frozen-tournament path therefore selects sampling here.
        if isinstance(observation, np.ndarray) and observation.shape[-1] == ACTOR_OBS_SIZE:
            deterministic = False
        return super().predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
            action_masks=action_masks,
        )


def parameter_counts(policy: Phase2Policy) -> dict[str, int]:
    actor = sum(p.numel() for p in policy.mlp_extractor.actor.parameters())
    actor += sum(p.numel() for p in policy.action_net.parameters())
    critic = sum(p.numel() for p in policy.mlp_extractor.critic.parameters())
    critic += sum(p.numel() for p in policy.value_net.parameters())
    return {"actor": actor, "critic": critic, "total": actor + critic}


def load_il_weights(policy: Phase2Policy, checkpoint: str | Path) -> dict:
    """Load the four supervised modules into an initialized PPO policy."""
    state = torch.load(str(checkpoint), map_location=policy.device, weights_only=False)
    policy.mlp_extractor.actor.load_state_dict(state["actor_body"])
    policy.action_net.load_state_dict(state["actor_head"])
    policy.mlp_extractor.critic.load_state_dict(state["critic_body"])
    policy.value_net.load_state_dict(state["critic_head"])
    return state
