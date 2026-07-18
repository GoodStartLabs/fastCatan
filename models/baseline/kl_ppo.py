"""P3-C1 (Conservative): IL-anchored MaskablePPO.

Change 1 of proposal docs/proposals/p3-c1-conservative.md: the warm-start twin
(W&B 33fw0lcs) catastrophically forgot its imitation init under un-anchored PPO
(pool promotion_mean 0.469->0.297->0.117; actor entropy 0.22->1.8 nats). This
module adds a distillation anchor to a *frozen* copy of the IL actor so the PPO
actor stays near the competent clone while improving on-policy.

The anchor is a masked distribution-matching term on the rollout states:
* forward KL (mass-covering; Kickstarting / InstructGPT-ptx form) -- default, or
* reverse KL (mode-seeking; RLHF penalty form).
Its coefficient anneals from ``kl_coef0`` to ``kl_coef_final`` over the budget so
the student can surpass the teacher late (Schmitt et al. 2018).

The reference net reads only the 1,084-float actor observation -- leakage-clean;
actor inputs are unchanged. Everything else is stock ``MaskablePPO``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from torch import nn

from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance

from models.baseline.policy import ACTOR_HIDDEN, ACTOR_OBS_SIZE, SplitMlpExtractor

_MASK_NEG = -1e8


class ReferenceActor(nn.Module):
    """Frozen copy of the IL actor (body + head); actor-obs only."""

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        # Same architecture the IL checkpoint was trained with (LayerNorm + 3
        # GELU layers), mirroring Phase2Policy.mlp_extractor.actor.
        self.actor = SplitMlpExtractor().actor
        self.head = nn.Linear(ACTOR_HIDDEN[-1], action_dim)

    @classmethod
    def from_il(cls, checkpoint: str | Path, device, action_dim: int) -> "ReferenceActor":
        state = th.load(str(checkpoint), map_location=device, weights_only=False)
        module = cls(action_dim)
        module.actor.load_state_dict(state["actor_body"])
        module.head.load_state_dict(state["actor_head"])
        module.to(device)
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)
        return module

    @th.no_grad()
    def forward(self, observation: th.Tensor) -> th.Tensor:
        latent = self.actor(observation[..., :ACTOR_OBS_SIZE])
        return self.head(latent)


class KLRegularizedMaskablePPO(MaskablePPO):
    """MaskablePPO with an annealed IL-anchor added to the loss.

    Anchor state is set as plain attributes after construction (kept out of the
    constructor signature to avoid coupling to the SB3 __init__):
      * ``kl_ref``        -- ReferenceActor or None (None => stock behaviour)
      * ``kl_coef0``      -- initial anchor coefficient beta_0
      * ``kl_coef_final`` -- final anchor coefficient beta_T
      * ``kl_form``       -- "forward" (default) or "reverse"
    """

    def _anchor_beta(self) -> float:
        remaining = float(self._current_progress_remaining)
        c0 = float(getattr(self, "kl_coef0", 0.0))
        cf = float(getattr(self, "kl_coef_final", c0))
        return cf + (c0 - cf) * remaining

    def _anchor_term(self, rollout_data) -> th.Tensor:
        obs = rollout_data.observations
        latent_pi = self.policy.mlp_extractor.forward_actor(obs)
        cur_logits = self.policy.action_net(latent_pi)
        ref_logits = self.kl_ref(obs)

        masks = getattr(rollout_data, "action_masks", None)
        if masks is not None:
            legal = masks.reshape(cur_logits.shape) > 0.5
            neg = th.full_like(cur_logits, _MASK_NEG)
            cur_logits = th.where(legal, cur_logits, neg)
            ref_logits = th.where(legal, ref_logits, neg)

        cur_logp = F.log_softmax(cur_logits, dim=-1)
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        if getattr(self, "kl_form", "forward") == "reverse":
            cur_p = cur_logp.exp()
            return (cur_p * (cur_logp - ref_logp)).sum(-1).mean()
        ref_p = ref_logp.exp()
        return (ref_p * (ref_logp - cur_logp)).sum(-1).mean()

    def train(self) -> None:
        """Stock MaskablePPO.train() plus the IL anchor term."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        anchor_vals: list[float] = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                if getattr(self, "kl_ref", None) is not None:
                    anchor = self._anchor_term(rollout_data)
                    loss = loss + self._anchor_beta() * anchor
                    anchor_vals.append(anchor.item())

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        if getattr(self, "kl_ref", None) is not None:
            self.logger.record("train/kl_anchor", float(np.mean(anchor_vals)) if anchor_vals else 0.0)
            self.logger.record("train/kl_beta", self._anchor_beta())
