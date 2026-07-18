"""Loss modules shared by on-policy PPO and search-target fine-tuning.

The trainer owns batching, optimization, checkpointing, and instrumentation.
Only the objective changes: :class:`PPOLoss` consumes on-policy transitions,
while :class:`DistillationLoss` consumes a search policy and value target.  This
keeps stage3-style search distillation a loss seam rather than a second trainer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def masked_log_probs(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Log probabilities over legal actions only."""
    if logits.shape != legal.shape:
        raise ValueError(f"logit/mask mismatch: {logits.shape} != {legal.shape}")
    if not bool(legal.any(dim=1).all()):
        raise ValueError("empty legal-action row")
    return F.log_softmax(logits.float().masked_fill(~legal, -1e9), dim=-1)


@dataclass
class PPOBatch:
    obs: torch.Tensor
    legal: torch.Tensor
    actions: torch.Tensor
    old_logp: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


@dataclass
class DistillationBatch:
    obs: torch.Tensor
    legal: torch.Tensor
    policy_targets: torch.Tensor
    value_targets: torch.Tensor


class PPOLoss:
    """Clipped PPO objective with an optional frozen forward-KL anchor."""

    def __init__(
        self,
        clip_coef: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        anchor_ref: torch.nn.Module | None = None,
        anchor_coef: float = 0.0,
    ) -> None:
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.anchor_ref = anchor_ref
        self.anchor_coef = anchor_coef

    def __call__(self, net: torch.nn.Module, batch: PPOBatch):
        logits, values = net(batch.obs)
        logp_all = masked_log_probs(logits, batch.legal)
        logp = logp_all.gather(1, batch.actions[:, None]).squeeze(1)
        probs = logp_all.exp()
        entropy = -(probs * logp_all).sum(dim=1).mean()

        ratio = (logp - batch.old_logp).exp()
        unclipped = ratio * batch.advantages
        clipped = ratio.clamp(1.0 - self.clip_coef,
                              1.0 + self.clip_coef) * batch.advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = F.mse_loss(values.float(), batch.returns)

        anchor_kl = torch.zeros((), device=batch.obs.device)
        if self.anchor_ref is not None and self.anchor_coef:
            with torch.no_grad():
                ref_logits, _ = self.anchor_ref(batch.obs)
                ref_logp = masked_log_probs(ref_logits, batch.legal)
            # KL(reference || learner): validated C1/C2/C3 forward anchor.
            ref_prob = ref_logp.exp()
            anchor_kl = (ref_prob * (ref_logp - logp_all)).sum(dim=1).mean()

        loss = (policy_loss + self.value_coef * value_loss
                - self.entropy_coef * entropy
                + self.anchor_coef * anchor_kl)
        log_ratio = logp - batch.old_logp
        approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > self.clip_coef).float().mean()
        metrics = {
            "loss": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
            "approx_kl": approx_kl.detach(),
            "clip_fraction": clip_fraction.detach(),
            "anchor_kl": anchor_kl.detach(),
        }
        return loss, metrics


class DistillationLoss:
    """Search policy CE/KL plus search-value regression.

    ``policy_targets`` may be integer action labels ``(B,)`` or dense visit
    distributions ``(B, NUM_ACTIONS)``.  Dense targets are renormalized on the
    legal support, matching stage3-style MCTS shards.
    """

    def __init__(self, value_coef: float = 1.0) -> None:
        self.value_coef = value_coef

    def __call__(self, net: torch.nn.Module, batch: DistillationBatch):
        logits, values = net(batch.obs)
        logp = masked_log_probs(logits, batch.legal)
        targets = batch.policy_targets
        if targets.ndim == 1:
            policy_loss = F.nll_loss(logp, targets.long())
        else:
            target = targets.float().masked_fill(~batch.legal, 0.0)
            target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)
            policy_loss = -(target * logp).sum(dim=1).mean()
        value_loss = F.mse_loss(values.float(), batch.value_targets.float())
        loss = policy_loss + self.value_coef * value_loss
        return loss, {
            "loss": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
        }
