"""High-throughput PPO directly on :class:`fastcatan.BatchedEnv`."""

from .losses import DistillationBatch, DistillationLoss, PPOBatch, PPOLoss
from .trainer import BatchedTrainer, TrainerConfig, compute_gae, terminal_rewards

__all__ = [
    "BatchedTrainer",
    "DistillationBatch",
    "DistillationLoss",
    "PPOBatch",
    "PPOLoss",
    "TrainerConfig",
    "compute_gae",
    "terminal_rewards",
]
