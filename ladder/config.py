"""Frozen harness-level defaults for ladder v1."""

from __future__ import annotations

LADDER_VERSION = "1.0-v1"
SCHEMA_VERSION = 1

# Board seeds are derived explicitly; BatchedEnv's advancing reset counter is
# deliberately not used by the ladder.
MASTER_SEED = 0x6A09E667F3BCC909
SMOKE_GAMES_PER_OPPONENT_MODE = 256
FULL_GAMES_PER_OPPONENT_MODE = 1024
ROTATIONS_PER_BLOCK = 4

INCUMBENT = "balanced-strong"
WANDB_PROJECT = "goodsettler-eval"

NO_WINNER_POLICY = "loss_for_all_seats"
