"""Policy adapters: wrap a trained fastcatan checkpoint as a bridge PolicyFn.

`CatanatronBridge` (bridge/catanatron_bridge.py) drives any callable with the
signature

    policy(obs, mask, rng) -> int
        obs:  np.ndarray[float32, (OBS_SIZE,)]   from bridge.obs_encoder.encode_obs
        mask: list[int]                          sorted legal *fast* action IDs
        rng:  random.Random
    returns one fast action ID (must be in `mask`).

This module turns a saved checkpoint into that callable. `build_policy(algo,
ckpt, ...)` is the single entry point and is intentionally a registry mirroring
models/eval.py, so swapping the thesis "final model" is a one-flag change and
new algos (dqn/a2c/muzero) drop in later without touching the tournament code.
Only `ppo` is wired today — that is the trained agent under M4 eval.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np

import fastcatan


NUM_ACTIONS = fastcatan.NUM_ACTIONS
OBS_SIZE = fastcatan.OBS_SIZE

PolicyFn = Callable[[np.ndarray, "list[int]", random.Random], int]


def _mask_list_to_bool(mask: "list[int]") -> np.ndarray:
    """Bridge hands a sorted list of legal fast IDs (often a sub-space, e.g.
    only STEAL_* ids). MaskablePPO.predict wants a bool[NUM_ACTIONS]."""
    b = np.zeros(NUM_ACTIONS, dtype=bool)
    b[mask] = True
    return b


def build_ppo_policy(ckpt: Path, deterministic: bool = False) -> PolicyFn:
    """Load a MaskablePPO checkpoint and expose it as a bridge PolicyFn.

    deterministic=False (sampling) is the default and the validated eval mode:
    argmax trips the within-turn trade-compose loop (see models/PLAN.md), and
    inside the bridge the compose sub-loop re-queries the policy, so sampling
    keeps it from dead-ending into the per-call _COMPOSE_LOOP_CAP fallback.
    """
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(ckpt))
    got = int(model.observation_space.shape[0])
    if got != OBS_SIZE:
        raise ValueError(
            f"checkpoint obs dim {got} != fastcatan.OBS_SIZE {OBS_SIZE}; "
            f"the obs interface changed since this checkpoint was trained — "
            f"retrain or rebuild the matching fastcatan."
        )
    if int(model.action_space.n) != NUM_ACTIONS:
        raise ValueError(
            f"checkpoint action dim {model.action_space.n} != "
            f"fastcatan.NUM_ACTIONS {NUM_ACTIONS}."
        )

    def pick(obs: np.ndarray, mask: "list[int]", rng: random.Random) -> int:
        bool_mask = _mask_list_to_bool(mask)
        action, _ = model.predict(
            obs, action_masks=bool_mask, deterministic=deterministic
        )
        a = int(action)
        # MaskablePPO already restricts to bool_mask; this is belt-and-braces
        # (the bridge re-checks too) and keeps a clean fallback if a sub-space
        # mask is ever empty-after-masking.
        return a if a in mask else rng.choice(mask)

    return pick


def build_policy(algo: str, ckpt: Path, deterministic: bool = False) -> PolicyFn:
    if algo == "ppo":
        return build_ppo_policy(ckpt, deterministic)
    raise ValueError(
        f"algo {algo!r} is not wired into the bridge yet (only 'ppo'). "
        f"Add a loader here mirroring models/eval.py's load_{algo}()."
    )
