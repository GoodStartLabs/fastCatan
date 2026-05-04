"""High-throughput Catan simulator (C++ core via nanobind)."""

from ._fastcatan import (  # noqa: F401
    BatchedEnv,
    Env,
    OBS_SIZE,
    MASK_WORDS,
    NUM_ACTIONS,
    NUM_PLAYERS,
    NUM_NODES,
    NUM_EDGES,
    NUM_HEXES,
    NUM_PORTS,
    action,
)

# Gym wrapper is a soft import — only available if `gymnasium` is installed.
try:
    from .gym_env import (  # noqa: F401
        GymEnv,
        random_legal_policy,
        lowest_legal_policy,
        unpack_mask,
    )
    _GYM = True
except ImportError:
    _GYM = False

try:
    from .pettingzoo_env import CatanAECEnv  # noqa: F401
    _PZ = True
except ImportError:
    _PZ = False

__all__ = [
    "BatchedEnv",
    "Env",
    "OBS_SIZE",
    "MASK_WORDS",
    "NUM_ACTIONS",
    "NUM_PLAYERS",
    "NUM_NODES",
    "NUM_EDGES",
    "NUM_HEXES",
    "NUM_PORTS",
    "action",
]
if _GYM:
    __all__ += ["GymEnv", "random_legal_policy", "lowest_legal_policy", "unpack_mask"]
if _PZ:
    __all__ += ["CatanAECEnv"]

__version__ = "0.1.0"
