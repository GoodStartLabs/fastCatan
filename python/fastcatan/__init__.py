"""High-throughput Catan simulator (C++ core via nanobind)."""

from ._fastcatan import (  # noqa: F401
    BatchedEnv,
    Env,
    OBS_SIZE,
    OBS_FULL_SIZE,
    MASK_WORDS,
    NUM_ACTIONS,
    NUM_PLAYERS,
    NUM_NODES,
    NUM_EDGES,
    NUM_HEXES,
    NUM_PORTS,
    SNAPSHOT_BYTES,
    SIG_INTS,
    SKIP_ACTION,
    action,
)

__all__ = [
    "BatchedEnv",
    "Env",
    "OBS_SIZE",
    "OBS_FULL_SIZE",
    "MASK_WORDS",
    "NUM_ACTIONS",
    "NUM_PLAYERS",
    "NUM_NODES",
    "NUM_EDGES",
    "NUM_HEXES",
    "NUM_PORTS",
    "SNAPSHOT_BYTES",
    "SIG_INTS",
    "SKIP_ACTION",
    "action",
]

__version__ = "0.1.0"
