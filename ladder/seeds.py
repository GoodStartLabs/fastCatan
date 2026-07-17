"""Call-order-independent seed derivation shared by every ladder matchup."""

from __future__ import annotations

MASK64 = (1 << 64) - 1
GOLDEN_GAMMA = 0x9E3779B97F4A7C15


def splitmix64(x: int) -> int:
    """Return the next SplitMix64 value for the supplied 64-bit state.

    This is bit-for-bit the helper in ``include/rng.hpp``: it advances the
    supplied value by GOLDEN_GAMMA before applying the finalizer.
    """
    z = (int(x) + GOLDEN_GAMMA) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def derive_board_seed(master_seed: int, game_index: int) -> int:
    """Derive board ``game_index`` exactly as fastCatan's batched seed family.

    The explicit index makes pairing independent of environment construction or
    reset call order.  A rotation block reuses one returned seed four times.
    """
    if game_index < 0:
        raise ValueError("game_index must be non-negative")
    mixed = (int(master_seed) ^ ((int(game_index) * GOLDEN_GAMMA) & MASK64)) & MASK64
    return splitmix64(mixed)


def derive_policy_seed(master_seed: int, block: int, rotation: int, seat: int) -> int:
    """Separate deterministic randomness for policies from the board stream."""
    if min(block, rotation, seat) < 0:
        raise ValueError("policy seed coordinates must be non-negative")
    index = (block << 6) | (rotation << 3) | seat
    return splitmix64((int(master_seed) ^ 0xD1B54A32D192ED03 ^ index) & MASK64)
