"""Force fastcatan's per-env RNG to a chosen ``bounded(N)`` outcome.

The differential test (``test_differential.py``) drives fastcatan through the
same action stream as catanatron. Three actions consume one ``rng.bounded(N)``
draw each and would otherwise diverge purely by RNG, not by rule logic:

  - ROLL_DICE: ``bounded(36)`` -> dice pair -> sum (rules.cpp:439)
  - BUY_DEV:   ``bounded(deck_total)`` -> drawn card type (rules.cpp:705)
  - STEAL:     ``bounded(victim_handsize)`` -> stolen resource (rules.cpp:554)

To compare *production / draw / steal consequences* (real rules) instead of
the random index, we inject an RNG state into fastcatan's GameState that makes
the upcoming ``bounded(N)`` return the value catanatron actually produced.

This module is a bit-exact Python replica of ``include/rng.hpp`` (xoshiro128++
+ Lemire ``bounded`` + SplitMix64 seeding) plus a scanner that finds a seed
state yielding a target outcome. The replica is verified against the live C++
engine in ``__main__`` (and in the test suite) by injecting a forced state and
confirming the engine produces the predicted draw.
"""
from __future__ import annotations

from typing import Callable

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def _rotl(x: int, k: int) -> int:
    x &= MASK32
    return ((x << k) | (x >> (32 - k))) & MASK32


class Xoshiro128:
    """Mirror of catan::Xoshiro128 (rng.hpp:10-43)."""

    __slots__ = ("s",)

    def __init__(self, s):
        self.s = [s[0] & MASK32, s[1] & MASK32, s[2] & MASK32, s[3] & MASK32]

    def next(self) -> int:
        s = self.s
        result = (_rotl((s[0] + s[3]) & MASK32, 7) + s[0]) & MASK32
        t = (s[1] << 9) & MASK32
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl(s[3], 11)
        return result

    def bounded(self, bound: int) -> int:
        # Lemire (rng.hpp:31-42).
        product = self.next() * bound
        lo = product & MASK32
        if lo < bound:
            threshold = ((0x100000000 - bound) & MASK32) % bound
            while lo < threshold:
                product = self.next() * bound
                lo = product & MASK32
        return product >> 32


def _splitmix64(x: int) -> tuple[int, int]:
    """Returns (next_x, output) mirroring rng.hpp:48-53 (x is by-ref in C++)."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return x, (z ^ (z >> 31))


def seed_state(key: int) -> tuple[int, int, int, int]:
    """Mirror of catan::xoshiro_seed (rng.hpp:57-65)."""
    x = key if key != 0 else 0x123456789ABCDEF0
    x, a = _splitmix64(x)
    x, b = _splitmix64(x)
    return (a & MASK32, (a >> 32) & MASK32, b & MASK32, (b >> 32) & MASK32)


def find_state(bound: int, accept: Callable[[int], bool],
               max_scan: int = 2_000_000) -> tuple[int, int, int, int]:
    """Find an xoshiro seed state whose first ``bounded(bound)`` is accepted.

    Scans SplitMix64-seeded states (the same family the C++ engine uses), so
    every returned state is a valid non-zero xoshiro state.
    """
    for k in range(max_scan):
        st = seed_state(k)
        if accept(Xoshiro128(st).bounded(bound)):
            return st
    raise RuntimeError(
        f"no rng state found for bound={bound} within {max_scan} scans"
    )


# ---------------------------------------------------------------------------
# Outcome-specific finders
# ---------------------------------------------------------------------------

# Cache: dice sums are reused constantly; the deck/steal bands vary per ply.
_DICE_CACHE: dict[int, tuple[int, int, int, int]] = {}


def state_for_dice_sum(target_sum: int) -> tuple[int, int, int, int]:
    """RNG state so ROLL's ``bounded(36)`` yields a pair summing to target.

    pair -> d1 = pair//6 + 1, d2 = pair%6 + 1 (rules.cpp:440-441).
    """
    if target_sum in _DICE_CACHE:
        return _DICE_CACHE[target_sum]

    def ok(pair: int) -> bool:
        d1 = pair // 6 + 1
        d2 = pair % 6 + 1
        return d1 + d2 == target_sum

    st = find_state(36, ok)
    _DICE_CACHE[target_sum] = st
    return st


def state_for_band(total: int, lo: int, hi: int) -> tuple[int, int, int, int]:
    """RNG state so ``bounded(total)`` lands in [lo, hi).

    Used for BUY_DEV (drawn card type) and STEAL (stolen resource), where the
    engine walks a cumulative count array and the index in [lo, hi) selects the
    desired bucket (rules.cpp:705-712, 554-560).
    """
    return find_state(total, lambda v: lo <= v < hi)


if __name__ == "__main__":
    import numpy as np
    import fastcatan as fc
    from bridge import state_mirror as M

    # Verify the replica against the C++ engine: drive an env to a state where
    # ROLL_DICE is legal, force each dice sum, and confirm the engine rolls it.
    def fresh_rollable_env():
        env = fc.Env()
        env.reset(7)
        mask = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
        for _ in range(200):
            env.action_mask(mask)
            def legal_bit(i):
                return (int(mask[i >> 6]) >> (i & 63)) & 1
            if legal_bit(fc.action.ROLL_DICE):
                return env
            legal = [i for i in range(fc.NUM_ACTIONS) if legal_bit(i)]
            if not legal:
                break
            # Avoid ending the turn so we stay near a roll point.
            env.step(int(legal[0]))
        raise RuntimeError("could not reach a ROLL-legal state")

    base = fresh_rollable_env()
    base_snap = base.snapshot()

    for target in range(2, 13):
        snap = M.parse_snapshot(base_snap)
        st = state_for_dice_sum(target)
        snap.gs.rng[:] = st
        env = fc.Env()
        env.reset(0)
        env.load_snapshot(M.to_bytes(snap))
        env.step(fc.action.ROLL_DICE)
        assert env.dice_roll == target, (
            f"forced sum {target} but engine rolled {env.dice_roll}"
        )
    print("rng_force: dice forcing verified against C++ engine for sums 2..12 — OK")
