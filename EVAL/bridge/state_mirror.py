"""Byte-exact ctypes mirror of the C++ ``GameState`` + ``BoardLayout``.

This is the foundation of the cross-engine differential test
(``bridge/tests/test_differential.py``). The C++ ``snapshot()`` /
``load_snapshot()`` bindings are a raw ``memcpy`` of these two structs
(bindings.cpp:70-82), so a Python struct with the identical layout lets us
*construct* an arbitrary fastcatan state and inject it via ``load_snapshot``.

Layout mirrored from ``include/state.hpp``:
  - ``GameState`` is ``alignas(64)`` and ``static_assert``-pinned at 384 bytes.
  - ``BoardLayout`` is 47 bytes (19+19+9).

We use ``_pack_ = 1`` and lay out every padding byte explicitly so the
result is deterministic across platforms rather than relying on ctypes'
auto-alignment matching the C++ compiler. The two internal pads (before the
4-aligned RNG and the 8-aligned mask) and the trailing pad to 64-byte
alignment are computed from the field offsets in state.hpp.

The ``validate_against_env`` round-trip is the correctness gate: it parses a
real fastcatan snapshot and checks every accessor-visible field matches, so
a layout mistake fails loudly instead of silently corrupting injected state.
"""
from __future__ import annotations

import ctypes

import fastcatan as fc

U8 = ctypes.c_uint8
U16 = ctypes.c_uint16
U32 = ctypes.c_uint32
U64 = ctypes.c_uint64

NUM_NODES = 54
NUM_EDGES = 72
NUM_HEXES = 19
NUM_PORTS = 9

# node[] level encoding (state.hpp:16-19)
NODE_EMPTY = 0
NODE_SETTLEMENT = 1
NODE_CITY = 2


class CBoardLayout(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("hex_resource", U8 * 19),
        ("hex_number", U8 * 19),
        ("port_type", U8 * 9),
    ]


class CGameState(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        # --- Board state ---
        ("node", U8 * 54),
        ("edge", U8 * 72),
        ("robber_hex", U8),
        # --- Turn / phase state ---
        ("dice_roll", U8),
        ("turn_count", U16),          # offset 128, 2-aligned naturally
        ("phase", U8),
        ("flag", U8),
        ("start_player", U8),
        # --- Per-player private state ---
        ("player_resources", (U8 * 5) * 4),
        ("player_dev", (U8 * 5) * 4),
        ("player_dev_bought_this_turn", (U8 * 5) * 4),
        ("player_vp", U8 * 4),
        ("player_ports", U8 * 4),
        # --- Per-player counters ---
        ("player_knights_played", U8 * 4),
        ("player_road_length", U8 * 4),
        ("player_settlement_count", U8 * 4),
        ("player_city_count", U8 * 4),
        ("player_road_count", U8 * 4),
        # --- Awards / turn flags ---
        ("longest_road_owner", U8),
        ("largest_army_owner", U8),
        ("dev_card_played", U8),      # C++ bool, 1 byte
        ("current_player", U8),
        ("discarding_player", U8),
        ("free_roads_remaining", U8),
        # --- Per-player public state ---
        ("player_handsize", U8 * 4),
        ("player_total_dev", U8 * 4),
        ("player_vp_without_dev", U8 * 4),
        ("player_discard_remaining", U8 * 4),
        # --- Bank ---
        ("bank", U8 * 5),
        ("dev_deck", U8 * 5),
        # --- Trade scratch ---
        ("trade_give", U8 * 5),
        ("trade_want", U8 * 5),
        ("trade_response", U8),
        ("trade_proposer", U8),
        # ends at offset 265; RNG (uint32) needs 4-byte alignment -> pad 3
        ("_pad_rng", U8 * 3),
        ("rng", U32 * 4),             # offset 268, 16 bytes -> 284
        # action_mask (uint64) needs 8-byte alignment -> pad 4
        ("_pad_mask", U8 * 4),
        ("action_mask", U64 * 5),     # offset 288, 40 bytes -> 328
        ("road_node_member", U64 * 4),  # offset 328, 32 bytes -> 360
        # struct is alignas(64) -> sizeof rounds up to 384 -> trailing pad 24
        ("_pad_tail", U8 * 24),
    ]


class CSnapshot(ctypes.Structure):
    """GameState immediately followed by BoardLayout, matching snapshot()."""
    _pack_ = 1
    _fields_ = [
        ("gs", CGameState),
        ("board", CBoardLayout),
    ]


# Compile-time-ish guards mirroring the C++ static_asserts.
assert ctypes.sizeof(CGameState) == 384, (
    f"CGameState is {ctypes.sizeof(CGameState)} bytes, expected 384 "
    "(layout drifted from state.hpp)"
)
assert ctypes.sizeof(CBoardLayout) == 47, (
    f"CBoardLayout is {ctypes.sizeof(CBoardLayout)} bytes, expected 47"
)
assert ctypes.sizeof(CSnapshot) == 431, ctypes.sizeof(CSnapshot)


def parse_snapshot(data: bytes) -> CSnapshot:
    if len(data) != 431:
        raise ValueError(f"snapshot must be 431 bytes, got {len(data)}")
    return CSnapshot.from_buffer_copy(data)


def to_bytes(snap: CSnapshot) -> bytes:
    return bytes(memoryview(snap).cast("B"))


# Node packing (state.hpp:16-25): bits 0-1 = level, bits 2-4 = owner.
def node_pack(level: int, owner: int) -> int:
    return (level & 0x03) | ((owner & 0x03) << 2)


def node_level(n: int) -> int:
    return n & 0x03


def node_owner(n: int) -> int:
    return (n >> 2) & 0x03


def validate_against_env(seed: int = 42, steps: int = 60) -> None:
    """Round-trip gate: snapshot a live env, parse with the ctypes mirror,
    and assert every accessor-visible field matches. Steps the env first so
    a non-trivial state (buildings, resources, robber moved) is exercised.

    Raises AssertionError on any mismatch — a layout bug in the mirror.
    """
    import numpy as np

    env = fc.Env()
    env.reset(seed)
    # Advance to a rich mid-game state via lowest-legal actions.
    mask = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
    for _ in range(steps):
        env.action_mask(mask)
        legal = [i for i in range(fc.NUM_ACTIONS)
                 if (int(mask[i >> 6]) >> (i & 63)) & 1]
        if not legal:
            break
        env.step(int(legal[0]))
        if env.phase == 3:  # ENDED
            break

    snap = parse_snapshot(env.snapshot())
    gs = snap.gs

    def check(name, got, exp):
        assert got == exp, f"{name}: mirror={got} accessor={exp}"

    check("current_player", gs.current_player, env.current_player)
    check("dice_roll", gs.dice_roll, env.dice_roll)
    check("turn_count", gs.turn_count, env.turn_count)
    check("phase", gs.phase, env.phase)
    check("flag", gs.flag, env.flag)
    check("longest_road_owner", gs.longest_road_owner, env.longest_road_owner)
    check("largest_army_owner", gs.largest_army_owner, env.largest_army_owner)
    for r in range(5):
        check(f"bank[{r}]", gs.bank[r], env.bank(r))
    for seat in range(4):
        check(f"vp[{seat}]", gs.player_vp[seat], env.player_vp(seat))
        check(f"vp_public[{seat}]", gs.player_vp_without_dev[seat],
              env.player_vp_public(seat))
        check(f"handsize[{seat}]", gs.player_handsize[seat],
              env.player_handsize(seat))
        check(f"settle_cnt[{seat}]", gs.player_settlement_count[seat],
              env.player_settlement_count(seat))
        check(f"city_cnt[{seat}]", gs.player_city_count[seat],
              env.player_city_count(seat))
        check(f"road_cnt[{seat}]", gs.player_road_count[seat],
              env.player_road_count(seat))
        check(f"knights[{seat}]", gs.player_knights_played[seat],
              env.player_knights_played(seat))
        check(f"road_len[{seat}]", gs.player_road_length[seat],
              env.player_road_length(seat))
        check(f"ports[{seat}]", gs.player_ports[seat], env.player_ports(seat))
        for r in range(5):
            check(f"res[{seat}][{r}]", gs.player_resources[seat][r],
                  env.player_resource(seat, r))

    # action_mask: mirror must equal the env's incremental mask.
    env.action_mask(mask)
    for w in range(5):
        check(f"action_mask[{w}]", gs.action_mask[w], int(mask[w]))

    # Round-trip: re-serialize and confirm load_snapshot accepts it unchanged.
    env.load_snapshot(to_bytes(snap))
    assert env.current_player == gs.current_player


if __name__ == "__main__":
    for s in (0, 1, 7, 42, 123):
        validate_against_env(seed=s)
    print("state_mirror: layout validated against live env (sizeof=384) — OK")
