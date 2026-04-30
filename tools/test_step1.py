#!/usr/bin/env python3
"""Cheap Python tests for step_one slice 1 (initial placement).

Loads build/libfastcatan.{dylib,so} via ctypes. No nanobind, no CMake.

Build first:
    bash tools/build_lib.sh

Run:
    python3 tools/test_step1.py
"""

from __future__ import annotations
import ctypes
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Constants mirroring rules.hpp / topology.hpp
# ---------------------------------------------------------------------
SETTLE_BASE = 0
CITY_BASE = 54
ROAD_BASE = 108

NUM_NODES = 54
NUM_EDGES = 72
NUM_HEXES = 19
NUM_PORTS = 9

NO_PLAYER = 0xFF

PHASE_INITIAL_1 = 0
PHASE_INITIAL_2 = 1
PHASE_MAIN      = 2
PHASE_ENDED     = 3

NODE_EMPTY      = 0
NODE_SETTLEMENT = 1
NODE_CITY       = 2

# ---------------------------------------------------------------------
# Topology tables (parsed from include/topology.hpp by viz_topology.py
# style; here we duplicate the small subset needed for this test).
# ---------------------------------------------------------------------
import re

def _parse_table(text: str, name: str) -> tuple[tuple[int, ...], ...]:
    pat = re.compile(rf"\b{re.escape(name)}\s*=\s*\{{\{{(?P<body>.*?)\}}\}}\s*;",
                     re.DOTALL)
    m = pat.search(text)
    if not m: raise KeyError(name)
    rows = re.findall(r"\{\{([^{}]*)\}\}", m.group("body"))
    out = []
    for r in rows:
        vals = [int(x, 16) for x in re.findall(r"0x[0-9A-Fa-f]+", r)]
        while vals and vals[-1] == 0xFF:
            vals.pop()
        out.append(tuple(vals))
    return tuple(out)

REPO = Path(__file__).resolve().parents[1]
HDR = (REPO / "include" / "topology.hpp").read_text()
NODE_NODE = _parse_table(HDR, "node_to_node")
NODE_EDGE = _parse_table(HDR, "node_to_edge")
EDGE_NODE = _parse_table(HDR, "edge_to_node")

# ---------------------------------------------------------------------
# Load shared library
# ---------------------------------------------------------------------
LIB_DIR = REPO / "build"
LIB_NAME = "libfastcatan.dylib" if sys.platform == "darwin" else "libfastcatan.so"
LIB_PATH = LIB_DIR / LIB_NAME

if not LIB_PATH.exists():
    print(f"missing {LIB_PATH}; run `bash tools/build_lib.sh` first")
    sys.exit(1)

lib = ctypes.CDLL(str(LIB_PATH))

def _bind(name, restype, *argtypes):
    f = getattr(lib, name)
    f.restype = restype
    f.argtypes = list(argtypes)
    return f

VP = ctypes.c_void_p
U8, U16, U32, U64, I = (ctypes.c_uint8, ctypes.c_uint16,
                        ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int)

create  = _bind("fcatan_create",  VP)
destroy = _bind("fcatan_destroy", None, VP)
reset_  = _bind("fcatan_reset",   None, VP, U64)
step_   = _bind("fcatan_step",    U8,  VP, U32)

phase           = _bind("fcatan_phase",            U8, VP)
flag            = _bind("fcatan_flag",             U8, VP)
current_player  = _bind("fcatan_current_player",   U8, VP)
start_player    = _bind("fcatan_start_player",     U8, VP)
robber_hex      = _bind("fcatan_robber_hex",       U8, VP)
dice_roll       = _bind("fcatan_dice_roll",        U8, VP)
turn_count      = _bind("fcatan_turn_count",       U16, VP)
longest_owner   = _bind("fcatan_longest_road_owner", U8, VP)
largest_owner   = _bind("fcatan_largest_army_owner", U8, VP)

node_byte = _bind("fcatan_node", U8, VP, I)
edge_byte = _bind("fcatan_edge", U8, VP, I)
hex_resource = _bind("fcatan_hex_resource", U8, VP, I)
hex_number   = _bind("fcatan_hex_number",   U8, VP, I)
port_type    = _bind("fcatan_port_type",    U8, VP, I)
port_layout  = _bind("fcatan_port_layout",  U8, VP)

p_vp        = _bind("fcatan_player_vp",            U8, VP, I)
p_vp_pub    = _bind("fcatan_player_vp_public",     U8, VP, I)
p_hand      = _bind("fcatan_player_handsize",      U8, VP, I)
p_res       = _bind("fcatan_player_resource",     U8, VP, I, I)
p_settle_n  = _bind("fcatan_player_settlement_count", U8, VP, I)
p_city_n    = _bind("fcatan_player_city_count",       U8, VP, I)
p_road_n    = _bind("fcatan_player_road_count",       U8, VP, I)
p_ports     = _bind("fcatan_player_ports",         U8, VP, I)

bank     = _bind("fcatan_bank",     U8, VP, I)
dev_deck = _bind("fcatan_dev_deck", U8, VP, I)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def node_level(b: int) -> int: return b & 0x03
def node_owner(b: int) -> int: return (b >> 2) & 0x07

class Env:
    def __init__(self):
        self.h = create()
    def __del__(self):
        try: destroy(self.h)
        except Exception: pass
    def reset(self, seed: int): reset_(self.h, seed)
    def step(self, action: int) -> bool: return step_(self.h, action) != 0
    def __getattr__(self, name):
        # convenience: e.phase, e.current_player, etc.
        f = globals().get(name)
        if f is None: raise AttributeError(name)
        return f(self.h)


def first_legal_settle(e: Env) -> int:
    for n in range(NUM_NODES):
        b = node_byte(e.h, n)
        if node_level(b) != NODE_EMPTY: continue
        if any(node_level(node_byte(e.h, nb)) != NODE_EMPTY for nb in NODE_NODE[n]):
            continue
        return n
    return -1


def first_legal_road(e: Env) -> int:
    """Road touching player's most recently placed (un-roaded) settlement."""
    pl = current_player(e.h)
    for ed in range(NUM_EDGES):
        if edge_byte(e.h, ed) != NO_PLAYER: continue
        for n in EDGE_NODE[ed]:
            b = node_byte(e.h, n)
            if node_level(b) != NODE_SETTLEMENT or node_owner(b) != pl: continue
            already = any(edge_byte(e.h, e2) == pl for e2 in NODE_EDGE[n])
            if not already: return ed
    return -1


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def fail(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        return 1
    return 0


def play_initial(e: Env) -> tuple[list[int], list[int]]:
    nodes_placed, edges_placed = [], []
    while phase(e.h) != PHASE_MAIN:
        n = first_legal_settle(e)
        e.step(SETTLE_BASE + n)
        nodes_placed.append(n)
        ed = first_legal_road(e)
        e.step(ROAD_BASE + ed)
        edges_placed.append(ed)
    return nodes_placed, edges_placed


def test_basic(seed: int = 42) -> int:
    e = Env()
    e.reset(seed)
    fails = 0

    fails += fail(phase(e.h) == PHASE_INITIAL_1, f"start phase={phase(e.h)}")
    fails += fail(current_player(e.h) == start_player(e.h),
                  "current != start at reset")

    nodes, edges = play_initial(e)
    fails += fail(len(nodes) == 8, f"placed {len(nodes)} settlements (want 8)")
    fails += fail(len(edges) == 8, f"placed {len(edges)} roads (want 8)")
    fails += fail(phase(e.h) == PHASE_MAIN, f"end phase={phase(e.h)} (want MAIN)")
    fails += fail(current_player(e.h) == start_player(e.h),
                  "MAIN current != start_player")

    # invariants
    for pl in range(4):
        fails += fail(p_settle_n(e.h, pl) == 3, f"p{pl} sett_left={p_settle_n(e.h, pl)}")
        fails += fail(p_road_n(e.h, pl)   == 13, f"p{pl} road_left={p_road_n(e.h, pl)}")
        fails += fail(p_vp(e.h, pl) == 2, f"p{pl} vp={p_vp(e.h, pl)}")
        fails += fail(p_vp_pub(e.h, pl) == 2, f"p{pl} vp_pub={p_vp_pub(e.h, pl)}")

    # resource conservation: 95 = bank + hands
    bsum = sum(bank(e.h, r) for r in range(5))
    hsum = sum(p_res(e.h, pl, r) for pl in range(4) for r in range(5))
    fails += fail(bsum + hsum == 95, f"resource sum {bsum + hsum} != 95")

    # distance rule
    for n in range(NUM_NODES):
        if node_level(node_byte(e.h, n)) == NODE_EMPTY: continue
        for nb in NODE_NODE[n]:
            if node_level(node_byte(e.h, nb)) != NODE_EMPTY:
                fails += fail(False, f"distance violated: {n}↔{nb}")

    # bank doesn't go negative; total dev deck unchanged
    for r in range(5):
        fails += fail(bank(e.h, r) <= 19, f"bank[{r}]={bank(e.h, r)}")
    expected_dev = [14, 5, 2, 2, 2]
    for d in range(5):
        fails += fail(dev_deck(e.h, d) == expected_dev[d],
                      f"dev_deck[{d}]={dev_deck(e.h, d)}")

    # awards still unowned
    fails += fail(longest_owner(e.h) == NO_PLAYER, "longest_owner should be unowned")
    fails += fail(largest_owner(e.h) == NO_PLAYER, "largest_owner should be unowned")

    return fails


def test_determinism(seed: int = 42) -> int:
    e1, e2 = Env(), Env()
    e1.reset(seed); e2.reset(seed)
    n1, _ = play_initial(e1)
    n2, _ = play_initial(e2)
    fails = fail(n1 == n2, f"placement order diverged: {n1} vs {n2}")
    # also check final state byte-equal in a few representative fields
    for pl in range(4):
        fails += fail(p_hand(e1.h, pl) == p_hand(e2.h, pl),
                      f"hand p{pl} diverges")
        fails += fail(p_ports(e1.h, pl) == p_ports(e2.h, pl),
                      f"ports p{pl} diverges")
    return fails


def test_seed_independence() -> int:
    e1, e2 = Env(), Env()
    e1.reset(42); e2.reset(43)
    # Boards should differ on at least one hex resource.
    diff = any(hex_resource(e1.h, h) != hex_resource(e2.h, h) for h in range(NUM_HEXES))
    return fail(diff, "different seeds produced identical boards")


def test_invalid_action_is_nop() -> int:
    e = Env()
    e.reset(7)
    # try a city action in initial phase — illegal
    h_before = bytes(node_byte(e.h, n) for n in range(NUM_NODES))
    cp_before = current_player(e.h)
    e.step(CITY_BASE + 0)
    h_after = bytes(node_byte(e.h, n) for n in range(NUM_NODES))
    fails = fail(h_before == h_after, "illegal action mutated nodes")
    fails += fail(current_player(e.h) == cp_before, "illegal action advanced player")
    return fails


def test_fuzz(n_seeds: int = 200) -> int:
    fails = 0
    for seed in range(n_seeds):
        e = Env()
        e.reset(seed)
        play_initial(e)
        if phase(e.h) != PHASE_MAIN:
            fails += fail(False, f"seed={seed} stuck in phase {phase(e.h)}")
        bsum = sum(bank(e.h, r) for r in range(5))
        hsum = sum(p_res(e.h, pl, r) for pl in range(4) for r in range(5))
        if bsum + hsum != 95:
            fails += fail(False, f"seed={seed} resource sum {bsum + hsum}")
        # distance rule
        for n in range(NUM_NODES):
            if node_level(node_byte(e.h, n)) == NODE_EMPTY: continue
            for nb in NODE_NODE[n]:
                if node_level(node_byte(e.h, nb)) != NODE_EMPTY:
                    fails += fail(False, f"seed={seed} distance: {n}↔{nb}")
                    break
    print(f"fuzz {n_seeds} seeds: {fails} failures")
    return fails


def main():
    total = 0
    print("== test_basic(seed=42) ==");           total += test_basic(42)
    print("== test_basic(seed=7) ==");            total += test_basic(7)
    print("== test_determinism ==");              total += test_determinism(42)
    print("== test_seed_independence ==");        total += test_seed_independence()
    print("== test_invalid_action_is_nop ==");    total += test_invalid_action_is_nop()
    print("== test_fuzz(200) ==");                total += test_fuzz(200)
    print()
    if total == 0:
        print("ALL TESTS PASS")
    else:
        print(f"FAIL — {total} assertions failed")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
