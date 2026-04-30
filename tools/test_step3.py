#!/usr/bin/env python3
"""Cheap Python tests for step_one slice 3 (build settlement / city / road in MAIN)."""

from __future__ import annotations
import ctypes
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
SETTLE_BASE, CITY_BASE, ROAD_BASE = 0, 54, 108
ROLL_DICE, END_TURN = 180, 181

NUM_NODES, NUM_EDGES, NUM_HEXES = 54, 72, 19
NO_PLAYER = 0xFF

PHASE_INITIAL_1, PHASE_INITIAL_2, PHASE_MAIN, PHASE_ENDED = 0, 1, 2, 3
FLAG_NONE = 0
NODE_EMPTY, NODE_SETTLEMENT, NODE_CITY = 0, 1, 2

R_BRICK, R_LUMBER, R_WOOL, R_GRAIN, R_ORE = 0, 1, 2, 3, 4

COST_SETTLE = (1, 1, 1, 1, 0)
COST_CITY   = (0, 0, 0, 2, 3)
COST_ROAD   = (1, 1, 0, 0, 0)

# ---------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
HDR = (REPO / "include" / "topology.hpp").read_text()

def _parse(text, name):
    m = re.search(rf"\b{re.escape(name)}\s*=\s*\{{\{{(?P<body>.*?)\}}\}}\s*;", text, re.DOTALL)
    rows = re.findall(r"\{\{([^{}]*)\}\}", m.group("body"))
    out = []
    for r in rows:
        v = [int(x, 16) for x in re.findall(r"0x[0-9A-Fa-f]+", r)]
        while v and v[-1] == 0xFF: v.pop()
        out.append(tuple(v))
    return tuple(out)

NODE_NODE = _parse(HDR, "node_to_node")
NODE_EDGE = _parse(HDR, "node_to_edge")
EDGE_NODE = _parse(HDR, "edge_to_node")

# ---------------------------------------------------------------------
# Lib
# ---------------------------------------------------------------------
LIB = REPO / "build" / ("libfastcatan.dylib" if sys.platform == "darwin" else "libfastcatan.so")
if not LIB.exists():
    print(f"missing {LIB}; run `bash tools/build_lib.sh`"); sys.exit(1)
lib = ctypes.CDLL(str(LIB))

VP, U8, U16, U32, U64, I = (ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint16,
                             ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int)

def _b(name, restype, *argtypes):
    f = getattr(lib, name); f.restype = restype; f.argtypes = list(argtypes); return f

create  = _b("fcatan_create",  VP)
destroy = _b("fcatan_destroy", None, VP)
reset_  = _b("fcatan_reset",   None, VP, U64)
step_   = _b("fcatan_step",    U8,   VP, U32)

phase          = _b("fcatan_phase",            U8, VP)
flag           = _b("fcatan_flag",             U8, VP)
current_player = _b("fcatan_current_player",   U8, VP)
dice_roll      = _b("fcatan_dice_roll",        U8, VP)

node_byte = _b("fcatan_node", U8, VP, I)
edge_byte = _b("fcatan_edge", U8, VP, I)

p_vp        = _b("fcatan_player_vp",                  U8, VP, I)
p_vp_pub    = _b("fcatan_player_vp_public",           U8, VP, I)
p_hand      = _b("fcatan_player_handsize",            U8, VP, I)
p_res       = _b("fcatan_player_resource",            U8, VP, I, I)
p_settle_n  = _b("fcatan_player_settlement_count",    U8, VP, I)
p_city_n    = _b("fcatan_player_city_count",          U8, VP, I)
p_road_n    = _b("fcatan_player_road_count",          U8, VP, I)
bank        = _b("fcatan_bank",                       U8, VP, I)

give_res = _b("fcatan_give_resources", None, VP, I, I, U8)
set_vp   = _b("fcatan_set_player_vp",  None, VP, I, U8)

def node_level(b): return b & 0x03
def node_owner(b): return (b >> 2) & 0x07

class Env:
    def __init__(self): self.h = create()
    def __del__(self):
        try: destroy(self.h)
        except Exception: pass
    def reset(self, seed): reset_(self.h, seed)
    def step(self, action): return step_(self.h, action) != 0


def first_legal_settle_initial(e):
    for n in range(NUM_NODES):
        if node_level(node_byte(e.h, n)) != NODE_EMPTY: continue
        if any(node_level(node_byte(e.h, nb)) != NODE_EMPTY for nb in NODE_NODE[n]):
            continue
        return n
    return -1

def first_unroaded_settle_road(e):
    pl = current_player(e.h)
    for ed in range(NUM_EDGES):
        if edge_byte(e.h, ed) != NO_PLAYER: continue
        for n in EDGE_NODE[ed]:
            b = node_byte(e.h, n)
            if node_level(b) != NODE_SETTLEMENT or node_owner(b) != pl: continue
            already = any(edge_byte(e.h, e2) == pl for e2 in NODE_EDGE[n])
            if not already: return ed
    return -1

def play_initial(e):
    while phase(e.h) != PHASE_MAIN:
        e.step(SETTLE_BASE + first_legal_settle_initial(e))
        e.step(ROAD_BASE   + first_unroaded_settle_road(e))


def give_kit(e, pl, kit):
    """kit: tuple of 5 ints — counts of (brick, lumber, wool, grain, ore)."""
    for r, n in enumerate(kit):
        if n: give_res(e.h, pl, r, n)

def hand_kit(e, pl):
    return tuple(p_res(e.h, pl, r) for r in range(5))


def find_player_settle_node(e, pl):
    """Return some settlement node owned by player."""
    for n in range(NUM_NODES):
        b = node_byte(e.h, n)
        if node_level(b) == NODE_SETTLEMENT and node_owner(b) == pl:
            return n
    return -1

def find_player_road_edge(e, pl):
    for ed in range(NUM_EDGES):
        if edge_byte(e.h, ed) == pl:
            return ed
    return -1

def find_road_legal_for_main(e, pl):
    """Edge where road would be legal in MAIN: empty + connects to player without crossing opponent."""
    for ed in range(NUM_EDGES):
        if edge_byte(e.h, ed) != NO_PLAYER: continue
        for v in EDGE_NODE[ed]:
            n = node_byte(e.h, v)
            lvl = node_level(n)
            if lvl != NODE_EMPTY and node_owner(n) != pl: continue
            if lvl != NODE_EMPTY and node_owner(n) == pl: return ed
            for e2 in NODE_EDGE[v]:
                if e2 == ed: continue
                if edge_byte(e.h, e2) == pl: return ed
    return -1


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_build_road_basic(seed=42):
    e = Env(); e.reset(seed); play_initial(e)
    fails = 0
    pl = current_player(e.h)
    # ensure rolled (non-7)
    while True:
        e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: break
        e.reset(seed + 1); play_initial(e); pl = current_player(e.h)

    give_kit(e, pl, COST_ROAD)
    target = find_road_legal_for_main(e, pl)
    fails += fail(target >= 0, "no legal road found")

    bank_b = [bank(e.h, r) for r in range(5)]
    hand_b = hand_kit(e, pl)
    road_n_b = p_road_n(e.h, pl)

    e.step(ROAD_BASE + target)

    fails += fail(edge_byte(e.h, target) == pl, f"road not placed at {target}")
    fails += fail(p_road_n(e.h, pl) == road_n_b - 1, "road count not decremented")
    # cost paid
    fails += fail(p_res(e.h, pl, R_BRICK)  == hand_b[R_BRICK]  - 1, "brick not paid")
    fails += fail(p_res(e.h, pl, R_LUMBER) == hand_b[R_LUMBER] - 1, "lumber not paid")
    fails += fail(bank(e.h, R_BRICK)  == bank_b[R_BRICK]  + 1, "brick not returned to bank")
    fails += fail(bank(e.h, R_LUMBER) == bank_b[R_LUMBER] + 1, "lumber not returned to bank")
    return fails


def test_build_road_unaffordable(seed=42):
    e = Env(); e.reset(seed); play_initial(e); pl = current_player(e.h)
    while True:
        e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: break
        e.reset(seed + 1); play_initial(e); pl = current_player(e.h)

    target = find_road_legal_for_main(e, pl)
    edge_b = edge_byte(e.h, target)
    hand_b = hand_kit(e, pl)
    e.step(ROAD_BASE + target)
    fails = fail(edge_byte(e.h, target) == edge_b, "road placed without resources")
    fails += fail(hand_kit(e, pl) == hand_b, "hand changed on illegal build")
    return fails


def test_build_road_disconnected(seed=42):
    e = Env(); e.reset(seed); play_initial(e); pl = current_player(e.h)
    while True:
        e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: break
        e.reset(seed + 1); play_initial(e); pl = current_player(e.h)

    give_kit(e, pl, COST_ROAD)
    # find an empty edge where neither endpoint connects to player
    target = -1
    for ed in range(NUM_EDGES):
        if edge_byte(e.h, ed) != NO_PLAYER: continue
        ok = True
        for v in EDGE_NODE[ed]:
            n = node_byte(e.h, v)
            if node_level(n) != NODE_EMPTY and node_owner(n) == pl: ok = False; break
            for e2 in NODE_EDGE[v]:
                if edge_byte(e.h, e2) == pl: ok = False; break
            if not ok: break
        if ok: target = ed; break
    if target < 0: return 0  # all edges connect, nothing to test
    hand_b = hand_kit(e, pl)
    e.step(ROAD_BASE + target)
    fails = fail(edge_byte(e.h, target) == NO_PLAYER, "disconnected road placed")
    fails += fail(hand_kit(e, pl) == hand_b, "hand drained on illegal road")
    return fails


def test_build_settlement_basic(seed=42):
    e = Env(); e.reset(seed); play_initial(e); pl = current_player(e.h)
    while True:
        e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: break
        e.reset(seed + 1); play_initial(e); pl = current_player(e.h)

    # Need to extend road first to reach a new node, then build settlement there
    give_kit(e, pl, (10, 10, 10, 10, 10))  # plenty for several builds

    # Place 2 roads to reach far enough
    placed_roads = 0
    while placed_roads < 2:
        ed = find_road_legal_for_main(e, pl)
        if ed < 0: break
        before = edge_byte(e.h, ed)
        e.step(ROAD_BASE + ed)
        if edge_byte(e.h, ed) == pl: placed_roads += 1
        else: break

    # Find a node adjacent to player road, currently empty, distance OK
    target = -1
    for n in range(NUM_NODES):
        if node_level(node_byte(e.h, n)) != NODE_EMPTY: continue
        # neighbors empty?
        if any(node_level(node_byte(e.h, nb)) != NODE_EMPTY for nb in NODE_NODE[n]): continue
        # adjacent to player road?
        if any(edge_byte(e.h, e2) == pl for e2 in NODE_EDGE[n]):
            target = n; break

    if target < 0: return 0  # couldn't set up; not a failure of slice 3 logic

    settle_b = p_settle_n(e.h, pl)
    vp_b = p_vp(e.h, pl)
    hand_b = hand_kit(e, pl)
    bank_b = [bank(e.h, r) for r in range(5)]

    e.step(SETTLE_BASE + target)

    fails = fail(node_level(node_byte(e.h, target)) == NODE_SETTLEMENT, "settlement not placed")
    fails += fail(node_owner(node_byte(e.h, target)) == pl, "wrong owner")
    fails += fail(p_settle_n(e.h, pl) == settle_b - 1, "settle count not decremented")
    fails += fail(p_vp(e.h, pl) == vp_b + 1, "VP not incremented")
    fails += fail(p_vp_pub(e.h, pl) == vp_b + 1, "public VP not incremented")
    # cost: 1 of each except ore
    for r, c in enumerate(COST_SETTLE):
        fails += fail(p_res(e.h, pl, r) == hand_b[r] - c,
                      f"resource {r} cost {c} not paid (was {hand_b[r]} now {p_res(e.h, pl, r)})")
        fails += fail(bank(e.h, r) == bank_b[r] + c, f"bank {r} not refunded")
    return fails


def test_build_settle_distance_violation(seed=42):
    e = Env(); e.reset(seed); play_initial(e); pl = current_player(e.h)
    while True:
        e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: break
        e.reset(seed + 1); play_initial(e); pl = current_player(e.h)

    give_kit(e, pl, (10, 10, 10, 10, 10))
    # find an empty node adjacent to an existing settlement (any) — distance violation
    target = -1
    for n in range(NUM_NODES):
        if node_level(node_byte(e.h, n)) != NODE_EMPTY: continue
        # has a neighbor with a settlement
        if any(node_level(node_byte(e.h, nb)) != NODE_EMPTY for nb in NODE_NODE[n]):
            target = n; break
    if target < 0: return 0
    hand_b = hand_kit(e, pl)
    e.step(SETTLE_BASE + target)
    fails = fail(node_level(node_byte(e.h, target)) == NODE_EMPTY, "settlement built too close")
    fails += fail(hand_kit(e, pl) == hand_b, "resources spent on illegal settle")
    return fails


def test_build_settle_no_road(seed=42):
    e = Env(); e.reset(seed); play_initial(e); pl = current_player(e.h)
    while True:
        e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: break
        e.reset(seed + 1); play_initial(e); pl = current_player(e.h)

    give_kit(e, pl, (10, 10, 10, 10, 10))
    # Find a node that:
    # - is empty
    # - all neighbors empty (distance OK)
    # - NOT adjacent to a player road
    target = -1
    for n in range(NUM_NODES):
        if node_level(node_byte(e.h, n)) != NODE_EMPTY: continue
        if any(node_level(node_byte(e.h, nb)) != NODE_EMPTY for nb in NODE_NODE[n]): continue
        if not any(edge_byte(e.h, e2) == pl for e2 in NODE_EDGE[n]):
            target = n; break
    if target < 0: return 0
    hand_b = hand_kit(e, pl)
    e.step(SETTLE_BASE + target)
    fails = fail(node_level(node_byte(e.h, target)) == NODE_EMPTY,
                 "settlement built without adjacent road")
    fails += fail(hand_kit(e, pl) == hand_b, "resources spent on illegal settle")
    return fails


def test_build_city_basic(seed=42):
    e = Env(); e.reset(seed); play_initial(e); pl = current_player(e.h)
    while True:
        e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: break
        e.reset(seed + 1); play_initial(e); pl = current_player(e.h)

    give_kit(e, pl, COST_CITY)
    target = find_player_settle_node(e, pl)
    fails = fail(target >= 0, "no settlement to upgrade")

    settle_b = p_settle_n(e.h, pl)
    city_b = p_city_n(e.h, pl)
    vp_b = p_vp(e.h, pl)
    hand_b = hand_kit(e, pl)
    bank_b = [bank(e.h, r) for r in range(5)]

    e.step(CITY_BASE + target)

    fails += fail(node_level(node_byte(e.h, target)) == NODE_CITY, "city not placed")
    fails += fail(node_owner(node_byte(e.h, target)) == pl, "wrong owner on city")
    fails += fail(p_city_n(e.h, pl) == city_b - 1, "city count not decremented")
    fails += fail(p_settle_n(e.h, pl) == settle_b + 1, "settlement piece not returned")
    fails += fail(p_vp(e.h, pl) == vp_b + 1, "VP not incremented (city = +1 over settlement)")
    for r, c in enumerate(COST_CITY):
        fails += fail(p_res(e.h, pl, r) == hand_b[r] - c, f"city cost {r} not paid")
        fails += fail(bank(e.h, r) == bank_b[r] + c, f"city bank refund {r}")
    return fails


def test_city_only_on_own_settle(seed=42):
    e = Env(); e.reset(seed); play_initial(e); pl = current_player(e.h)
    while True:
        e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: break
        e.reset(seed + 1); play_initial(e); pl = current_player(e.h)

    give_kit(e, pl, COST_CITY)
    fails = 0
    # try city on opponent settlement
    other = -1
    for n in range(NUM_NODES):
        b = node_byte(e.h, n)
        if node_level(b) == NODE_SETTLEMENT and node_owner(b) != pl:
            other = n; break
    if other >= 0:
        before_lvl = node_level(node_byte(e.h, other))
        before_owner = node_owner(node_byte(e.h, other))
        e.step(CITY_BASE + other)
        fails += fail(node_level(node_byte(e.h, other)) == before_lvl, "city placed on opponent")
        fails += fail(node_owner(node_byte(e.h, other)) == before_owner, "city stole node")

    # try city on empty node
    empty = -1
    for n in range(NUM_NODES):
        if node_level(node_byte(e.h, n)) == NODE_EMPTY:
            empty = n; break
    if empty >= 0:
        e.step(CITY_BASE + empty)
        fails += fail(node_level(node_byte(e.h, empty)) == NODE_EMPTY,
                      "city placed on empty node")
    return fails


def test_resource_conservation_after_builds(seed=42):
    """Build many things, assert bank + hands always = 95 - paid_to_bank."""
    e = Env(); e.reset(seed); play_initial(e); pl = current_player(e.h)
    while True:
        e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: break
        e.reset(seed + 1); play_initial(e); pl = current_player(e.h)

    fails = 0

    def total_in_play():
        return sum(bank(e.h, r) for r in range(5)) + sum(
            p_res(e.h, p, r) for p in range(4) for r in range(5)
        )

    target_total = total_in_play()  # reset+roll already ran, this is the post-roll total
    fails += fail(target_total == 95, f"pre-build total {target_total} != 95")

    # Buy a kit so player can build
    give_kit(e, pl, (10, 10, 10, 10, 10))
    # Now player has 10 of each, bank decremented accordingly. Total still 95.
    fails += fail(total_in_play() == 95, "total != 95 after give_kit")

    # Build a few roads
    for _ in range(3):
        ed = find_road_legal_for_main(e, pl)
        if ed < 0: break
        e.step(ROAD_BASE + ed)
    fails += fail(total_in_play() == 95, "total != 95 after roads")

    # Build settlement
    target = -1
    for n in range(NUM_NODES):
        if node_level(node_byte(e.h, n)) != NODE_EMPTY: continue
        if any(node_level(node_byte(e.h, nb)) != NODE_EMPTY for nb in NODE_NODE[n]): continue
        if any(edge_byte(e.h, e2) == pl for e2 in NODE_EDGE[n]):
            target = n; break
    if target >= 0:
        e.step(SETTLE_BASE + target)
    fails += fail(total_in_play() == 95, "total != 95 after settle")

    # Upgrade existing settlement to city
    n = find_player_settle_node(e, pl)
    if n >= 0:
        e.step(CITY_BASE + n)
    fails += fail(total_in_play() == 95, "total != 95 after city")
    return fails


def test_winning_triggers_end(seed=42):
    """Force a player to 9 VP, then a single city upgrade pushes to 10
    and triggers Phase::ENDED. Doesn't depend on board reachability."""
    e = Env(); e.reset(seed); play_initial(e); pl = current_player(e.h)
    while True:
        e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: break
        e.reset(seed + 1); play_initial(e); pl = current_player(e.h)

    fails = 0
    set_vp(e.h, pl, 9)
    fails += fail(phase(e.h) == PHASE_MAIN, "phase changed by set_vp alone")

    give_kit(e, pl, COST_CITY)
    n = find_player_settle_node(e, pl)
    fails += fail(n >= 0, "no settlement to upgrade for win test")
    e.step(CITY_BASE + n)

    fails += fail(p_vp(e.h, pl) == 10, f"vp={p_vp(e.h, pl)} (want 10)")
    fails += fail(phase(e.h) == PHASE_ENDED, f"phase={phase(e.h)} (want ENDED)")

    # Subsequent step on terminal state is a no-op.
    e.step(ROLL_DICE)
    fails += fail(phase(e.h) == PHASE_ENDED, "phase regressed after step on ENDED")
    return fails


def test_no_win_below_10(seed=42):
    """Build to VP=9 and verify phase stays MAIN until 10 hit."""
    e = Env(); e.reset(seed); play_initial(e); pl = current_player(e.h)
    while True:
        e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: break
        e.reset(seed + 1); play_initial(e); pl = current_player(e.h)

    set_vp(e.h, pl, 9)
    give_kit(e, pl, COST_ROAD)
    fails = 0
    ed = find_road_legal_for_main(e, pl)
    if ed >= 0:
        e.step(ROAD_BASE + ed)
    fails += fail(phase(e.h) == PHASE_MAIN, "road build triggered ENDED at 9 VP")
    fails += fail(p_vp(e.h, pl) == 9, f"vp shifted to {p_vp(e.h, pl)}")
    return fails


def main():
    total = 0
    print("== test_build_road_basic ==");                  total += test_build_road_basic()
    print("== test_build_road_unaffordable ==");           total += test_build_road_unaffordable()
    print("== test_build_road_disconnected ==");           total += test_build_road_disconnected()
    print("== test_build_settlement_basic ==");            total += test_build_settlement_basic()
    print("== test_build_settle_distance_violation ==");   total += test_build_settle_distance_violation()
    print("== test_build_settle_no_road ==");              total += test_build_settle_no_road()
    print("== test_build_city_basic ==");                  total += test_build_city_basic()
    print("== test_city_only_on_own_settle ==");           total += test_city_only_on_own_settle()
    print("== test_resource_conservation_after_builds =="); total += test_resource_conservation_after_builds()
    print("== test_winning_triggers_end ==");              total += test_winning_triggers_end()
    print("== test_no_win_below_10 ==");                   total += test_no_win_below_10()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} assertions failed")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
