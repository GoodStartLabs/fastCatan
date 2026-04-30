#!/usr/bin/env python3
"""Cheap Python tests for step_one slice 5 (bank/port trade)."""

from __future__ import annotations
import ctypes, re, sys
from pathlib import Path

SETTLE_BASE, CITY_BASE, ROAD_BASE = 0, 54, 108
ROLL_DICE, END_TURN = 180, 181
TRADE_BASE = 210
TRADE_END  = 235

NUM_NODES, NUM_EDGES, NUM_HEXES, NUM_PLAYERS = 54, 72, 19, 4
NO_PLAYER = 0xFF
PHASE_MAIN = 2
FLAG_NONE = 0
NODE_EMPTY, NODE_SETTLEMENT = 0, 1

R_BRICK, R_LUMBER, R_WOOL, R_GRAIN, R_ORE = 0, 1, 2, 3, 4

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
LIB = REPO / "build" / ("libfastcatan.dylib" if sys.platform == "darwin" else "libfastcatan.so")
if not LIB.exists(): print(f"missing {LIB}; build first"); sys.exit(1)
lib = ctypes.CDLL(str(LIB))
VP, U8, U16, U32, U64, I = (ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint16,
                             ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int)
def _b(name, restype, *argtypes):
    f = getattr(lib, name); f.restype = restype; f.argtypes = list(argtypes); return f

create  = _b("fcatan_create",  VP)
destroy = _b("fcatan_destroy", None, VP)
reset_  = _b("fcatan_reset",   None, VP, U64)
step_   = _b("fcatan_step",    U8,   VP, U32)

phase           = _b("fcatan_phase",            U8, VP)
flag            = _b("fcatan_flag",             U8, VP)
current_player  = _b("fcatan_current_player",   U8, VP)
dice_roll       = _b("fcatan_dice_roll",        U8, VP)

node_byte = _b("fcatan_node", U8, VP, I)
edge_byte = _b("fcatan_edge", U8, VP, I)
p_hand   = _b("fcatan_player_handsize", U8, VP, I)
p_res    = _b("fcatan_player_resource", U8, VP, I, I)
p_ports  = _b("fcatan_player_ports", U8, VP, I)
bank     = _b("fcatan_bank", U8, VP, I)

give_res = _b("fcatan_give_resources", None, VP, I, I, U8)

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

def to_post_roll_no7(e, base_seed):
    """Reset+initial+roll until non-7 result. Returns the seed used."""
    seed = base_seed
    while True:
        e.reset(seed); play_initial(e)
        e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: return seed
        seed += 1


def trade_action(give, get):
    return TRADE_BASE + give * 5 + get


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_bank_trade_4to1():
    e = Env(); to_post_roll_no7(e, 42)
    pl = current_player(e.h)

    # Find a player with no ports (cleanest 4:1 test). If our pl has ports, pick
    # a give-resource that's NOT covered by 2:1 OR set ports to 0 via giving them
    # a resource without 3:1 either. To keep test robust, just pick give to be
    # any resource where pl doesn't have a 2:1 nor a 3:1.
    give = -1
    for r in range(5):
        # would resolve to 4:1 only if no 2:1[r] and no 3:1
        if (p_ports(e.h, pl) >> r) & 1: continue
        if (p_ports(e.h, pl) >> 5) & 1: continue
        give = r; break
    if give < 0:
        # player has 3:1 — fall back: simulate by stripping ports... no setter
        # available. Skip.
        return 0

    get = (give + 1) % 5
    fails = 0
    give_res(e.h, pl, give, 4)
    hand_b = p_hand(e.h, pl); res_g_b = p_res(e.h, pl, give); res_w_b = p_res(e.h, pl, get)
    bank_g_b = bank(e.h, give); bank_w_b = bank(e.h, get)

    e.step(trade_action(give, get))
    fails += fail(p_res(e.h, pl, give) == res_g_b - 4, f"give res: {res_g_b} -> {p_res(e.h, pl, give)} (want -4)")
    fails += fail(p_res(e.h, pl, get) == res_w_b + 1, f"get res: {res_w_b} -> {p_res(e.h, pl, get)} (want +1)")
    fails += fail(bank(e.h, give) == bank_g_b + 4, f"bank give: {bank_g_b} -> {bank(e.h, give)}")
    fails += fail(bank(e.h, get) == bank_w_b - 1, f"bank get: {bank_w_b} -> {bank(e.h, get)}")
    fails += fail(p_hand(e.h, pl) == hand_b - 3, f"hand: {hand_b} -> {p_hand(e.h, pl)} (want -3)")
    return fails


def test_bank_trade_unaffordable():
    e = Env(); to_post_roll_no7(e, 42)
    pl = current_player(e.h)
    # find a give where ratio is 4 and player has < 4 of it
    give = -1
    for r in range(5):
        if (p_ports(e.h, pl) >> r) & 1: continue
        if (p_ports(e.h, pl) >> 5) & 1: continue
        if p_res(e.h, pl, r) < 4: give = r; break
    if give < 0: return 0
    get = (give + 1) % 5
    fails = 0
    hand_b = p_hand(e.h, pl); res_b = p_res(e.h, pl, give)
    e.step(trade_action(give, get))
    fails += fail(p_res(e.h, pl, give) == res_b, "give res changed")
    fails += fail(p_hand(e.h, pl) == hand_b, "hand changed on illegal trade")
    return fails


def test_trade_same_resource_nop():
    e = Env(); to_post_roll_no7(e, 42)
    pl = current_player(e.h)
    give_res(e.h, pl, R_BRICK, 5)
    fails = 0
    hand_b = p_hand(e.h, pl); res_b = p_res(e.h, pl, R_BRICK)
    e.step(trade_action(R_BRICK, R_BRICK))
    fails += fail(p_res(e.h, pl, R_BRICK) == res_b, "diagonal trade changed resource")
    fails += fail(p_hand(e.h, pl) == hand_b, "diagonal trade changed hand")
    return fails


def test_port_3to1_when_player_has_generic():
    """Find a player with a 3:1 generic port and verify ratio=3."""
    for seed in range(200):
        e = Env(); e.reset(seed); play_initial(e)
        # any player with bit 5 set?
        target = -1
        for p in range(NUM_PLAYERS):
            if (p_ports(e.h, p) >> 5) & 1:
                target = p; break
        if target < 0: continue

        # Roll non-7 with that player as current
        while current_player(e.h) != target:
            # advance turns until current matches target
            if dice_roll(e.h) == 0:
                e.step(ROLL_DICE)
                if flag(e.h) != FLAG_NONE: break
            e.step(END_TURN)
        if current_player(e.h) != target: continue
        if dice_roll(e.h) == 0: e.step(ROLL_DICE)
        if dice_roll(e.h) == 7 or flag(e.h) != FLAG_NONE: continue

        # find a resource where 2:1 isn't available (so 3:1 wins)
        give = -1
        for r in range(5):
            if not ((p_ports(e.h, target) >> r) & 1):
                give = r; break
        if give < 0: continue
        get = (give + 1) % 5

        give_res(e.h, target, give, 3)
        hand_b = p_hand(e.h, target); res_g_b = p_res(e.h, target, give)
        e.step(trade_action(give, get))
        fails = 0
        fails += fail(p_res(e.h, target, give) == res_g_b - 3, "3:1 give wrong")
        fails += fail(p_res(e.h, target, get) == 1 or p_res(e.h, target, get) > 0, "3:1 get not received")
        fails += fail(p_hand(e.h, target) == hand_b - 2, f"3:1 hand delta: {hand_b - p_hand(e.h, target)} (want 2)")
        return fails
    return 0  # no seed produced a 3:1 generic port owner — skip


def test_port_2to1_when_player_has_specific():
    """Find a player with a 2:1 specific port and verify ratio=2 for that resource."""
    for seed in range(300):
        e = Env(); e.reset(seed); play_initial(e)
        target = -1; res = -1
        for p in range(NUM_PLAYERS):
            for r in range(5):
                if (p_ports(e.h, p) >> r) & 1:
                    target = p; res = r; break
            if target >= 0: break
        if target < 0: continue

        # Cycle to make `target` current_player (post-roll, no flag)
        ok = False
        for _ in range(20):
            if current_player(e.h) == target:
                if dice_roll(e.h) == 0: e.step(ROLL_DICE)
                if dice_roll(e.h) != 7 and flag(e.h) == FLAG_NONE:
                    ok = True; break
                # 7 rolled — skip
                break
            else:
                if dice_roll(e.h) == 0:
                    e.step(ROLL_DICE)
                    if flag(e.h) != FLAG_NONE: break
                e.step(END_TURN)
        if not ok: continue

        get = (res + 1) % 5
        give_res(e.h, target, res, 2)
        hand_b = p_hand(e.h, target); res_g_b = p_res(e.h, target, res); res_w_b = p_res(e.h, target, get)
        e.step(trade_action(res, get))
        fails = 0
        fails += fail(p_res(e.h, target, res) == res_g_b - 2, f"2:1 give: {res_g_b} -> {p_res(e.h, target, res)}")
        fails += fail(p_res(e.h, target, get) == res_w_b + 1, f"2:1 get not received")
        fails += fail(p_hand(e.h, target) == hand_b - 1, f"2:1 hand delta: {hand_b - p_hand(e.h, target)} (want 1)")
        return fails
    return 0  # no seed found — skip


def test_resource_conservation_via_trade():
    e = Env(); to_post_roll_no7(e, 42)
    pl = current_player(e.h)
    give_res(e.h, pl, R_BRICK, 8)
    total_b = sum(bank(e.h, r) for r in range(5)) + sum(p_res(e.h, p, r) for p in range(4) for r in range(5))
    fails = fail(total_b == 95, f"pre-trade total {total_b} != 95")
    # do some trades regardless of port config
    e.step(trade_action(R_BRICK, R_GRAIN))
    e.step(trade_action(R_BRICK, R_ORE))
    total_a = sum(bank(e.h, r) for r in range(5)) + sum(p_res(e.h, p, r) for p in range(4) for r in range(5))
    fails += fail(total_a == 95, f"post-trade total {total_a} != 95")
    return fails


def test_trade_blocked_in_subphase():
    """During DISCARD, MOVE_ROBBER, ROBBER_STEAL, trade is a no-op."""
    # Find a 7-roll seed
    seed = None
    for s in range(2000):
        e = Env(); e.reset(s); play_initial(e); e.step(ROLL_DICE)
        if dice_roll(e.h) == 7: seed = s; break
    if seed is None: return 0
    e = Env(); e.reset(seed); play_initial(e); pl = current_player(e.h)
    # force a discarder
    other = (pl + 1) & 3
    give_res(e.h, other, R_BRICK, max(0, 9 - p_hand(e.h, other)))
    e.step(ROLL_DICE)
    if flag(e.h) == FLAG_NONE: return 0  # no discard happened

    fails = 0
    pre_hand = [p_hand(e.h, p) for p in range(4)]
    pre_bank = [bank(e.h, r) for r in range(5)]
    e.step(trade_action(R_BRICK, R_GRAIN))
    fails += fail(all(p_hand(e.h, p) == pre_hand[p] for p in range(4)), "trade in sub-phase changed hand")
    fails += fail(all(bank(e.h, r) == pre_bank[r] for r in range(5)), "trade in sub-phase changed bank")
    return fails


def test_trade_no_bank_supply():
    """If bank is out of `get` resource, trade is nop."""
    e = Env(); to_post_roll_no7(e, 42)
    pl = current_player(e.h)
    give_res(e.h, pl, R_BRICK, 10)
    # Drain bank of grain to 0 by giving it all to someone
    grain_in_bank = bank(e.h, R_GRAIN)
    if grain_in_bank > 0:
        # give all grain to other player
        for p in range(NUM_PLAYERS):
            if p != pl:
                give_res(e.h, p, R_GRAIN, grain_in_bank)
                break
    fails = fail(bank(e.h, R_GRAIN) == 0, "couldn't drain bank for test")
    res_b = p_res(e.h, pl, R_BRICK); hand_b = p_hand(e.h, pl)
    e.step(trade_action(R_BRICK, R_GRAIN))
    fails += fail(p_res(e.h, pl, R_BRICK) == res_b, "brick changed when bank empty")
    fails += fail(p_hand(e.h, pl) == hand_b, "hand changed when bank empty")
    return fails


def main():
    total = 0
    print("== test_bank_trade_4to1 ==");                  total += test_bank_trade_4to1()
    print("== test_bank_trade_unaffordable ==");          total += test_bank_trade_unaffordable()
    print("== test_trade_same_resource_nop ==");          total += test_trade_same_resource_nop()
    print("== test_port_3to1_when_player_has_generic =="); total += test_port_3to1_when_player_has_generic()
    print("== test_port_2to1_when_player_has_specific =="); total += test_port_2to1_when_player_has_specific()
    print("== test_resource_conservation_via_trade =="); total += test_resource_conservation_via_trade()
    print("== test_trade_blocked_in_subphase ==");       total += test_trade_blocked_in_subphase()
    print("== test_trade_no_bank_supply ==");            total += test_trade_no_bank_supply()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} assertions failed")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
