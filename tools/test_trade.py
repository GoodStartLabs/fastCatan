#!/usr/bin/env python3
"""Tests for player-to-player trade (compose -> respond -> confirm)."""

from __future__ import annotations
import ctypes, re, sys
from pathlib import Path

# Action IDs
SETTLE_BASE, CITY_BASE, ROAD_BASE = 0, 54, 108
ROLL_DICE, END_TURN = 180, 181
DISCARD_BASE, MOVE_ROBBER_BASE, STEAL_BASE = 182, 187, 206
TRADE_BASE = 210
BUY_DEV, PLAY_KNIGHT = 235, 236
PLAY_ROAD_BUILDING = 237
PLAY_YEAR_OF_PLENTY = 238
PLAY_MONOPOLY = 263

TRADE_ADD_GIVE_BASE    = 268
TRADE_ADD_WANT_BASE    = 273
TRADE_OPEN             = 278
TRADE_ACCEPT           = 279
TRADE_DECLINE          = 280
TRADE_CONFIRM_BASE     = 281
TRADE_CANCEL           = 285

NUM_NODES, NUM_EDGES, NUM_PLAYERS = 54, 72, 4
NO_PLAYER = 0xFF
NUM_RESOURCES = 5

PHASE_MAIN = 2
FLAG_NONE, FLAG_DISCARD, FLAG_MOVE_ROBBER, FLAG_STEAL, FLAG_YOP, FLAG_MONO, FLAG_PLACE_ROAD, FLAG_TRADE_PENDING = 0, 1, 2, 3, 4, 5, 6, 7
NODE_EMPTY, NODE_SETTLEMENT = 0, 1

R_BRICK, R_LUMBER, R_WOOL, R_GRAIN, R_ORE = 0, 1, 2, 3, 4

TR_PENDING, TR_ACCEPT, TR_DECLINE, TR_NA = 0, 1, 2, 3

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
bank     = _b("fcatan_bank", U8, VP, I)

trade_give     = _b("fcatan_trade_give",         U8, VP, I)
trade_want     = _b("fcatan_trade_want",         U8, VP, I)
trade_response = _b("fcatan_trade_response_byte", U8, VP)
trade_proposer = _b("fcatan_trade_proposer",     U8, VP)

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


def to_post_roll_no7(e, base_seed=42):
    seed = base_seed
    while True:
        e.reset(seed); play_initial(e)
        e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: return seed
        seed += 1


def get_response(env, p):
    return (trade_response(env.h) >> (2*p)) & 0x3


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_compose_add():
    e = Env(); to_post_roll_no7(e)
    pl = current_player(e.h)
    give_res(e.h, pl, R_BRICK, 3)
    fails = 0

    e.step(TRADE_ADD_GIVE_BASE + R_BRICK)
    fails += fail(trade_give(e.h, R_BRICK) == 1, f"add give: {trade_give(e.h, R_BRICK)}")
    e.step(TRADE_ADD_GIVE_BASE + R_BRICK)
    fails += fail(trade_give(e.h, R_BRICK) == 2, f"add give 2x: {trade_give(e.h, R_BRICK)}")

    e.step(TRADE_ADD_WANT_BASE + R_GRAIN)
    fails += fail(trade_want(e.h, R_GRAIN) == 1, f"add want: {trade_want(e.h, R_GRAIN)}")

    return fails


def test_compose_cant_give_more_than_owned():
    e = Env(); to_post_roll_no7(e)
    pl = current_player(e.h)
    # Set brick to exactly 2 (give_res adds; first count current then top up)
    have = p_res(e.h, pl, R_BRICK)
    if have < 2: give_res(e.h, pl, R_BRICK, 2 - have)
    fails = 0
    e.step(TRADE_ADD_GIVE_BASE + R_BRICK)
    e.step(TRADE_ADD_GIVE_BASE + R_BRICK)
    n = trade_give(e.h, R_BRICK)
    e.step(TRADE_ADD_GIVE_BASE + R_BRICK)  # exceeds owned
    fails += fail(trade_give(e.h, R_BRICK) == n,
                  f"add beyond owned: {n} -> {trade_give(e.h, R_BRICK)}")
    return fails


def test_open_validation():
    """TRADE_OPEN no-op when scratch empty / one-sided / unaffordable."""
    e = Env(); to_post_roll_no7(e)
    pl = current_player(e.h)
    fails = 0

    # Empty scratch — open is nop
    e.step(TRADE_OPEN)
    fails += fail(flag(e.h) == FLAG_NONE, "OPEN with empty scratch set flag")

    # Only give, no want
    give_res(e.h, pl, R_BRICK, 1)
    e.step(TRADE_ADD_GIVE_BASE + R_BRICK)
    e.step(TRADE_OPEN)
    fails += fail(flag(e.h) == FLAG_NONE, "OPEN one-sided set flag")

    # Add want, then OPEN should succeed
    e.step(TRADE_ADD_WANT_BASE + R_GRAIN)
    e.step(TRADE_OPEN)
    fails += fail(flag(e.h) == FLAG_TRADE_PENDING, f"OPEN didn't set flag (got {flag(e.h)})")
    fails += fail(trade_proposer(e.h) == pl, "trade_proposer not set")
    fails += fail(current_player(e.h) != pl, "current_player still proposer after OPEN")
    return fails


def test_full_trade_accept_and_confirm():
    e = Env(); to_post_roll_no7(e)
    pl = current_player(e.h)
    # Proposer wants grain; first opponent has grain
    give_res(e.h, pl, R_BRICK, 2)
    other = (pl + 1) & 3
    give_res(e.h, other, R_GRAIN, 2)

    e.step(TRADE_ADD_GIVE_BASE + R_BRICK)
    e.step(TRADE_ADD_WANT_BASE + R_GRAIN)
    e.step(TRADE_OPEN)

    fails = 0
    fails += fail(flag(e.h) == FLAG_TRADE_PENDING, "OPEN failed")

    # Cycle through responses: first opponent accepts, others decline
    safety = 10
    while flag(e.h) == FLAG_TRADE_PENDING and current_player(e.h) != pl and safety > 0:
        safety -= 1
        cp = current_player(e.h)
        if cp == other:
            e.step(TRADE_ACCEPT)
        else:
            e.step(TRADE_DECLINE)

    fails += fail(current_player(e.h) == pl, "didn't return to proposer")
    fails += fail(get_response(e, other) == TR_ACCEPT, f"other response: {get_response(e, other)}")

    # Snapshot before confirm
    pl_brick_b = p_res(e.h, pl, R_BRICK)
    pl_grain_b = p_res(e.h, pl, R_GRAIN)
    other_brick_b = p_res(e.h, other, R_BRICK)
    other_grain_b = p_res(e.h, other, R_GRAIN)
    pl_hand_b = p_hand(e.h, pl)
    other_hand_b = p_hand(e.h, other)

    e.step(TRADE_CONFIRM_BASE + other)

    fails += fail(flag(e.h) == FLAG_NONE, f"flag={flag(e.h)} after confirm")
    fails += fail(p_res(e.h, pl, R_BRICK) == pl_brick_b - 1, "pl brick wrong")
    fails += fail(p_res(e.h, pl, R_GRAIN) == pl_grain_b + 1, "pl grain wrong")
    fails += fail(p_res(e.h, other, R_BRICK) == other_brick_b + 1, "other brick wrong")
    fails += fail(p_res(e.h, other, R_GRAIN) == other_grain_b - 1, "other grain wrong")
    fails += fail(p_hand(e.h, pl) == pl_hand_b, "pl hand changed (should be net 0)")
    fails += fail(p_hand(e.h, other) == other_hand_b, "other hand changed")
    fails += fail(trade_give(e.h, R_BRICK) == 0, "scratch not cleared")
    fails += fail(trade_proposer(e.h) == NO_PLAYER, "proposer not cleared")
    return fails


def test_proposer_picks_among_multiple_acceptors():
    """Two opponents accept; proposer picks one."""
    e = Env(); to_post_roll_no7(e)
    pl = current_player(e.h)
    give_res(e.h, pl, R_BRICK, 1)
    # Give grain to two opponents so they can accept
    o1, o2 = (pl + 1) & 3, (pl + 2) & 3
    give_res(e.h, o1, R_GRAIN, 1)
    give_res(e.h, o2, R_GRAIN, 1)
    o1_brick_pre = p_res(e.h, o1, R_BRICK)
    o2_brick_pre = p_res(e.h, o2, R_BRICK)

    e.step(TRADE_ADD_GIVE_BASE + R_BRICK)
    e.step(TRADE_ADD_WANT_BASE + R_GRAIN)
    e.step(TRADE_OPEN)

    safety = 10
    while flag(e.h) == FLAG_TRADE_PENDING and current_player(e.h) != pl and safety > 0:
        safety -= 1
        cp = current_player(e.h)
        if cp == o1 or cp == o2:
            e.step(TRADE_ACCEPT)
        else:
            e.step(TRADE_DECLINE)

    fails = 0
    fails += fail(get_response(e, o1) == TR_ACCEPT, "o1 should accept")
    fails += fail(get_response(e, o2) == TR_ACCEPT, "o2 should accept")
    fails += fail(current_player(e.h) == pl, "proposer not active for confirm")

    # Confirm with o2
    e.step(TRADE_CONFIRM_BASE + o2)
    fails += fail(flag(e.h) == FLAG_NONE, "flag not cleared")
    fails += fail(p_res(e.h, o2, R_BRICK) == o2_brick_pre + 1, "o2 didn't get brick")
    fails += fail(p_res(e.h, o1, R_BRICK) == o1_brick_pre, "o1 incorrectly got brick (should be untouched)")
    return fails


def test_responder_cant_accept_without_resources():
    """Responder lacking the wanted bundle is force-declined."""
    e = Env(); to_post_roll_no7(e)
    pl = current_player(e.h)
    give_res(e.h, pl, R_BRICK, 1)
    other = (pl + 1) & 3
    # other has no grain
    have_grain = p_res(e.h, other, R_GRAIN)
    # zero out other's grain (no easy setter; use give_res with negative impossible)
    # Instead, just request 5 grain that nobody has
    e.step(TRADE_ADD_GIVE_BASE + R_BRICK)
    for _ in range(5):
        e.step(TRADE_ADD_WANT_BASE + R_GRAIN)
    e.step(TRADE_OPEN)

    fails = 0
    safety = 10
    while flag(e.h) == FLAG_TRADE_PENDING and current_player(e.h) != pl and safety > 0:
        safety -= 1
        cp = current_player(e.h)
        e.step(TRADE_ACCEPT)  # try to accept; should auto-decline if can't pay

    # All forced to decline (insufficient grain). Proposer must cancel.
    for p in range(NUM_PLAYERS):
        if p == pl: continue
        if p_res(e.h, p, R_GRAIN) < 5:
            fails += fail(get_response(e, p) == TR_DECLINE,
                          f"p{p} should be declined (has {p_res(e.h, p, R_GRAIN)} grain)")
    return fails


def test_cancel_clears_scratch():
    e = Env(); to_post_roll_no7(e)
    pl = current_player(e.h)
    give_res(e.h, pl, R_BRICK, 2)
    e.step(TRADE_ADD_GIVE_BASE + R_BRICK)
    e.step(TRADE_ADD_WANT_BASE + R_GRAIN)

    fails = 0
    fails += fail(trade_give(e.h, R_BRICK) == 1, "setup failed")

    e.step(TRADE_CANCEL)
    fails += fail(trade_give(e.h, R_BRICK) == 0, "cancel didn't clear give")
    fails += fail(trade_want(e.h, R_GRAIN) == 0, "cancel didn't clear want")
    fails += fail(flag(e.h) == FLAG_NONE, "flag changed by cancel of empty")
    return fails


def test_cancel_during_pending():
    e = Env(); to_post_roll_no7(e)
    pl = current_player(e.h)
    give_res(e.h, pl, R_BRICK, 1)
    other = (pl + 1) & 3
    give_res(e.h, other, R_GRAIN, 1)

    e.step(TRADE_ADD_GIVE_BASE + R_BRICK)
    e.step(TRADE_ADD_WANT_BASE + R_GRAIN)
    e.step(TRADE_OPEN)

    # Walk through responses without confirming
    safety = 10
    while flag(e.h) == FLAG_TRADE_PENDING and current_player(e.h) != pl and safety > 0:
        safety -= 1
        e.step(TRADE_DECLINE)

    fails = 0
    fails += fail(current_player(e.h) == pl, "didn't return to proposer")
    # All declined → proposer can only cancel
    e.step(TRADE_CANCEL)
    fails += fail(flag(e.h) == FLAG_NONE, "cancel didn't clear flag")
    fails += fail(trade_proposer(e.h) == NO_PLAYER, "proposer not cleared")
    fails += fail(trade_give(e.h, R_BRICK) == 0, "scratch not cleared")
    return fails


def test_resource_conservation_through_trade():
    e = Env(); to_post_roll_no7(e)
    pl = current_player(e.h)
    give_res(e.h, pl, R_BRICK, 1)
    other = (pl + 1) & 3
    give_res(e.h, other, R_GRAIN, 1)

    total_b = sum(bank(e.h, r) for r in range(5)) + sum(p_res(e.h, p, r) for p in range(4) for r in range(5))
    fails = fail(total_b == 95, f"pre-trade total {total_b}")

    e.step(TRADE_ADD_GIVE_BASE + R_BRICK)
    e.step(TRADE_ADD_WANT_BASE + R_GRAIN)
    e.step(TRADE_OPEN)
    safety = 10
    while flag(e.h) == FLAG_TRADE_PENDING and current_player(e.h) != pl and safety > 0:
        safety -= 1
        cp = current_player(e.h)
        if cp == other: e.step(TRADE_ACCEPT)
        else: e.step(TRADE_DECLINE)
    e.step(TRADE_CONFIRM_BASE + other)

    total_a = sum(bank(e.h, r) for r in range(5)) + sum(p_res(e.h, p, r) for p in range(4) for r in range(5))
    fails += fail(total_a == 95, f"post-trade total {total_a}")
    return fails


def test_end_turn_clears_uncommitted_scratch():
    e = Env(); to_post_roll_no7(e)
    pl = current_player(e.h)
    give_res(e.h, pl, R_BRICK, 1)
    e.step(TRADE_ADD_GIVE_BASE + R_BRICK)
    e.step(TRADE_ADD_WANT_BASE + R_GRAIN)
    fails = fail(trade_give(e.h, R_BRICK) == 1, "setup failed")
    e.step(END_TURN)
    fails += fail(trade_give(e.h, R_BRICK) == 0, "scratch not cleared on end_turn")
    fails += fail(trade_want(e.h, R_GRAIN) == 0, "want not cleared on end_turn")
    fails += fail(trade_proposer(e.h) == NO_PLAYER, "proposer not cleared on end_turn")
    return fails


def main():
    total = 0
    print("== test_compose_add ==");                         total += test_compose_add()
    print("== test_compose_cant_give_more_than_owned ==");   total += test_compose_cant_give_more_than_owned()
    print("== test_open_validation ==");                     total += test_open_validation()
    print("== test_full_trade_accept_and_confirm ==");       total += test_full_trade_accept_and_confirm()
    print("== test_proposer_picks_among_multiple_acceptors =="); total += test_proposer_picks_among_multiple_acceptors()
    print("== test_responder_cant_accept_without_resources =="); total += test_responder_cant_accept_without_resources()
    print("== test_cancel_clears_scratch ==");               total += test_cancel_clears_scratch()
    print("== test_cancel_during_pending ==");               total += test_cancel_during_pending()
    print("== test_resource_conservation_through_trade =="); total += test_resource_conservation_through_trade()
    print("== test_end_turn_clears_uncommitted_scratch =="); total += test_end_turn_clears_uncommitted_scratch()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} assertions failed")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
