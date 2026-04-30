#!/usr/bin/env python3
"""Cheap Python tests for step_one slice 4 (robber: discard / move / steal)."""

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
DISCARD_BASE = 182
MOVE_ROBBER_BASE = 187
STEAL_BASE = 206

NUM_NODES, NUM_EDGES, NUM_HEXES, NUM_PLAYERS = 54, 72, 19, 4
NO_PLAYER = 0xFF

PHASE_MAIN, PHASE_ENDED = 2, 3
FLAG_NONE, FLAG_DISCARD, FLAG_MOVE_ROBBER, FLAG_STEAL = 0, 1, 2, 3

NODE_EMPTY, NODE_SETTLEMENT, NODE_CITY = 0, 1, 2
DESERT = 5

R_BRICK, R_LUMBER, R_WOOL, R_GRAIN, R_ORE = 0, 1, 2, 3, 4
RESOURCE_NAMES = ['brick', 'lumber', 'wool', 'grain', 'ore']

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
HEX_NODE  = _parse(HDR, "hex_to_node")

# ---------------------------------------------------------------------
# Lib
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
rolling_player  = _b("fcatan_rolling_player",   U8, VP)
dice_roll       = _b("fcatan_dice_roll",        U8, VP)
robber_hex      = _b("fcatan_robber_hex",       U8, VP)
discard_left    = _b("fcatan_player_discard_remaining", U8, VP, I)

node_byte = _b("fcatan_node", U8, VP, I)
edge_byte = _b("fcatan_edge", U8, VP, I)
hex_resource = _b("fcatan_hex_resource", U8, VP, I)

p_hand   = _b("fcatan_player_handsize", U8, VP, I)
p_res    = _b("fcatan_player_resource", U8, VP, I, I)
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


def hand_total(e):
    return sum(p_res(e.h, p, r) for p in range(NUM_PLAYERS) for r in range(5))

def bank_total(e):
    return sum(bank(e.h, r) for r in range(5))


def find_seed_with_first_roll_seven():
    for seed in range(2000):
        e = Env(); e.reset(seed); play_initial(e)
        e.step(ROLL_DICE)
        if dice_roll(e.h) == 7:
            return seed
    return None


def hex_opponents_owners(e, hex_id, of_player):
    """Return set of unique opponent owners of settle/city on this hex."""
    out = set()
    for v in HEX_NODE[hex_id]:
        b = node_byte(e.h, v)
        if node_level(b) == NODE_EMPTY: continue
        owner = node_owner(b)
        if owner != of_player: out.add(owner)
    return out


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_seven_no_discard():
    """7 rolled but everyone has ≤7 cards → flag=MOVE_ROBBER directly."""
    seed = find_seed_with_first_roll_seven()
    if seed is None: return 0
    e = Env(); e.reset(seed); play_initial(e)
    fails = 0
    # Right after initial placement, hand sizes are small (≤3)
    for p in range(NUM_PLAYERS):
        if p_hand(e.h, p) > 7:
            return 0  # skip this scenario
    pre_player = current_player(e.h)
    e.step(ROLL_DICE)
    fails += fail(dice_roll(e.h) == 7, "expected 7")
    fails += fail(flag(e.h) == FLAG_MOVE_ROBBER, f"flag={flag(e.h)} (want MOVE_ROBBER)")
    fails += fail(rolling_player(e.h) == pre_player, "rolling_player not preserved")
    fails += fail(current_player(e.h) == pre_player, "current_player changed without discards")
    return fails


def test_seven_with_discards():
    """One player has >7 cards → DISCARD_RESOURCES, current_player rotates."""
    seed = find_seed_with_first_roll_seven()
    if seed is None: return 0
    e = Env(); e.reset(seed); play_initial(e)
    pre = current_player(e.h)
    other = (pre + 1) & 3

    # Push other player to 9 cards
    give_res(e.h, other, R_BRICK, 8)  # other now has at least 8 brick + their starting cards
    # Their handsize should be > 7
    if p_hand(e.h, other) <= 7:
        give_res(e.h, other, R_BRICK, 9 - p_hand(e.h, other))

    fails = 0
    e.step(ROLL_DICE)
    fails += fail(dice_roll(e.h) == 7, f"expected 7, got {dice_roll(e.h)}")
    fails += fail(flag(e.h) == FLAG_DISCARD, f"flag={flag(e.h)} (want DISCARD)")
    fails += fail(rolling_player(e.h) == pre, "rolling_player not set to pre-roll player")
    fails += fail(discard_left(e.h, other) == p_hand(e.h, other) // 2,
                  f"discard_remaining wrong: {discard_left(e.h, other)} for hand {p_hand(e.h, other)}")
    fails += fail(current_player(e.h) == other,
                  f"current_player should be discarder {other}, got {current_player(e.h)}")
    return fails


def test_discard_one_card():
    seed = find_seed_with_first_roll_seven()
    if seed is None: return 0
    e = Env(); e.reset(seed); play_initial(e)
    pre = current_player(e.h)
    other = (pre + 1) & 3
    give_res(e.h, other, R_BRICK, 12)  # plenty
    e.step(ROLL_DICE)
    if flag(e.h) != FLAG_DISCARD: return 0
    discarder = current_player(e.h)

    fails = 0
    # discarder must have at least some brick to discard
    hand_b = p_hand(e.h, discarder)
    bank_b = bank(e.h, R_BRICK)
    res_b  = p_res(e.h, discarder, R_BRICK)
    rem_b  = discard_left(e.h, discarder)

    e.step(DISCARD_BASE + R_BRICK)

    fails += fail(p_hand(e.h, discarder) == hand_b - 1, "hand didn't drop on discard")
    fails += fail(bank(e.h, R_BRICK) == bank_b + 1, "bank didn't refund discard")
    fails += fail(p_res(e.h, discarder, R_BRICK) == res_b - 1, "brick count didn't drop")
    fails += fail(discard_left(e.h, discarder) == rem_b - 1, "discard_remaining not decremented")
    return fails


def test_discard_no_resource_nop():
    """Discarding a resource you don't have is a no-op."""
    seed = find_seed_with_first_roll_seven()
    if seed is None: return 0
    e = Env(); e.reset(seed); play_initial(e)
    pre = current_player(e.h)
    other = (pre + 1) & 3
    give_res(e.h, other, R_BRICK, 12)  # only brick
    e.step(ROLL_DICE)
    if flag(e.h) != FLAG_DISCARD: return 0
    discarder = current_player(e.h)

    # Try discarding wool (might or might not have any depending on initial placement)
    if p_res(e.h, discarder, R_WOOL) > 0:
        return 0  # has wool, can't test "no-op" cleanly
    fails = 0
    rem_b = discard_left(e.h, discarder)
    hand_b = p_hand(e.h, discarder)
    e.step(DISCARD_BASE + R_WOOL)
    fails += fail(discard_left(e.h, discarder) == rem_b, "discard remaining changed on nop")
    fails += fail(p_hand(e.h, discarder) == hand_b, "hand changed on nop")
    return fails


def test_full_discard_then_move_robber():
    """Discard all required cards, transition to MOVE_ROBBER, current=rolling."""
    seed = find_seed_with_first_roll_seven()
    if seed is None: return 0
    e = Env(); e.reset(seed); play_initial(e)
    pre = current_player(e.h)
    other = (pre + 1) & 3

    # Push other to 8 cards (must discard 4) by giving 8 brick
    delta = max(0, 8 - p_hand(e.h, other))
    if delta:
        give_res(e.h, other, R_BRICK, delta)
    # Ensure rolling player <=7 so only `other` discards
    if p_hand(e.h, pre) > 7:
        return 0  # skip

    e.step(ROLL_DICE)
    if flag(e.h) != FLAG_DISCARD: return 0

    fails = 0
    safety = 0
    while flag(e.h) == FLAG_DISCARD and safety < 30:
        safety += 1
        # discard whatever resource the discarder has
        d = current_player(e.h)
        for r in range(5):
            if p_res(e.h, d, r) > 0:
                e.step(DISCARD_BASE + r)
                break

    fails += fail(flag(e.h) == FLAG_MOVE_ROBBER, f"flag after discards={flag(e.h)} (want MOVE_ROBBER)")
    fails += fail(current_player(e.h) == rolling_player(e.h),
                  "current should return to rolling player after discards")
    return fails


def test_move_robber_invalid_same_hex():
    seed = find_seed_with_first_roll_seven()
    if seed is None: return 0
    e = Env(); e.reset(seed); play_initial(e)
    pre = current_player(e.h)
    e.step(ROLL_DICE)
    if flag(e.h) != FLAG_MOVE_ROBBER: return 0
    cur_hex = robber_hex(e.h)
    fails = 0
    e.step(MOVE_ROBBER_BASE + cur_hex)
    fails += fail(robber_hex(e.h) == cur_hex, "robber moved to same hex")
    fails += fail(flag(e.h) == FLAG_MOVE_ROBBER, "flag cleared on invalid move")
    return fails


def test_move_robber_no_victims():
    """Move robber to a hex with no opponents → flag clears, no steal."""
    seed = find_seed_with_first_roll_seven()
    if seed is None: return 0
    e = Env(); e.reset(seed); play_initial(e)
    pre = current_player(e.h)
    e.step(ROLL_DICE)
    if flag(e.h) != FLAG_MOVE_ROBBER: return 0

    # Find a hex (different from current robber) with no opponents
    target = -1
    for h in range(NUM_HEXES):
        if h == robber_hex(e.h): continue
        if len(hex_opponents_owners(e, h, pre)) == 0:
            target = h; break
    if target < 0: return 0  # rare: all hexes have opponents

    fails = 0
    bank_b = bank_total(e); hand_b = hand_total(e)
    e.step(MOVE_ROBBER_BASE + target)
    fails += fail(robber_hex(e.h) == target, "robber not moved")
    fails += fail(flag(e.h) == FLAG_NONE, f"flag={flag(e.h)} after no-victim move (want NONE)")
    fails += fail(bank_total(e) == bank_b, "bank changed with no steal")
    fails += fail(hand_total(e) == hand_b, "hands changed with no steal")
    return fails


def test_move_robber_one_victim_auto_steal():
    seed = find_seed_with_first_roll_seven()
    if seed is None: return 0
    e = Env(); e.reset(seed); play_initial(e)
    pre = current_player(e.h)
    e.step(ROLL_DICE)
    if flag(e.h) != FLAG_MOVE_ROBBER: return 0

    # Find a hex with exactly 1 opponent owner who has >0 cards
    target = -1
    victim = -1
    for h in range(NUM_HEXES):
        if h == robber_hex(e.h): continue
        owners = hex_opponents_owners(e, h, pre)
        owners = {o for o in owners if p_hand(e.h, o) > 0}
        if len(owners) == 1:
            target = h; victim = next(iter(owners)); break
    if target < 0: return 0

    fails = 0
    victim_hand_b = p_hand(e.h, victim)
    self_hand_b   = p_hand(e.h, pre)
    e.step(MOVE_ROBBER_BASE + target)
    fails += fail(flag(e.h) == FLAG_NONE, f"flag={flag(e.h)} after auto-steal (want NONE)")
    fails += fail(p_hand(e.h, victim) == victim_hand_b - 1,
                  f"victim hand: {victim_hand_b}->{p_hand(e.h, victim)}")
    fails += fail(p_hand(e.h, pre) == self_hand_b + 1,
                  f"thief hand: {self_hand_b}->{p_hand(e.h, pre)}")
    return fails


def test_move_robber_multi_victim_then_steal():
    seed = find_seed_with_first_roll_seven()
    if seed is None: return 0
    e = Env(); e.reset(seed); play_initial(e)
    pre = current_player(e.h)
    # Make sure opponents have cards
    for p in range(NUM_PLAYERS):
        if p != pre: give_res(e.h, p, R_BRICK, 3)

    e.step(ROLL_DICE)
    if flag(e.h) != FLAG_MOVE_ROBBER: return 0

    target = -1; victims = []
    for h in range(NUM_HEXES):
        if h == robber_hex(e.h): continue
        owners = sorted(hex_opponents_owners(e, h, pre))
        if len(owners) >= 2:
            target = h; victims = owners; break
    if target < 0: return 0  # rare board

    fails = 0
    e.step(MOVE_ROBBER_BASE + target)
    fails += fail(flag(e.h) == FLAG_STEAL, f"flag={flag(e.h)} after multi (want STEAL)")
    fails += fail(robber_hex(e.h) == target, "robber not moved")
    # Now steal from first victim
    victim = victims[0]
    victim_hand_b = p_hand(e.h, victim); self_hand_b = p_hand(e.h, pre)
    e.step(STEAL_BASE + victim)
    fails += fail(flag(e.h) == FLAG_NONE, f"flag={flag(e.h)} after steal (want NONE)")
    fails += fail(p_hand(e.h, victim) == victim_hand_b - 1, "victim hand wrong post-steal")
    fails += fail(p_hand(e.h, pre) == self_hand_b + 1, "thief hand wrong post-steal")
    return fails


def test_steal_invalid_target():
    """Steal action targeting non-victim should be a no-op."""
    seed = find_seed_with_first_roll_seven()
    if seed is None: return 0
    e = Env(); e.reset(seed); play_initial(e)
    pre = current_player(e.h)
    for p in range(NUM_PLAYERS):
        if p != pre: give_res(e.h, p, R_BRICK, 3)
    e.step(ROLL_DICE)
    if flag(e.h) != FLAG_MOVE_ROBBER: return 0

    target = -1; victims = []
    for h in range(NUM_HEXES):
        if h == robber_hex(e.h): continue
        owners = sorted(hex_opponents_owners(e, h, pre))
        if len(owners) >= 2:
            target = h; victims = owners; break
    if target < 0: return 0

    e.step(MOVE_ROBBER_BASE + target)
    if flag(e.h) != FLAG_STEAL: return 0

    # Pick a non-victim opponent
    non_victim = -1
    for p in range(NUM_PLAYERS):
        if p != pre and p not in victims:
            non_victim = p; break
    if non_victim < 0: return 0

    fails = 0
    hand_b = p_hand(e.h, pre)
    e.step(STEAL_BASE + non_victim)
    fails += fail(flag(e.h) == FLAG_STEAL, "flag cleared on invalid steal target")
    fails += fail(p_hand(e.h, pre) == hand_b, "thief hand changed on invalid steal")

    # Also try stealing from self
    e.step(STEAL_BASE + pre)
    fails += fail(flag(e.h) == FLAG_STEAL, "flag cleared on self-steal")
    fails += fail(p_hand(e.h, pre) == hand_b, "thief hand changed on self-steal")
    return fails


def test_resource_conservation_through_seven():
    """Total = 95 must hold across discards + steal."""
    seed = find_seed_with_first_roll_seven()
    if seed is None: return 0
    e = Env(); e.reset(seed); play_initial(e)
    pre = current_player(e.h)
    # Push another player to discard
    other = (pre + 1) & 3
    give_res(e.h, other, R_BRICK, 8 - p_hand(e.h, other) + 1)
    for p in range(NUM_PLAYERS):
        if p != pre: give_res(e.h, p, R_GRAIN, 1)

    fails = 0
    fails += fail(bank_total(e) + hand_total(e) == 95, "pre-roll total != 95")

    e.step(ROLL_DICE)
    fails += fail(bank_total(e) + hand_total(e) == 95, "post-roll total != 95")

    # Walk through all sub-phases, doing whatever's legal
    safety = 0
    while flag(e.h) != FLAG_NONE and safety < 50:
        safety += 1
        f = flag(e.h)
        if f == FLAG_DISCARD:
            d = current_player(e.h)
            for r in range(5):
                if p_res(e.h, d, r) > 0:
                    e.step(DISCARD_BASE + r); break
        elif f == FLAG_MOVE_ROBBER:
            for h in range(NUM_HEXES):
                if h != robber_hex(e.h):
                    e.step(MOVE_ROBBER_BASE + h); break
        elif f == FLAG_STEAL:
            for p in range(NUM_PLAYERS):
                if p != pre and p_hand(e.h, p) > 0:
                    e.step(STEAL_BASE + p); break
            else:
                break
        fails += fail(bank_total(e) + hand_total(e) == 95,
                      f"sub-phase {f}: total={bank_total(e) + hand_total(e)}")
    return fails


def main():
    total = 0
    print("== test_seven_no_discard =="); total += test_seven_no_discard()
    print("== test_seven_with_discards =="); total += test_seven_with_discards()
    print("== test_discard_one_card =="); total += test_discard_one_card()
    print("== test_discard_no_resource_nop =="); total += test_discard_no_resource_nop()
    print("== test_full_discard_then_move_robber =="); total += test_full_discard_then_move_robber()
    print("== test_move_robber_invalid_same_hex =="); total += test_move_robber_invalid_same_hex()
    print("== test_move_robber_no_victims =="); total += test_move_robber_no_victims()
    print("== test_move_robber_one_victim_auto_steal =="); total += test_move_robber_one_victim_auto_steal()
    print("== test_move_robber_multi_victim_then_steal =="); total += test_move_robber_multi_victim_then_steal()
    print("== test_steal_invalid_target =="); total += test_steal_invalid_target()
    print("== test_resource_conservation_through_seven =="); total += test_resource_conservation_through_seven()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} assertions failed")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
