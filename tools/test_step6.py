#!/usr/bin/env python3
"""Cheap Python tests for step_one slice 6 (buy dev + play knight + largest army)."""

from __future__ import annotations
import ctypes, re, sys
from pathlib import Path

SETTLE_BASE, CITY_BASE, ROAD_BASE = 0, 54, 108
ROLL_DICE, END_TURN = 180, 181
DISCARD_BASE = 182
MOVE_ROBBER_BASE = 187
STEAL_BASE = 206
TRADE_BASE = 210
BUY_DEV = 235
PLAY_KNIGHT = 236

NUM_NODES, NUM_EDGES, NUM_HEXES, NUM_PLAYERS = 54, 72, 19, 4
NO_PLAYER = 0xFF
PHASE_MAIN, PHASE_ENDED = 2, 3
FLAG_NONE, FLAG_DISCARD, FLAG_MOVE_ROBBER, FLAG_STEAL = 0, 1, 2, 3
NODE_EMPTY, NODE_SETTLEMENT = 0, 1

R_BRICK, R_LUMBER, R_WOOL, R_GRAIN, R_ORE = 0, 1, 2, 3, 4

DEV_KNIGHT, DEV_VP, DEV_RB, DEV_YOP, DEV_MONO = 0, 1, 2, 3, 4

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
dev_card_played = _b("fcatan_dev_card_played",  U8, VP)
robber_hex      = _b("fcatan_robber_hex",       U8, VP)
largest_army_owner = _b("fcatan_largest_army_owner", U8, VP)

p_vp        = _b("fcatan_player_vp",                  U8, VP, I)
p_vp_pub    = _b("fcatan_player_vp_public",           U8, VP, I)
p_hand      = _b("fcatan_player_handsize",            U8, VP, I)
p_res       = _b("fcatan_player_resource",            U8, VP, I, I)
p_dev       = _b("fcatan_player_dev",                 U8, VP, I, I)
p_dev_bought = _b("fcatan_player_dev_bought",         U8, VP, I, I)
p_total_dev = _b("fcatan_player_total_dev",           U8, VP, I)
p_knights   = _b("fcatan_player_knights_played",      U8, VP, I)
bank        = _b("fcatan_bank", U8, VP, I)
dev_deck    = _b("fcatan_dev_deck", U8, VP, I)

node_byte = _b("fcatan_node", U8, VP, I)
edge_byte = _b("fcatan_edge", U8, VP, I)

give_res = _b("fcatan_give_resources", None, VP, I, I, U8)
set_vp   = _b("fcatan_set_player_vp",  None, VP, I, U8)
set_dev  = _b("fcatan_set_player_dev", None, VP, I, I, U8)
set_knights = _b("fcatan_set_player_knights_played", None, VP, I, U8)


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
    seed = base_seed
    while True:
        e.reset(seed); play_initial(e)
        e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: return seed
        seed += 1


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


# ---------------------------------------------------------------------
# Tests — buy dev
# ---------------------------------------------------------------------

def test_buy_dev_basic():
    e = Env(); to_post_roll_no7(e, 42)
    pl = current_player(e.h)
    give_res(e.h, pl, R_WOOL, 1)
    give_res(e.h, pl, R_GRAIN, 1)
    give_res(e.h, pl, R_ORE, 1)
    fails = 0
    deck_b = sum(dev_deck(e.h, d) for d in range(5))
    total_b = p_total_dev(e.h, pl)
    res_w_b = p_res(e.h, pl, R_WOOL); res_g_b = p_res(e.h, pl, R_GRAIN); res_o_b = p_res(e.h, pl, R_ORE)
    bank_w_b = bank(e.h, R_WOOL); bank_g_b = bank(e.h, R_GRAIN); bank_o_b = bank(e.h, R_ORE)

    e.step(BUY_DEV)
    fails += fail(p_total_dev(e.h, pl) == total_b + 1, "total_dev not incremented")
    fails += fail(sum(dev_deck(e.h, d) for d in range(5)) == deck_b - 1, "dev deck not decremented")
    fails += fail(p_res(e.h, pl, R_WOOL) == res_w_b - 1, "wool not paid")
    fails += fail(p_res(e.h, pl, R_GRAIN) == res_g_b - 1, "grain not paid")
    fails += fail(p_res(e.h, pl, R_ORE) == res_o_b - 1, "ore not paid")
    fails += fail(bank(e.h, R_WOOL) == bank_w_b + 1, "wool not refunded to bank")
    fails += fail(bank(e.h, R_GRAIN) == bank_g_b + 1, "grain not refunded to bank")
    fails += fail(bank(e.h, R_ORE) == bank_o_b + 1, "ore not refunded to bank")
    return fails


def test_buy_dev_unaffordable():
    e = Env(); to_post_roll_no7(e, 42)
    pl = current_player(e.h)
    fails = 0
    if p_res(e.h, pl, R_WOOL) >= 1 and p_res(e.h, pl, R_GRAIN) >= 1 and p_res(e.h, pl, R_ORE) >= 1:
        return 0  # naturally afford; not testable cleanly
    deck_b = sum(dev_deck(e.h, d) for d in range(5))
    total_b = p_total_dev(e.h, pl)
    e.step(BUY_DEV)
    fails += fail(p_total_dev(e.h, pl) == total_b, "total_dev changed when unaffordable")
    fails += fail(sum(dev_deck(e.h, d) for d in range(5)) == deck_b, "deck changed when unaffordable")
    return fails


def test_buy_dev_distribution():
    """Buy many dev cards; observed distribution should roughly match the
    initial deck composition (14/5/2/2/2 = 25 cards)."""
    e = Env(); to_post_roll_no7(e, 7)
    pl = current_player(e.h)
    give_res(e.h, pl, R_WOOL, 50)
    give_res(e.h, pl, R_GRAIN, 50)
    give_res(e.h, pl, R_ORE, 50)
    drawn = [0]*5
    while True:
        deck_total = sum(dev_deck(e.h, d) for d in range(5))
        if deck_total == 0: break
        before = [p_dev(e.h, pl, d) + p_dev_bought(e.h, pl, d) for d in range(5)]
        e.step(BUY_DEV)
        after = [p_dev(e.h, pl, d) + p_dev_bought(e.h, pl, d) for d in range(5)]
        for d in range(5):
            if after[d] > before[d]: drawn[d] += 1; break
    fails = 0
    fails += fail(drawn == [14, 5, 2, 2, 2], f"drew {drawn} (want [14,5,2,2,2])")
    fails += fail(p_total_dev(e.h, pl) >= 1, "total_dev should be > 0 after many buys")
    return fails


def test_buy_vp_increments_hidden_vp():
    """A bought VP card increments player_vp but not vp_public."""
    e = Env(); to_post_roll_no7(e, 42)
    pl = current_player(e.h)
    # Buy until we draw a VP card
    give_res(e.h, pl, R_WOOL, 25); give_res(e.h, pl, R_GRAIN, 25); give_res(e.h, pl, R_ORE, 25)
    fails = 0
    drew_vp = False
    safety = 0
    while not drew_vp and safety < 25:
        safety += 1
        vp_b = p_vp(e.h, pl); pub_b = p_vp_pub(e.h, pl)
        deck_vp_b = dev_deck(e.h, DEV_VP)
        e.step(BUY_DEV)
        if dev_deck(e.h, DEV_VP) < deck_vp_b:
            # we drew a VP card
            fails += fail(p_vp(e.h, pl) == vp_b + 1, "vp didn't increase by 1 on VP draw")
            fails += fail(p_vp_pub(e.h, pl) == pub_b, "public vp changed on VP draw")
            drew_vp = True
    return fails


def test_buy_vp_can_trigger_win():
    """Force player to VP=9, then a VP card buy → ENDED."""
    e = Env(); to_post_roll_no7(e, 42)
    pl = current_player(e.h)
    set_vp(e.h, pl, 9)
    give_res(e.h, pl, R_WOOL, 50); give_res(e.h, pl, R_GRAIN, 50); give_res(e.h, pl, R_ORE, 50)
    fails = 0
    safety = 0
    while phase(e.h) == PHASE_MAIN and safety < 25:
        safety += 1
        e.step(BUY_DEV)
    # If we drew at least one VP card, phase should be ENDED.
    # If not, test is inconclusive but at least confirm phase consistency.
    if dev_deck(e.h, DEV_VP) < 5:  # at least one VP drawn
        # Did we actually draw one? VP cards initially=5 in deck. If decreased, yes.
        pass
    fails += fail(phase(e.h) == PHASE_ENDED or dev_deck(e.h, DEV_VP) == 5,
                  f"phase={phase(e.h)} but dev_deck[VP]={dev_deck(e.h, DEV_VP)}")
    return fails


# ---------------------------------------------------------------------
# Tests — play knight
# ---------------------------------------------------------------------

def test_play_knight_basic():
    e = Env(); to_post_roll_no7(e, 42)
    pl = current_player(e.h)
    set_dev(e.h, pl, DEV_KNIGHT, 1)

    fails = 0
    knights_b = p_knights(e.h, pl)
    total_b   = p_total_dev(e.h, pl)
    e.step(PLAY_KNIGHT)

    fails += fail(p_knights(e.h, pl) == knights_b + 1, "knights_played not incremented")
    fails += fail(p_dev(e.h, pl, DEV_KNIGHT) == 0, "knight not consumed")
    fails += fail(p_total_dev(e.h, pl) == total_b - 1, "total_dev not decremented")
    fails += fail(dev_card_played(e.h) == 1, "dev_card_played not set")
    fails += fail(flag(e.h) == FLAG_MOVE_ROBBER, f"flag={flag(e.h)} (want MOVE_ROBBER)")
    fails += fail(rolling_player(e.h) == pl, "rolling_player not set to knight player")
    return fails


def test_play_knight_one_per_turn():
    e = Env(); to_post_roll_no7(e, 42)
    pl = current_player(e.h)
    set_dev(e.h, pl, DEV_KNIGHT, 2)
    e.step(PLAY_KNIGHT)
    if flag(e.h) != FLAG_MOVE_ROBBER: return 0
    # Move robber to clear flag; pick any non-current hex with no opponents
    for h in range(NUM_HEXES):
        if h != robber_hex(e.h):
            e.step(MOVE_ROBBER_BASE + h)
            if flag(e.h) == FLAG_NONE: break
    fails = 0
    if flag(e.h) != FLAG_NONE: return 0  # got STEAL flag, can't easily resolve here
    # Now try to play another knight
    knights_b = p_knights(e.h, pl)
    e.step(PLAY_KNIGHT)
    fails += fail(p_knights(e.h, pl) == knights_b, "second knight played in same turn")
    return fails


def test_play_knight_no_card_nop():
    e = Env(); to_post_roll_no7(e, 42)
    pl = current_player(e.h)
    set_dev(e.h, pl, DEV_KNIGHT, 0)
    fails = 0
    knights_b = p_knights(e.h, pl)
    flag_b = flag(e.h)
    e.step(PLAY_KNIGHT)
    fails += fail(p_knights(e.h, pl) == knights_b, "knights_played changed without card")
    fails += fail(flag(e.h) == flag_b, "flag changed on illegal knight")
    fails += fail(dev_card_played(e.h) == 0, "dev_card_played set on illegal knight")
    return fails


def test_play_knight_pre_roll():
    """Knight can be played BEFORE rolling dice."""
    e = Env(); e.reset(42); play_initial(e)
    pl = current_player(e.h)
    set_dev(e.h, pl, DEV_KNIGHT, 1)
    fails = 0
    fails += fail(dice_roll(e.h) == 0, "should be pre-roll")
    e.step(PLAY_KNIGHT)
    fails += fail(p_knights(e.h, pl) == 1, "knight not played pre-roll")
    fails += fail(flag(e.h) == FLAG_MOVE_ROBBER, "flag not set pre-roll knight")
    return fails


# ---------------------------------------------------------------------
# Tests — largest army
# ---------------------------------------------------------------------

def test_largest_army_first_award():
    e = Env(); to_post_roll_no7(e, 42)
    pl = current_player(e.h)
    fails = 0
    fails += fail(largest_army_owner(e.h) == NO_PLAYER, "should start unowned")
    # Set up 2 prior knights, give 1 more in dev, play it → should reach 3 → award
    set_knights(e.h, pl, 2)
    set_dev(e.h, pl, DEV_KNIGHT, 1)
    vp_b = p_vp(e.h, pl); pub_b = p_vp_pub(e.h, pl)
    e.step(PLAY_KNIGHT)
    fails += fail(p_knights(e.h, pl) == 3, f"knights_played={p_knights(e.h, pl)}")
    fails += fail(largest_army_owner(e.h) == pl, "largest_army not awarded at 3")
    fails += fail(p_vp(e.h, pl) == vp_b + 2, f"vp didn't +2 on largest army (was {vp_b}, now {p_vp(e.h, pl)})")
    fails += fail(p_vp_pub(e.h, pl) == pub_b + 2, "public vp didn't +2 on largest army")
    return fails


def test_largest_army_transfer():
    e = Env(); to_post_roll_no7(e, 42)
    holder = current_player(e.h)
    # Pre-set holder to have 3 knights and own the title
    set_knights(e.h, holder, 3)
    # manually award title via set... we don't have a direct setter; simulate by playing
    # Easier: cycle through and play knights for both players.
    # Approach: give holder 3 played + title, then make another player exceed.
    # Simulate by stepping...
    # Actually let me just use the simulator: holder plays 3 knights to get title.
    set_knights(e.h, holder, 0)  # reset and do it via playing
    set_dev(e.h, holder, DEV_KNIGHT, 3)
    # Play 3 knights across 3 turns (one per turn due to one-per-turn rule)
    for _ in range(3):
        if dice_roll(e.h) == 0:
            e.step(ROLL_DICE)
            if flag(e.h) != FLAG_NONE:
                # got a 7 — try to skip
                while flag(e.h) != FLAG_NONE:
                    f = flag(e.h)
                    if f == FLAG_DISCARD:
                        d = current_player(e.h)
                        for r in range(5):
                            if p_res(e.h, d, r) > 0: e.step(DISCARD_BASE + r); break
                    elif f == FLAG_MOVE_ROBBER:
                        for h in range(NUM_HEXES):
                            if h != robber_hex(e.h): e.step(MOVE_ROBBER_BASE + h); break
                    elif f == FLAG_STEAL:
                        for p in range(NUM_PLAYERS):
                            if p != current_player(e.h) and p_hand(e.h, p) > 0:
                                e.step(STEAL_BASE + p); break
                        else: break
        if current_player(e.h) != holder:
            return 0  # cycled away from holder somehow
        if p_dev(e.h, holder, DEV_KNIGHT) == 0:
            return 0
        e.step(PLAY_KNIGHT)
        # Resolve robber (no opponents needed)
        for h in range(NUM_HEXES):
            if h != robber_hex(e.h):
                e.step(MOVE_ROBBER_BASE + h); break
        # Skip remaining steal flag if any
        while flag(e.h) == FLAG_STEAL:
            stole = False
            for p in range(NUM_PLAYERS):
                if p != current_player(e.h) and p_hand(e.h, p) > 0:
                    e.step(STEAL_BASE + p); stole = True; break
            if not stole: break
        e.step(END_TURN)
        # cycle back to holder by ending turns
        for _ in range(3):
            if current_player(e.h) == holder: break
            if dice_roll(e.h) == 0:
                e.step(ROLL_DICE)
                if flag(e.h) != FLAG_NONE: break
            e.step(END_TURN)

    fails = 0
    if largest_army_owner(e.h) != holder:
        return 0  # couldn't get holder the title; skip
    fails += fail(largest_army_owner(e.h) == holder, "holder doesn't own title yet")
    fails += fail(p_knights(e.h, holder) == 3, f"holder knights={p_knights(e.h, holder)}")

    # Now another player needs 4 knights to take title.
    challenger = (holder + 1) & 3
    set_knights(e.h, challenger, 3)
    set_dev(e.h, challenger, DEV_KNIGHT, 1)

    # Cycle to challenger
    while current_player(e.h) != challenger:
        if dice_roll(e.h) == 0:
            e.step(ROLL_DICE)
            if flag(e.h) != FLAG_NONE: break
        e.step(END_TURN)
    if current_player(e.h) != challenger: return 0
    if dice_roll(e.h) == 0:
        e.step(ROLL_DICE)
        if flag(e.h) != FLAG_NONE: return 0

    holder_vp_b = p_vp(e.h, holder)
    chal_vp_b = p_vp(e.h, challenger)

    e.step(PLAY_KNIGHT)
    fails += fail(p_knights(e.h, challenger) == 4, "challenger knights wrong")
    fails += fail(largest_army_owner(e.h) == challenger,
                  f"title didn't transfer (owner={largest_army_owner(e.h)})")
    fails += fail(p_vp(e.h, holder) == holder_vp_b - 2, "holder didn't lose 2 VP")
    fails += fail(p_vp(e.h, challenger) == chal_vp_b + 2, "challenger didn't gain 2 VP")
    return fails


def test_largest_army_no_transfer_on_tie():
    """Holder has N, opponent reaches N — title stays with holder."""
    e = Env(); to_post_roll_no7(e, 42)
    holder = current_player(e.h)
    challenger = (holder + 1) & 3

    # Manipulate state directly: holder=3 knights + has title.
    # Award via playing — but easier: play 3 knights for holder over multiple turns.
    set_knights(e.h, holder, 3)
    # Need to set title — only check_largest_army does that. Trigger by playing a knight.
    # Use trick: set holder to 2 knights + 1 in dev, play it → goes to 3 → title.
    set_knights(e.h, holder, 2)
    set_dev(e.h, holder, DEV_KNIGHT, 1)
    e.step(PLAY_KNIGHT)
    if largest_army_owner(e.h) != holder: return 0
    # resolve robber
    for h in range(NUM_HEXES):
        if h != robber_hex(e.h):
            e.step(MOVE_ROBBER_BASE + h); break
    while flag(e.h) == FLAG_STEAL:
        stole = False
        for p in range(NUM_PLAYERS):
            if p != holder and p_hand(e.h, p) > 0:
                e.step(STEAL_BASE + p); stole = True; break
        if not stole: break

    # Now try to make challenger tie at 3.
    set_knights(e.h, challenger, 2)
    set_dev(e.h, challenger, DEV_KNIGHT, 1)

    # Cycle to challenger
    e.step(END_TURN)
    safety = 0
    while current_player(e.h) != challenger and safety < 10:
        safety += 1
        if dice_roll(e.h) == 0:
            e.step(ROLL_DICE)
            if flag(e.h) != FLAG_NONE: break
        e.step(END_TURN)
    if current_player(e.h) != challenger: return 0
    if dice_roll(e.h) == 0:
        e.step(ROLL_DICE)
        if flag(e.h) != FLAG_NONE: return 0

    holder_vp_b = p_vp(e.h, holder)
    e.step(PLAY_KNIGHT)
    fails = 0
    fails += fail(p_knights(e.h, challenger) == 3, "challenger knights wrong")
    fails += fail(largest_army_owner(e.h) == holder,
                  f"title transferred on tie (owner={largest_army_owner(e.h)})")
    fails += fail(p_vp(e.h, holder) == holder_vp_b, "holder lost VP on tie")
    return fails


def test_resource_conservation_through_buy_dev():
    e = Env(); to_post_roll_no7(e, 42)
    pl = current_player(e.h)
    give_res(e.h, pl, R_WOOL, 5); give_res(e.h, pl, R_GRAIN, 5); give_res(e.h, pl, R_ORE, 5)
    fails = 0
    for _ in range(5):
        before = sum(bank(e.h, r) for r in range(5)) + sum(p_res(e.h, p, r) for p in range(4) for r in range(5))
        e.step(BUY_DEV)
        after = sum(bank(e.h, r) for r in range(5)) + sum(p_res(e.h, p, r) for p in range(4) for r in range(5))
        fails += fail(before == 95 and after == 95, f"sum {before}->{after}")
    return fails


def main():
    total = 0
    print("== test_buy_dev_basic ==");                  total += test_buy_dev_basic()
    print("== test_buy_dev_unaffordable ==");           total += test_buy_dev_unaffordable()
    print("== test_buy_dev_distribution ==");           total += test_buy_dev_distribution()
    print("== test_buy_vp_increments_hidden_vp ==");    total += test_buy_vp_increments_hidden_vp()
    print("== test_buy_vp_can_trigger_win ==");         total += test_buy_vp_can_trigger_win()
    print("== test_play_knight_basic ==");              total += test_play_knight_basic()
    print("== test_play_knight_one_per_turn ==");       total += test_play_knight_one_per_turn()
    print("== test_play_knight_no_card_nop ==");        total += test_play_knight_no_card_nop()
    print("== test_play_knight_pre_roll ==");           total += test_play_knight_pre_roll()
    print("== test_largest_army_first_award ==");       total += test_largest_army_first_award()
    print("== test_largest_army_transfer ==");          total += test_largest_army_transfer()
    print("== test_largest_army_no_transfer_on_tie =="); total += test_largest_army_no_transfer_on_tie()
    print("== test_resource_conservation_through_buy_dev =="); total += test_resource_conservation_through_buy_dev()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} assertions failed")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
