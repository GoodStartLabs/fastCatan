#!/usr/bin/env python3
"""Cheap Python tests for step_one slice 2 (roll dice + production + end turn).

Build first:
    bash tools/build_lib.sh

Run:
    python3 tools/test_step2.py
"""

from __future__ import annotations
import ctypes
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Constants (mirror rules.hpp, state.hpp, topology.hpp)
# ---------------------------------------------------------------------
SETTLE_BASE = 0
CITY_BASE   = 54
ROAD_BASE   = 108
ROLL_DICE   = 180
END_TURN    = 181

NUM_NODES = 54
NUM_EDGES = 72
NUM_HEXES = 19

NO_PLAYER = 0xFF
NO_HEX    = 0xFF

PHASE_INITIAL_1 = 0
PHASE_INITIAL_2 = 1
PHASE_MAIN      = 2
PHASE_ENDED     = 3

FLAG_NONE              = 0
FLAG_DISCARD_RESOURCES = 1
FLAG_MOVE_ROBBER       = 2

NODE_EMPTY      = 0
NODE_SETTLEMENT = 1
NODE_CITY       = 2

DESERT_CODE = 5

# ---------------------------------------------------------------------
# Topology parse
# ---------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
HDR = (REPO / "include" / "topology.hpp").read_text()

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

NODE_NODE = _parse_table(HDR, "node_to_node")
NODE_EDGE = _parse_table(HDR, "node_to_edge")
EDGE_NODE = _parse_table(HDR, "edge_to_node")
HEX_NODE  = _parse_table(HDR, "hex_to_node")

# ---------------------------------------------------------------------
# Library load
# ---------------------------------------------------------------------
LIB_NAME = "libfastcatan.dylib" if sys.platform == "darwin" else "libfastcatan.so"
LIB_PATH = REPO / "build" / LIB_NAME
if not LIB_PATH.exists():
    print(f"missing {LIB_PATH}; run `bash tools/build_lib.sh`")
    sys.exit(1)
lib = ctypes.CDLL(str(LIB_PATH))

VP, U8, U16, U32, U64, I = (ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint16,
                             ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int)

def _bind(name, restype, *argtypes):
    f = getattr(lib, name); f.restype = restype; f.argtypes = list(argtypes); return f

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

node_byte    = _bind("fcatan_node",         U8, VP, I)
edge_byte    = _bind("fcatan_edge",         U8, VP, I)
hex_resource = _bind("fcatan_hex_resource", U8, VP, I)
hex_number   = _bind("fcatan_hex_number",   U8, VP, I)

p_vp        = _bind("fcatan_player_vp",            U8, VP, I)
p_hand      = _bind("fcatan_player_handsize",      U8, VP, I)
p_res       = _bind("fcatan_player_resource",      U8, VP, I, I)
bank        = _bind("fcatan_bank",                 U8, VP, I)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def node_level(b): return b & 0x03
def node_owner(b): return (b >> 2) & 0x07

class Env:
    def __init__(self):       self.h = create()
    def __del__(self):
        try: destroy(self.h)
        except Exception: pass
    def reset(self, seed):    reset_(self.h, seed)
    def step(self, action):   return step_(self.h, action) != 0


def first_legal_settle(e):
    for n in range(NUM_NODES):
        if node_level(node_byte(e.h, n)) != NODE_EMPTY: continue
        if any(node_level(node_byte(e.h, nb)) != NODE_EMPTY for nb in NODE_NODE[n]):
            continue
        return n
    return -1

def first_legal_road(e):
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
        e.step(SETTLE_BASE + first_legal_settle(e))
        e.step(ROAD_BASE   + first_legal_road(e))


def total_resources_owed(e, roll):
    """Compute expected total resources distributed for a given roll
    (ignoring bank shortage). Sums settlements (1) + cities (2) on
    matching non-robber hexes, by resource."""
    owed = [0]*5  # per resource
    for h in range(NUM_HEXES):
        if hex_number(e.h, h) != roll: continue
        if h == robber_hex(e.h): continue
        res = hex_resource(e.h, h)
        if res == DESERT_CODE: continue
        for v in HEX_NODE[h]:
            b = node_byte(e.h, v)
            lvl = node_level(b)
            if lvl == NODE_EMPTY: continue
            owed[res] += 1 if lvl == NODE_SETTLEMENT else 2
    return owed


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_roll_basic(seed=42):
    e = Env(); e.reset(seed); play_initial(e)
    fails = 0
    fails += fail(dice_roll(e.h) == 0, "dice_roll should be 0 pre-roll")
    e.step(ROLL_DICE)
    r = dice_roll(e.h)
    fails += fail(2 <= r <= 12, f"dice out of range: {r}")
    fails += fail(r != 7 or flag(e.h) != FLAG_NONE,
                  "7 rolled but no robber/discard flag set")
    if r != 7:
        fails += fail(flag(e.h) == FLAG_NONE,
                      f"non-7 roll set flag={flag(e.h)}")
    return fails


def test_dice_distribution(n_rolls=20000):
    """Roll many times across many turns/seeds; histogram should look like 2d6."""
    e = Env(); e.reset(0); play_initial(e)
    hist = [0]*13
    for i in range(n_rolls):
        # Roll, then end turn (or skip end if flag set — just reset state)
        if dice_roll(e.h) == 0 and flag(e.h) == FLAG_NONE:
            e.step(ROLL_DICE)
        r = dice_roll(e.h)
        if 2 <= r <= 12: hist[r] += 1
        # If flag set, can't end turn — reset to fresh state to keep rolling
        if flag(e.h) != FLAG_NONE or dice_roll(e.h) != 0:
            # New game with new seed, replay initial placement
            e.reset(i + 1)
            play_initial(e)
        elif phase(e.h) != PHASE_MAIN:
            e.reset(i + 1); play_initial(e)
    # 2d6 expected: P(7) = 6/36, P(2)=P(12) = 1/36
    # Rough bounds: 7 most common, 2/12 rarest by ~6x
    fails = 0
    fails += fail(hist[7] > hist[2] * 3, f"hist[7]={hist[7]} not >> hist[2]={hist[2]}")
    fails += fail(hist[7] > hist[12] * 3, f"hist[7]={hist[7]} not >> hist[12]={hist[12]}")
    print(f"  dist: {hist[2:13]}")
    return fails


def test_production_non7(seeds=range(50)):
    fails = 0
    for seed in seeds:
        e = Env(); e.reset(seed); play_initial(e)

        # snapshot pre-roll
        hand_before = [p_hand(e.h, p) for p in range(4)]
        bank_before = [bank(e.h, r) for r in range(5)]
        res_before  = [[p_res(e.h, p, r) for r in range(5)] for p in range(4)]

        e.step(ROLL_DICE)
        r = dice_roll(e.h)
        if r == 7: continue  # not testing 7 here

        owed = total_resources_owed(e, r)

        # Per-resource distributed = owed (assuming bank covers; in M1 + slice 2,
        # initial hand is small so bank shortage shouldn't occur on first roll).
        for res in range(5):
            distributed = sum(p_res(e.h, p, res) - res_before[p][res] for p in range(4))
            bank_drop = bank_before[res] - bank(e.h, res)
            fails += fail(distributed == bank_drop,
                          f"seed={seed} r={r} res={res}: distributed={distributed} bank_drop={bank_drop}")
            # Distributed should equal owed (bank had >= owed at this point)
            if owed[res] <= bank_before[res]:
                fails += fail(distributed == owed[res],
                              f"seed={seed} r={r} res={res}: dist={distributed} owed={owed[res]}")

        # handsize accounting
        for p in range(4):
            gained = sum(p_res(e.h, p, r2) - res_before[p][r2] for r2 in range(5))
            hand_delta = p_hand(e.h, p) - hand_before[p]
            fails += fail(gained == hand_delta,
                          f"seed={seed} p{p}: gained={gained} hand_delta={hand_delta}")

        # global conservation
        bsum = sum(bank(e.h, r) for r in range(5))
        hsum = sum(p_res(e.h, p, r) for p in range(4) for r in range(5))
        fails += fail(bsum + hsum == 95, f"seed={seed} sum={bsum + hsum}")
    return fails


def test_seven_sets_flag():
    """Find a seed where the first roll is 7. Verify flag set, no production."""
    for seed in range(1000):
        e = Env(); e.reset(seed); play_initial(e)
        bank_before = [bank(e.h, r) for r in range(5)]
        e.step(ROLL_DICE)
        if dice_roll(e.h) == 7:
            fails = 0
            fails += fail(flag(e.h) in (FLAG_DISCARD_RESOURCES, FLAG_MOVE_ROBBER),
                          f"7 roll didn't set flag (got {flag(e.h)})")
            for r in range(5):
                fails += fail(bank(e.h, r) == bank_before[r],
                              f"7 roll changed bank[{r}]: {bank_before[r]}->{bank(e.h, r)}")
            # end_turn should be a no-op while flag set
            cp_before = current_player(e.h)
            e.step(END_TURN)
            fails += fail(current_player(e.h) == cp_before,
                          "end_turn advanced while flag active")
            print(f"  found 7 at seed={seed} flag={flag(e.h)}")
            return fails
    print("  warning: no 7 in 1000 seeds — sample too small")
    return 0


def test_end_turn(seed=42):
    e = Env(); e.reset(seed); play_initial(e)
    fails = 0

    cp = current_player(e.h)
    tc = turn_count(e.h)

    # end_turn before roll: nop
    e.step(END_TURN)
    fails += fail(current_player(e.h) == cp, "end_turn advanced before roll")
    fails += fail(turn_count(e.h) == tc, "turn_count incremented before roll")

    # roll, then end_turn (if non-7)
    e.step(ROLL_DICE)
    if dice_roll(e.h) == 7:
        return fails  # skip; needs slice 4
    e.step(END_TURN)
    fails += fail(current_player(e.h) == (cp + 1) & 0x03, "end_turn didn't advance")
    fails += fail(turn_count(e.h) == tc + 1, "turn_count didn't increment")
    fails += fail(dice_roll(e.h) == 0, "dice_roll didn't reset")
    return fails


def test_double_roll_blocked(seed=42):
    e = Env(); e.reset(seed); play_initial(e)
    e.step(ROLL_DICE)
    r1 = dice_roll(e.h)
    bank_after_first = [bank(e.h, r) for r in range(5)]
    e.step(ROLL_DICE)
    fails = fail(dice_roll(e.h) == r1, "second roll changed dice_roll")
    for r in range(5):
        fails += fail(bank(e.h, r) == bank_after_first[r],
                      f"second roll changed bank[{r}]")
    return fails


def test_full_round(seed=42, n_rounds=10):
    """Play several full rounds; assert resource conservation always holds."""
    e = Env(); e.reset(seed); play_initial(e)
    fails = 0
    rounds_done = 0
    safety = 0
    while rounds_done < n_rounds and safety < 1000:
        safety += 1
        if phase(e.h) != PHASE_MAIN: break
        if flag(e.h) != FLAG_NONE: break  # 7 rolled, slice 4 territory

        if dice_roll(e.h) == 0:
            e.step(ROLL_DICE)
            if flag(e.h) != FLAG_NONE: break
        else:
            e.step(END_TURN)
            rounds_done += 1
            bsum = sum(bank(e.h, r) for r in range(5))
            hsum = sum(p_res(e.h, p, r) for p in range(4) for r in range(5))
            fails += fail(bsum + hsum == 95,
                          f"round {rounds_done}: sum={bsum + hsum}")
    fails += fail(rounds_done > 0, "no rounds completed")
    return fails


def test_determinism_main(seed=42):
    """Same seed → same dice sequence in MAIN."""
    rolls = []
    for _ in range(2):
        e = Env(); e.reset(seed); play_initial(e)
        seq = []
        for _ in range(20):
            if phase(e.h) != PHASE_MAIN: break
            if flag(e.h) != FLAG_NONE: break
            e.step(ROLL_DICE)
            seq.append(dice_roll(e.h))
            if flag(e.h) != FLAG_NONE: break
            e.step(END_TURN)
        rolls.append(tuple(seq))
    return fail(rolls[0] == rolls[1], f"dice sequence diverged: {rolls[0]} vs {rolls[1]}")


def main():
    total = 0
    print("== test_roll_basic =="); total += test_roll_basic()
    print("== test_dice_distribution =="); total += test_dice_distribution(20000)
    print("== test_production_non7 =="); total += test_production_non7()
    print("== test_seven_sets_flag =="); total += test_seven_sets_flag()
    print("== test_end_turn =="); total += test_end_turn()
    print("== test_double_roll_blocked =="); total += test_double_roll_blocked()
    print("== test_full_round =="); total += test_full_round()
    print("== test_determinism_main =="); total += test_determinism_main()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} assertions failed")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
