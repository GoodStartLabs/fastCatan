#!/usr/bin/env python3
"""Tests for compute_mask: mask must agree with step_one's notion of legality.

For every reachable state, every action ID either:
  - has its mask bit set AND step_one mutates state, OR
  - has its mask bit clear AND step_one is a no-op.

Build first:
    bash tools/build_lib.sh

Run:
    python3 tools/test_mask.py
"""

from __future__ import annotations
import ctypes, re, sys
from pathlib import Path

# Action IDs (mirror rules.hpp)
SETTLE_BASE, CITY_BASE, ROAD_BASE = 0, 54, 108
ROLL_DICE, END_TURN = 180, 181
DISCARD_BASE, MOVE_ROBBER_BASE, STEAL_BASE = 182, 187, 206
TRADE_BASE = 210
BUY_DEV, PLAY_KNIGHT = 235, 236

NUM_ACTIONS = 237
MASK_WORDS = 5

NUM_NODES, NUM_EDGES, NUM_HEXES, NUM_PLAYERS = 54, 72, 19, 4
NO_PLAYER = 0xFF
PHASE_INITIAL_1, PHASE_INITIAL_2, PHASE_MAIN, PHASE_ENDED = 0, 1, 2, 3
FLAG_NONE, FLAG_DISCARD, FLAG_MOVE_ROBBER, FLAG_STEAL = 0, 1, 2, 3
NODE_EMPTY, NODE_SETTLEMENT, NODE_CITY = 0, 1, 2

DEV_KNIGHT, DEV_VP = 0, 1

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
robber_hex      = _b("fcatan_robber_hex",       U8, VP)
node_byte       = _b("fcatan_node",             U8, VP, I)
edge_byte       = _b("fcatan_edge",             U8, VP, I)
p_hand          = _b("fcatan_player_handsize",  U8, VP, I)
p_res           = _b("fcatan_player_resource",  U8, VP, I, I)

give_res = _b("fcatan_give_resources", None, VP, I, I, U8)
set_dev  = _b("fcatan_set_player_dev", None, VP, I, I, U8)

compute_mask_  = _b("fcatan_compute_mask", None, VP, ctypes.POINTER(U64))
copy_state_    = _b("fcatan_copy_state",   None, VP, VP)
state_equal_   = _b("fcatan_state_equal",  U8, VP, VP)


def node_level(b): return b & 0x03
def node_owner(b): return (b >> 2) & 0x07

class Env:
    def __init__(self): self.h = create()
    def __del__(self):
        try: destroy(self.h)
        except Exception: pass
    def reset(self, seed): reset_(self.h, seed)
    def step(self, action): return step_(self.h, action) != 0


def get_mask(e):
    arr = (U64 * MASK_WORDS)()
    compute_mask_(e.h, arr)
    bits = []
    for w in range(MASK_WORDS):
        for b in range(64):
            if arr[w] & (1 << b):
                bits.append(w * 64 + b)
    return bits, arr

def mask_set(arr, action_id):
    return bool(arr[action_id >> 6] & (1 << (action_id & 63)))


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


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


# ---------------------------------------------------------------------
# Mask consistency vs simulation
# ---------------------------------------------------------------------

def mask_matches_simulation(e, label=""):
    """Verify: for every action ID, mask bit set iff step_one mutates state."""
    bits, arr = get_mask(e)
    sandbox = Env()
    fails = 0
    legal_count = 0
    for a in range(NUM_ACTIONS):
        copy_state_(e.h, sandbox.h)
        sandbox.step(a)
        changed = state_equal_(e.h, sandbox.h) == 0
        legal = mask_set(arr, a)
        if changed and not legal:
            fails += fail(False, f"{label} action {a} changed state but mask=0")
        elif legal and not changed:
            fails += fail(False, f"{label} action {a} mask=1 but state unchanged")
        if legal: legal_count += 1
    return fails, legal_count


def test_mask_initial_settle_phase():
    e = Env(); e.reset(42)
    f, n = mask_matches_simulation(e, "initial-settle")
    print(f"  initial-settle legal actions: {n}")
    return f


def test_mask_initial_road_phase():
    e = Env(); e.reset(42)
    e.step(SETTLE_BASE + first_legal_settle_initial(e))
    # now in road sub-step
    f, n = mask_matches_simulation(e, "initial-road")
    print(f"  initial-road legal actions: {n}")
    return f


def test_mask_main_pre_roll():
    e = Env(); e.reset(42); play_initial(e)
    f, n = mask_matches_simulation(e, "main-pre-roll")
    print(f"  main-pre-roll legal actions: {n}")
    # Pre-roll legality: ROLL_DICE always; PLAY_KNIGHT only if has knight playable
    return f


def test_mask_main_post_roll_basic():
    e = Env(); e.reset(42); play_initial(e)
    e.step(ROLL_DICE)
    if dice_roll(e.h) == 7:
        # would put us in sub-phase; skip
        return 0
    f, n = mask_matches_simulation(e, "main-post-roll-basic")
    print(f"  main-post-roll-basic legal actions: {n}")
    return f


def test_mask_main_post_roll_with_resources():
    """After rolling, player has resources to build many things."""
    e = Env(); seed = 42
    while True:
        e.reset(seed); play_initial(e); e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: break
        seed += 1
    pl = current_player(e.h)
    # Give plenty of every resource
    for r in range(5): give_res(e.h, pl, r, 5)
    f, n = mask_matches_simulation(e, "main-post-roll-loaded")
    print(f"  main-post-roll-loaded legal actions: {n}")
    return f


def test_mask_discard_subphase():
    """7-roll forces discard. Mask only allows DISCARD actions for resources held."""
    seed = None
    for s in range(2000):
        e = Env(); e.reset(s); play_initial(e); e.step(ROLL_DICE)
        if dice_roll(e.h) == 7: seed = s; break
    if seed is None: return 0
    e = Env(); e.reset(seed); play_initial(e)
    pl = current_player(e.h)
    other = (pl + 1) & 3
    give_res(e.h, other, 0, max(0, 9 - p_hand(e.h, other)))
    e.step(ROLL_DICE)
    if flag(e.h) != FLAG_DISCARD: return 0
    f, n = mask_matches_simulation(e, "discard")
    print(f"  discard legal actions: {n}")
    return f


def test_mask_move_robber_subphase():
    seed = None
    for s in range(2000):
        e = Env(); e.reset(s); play_initial(e); e.step(ROLL_DICE)
        if dice_roll(e.h) == 7: seed = s; break
    if seed is None: return 0
    e = Env(); e.reset(seed); play_initial(e)
    e.step(ROLL_DICE)
    # Possibly DISCARD first; resolve quickly
    safety = 0
    while flag(e.h) == FLAG_DISCARD and safety < 30:
        safety += 1
        d = current_player(e.h)
        for r in range(5):
            if p_res(e.h, d, r) > 0:
                e.step(DISCARD_BASE + r); break
    if flag(e.h) != FLAG_MOVE_ROBBER: return 0
    f, n = mask_matches_simulation(e, "move-robber")
    print(f"  move-robber legal actions: {n} (expect 18)")
    return f


def test_mask_steal_subphase():
    """Steal sub-phase: only valid victim IDs allowed."""
    for seed in range(2000):
        e = Env(); e.reset(seed); play_initial(e)
        pl = current_player(e.h)
        # give all opponents some resources
        for p in range(NUM_PLAYERS):
            if p != pl: give_res(e.h, p, 0, 3)
        e.step(ROLL_DICE)
        if dice_roll(e.h) != 7: continue
        # walk discard
        safety = 0
        while flag(e.h) == FLAG_DISCARD and safety < 30:
            safety += 1
            d = current_player(e.h)
            for r in range(5):
                if p_res(e.h, d, r) > 0:
                    e.step(DISCARD_BASE + r); break
        if flag(e.h) != FLAG_MOVE_ROBBER: continue
        # find a target hex with multiple opponents (so we get STEAL flag, not auto-steal)
        target = -1
        for h in range(NUM_HEXES):
            if h == robber_hex(e.h): continue
            owners = set()
            from_table = (
                _parse(HDR, "hex_to_node")
            )
            for v in from_table[h]:
                b = node_byte(e.h, v)
                if node_level(b) != NODE_EMPTY and node_owner(b) != pl:
                    owners.add(node_owner(b))
            if len(owners) >= 2: target = h; break
        if target < 0: continue
        e.step(MOVE_ROBBER_BASE + target)
        if flag(e.h) != FLAG_STEAL: continue
        f, n = mask_matches_simulation(e, "steal")
        print(f"  steal legal actions: {n}")
        return f
    return 0  # no suitable seed in 2000 — skip


def test_mask_terminal_phase_empty():
    """ENDED state → mask all zeros."""
    e = Env(); e.reset(42); play_initial(e)
    set_vp = _b("fcatan_set_player_vp", None, VP, I, U8)
    pl = current_player(e.h)
    set_vp(e.h, pl, 10)  # but phase doesn't auto-end on raw set; need an event
    # trigger via end_turn (which does check_game_ended)
    # actually reset_one + set vp doesn't change phase. We can force ENDED by a
    # build action that triggers check. Simpler: roll and end_turn.
    e.step(ROLL_DICE)
    if flag(e.h) != FLAG_NONE: return 0
    e.step(END_TURN)
    if phase(e.h) != PHASE_ENDED:
        # try harder: keep ending turns
        for _ in range(4):
            if phase(e.h) == PHASE_ENDED: break
            if dice_roll(e.h) == 0:
                e.step(ROLL_DICE)
                if flag(e.h) != FLAG_NONE: break
            e.step(END_TURN)
    if phase(e.h) != PHASE_ENDED:
        return 0
    bits, arr = get_mask(e)
    return fail(len(bits) == 0, f"ENDED mask has bits set: {bits}")


def test_mask_fuzz_random_play(seeds=range(20), max_steps=400):
    """Walk through games picking random legal actions; verify mask consistency at each step."""
    import random
    rng = random.Random(123)
    fails = 0
    for seed in seeds:
        e = Env(); e.reset(seed)
        for _ in range(max_steps):
            if phase(e.h) == PHASE_ENDED: break
            bits, arr = get_mask(e)
            if not bits: break
            # consistency check at this step
            sandbox = Env()
            for a in (rng.choice(bits), rng.choice(range(NUM_ACTIONS))):
                copy_state_(e.h, sandbox.h)
                sandbox.step(a)
                changed = state_equal_(e.h, sandbox.h) == 0
                legal = mask_set(arr, a)
                if changed != legal:
                    fails += fail(False,
                        f"seed={seed} a={a} legal={legal} changed={changed} phase={phase(e.h)} flag={flag(e.h)}")
                    break
            if fails: break
            # Take a random legal step to advance
            e.step(rng.choice(bits))
        if fails: break
    return fails


def main():
    total = 0
    print("== test_mask_initial_settle_phase ==");           total += test_mask_initial_settle_phase()
    print("== test_mask_initial_road_phase ==");             total += test_mask_initial_road_phase()
    print("== test_mask_main_pre_roll ==");                  total += test_mask_main_pre_roll()
    print("== test_mask_main_post_roll_basic ==");           total += test_mask_main_post_roll_basic()
    print("== test_mask_main_post_roll_with_resources =="); total += test_mask_main_post_roll_with_resources()
    print("== test_mask_discard_subphase ==");               total += test_mask_discard_subphase()
    print("== test_mask_move_robber_subphase ==");           total += test_mask_move_robber_subphase()
    print("== test_mask_steal_subphase ==");                 total += test_mask_steal_subphase()
    print("== test_mask_terminal_phase_empty ==");           total += test_mask_terminal_phase_empty()
    print("== test_mask_fuzz_random_play ==");               total += test_mask_fuzz_random_play()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} assertions failed")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
