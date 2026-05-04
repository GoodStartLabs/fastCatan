#!/usr/bin/env python3
"""Hand-built corpus for longest road algorithm.

Each test sets node + edge state directly, then calls recompute_awards and
asserts the resulting per-player road length matches the known answer.

Build first:
    bash tools/build_lib.sh
"""

from __future__ import annotations
import ctypes, sys
from pathlib import Path

NUM_NODES, NUM_EDGES = 54, 72
NO_PLAYER = 0xFF

NODE_EMPTY, NODE_SETTLEMENT, NODE_CITY = 0, 1, 2

REPO = Path(__file__).resolve().parents[1]
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
phase   = _b("fcatan_phase",   U8, VP)
node_byte = _b("fcatan_node",  U8, VP, I)
edge_byte = _b("fcatan_edge",  U8, VP, I)
set_node = _b("fcatan_set_node", None, VP, I, U8, U8)
set_edge = _b("fcatan_set_edge", None, VP, I, U8)
recompute_awards = _b("fcatan_recompute_awards", None, VP)
road_length = _b("fcatan_player_road_length", U8, VP, I)
longest_owner = _b("fcatan_longest_road_owner", U8, VP)
p_vp = _b("fcatan_player_vp", U8, VP, I)
p_vp_pub = _b("fcatan_player_vp_public", U8, VP, I)


def clear_board(env):
    """Wipe all nodes/edges to empty. Doesn't touch players or bank."""
    for n in range(NUM_NODES):
        set_node(env, n, NODE_EMPTY, 0)
    for e in range(NUM_EDGES):
        set_edge(env, e, NO_PLAYER)


def set_road(env, edges, player):
    for e in edges:
        set_edge(env, e, player)


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_linear_road_5():
    """Top edge of board: edges 0..4 form a line v0-v1-v2-v3-v4-v5."""
    env = create(); reset_(env, 1); clear_board(env)
    set_road(env, [0, 1, 2, 3, 4], 0)
    recompute_awards(env)
    fails = fail(road_length(env, 0) == 5, f"length = {road_length(env, 0)} (want 5)")
    fails += fail(longest_owner(env) == 0, f"owner = {longest_owner(env)} (want 0)")
    destroy(env)
    return fails


def test_cycle_around_hex_6():
    """All 6 boundary edges of hex 0 form a hex cycle. Longest simple path = 6."""
    env = create(); reset_(env, 1); clear_board(env)
    set_road(env, [0, 1, 6, 7, 11, 12], 0)
    recompute_awards(env)
    fails = fail(road_length(env, 0) == 6, f"length = {road_length(env, 0)} (want 6)")
    destroy(env)
    return fails


def test_cut_by_opponent_settlement():
    """Line of 4 (edges 0,1,2,3) cut by opponent settlement at v2.
    Player 0's max simple path = 2 (v0-v1-v2 = 2 edges, or v2-v3-v4 = 2)."""
    env = create(); reset_(env, 1); clear_board(env)
    set_road(env, [0, 1, 2, 3], 0)
    set_node(env, 2, NODE_SETTLEMENT, 1)  # opponent at v2 cuts the road
    recompute_awards(env)
    fails = fail(road_length(env, 0) == 2, f"length = {road_length(env, 0)} (want 2)")
    destroy(env)
    return fails


def test_branch_y_shape():
    """T at v2: e0(v0-v1), e1(v1-v2), e2(v2-v3), e7(v2-v10).
    Longest simple path through fork: v0-v1-v2-v3 = 3 OR v0-v1-v2-v10 = 3.
    Both arms equal so length = 3 either way."""
    env = create(); reset_(env, 1); clear_board(env)
    set_road(env, [0, 1, 2, 7], 0)
    recompute_awards(env)
    fails = fail(road_length(env, 0) == 3, f"length = {road_length(env, 0)} (want 3)")
    destroy(env)
    return fails


def test_below_threshold_no_award():
    """Length 4 = no longest road awarded (threshold is 5)."""
    env = create(); reset_(env, 1); clear_board(env)
    set_road(env, [0, 1, 2, 3], 0)
    recompute_awards(env)
    fails = fail(road_length(env, 0) == 4, f"length = {road_length(env, 0)}")
    fails += fail(longest_owner(env) == NO_PLAYER, f"award given below threshold (owner={longest_owner(env)})")
    destroy(env)
    return fails


def test_threshold_award_at_5():
    """First crossing of threshold awards +2 VP."""
    env = create(); reset_(env, 1); clear_board(env)
    vp_before = p_vp(env, 0)
    pub_before = p_vp_pub(env, 0)
    set_road(env, [0, 1, 2, 3, 4], 0)
    recompute_awards(env)
    fails = fail(longest_owner(env) == 0, f"owner = {longest_owner(env)}")
    fails += fail(p_vp(env, 0) == vp_before + 2, f"VP = {p_vp(env, 0)} (want {vp_before + 2})")
    fails += fail(p_vp_pub(env, 0) == pub_before + 2, "public VP did not get +2")
    destroy(env)
    return fails


def test_no_transfer_on_tie():
    """Holder at length 5; opponent reaches 5. No transfer (incumbent keeps)."""
    env = create(); reset_(env, 1); clear_board(env)
    set_road(env, [0, 1, 2, 3, 4], 0)
    recompute_awards(env)
    fails = fail(longest_owner(env) == 0, "p0 should hold first")
    vp_p0_before = p_vp(env, 0)

    # P1 builds a 5-length road on a different part of board (edges 18..22 line on left side)
    # edges 18, 23, 24, 25, 26 form a path along v17-v16-v17... let me verify by edge_to_node
    # Use edges 23..27 which form a line: edge_to_node[23]=(16,17), [24]=(17,18), [25]=(18,19), [26]=(19,20), [27]=(20,21)
    set_road(env, [23, 24, 25, 26, 27], 1)
    recompute_awards(env)
    fails += fail(longest_owner(env) == 0, f"title transferred on tie (owner={longest_owner(env)})")
    fails += fail(p_vp(env, 0) == vp_p0_before, "p0 lost VP on tie")
    destroy(env)
    return fails


def test_transfer_on_strict_exceed():
    """Holder at 5; opponent reaches 6. Transfer."""
    env = create(); reset_(env, 1); clear_board(env)
    set_road(env, [0, 1, 2, 3, 4], 0)
    recompute_awards(env)
    fails = fail(longest_owner(env) == 0, "p0 should hold first")

    # P1 builds a longer road than 5. Use 6 edges along left side.
    # Edges 23..28 form 6-length: (16,17),(17,18),(18,19),(19,20),(20,21),(21,22)
    set_road(env, [23, 24, 25, 26, 27, 28], 1)
    recompute_awards(env)
    fails += fail(longest_owner(env) == 1, f"title not transferred (owner={longest_owner(env)})")
    fails += fail(road_length(env, 1) == 6, f"p1 length = {road_length(env, 1)}")
    destroy(env)
    return fails


def test_cut_below_threshold_loses_title():
    """Holder at 5; opponent settlement cuts road to length 2. Title lost."""
    env = create(); reset_(env, 1); clear_board(env)
    set_road(env, [0, 1, 2, 3, 4], 0)
    recompute_awards(env)
    fails = fail(longest_owner(env) == 0, "p0 should hold first")
    vp_before = p_vp(env, 0)

    # Opponent cuts the road in the middle (settlement at v3).
    set_node(env, 3, NODE_SETTLEMENT, 1)
    recompute_awards(env)
    fails += fail(longest_owner(env) == NO_PLAYER, f"title not lost on cut (owner={longest_owner(env)})")
    fails += fail(p_vp(env, 0) == vp_before - 2, f"VP penalty not applied: {vp_before} -> {p_vp(env, 0)}")
    destroy(env)
    return fails


def test_disconnected_components_max_length():
    """Two separate roads: lengths 3 and 4. Max = 4."""
    env = create(); reset_(env, 1); clear_board(env)
    set_road(env, [0, 1, 2], 0)             # line of 3 on top
    set_road(env, [23, 24, 25, 26], 0)       # line of 4 on left
    recompute_awards(env)
    fails = fail(road_length(env, 0) == 4, f"length = {road_length(env, 0)} (want 4)")
    destroy(env)
    return fails


def test_own_settlement_does_not_block():
    """Player's own settlement at a road node does NOT cut the road."""
    env = create(); reset_(env, 1); clear_board(env)
    set_road(env, [0, 1, 2, 3, 4], 0)
    set_node(env, 2, NODE_SETTLEMENT, 0)  # player 0's own settlement mid-road
    recompute_awards(env)
    fails = fail(road_length(env, 0) == 5, f"own settle should not cut: {road_length(env, 0)}")
    destroy(env)
    return fails


def test_per_player_independent():
    """Different players' roads are computed independently."""
    env = create(); reset_(env, 1); clear_board(env)
    set_road(env, [0, 1, 2], 0)
    set_road(env, [23, 24, 25, 26, 27], 1)
    recompute_awards(env)
    fails = fail(road_length(env, 0) == 3, f"p0 = {road_length(env, 0)}")
    fails += fail(road_length(env, 1) == 5, f"p1 = {road_length(env, 1)}")
    fails += fail(longest_owner(env) == 1, "title should be p1")
    destroy(env)
    return fails


def main():
    total = 0
    print("== test_linear_road_5 ==");                 total += test_linear_road_5()
    print("== test_cycle_around_hex_6 ==");            total += test_cycle_around_hex_6()
    print("== test_cut_by_opponent_settlement ==");    total += test_cut_by_opponent_settlement()
    print("== test_branch_y_shape ==");                total += test_branch_y_shape()
    print("== test_below_threshold_no_award ==");      total += test_below_threshold_no_award()
    print("== test_threshold_award_at_5 ==");          total += test_threshold_award_at_5()
    print("== test_no_transfer_on_tie ==");            total += test_no_transfer_on_tie()
    print("== test_transfer_on_strict_exceed ==");     total += test_transfer_on_strict_exceed()
    print("== test_cut_below_threshold_loses_title =="); total += test_cut_below_threshold_loses_title()
    print("== test_disconnected_components_max_length =="); total += test_disconnected_components_max_length()
    print("== test_own_settlement_does_not_block =="); total += test_own_settlement_does_not_block()
    print("== test_per_player_independent ==");        total += test_per_player_independent()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} failures")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
