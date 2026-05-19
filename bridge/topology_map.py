"""Topology mapping between fastcatan IDs and Catanatron board geometry.

Both engines model the standard Catan board (19 land hexes, 54 land nodes,
72 land edges) but number objects differently:
  - fastcatan: integer IDs 0..N-1 in row-major spatial order. Tables are
    mirrored verbatim from `include/topology.hpp`.
  - catanatron: cube coords (x, y, z) for hex tiles. Node IDs are integers
    but enumerated alongside port/water nodes (range 0..95; only 54 of
    those are land nodes — happen to be contiguous 0..53 in practice but
    indices don't line up with fastcatan). Edges are (node_a, node_b)
    tuples.

Built at module load. Hex mapping is hard-coded by y-row inspection (see
plan file). Node/edge mappings are solved by hexagon-cycle matching with
one rotation+reflection seed at the center hex, then propagated by shared
vertices between adjacent hex pairs.
"""

from __future__ import annotations

from typing import Iterable

NUM_HEXES = 19
NUM_NODES = 54
NUM_EDGES = 72

# ---------------------------------------------------------------------------
# fastcatan topology (mirrored from include/topology.hpp)
# ---------------------------------------------------------------------------


def _strip_sentinel(arr: Iterable[Iterable[int]]) -> list[list[int]]:
    return [[x for x in row if x != 0xFF] for row in arr]


FAST_HEX_TO_HEX: list[list[int]] = _strip_sentinel([
    [0x01, 0x03, 0x04, 0xFF, 0xFF, 0xFF],
    [0x00, 0x02, 0x04, 0x05, 0xFF, 0xFF],
    [0x01, 0x05, 0x06, 0xFF, 0xFF, 0xFF],
    [0x00, 0x04, 0x07, 0x08, 0xFF, 0xFF],
    [0x00, 0x01, 0x03, 0x05, 0x08, 0x09],
    [0x01, 0x02, 0x04, 0x06, 0x09, 0x0A],
    [0x02, 0x05, 0x0A, 0x0B, 0xFF, 0xFF],
    [0x03, 0x08, 0x0C, 0xFF, 0xFF, 0xFF],
    [0x03, 0x04, 0x07, 0x09, 0x0C, 0x0D],
    [0x04, 0x05, 0x08, 0x0A, 0x0D, 0x0E],
    [0x05, 0x06, 0x09, 0x0B, 0x0E, 0x0F],
    [0x06, 0x0A, 0x0F, 0xFF, 0xFF, 0xFF],
    [0x07, 0x08, 0x0D, 0x10, 0xFF, 0xFF],
    [0x08, 0x09, 0x0C, 0x0E, 0x10, 0x11],
    [0x09, 0x0A, 0x0D, 0x0F, 0x11, 0x12],
    [0x0A, 0x0B, 0x0E, 0x12, 0xFF, 0xFF],
    [0x0C, 0x0D, 0x11, 0xFF, 0xFF, 0xFF],
    [0x0D, 0x0E, 0x10, 0x12, 0xFF, 0xFF],
    [0x0E, 0x0F, 0x11, 0xFF, 0xFF, 0xFF],
])

# Ordered hexagon-cycle vertex listing for each fastcatan hex.
# Six entries per hex; semantic order: top-left, top-mid, top-right,
# bottom-right, bottom-mid, bottom-left (forming a closed 6-cycle in
# `hex_to_edge`). See topology.hpp lines 47-67 reordered into a cycle.
FAST_HEX_TO_NODE_RAW: list[list[int]] = [
    [0x00, 0x01, 0x02, 0x08, 0x09, 0x0A],
    [0x02, 0x03, 0x04, 0x0A, 0x0B, 0x0C],
    [0x04, 0x05, 0x06, 0x0C, 0x0D, 0x0E],
    [0x07, 0x08, 0x09, 0x11, 0x12, 0x13],
    [0x09, 0x0A, 0x0B, 0x13, 0x14, 0x15],
    [0x0B, 0x0C, 0x0D, 0x15, 0x16, 0x17],
    [0x0D, 0x0E, 0x0F, 0x17, 0x18, 0x19],
    [0x10, 0x11, 0x12, 0x1B, 0x1C, 0x1D],
    [0x12, 0x13, 0x14, 0x1D, 0x1E, 0x1F],
    [0x14, 0x15, 0x16, 0x1F, 0x20, 0x21],
    [0x16, 0x17, 0x18, 0x21, 0x22, 0x23],
    [0x18, 0x19, 0x1A, 0x23, 0x24, 0x25],
    [0x1C, 0x1D, 0x1E, 0x26, 0x27, 0x28],
    [0x1E, 0x1F, 0x20, 0x28, 0x29, 0x2A],
    [0x20, 0x21, 0x22, 0x2A, 0x2B, 0x2C],
    [0x22, 0x23, 0x24, 0x2C, 0x2D, 0x2E],
    [0x27, 0x28, 0x29, 0x2F, 0x30, 0x31],
    [0x29, 0x2A, 0x2B, 0x31, 0x32, 0x33],
    [0x2B, 0x2C, 0x2D, 0x33, 0x34, 0x35],
]


def _to_cycle(nodes6: list[int]) -> list[int]:
    """Reorder the row-major 6-node list (TL, TM, TR, BL, BM, BR) into a
    counter-clockwise 6-cycle around the hex: TL, TM, TR, BR, BM, BL.
    """
    tl, tm, tr, bl, bm, br = nodes6
    return [tl, tm, tr, br, bm, bl]


FAST_HEX_TO_NODE_CYCLE: list[list[int]] = [
    _to_cycle(row) for row in FAST_HEX_TO_NODE_RAW
]

FAST_EDGE_TO_NODE: list[tuple[int, int]] = [
    (0x00, 0x01), (0x01, 0x02), (0x02, 0x03), (0x03, 0x04), (0x04, 0x05), (0x05, 0x06),
    (0x00, 0x08), (0x02, 0x0A), (0x04, 0x0C), (0x06, 0x0E),
    (0x07, 0x08), (0x08, 0x09), (0x09, 0x0A), (0x0A, 0x0B), (0x0B, 0x0C),
    (0x0C, 0x0D), (0x0D, 0x0E), (0x0E, 0x0F),
    (0x07, 0x11), (0x09, 0x13), (0x0B, 0x15), (0x0D, 0x17), (0x0F, 0x19),
    (0x10, 0x11), (0x11, 0x12), (0x12, 0x13), (0x13, 0x14), (0x14, 0x15),
    (0x15, 0x16), (0x16, 0x17), (0x17, 0x18), (0x18, 0x19), (0x19, 0x1A),
    (0x10, 0x1B), (0x12, 0x1D), (0x14, 0x1F), (0x16, 0x21), (0x18, 0x23), (0x1A, 0x25),
    (0x1B, 0x1C), (0x1C, 0x1D), (0x1D, 0x1E), (0x1E, 0x1F), (0x1F, 0x20),
    (0x20, 0x21), (0x21, 0x22), (0x22, 0x23), (0x23, 0x24), (0x24, 0x25),
    (0x1C, 0x26), (0x1E, 0x28), (0x20, 0x2A), (0x22, 0x2C), (0x24, 0x2E),
    (0x26, 0x27), (0x27, 0x28), (0x28, 0x29), (0x29, 0x2A), (0x2A, 0x2B),
    (0x2B, 0x2C), (0x2C, 0x2D), (0x2D, 0x2E),
    (0x27, 0x2F), (0x29, 0x31), (0x2B, 0x33), (0x2D, 0x35),
    (0x2F, 0x30), (0x30, 0x31), (0x31, 0x32), (0x32, 0x33), (0x33, 0x34), (0x34, 0x35),
]
assert len(FAST_EDGE_TO_NODE) == NUM_EDGES

# fastcatan node -> set of touching fastcatan hex IDs (signature for matching)
FAST_NODE_TO_HEX: list[frozenset[int]] = [frozenset() for _ in range(NUM_NODES)]
_tmp_node_hex: list[set[int]] = [set() for _ in range(NUM_NODES)]
for _h, _nodes in enumerate(FAST_HEX_TO_NODE_RAW):
    for _n in _nodes:
        _tmp_node_hex[_n].add(_h)
FAST_NODE_TO_HEX = [frozenset(s) for s in _tmp_node_hex]

# ---------------------------------------------------------------------------
# Hex mapping: fastcatan hex ID -> catanatron cube coord.
# Derived from row-major (top-to-bottom, left-to-right) inspection of
# catanatron's land tiles, grouped by y, sorted by x. See plan file.
# ---------------------------------------------------------------------------

FAST_HEX_TO_COORD: list[tuple[int, int, int]] = [
    (-2,  2,  0), (-1,  2, -1), ( 0,  2, -2),
    (-2,  1,  1), (-1,  1,  0), ( 0,  1, -1), ( 1,  1, -2),
    (-2,  0,  2), (-1,  0,  1), ( 0,  0,  0), ( 1,  0, -1), ( 2,  0, -2),
    (-1, -1,  2), ( 0, -1,  1), ( 1, -1,  0), ( 2, -1, -1),
    ( 0, -2,  2), ( 1, -2,  1), ( 2, -2,  0),
]
assert len(FAST_HEX_TO_COORD) == NUM_HEXES

COORD_TO_FAST_HEX: dict[tuple[int, int, int], int] = {
    c: i for i, c in enumerate(FAST_HEX_TO_COORD)
}

# ---------------------------------------------------------------------------
# Module-load solver: node + edge mappings via hexagon cycle alignment.
# ---------------------------------------------------------------------------


def _build_node_edge_mapping():
    """Solve fastcatan node IDs <-> catanatron node IDs by aligning the
    hexagon cycle of each tile, anchored at the center hex with brute-force
    rotation+reflection.
    """
    from catanatron import Color
    from catanatron.game import Game
    from catanatron.models.enums import NodeRef
    from catanatron.models.player import RandomPlayer

    g = Game([RandomPlayer(c) for c in Color])
    m = g.state.board.map

    # catanatron NodeRef cycle in same orientation as fastcatan TL,TM,TR,BR,BM,BL
    # (counter-clockwise starting from top-left). NodeRef enum order on a
    # pointy-top hex: NORTH, NORTHEAST, SOUTHEAST, SOUTH, SOUTHWEST, NORTHWEST.
    # Our fastcatan cycle goes TL -> TM -> TR -> BR -> BM -> BL. We don't yet
    # know which catanatron NodeRef corresponds to fastcatan's TL, so we try
    # all 12 orientations (6 rotations * 2 mirror) and pick the one that
    # produces a globally consistent mapping.
    nref_cycle = [NodeRef.NORTH, NodeRef.NORTHEAST, NodeRef.SOUTHEAST,
                  NodeRef.SOUTH, NodeRef.SOUTHWEST, NodeRef.NORTHWEST]

    def cat_cycle_nodes(coord: tuple, start_idx: int, reverse: bool) -> list[int]:
        tile = m.tiles[coord]
        idxs = [(start_idx + i) % 6 for i in range(6)]
        if reverse:
            idxs = list(reversed(idxs))
        return [tile.nodes[nref_cycle[i]] for i in idxs]

    center_coord = (0, 0, 0)
    fast_center_hex = 9
    fast_center_cycle = FAST_HEX_TO_NODE_CYCLE[fast_center_hex]

    for start in range(6):
        for reverse in (False, True):
            cat_cycle = cat_cycle_nodes(center_coord, start, reverse)
            node_f2c: dict[int, int] = dict(zip(fast_center_cycle, cat_cycle))
            processed_hexes: set[int] = {fast_center_hex}
            ok = True

            # Fixed-point propagation: repeatedly scan unprocessed hexes,
            # processing any that share a vertex with already-processed ones.
            while True:
                made_progress = False
                for fast_h, cat_coord in enumerate(FAST_HEX_TO_COORD):
                    if fast_h in processed_hexes:
                        continue
                    fast_cycle = FAST_HEX_TO_NODE_CYCLE[fast_h]
                    seed = None
                    for i, fn in enumerate(fast_cycle):
                        if fn in node_f2c:
                            seed = (i, node_f2c[fn])
                            break
                    if seed is None:
                        continue

                    tile = m.tiles[cat_coord]
                    cat_id_at_pos = {tile.nodes[nref_cycle[i]]: i for i in range(6)}
                    if seed[1] not in cat_id_at_pos:
                        ok = False
                        break
                    cat_pos = cat_id_at_pos[seed[1]]
                    fast_pos = seed[0]

                    aligned = False
                    for try_reverse in (False, True):
                        candidate = {}
                        for off in range(6):
                            fp = (fast_pos + off) % 6
                            cp = (cat_pos - off) % 6 if try_reverse else (cat_pos + off) % 6
                            fn = fast_cycle[fp]
                            cn = tile.nodes[nref_cycle[cp]]
                            candidate[fn] = cn
                        if all(node_f2c.get(fn, cn) == cn for fn, cn in candidate.items()):
                            node_f2c.update(candidate)
                            processed_hexes.add(fast_h)
                            made_progress = True
                            aligned = True
                            break
                    if not aligned:
                        ok = False
                        break
                if not ok or not made_progress:
                    break

            if not ok or len(processed_hexes) != NUM_HEXES or len(node_f2c) != NUM_NODES:
                continue

            # Validate: every fastcatan node's hex signature should match
            # the catanatron node's hex coord signature.
            valid = True
            for fn in range(NUM_NODES):
                cn = node_f2c[fn]
                fast_hex_set = set(FAST_HEX_TO_COORD[h] for h in FAST_NODE_TO_HEX[fn])
                cat_hex_set = set()
                for coord, tile in m.tiles.items():
                    from catanatron.models.tiles import LandTile
                    if not isinstance(tile, LandTile):
                        continue
                    if cn in tile.nodes.values():
                        cat_hex_set.add(coord)
                if fast_hex_set != cat_hex_set:
                    valid = False
                    break
            if valid:
                return node_f2c

    raise RuntimeError(
        "could not solve node ID isomorphism between fastcatan and catanatron"
    )


NODE_FAST_TO_CAT: list[int]
NODE_CAT_TO_FAST: dict[int, int]
EDGE_FAST_TO_TUPLE: list[tuple[int, int]]
EDGE_TUPLE_TO_FAST: dict[tuple[int, int], int]


def _build_edge_mapping(node_f2c: dict[int, int]) -> tuple[
    list[tuple[int, int]], dict[tuple[int, int], int]
]:
    fast_to_tuple: list[tuple[int, int]] = []
    for a, b in FAST_EDGE_TO_NODE:
        ca, cb = node_f2c[a], node_f2c[b]
        fast_to_tuple.append(tuple(sorted([ca, cb])))
    tuple_to_fast: dict[tuple[int, int], int] = {}
    for i, t in enumerate(fast_to_tuple):
        tuple_to_fast[t] = i
        tuple_to_fast[(t[1], t[0])] = i  # both orderings
    return fast_to_tuple, tuple_to_fast


_node_f2c = _build_node_edge_mapping()
NODE_FAST_TO_CAT = [_node_f2c[i] for i in range(NUM_NODES)]
NODE_CAT_TO_FAST = {c: f for f, c in _node_f2c.items()}
EDGE_FAST_TO_TUPLE, EDGE_TUPLE_TO_FAST = _build_edge_mapping(_node_f2c)
