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
# Hex/node mappings: derived from the networkx-solved isomorphism below.
# The isomorphism is the unique mapping that (a) is structurally valid
# between fastcatan's bipartite hex-node graph and catanatron's land-only
# equivalent and (b) aligns the 9 fastcatan port edges onto the 9
# catanatron port edges.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module-load solver: node + edge mappings via hexagon cycle alignment.
# ---------------------------------------------------------------------------


def _build_node_edge_mapping():
    """Find the node-ID isomorphism between fastcatan and catanatron's
    land-only board graphs via networkx VF2.

    Both graphs are bipartite (54 land nodes + 19 land hexes); edges are
    land-edge node-pairs plus node-hex incidence. networkx enumerates
    isomorphisms; we iterate until we find one whose mapping aligns
    fastcatan's 9 port edges onto catanatron's 9 port edges. That
    constraint pins the canonical absolute orientation.
    """
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher

    from catanatron import Color
    from catanatron.game import Game
    from catanatron.models.enums import EdgeRef
    from catanatron.models.player import RandomPlayer
    from catanatron.models.tiles import LandTile, Port

    GF = nx.Graph()
    for n in range(NUM_NODES):
        GF.add_node(("n", n), kind="node")
    for h in range(NUM_HEXES):
        GF.add_node(("h", h), kind="hex")
    for a, b in FAST_EDGE_TO_NODE:
        GF.add_edge(("n", a), ("n", b))
    for h, nodes in enumerate(FAST_HEX_TO_NODE_RAW):
        for n in nodes:
            GF.add_edge(("n", n), ("h", h))

    g = Game([RandomPlayer(c) for c in Color])
    m = g.state.board.map

    cat_land_tiles = [(c, t) for c, t in m.tiles.items() if isinstance(t, LandTile)]
    cat_land_nodes: set[int] = set()
    for _, tile in cat_land_tiles:
        cat_land_nodes.update(tile.nodes.values())
    cat_land_edges: set[tuple[int, int]] = set()
    for _, tile in cat_land_tiles:
        for e in tile.edges.values():
            cat_land_edges.add(tuple(sorted(e)))

    GC = nx.Graph()
    for n in cat_land_nodes:
        GC.add_node(("n", n), kind="node")
    for coord, _ in cat_land_tiles:
        GC.add_node(("h", coord), kind="hex")
    for a, b in cat_land_edges:
        GC.add_edge(("n", a), ("n", b))
    for coord, tile in cat_land_tiles:
        for nid in tile.nodes.values():
            GC.add_edge(("n", nid), ("h", coord))

    cat_port_edges: set[frozenset[int]] = set()
    for coord, tile in m.tiles.items():
        if isinstance(tile, Port):
            edge = tile.edges[EdgeRef[tile.direction.name]]
            cat_port_edges.add(frozenset(edge))

    matcher = GraphMatcher(GF, GC, node_match=lambda a, b: a["kind"] == b["kind"])

    for iso in matcher.isomorphisms_iter():
        node_map = {k[1]: v[1] for k, v in iso.items() if k[0] == "n"}
        hex_map = {k[1]: v[1] for k, v in iso.items() if k[0] == "h"}
        # Port-edge alignment check.
        fast_port_edges_mapped = set()
        for a, b in FAST_PORT_TO_NODE:
            fast_port_edges_mapped.add(frozenset({node_map[a], node_map[b]}))
        if fast_port_edges_mapped == cat_port_edges:
            return node_map, hex_map

    raise RuntimeError(
        "no fastcatan<->catanatron isomorphism aligns the 9 port edges; "
        "their standard board layouts may differ"
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


# fastcatan port_to_node (from include/topology.hpp lines 487-499). Defined
# before the node-edge solver because the port-edge constraint pins the
# canonical absolute orientation.
FAST_PORT_TO_NODE: list[tuple[int, int]] = [
    (0x00, 0x01), (0x03, 0x04), (0x0E, 0x0F), (0x1A, 0x25), (0x2D, 0x2E),
    (0x32, 0x33), (0x2F, 0x30), (0x1C, 0x26), (0x07, 0x11),
]
NUM_PORTS = 9


_node_f2c, _hex_f2c = _build_node_edge_mapping()
NODE_FAST_TO_CAT = [_node_f2c[i] for i in range(NUM_NODES)]
NODE_CAT_TO_FAST = {c: f for f, c in _node_f2c.items()}
EDGE_FAST_TO_TUPLE, EDGE_TUPLE_TO_FAST = _build_edge_mapping(_node_f2c)

FAST_HEX_TO_COORD: list[tuple[int, int, int]] = [_hex_f2c[i] for i in range(NUM_HEXES)]
COORD_TO_FAST_HEX: dict[tuple[int, int, int], int] = {
    c: i for i, c in enumerate(FAST_HEX_TO_COORD)
}


def _build_port_mapping() -> list[tuple[int, int, int]]:
    """For each fastcatan port slot, find the Catanatron Port tile coord
    whose direction-edge endpoints match fastcatan's port_to_node entry.

    A Catanatron Port tile's `direction` attribute names the EdgeRef that
    is the actual port (trading) edge — that's the one shared with the
    adjacent land tile in the trading sense. (A port hex can have
    additional land-adjacent edges at its corners, but only the direction
    edge counts as "the port".)
    """
    from catanatron import Color
    from catanatron.game import Game
    from catanatron.models.enums import EdgeRef
    from catanatron.models.player import RandomPlayer
    from catanatron.models.tiles import Port

    g = Game([RandomPlayer(c) for c in Color])
    m = g.state.board.map

    port_info: list[tuple[tuple, frozenset[int]]] = []
    for coord, tile in m.tiles.items():
        if isinstance(tile, Port):
            edge = tile.edges[EdgeRef[tile.direction.name]]
            port_info.append((coord, frozenset(edge)))

    result: list[tuple[int, int, int]] = []
    for fast_port_idx, (fa, fb) in enumerate(FAST_PORT_TO_NODE):
        cat_a = NODE_FAST_TO_CAT[fa]
        cat_b = NODE_FAST_TO_CAT[fb]
        target = frozenset({cat_a, cat_b})
        match = None
        for coord, edge_nodes in port_info:
            if edge_nodes == target:
                match = coord
                break
        if match is None:
            raise RuntimeError(
                f"could not locate catanatron port tile for fastcatan port "
                f"{fast_port_idx} (cat nodes {target})"
            )
        result.append(match)
    return result


PORT_FAST_TO_COORD: list[tuple[int, int, int]] = _build_port_mapping()
