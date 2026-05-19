"""Validation tests for bridge.topology_map.

Phase-1 gate: if any of these fail, the entire bridge pipeline rests on
faulty topology. Subsequent translators (obs encoder, action codec) would
silently corrupt the NN's input/output.
"""

from __future__ import annotations

import pytest

from catanatron import Color
from catanatron.game import Game
from catanatron.models.player import RandomPlayer
from catanatron.models.tiles import LandTile

from bridge import topology_map as T


@pytest.fixture(scope="module")
def cat_map():
    g = Game([RandomPlayer(c) for c in Color])
    return g.state.board.map


def test_counts():
    assert T.NUM_HEXES == 19
    assert T.NUM_NODES == 54
    assert T.NUM_EDGES == 72
    assert len(T.FAST_HEX_TO_COORD) == 19
    assert len(T.NODE_FAST_TO_CAT) == 54
    assert len(T.EDGE_FAST_TO_TUPLE) == 72


def test_hex_coords_unique_and_match_catanatron_land_tiles(cat_map):
    fast_coords = set(T.FAST_HEX_TO_COORD)
    assert len(fast_coords) == 19, "fastcatan hex coords must be unique"

    cat_land_coords = {
        c for c, t in cat_map.tiles.items() if isinstance(t, LandTile)
    }
    assert fast_coords == cat_land_coords, (
        f"land coord mismatch — only in fast: {fast_coords - cat_land_coords}, "
        f"only in cat: {cat_land_coords - fast_coords}"
    )


def test_hex_inverse_lookup_round_trips():
    for fast_id, coord in enumerate(T.FAST_HEX_TO_COORD):
        assert T.COORD_TO_FAST_HEX[coord] == fast_id


def test_node_mapping_is_bijection(cat_map):
    fast_ids = set(range(T.NUM_NODES))
    cat_ids = set(T.NODE_FAST_TO_CAT)
    assert len(cat_ids) == T.NUM_NODES, "duplicate catanatron node IDs in mapping"

    # Every cat node in our mapping must be a land node in catanatron.
    cat_land_nodes: set[int] = set()
    for coord, tile in cat_map.tiles.items():
        if isinstance(tile, LandTile):
            for nid in tile.nodes.values():
                cat_land_nodes.add(nid)
    assert len(cat_land_nodes) == 54
    assert cat_ids == cat_land_nodes, (
        f"mapped cat ids != catanatron's land node ids — diff "
        f"{cat_ids ^ cat_land_nodes}"
    )

    # Inverse dict round-trips.
    for fast in fast_ids:
        cat = T.NODE_FAST_TO_CAT[fast]
        assert T.NODE_CAT_TO_FAST[cat] == fast


def test_node_hex_signatures_agree(cat_map):
    """For every fastcatan node, its set of touching hex IDs must map onto
    the same set of cube coords that the corresponding catanatron node
    touches. Catches rotation / reflection bugs."""
    for fast_node in range(T.NUM_NODES):
        cat_node = T.NODE_FAST_TO_CAT[fast_node]
        fast_hexes = {T.FAST_HEX_TO_COORD[h] for h in T.FAST_NODE_TO_HEX[fast_node]}
        cat_hexes: set = set()
        for coord, tile in cat_map.tiles.items():
            if isinstance(tile, LandTile) and cat_node in tile.nodes.values():
                cat_hexes.add(coord)
        assert fast_hexes == cat_hexes, (
            f"fast node {fast_node} (cat {cat_node}): "
            f"fast hexes {fast_hexes} != cat hexes {cat_hexes}"
        )


def test_edge_round_trips():
    for fast_id, tpl in enumerate(T.EDGE_FAST_TO_TUPLE):
        assert T.EDGE_TUPLE_TO_FAST[tpl] == fast_id
        # Both orderings registered.
        assert T.EDGE_TUPLE_TO_FAST[(tpl[1], tpl[0])] == fast_id


def test_edge_coverage_matches_catanatron(cat_map):
    """Every fastcatan edge corresponds to a real land-edge in catanatron,
    and vice versa."""
    fast_edges_as_tuples = {tuple(sorted(t)) for t in T.EDGE_FAST_TO_TUPLE}
    assert len(fast_edges_as_tuples) == T.NUM_EDGES

    cat_land_edges: set[tuple[int, int]] = set()
    for coord, tile in cat_map.tiles.items():
        if isinstance(tile, LandTile):
            for ref, e in tile.edges.items():
                cat_land_edges.add(tuple(sorted(e)))
    # Catanatron's land edges are exactly the 72 we mapped.
    assert fast_edges_as_tuples == cat_land_edges, (
        f"edge sets differ — only in fast: {fast_edges_as_tuples - cat_land_edges}, "
        f"only in cat: {cat_land_edges - fast_edges_as_tuples}"
    )


def test_edge_endpoints_match_node_mapping():
    """The endpoints of each translated edge tuple must equal the node
    mapping of fastcatan's edge_to_node entry."""
    from bridge.topology_map import FAST_EDGE_TO_NODE, NODE_FAST_TO_CAT, EDGE_FAST_TO_TUPLE
    for fast_id, (a, b) in enumerate(FAST_EDGE_TO_NODE):
        expected = tuple(sorted([NODE_FAST_TO_CAT[a], NODE_FAST_TO_CAT[b]]))
        assert EDGE_FAST_TO_TUPLE[fast_id] == expected


def test_center_hex_anchor():
    """Hex 9 should map to cube coord (0, 0, 0) — the standard board's
    center tile."""
    assert T.FAST_HEX_TO_COORD[9] == (0, 0, 0)
    assert T.COORD_TO_FAST_HEX[(0, 0, 0)] == 9
