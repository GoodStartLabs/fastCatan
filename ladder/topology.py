"""Small public-board topology subset needed by legal-info heuristics."""

from __future__ import annotations

HEX_TO_NODES = (
    (0, 1, 2, 8, 9, 10),
    (2, 3, 4, 10, 11, 12),
    (4, 5, 6, 12, 13, 14),
    (7, 8, 9, 17, 18, 19),
    (9, 10, 11, 19, 20, 21),
    (11, 12, 13, 21, 22, 23),
    (13, 14, 15, 23, 24, 25),
    (16, 17, 18, 27, 28, 29),
    (18, 19, 20, 29, 30, 31),
    (20, 21, 22, 31, 32, 33),
    (22, 23, 24, 33, 34, 35),
    (24, 25, 26, 35, 36, 37),
    (28, 29, 30, 38, 39, 40),
    (30, 31, 32, 40, 41, 42),
    (32, 33, 34, 42, 43, 44),
    (34, 35, 36, 44, 45, 46),
    (39, 40, 41, 47, 48, 49),
    (41, 42, 43, 49, 50, 51),
    (43, 44, 45, 51, 52, 53),
)

PORT_TO_NODES = (
    (0, 1), (3, 4), (14, 15), (26, 37), (45, 46),
    (50, 51), (47, 48), (28, 38), (7, 17),
)

NODE_TO_HEXES: tuple[tuple[int, ...], ...] = tuple(
    tuple(h for h, nodes in enumerate(HEX_TO_NODES) if node in nodes)
    for node in range(54)
)

NODE_TO_PORT: dict[int, int] = {
    node: port for port, nodes in enumerate(PORT_TO_NODES) for node in nodes
}

DICE_PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
