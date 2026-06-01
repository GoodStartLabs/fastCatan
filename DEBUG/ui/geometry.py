"""Hex/node/edge geometry for board rendering.

Parses `include/topology.hpp` (single source of truth for IDs) and computes
2D positions for every hex/node/edge. Self-contained so `ui/` has no
dependency on `visual/viz_topology.py` (which is a CLI script, not a module).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

SQRT3 = math.sqrt(3.0)
SIZE = 1.0  # hex circumradius

# Pointy-top hex corner offsets in HEX_NODE order: UL, T, UR, BL, B, BR.
CORNER_OFFSETS = (
    (-SQRT3 / 2.0, +0.5),
    ( 0.0,         +1.0),
    (+SQRT3 / 2.0, +0.5),
    (-SQRT3 / 2.0, -0.5),
    ( 0.0,         -1.0),
    (+SQRT3 / 2.0, -0.5),
)

HEX_ROWS = (3, 4, 5, 4, 3)


# ---------------------------------------------------------------------------
# topology.hpp parsing
# ---------------------------------------------------------------------------

def _read_header() -> str:
    here = Path(__file__).resolve()
    for cand in (here.parents[1] / "include" / "topology.hpp",
                 here.parents[2] / "include" / "topology.hpp"):
        if cand.exists():
            return cand.read_text()
    raise FileNotFoundError("topology.hpp not found near ui/")


def _parse_table(text: str, name: str) -> tuple[tuple[int, ...], ...]:
    pat = re.compile(
        rf"\b{re.escape(name)}\s*=\s*\{{\{{(?P<body>.*?)\}}\}}\s*;",
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        raise KeyError(f"table {name!r} not found in topology.hpp")
    rows = re.findall(r"\{\{([^{}]*)\}\}", m.group("body"))
    out: list[tuple[int, ...]] = []
    for r in rows:
        vals = [int(x, 16) for x in re.findall(r"0x[0-9A-Fa-f]+", r)]
        while vals and vals[-1] == 0xFF:
            vals.pop()
        out.append(tuple(vals))
    return tuple(out)


_HDR = _read_header()
HEX_NODE = _parse_table(_HDR, "hex_to_node")
EDGE_NODE = _parse_table(_HDR, "edge_to_node")
PORT_NODE = _parse_table(_HDR, "port_to_node")

NUM_HEXES = len(HEX_NODE)
NUM_NODES = max(max(row) for row in HEX_NODE) + 1
NUM_EDGES = len(EDGE_NODE)
NUM_PORTS = len(PORT_NODE)

assert (NUM_HEXES, NUM_NODES, NUM_EDGES, NUM_PORTS) == (19, 54, 72, 9), (
    f"unexpected topology sizes: {(NUM_HEXES, NUM_NODES, NUM_EDGES, NUM_PORTS)}"
)

HEX_TO_ROWCOL = []
for _r, _w in enumerate(HEX_ROWS):
    for _c in range(_w):
        HEX_TO_ROWCOL.append((_r, _c))
assert len(HEX_TO_ROWCOL) == NUM_HEXES


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

def hex_center(hid: int) -> tuple[float, float]:
    r, c = HEX_TO_ROWCOL[hid]
    width = HEX_ROWS[r]
    x = (c - (width - 1) / 2.0) * SQRT3 * SIZE
    y = -1.5 * r * SIZE
    return x, y


def _compute_node_positions() -> list[tuple[float, float]]:
    pos: list[tuple[float, float] | None] = [None] * NUM_NODES
    for hid in range(NUM_HEXES):
        cx, cy = hex_center(hid)
        for k, nid in enumerate(HEX_NODE[hid]):
            ox, oy = CORNER_OFFSETS[k]
            x, y = cx + SIZE * ox, cy + SIZE * oy
            if pos[nid] is None:
                pos[nid] = (x, y)
            else:
                px, py = pos[nid]
                if abs(px - x) > 1e-6 or abs(py - y) > 1e-6:
                    raise AssertionError(
                        f"node {nid} mismatch: ({px:.3f},{py:.3f}) vs ({x:.3f},{y:.3f})"
                    )
    if any(p is None for p in pos):
        missing = [i for i, p in enumerate(pos) if p is None]
        raise AssertionError(f"unplaced nodes: {missing}")
    return pos  # type: ignore[return-value]


NODE_XY: tuple[tuple[float, float], ...] = tuple(_compute_node_positions())
HEX_XY: tuple[tuple[float, float], ...] = tuple(hex_center(h) for h in range(NUM_HEXES))


def edge_midpoint(eid: int) -> tuple[float, float]:
    a, b = EDGE_NODE[eid]
    ax, ay = NODE_XY[a]
    bx, by = NODE_XY[b]
    return ((ax + bx) / 2.0, (ay + by) / 2.0)


def edge_endpoints(eid: int) -> tuple[tuple[float, float], tuple[float, float]]:
    a, b = EDGE_NODE[eid]
    return NODE_XY[a], NODE_XY[b]


PORT_NODES_FLAT: frozenset[int] = frozenset(n for pair in PORT_NODE for n in pair)


def port_midpoint(pid: int) -> tuple[float, float]:
    a, b = PORT_NODE[pid]
    ax, ay = NODE_XY[a]
    bx, by = NODE_XY[b]
    return ((ax + bx) / 2.0, (ay + by) / 2.0)


def bbox(pad: float = 1.2) -> tuple[float, float, float, float]:
    xs = [p[0] for p in NODE_XY]
    ys = [p[1] for p in NODE_XY]
    return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)
