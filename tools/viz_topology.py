#!/usr/bin/env python3
"""Visual sanity-check for include/topology.hpp.

Renders the standard 19-hex Catan board with all IDs labeled:
  - hex IDs at hex centers
  - node IDs at corners (port-nodes shown in red)
  - edge IDs at edge midpoints

Usage:
    python3 tools/viz_topology.py [--save out.png] [--no-edges]

Verify by eye: every node/edge ID matches its position on the board, and
ports sit on the expected coastal edges.
"""

from __future__ import annotations
import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle


# ---------------------------------------------------------------------
# Parse topology.hpp — single source of truth.
# ---------------------------------------------------------------------

def _read_header() -> str:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "include" / "topology.hpp",
        here.parents[2] / "include" / "topology.hpp",
    ]
    for path in candidates:
        if path.exists():
            return path.read_text()
    raise FileNotFoundError("topology.hpp not found near tools/")

def _parse_table(text: str, name: str) -> tuple[tuple[int, ...], ...]:
    """Extract `inline constexpr std::array<...> NAME = {{ ... }};`
    Returns rows of ints with 0xFF sentinels stripped."""
    # find the assignment block (greedy until matching closing }};)
    pat = re.compile(
        rf"\b{re.escape(name)}\s*=\s*\{{\{{(?P<body>.*?)\}}\}}\s*;",
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        raise KeyError(f"table {name!r} not found in topology.hpp")
    body = m.group("body")
    rows = re.findall(r"\{\{([^{}]*)\}\}", body)
    out = []
    for r in rows:
        vals = [int(x, 16) for x in re.findall(r"0x[0-9A-Fa-f]+", r)]
        # strip trailing sentinels for variable-width tables; keep all for fixed
        while vals and vals[-1] == 0xFF:
            vals.pop()
        out.append(tuple(vals))
    return tuple(out)

_HDR = _read_header()
HEX_NODE  = _parse_table(_HDR, "hex_to_node")
EDGE_NODE = _parse_table(_HDR, "edge_to_node")
PORT_NODE = _parse_table(_HDR, "port_to_node")

NUM_HEXES = len(HEX_NODE)
NUM_NODES = max(max(row) for row in HEX_NODE) + 1
NUM_EDGES = len(EDGE_NODE)
NUM_PORTS = len(PORT_NODE)

assert (NUM_HEXES, NUM_NODES, NUM_EDGES, NUM_PORTS) == (19, 54, 72, 9), (
    f"unexpected counts: {(NUM_HEXES, NUM_NODES, NUM_EDGES, NUM_PORTS)}"
)
assert all(len(row) == 6 for row in HEX_NODE), "hex_to_node rows must be 6-wide"
assert all(len(row) == 2 for row in EDGE_NODE), "edge_to_node rows must be 2-wide"
assert all(len(row) == 2 for row in PORT_NODE), "port_to_node rows must be 2-wide"

# Hex layout: rows of 3, 4, 5, 4, 3.
HEX_ROWS = [3, 4, 5, 4, 3]
HEX_TO_ROWCOL = []
for r, w in enumerate(HEX_ROWS):
    for c in range(w):
        HEX_TO_ROWCOL.append((r, c))
assert len(HEX_TO_ROWCOL) == NUM_HEXES


# ---------------------------------------------------------------------
# Geometry: pointy-top hexes, half-row offset.
# ---------------------------------------------------------------------

SQRT3 = math.sqrt(3.0)
SIZE = 1.0  # hex circumradius

def hex_center(hid: int) -> tuple[float, float]:
    r, c = HEX_TO_ROWCOL[hid]
    width = HEX_ROWS[r]
    x = (c - (width - 1) / 2.0) * SQRT3 * SIZE
    y = -1.5 * r * SIZE
    return x, y

# Corner offsets in HEX_NODE order: UL, T, UR, BL, B, BR.
CORNER_OFFSETS = [
    (-SQRT3 / 2.0,  +0.5),  # UL
    ( 0.0,          +1.0),  # T
    (+SQRT3 / 2.0,  +0.5),  # UR
    (-SQRT3 / 2.0,  -0.5),  # BL
    ( 0.0,          -1.0),  # B
    (+SQRT3 / 2.0,  -0.5),  # BR
]

def compute_node_positions() -> list[tuple[float, float]]:
    pos = [None] * NUM_NODES
    for hid in range(NUM_HEXES):
        cx, cy = hex_center(hid)
        for k, nid in enumerate(HEX_NODE[hid]):
            ox, oy = CORNER_OFFSETS[k]
            x, y = cx + SIZE * ox, cy + SIZE * oy
            if pos[nid] is None:
                pos[nid] = (x, y)
            else:
                # consistency check: same node from another hex must match
                px, py = pos[nid]
                if abs(px - x) > 1e-6 or abs(py - y) > 1e-6:
                    raise AssertionError(
                        f"node {nid} mismatch: ({px:.3f},{py:.3f}) vs ({x:.3f},{y:.3f})"
                    )
    if any(p is None for p in pos):
        missing = [i for i, p in enumerate(pos) if p is None]
        raise AssertionError(f"unplaced nodes: {missing}")
    return pos


# ---------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------

def draw(out_path: str | None, draw_edges: bool):
    nodes = compute_node_positions()
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set_aspect("equal")
    ax.axis("off")

    # hex polygons + IDs
    for hid in range(NUM_HEXES):
        cx, cy = hex_center(hid)
        hexagon = RegularPolygon(
            (cx, cy), numVertices=6, radius=SIZE,
            orientation=math.radians(0),  # pointy-top
            edgecolor="#888", facecolor="#f7e9b0", linewidth=1.2,
        )
        ax.add_patch(hexagon)
        ax.text(cx, cy + 0.0, f"0x{hid:02X}",
                ha="center", va="center", fontsize=11,
                color="#a04000", fontweight="bold")

    # edges with IDs
    if draw_edges:
        for eid, (a, b) in enumerate(EDGE_NODE):
            ax_, ay_ = nodes[a]
            bx_, by_ = nodes[b]
            ax.plot([ax_, bx_], [ay_, by_],
                    color="#555", linewidth=0.8, zorder=2)
            mx, my = (ax_ + bx_) / 2.0, (ay_ + by_) / 2.0
            # offset label slightly perpendicular
            dx, dy = bx_ - ax_, by_ - ay_
            length = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / length, dx / length
            offset = 0.10
            ax.text(mx + nx * offset, my + ny * offset, f"0x{eid:02X}",
                    ha="center", va="center", fontsize=6.5,
                    color="#1a5e1a", zorder=3)

    # nodes — port-nodes in red, others white
    PORT_NODES = {n for pair in PORT_NODE for n in pair}
    RED = "#e74c3c"
    for nid, (x, y) in enumerate(nodes):
        if nid in PORT_NODES:
            ax.add_patch(Circle((x, y), 0.16, facecolor=RED,
                                edgecolor="black", linewidth=1.0, zorder=4))
            tcolor = "white"
        else:
            ax.add_patch(Circle((x, y), 0.13, facecolor="white",
                                edgecolor="black", linewidth=1.0, zorder=4))
            tcolor = "black"
        ax.text(x, y, f"{nid:02X}", ha="center", va="center",
                fontsize=7.0, color=tcolor, fontweight="bold", zorder=5)

    pad = 1.2
    xs = [p[0] for p in nodes]
    ys = [p[1] for p in nodes]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=160, bbox_inches="tight")
        print(f"saved {out}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", help="output PNG path; omit to open window")
    p.add_argument("--no-edges", action="store_true",
                   help="skip edge drawing/labels (less cluttered)")
    args = p.parse_args()
    draw(args.save, draw_edges=not args.no_edges)


if __name__ == "__main__":
    main()
