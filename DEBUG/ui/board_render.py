"""Matplotlib board renderer driven by a decoded `BoardView`.

`draw_board(ax, view, *, current_player, options)` paints the full board:
  - hex polygons coloured by resource, number token at centre
  - settlements as filled triangles, cities as filled squares (per-seat colour)
  - roads as thick coloured segments
  - robber as a black disk on its hex
  - ports as wedges + labels on coastal nodes

`view` is POV-relative (relseat 0 = self). To recover absolute seats for
colouring, the caller passes the recorded `current_player`. Owner relseat r
→ absolute seat `(current_player + r) % 4`.

Mask overlay is layered on top via `draw_mask_overlay(ax, mask, ...)` in
`mask_view.py` (M4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Polygon, RegularPolygon

from ui import geometry as G
from ui.obs_decoder import BoardView

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

# Per-seat colours (absolute seat 0..3). Picked for contrast on the hex palette.
SEAT_COLORS = ("#d33", "#39c", "#e9a000", "#5cb85c")
SEAT_NAMES = ("P0", "P1", "P2", "P3")

# Hex resource fills (decoder string -> hex).
HEX_FILL = {
    "brick":  "#c0552c",
    "lumber": "#3f7e3f",
    "wool":   "#a8d479",
    "grain":  "#e6c84a",
    "ore":    "#888a8e",
    "desert": "#e6d4a8",
    "unknown": "#dddddd",
}

NUMBER_HOT = {6, 8}  # bold red token like the printed board

ROBBER_COLOR = "#111"


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class RenderOptions:
    show_node_ids: bool = False
    show_edge_ids: bool = False
    show_hex_ids: bool = False
    show_ports: bool = True
    show_empty_nodes: bool = True
    hex_alpha: float = 0.85
    title: str | None = None


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _seat_color(rel: int, current_player: int) -> str:
    return SEAT_COLORS[(current_player + rel) & 0x3]


def _seat_name(rel: int, current_player: int) -> str:
    return SEAT_NAMES[(current_player + rel) & 0x3]


def _draw_hex(ax: Axes, hid: int, resource: str, number: int,
              robber: bool, opts: RenderOptions) -> None:
    cx, cy = G.HEX_XY[hid]
    hexagon = RegularPolygon(
        (cx, cy), numVertices=6, radius=G.SIZE,
        orientation=0.0,
        edgecolor="#666",
        facecolor=HEX_FILL.get(resource, "#dddddd"),
        linewidth=1.0,
        alpha=opts.hex_alpha,
        zorder=1,
    )
    ax.add_patch(hexagon)

    if number > 0 and resource != "desert":
        token_color = "white"
        text_color = "#cc1f1f" if number in NUMBER_HOT else "#222"
        ax.add_patch(Circle((cx, cy), 0.32, facecolor=token_color,
                            edgecolor="#222", linewidth=0.8, zorder=3))
        weight = "bold" if number in NUMBER_HOT else "normal"
        ax.text(cx, cy, str(number), ha="center", va="center",
                fontsize=13, color=text_color, fontweight=weight, zorder=4)
        # Probability dots
        dots = 6 - abs(7 - number) if number != 7 else 0
        if dots > 0:
            dot_y = cy - 0.20
            spacing = 0.05
            x0 = cx - (dots - 1) * spacing / 2.0
            for i in range(dots):
                ax.add_patch(Circle((x0 + i * spacing, dot_y), 0.012,
                                    facecolor=text_color, zorder=5))

    if opts.show_hex_ids:
        ax.text(cx, cy + 0.55, f"h{hid:02X}",
                ha="center", va="center", fontsize=6, color="#444",
                zorder=4)

    if robber:
        ax.add_patch(Circle((cx + 0.32, cy + 0.32), 0.16,
                            facecolor=ROBBER_COLOR, edgecolor="white",
                            linewidth=1.2, zorder=6))


def _draw_road(ax: Axes, eid: int, color: str) -> None:
    (ax_, ay_), (bx_, by_) = G.edge_endpoints(eid)
    ax.plot([ax_, bx_], [ay_, by_],
            color=color, linewidth=4.5, solid_capstyle="round",
            zorder=4)


def _draw_settlement(ax: Axes, x: float, y: float, color: str) -> None:
    # Filled triangle (roof).
    s = 0.16
    pts = [(x, y + s), (x - s * 0.9, y - s * 0.6), (x + s * 0.9, y - s * 0.6)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color,
                         edgecolor="black", linewidth=0.8, zorder=7))


def _draw_city(ax: Axes, x: float, y: float, color: str) -> None:
    s = 0.18
    pts = [(x - s, y - s * 0.7), (x - s, y + s * 0.3),
           (x - 0.02, y + s * 0.3), (x - 0.02, y + s),
           (x + s, y + s), (x + s, y - s * 0.7)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color,
                         edgecolor="black", linewidth=0.8, zorder=7))


def _draw_empty_node(ax: Axes, x: float, y: float) -> None:
    ax.add_patch(Circle((x, y), 0.04, facecolor="#bbb",
                        edgecolor="#555", linewidth=0.4, zorder=3))


def _draw_port(ax: Axes, pid: int, ptype: str) -> None:
    mx, my = G.port_midpoint(pid)
    label = "3:1" if ptype == "3:1" else f"2:1 {ptype[0].upper()}"
    color = HEX_FILL.get(ptype, "#5b8")
    ax.add_patch(Circle((mx, my), 0.16, facecolor=color,
                        edgecolor="black", linewidth=0.8, alpha=0.85,
                        zorder=5))
    ax.text(mx, my, label, ha="center", va="center",
            fontsize=5.0, color="black", fontweight="bold", zorder=6)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def draw_board(
    ax: Axes,
    view: BoardView,
    *,
    current_player: int,
    options: RenderOptions | None = None,
) -> None:
    """Render the full board onto `ax`. Does not call `plt.show`."""
    opts = options or RenderOptions()
    ax.set_aspect("equal")
    ax.axis("off")

    # Hexes
    for hid in range(G.NUM_HEXES):
        _draw_hex(
            ax, hid,
            resource=view.hex_resources[hid],
            number=view.hex_numbers[hid],
            robber=(view.robber_hex == hid),
            opts=opts,
        )

    # Ports
    if opts.show_ports:
        for pid, ptype in enumerate(view.port_types):
            _draw_port(ax, pid, ptype)

    # Roads
    for eid, owner_rel in enumerate(view.edges):
        if owner_rel is None:
            continue
        _draw_road(ax, eid, _seat_color(owner_rel, current_player))

    # Nodes
    for nid, nstate in enumerate(view.nodes):
        x, y = G.NODE_XY[nid]
        if nstate.kind is None:
            if opts.show_empty_nodes:
                _draw_empty_node(ax, x, y)
        else:
            color = _seat_color(nstate.owner_rel, current_player)
            if nstate.kind == "settlement":
                _draw_settlement(ax, x, y, color)
            elif nstate.kind == "city":
                _draw_city(ax, x, y, color)

        if opts.show_node_ids:
            ax.text(x + 0.18, y + 0.05, f"{nid:02X}",
                    ha="left", va="center", fontsize=5.0,
                    color="#222", zorder=8)

    if opts.show_edge_ids:
        for eid in range(G.NUM_EDGES):
            mx, my = G.edge_midpoint(eid)
            ax.text(mx, my, f"{eid:02X}", ha="center", va="center",
                    fontsize=4.5, color="#114", zorder=3)

    # Frame the board
    xmin, xmax, ymin, ymax = G.bbox(pad=1.2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if opts.title is not None:
        ax.set_title(opts.title, fontsize=10)


# ---------------------------------------------------------------------------
# Standalone driver
# ---------------------------------------------------------------------------

def _main(argv: list[str]) -> int:
    """`python -m ui.board_render --log <p> --step N [--out png] [--pov S]`"""
    import argparse

    import numpy as np

    import fastcatan
    from ui.log_format import decode_snap, read_log
    from ui.obs_decoder import decode_from_env

    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--pov", type=int, default=None,
                   help="POV seat (0..3); default = step's current_player")
    p.add_argument("--out", default=None, help="output PNG; omit for window")
    p.add_argument("--ids", action="store_true",
                   help="draw node/edge/hex IDs")
    args = p.parse_args(argv)

    g = read_log(args.log)
    if not (0 <= args.step < len(g.steps)):
        p.error(f"--step out of range [0, {len(g.steps)})")
    target = g.steps[args.step]

    snap_info = g.nearest_snapshot(args.step)
    if snap_info is None:
        p.error("no snapshot at or before requested step")
    snap_idx, snap_raw = snap_info

    env = fastcatan.Env()
    env.reset(0)
    env.load_snapshot(snap_raw)
    for i in range(snap_idx + 1, args.step + 1):
        env.step(int(g.steps[i].a))

    pov = args.pov if args.pov is not None else int(env.current_player)
    view = decode_from_env(env, pov)

    fig, ax = plt.subplots(figsize=(10, 10))
    opts = RenderOptions(
        show_node_ids=args.ids,
        show_edge_ids=args.ids,
        show_hex_ids=args.ids,
        title=(f"step {args.step}/{len(g.steps)-1}  "
               f"phase={view.phase}  flag={view.flag}  "
               f"P{int(env.current_player)} to act  "
               f"roll={view.last_roll}  POV=P{pov}"),
    )
    draw_board(ax, view, current_player=int(env.current_player), options=opts)

    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=140, bbox_inches="tight")
        print(f"saved {args.out}")
    else:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
