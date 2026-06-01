"""Mask interpretation + on-board overlay.

`bucket_mask(mask)`        : categorize legal action IDs into named buckets.
`draw_mask_overlay(ax, …)` : highlight legal spatial actions (settle/city/
                              road/move_robber) on the board axis.
`render_mask_chips(ax, …)` : compact list of non-spatial legal actions in a
                              side panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, RegularPolygon

from ui import geometry as G
from ui.action_names import (
    ActionInfo,
    CAT_CITY,
    CAT_MOVE_ROBBER,
    CAT_ROAD,
    CAT_SETTLE,
    info,
    mask_to_ids,
)


# Highlight colours (translucent, layered above board art).
HILITE_SETTLE = "#22c55e"   # green
HILITE_CITY   = "#f59e0b"   # amber (city upgrade signal)
HILITE_ROAD   = "#22c55e"   # green
HILITE_ROBBER = "#ef4444"   # red


@dataclass
class MaskBuckets:
    """Legal IDs split by category. `spatial` is the union of board-overlay
    categories; `chips` is everything else (one-line strings)."""
    by_category: dict[str, list[ActionInfo]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.by_category.values())


def bucket_mask(mask: np.ndarray) -> MaskBuckets:
    out: dict[str, list[ActionInfo]] = {}
    for aid in mask_to_ids(mask):
        ai = info(aid)
        out.setdefault(ai.category, []).append(ai)
    return MaskBuckets(by_category=out)


# ---------------------------------------------------------------------------
# Board overlay
# ---------------------------------------------------------------------------

def draw_mask_overlay(ax: Axes, mask: np.ndarray) -> None:
    """Highlight legal spatial actions on `ax` (already has a board drawn)."""
    buckets = bucket_mask(mask)

    for ai in buckets.by_category.get(CAT_SETTLE, ()):
        x, y = G.NODE_XY[ai.payload]  # type: ignore[index]
        ax.add_patch(Circle((x, y), 0.22, facecolor="none",
                            edgecolor=HILITE_SETTLE, linewidth=2.2,
                            alpha=0.95, zorder=9))

    for ai in buckets.by_category.get(CAT_CITY, ()):
        x, y = G.NODE_XY[ai.payload]  # type: ignore[index]
        ax.add_patch(Circle((x, y), 0.27, facecolor="none",
                            edgecolor=HILITE_CITY, linewidth=2.2,
                            alpha=0.95, zorder=9))

    for ai in buckets.by_category.get(CAT_ROAD, ()):
        eid = ai.payload  # type: ignore[assignment]
        (ax_, ay_), (bx_, by_) = G.edge_endpoints(eid)  # type: ignore[arg-type]
        ax.plot([ax_, bx_], [ay_, by_],
                color=HILITE_ROAD, linewidth=6.5, alpha=0.45,
                solid_capstyle="round", zorder=3)

    for ai in buckets.by_category.get(CAT_MOVE_ROBBER, ()):
        cx, cy = G.HEX_XY[ai.payload]  # type: ignore[index]
        ax.add_patch(RegularPolygon(
            (cx, cy), numVertices=6, radius=G.SIZE * 0.93,
            orientation=0.0,
            edgecolor=HILITE_ROBBER, facecolor="none",
            linewidth=2.4, alpha=0.95, zorder=8,
        ))


# ---------------------------------------------------------------------------
# Non-spatial legal-action chip list
# ---------------------------------------------------------------------------

# Categories rendered as chips (everything that doesn't get a board halo).
_CHIP_CATEGORY_ORDER = (
    "roll", "end_turn",
    "buy_dev", "play_knight", "play_road_building",
    "play_year_of_plenty", "play_monopoly",
    "trade_bank",
    "trade_add_give", "trade_add_want",
    "trade_open", "trade_accept", "trade_decline",
    "trade_confirm", "trade_cancel",
    "discard", "steal",
)


def chip_lines(buckets: MaskBuckets, *, per_cat_limit: int = 8) -> list[str]:
    """Compact label list for the side panel."""
    lines: list[str] = []
    for cat in _CHIP_CATEGORY_ORDER:
        ais = buckets.by_category.get(cat, [])
        if not ais:
            continue
        if len(ais) <= per_cat_limit:
            labels = [ai.label for ai in ais]
            lines.append(f"{cat}: {', '.join(labels)}")
        else:
            shown = ", ".join(ai.label for ai in ais[:per_cat_limit])
            lines.append(f"{cat} ({len(ais)}): {shown}, …")
    return lines


def spatial_summary(buckets: MaskBuckets) -> str:
    """One-line count of spatial legal actions for the side panel header."""
    n = (len(buckets.by_category.get(CAT_SETTLE, ()))
         + len(buckets.by_category.get(CAT_CITY, ()))
         + len(buckets.by_category.get(CAT_ROAD, ()))
         + len(buckets.by_category.get(CAT_MOVE_ROBBER, ())))
    parts: list[str] = []
    for cat, sym in ((CAT_SETTLE, "S"), (CAT_CITY, "C"),
                     (CAT_ROAD, "R"), (CAT_MOVE_ROBBER, "Rob")):
        k = len(buckets.by_category.get(cat, ()))
        if k:
            parts.append(f"{sym}{k}")
    return f"legal: {buckets.total}  spatial[{', '.join(parts) if parts else '-'}]"
