"""Side-panel renderer for the replayer.

Paints a text block + per-seat scoreboard onto a matplotlib axis sitting
next to the board. Reads everything from a `BoardView`; absolute seat
numbers are derived from the recorded `current_player` (POV is relseat 0).
"""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from ui.board_render import SEAT_COLORS, SEAT_NAMES
from ui.obs_decoder import (
    BoardView,
    DEV_NAMES,
    RES_NAMES,
)


def _rel_to_abs(rel: int, current_player: int) -> int:
    return (current_player + rel) & 0x3


def _abs_seat_color(seat_abs: int) -> str:
    return SEAT_COLORS[seat_abs & 0x3]


def draw_state_panel(
    ax: Axes,
    view: BoardView,
    *,
    current_player: int,
    pov_seat: int,
    step_idx: int,
    total_steps: int,
    action_id: int | None,
    action_desc: str | None = None,
    reward: float | None = None,
    done: bool = False,
    extra_lines: list[str] | None = None,
) -> None:
    """Render the side-panel text. `ax` should be a small axis with no spines."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    y = 0.98
    line_h = 0.018
    small_h = 0.015

    def put(text: str, *, size: int = 9, color: str = "#222",
            weight: str = "normal", indent: float = 0.02) -> None:
        nonlocal y
        ax.text(indent, y, text, ha="left", va="top", fontsize=size,
                color=color, fontweight=weight, family="monospace",
                transform=ax.transAxes)
        y -= line_h if size >= 9 else small_h

    # Header
    put(f"step {step_idx} / {total_steps - 1}", size=11, weight="bold")
    put(f"phase = {view.phase}", size=9)
    put(f"flag  = {view.flag}", size=9)
    roll_str = "—" if view.last_roll == 0 else str(view.last_roll)
    put(f"roll  = {roll_str}    turn = {int(view.turn_norm * 400)}", size=9)
    put(f"POV   = P{pov_seat}    free_roads = {view.free_roads}", size=9)
    if action_id is not None:
        adesc = action_desc or f"id={action_id}"
        put(f"action: {adesc}", size=9, color="#114")
    if reward is not None and reward != 0:
        put(f"reward: {reward:+.0f}", size=9,
            color="#0a0" if reward > 0 else "#a00")
    if done:
        put("DONE", size=11, weight="bold", color="#a00")
    y -= line_h * 0.4

    # Per-seat scoreboard
    put("seat | VP | hand | dev | kn | road | s/c/r | LR/LA",
        size=8, weight="bold")
    cur_abs = current_player & 0x3
    for p in view.players:
        seat_abs = _rel_to_abs(p.rel, current_player)
        marker = "*" if p.is_self else " "
        active = "<" if (seat_abs == cur_abs) else " "
        is_lr = "L" if view.longest_road_rel == p.rel else " "
        is_la = "A" if view.largest_army_rel == p.rel else " "
        line = (f"P{seat_abs}{marker}{active}  "
                f"{p.vp:2d}  {p.handsize:2d}    "
                f"{p.total_dev}   {p.knights_played}   "
                f"{p.road_length:2d}    "
                f"{p.settlements_left}/{p.cities_left}/{p.roads_left:2d}  "
                f"{is_lr}{is_la}")
        # Colour swatch
        sw_x = 0.005
        sw_w = 0.012
        ax.add_patch(Rectangle(
            (sw_x, y - small_h * 0.6), sw_w, small_h * 0.9,
            facecolor=_abs_seat_color(seat_abs), edgecolor="none",
            transform=ax.transAxes, clip_on=False,
        ))
        put(line, size=8, indent=0.025)
    y -= line_h * 0.4

    # Bank / dev deck
    put("bank: " + "  ".join(f"{n[0]}{c}" for n, c in zip(RES_NAMES, view.bank)),
        size=8)
    put("deck: " + "  ".join(f"{n[:2]}{c}" for n, c in zip(DEV_NAMES, view.dev_deck)),
        size=8)
    y -= line_h * 0.4

    # Self private
    put("YOU (POV)", size=9, weight="bold")
    put("hand: " + "  ".join(f"{n[0]}{c}" for n, c in zip(RES_NAMES, view.self_hand)),
        size=8)
    put("dev play: " + "  ".join(f"{n[:2]}{c}" for n, c
                                  in zip(DEV_NAMES, view.self_dev_playable)),
        size=8)
    pending_nonzero = any(view.self_dev_pending)
    if pending_nonzero:
        put("dev pend: " + "  ".join(f"{n[:2]}{c}" for n, c
                                      in zip(DEV_NAMES, view.self_dev_pending)),
            size=8)
    if view.self_dev_played_flag:
        put("played dev card this turn", size=8, color="#666")
    y -= line_h * 0.4

    # Trade scratch
    t = view.trade
    if t.proposer_rel is not None or any(t.give) or any(t.want):
        put("TRADE", size=9, weight="bold")
        if t.proposer_rel is not None:
            put(f"proposer: P{_rel_to_abs(t.proposer_rel, current_player)} "
                f"(rel{t.proposer_rel})", size=8)
        give = "  ".join(f"{n[0]}{c}" for n, c in zip(RES_NAMES, t.give) if c)
        want = "  ".join(f"{n[0]}{c}" for n, c in zip(RES_NAMES, t.want) if c)
        if give:
            put(f"give: {give}", size=8)
        if want:
            put(f"want: {want}", size=8)
        for opp_rel, resp in zip((1, 2, 3), t.responses):
            seat_abs = _rel_to_abs(opp_rel, current_player)
            put(f"  P{seat_abs}: {resp}", size=8)
        y -= line_h * 0.4

    if extra_lines:
        for line in extra_lines:
            put(line, size=8, color="#444")
