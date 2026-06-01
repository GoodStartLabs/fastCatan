"""Replay a recorded fastcatan game.

CLI:
    python -m ui.replay LOG [--start N] [--pov S] [--auto] [--delay F]
                            [--out DIR] [--no-mask] [--ids]

Modes
-----
Interactive (default): opens a matplotlib window. Keys:
    Right / Space   next step
    Left            prev step
    Home            first step
    End             last step
    a               toggle autoplay
    m               toggle mask overlay
    i               toggle ID labels
    1/2/3/4         set POV seat 0/1/2/3
    p               reset POV to step's current_player
    q / Escape      quit

Frame-dump (`--out DIR`): renders every step into DIR/frameNNNN.png with
no GUI. Useful for sharing or building videos.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

import fastcatan
from ui import obs_layout as L
from ui.action_names import info as action_info
from ui.board_render import RenderOptions, draw_board
from ui.log_format import GameLog, StepRecord, decode_snap, read_log
from ui.mask_view import (
    bucket_mask,
    chip_lines,
    draw_mask_overlay,
    spatial_summary,
)
from ui.obs_decoder import decode
from ui.state_panel import draw_state_panel


# ---------------------------------------------------------------------------
# Env walker
# ---------------------------------------------------------------------------

class Walker:
    """Holds a fastcatan env synced to a chosen log step.

    Seeks by loading the nearest snapshot and replaying forward; subsequent
    moves of `step_idx` to a later step replay incrementally (no reload).
    """

    def __init__(self, game: GameLog):
        self.game = game
        self.env = fastcatan.Env()
        self.env.reset(0)
        self.step_idx = -1
        self._obs_buf = np.zeros(L.TOTAL_WIDTH, dtype=np.float32)
        self._mask_buf = np.zeros(int(fastcatan.MASK_WORDS), dtype=np.uint64)
        self.seek(0)

    @property
    def total(self) -> int:
        return len(self.game.steps)

    def seek(self, target: int) -> None:
        target = max(0, min(self.total - 1, target))
        # Cheap forward step?
        if target > self.step_idx and (target - self.step_idx) < 60:
            for i in range(self.step_idx + 1, target + 1):
                self.env.step(int(self.game.steps[i].a))
            self.step_idx = target
            return
        # Snapshot-anchored seek.
        snap = self.game.nearest_snapshot(target)
        if snap is None:
            # No snap recorded before this step — fall back to full replay.
            self.env.reset(0)
            base = -1
        else:
            snap_idx, snap_raw = snap
            self.env.load_snapshot(snap_raw)
            base = snap_idx
        for i in range(base + 1, target + 1):
            self.env.step(int(self.game.steps[i].a))
        self.step_idx = target

    def view_state(self, pov: int):
        """Return (BoardView, mask, current_step_record, current_player)."""
        self.env.write_obs(int(pov), self._obs_buf)
        self.env.action_mask(self._mask_buf)
        view = decode(self._obs_buf)
        rec = self.game.steps[self.step_idx]
        return view, self._mask_buf.copy(), rec, int(self.env.current_player)


# ---------------------------------------------------------------------------
# Frame painter
# ---------------------------------------------------------------------------

@dataclass
class ViewOptions:
    show_ids: bool = False
    show_mask: bool = True


def paint_frame(
    fig: Figure,
    walker: Walker,
    pov: int,
    opts: ViewOptions,
    final_winner: Optional[int],
) -> None:
    fig.clear()
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.0], wspace=0.02)
    ax_board = fig.add_subplot(gs[0, 0])
    ax_panel = fig.add_subplot(gs[0, 1])

    view, mask, rec, current_player = walker.view_state(pov)

    title = (
        f"step {walker.step_idx}/{walker.total - 1}  "
        f"P{current_player} to act  phase={view.phase}  flag={view.flag}  "
        f"roll={view.last_roll or '-'}  POV=P{pov}"
    )
    ropts = RenderOptions(
        show_node_ids=opts.show_ids,
        show_edge_ids=opts.show_ids,
        show_hex_ids=opts.show_ids,
        title=title,
    )
    draw_board(ax_board, view, current_player=current_player, options=ropts)
    if opts.show_mask:
        draw_mask_overlay(ax_board, mask)

    # Build side panel
    buckets = bucket_mask(mask)
    action_desc = None
    if rec.a is not None:
        try:
            action_desc = f"{rec.a}  {action_info(rec.a).label}"
        except ValueError:
            action_desc = f"{rec.a}  ?"
    extra: list[str] = [spatial_summary(buckets)]
    extra.extend(chip_lines(buckets, per_cat_limit=4))
    if final_winner is not None and walker.step_idx == walker.total - 1:
        extra.append(f"winner: P{final_winner}")

    draw_state_panel(
        ax_panel,
        view,
        current_player=current_player,
        pov_seat=pov,
        step_idx=walker.step_idx,
        total_steps=walker.total,
        action_id=rec.a,
        action_desc=action_desc,
        reward=rec.r if rec.r else None,
        done=bool(rec.d),
        extra_lines=extra,
    )

    fig.suptitle("")  # rely on the board title


# ---------------------------------------------------------------------------
# Interactive driver
# ---------------------------------------------------------------------------

class InteractiveReplayer:
    def __init__(self, game: GameLog, *, start: int, pov: Optional[int],
                 view_opts: ViewOptions, auto: bool, delay: float):
        self.game = game
        self.walker = Walker(game)
        self.walker.seek(start)
        self.pov_override = pov
        self.view_opts = view_opts
        self.auto = auto
        self.delay = max(0.05, float(delay))
        self.fig: Figure = plt.figure(figsize=(15, 10))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self._timer = self.fig.canvas.new_timer(interval=int(self.delay * 1000))
        self._timer.add_callback(self._tick)
        self._closed = False
        self.redraw()
        if self.auto:
            self._timer.start()

    @property
    def pov(self) -> int:
        if self.pov_override is not None:
            return self.pov_override
        return int(self.game.steps[self.walker.step_idx].cp)

    def redraw(self) -> None:
        paint_frame(
            self.fig, self.walker, self.pov, self.view_opts,
            self.game.final.winner if self.game.final else None,
        )
        self.fig.canvas.draw_idle()

    def _tick(self) -> None:
        if self._closed:
            return
        if self.walker.step_idx >= self.walker.total - 1:
            self._timer.stop()
            self.auto = False
            return
        self.walker.seek(self.walker.step_idx + 1)
        self.redraw()

    def _on_close(self, _event) -> None:
        self._closed = True
        self._timer.stop()

    def _on_key(self, event) -> None:
        k = event.key
        if k in ("right", " ", "space"):
            self.walker.seek(self.walker.step_idx + 1); self.redraw()
        elif k == "left":
            self.walker.seek(self.walker.step_idx - 1); self.redraw()
        elif k == "home":
            self.walker.seek(0); self.redraw()
        elif k == "end":
            self.walker.seek(self.walker.total - 1); self.redraw()
        elif k == "a":
            self.auto = not self.auto
            if self.auto:
                self._timer.start()
            else:
                self._timer.stop()
        elif k == "m":
            self.view_opts.show_mask = not self.view_opts.show_mask
            self.redraw()
        elif k == "i":
            self.view_opts.show_ids = not self.view_opts.show_ids
            self.redraw()
        elif k in ("1", "2", "3", "4"):
            self.pov_override = int(k) - 1
            self.redraw()
        elif k == "p":
            self.pov_override = None
            self.redraw()
        elif k in ("q", "escape"):
            plt.close(self.fig)

    def run(self) -> None:
        plt.show()


# ---------------------------------------------------------------------------
# Frame-dump mode
# ---------------------------------------------------------------------------

def dump_frames(game: GameLog, out_dir: Path, *, pov: Optional[int],
                view_opts: ViewOptions, start: int, end: Optional[int]) -> int:
    walker = Walker(game)
    end = (walker.total - 1) if end is None else min(end, walker.total - 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 10))
    n = 0
    winner = game.final.winner if game.final else None
    for i in range(start, end + 1):
        walker.seek(i)
        pov_use = pov if pov is not None else int(game.steps[i].cp)
        paint_frame(fig, walker, pov_use, view_opts, winner)
        p = out_dir / f"frame{i:05d}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        n += 1
    plt.close(fig)
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Replay a recorded fastcatan game.")
    p.add_argument("log", help="path to .jsonl.gz log (bare name resolves against DEBUG/logs/)")
    p.add_argument("--start", type=int, default=0, help="seek to this step")
    p.add_argument("--end", type=int, default=None,
                   help="(--out only) last step to render")
    p.add_argument("--pov", type=int, default=None,
                   help="POV seat 0..3; default = step's current_player")
    p.add_argument("--auto", action="store_true", help="start in autoplay mode")
    p.add_argument("--delay", type=float, default=0.4,
                   help="seconds per frame in autoplay")
    p.add_argument("--out", default=None,
                   help="dump frames to this directory instead of opening a window")
    p.add_argument("--no-mask", action="store_true",
                   help="disable mask overlay")
    p.add_argument("--ids", action="store_true",
                   help="draw node/edge/hex IDs")
    args = p.parse_args(argv)

    log_path = Path(args.log)
    if not log_path.exists():
        # drag-drop convenience: a bare filename resolves against the drop folder
        drop = Path(__file__).resolve().parents[1] / "logs" / args.log
        if drop.exists():
            log_path = drop
    game = read_log(str(log_path))
    if not game.steps:
        print("log has no steps", file=sys.stderr)
        return 1
    view_opts = ViewOptions(show_ids=args.ids, show_mask=(not args.no_mask))

    if args.out:
        n = dump_frames(
            game, Path(args.out),
            pov=args.pov, view_opts=view_opts,
            start=max(0, args.start), end=args.end,
        )
        print(f"wrote {n} frames to {args.out}")
        return 0

    replayer = InteractiveReplayer(
        game, start=max(0, args.start), pov=args.pov,
        view_opts=view_opts, auto=args.auto, delay=args.delay,
    )
    replayer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
