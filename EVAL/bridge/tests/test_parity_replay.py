"""Cross-engine parity replay.

For each ply of a catanatron game, the bridge's obs_encoder produces a
fastcatan-shaped obs vector. Decoding that vector should round-trip back to
the SAME catanatron ground-truth state.

This test runs catanatron games step-by-step and, after every ply, decodes
the encoded obs for one POV and compares structured fields against
catanatron's authoritative state. Any mismatch is appended to a JSONL diff
log (`bridge/tests/parity_diffs.jsonl`) and the test fails.

What's compared per step (POV: a chosen seat that stays fixed for the game):
  - robber_hex                         (fast hex ID 0..18)
  - bank[5]                            (fast resource order)
  - per-player VP                      (relseat-mapped)
  - per-player longest road length
  - per-player knights played
  - self hand[5]                       (fast resource order)
  - longest road owner relseat / NO
  - largest army owner relseat / NO

What's NOT compared (different encoding semantics or known-lossy):
  - dev card draws (private to engine RNG)
  - trade scratch (different lifecycle in catanatron)
  - flag (fastcatan has more flags than catanatron exposes)

Run a single seed:  pytest bridge/tests/test_parity_replay.py -k seed_42
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from catanatron import Color
from catanatron.game import Game
from catanatron.models.player import RandomPlayer
from catanatron.models.enums import RESOURCES

import fastcatan
from bridge import topology_map as T
from bridge.action_codec import RES_FAST_TO_CAT, RES_CAT_TO_FAST
from bridge.obs_encoder import encode_obs
from ui.obs_decoder import decode


# Fast resource idx -> catanatron RESOURCES idx
FAST_TO_CAT_IDX = [RESOURCES.index(name) for name in RES_FAST_TO_CAT]

# Sink for diffs. Overwritten on each test session start (per pytest run).
DIFFS_PATH = Path(__file__).parent / "parity_diffs.jsonl"

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


# ---------------------------------------------------------------------------
# Diff capture
# ---------------------------------------------------------------------------

class DiffLogger:
    def __init__(self, path: Path, seed: int, pov_color: Color):
        self.path = path
        self.seed = seed
        self.pov = str(pov_color)
        self.records: list[dict] = []

    def add(self, step: int, last_action: str, field: str,
            decoded: Any, expected: Any, extra: dict | None = None):
        rec = {
            "seed": self.seed,
            "pov": self.pov,
            "step": step,
            "last_action": last_action,
            "field": field,
            "decoded": _jsonable(decoded),
            "expected": _jsonable(expected),
        }
        if extra:
            rec["extra"] = extra
        self.records.append(rec)

    def flush(self):
        if not self.records:
            return
        with self.path.open("a") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")


def _jsonable(x):
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# ---------------------------------------------------------------------------
# Ground truth probes (catanatron state -> fast-shaped expected values)
# ---------------------------------------------------------------------------

def _expected_bank(game) -> list[int]:
    """Catanatron resource_freqdeck reordered to fastcatan order."""
    cat = game.state.resource_freqdeck
    return [int(cat[FAST_TO_CAT_IDX[r]]) for r in range(5)]


def _expected_robber_hex(game) -> int:
    coord = game.state.board.robber_coordinate
    return int(T.COORD_TO_FAST_HEX[coord])


def _expected_self_hand_fast(game, color: Color) -> list[int]:
    """Self hand in fast resource order."""
    seat = game.state.color_to_index[color]
    ps = game.state.player_state
    cat_hand = [
        ps[f"P{seat}_WOOD_IN_HAND"],
        ps[f"P{seat}_BRICK_IN_HAND"],
        ps[f"P{seat}_SHEEP_IN_HAND"],
        ps[f"P{seat}_WHEAT_IN_HAND"],
        ps[f"P{seat}_ORE_IN_HAND"],
    ]
    return [int(cat_hand[FAST_TO_CAT_IDX[r]]) for r in range(5)]


def _player_seat_for_relseat(self_seat: int, relseat: int) -> int:
    return (self_seat + relseat) % 4


def _expected_vp(game, abs_seat: int, is_self: bool) -> int:
    """Self uses ACTUAL_VICTORY_POINTS; others see public VP."""
    ps = game.state.player_state
    key = f"P{abs_seat}_ACTUAL_VICTORY_POINTS" if is_self else f"P{abs_seat}_VICTORY_POINTS"
    return int(ps[key])


def _expected_lr_length(game, abs_seat: int) -> int:
    return int(game.state.player_state[f"P{abs_seat}_LONGEST_ROAD_LENGTH"])


def _expected_knights(game, abs_seat: int) -> int:
    return int(game.state.player_state[f"P{abs_seat}_PLAYED_KNIGHT"])


def _expected_bonus_owner_relseat(game, self_seat: int, key_suffix: str) -> int | None:
    """Returns relseat of bonus holder, or None if no one holds it."""
    for s in range(4):
        if game.state.player_state[f"P{s}_{key_suffix}"]:
            return (s - self_seat) % 4
    return None


# ---------------------------------------------------------------------------
# Per-step parity check
# ---------------------------------------------------------------------------

def _check_step(game, pov_color: Color, step: int, last_action: str,
                logger: DiffLogger) -> int:
    """Encode obs from pov, decode, compare to catanatron ground truth.
    Returns count of mismatches discovered this step."""
    obs = encode_obs(game, pov_color)
    view = decode(obs)
    self_seat = game.state.color_to_index[pov_color]

    n_fail = 0

    # robber_hex
    exp_robber = _expected_robber_hex(game)
    if view.robber_hex != exp_robber:
        logger.add(step, last_action, "robber_hex", view.robber_hex, exp_robber)
        n_fail += 1

    # bank
    exp_bank = _expected_bank(game)
    if list(view.bank) != exp_bank:
        logger.add(step, last_action, "bank", list(view.bank), exp_bank)
        n_fail += 1

    # self hand
    exp_hand = _expected_self_hand_fast(game, pov_color)
    if list(view.self_hand) != exp_hand:
        logger.add(step, last_action, "self_hand", list(view.self_hand), exp_hand)
        n_fail += 1

    # per-player VP (relseat-mapped)
    for relseat in range(4):
        abs_seat = _player_seat_for_relseat(self_seat, relseat)
        exp_vp = _expected_vp(game, abs_seat, is_self=(relseat == 0))
        got_vp = view.players[relseat].vp
        if got_vp != exp_vp:
            logger.add(step, last_action, f"player[{relseat}].vp", got_vp, exp_vp,
                       extra={"abs_seat": abs_seat, "is_self": relseat == 0})
            n_fail += 1

    # per-player longest road length
    for relseat in range(4):
        abs_seat = _player_seat_for_relseat(self_seat, relseat)
        exp_lr = _expected_lr_length(game, abs_seat)
        got_lr = view.players[relseat].road_length
        if got_lr != exp_lr:
            logger.add(step, last_action, f"player[{relseat}].longest_road",
                       got_lr, exp_lr, extra={"abs_seat": abs_seat})
            n_fail += 1

    # per-player knights played
    for relseat in range(4):
        abs_seat = _player_seat_for_relseat(self_seat, relseat)
        exp_k = _expected_knights(game, abs_seat)
        got_k = view.players[relseat].knights_played
        if got_k != exp_k:
            logger.add(step, last_action, f"player[{relseat}].knights",
                       got_k, exp_k, extra={"abs_seat": abs_seat})
            n_fail += 1

    # bonus owners
    exp_lr_owner = _expected_bonus_owner_relseat(game, self_seat, "HAS_ROAD")
    if view.longest_road_rel != exp_lr_owner:
        logger.add(step, last_action, "longest_road_rel",
                   view.longest_road_rel, exp_lr_owner)
        n_fail += 1

    exp_la_owner = _expected_bonus_owner_relseat(game, self_seat, "HAS_ARMY")
    if view.largest_army_rel != exp_la_owner:
        logger.add(step, last_action, "largest_army_rel",
                   view.largest_army_rel, exp_la_owner)
        n_fail += 1

    return n_fail


# ---------------------------------------------------------------------------
# Driver: one full parity replay
# ---------------------------------------------------------------------------

def _replay_one(seed: int, pov_color: Color, max_ticks: int = 1500) -> tuple[int, int]:
    """Returns (steps_checked, total_mismatches). Appends to DIFFS_PATH on diffs."""
    players = [RandomPlayer(c) for c in COLORS]
    game = Game(players, seed=seed)

    logger = DiffLogger(DIFFS_PATH, seed=seed, pov_color=pov_color)
    steps = 0
    mismatches = 0
    last_action = "<init>"

    # Check initial state before any ply
    mismatches += _check_step(game, pov_color, step=0, last_action=last_action, logger=logger)
    steps += 1

    for _ in range(max_ticks):
        if game.winning_color() is not None:
            break
        action_rec = game.play_tick()
        last_action = str(action_rec)
        mismatches += _check_step(game, pov_color, step=steps,
                                  last_action=last_action, logger=logger)
        steps += 1

    logger.flush()
    return steps, mismatches


# ---------------------------------------------------------------------------
# Pytest entrypoints
# ---------------------------------------------------------------------------

def setup_module(_m):
    """Truncate the diff log at the start of each pytest session."""
    if DIFFS_PATH.exists():
        DIFFS_PATH.unlink()


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
@pytest.mark.parametrize("pov_color", [Color.RED, Color.WHITE])
def test_parity_replay(seed, pov_color):
    steps, mismatches = _replay_one(seed=seed, pov_color=pov_color)
    assert mismatches == 0, (
        f"{mismatches} parity mismatches across {steps} steps. "
        f"See {DIFFS_PATH} for the diff log."
    )
