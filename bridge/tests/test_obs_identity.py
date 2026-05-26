"""Obs-identity: C++ write_obs vs bridge encode_obs must agree bit-for-bit.

The simulator (src/catan/obs.cpp) and the catanatron-eval encoder
(bridge/obs_encoder.py) are two independent implementations of the SAME obs
layout. If they drift — a field in the wrong slot, a missed normalization
divisor, a different one-hot — an agent trained in fastcatan sees a different
obs at catanatron eval and the bridge is silently wrong.

This test pins them together. For each catanatron state we serialize it into a
fastcatan GameState (bridge.state_inject), run the C++ write_obs from each
player's POV, and require it to equal encode_obs(game, color) element-wise.

It exercises layout + normalization parity. The underlying *field values*
(phase, dev pending, longest road, ...) are validated against catanatron's raw
state separately by test_differential.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from catanatron import Color
from catanatron.game import Game
from catanatron.models.player import RandomPlayer

import fastcatan as fc
from bridge import state_inject as SI
from bridge import state_mirror as M
from bridge.obs_encoder import encode_obs
from ui import obs_layout as L

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]

# (name, slice) spans over the obs vector, for readable mismatch diagnostics.
_SPANS = [
    *[(f"player_block[{i}]", s) for i, s in enumerate(L.PLAYER_BLOCKS)],
    ("self_res", L.SELF_RES), ("self_dev_playable", L.SELF_DEV_PLAYABLE),
    ("self_dev_pending", L.SELF_DEV_PENDING), ("self_played_flag", L.SELF_DEV_PLAYED_FLAG),
    ("nodes", L.NODES), ("edges", L.EDGES), ("hex_res", L.HEX_RES),
    ("hex_nums", L.HEX_NUMS), ("port_types", L.PORT_TYPES), ("robber", L.ROBBER),
    ("phase", L.PHASE), ("flag", L.FLAG), ("last_roll", L.LAST_ROLL),
    ("turn_norm", L.TURN_NORM), ("bank", L.BANK), ("dev_deck", L.DEV_DECK),
    ("lr_owner", L.LR_OWNER), ("la_owner", L.LA_OWNER), ("start_player", L.START_PLAYER),
    ("free_roads", L.FREE_ROADS), ("trade_proposer", L.TRADE_PROPOSER),
    ("trade_give", L.TRADE_GIVE), ("trade_want", L.TRADE_WANT),
    ("trade_responses", L.TRADE_RESPONSES),
]


def _slot_name(idx: int) -> str:
    for name, s in _SPANS:
        if s.start <= idx < s.stop:
            return f"{name}[+{idx - s.start}]"
    return "?"


def _fast_obs(env, game, pov_color) -> np.ndarray:
    """write_obs from fastcatan after injecting catanatron's state."""
    # current_player must reflect whose turn it is (drives the is_current bit),
    # independent of the POV we encode from.
    cur_seat = game.state.color_to_index[game.state.current_color()]
    gs, board = SI.build_cgs(game, actor_seat=cur_seat)
    snap = M.CSnapshot()
    snap.gs = gs
    snap.board = board
    env.load_snapshot(M.to_bytes(snap))
    buf = np.zeros(fc.OBS_SIZE, dtype=np.float32)
    env.write_obs(game.state.color_to_index[pov_color], buf)
    return buf


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_obs_identity(seed):
    random.seed(seed)
    np.random.seed(seed)
    game = Game([RandomPlayer(c) for c in COLORS], seed=seed)
    env = fc.Env()
    env.reset(0)

    checks = 0
    for _ in range(120):
        if game.winning_color() is not None:
            break
        # The obs is only ever consumed from the *current* player's POV — in
        # training write_obs(LEARNER_SEAT) is read on the learner's turn, and at
        # eval encode_obs is called for the player about to act. Some fields are
        # current-player-global (e.g. dev_card_played is a single GameState
        # bool, not per-player), so they only line up from the current POV.
        # The current player rotates across seats, so this still covers all
        # relseat orderings over a game.
        for color in [game.state.current_color()]:
            fast = _fast_obs(env, game, color)
            enc = encode_obs(game, color)
            if not np.allclose(fast, enc, atol=1e-5):
                d = np.abs(fast - enc)
                idx = int(np.argmax(d))
                bad = np.flatnonzero(d > 1e-5)
                slots = sorted({_slot_name(int(i)) for i in bad})
                raise AssertionError(
                    f"seed {seed} {color}: obs mismatch in {len(bad)} slots {slots}; "
                    f"worst idx {idx} ({_slot_name(idx)}) "
                    f"cpp={fast[idx]:.4f} bridge={enc[idx]:.4f}"
                )
            checks += 1
        game.play_tick()

    assert checks > 0
