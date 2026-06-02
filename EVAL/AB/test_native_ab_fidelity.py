"""Fidelity gate: fastcatan's native alpha-beta == Catanatron's AlphaBetaPlayer.

The native player (``Env.ab_value`` / ``Env.ab_decide``, C++ src/catan/search.cpp)
is a port of catanatron.players.minimax.AlphaBetaPlayer + value.base_fn. This test
proves the port is faithful, using the bridge to inject byte-identical states into
both engines:

  1. ``test_value_function_matches_base_fn`` — ab_value equals base_fn(DEFAULT_WEIGHTS)
     to machine precision on every seat of many random positions.
  2. ``test_optimal_value_agreement_depth1`` — on deterministic, 1:1-action-space
     decisions, Catanatron's chosen move achieves *exactly* the best depth-1 value
     fastcatan's search finds (so any raw action difference is a pure value tie).

Run (anaconda interpreter, catanatron + bridge on the path):
    PYTHONPATH=EVAL /home/sinan/anaconda3/bin/python -m pytest \
        EVAL/AB/test_native_ab_fidelity.py -q

Two deliberate (documented) deviations from Catanatron's expectimax — BUY_DEV
forks the true remaining deck, robber-steal forks the victim's real hand (both
more correct than the reference) — are out of scope here: the value function is
deviation-free, and the depth-1 agreement is measured on deterministic decisions.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from catanatron import Color
from catanatron.game import Game
from catanatron.models.player import RandomPlayer
from catanatron.models.enums import ActionType
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import base_fn, DEFAULT_WEIGHTS

import fastcatan as fc
from bridge import state_inject as SI
from bridge import state_mirror as M
from bridge.action_codec import encode_to_fast_ids

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]

# Action types whose fastcatan action-space structure is NOT 1:1 with
# catanatron's (robber decomposes into hex + separate steal; discard is
# per-card; maritime auto-resolves the port ratio; p2p trades use a compose
# protocol catanatron's AB can't even search). Excluded from move agreement.
_NON_11 = {
    ActionType.MOVE_ROBBER, ActionType.DISCARD_RESOURCE, ActionType.MARITIME_TRADE,
    ActionType.OFFER_TRADE, ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE,
    ActionType.CONFIRM_TRADE, ActionType.CANCEL_TRADE,
    # ROLL / BUY_DEV are chance actions — excluded from the *deterministic*
    # depth-1 value check (their expectimax value involves forks).
    ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD,
}


def _inject_bytes(game, env, actor_seat=None):
    gs, board = SI.build_cgs(game, actor_seat=actor_seat)
    snap = M.CSnapshot()
    snap.gs = gs
    snap.board = board
    data = M.to_bytes(snap)
    env.load_snapshot(data)
    return data


def test_value_function_matches_base_fn():
    catval = base_fn(DEFAULT_WEIGHTS)
    env = fc.Env()
    env.reset(0)
    worst = 0.0
    n = 0
    for seed in range(12):
        random.seed(seed)
        np.random.seed(seed)
        game = Game([RandomPlayer(c) for c in COLORS], seed=seed)
        for tick in range(400):
            if game.winning_color() is not None:
                break
            game.play_tick()
            if tick % 5:
                continue
            _inject_bytes(game, env)
            for seat in range(4):
                cv = catval(game, game.state.colors[seat])  # seat <-> state.colors[seat]
                fv = env.ab_value(seat)
                worst = max(worst, abs(fv - cv) / max(abs(cv), 1.0))
                n += 1
    assert n > 1000, f"too few samples ({n})"
    assert worst < 1e-9, f"ab_value diverges from base_fn (worst rel error {worst:.3e})"


def test_optimal_value_agreement_depth1():
    """Catanatron's depth-1 pick achieves fastcatan's best depth-1 value (ties OK)."""
    env = fc.Env()
    scratch = fc.Env()
    env.reset(0)
    decisions = 0
    cat_optimal = 0  # catanatron's pick == fastcatan's best value
    my_optimal = 0   # fastcatan's pick == fastcatan's best value
    for seed in range(12):
        random.seed(seed)
        np.random.seed(seed)
        game = Game([RandomPlayer(c) for c in COLORS], seed=seed)
        cti = game.state.color_to_index
        for _ in range(300):
            if game.winning_color() is not None:
                break
            pa = game.playable_actions
            types = {a.action_type for a in pa}
            if len(pa) >= 2 and not (types & _NON_11):
                color = game.state.current_color()
                seat = cti[color]
                cat_a = AlphaBetaPlayer(color, depth=1, prunning=False).decide(game, pa)
                data = _inject_bytes(game, env, actor_seat=seat)
                fa = env.ab_decide(seat, 1, False)

                def val(aid):
                    scratch.load_snapshot(data)
                    scratch.step(int(aid))
                    return scratch.ab_value(seat)

                best = max(val(fid) for a in pa for fid in encode_to_fast_ids(a))
                cat_val = val(encode_to_fast_ids(cat_a)[0])
                my_val = val(fa)
                decisions += 1
                if abs(cat_val - best) / max(abs(best), 1.0) < 1e-9:
                    cat_optimal += 1
                if abs(my_val - best) / max(abs(best), 1.0) < 1e-9:
                    my_optimal += 1
            game.play_tick()
    assert decisions > 100, f"too few clean decisions ({decisions})"
    assert my_optimal == decisions, (
        f"fastcatan search argmax wrong on {decisions - my_optimal}/{decisions}")
    assert cat_optimal == decisions, (
        f"catanatron pick beat fastcatan best on {decisions - cat_optimal}/{decisions} "
        f"(value function or search diverged)")
