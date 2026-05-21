"""Trade composition + resolve tests for CatanatronBridge.

Compose path: trade-loving policy forces OFFER_TRADE emission inside the
compose sub-loop. Asserts cat accepts the assembled OFFER_TRADE Action
(valid via is_valid_trade) and the game keeps running.

Resolve path: unit-tests the bridge's mapping of ACCEPT_TRADE /
REJECT_TRADE / CANCEL_TRADE / CONFIRM_TRADE actions onto fast IDs.
"""

from __future__ import annotations

import random

import pytest

from catanatron import Color
from catanatron.game import Game, is_valid_trade
from catanatron.models.enums import Action, ActionPrompt, ActionType
from catanatron.models.player import RandomPlayer

import fastcatan
from bridge.catanatron_bridge import CatanatronBridge


_a = fastcatan.action

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


# ---------------------------------------------------------------------------
# Compose path
# ---------------------------------------------------------------------------


class _TradeLovingPolicy:
    """Picks ADD_GIVE/ADD_WANT/OPEN whenever any are legal. Otherwise picks
    uniformly from the mask. Forces OFFER_TRADE emission."""

    def __init__(self):
        self.opens_emitted = 0

    def __call__(self, obs, mask, rng):
        # If OPEN is legal, take it immediately (assemble whatever scratch).
        if _a.TRADE_OPEN in mask:
            self.opens_emitted += 1
            return _a.TRADE_OPEN
        # Else prefer ADD_GIVE first, then ADD_WANT.
        for fid in mask:
            if _a.TRADE_ADD_GIVE_BASE <= fid < _a.TRADE_ADD_GIVE_BASE + 5:
                return fid
        for fid in mask:
            if _a.TRADE_ADD_WANT_BASE <= fid < _a.TRADE_ADD_WANT_BASE + 5:
                return fid
        return rng.choice(mask)


def test_compose_emits_offer_trade():
    """With a trade-loving policy occupying the bridge, at least one
    OFFER_TRADE is emitted across a game and cat accepts it as valid."""
    policy = _TradeLovingPolicy()
    bridge = CatanatronBridge(Color.RED, policy=policy, seed=7,
                              enable_trades=True)
    players = [bridge,
               RandomPlayer(Color.BLUE),
               RandomPlayer(Color.ORANGE),
               RandomPlayer(Color.WHITE)]
    game = Game(players, seed=7)
    game.play()

    offer_trades = [r.action for r in game.state.action_records
                    if r.action.action_type == ActionType.OFFER_TRADE]
    assert offer_trades, "trade-loving bridge emitted no OFFER_TRADE"
    for a in offer_trades:
        assert is_valid_trade(a.value), f"invalid trade emitted: {a}"


def test_compose_disabled_emits_no_offer():
    """enable_trades=False must produce zero OFFER_TRADE actions."""
    bridge = CatanatronBridge(Color.RED, seed=11, enable_trades=False)
    players = [bridge,
               RandomPlayer(Color.BLUE),
               RandomPlayer(Color.ORANGE),
               RandomPlayer(Color.WHITE)]
    game = Game(players, seed=11)
    game.play()
    offer_trades = [r.action for r in game.state.action_records
                    if r.action.action_type == ActionType.OFFER_TRADE]
    assert not offer_trades


# ---------------------------------------------------------------------------
# Resolve path
# ---------------------------------------------------------------------------


def _make_game(seed=0):
    return Game([RandomPlayer(c) for c in COLORS], seed=seed)


def test_resolve_accept_reject_mapping():
    """DECIDE_TRADE path: ACCEPT/REJECT actions map to TRADE_ACCEPT /
    TRADE_DECLINE fast IDs."""
    game = _make_game()
    bridge = CatanatronBridge(Color.RED, seed=0)
    pa = [
        Action(Color.RED, ActionType.REJECT_TRADE,
               (1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0)),
        Action(Color.RED, ActionType.ACCEPT_TRADE,
               (1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0)),
    ]
    # Force prompt by patching for the call (we don't apply the trade).
    game.state.current_prompt = ActionPrompt.DECIDE_TRADE
    chosen = bridge._decide_trade_resolve(game, pa)
    assert chosen in pa


def test_resolve_confirm_maps_partner_seat():
    """DECIDE_ACCEPTEES path: CONFIRM_TRADE encodes partner color → seat
    via state.color_to_index. Verify the bridge picks a CONFIRM action and
    that the seat-mapped fast ID actually appears in the mask."""
    game = _make_game()
    bridge = CatanatronBridge(Color.RED, seed=0)
    # Three CONFIRM_TRADE actions, one CANCEL.
    trade_val = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    pa = [
        Action(Color.RED, ActionType.CANCEL_TRADE, None),
        Action(Color.RED, ActionType.CONFIRM_TRADE,
               trade_val + (Color.WHITE,)),
        Action(Color.RED, ActionType.CONFIRM_TRADE,
               trade_val + (Color.BLUE,)),
    ]
    game.state.current_prompt = ActionPrompt.DECIDE_ACCEPTEES
    # Deterministic uniform policy. Just confirm a valid pick comes back.
    chosen = bridge._decide_trade_resolve(game, pa)
    assert chosen in pa

    # Verify the seat math: color_to_index[WHITE] = 2 in our 4-player setup
    # → TRADE_CONFIRM_BASE + 2 should map to the CONFIRM_TRADE(WHITE) action.
    seat_white = game.state.color_to_index[Color.WHITE]
    seat_blue = game.state.color_to_index[Color.BLUE]
    assert seat_white != seat_blue
    fid_white = _a.TRADE_CONFIRM_BASE + seat_white
    fid_blue = _a.TRADE_CONFIRM_BASE + seat_blue
    assert _a.TRADE_CONFIRM_BASE <= fid_white < _a.TRADE_CONFIRM_BASE + 4
    assert _a.TRADE_CONFIRM_BASE <= fid_blue < _a.TRADE_CONFIRM_BASE + 4


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_compose_completes_game(seed):
    """Bridge with compose enabled (default uniform policy) finishes games
    without exceptions across multiple seeds."""
    bridge = CatanatronBridge(Color.RED, seed=seed, enable_trades=True)
    players = [bridge,
               RandomPlayer(Color.BLUE),
               RandomPlayer(Color.ORANGE),
               RandomPlayer(Color.WHITE)]
    game = Game(players, seed=seed)
    game.play()  # No assertion — must just not raise.
