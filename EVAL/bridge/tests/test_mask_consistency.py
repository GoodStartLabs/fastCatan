"""Mask consistency: every catanatron playable_action (excluding trades)
must reverse-map to a fastcatan ID sequence via encode_to_fast_ids.

Trade actions (OFFER_TRADE / CONFIRM_TRADE / ACCEPT / REJECT / CANCEL) are
explicitly skipped — the bridge MVP runs with --no-player-trading on the
catanatron side. MOVE_ROBBER encodes to hex-only (length 1); the bridge
pairs in a STEAL via its 2-step sub-policy.
"""

from __future__ import annotations

import pytest

from catanatron import Color
from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import ActionType
from catanatron.models.player import RandomPlayer

from bridge.action_codec import encode_to_fast_ids


COLORS = [Color.RED, Color.WHITE, Color.BLUE, Color.ORANGE]

TRADE_TYPES = {
    ActionType.OFFER_TRADE,
    ActionType.CONFIRM_TRADE,
    ActionType.ACCEPT_TRADE,
    ActionType.REJECT_TRADE,
    ActionType.CANCEL_TRADE,
}


def _walk_states(seed: int, ticks: int):
    """Yield (game, playable_actions) at each tick of a random game."""
    g = Game([RandomPlayer(c) for c in COLORS], seed=seed)
    for _ in range(ticks):
        pa = generate_playable_actions(g.state)
        yield g, pa
        if not pa:
            return
        g.play_tick()


@pytest.mark.parametrize("seed", [1, 7, 42, 99, 1337])
def test_every_non_trade_action_maps(seed):
    """For 300 ticks of 5 different random games, every non-trade
    playable_action must produce a non-empty fast-ID sequence."""
    for _, pa in _walk_states(seed, ticks=300):
        for a in pa:
            if a.action_type in TRADE_TYPES:
                continue
            seq = encode_to_fast_ids(a)
            assert seq, f"empty fast-id seq for {a}"
            assert all(isinstance(x, int) and x >= 0 for x in seq), seq


def test_move_robber_returns_hex_id():
    """MOVE_ROBBER cat Actions encode to a single MOVE_ROBBER_BASE+hex ID
    (the STEAL pairing is added by the bridge's sub-policy, not the codec)."""
    import fastcatan
    _a = fastcatan.action
    seen_any = False
    for _, pa in _walk_states(seed=7, ticks=500):
        for a in pa:
            if a.action_type != ActionType.MOVE_ROBBER:
                continue
            seen_any = True
            seq = encode_to_fast_ids(a)
            assert len(seq) == 1
            fid = seq[0]
            assert _a.MOVE_ROBBER_BASE <= fid < _a.MOVE_ROBBER_BASE + 19
        if seen_any:
            return
    pytest.skip("no MOVE_ROBBER actions encountered in 500 ticks of seed=7")
