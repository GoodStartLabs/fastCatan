"""Tests for action_codec — forward/reverse translation of stateless actions
and round-trip identities."""

from __future__ import annotations

import pytest

from catanatron import Color
from catanatron.models.enums import Action, ActionType

import fastcatan
from bridge import action_codec as C
from bridge import topology_map as T


_a = fastcatan.action


# ---------------------------------------------------------------------------
# Resource permutation
# ---------------------------------------------------------------------------


def test_res_permutation_bijection():
    assert len(C.RES_FAST_TO_CAT) == 5
    assert set(C.RES_FAST_TO_CAT) == {"WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"}
    for fi in range(5):
        cat_name = C.RES_FAST_TO_CAT[fi]
        assert C.RES_CAT_TO_FAST[cat_name] == fi


def test_freqdeck_round_trips():
    fast = [3, 1, 4, 1, 5]  # arbitrary
    cat = C.fast_freqdeck_to_cat(fast)
    back = C.cat_freqdeck_to_fast(cat)
    assert back == fast


# ---------------------------------------------------------------------------
# Forward decode coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fast_id", range(_a.SETTLE_BASE, _a.SETTLE_BASE + 54))
def test_decode_settle(fast_id):
    a = C.decode_simple(fast_id, Color.RED)
    assert a is not None
    assert a.action_type == ActionType.BUILD_SETTLEMENT
    assert a.color == Color.RED
    expected_node = T.NODE_FAST_TO_CAT[fast_id - _a.SETTLE_BASE]
    assert a.value == expected_node


@pytest.mark.parametrize("fast_id", range(_a.CITY_BASE, _a.CITY_BASE + 54))
def test_decode_city(fast_id):
    a = C.decode_simple(fast_id, Color.RED)
    assert a is not None
    assert a.action_type == ActionType.BUILD_CITY


@pytest.mark.parametrize("fast_id", range(_a.ROAD_BASE, _a.ROAD_BASE + 72))
def test_decode_road(fast_id):
    a = C.decode_simple(fast_id, Color.BLUE)
    assert a is not None
    assert a.action_type == ActionType.BUILD_ROAD
    expected_tuple = T.EDGE_FAST_TO_TUPLE[fast_id - _a.ROAD_BASE]
    assert a.value == expected_tuple


def test_decode_roll_and_end_turn():
    a = C.decode_simple(_a.ROLL_DICE, Color.RED)
    assert a == Action(Color.RED, ActionType.ROLL, None)

    a = C.decode_simple(_a.END_TURN, Color.RED)
    assert a == Action(Color.RED, ActionType.END_TURN, None)


def test_decode_discard():
    # fastcatan resource 0 = brick
    a = C.decode_simple(_a.DISCARD_BASE + 0, Color.RED)
    assert a == Action(Color.RED, ActionType.DISCARD_RESOURCE, "BRICK")
    a = C.decode_simple(_a.DISCARD_BASE + 1, Color.RED)
    assert a == Action(Color.RED, ActionType.DISCARD_RESOURCE, "WOOD")
    a = C.decode_simple(_a.DISCARD_BASE + 4, Color.RED)
    assert a == Action(Color.RED, ActionType.DISCARD_RESOURCE, "ORE")


def test_decode_move_robber_defers():
    for h in range(19):
        assert C.decode_simple(_a.MOVE_ROBBER_BASE + h, Color.RED) is None


def test_decode_steal_defers():
    for p in range(4):
        assert C.decode_simple(_a.STEAL_BASE + p, Color.RED) is None


def test_decode_maritime_trade():
    # fastcatan trade: give wool(2) get ore(4) -> offset = 2*5+4 = 14
    a = C.decode_simple(_a.TRADE_BASE + 14, Color.RED)
    assert a is not None
    assert a.action_type == ActionType.MARITIME_TRADE
    assert a.value == ("SHEEP", "SHEEP", "SHEEP", "SHEEP", "ORE")


def test_decode_dev_cards():
    assert C.decode_simple(_a.BUY_DEV, Color.RED) == \
        Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None)
    assert C.decode_simple(_a.PLAY_KNIGHT, Color.RED) == \
        Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None)
    assert C.decode_simple(_a.PLAY_ROAD_BUILDING, Color.RED) == \
        Action(Color.RED, ActionType.PLAY_ROAD_BUILDING, None)
    # YOP: offset = 1*5+3 = 8 -> (wood, wheat)
    a = C.decode_simple(_a.PLAY_YEAR_OF_PLENTY + 8, Color.RED)
    assert a == Action(Color.RED, ActionType.PLAY_YEAR_OF_PLENTY, ("WOOD", "WHEAT"))
    # Monopoly: offset 2 -> sheep
    a = C.decode_simple(_a.PLAY_MONOPOLY + 2, Color.RED)
    assert a == Action(Color.RED, ActionType.PLAY_MONOPOLY, "SHEEP")


def test_decode_trade_add_defers():
    for r in range(5):
        assert C.decode_simple(_a.TRADE_ADD_GIVE_BASE + r, Color.RED) is None
        assert C.decode_simple(_a.TRADE_ADD_WANT_BASE + r, Color.RED) is None


def test_decode_trade_simple_responses():
    assert C.decode_simple(_a.TRADE_ACCEPT, Color.RED) == \
        Action(Color.RED, ActionType.ACCEPT_TRADE, None)
    assert C.decode_simple(_a.TRADE_DECLINE, Color.RED) == \
        Action(Color.RED, ActionType.REJECT_TRADE, None)
    assert C.decode_simple(_a.TRADE_CANCEL, Color.RED) == \
        Action(Color.RED, ActionType.CANCEL_TRADE, None)


def test_decode_trade_open_defers():
    assert C.decode_simple(_a.TRADE_OPEN, Color.RED) is None


def test_decode_trade_confirm_defers():
    for p in range(4):
        assert C.decode_simple(_a.TRADE_CONFIRM_BASE + p, Color.RED) is None


def test_decode_move_robber_emit():
    a = C.decode_move_robber(9, None, Color.RED)
    assert a == Action(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), None))

    a = C.decode_move_robber(9, Color.BLUE, Color.RED)
    assert a == Action(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.BLUE))


def test_decode_offer_trade():
    give = [1, 0, 0, 0, 0]  # 1 brick (fast)
    want = [0, 1, 0, 0, 0]  # 1 lumber/wood (fast)
    a = C.decode_offer_trade(give, want, Color.RED)
    assert a.action_type == ActionType.OFFER_TRADE
    # Cat order: WOOD, BRICK, SHEEP, WHEAT, ORE
    expected_give = (0, 1, 0, 0, 0)  # 1 BRICK at cat idx 1
    expected_want = (1, 0, 0, 0, 0)  # 1 WOOD at cat idx 0
    assert a.value == expected_give + expected_want


# ---------------------------------------------------------------------------
# Reverse encode
# ---------------------------------------------------------------------------


def test_encode_settle():
    for node_fast in range(54):
        cat_node = T.NODE_FAST_TO_CAT[node_fast]
        action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, cat_node)
        seq = C.encode_to_fast_ids(action)
        assert seq == [_a.SETTLE_BASE + node_fast]


def test_encode_road():
    for edge_fast in range(72):
        tpl = T.EDGE_FAST_TO_TUPLE[edge_fast]
        action = Action(Color.RED, ActionType.BUILD_ROAD, tpl)
        assert C.encode_to_fast_ids(action) == [_a.ROAD_BASE + edge_fast]
        # Reverse ordering too
        action = Action(Color.RED, ActionType.BUILD_ROAD, (tpl[1], tpl[0]))
        assert C.encode_to_fast_ids(action) == [_a.ROAD_BASE + edge_fast]


def test_encode_move_robber_returns_hex_only():
    a = Action(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), None))
    seq = C.encode_to_fast_ids(a)
    assert seq == [_a.MOVE_ROBBER_BASE + 9]

    a = Action(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.BLUE))
    seq = C.encode_to_fast_ids(a)
    assert seq == [_a.MOVE_ROBBER_BASE + 9]
    # Bridge will append STEAL_BASE+seat itself.


def test_encode_maritime_trade():
    # Catanatron 4:1 trade: (X, X, X, X, Y) where X, Y are cat strings
    a = Action(Color.RED, ActionType.MARITIME_TRADE,
               ("SHEEP", "SHEEP", "SHEEP", "SHEEP", "ORE"))
    seq = C.encode_to_fast_ids(a)
    # fast: give wool(2) get ore(4) -> TRADE_BASE + 14
    assert seq == [_a.TRADE_BASE + 14]

    # Port trade (3:1): (X, X, X, None, Y)
    a = Action(Color.RED, ActionType.MARITIME_TRADE,
               ("ORE", "ORE", "ORE", None, "WOOD"))
    seq = C.encode_to_fast_ids(a)
    # fast: give ore(4) get lumber(1) -> TRADE_BASE + 4*5+1 = 21
    assert seq == [_a.TRADE_BASE + 21]


def test_encode_offer_trade_decomposes():
    # Give: 2 BRICK, want: 1 WHEAT
    give = (0, 2, 0, 0, 0)  # cat: 2 at idx 1 = BRICK
    want = (0, 0, 0, 1, 0)  # cat: 1 at idx 3 = WHEAT
    a = Action(Color.RED, ActionType.OFFER_TRADE, give + want)
    seq = C.encode_to_fast_ids(a)
    # fast: 2x ADD_GIVE_BASE+0 (brick) + 1x ADD_WANT_BASE+3 (wheat) + OPEN
    assert seq == [
        _a.TRADE_ADD_GIVE_BASE + 0,
        _a.TRADE_ADD_GIVE_BASE + 0,
        _a.TRADE_ADD_WANT_BASE + 3,
        _a.TRADE_OPEN,
    ]


def test_encode_dev_cards():
    assert C.encode_to_fast_ids(Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None)) == [_a.BUY_DEV]
    assert C.encode_to_fast_ids(Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None)) == [_a.PLAY_KNIGHT]
    assert C.encode_to_fast_ids(Action(Color.RED, ActionType.PLAY_ROAD_BUILDING, None)) == [_a.PLAY_ROAD_BUILDING]
    a = Action(Color.RED, ActionType.PLAY_YEAR_OF_PLENTY, ("WOOD", "WHEAT"))
    assert C.encode_to_fast_ids(a) == [_a.PLAY_YEAR_OF_PLENTY + 1*5 + 3]
    a = Action(Color.RED, ActionType.PLAY_MONOPOLY, "SHEEP")
    assert C.encode_to_fast_ids(a) == [_a.PLAY_MONOPOLY + 2]


def test_encode_discard():
    a = Action(Color.RED, ActionType.DISCARD_RESOURCE, "BRICK")
    assert C.encode_to_fast_ids(a) == [_a.DISCARD_BASE + 0]
    a = Action(Color.RED, ActionType.DISCARD_RESOURCE, "ORE")
    assert C.encode_to_fast_ids(a) == [_a.DISCARD_BASE + 4]


# ---------------------------------------------------------------------------
# Round-trip: decode_simple(encode(action)[0]) == action  (for stateless 1-ID actions)
# ---------------------------------------------------------------------------


def test_round_trip_simple_actions():
    cases = [
        Action(Color.RED, ActionType.BUILD_SETTLEMENT, T.NODE_FAST_TO_CAT[0]),
        Action(Color.RED, ActionType.BUILD_CITY, T.NODE_FAST_TO_CAT[10]),
        Action(Color.RED, ActionType.BUILD_ROAD, T.EDGE_FAST_TO_TUPLE[5]),
        Action(Color.RED, ActionType.ROLL, None),
        Action(Color.RED, ActionType.END_TURN, None),
        Action(Color.RED, ActionType.DISCARD_RESOURCE, "WHEAT"),
        Action(Color.RED, ActionType.MARITIME_TRADE,
               ("SHEEP", "SHEEP", "SHEEP", "SHEEP", "ORE")),
        Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None),
        Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None),
        Action(Color.RED, ActionType.PLAY_ROAD_BUILDING, None),
        Action(Color.RED, ActionType.PLAY_YEAR_OF_PLENTY, ("WHEAT", "ORE")),
        Action(Color.RED, ActionType.PLAY_MONOPOLY, "WOOD"),
        Action(Color.RED, ActionType.ACCEPT_TRADE, None),
        Action(Color.RED, ActionType.REJECT_TRADE, None),
        Action(Color.RED, ActionType.CANCEL_TRADE, None),
    ]
    for action in cases:
        seq = C.encode_to_fast_ids(action)
        assert len(seq) == 1, f"expected single ID for {action.action_type}, got {seq}"
        round_tripped = C.decode_simple(seq[0], Color.RED)
        assert round_tripped == action, \
            f"round-trip failure: {action} -> {seq[0]} -> {round_tripped}"
