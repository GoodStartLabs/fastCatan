"""Bidirectional action codec: fastcatan flat IDs <-> Catanatron Actions.

Pure-function layer where possible. Stateful translations (MOVE_ROBBER + STEAL
pairing, trade composition) are handled by the bridge orchestrator, which
calls into helpers here with the assembled state.

Forward (NN -> Catanatron):
    decode_simple(fast_id, color) -> Action | None
        None means "deferred — needs accompanying state to emit"
        (e.g. MOVE_ROBBER_BASE without STEAL_BASE, trade ADD_* before OPEN).

    decode_move_robber(hex_fast_id, victim_color_or_none, color) -> Action
    decode_offer_trade(give5_fast, want5_fast, color) -> Action
    decode_confirm_trade(give5_fast, want5_fast, partner_color, color) -> Action

Reverse (Catanatron action committed by engine -> fastcatan ID sequence):
    encode_to_fast_ids(action, mirror_helpers) -> list[int]
        Returns the fastcatan IDs that, when applied in order to a fastcatan
        Env in the same state, reproduce this catanatron action.

Resource ordering:
    fastcatan: [brick, lumber, wool, grain, ore] (indices 0..4)
    catanatron: [WOOD, BRICK, SHEEP, WHEAT, ORE] (RESOURCES list order)
"""

from __future__ import annotations

from typing import Optional, Sequence

from catanatron import Color
from catanatron.models.enums import Action, ActionType, RESOURCES

import fastcatan
from bridge import topology_map as T


_a = fastcatan.action

# ---------------------------------------------------------------------------
# Resource permutation
# ---------------------------------------------------------------------------

# fastcatan index -> catanatron resource string
RES_FAST_TO_CAT: list[str] = ["BRICK", "WOOD", "SHEEP", "WHEAT", "ORE"]
# catanatron resource string -> fastcatan index
RES_CAT_TO_FAST: dict[str, int] = {s: i for i, s in enumerate(RES_FAST_TO_CAT)}

# fastcatan index -> catanatron RESOURCES list position (0..4)
RES_FAST_TO_CAT_IDX: list[int] = [RESOURCES.index(s) for s in RES_FAST_TO_CAT]
# catanatron RESOURCES list position -> fastcatan index
RES_CAT_IDX_TO_FAST: list[int] = [
    RES_FAST_TO_CAT.index(s) for s in RESOURCES
]


def fast_freqdeck_to_cat(fast_freqdeck: Sequence[int]) -> tuple[int, ...]:
    """Convert a 5-vector indexed by fastcatan resource order into
    catanatron's RESOURCES list order."""
    out = [0] * 5
    for fi, count in enumerate(fast_freqdeck):
        out[RES_FAST_TO_CAT_IDX[fi]] = count
    return tuple(out)


def cat_freqdeck_to_fast(cat_freqdeck: Sequence[int]) -> list[int]:
    out = [0] * 5
    for ci, count in enumerate(cat_freqdeck):
        out[RES_CAT_IDX_TO_FAST[ci]] = count
    return out


# ---------------------------------------------------------------------------
# Forward: fastcatan ID -> Catanatron Action (stateless cases)
# ---------------------------------------------------------------------------


def decode_simple(fast_id: int, color: Color) -> Optional[Action]:
    """Translate a fastcatan ID to a Catanatron Action when no extra state
    is required.

    Returns None for IDs that need stateful pairing (MOVE_ROBBER, STEAL,
    trade composition actions). The orchestrator collects those into the
    helper-emitter calls below.
    """
    if _a.SETTLE_BASE <= fast_id < _a.SETTLE_BASE + T.NUM_NODES:
        node_fast = fast_id - _a.SETTLE_BASE
        return Action(color, ActionType.BUILD_SETTLEMENT, T.NODE_FAST_TO_CAT[node_fast])

    if _a.CITY_BASE <= fast_id < _a.CITY_BASE + T.NUM_NODES:
        node_fast = fast_id - _a.CITY_BASE
        return Action(color, ActionType.BUILD_CITY, T.NODE_FAST_TO_CAT[node_fast])

    if _a.ROAD_BASE <= fast_id < _a.ROAD_BASE + T.NUM_EDGES:
        edge_fast = fast_id - _a.ROAD_BASE
        return Action(color, ActionType.BUILD_ROAD, T.EDGE_FAST_TO_TUPLE[edge_fast])

    if fast_id == _a.ROLL_DICE:
        return Action(color, ActionType.ROLL, None)

    if fast_id == _a.END_TURN:
        return Action(color, ActionType.END_TURN, None)

    if _a.DISCARD_BASE <= fast_id < _a.DISCARD_BASE + 5:
        res_fast = fast_id - _a.DISCARD_BASE
        return Action(color, ActionType.DISCARD_RESOURCE, RES_FAST_TO_CAT[res_fast])

    if _a.MOVE_ROBBER_BASE <= fast_id < _a.MOVE_ROBBER_BASE + T.NUM_HEXES:
        return None  # paired with STEAL

    if _a.STEAL_BASE <= fast_id < _a.STEAL_BASE + 4:
        return None  # paired with MOVE_ROBBER

    if _a.TRADE_BASE <= fast_id < _a.TRADE_BASE + 25:
        # bank/port trade. fastcatan offset = give * 5 + get.
        off = fast_id - _a.TRADE_BASE
        give_f = off // 5
        get_f = off % 5
        # Catanatron MARITIME_TRADE value is 5-tuple ('giving4 of X', possibly
        # None placeholders for port trades, 'getting Y'). The exact tuple
        # depends on the player's port access; we emit a generic 4:1 shape
        # and rely on Catanatron's playable_actions filtering to pick the
        # better-ratio variant if available. Actual emission resolved against
        # playable_actions in the bridge (see resolve_maritime_trade).
        give_name = RES_FAST_TO_CAT[give_f]
        get_name = RES_FAST_TO_CAT[get_f]
        return Action(color, ActionType.MARITIME_TRADE,
                      (give_name, give_name, give_name, give_name, get_name))

    if fast_id == _a.BUY_DEV:
        return Action(color, ActionType.BUY_DEVELOPMENT_CARD, None)

    if fast_id == _a.PLAY_KNIGHT:
        return Action(color, ActionType.PLAY_KNIGHT_CARD, None)

    if fast_id == _a.PLAY_ROAD_BUILDING:
        return Action(color, ActionType.PLAY_ROAD_BUILDING, None)

    if _a.PLAY_YEAR_OF_PLENTY <= fast_id < _a.PLAY_YEAR_OF_PLENTY + 25:
        off = fast_id - _a.PLAY_YEAR_OF_PLENTY
        r1 = off // 5
        r2 = off % 5
        return Action(color, ActionType.PLAY_YEAR_OF_PLENTY,
                      (RES_FAST_TO_CAT[r1], RES_FAST_TO_CAT[r2]))

    if _a.PLAY_MONOPOLY <= fast_id < _a.PLAY_MONOPOLY + 5:
        res_fast = fast_id - _a.PLAY_MONOPOLY
        return Action(color, ActionType.PLAY_MONOPOLY, RES_FAST_TO_CAT[res_fast])

    # Trade composition: deferred
    if _a.TRADE_ADD_GIVE_BASE <= fast_id < _a.TRADE_ADD_GIVE_BASE + 5:
        return None
    if _a.TRADE_ADD_WANT_BASE <= fast_id < _a.TRADE_ADD_WANT_BASE + 5:
        return None
    if fast_id == _a.TRADE_OPEN:
        return None  # bridge calls decode_offer_trade with scratch
    if fast_id == _a.TRADE_ACCEPT:
        return Action(color, ActionType.ACCEPT_TRADE, None)
    if fast_id == _a.TRADE_DECLINE:
        return Action(color, ActionType.REJECT_TRADE, None)
    if _a.TRADE_CONFIRM_BASE <= fast_id < _a.TRADE_CONFIRM_BASE + 4:
        return None  # bridge calls decode_confirm_trade with scratch + partner
    if fast_id == _a.TRADE_CANCEL:
        return Action(color, ActionType.CANCEL_TRADE, None)

    raise ValueError(f"unmapped fastcatan action id {fast_id}")


def decode_move_robber(hex_fast: int, victim_color: Optional[Color], color: Color) -> Action:
    coord = T.FAST_HEX_TO_COORD[hex_fast]
    return Action(color, ActionType.MOVE_ROBBER, (coord, victim_color))


def decode_offer_trade(give_fast5: Sequence[int], want_fast5: Sequence[int],
                       color: Color) -> Action:
    give = fast_freqdeck_to_cat(give_fast5)
    want = fast_freqdeck_to_cat(want_fast5)
    return Action(color, ActionType.OFFER_TRADE, give + want)


def decode_confirm_trade(give_fast5: Sequence[int], want_fast5: Sequence[int],
                          partner_color: Color, color: Color) -> Action:
    give = fast_freqdeck_to_cat(give_fast5)
    want = fast_freqdeck_to_cat(want_fast5)
    return Action(color, ActionType.CONFIRM_TRADE, give + want + (partner_color,))


# ---------------------------------------------------------------------------
# Reverse: Catanatron Action committed -> list[fastcatan ID]
# ---------------------------------------------------------------------------


def encode_to_fast_ids(action: Action) -> list[int]:
    """Translate a committed Catanatron action into fastcatan ID(s).

    Contract: returns the *keying* ID(s) the bridge uses as mask keys in
    rep_map. Most actions yield length-1 lists. OFFER_TRADE expands to
    N+M+1 (ADD_GIVE × N, ADD_WANT × M, OPEN) for completeness.

    Incomplete cases (bridge resolves the missing seat/sub-ID itself):
      - MOVE_ROBBER: returns only [MOVE_ROBBER_BASE+hex]. The paired
        STEAL_BASE+seat ID is computed in `_decide_move_robber` from the
        bridge's color->seat map (see catanatron_bridge.py:154).
      - CONFIRM_TRADE: returns [TRADE_CONFIRM_BASE]. `_decide_trade_resolve`
        adds the partner seat offset directly (catanatron_bridge.py:208).

    Callers should use `fids[0]` as the mask key. The function is not a
    full state-reproducing encoder for these two action types — there is
    no mirror env to step, so the missing IDs are only needed for the
    bridge's local keying.
    """
    at = action.action_type
    v = action.value

    if at == ActionType.BUILD_SETTLEMENT:
        return [_a.SETTLE_BASE + T.NODE_CAT_TO_FAST[v]]

    if at == ActionType.BUILD_CITY:
        return [_a.CITY_BASE + T.NODE_CAT_TO_FAST[v]]

    if at == ActionType.BUILD_ROAD:
        return [_a.ROAD_BASE + T.EDGE_TUPLE_TO_FAST[tuple(v)]]

    if at == ActionType.ROLL:
        return [_a.ROLL_DICE]

    if at == ActionType.END_TURN:
        return [_a.END_TURN]

    if at == ActionType.DISCARD_RESOURCE:
        return [_a.DISCARD_BASE + RES_CAT_TO_FAST[v]]

    if at == ActionType.MOVE_ROBBER:
        coord, victim_color = v
        hex_fast = T.COORD_TO_FAST_HEX[coord]
        seq = [_a.MOVE_ROBBER_BASE + hex_fast]
        if victim_color is not None:
            # Need the fastcatan seat index of the victim. Bridge supplies it.
            # Encoded inline once seat layout is fixed at bridge init; here we
            # surface the raw color via a sentinel that the bridge resolves.
            # NOTE: This function returns the hex move only; the bridge
            # appends the STEAL_BASE+seat ID itself (it knows the seat order).
            pass
        return seq

    if at == ActionType.MARITIME_TRADE:
        # value = (g, g, g, g, t) with possible Nones in slots [2], [3].
        # Determine give resource by first non-None of value[0..3], get by value[4].
        give_name = next(s for s in v[:4] if s is not None)
        get_name = v[4]
        return [_a.TRADE_BASE + RES_CAT_TO_FAST[give_name] * 5 + RES_CAT_TO_FAST[get_name]]

    if at == ActionType.BUY_DEVELOPMENT_CARD:
        return [_a.BUY_DEV]

    if at == ActionType.PLAY_KNIGHT_CARD:
        return [_a.PLAY_KNIGHT]

    if at == ActionType.PLAY_ROAD_BUILDING:
        return [_a.PLAY_ROAD_BUILDING]

    if at == ActionType.PLAY_YEAR_OF_PLENTY:
        r1, r2 = v
        return [_a.PLAY_YEAR_OF_PLENTY + RES_CAT_TO_FAST[r1] * 5 + RES_CAT_TO_FAST[r2]]

    if at == ActionType.PLAY_MONOPOLY:
        return [_a.PLAY_MONOPOLY + RES_CAT_TO_FAST[v]]

    if at == ActionType.OFFER_TRADE:
        give_cat = v[:5]
        want_cat = v[5:10]
        give_fast = cat_freqdeck_to_fast(give_cat)
        want_fast = cat_freqdeck_to_fast(want_cat)
        seq: list[int] = []
        for ri, count in enumerate(give_fast):
            seq.extend([_a.TRADE_ADD_GIVE_BASE + ri] * count)
        for ri, count in enumerate(want_fast):
            seq.extend([_a.TRADE_ADD_WANT_BASE + ri] * count)
        seq.append(_a.TRADE_OPEN)
        return seq

    if at == ActionType.ACCEPT_TRADE:
        return [_a.TRADE_ACCEPT]
    if at == ActionType.REJECT_TRADE:
        return [_a.TRADE_DECLINE]
    if at == ActionType.CANCEL_TRADE:
        return [_a.TRADE_CANCEL]

    if at == ActionType.CONFIRM_TRADE:
        # Bridge supplies partner seat (last element of value is the Color).
        # Codec returns only the CONFIRM ID with a placeholder seat slot of 0;
        # bridge fixes up the seat index using its color->seat map.
        return [_a.TRADE_CONFIRM_BASE]  # bridge adds seat offset

    raise ValueError(f"unhandled catanatron action_type {at}")
