"""Catanatron Game -> fastcatan obs vector encoder.

Section-by-section mirror of src/catan/obs.cpp. Output: numpy float32 of
length fastcatan.OBS_SIZE, layout identical to fastcatan's write_obs.

Resource ordering note: fastcatan uses [brick, lumber, wool, grain, ore].
Catanatron's RESOURCES = [WOOD, BRICK, SHEEP, WHEAT, ORE]. All resource-
indexed fields are permuted via bridge.action_codec.RES_FAST_TO_CAT.

Phase / flag approximations:
    Phase: 0=INITIAL_PLACEMENT_1, 1=INITIAL_PLACEMENT_2, 2=MAIN, 3=ENDED
    Flag: 0=NONE 1=DISCARD 2=MOVE_ROBBER 3=ROBBER_STEAL 4=YOP 5=MONOPOLY
          6=PLACE_ROAD 7=TRADE_PENDING
Several fastcatan-internal flags (YOP/MONOPOLY scratch, ROBBER_STEAL split)
collapse to NONE under catanatron since catanatron emits those actions
in one shot.

Trade scratch (last 27 floats): catanatron has no `scratch` for trade
composition — trades exist only as committed OFFER_TRADE. The encoder
treats trade scratch as zeros UNLESS `state.is_resolving_trade` and
`state.current_trade` is populated, in which case we mirror the offered
freqdeck.
"""

from __future__ import annotations

import numpy as np

from catanatron import Color
from catanatron.game import Game
from catanatron.models.enums import ActionType, RESOURCES
from catanatron.models.tiles import LandTile, Port
from catanatron.state_functions import (
    get_dev_cards_in_hand,
    get_longest_road_length,
    get_played_dev_cards,
    player_has_rolled,
    player_key,
)

import fastcatan
from bridge import topology_map as T
from bridge.action_codec import RES_CAT_TO_FAST, RES_FAST_TO_CAT


# Fastcatan-aligned ordered lists.
DEV_TYPES_FAST = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]
# Note: fastcatan obs.cpp comment says "[resources(5), dev_playable(5),
# dev_bought_pending(5), dev_card_played]". The 5 dev types in fastcatan
# slot order match `DEV_TYPES_FAST` above.


def _seat_of(state, color: Color) -> int:
    return state.color_to_index[color]


def _relseat(self_seat: int, player_seat: int) -> int:
    return (player_seat + 4 - self_seat) & 0x3


def _player_block(state, seat: int, is_self: bool) -> list[float]:
    key = player_key(state, state.colors[seat])
    ps = state.player_state
    vp = ps[f"{key}_ACTUAL_VICTORY_POINTS"] if is_self else ps[f"{key}_VICTORY_POINTS"]
    handsize = sum(ps[f"{key}_{r}_IN_HAND"] for r in RESOURCES)
    total_dev = sum(ps[f"{key}_{d}_IN_HAND"] for d in DEV_TYPES_FAST)
    knights = ps[f"{key}_PLAYED_KNIGHT"]
    road_len = get_longest_road_length(state, state.colors[seat])
    settle_left = ps[f"{key}_SETTLEMENTS_AVAILABLE"]
    city_left = ps[f"{key}_CITIES_AVAILABLE"]
    road_left = ps[f"{key}_ROADS_AVAILABLE"]

    # Port ownership: fastcatan bits 0..4 = 2:1 port for each resource (in
    # fastcatan res order), bit 5 = 3:1 generic.
    port_resources = state.board.get_player_port_resources(state.colors[seat])
    ports_bits = [0.0] * 6
    for r in port_resources:
        if r is None:
            ports_bits[5] = 1.0
        else:
            ports_bits[RES_CAT_TO_FAST[r]] = 1.0

    # Discard remaining: catanatron tracks per-player via state.discard_counts.
    # When player is in DISCARD prompt and hasn't discarded the required #
    # yet, this is non-zero. Approximation: 0 if not discarding, else owed.
    discard_left = 0
    if state.is_discarding and state.current_color() == state.colors[seat]:
        # Required = floor(handsize / 2). discard_counts[seat] tracks how many
        # the player has already discarded this round.
        already = state.discard_counts[seat]
        owed = max(0, (handsize // 2) - already)
        discard_left = owed

    is_current = 1.0 if state.current_color() == state.colors[seat] else 0.0

    return [
        float(vp), float(handsize), float(total_dev),
        float(knights), float(road_len),
        float(settle_left), float(city_left), float(road_left),
        *ports_bits,
        float(discard_left),
        is_current,
    ]


def _onehot(idx: int, n: int) -> list[float]:
    out = [0.0] * n
    if 0 <= idx < n:
        out[idx] = 1.0
    return out


def _phase_value(state) -> int:
    """0/1/2/3 matching fastcatan."""
    # winning color lookup: 10+ VP including hidden VP cards.
    for c in state.colors:
        key = player_key(state, c)
        if state.player_state.get(f"{key}_ACTUAL_VICTORY_POINTS", 0) >= 10:
            return 3  # ENDED
    if state.is_initial_build_phase:
        placed_settlements = 0
        for c in state.colors:
            placed_settlements += len(
                state.buildings_by_color.get(c, {}).get("SETTLEMENT", [])
            )
        return 0 if placed_settlements < 4 else 1
    return 2  # MAIN


def _flag_value(state) -> int:
    if state.is_resolving_trade:
        return 7  # TRADE_PENDING
    if state.is_discarding:
        return 1
    if state.current_prompt.name == "MOVE_ROBBER":
        return 2
    if state.is_road_building and state.free_roads_available > 0:
        return 6  # PLACE_ROAD
    return 0  # NONE (catanatron collapses YOP/MONOPOLY/ROBBER_STEAL to single actions)


def _last_dice_roll(state) -> int:
    """Sum of last ROLL since last END_TURN, or 0 if not rolled yet this turn."""
    cur_color = state.current_color()
    cur_key = player_key(state, cur_color)
    if not state.player_state.get(f"{cur_key}_HAS_ROLLED", False):
        return 0
    # Walk action_records backwards to find this turn's roll.
    for rec in reversed(state.action_records):
        at = rec.action.action_type
        if at == ActionType.END_TURN:
            return 0
        if at == ActionType.ROLL and rec.result is not None:
            d1, d2 = rec.result
            return d1 + d2
    return 0


def _start_player_seat(state) -> int:
    """The seat of the first player to act in initial placement. With
    catanatron's standard reset, this is seat 0 (the first color)."""
    # We could walk action_records to find the first BUILD_SETTLEMENT
    # action; for the default Game ctor it's always seat 0.
    return 0


def encode_obs(game: Game, pov_color: Color,
               compose_scratch: "tuple[list[int], list[int]] | None" = None) -> np.ndarray:
    """Encode catanatron Game state into a fastcatan-formatted obs vector
    from `pov_color`'s perspective. Output: float32 array, len OBS_SIZE.

    `compose_scratch=(give_fast5, want_fast5)` overrides the trade scratch
    section with a bridge-maintained scratch (used during the compose
    sub-loop for OFFER_TRADE). When provided, the proposer slot is forced
    to self (relseat 0) and per-opponent responses are N/A.
    """
    state = game.state
    self_seat = _seat_of(state, pov_color)

    out: list[float] = []

    # --- Per-player blocks in relative seat order ---
    for rel in range(4):
        seat = (self_seat + rel) & 0x3
        out.extend(_player_block(state, seat, is_self=(rel == 0)))

    # --- Self private ---
    self_key = player_key(state, pov_color)
    ps = state.player_state
    # 5 resources in fastcatan order
    for r_fast in range(5):
        cat_name = RES_FAST_TO_CAT[r_fast]
        out.append(float(ps[f"{self_key}_{cat_name}_IN_HAND"]))
    # 5 dev cards owned (playable). We report total in hand; the "owned at
    # start" flag distinguishes playable-this-turn but fastcatan stores
    # the raw count, not playability.
    for dev in DEV_TYPES_FAST:
        out.append(float(ps.get(f"{self_key}_{dev}_IN_HAND", 0)))
    # 5 dev cards pending (bought this turn, can't play until next turn).
    # Catanatron's `_OWNED_AT_START` flag tracks this for K/M/RB/YOP only;
    # VP cards have no such flag (not pending-restricted).
    for dev in DEV_TYPES_FAST:
        owned = ps.get(f"{self_key}_{dev}_IN_HAND", 0)
        owned_at_start = ps.get(f"{self_key}_{dev}_OWNED_AT_START", True)
        pending = owned if (owned and not owned_at_start) else 0
        out.append(float(pending))
    # dev_card_played flag
    out.append(float(ps[f"{self_key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]))

    # --- Board nodes (54 * 8 = 432) ---
    buildings = state.board.buildings
    for n_fast in range(T.NUM_NODES):
        n_cat = T.NODE_FAST_TO_CAT[n_fast]
        b = buildings.get(n_cat)
        if b is None:
            out.extend([0.0] * 8)
            continue
        owner_color, kind = b
        owner_seat = state.color_to_index[owner_color]
        rel = _relseat(self_seat, owner_seat)
        is_settle = (kind == "SETTLEMENT")
        is_city = (kind == "CITY")
        for r in range(4):
            out.append(1.0 if (r == rel and is_settle) else 0.0)
            out.append(1.0 if (r == rel and is_city) else 0.0)

    # --- Board edges (72 * 4 = 288) ---
    roads = state.board.roads
    for e_fast in range(T.NUM_EDGES):
        cat_edge = T.EDGE_FAST_TO_TUPLE[e_fast]
        owner = roads.get(cat_edge) or roads.get((cat_edge[1], cat_edge[0]))
        if owner is None:
            out.extend([0.0] * 4)
            continue
        owner_seat = state.color_to_index[owner]
        rel = _relseat(self_seat, owner_seat)
        for r in range(4):
            out.append(1.0 if r == rel else 0.0)

    # --- Hex resources (19 * 6 one-hot, fastcatan res order + desert) ---
    # fastcatan resource codes: 0=brick, 1=lumber, 2=wool, 3=grain, 4=ore, 5=desert
    cat_to_fast_res = {"BRICK": 0, "WOOD": 1, "SHEEP": 2, "WHEAT": 3, "ORE": 4}
    for h_fast in range(T.NUM_HEXES):
        coord = T.FAST_HEX_TO_COORD[h_fast]
        tile = state.board.map.tiles[coord]
        assert isinstance(tile, LandTile)
        r = tile.resource
        idx = cat_to_fast_res[r] if r is not None else 5
        out.extend(_onehot(idx, 6))

    # --- Hex numbers (normalized /12) ---
    for h_fast in range(T.NUM_HEXES):
        coord = T.FAST_HEX_TO_COORD[h_fast]
        tile = state.board.map.tiles[coord]
        num = tile.number if tile.number is not None else 0
        out.append(num / 12.0)

    # --- Port types (9 * 6 one-hot) ---
    for p_fast in range(T.NUM_PORTS):
        coord = T.PORT_FAST_TO_COORD[p_fast]
        port = state.board.map.tiles[coord]
        assert isinstance(port, Port)
        r = port.resource
        idx = cat_to_fast_res[r] if r is not None else 5
        out.extend(_onehot(idx, 6))

    # --- Robber one-hot ---
    robber_coord = state.board.robber_coordinate
    robber_fast = T.COORD_TO_FAST_HEX.get(robber_coord, -1)
    out.extend(_onehot(robber_fast, T.NUM_HEXES))

    # --- Game state ---
    out.extend(_onehot(_phase_value(state), 4))
    out.extend(_onehot(_flag_value(state), 8))
    out.extend(_onehot(_last_dice_roll(state), 13))
    out.append(state.num_turns / 400.0)
    # Bank (5, fastcatan order)
    for r_fast in range(5):
        cat_idx = RESOURCES.index(RES_FAST_TO_CAT[r_fast])
        out.append(float(state.resource_freqdeck[cat_idx]))
    # Dev deck remaining (5, fastcatan dev order)
    deck = state.development_listdeck
    for dev in DEV_TYPES_FAST:
        out.append(float(deck.count(dev)))

    # Longest road owner (5 slots: self, +1, +2, +3, none)
    lr_color = None
    for c in state.colors:
        if state.player_state[f"{player_key(state, c)}_HAS_ROAD"]:
            lr_color = c
            break
    if lr_color is None:
        out.extend(_onehot(4, 5))
    else:
        out.extend(_onehot(_relseat(self_seat, state.color_to_index[lr_color]), 5))

    # Largest army owner
    la_color = None
    for c in state.colors:
        if state.player_state[f"{player_key(state, c)}_HAS_ARMY"]:
            la_color = c
            break
    if la_color is None:
        out.extend(_onehot(4, 5))
    else:
        out.extend(_onehot(_relseat(self_seat, state.color_to_index[la_color]), 5))

    # Start player (4 slots, relseat)
    out.extend(_onehot(_relseat(self_seat, _start_player_seat(state)), 4))

    # Free roads remaining
    out.append(float(state.free_roads_available))

    # --- Trade scratch (27) ---
    # Trade proposer (5)
    if compose_scratch is not None:
        give_fast, want_fast = compose_scratch
        out.extend(_onehot(0, 5))  # proposer = self (relseat 0)
        out.extend(float(x) for x in give_fast)
        out.extend(float(x) for x in want_fast)
        for _ in range(3):
            out.extend(_onehot(3, 4))  # N/A for all opponents
        arr = np.array(out, dtype=np.float32)
        assert arr.shape == (fastcatan.OBS_SIZE,), \
            f"obs size mismatch: got {arr.shape[0]}, expected {fastcatan.OBS_SIZE}"
        return arr

    if state.is_resolving_trade and state.current_trade is not None:
        # current_trade: tuple(give[5_cat], want[5_cat], proposer_color)
        # Format may vary; conservatively skip.
        # Catanatron's current_trade structure documented as a tuple.
        # For now: parse defensively.
        ct = state.current_trade
        proposer_seat = None
        if isinstance(ct, tuple) and len(ct) >= 11:
            # current_trade = (*OFFER_TRADE.value, current_turn_index).
            # The trailing element is the proposer's seat index (int).
            v = ct[10]
            if isinstance(v, int):
                proposer_seat = v
            else:
                # Defensive: older builds may store the proposer Color here.
                proposer_seat = state.color_to_index.get(v)
        if proposer_seat is not None:
            out.extend(_onehot(_relseat(self_seat, proposer_seat), 5))
        else:
            out.extend(_onehot(4, 5))
        # Give (5, fastcatan order)
        give_cat = list(ct[:5])
        for r_fast in range(5):
            cat_idx = RESOURCES.index(RES_FAST_TO_CAT[r_fast])
            out.append(float(give_cat[cat_idx]))
        # Want (5, fastcatan order)
        want_cat = list(ct[5:10])
        for r_fast in range(5):
            cat_idx = RESOURCES.index(RES_FAST_TO_CAT[r_fast])
            out.append(float(want_cat[cat_idx]))
    else:
        out.extend(_onehot(4, 5))  # proposer = none
        out.extend([0.0] * 5)      # give
        out.extend([0.0] * 5)      # want

    # Per-opponent response (3 * 4)
    # Catanatron `acceptees` is tuple of bool per player slot for current trade.
    # 4 states per opponent: 0=PENDING, 1=ACCEPT, 2=DECLINE, 3=N/A.
    # Without explicit per-color tracking, simplest mapping:
    for rel in range(1, 4):
        seat = (self_seat + rel) & 0x3
        if state.is_resolving_trade and seat < len(state.acceptees):
            v = 1 if state.acceptees[seat] else 0  # PENDING or ACCEPT only
        else:
            v = 3  # N/A
        out.extend(_onehot(v, 4))

    arr = np.array(out, dtype=np.float32)
    assert arr.shape == (fastcatan.OBS_SIZE,), \
        f"obs size mismatch: got {arr.shape[0]}, expected {fastcatan.OBS_SIZE}"
    return arr
