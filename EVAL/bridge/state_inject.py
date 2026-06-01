"""Serialize a catanatron ``Game`` state into a byte-exact fastcatan
``GameState`` + ``BoardLayout`` (``state_mirror.CSnapshot``) for injection via
``Env.load_snapshot``.

This is the inverse direction of ``obs_encoder`` (which reads catanatron into a
fastcatan *obs vector*); here we read catanatron into fastcatan's *full
internal state*. It reuses obs_encoder's verified catanatron-reading helpers so
the two stay consistent.

Used by ``tests/test_differential.py`` to drive both engines through the same
action stream and compare full post-state every ply.

Field encodings mirror ``include/state.hpp``:
  - node[]: bits 0-1 level (0 empty / 1 settlement / 2 city), bits 2-4 owner seat.
  - edge[]: owner seat 0..3, 0xFF (NO_PLAYER) empty.
  - resources / bank / dev ordered in fastcatan order via the obs_encoder maps.
"""
from __future__ import annotations

import fastcatan as fc
from catanatron.models.enums import RESOURCES
from catanatron.state_functions import (
    get_longest_road_length,
    player_key,
)

from bridge import state_mirror as M
from bridge import topology_map as T
from bridge.action_codec import RES_FAST_TO_CAT, RES_CAT_TO_FAST
from bridge.obs_encoder import (
    DEV_TYPES_FAST,
    _dev_bought_this_turn,
    _flag_value,
    _last_dice_roll,
    _per_opp_responses,
    _phase_value,
    _start_player_seat,
)

NO_PLAYER = 0xFF

# catanatron resource string -> fastcatan board resource code
#   0=brick 1=lumber 2=wool 3=grain 4=ore  (desert -> 5)
CAT_TO_FAST_RES = {"BRICK": 0, "WOOD": 1, "SHEEP": 2, "WHEAT": 3, "ORE": 4}
# fastcatan resource idx -> position in catanatron RESOURCES list
FAST_TO_CAT_IDX = [RESOURCES.index(name) for name in RES_FAST_TO_CAT]


def _fill_board(board: M.CBoardLayout, state) -> None:
    cmap = state.board.map
    for h in range(T.NUM_HEXES):
        coord = T.FAST_HEX_TO_COORD[h]
        tile = cmap.tiles[coord]
        r = tile.resource
        board.hex_resource[h] = CAT_TO_FAST_RES[r] if r is not None else 5
        board.hex_number[h] = tile.number if tile.number is not None else 0
    for p in range(T.NUM_PORTS):
        coord = T.PORT_FAST_TO_COORD[p]
        port = cmap.tiles[coord]
        r = port.resource
        board.port_type[p] = CAT_TO_FAST_RES[r] if r is not None else 5


def _bonus_owner_seat(state, key_suffix: str) -> int:
    for c in state.colors:
        if state.player_state[f"{player_key(state, c)}_{key_suffix}"]:
            return state.color_to_index[c]
    return NO_PLAYER


def build_cgs(game, actor_seat: int | None = None):
    """Build (CGameState, CBoardLayout) from a catanatron Game.

    ``actor_seat`` sets ``current_player`` and ``discarding_player`` — pass the
    seat of the action being replayed (``action.color``); defaults to
    catanatron's ``current_color()``.
    """
    state = game.state
    gs = M.CGameState()
    board = M.CBoardLayout()
    _fill_board(board, state)

    ps = state.player_state

    # --- Board nodes ---
    buildings = state.board.buildings
    for n_fast in range(T.NUM_NODES):
        n_cat = T.NODE_FAST_TO_CAT[n_fast]
        b = buildings.get(n_cat)
        if b is None:
            gs.node[n_fast] = 0
            continue
        owner_color, kind = b
        owner = state.color_to_index[owner_color]
        level = M.NODE_SETTLEMENT if kind == "SETTLEMENT" else M.NODE_CITY
        gs.node[n_fast] = M.node_pack(level, owner)

    # --- Board edges ---
    roads = state.board.roads
    for e_fast in range(T.NUM_EDGES):
        cat_edge = T.EDGE_FAST_TO_TUPLE[e_fast]
        owner = roads.get(cat_edge)
        if owner is None:
            owner = roads.get((cat_edge[1], cat_edge[0]))
        gs.edge[e_fast] = NO_PLAYER if owner is None else state.color_to_index[owner]

    # --- Robber ---
    gs.robber_hex = T.COORD_TO_FAST_HEX[state.board.robber_coordinate]

    # --- Turn / phase ---
    gs.dice_roll = _last_dice_roll(state)
    gs.turn_count = state.num_turns & 0xFFFF
    gs.phase = _phase_value(state)
    gs.flag = _flag_value(state)
    gs.start_player = _start_player_seat(state)

    cur_seat = state.color_to_index[state.current_color()]
    actor = cur_seat if actor_seat is None else actor_seat
    gs.current_player = actor
    gs.discarding_player = actor
    gs.dev_card_played = 1 if ps[
        f"{player_key(state, state.colors[actor])}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"
    ] else 0
    gs.free_roads_remaining = int(state.free_roads_available)

    # --- Per-player ---
    for seat in range(4):
        color = state.colors[seat]
        key = player_key(state, color)
        bought = _dev_bought_this_turn(state, color)

        for r_fast in range(5):
            cat_name = RES_FAST_TO_CAT[r_fast]
            gs.player_resources[seat][r_fast] = ps[f"{key}_{cat_name}_IN_HAND"]
        for d, dev in enumerate(DEV_TYPES_FAST):
            total = ps.get(f"{key}_{dev}_IN_HAND", 0)
            pend = bought.get(dev, 0)
            gs.player_dev[seat][d] = total - pend
            gs.player_dev_bought_this_turn[seat][d] = pend

        gs.player_vp[seat] = ps[f"{key}_ACTUAL_VICTORY_POINTS"]
        gs.player_vp_without_dev[seat] = ps[f"{key}_VICTORY_POINTS"]
        gs.player_handsize[seat] = sum(
            ps[f"{key}_{r}_IN_HAND"] for r in RESOURCES
        )
        gs.player_total_dev[seat] = sum(
            ps.get(f"{key}_{d}_IN_HAND", 0) for d in DEV_TYPES_FAST
        )
        gs.player_knights_played[seat] = ps[f"{key}_PLAYED_KNIGHT"]
        gs.player_road_length[seat] = get_longest_road_length(state, color)
        gs.player_settlement_count[seat] = ps[f"{key}_SETTLEMENTS_AVAILABLE"]
        gs.player_city_count[seat] = ps[f"{key}_CITIES_AVAILABLE"]
        gs.player_road_count[seat] = ps[f"{key}_ROADS_AVAILABLE"]

        # Port bitmask: bits 0..4 = 2:1 by fast resource, bit 5 = 3:1 generic.
        bits = 0
        for r in state.board.get_player_port_resources(color):
            bits |= (1 << 5) if r is None else (1 << RES_CAT_TO_FAST[r])
        gs.player_ports[seat] = bits

        gs.player_discard_remaining[seat] = (
            state.discard_counts[seat] if state.is_discarding else 0
        )

        # Longest-road component membership: catanatron's connected_components
        # (list of node sets) is the exact ground-truth member set, including
        # now-enemy nodes the player reached before the opponent settled.
        member = 0
        for comp in state.board.connected_components[color]:
            for n_cat in comp:
                member |= (1 << T.NODE_CAT_TO_FAST[n_cat])
        gs.road_node_member[seat] = member

    # --- Awards ---
    gs.longest_road_owner = _bonus_owner_seat(state, "HAS_ROAD")
    gs.largest_army_owner = _bonus_owner_seat(state, "HAS_ARMY")

    # --- Bank / dev deck ---
    bank_cat = state.resource_freqdeck
    for r_fast in range(5):
        gs.bank[r_fast] = bank_cat[FAST_TO_CAT_IDX[r_fast]]
    deck = state.development_listdeck
    for d, dev in enumerate(DEV_TYPES_FAST):
        gs.dev_deck[d] = deck.count(dev)

    # --- Trade scratch ---
    gs.trade_proposer = NO_PLAYER
    if state.is_resolving_trade and state.current_trade is not None:
        ct = state.current_trade
        proposer = ct[10] if (isinstance(ct, tuple) and len(ct) >= 11
                              and isinstance(ct[10], int)) else None
        if proposer is not None:
            gs.trade_proposer = proposer
        for r_fast in range(5):
            ci = FAST_TO_CAT_IDX[r_fast]
            gs.trade_give[r_fast] = int(ct[ci])
            gs.trade_want[r_fast] = int(ct[5 + ci])
        # trade_response: 2 bits/player, LSB-first (0 PENDING,1 ACCEPT,2 DECLINE,3 N/A)
        resp = _per_opp_responses(state, actor)
        packed = 0
        for seat in range(4):
            packed |= (resp[seat] & 0x3) << (2 * seat)
        gs.trade_response = packed

    return gs, board


def inject(env, game, actor_seat: int | None = None,
           rng_state: tuple[int, int, int, int] | None = None) -> None:
    """Build catanatron's state and load it into a fastcatan Env.

    ``rng_state`` (4×uint32) forces the next ``bounded()`` draw — set it before
    a stochastic action (ROLL/BUY_DEV/STEAL) to reproduce catanatron's outcome.
    """
    gs, board = build_cgs(game, actor_seat)
    if rng_state is not None:
        gs.rng[:] = rng_state
    snap = M.CSnapshot()
    snap.gs = gs
    snap.board = board
    env.load_snapshot(M.to_bytes(snap))


def read_fast(env) -> M.CGameState:
    """Parse the env's current state back into a CGameState for comparison."""
    return M.parse_snapshot(env.snapshot()).gs
