"""Decode a flat fastcatan obs vector into a structured `BoardView`.

The obs vector is POV-relative (`self` is relseat 0). Decoding preserves
the relseat convention; callers that need absolute seats translate using
the recorded `current_player` from the log:

    abs_seat = (current_player + relseat) % 4

Relseat 4 (used as the "none" slot in 5-wide one-hots like LR/LA owner and
trade proposer) is reported as `None`.

Phase / flag / dev-card names follow the engine. See `src/catan/obs.cpp`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import fastcatan
from ui import obs_layout as L


# ---------------------------------------------------------------------------
# Enums / labels
# ---------------------------------------------------------------------------

PHASE_NAMES = ("INITIAL_PLACEMENT_1", "INITIAL_PLACEMENT_2", "MAIN", "ENDED")
FLAG_NAMES = (
    "NONE", "DISCARD", "MOVE_ROBBER", "ROBBER_STEAL",
    "YEAR_OF_PLENTY", "MONOPOLY", "PLACE_ROAD", "TRADE_PENDING",
)
# Hex resource one-hot (6 slots): 0=brick, 1=lumber, 2=wool, 3=grain, 4=ore, 5=desert
HEX_RES_NAMES = ("brick", "lumber", "wool", "grain", "ore", "desert")
# Port one-hot (6 slots): res 0..4 + generic 3:1 in slot 5
PORT_NAMES = ("brick", "lumber", "wool", "grain", "ore", "3:1")
DEV_NAMES = ("KNIGHT", "VP", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY")
RES_NAMES = ("brick", "lumber", "wool", "grain", "ore")
# Trade response slots (4-wide one-hot per opponent).
TRADE_RESP_NAMES = ("PENDING", "ACCEPT", "DECLINE", "N/A")

# Normalization divisors — MUST match src/catan/obs.cpp (namespace norm) and
# bridge/obs_encoder.py. Count fields in the obs are stored as value/divisor;
# decode multiplies back to recover the integer game value.
N_VP = 10.0
N_HAND = 25.0
N_DEV = 10.0
N_KNIGHTS = 10.0
N_ROADLEN = 15.0
N_SETTLE = 5.0
N_CITY = 4.0
N_ROAD = 15.0
N_DISCARD = 10.0
N_RES = 19.0
N_BANK = 19.0
N_DEVDECK = 25.0
N_FREEROADS = 2.0
N_TRADE = 19.0


def _unq(x: float, divisor: float) -> int:
    """De-normalize a stored obs value (value/divisor) back to its integer."""
    return int(round(float(x) * divisor))


# ---------------------------------------------------------------------------
# Structured view
# ---------------------------------------------------------------------------

@dataclass
class PlayerBlock:
    """Per-player POV-relative summary. `is_self` is True for relseat 0."""
    rel: int             # 0..3
    is_self: bool
    vp: int              # full VP for self; public-only for opponents
    handsize: int
    total_dev: int
    knights_played: int
    road_length: int
    settlements_left: int
    cities_left: int
    roads_left: int
    ports: tuple[bool, ...]   # 6 bits: 2:1 by resource (5) + generic 3:1
    discard_owed: int
    is_current: bool


@dataclass
class NodeState:
    """`owner_rel` is relseat (0=self, 1/2/3 opponents); None if empty."""
    owner_rel: Optional[int]
    kind: Optional[str]       # "settlement" / "city" / None


@dataclass
class TradeView:
    proposer_rel: Optional[int]    # None when no active trade
    give: tuple[int, ...]          # 5 resource counts, fastcatan order
    want: tuple[int, ...]
    responses: tuple[str, ...]     # length 3, one per opponent relseat 1..3


@dataclass
class BoardView:
    # Sizes pinned at construction time for `__repr__` clarity.
    nodes: list[NodeState]
    edges: list[Optional[int]]                # owner_rel or None
    hex_resources: list[str]                  # length 19
    hex_numbers: list[int]                    # length 19 (0 for desert)
    port_types: list[str]                     # length 9
    robber_hex: Optional[int]                 # fastcatan hex id

    phase: str
    flag: str
    last_roll: int                            # 0 if not rolled yet this turn
    turn_norm: float                          # /400

    bank: tuple[int, ...]                     # 5
    dev_deck: tuple[int, ...]                 # 5

    longest_road_rel: Optional[int]
    largest_army_rel: Optional[int]
    start_player_rel: int
    free_roads: int

    self_hand: tuple[int, ...]                # 5
    self_dev_playable: tuple[int, ...]        # 5
    self_dev_pending: tuple[int, ...]         # 5
    self_dev_played_flag: bool

    players: list[PlayerBlock] = field(default_factory=list)
    trade: TradeView = field(default_factory=lambda: TradeView(None, (0,)*5, (0,)*5, ("N/A","N/A","N/A")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _argmax_onehot(view: np.ndarray) -> int:
    """Index of the 1.0 slot; -1 if all zero."""
    nz = np.flatnonzero(view > 0.5)
    if nz.size == 0:
        return -1
    return int(nz[0])


def _decode_player_block(block: np.ndarray, rel: int) -> PlayerBlock:
    assert block.shape == (L.PLAYER_BLOCK_W,)
    return PlayerBlock(
        rel=rel,
        is_self=(rel == 0),
        vp=_unq(block[L.PB_VP], N_VP),
        handsize=_unq(block[L.PB_HANDSIZE], N_HAND),
        total_dev=_unq(block[L.PB_TOTAL_DEV], N_DEV),
        knights_played=_unq(block[L.PB_KNIGHTS], N_KNIGHTS),
        road_length=_unq(block[L.PB_ROAD_LEN], N_ROADLEN),
        settlements_left=_unq(block[L.PB_SETTLE_LEFT], N_SETTLE),
        cities_left=_unq(block[L.PB_CITY_LEFT], N_CITY),
        roads_left=_unq(block[L.PB_ROAD_LEFT], N_ROAD),
        ports=tuple(bool(x) for x in block[L.PB_PORTS]),
        discard_owed=_unq(block[L.PB_DISCARD_LEFT], N_DISCARD),
        is_current=bool(block[L.PB_IS_CURRENT] > 0.5),
    )


def _decode_nodes(slab: np.ndarray) -> list[NodeState]:
    assert slab.shape == (L.NUM_NODES * L.NODE_CH,)
    out: list[NodeState] = []
    for n in range(L.NUM_NODES):
        ch = slab[n * L.NODE_CH:(n + 1) * L.NODE_CH]
        if not np.any(ch > 0.5):
            out.append(NodeState(None, None))
            continue
        # Layout: per relseat r=0..3, pairs (settle, city).
        for r in range(L.NUM_PLAYERS):
            s, c = ch[2 * r], ch[2 * r + 1]
            if s > 0.5:
                out.append(NodeState(r, "settlement"))
                break
            if c > 0.5:
                out.append(NodeState(r, "city"))
                break
        else:
            out.append(NodeState(None, None))  # unreachable
    return out


def _decode_edges(slab: np.ndarray) -> list[Optional[int]]:
    assert slab.shape == (L.NUM_EDGES * L.EDGE_CH,)
    out: list[Optional[int]] = []
    for e in range(L.NUM_EDGES):
        ch = slab[e * L.EDGE_CH:(e + 1) * L.EDGE_CH]
        idx = _argmax_onehot(ch)
        out.append(idx if idx >= 0 else None)
    return out


def _decode_hex_res(slab: np.ndarray) -> list[str]:
    assert slab.shape == (L.NUM_HEXES * 6,)
    out: list[str] = []
    for h in range(L.NUM_HEXES):
        idx = _argmax_onehot(slab[h * 6:(h + 1) * 6])
        out.append(HEX_RES_NAMES[idx] if 0 <= idx < 6 else "unknown")
    return out


def _decode_port_types(slab: np.ndarray) -> list[str]:
    assert slab.shape == (L.NUM_PORTS * 6,)
    out: list[str] = []
    for p in range(L.NUM_PORTS):
        idx = _argmax_onehot(slab[p * 6:(p + 1) * 6])
        out.append(PORT_NAMES[idx] if 0 <= idx < 6 else "unknown")
    return out


def _decode_owner_5slot(slab: np.ndarray) -> Optional[int]:
    """5-slot one-hot [self, +1, +2, +3, none] → relseat or None."""
    idx = _argmax_onehot(slab)
    if idx < 0 or idx == 4:
        return None
    return idx


def _decode_trade_responses(slab: np.ndarray) -> tuple[str, ...]:
    """12 floats = 3 × 4-slot one-hot. Returns 3 status strings."""
    assert slab.shape == (12,)
    out: list[str] = []
    for opp in range(3):
        idx = _argmax_onehot(slab[opp * 4:(opp + 1) * 4])
        out.append(TRADE_RESP_NAMES[idx] if 0 <= idx < 4 else "?")
    return tuple(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode(obs: np.ndarray) -> BoardView:
    """Decode a single 1D obs vector (length `fastcatan.OBS_SIZE`)."""
    if obs.ndim != 1 or obs.shape[0] != L.TOTAL_WIDTH:
        raise ValueError(
            f"expected 1D obs of length {L.TOTAL_WIDTH}, got shape {obs.shape}"
        )
    o = obs.astype(np.float32, copy=False)

    players = [
        _decode_player_block(o[s.start:s.stop], rel=i)
        for i, s in enumerate(L.PLAYER_BLOCKS)
    ]

    self_hand = tuple(_unq(x, N_RES) for x in o[L.SELF_RES.start:L.SELF_RES.stop])
    self_dev_playable = tuple(
        _unq(x, N_DEV) for x in o[L.SELF_DEV_PLAYABLE.start:L.SELF_DEV_PLAYABLE.stop]
    )
    self_dev_pending = tuple(
        _unq(x, N_DEV) for x in o[L.SELF_DEV_PENDING.start:L.SELF_DEV_PENDING.stop]
    )
    self_played_flag = bool(o[L.SELF_DEV_PLAYED_FLAG.start] > 0.5)

    nodes = _decode_nodes(o[L.NODES.start:L.NODES.stop])
    edges = _decode_edges(o[L.EDGES.start:L.EDGES.stop])
    hex_res = _decode_hex_res(o[L.HEX_RES.start:L.HEX_RES.stop])
    hex_nums = [int(round(x * 12.0)) for x in o[L.HEX_NUMS.start:L.HEX_NUMS.stop]]
    port_types = _decode_port_types(o[L.PORT_TYPES.start:L.PORT_TYPES.stop])

    robber_idx = _argmax_onehot(o[L.ROBBER.start:L.ROBBER.stop])
    robber = robber_idx if robber_idx >= 0 else None

    phase_idx = _argmax_onehot(o[L.PHASE.start:L.PHASE.stop])
    flag_idx = _argmax_onehot(o[L.FLAG.start:L.FLAG.stop])
    phase = PHASE_NAMES[phase_idx] if 0 <= phase_idx < 4 else "?"
    flag = FLAG_NAMES[flag_idx] if 0 <= flag_idx < 8 else "?"

    roll_idx = _argmax_onehot(o[L.LAST_ROLL.start:L.LAST_ROLL.stop])
    last_roll = roll_idx if roll_idx >= 0 else 0

    turn_norm = float(o[L.TURN_NORM.start])
    bank = tuple(_unq(x, N_BANK) for x in o[L.BANK.start:L.BANK.stop])
    dev_deck = tuple(_unq(x, N_DEVDECK) for x in o[L.DEV_DECK.start:L.DEV_DECK.stop])

    lr_rel = _decode_owner_5slot(o[L.LR_OWNER.start:L.LR_OWNER.stop])
    la_rel = _decode_owner_5slot(o[L.LA_OWNER.start:L.LA_OWNER.stop])
    sp_idx = _argmax_onehot(o[L.START_PLAYER.start:L.START_PLAYER.stop])
    start_player_rel = sp_idx if sp_idx >= 0 else 0
    free_roads = _unq(o[L.FREE_ROADS.start], N_FREEROADS)

    proposer_rel = _decode_owner_5slot(o[L.TRADE_PROPOSER.start:L.TRADE_PROPOSER.stop])
    give = tuple(_unq(x, N_TRADE) for x in o[L.TRADE_GIVE.start:L.TRADE_GIVE.stop])
    want = tuple(_unq(x, N_TRADE) for x in o[L.TRADE_WANT.start:L.TRADE_WANT.stop])
    responses = _decode_trade_responses(o[L.TRADE_RESPONSES.start:L.TRADE_RESPONSES.stop])

    return BoardView(
        nodes=nodes, edges=edges,
        hex_resources=hex_res, hex_numbers=hex_nums,
        port_types=port_types, robber_hex=robber,
        phase=phase, flag=flag, last_roll=last_roll, turn_norm=turn_norm,
        bank=bank, dev_deck=dev_deck,
        longest_road_rel=lr_rel, largest_army_rel=la_rel,
        start_player_rel=start_player_rel, free_roads=free_roads,
        self_hand=self_hand, self_dev_playable=self_dev_playable,
        self_dev_pending=self_dev_pending,
        self_dev_played_flag=self_played_flag,
        players=players,
        trade=TradeView(proposer_rel=proposer_rel, give=give, want=want, responses=responses),
    )


def decode_from_env(env: "fastcatan.Env", pov: int) -> BoardView:
    """Convenience: write obs from `pov` and decode it."""
    buf = np.zeros(L.TOTAL_WIDTH, dtype=np.float32)
    env.write_obs(pov, buf)
    return decode(buf)


# ---------------------------------------------------------------------------
# Pretty summary (used for `python -m ui.obs_decoder <log>`)
# ---------------------------------------------------------------------------

def summarize(view: BoardView, *, pov_seat: int | None = None) -> str:
    """One-page human summary of a BoardView."""
    lines: list[str] = []
    lines.append(f"phase={view.phase}  flag={view.flag}  last_roll={view.last_roll}  "
                 f"turn≈{view.turn_norm*400:.0f}  free_roads={view.free_roads}")
    if pov_seat is not None:
        lines.append(f"pov_seat={pov_seat}")
    lines.append(f"bank: {dict(zip(RES_NAMES, view.bank))}  dev_deck: "
                 f"{dict(zip(DEV_NAMES, view.dev_deck))}")
    lines.append(f"LR_owner_rel={view.longest_road_rel}  "
                 f"LA_owner_rel={view.largest_army_rel}  "
                 f"start_player_rel={view.start_player_rel}")
    lines.append("")
    lines.append("players (rel | vp | hand | dev | kn | road | s/c/r | ports | "
                 "disc | cur):")
    for p in view.players:
        ports = "".join("1" if b else "0" for b in p.ports)
        lines.append(
            f"  rel{p.rel}{'*' if p.is_self else ' '}: "
            f"vp={p.vp:2d} hand={p.handsize:2d} dev={p.total_dev} "
            f"kn={p.knights_played} rd_len={p.road_length} "
            f"left {p.settlements_left}/{p.cities_left}/{p.roads_left} "
            f"ports={ports} disc={p.discard_owed} "
            f"{'CUR' if p.is_current else '   '}"
        )
    lines.append("")
    lines.append(f"self hand:    {dict(zip(RES_NAMES, view.self_hand))}")
    lines.append(f"self dev:     playable={dict(zip(DEV_NAMES, view.self_dev_playable))}  "
                 f"pending={dict(zip(DEV_NAMES, view.self_dev_pending))}  "
                 f"played_this_turn={view.self_dev_played_flag}")
    lines.append("")
    n_built = sum(1 for n in view.nodes if n.kind is not None)
    n_roads = sum(1 for e in view.edges if e is not None)
    lines.append(f"board: {n_built} buildings, {n_roads} roads, robber@hex={view.robber_hex}")
    if view.trade.proposer_rel is not None or any(view.trade.give) or any(view.trade.want):
        lines.append(
            f"trade: proposer_rel={view.trade.proposer_rel} "
            f"give={dict(zip(RES_NAMES, view.trade.give))} "
            f"want={dict(zip(RES_NAMES, view.trade.want))} "
            f"responses={view.trade.responses}"
        )
    return "\n".join(lines)


def _main(argv: list[str]) -> int:
    import argparse

    from ui.log_format import decode_snap, read_log

    p = argparse.ArgumentParser(description="Dump decoded obs at a step.")
    p.add_argument("log", help="path to a .jsonl.gz log")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--pov", type=int, default=None,
                   help="seat (0..3) to use as POV; default = step's current_player")
    args = p.parse_args(argv)

    g = read_log(args.log)
    if not (0 <= args.step < len(g.steps)):
        p.error(f"--step out of range [0, {len(g.steps)})")
    target = g.steps[args.step]

    snap_info = g.nearest_snapshot(args.step)
    if snap_info is None:
        p.error("no snapshot at or before requested step")
    snap_idx, snap_raw = snap_info

    env = fastcatan.Env()
    env.reset(0)
    env.load_snapshot(snap_raw)
    # snapshot reflects state AFTER step `snap_idx`'s action — replay forward
    # from snap_idx+1 to (target.i) to align with the END of step `target.i`.
    for i in range(snap_idx + 1, target.i + 1):
        env.step(int(g.steps[i].a))

    pov = args.pov if args.pov is not None else int(env.current_player)
    view = decode_from_env(env, pov)
    print(summarize(view, pov_seat=pov))
    print(f"\n[step {target.i}] applied action={target.a}  r={target.r}  d={target.d}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
