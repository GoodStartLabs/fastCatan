"""Named slices over the flat float32 obs vector.

Single source of truth for the obs layout, mirroring `src/catan/obs.cpp`.
The decoder (`obs_decoder.py`), any obs-slot inspector, and the test that
guards layout drift all import from here.

Resource ordering (fastcatan): 0=brick, 1=lumber, 2=wool, 3=grain, 4=ore.
Hex resource one-hot adds slot 5 = desert.
Port type one-hot adds slot 5 = generic 3:1.

Count fields (vp, handsize, dev, road lengths, resources, bank, ...) are stored
NORMALIZED by structural Catan maxima (see src/catan/obs.cpp namespace norm and
bridge/obs_encoder.py N_*). `obs_decoder.py` multiplies them back to integers.
One-hots, port bits, and is_current are already 0/1.
"""

from __future__ import annotations

from dataclasses import dataclass

import fastcatan

NUM_PLAYERS = int(fastcatan.NUM_PLAYERS)
NUM_NODES = int(fastcatan.NUM_NODES)
NUM_EDGES = int(fastcatan.NUM_EDGES)
NUM_HEXES = int(fastcatan.NUM_HEXES)
NUM_PORTS = int(fastcatan.NUM_PORTS)
NUM_RES = 5

# Per-player block width (vp, handsize, total_dev, knights, road_len,
# settle_left, city_left, road_left, ports[6], discard_left, is_current).
PLAYER_BLOCK_W = 8 + 6 + 1 + 1  # = 16
SELF_PRIVATE_W = NUM_RES + 5 + 5 + 1  # res(5) + dev_playable(5) + dev_pending(5) + played_flag

NODE_CH = 2 * NUM_PLAYERS   # (settle, city) per player slot = 8
EDGE_CH = NUM_PLAYERS       # 4


@dataclass(frozen=True)
class Slice:
    """Half-open [start, stop) into the obs vector."""
    start: int
    stop: int

    @property
    def width(self) -> int:
        return self.stop - self.start


def _seq(start: int, *widths: int) -> tuple[Slice, ...]:
    out: list[Slice] = []
    p = start
    for w in widths:
        out.append(Slice(p, p + w))
        p += w
    return tuple(out)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_p = 0

# Per-player blocks: 4 × 16
PLAYER_BLOCKS = tuple(
    Slice(_p + i * PLAYER_BLOCK_W, _p + (i + 1) * PLAYER_BLOCK_W)
    for i in range(NUM_PLAYERS)
)
_p += NUM_PLAYERS * PLAYER_BLOCK_W

# Self private subfields
SELF_RES, SELF_DEV_PLAYABLE, SELF_DEV_PENDING, SELF_DEV_PLAYED_FLAG = _seq(
    _p, NUM_RES, 5, 5, 1
)
_p += SELF_PRIVATE_W

# Nodes: 54 × 8 (settle/city × player relseat 0..3)
NODES = Slice(_p, _p + NUM_NODES * NODE_CH)
_p = NODES.stop

# Edges: 72 × 4 (road × player relseat 0..3)
EDGES = Slice(_p, _p + NUM_EDGES * EDGE_CH)
_p = EDGES.stop

# Hex resources: 19 × 6 one-hot
HEX_RES = Slice(_p, _p + NUM_HEXES * 6)
_p = HEX_RES.stop

# Hex numbers: 19 floats, normalized by 12
HEX_NUMS = Slice(_p, _p + NUM_HEXES)
_p = HEX_NUMS.stop

# Port types: 9 × 6 one-hot
PORT_TYPES = Slice(_p, _p + NUM_PORTS * 6)
_p = PORT_TYPES.stop

# Robber hex one-hot
ROBBER = Slice(_p, _p + NUM_HEXES)
_p = ROBBER.stop

# Game state
PHASE = Slice(_p, _p + 4); _p = PHASE.stop
FLAG = Slice(_p, _p + 8); _p = FLAG.stop
LAST_ROLL = Slice(_p, _p + 13); _p = LAST_ROLL.stop
TURN_NORM = Slice(_p, _p + 1); _p = TURN_NORM.stop
BANK = Slice(_p, _p + NUM_RES); _p = BANK.stop
DEV_DECK = Slice(_p, _p + 5); _p = DEV_DECK.stop
LR_OWNER = Slice(_p, _p + 5); _p = LR_OWNER.stop     # 5 slots [self, +1, +2, +3, none]
LA_OWNER = Slice(_p, _p + 5); _p = LA_OWNER.stop
START_PLAYER = Slice(_p, _p + 4); _p = START_PLAYER.stop
FREE_ROADS = Slice(_p, _p + 1); _p = FREE_ROADS.stop

# Trade scratch
TRADE_PROPOSER = Slice(_p, _p + 5); _p = TRADE_PROPOSER.stop
TRADE_GIVE = Slice(_p, _p + NUM_RES); _p = TRADE_GIVE.stop
TRADE_WANT = Slice(_p, _p + NUM_RES); _p = TRADE_WANT.stop
TRADE_RESPONSES = Slice(_p, _p + 3 * 4); _p = TRADE_RESPONSES.stop  # 3 opponents × 4-slot one-hot

TOTAL_WIDTH = _p


# ---------------------------------------------------------------------------
# Per-player block subfields (offsets relative to a player-block Slice)
# ---------------------------------------------------------------------------

# layout: [vp, handsize, total_dev, knights, road_len,
#          settle_left, city_left, road_left,
#          ports[0..5],
#          discard_left, is_current]
PB_VP             = 0
PB_HANDSIZE       = 1
PB_TOTAL_DEV      = 2
PB_KNIGHTS        = 3
PB_ROAD_LEN       = 4
PB_SETTLE_LEFT    = 5
PB_CITY_LEFT      = 6
PB_ROAD_LEFT      = 7
PB_PORTS          = slice(8, 14)   # 6 slots
PB_DISCARD_LEFT   = 14
PB_IS_CURRENT     = 15


# ---------------------------------------------------------------------------
# Sanity guard
# ---------------------------------------------------------------------------

assert TOTAL_WIDTH == int(fastcatan.OBS_SIZE), (
    f"ui.obs_layout drift: computed {TOTAL_WIDTH} vs fastcatan.OBS_SIZE "
    f"{int(fastcatan.OBS_SIZE)} — update layout to match src/catan/obs.cpp"
)


__all__ = [
    "NUM_PLAYERS", "NUM_NODES", "NUM_EDGES", "NUM_HEXES", "NUM_PORTS", "NUM_RES",
    "PLAYER_BLOCK_W", "SELF_PRIVATE_W", "NODE_CH", "EDGE_CH",
    "Slice",
    "PLAYER_BLOCKS",
    "SELF_RES", "SELF_DEV_PLAYABLE", "SELF_DEV_PENDING", "SELF_DEV_PLAYED_FLAG",
    "NODES", "EDGES", "HEX_RES", "HEX_NUMS", "PORT_TYPES", "ROBBER",
    "PHASE", "FLAG", "LAST_ROLL", "TURN_NORM", "BANK", "DEV_DECK",
    "LR_OWNER", "LA_OWNER", "START_PLAYER", "FREE_ROADS",
    "TRADE_PROPOSER", "TRADE_GIVE", "TRADE_WANT", "TRADE_RESPONSES",
    "TOTAL_WIDTH",
    "PB_VP", "PB_HANDSIZE", "PB_TOTAL_DEV", "PB_KNIGHTS", "PB_ROAD_LEN",
    "PB_SETTLE_LEFT", "PB_CITY_LEFT", "PB_ROAD_LEFT",
    "PB_PORTS", "PB_DISCARD_LEFT", "PB_IS_CURRENT",
]
