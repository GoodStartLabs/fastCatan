"""Human-readable names + category for fastcatan flat action IDs.

Used by the mask viewer (M4) and the replayer (M5). Mirrors the layout in
`include/rules.hpp` / `bindings/pycatan/bindings.cpp` (action constants) and
`bridge/action_codec.py`.

Resource indices (fastcatan): 0=brick, 1=lumber, 2=wool, 3=grain, 4=ore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

import fastcatan

_a = fastcatan.action
NUM_ACTIONS = int(fastcatan.NUM_ACTIONS)
NUM_NODES = int(fastcatan.NUM_NODES)
NUM_EDGES = int(fastcatan.NUM_EDGES)
NUM_HEXES = int(fastcatan.NUM_HEXES)

RES_NAMES = ("brick", "lumber", "wool", "grain", "ore")

# Category labels used in the mask side-panel.
CAT_SETTLE       = "settle"
CAT_CITY         = "city"
CAT_ROAD         = "road"
CAT_ROLL         = "roll"
CAT_END_TURN     = "end_turn"
CAT_DISCARD      = "discard"
CAT_MOVE_ROBBER  = "move_robber"
CAT_STEAL        = "steal"
CAT_TRADE_BANK   = "trade_bank"
CAT_BUY_DEV      = "buy_dev"
CAT_PLAY_KNIGHT  = "play_knight"
CAT_PLAY_RB      = "play_road_building"
CAT_PLAY_YOP     = "play_year_of_plenty"
CAT_PLAY_MONO    = "play_monopoly"
CAT_TRADE_GIVE   = "trade_add_give"
CAT_TRADE_WANT   = "trade_add_want"
CAT_TRADE_OPEN   = "trade_open"
CAT_TRADE_ACCEPT = "trade_accept"
CAT_TRADE_DECL   = "trade_decline"
CAT_TRADE_CONF   = "trade_confirm"
CAT_TRADE_CANCEL = "trade_cancel"

# Categories with a spatial slot (used by the board overlay).
SPATIAL_CATS = frozenset({
    CAT_SETTLE, CAT_CITY, CAT_ROAD, CAT_MOVE_ROBBER,
})


@dataclass(frozen=True)
class ActionInfo:
    aid: int
    category: str
    # Optional payload — meaning depends on category:
    #   settle/city: node id
    #   road:        edge id
    #   move_robber: hex id
    #   steal:       seat (relseat, 0..3)
    #   discard/trade_add_*/play_monopoly: resource id
    #   trade_bank:  (give_res, get_res)
    #   play_year_of_plenty: (r1, r2)
    #   trade_confirm: partner relseat
    payload: object = None

    @property
    def label(self) -> str:
        return _format_label(self)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def info(aid: int) -> ActionInfo:
    if _a.SETTLE_BASE <= aid < _a.SETTLE_BASE + NUM_NODES:
        return ActionInfo(aid, CAT_SETTLE, aid - _a.SETTLE_BASE)
    if _a.CITY_BASE <= aid < _a.CITY_BASE + NUM_NODES:
        return ActionInfo(aid, CAT_CITY, aid - _a.CITY_BASE)
    if _a.ROAD_BASE <= aid < _a.ROAD_BASE + NUM_EDGES:
        return ActionInfo(aid, CAT_ROAD, aid - _a.ROAD_BASE)
    if aid == _a.ROLL_DICE:
        return ActionInfo(aid, CAT_ROLL)
    if aid == _a.END_TURN:
        return ActionInfo(aid, CAT_END_TURN)
    if _a.DISCARD_BASE <= aid < _a.DISCARD_BASE + 5:
        return ActionInfo(aid, CAT_DISCARD, aid - _a.DISCARD_BASE)
    if _a.MOVE_ROBBER_BASE <= aid < _a.MOVE_ROBBER_BASE + NUM_HEXES:
        return ActionInfo(aid, CAT_MOVE_ROBBER, aid - _a.MOVE_ROBBER_BASE)
    if _a.STEAL_BASE <= aid < _a.STEAL_BASE + 4:
        return ActionInfo(aid, CAT_STEAL, aid - _a.STEAL_BASE)
    if _a.TRADE_BASE <= aid < _a.TRADE_BASE + 25:
        off = aid - _a.TRADE_BASE
        return ActionInfo(aid, CAT_TRADE_BANK, (off // 5, off % 5))
    if aid == _a.BUY_DEV:
        return ActionInfo(aid, CAT_BUY_DEV)
    if aid == _a.PLAY_KNIGHT:
        return ActionInfo(aid, CAT_PLAY_KNIGHT)
    if aid == _a.PLAY_ROAD_BUILDING:
        return ActionInfo(aid, CAT_PLAY_RB)
    if _a.PLAY_YEAR_OF_PLENTY <= aid < _a.PLAY_YEAR_OF_PLENTY + 25:
        off = aid - _a.PLAY_YEAR_OF_PLENTY
        return ActionInfo(aid, CAT_PLAY_YOP, (off // 5, off % 5))
    if _a.PLAY_MONOPOLY <= aid < _a.PLAY_MONOPOLY + 5:
        return ActionInfo(aid, CAT_PLAY_MONO, aid - _a.PLAY_MONOPOLY)
    if _a.TRADE_ADD_GIVE_BASE <= aid < _a.TRADE_ADD_GIVE_BASE + 5:
        return ActionInfo(aid, CAT_TRADE_GIVE, aid - _a.TRADE_ADD_GIVE_BASE)
    if _a.TRADE_ADD_WANT_BASE <= aid < _a.TRADE_ADD_WANT_BASE + 5:
        return ActionInfo(aid, CAT_TRADE_WANT, aid - _a.TRADE_ADD_WANT_BASE)
    if aid == _a.TRADE_OPEN:
        return ActionInfo(aid, CAT_TRADE_OPEN)
    if aid == _a.TRADE_ACCEPT:
        return ActionInfo(aid, CAT_TRADE_ACCEPT)
    if aid == _a.TRADE_DECLINE:
        return ActionInfo(aid, CAT_TRADE_DECL)
    if _a.TRADE_CONFIRM_BASE <= aid < _a.TRADE_CONFIRM_BASE + 4:
        return ActionInfo(aid, CAT_TRADE_CONF, aid - _a.TRADE_CONFIRM_BASE)
    if aid == _a.TRADE_CANCEL:
        return ActionInfo(aid, CAT_TRADE_CANCEL)
    raise ValueError(f"unmapped action id {aid}")


def _format_label(ai: ActionInfo) -> str:
    c = ai.category
    pl = ai.payload
    if c == CAT_SETTLE:      return f"SETTLE @ n{pl:02X}"
    if c == CAT_CITY:        return f"CITY   @ n{pl:02X}"
    if c == CAT_ROAD:        return f"ROAD   @ e{pl:02X}"
    if c == CAT_ROLL:        return "ROLL_DICE"
    if c == CAT_END_TURN:    return "END_TURN"
    if c == CAT_DISCARD:     return f"DISCARD {RES_NAMES[pl]}"
    if c == CAT_MOVE_ROBBER: return f"MOVE_ROBBER @ h{pl:02X}"
    if c == CAT_STEAL:       return f"STEAL rel{pl}"
    if c == CAT_TRADE_BANK:
        g, t = pl  # type: ignore[misc]
        return f"BANK_TRADE 4 {RES_NAMES[g]} -> 1 {RES_NAMES[t]}"
    if c == CAT_BUY_DEV:     return "BUY_DEV_CARD"
    if c == CAT_PLAY_KNIGHT: return "PLAY_KNIGHT"
    if c == CAT_PLAY_RB:     return "PLAY_ROAD_BUILDING"
    if c == CAT_PLAY_YOP:
        r1, r2 = pl  # type: ignore[misc]
        return f"PLAY_YEAR_OF_PLENTY {RES_NAMES[r1]} + {RES_NAMES[r2]}"
    if c == CAT_PLAY_MONO:   return f"PLAY_MONOPOLY {RES_NAMES[pl]}"
    if c == CAT_TRADE_GIVE:  return f"TRADE_ADD_GIVE {RES_NAMES[pl]}"
    if c == CAT_TRADE_WANT:  return f"TRADE_ADD_WANT {RES_NAMES[pl]}"
    if c == CAT_TRADE_OPEN:  return "TRADE_OPEN"
    if c == CAT_TRADE_ACCEPT:return "TRADE_ACCEPT"
    if c == CAT_TRADE_DECL:  return "TRADE_DECLINE"
    if c == CAT_TRADE_CONF:  return f"TRADE_CONFIRM rel{pl}"
    if c == CAT_TRADE_CANCEL:return "TRADE_CANCEL"
    return f"?aid{ai.aid}"


def name(aid: int) -> str:
    """Compact one-liner. Convenience wrapper."""
    return info(aid).label


# ---------------------------------------------------------------------------
# Mask bitset utilities
# ---------------------------------------------------------------------------

def mask_to_ids(mask: np.ndarray) -> list[int]:
    """uint64[MASK_WORDS] bitmask -> sorted list of legal action IDs."""
    out: list[int] = []
    for w_idx in range(mask.shape[0]):
        w = int(mask[w_idx])
        base = w_idx * 64
        while w:
            bit = (w & -w).bit_length() - 1
            out.append(base + bit)
            w &= w - 1
    return out


def ids_to_mask(ids: Iterable[int]) -> np.ndarray:
    m = np.zeros(int(fastcatan.MASK_WORDS), dtype=np.uint64)
    for aid in ids:
        m[aid // 64] |= np.uint64(1) << np.uint64(aid % 64)
    return m
