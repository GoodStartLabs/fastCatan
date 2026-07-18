"""Freeze the FULL byte layout of the ctypes GameState mirror.

The `sizeof(CGameState) == 384` assert inside `state_mirror` is necessary but
NOT sufficient (substrate-map §4.2): a +1-byte field absorbed by a -1-byte pad
keeps the total at 384 while every field after it is silently misread — this
exact drift once produced a 25/25 differential failure (unmirrored
`trade_compose_count`). This test pins each field's (name, offset, size) across
the three mirror structs by hash. Any reorder / resize / retype of `GameState`,
`BoardLayout`, or the snapshot must re-mirror `state_mirror.py` AND update this
hash deliberately — never let the layout drift silently.
"""
import ctypes
import hashlib

from bridge import state_mirror as M


def _layout(cls):
    rows = [(cls.__name__, ctypes.sizeof(cls))]
    for name, *_ in cls._fields_:
        f = getattr(cls, name)
        rows.append((name, f.offset, f.size))
    return rows


def _signature():
    sig = []
    for cls in (M.CGameState, M.CBoardLayout, M.CSnapshot):
        sig.extend(_layout(cls))
    return sig


# sha256 of repr(_signature()) on the frozen substrate-v1 mirror.
EXPECTED_LAYOUT_HASH = "a739f178e6104517410cac61fdb4c42cfb391b4ed56d7d198a48a184ccecb1c6"


def test_state_mirror_layout_frozen():
    sig = _signature()
    h = hashlib.sha256(repr(sig).encode()).hexdigest()
    assert h == EXPECTED_LAYOUT_HASH, (
        "state_mirror ctypes layout changed.\n"
        f"  expected {EXPECTED_LAYOUT_HASH}\n  got      {h}\n"
        "If GameState/BoardLayout changed, re-mirror state_mirror.py (fields + "
        "explicit pads) and update EXPECTED_LAYOUT_HASH deliberately.\n"
        f"layout = {sig}"
    )
