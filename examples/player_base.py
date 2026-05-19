"""Base player interface + shared helpers for the test orchestrator."""
import abc
import random
import numpy as np

import fastcatan

def legal_actions(mask: np.ndarray) -> list[int]:
    """Return list of action IDs whose bit is set in the uint64 bitmask buffer."""
    out: list[int] = []
    for word_idx, word in enumerate(mask):
        w = int(word)
        base = word_idx * 64
        while w:
            bit = (w & -w).bit_length() - 1
            out.append(base + bit)
            w &= w - 1
    return out


def build_p2p_trade_filter() -> np.ndarray:
    """Bitmask of all player-to-player trade action IDs.

    Use as AND-NOT filter to suppress p2p trading. Bank/port trades stay enabled.
    """
    a = fastcatan.action
    ids = (
        list(range(a.TRADE_ADD_GIVE_BASE, a.TRADE_ADD_GIVE_BASE + 5))
        + list(range(a.TRADE_ADD_WANT_BASE, a.TRADE_ADD_WANT_BASE + 5))
        + [a.TRADE_OPEN, a.TRADE_ACCEPT, a.TRADE_DECLINE]
        + list(range(a.TRADE_CONFIRM_BASE, a.TRADE_CONFIRM_BASE + 4))
        + [a.TRADE_CANCEL]
    )
    m = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
    for aid in ids:
        m[aid // 64] |= np.uint64(1) << np.uint64(aid % 64)
    return m


class Player(abc.ABC):
    """Base class for all bots. Subclass and override `act`."""

    name: str = "?"

    def __init__(self, seed: int = 0, forbid: np.ndarray | None = None):
        self.rng = random.Random(seed)
        self.forbid = forbid

    @abc.abstractmethod
    def act(self, env, mask: np.ndarray) -> int:
        """Pick an action ID. `mask` is uint64[MASK_WORDS] of legal actions."""
        ...
