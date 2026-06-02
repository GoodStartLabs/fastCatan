"""Full-state stochastic MCTS over the C++ fastcatan simulator.

One MCTS instance owns a scratch ``fastcatan.Env`` used purely for branching: every
simulation does ``load_snapshot(node) -> reseed(rand) -> step(action)``. Because the
RNG lives inside GameState (state.hpp), reseed resamples all chance for the upcoming
step. Deterministic actions ignore the RNG, so their successor + signature are stable
and collapse to a single child; chance actions (dice / dev draw / robber steal) spread
across signature-keyed children in proportion to how often each outcome is sampled.

Tree node = a decision state. Edges = legal actions. Per-action stats N/W/Q/P are
stored as dense numpy arrays of length NUM_ACTIONS (cheap, vectorized PUCT). Children
are keyed by (action, outcome-signature) so chance fans out.

4-player credit assignment is max^n: the value head gives win-prob for a single POV,
so a leaf is evaluated from all 4 seat POVs into a length-4 vector; each node backs up
the value of *its* to-move seat, and selection maximizes that same seat's Q. No
zero-sum negation (that's only valid for 2-player).
"""
from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import torch

import fastcatan

OBS_SIZE = fastcatan.OBS_SIZE
NUM_ACTIONS = fastcatan.NUM_ACTIONS
MASK_WORDS = fastcatan.MASK_WORDS
NUM_PLAYERS = fastcatan.NUM_PLAYERS
WIN_VP = 10


def _unpack(mask_words: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """uint64[MASK_WORDS] -> (bool[NUM_ACTIONS], sorted legal id list)."""
    b = np.zeros(NUM_ACTIONS, dtype=bool)
    legal: list[int] = []
    for wi, word in enumerate(mask_words):
        w = int(word)
        base = wi * 64
        while w:
            bit = (w & -w).bit_length() - 1
            aid = base + bit
            if aid < NUM_ACTIONS:
                b[aid] = True
                legal.append(aid)
            w &= w - 1
    return b, legal


def p2p_trade_mask() -> np.ndarray:
    """bool[NUM_ACTIONS] marking every player-to-player trade action.

    AND-NOT this off a legal mask to forbid p2p trading (bank/port trades stay).
    Mirrors models/selfplay/selfplay_env._p2p_trade_mask_bool, inlined here to keep
    the alphazero package import-light. Default-ON for AZ because the M4 eval vs
    catanatron AlphaBeta runs --no-trades (AlphaBeta crashes on P2P trades), so
    trade-free training is the consistent regime AND it removes the trade-compose
    branching that stalls untrained self-play into the MAX_TURNS cap.
    """
    a = fastcatan.action
    ids = (
        list(range(a.TRADE_ADD_GIVE_BASE, a.TRADE_ADD_GIVE_BASE + 5))
        + list(range(a.TRADE_ADD_WANT_BASE, a.TRADE_ADD_WANT_BASE + 5))
        + [a.TRADE_OPEN, a.TRADE_ACCEPT, a.TRADE_DECLINE]
        + list(range(a.TRADE_CONFIRM_BASE, a.TRADE_CONFIRM_BASE + 4))
        + [a.TRADE_CANCEL]
    )
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    for i in ids:
        if i < NUM_ACTIONS:
            m[i] = True
    return m


def filter_p2p(mask_bool: np.ndarray, p2p_bool: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """AND-NOT p2p trades off ``mask_bool``; never strand a seat with no legal move."""
    filtered = mask_bool & ~p2p_bool
    if not filtered.any():
        return mask_bool, [int(i) for i in np.nonzero(mask_bool)[0]]
    return filtered, [int(i) for i in np.nonzero(filtered)[0]]


class Node:
    __slots__ = ("snap", "to_move", "legal", "mask", "P", "N", "W",
                 "total_N", "children", "expanded")

    def __init__(self, snap: bytes, to_move: int, mask: np.ndarray, legal: list[int]):
        self.snap = snap
        self.to_move = to_move
        self.legal = legal
        self.mask = mask
        self.P = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.N = np.zeros(NUM_ACTIONS, dtype=np.int32)
        self.W = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.total_N = 0
        # children[action] = {signature: Node}
        self.children: dict[int, dict[tuple, Node]] = {}
        self.expanded = False


class MCTS:
    def __init__(
        self,
        net: torch.nn.Module,
        device: str = "cpu",
        sims: int = 50,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
        seed: int = 0,
        suppress_p2p: bool = True,
    ):
        self.net = net
        self.device = device
        self.sims = sims
        self.c_puct = c_puct
        self.dir_alpha = dirichlet_alpha
        self.dir_frac = dirichlet_frac
        self.rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed ^ 0xA17ECA7)
        self._p2p_bool = p2p_trade_mask() if suppress_p2p else None

        self.env = fastcatan.Env()
        self._mask_buf = np.zeros(MASK_WORDS, dtype=np.uint64)
        self._obs4 = np.zeros((NUM_PLAYERS, OBS_SIZE), dtype=np.float32)

    # -------- env reads --------

    def _read_mask(self) -> tuple[np.ndarray, list[int]]:
        self.env.action_mask(self._mask_buf)
        mask, legal = _unpack(self._mask_buf)
        if self._p2p_bool is not None:
            mask, legal = filter_p2p(mask, self._p2p_bool)
        return mask, legal

    def _node_from_env(self) -> Node:
        """Build a node from the scratch env's CURRENT state (no expansion)."""
        mask, legal = self._read_mask()
        return Node(self.env.snapshot(), self.env.current_player, mask, legal)

    def _signature(self) -> tuple:
        """Observable outcome key — distinguishes chance results so they fan out.

        Captures dice + turn cursor + every seat's handsize and VP. This separates
        the 11 dice totals, robber steals (handsize shifts), and VP-card dev draws
        (VP bumps). It does NOT distinguish the *resource identity* of a steal or the
        non-VP dev drawn — those outcomes merge into one child, a v1 approximation.
        """
        e = self.env
        return (
            e.current_player, e.phase, e.flag, e.dice_roll,
            tuple(e.player_handsize(p) for p in range(NUM_PLAYERS)),
            tuple(e.player_vp(p) for p in range(NUM_PLAYERS)),
        )

    def _terminal_value(self) -> np.ndarray:
        v = np.full(NUM_PLAYERS, -1.0, dtype=np.float32)
        for p in range(NUM_PLAYERS):
            if self.env.player_vp(p) >= WIN_VP:
                v[p] = 1.0
                break
        return v

    # -------- net eval --------

    @torch.no_grad()
    def _expand_current(self, node: Node) -> np.ndarray:
        """Expand ``node`` using the env's CURRENT state (assumed == node's state).

        Forwards all 4 seat POVs in one batch: row to_move gives the policy prior,
        all rows give the length-4 value vector for max^n backup.
        """
        for pov in range(NUM_PLAYERS):
            self.env.write_obs(pov, self._obs4[pov])
        obs = torch.from_numpy(self._obs4).to(self.device)
        logits, values = self.net(obs)                       # (4,A), (4,)
        row = logits[node.to_move].float().cpu().numpy()
        row = np.where(node.mask, row, -np.inf)
        row -= row.max()
        p = np.exp(row)
        p[~node.mask] = 0.0
        s = p.sum()
        node.P = (p / s).astype(np.float32) if s > 0 else node.mask.astype(np.float32)
        node.expanded = True
        return values.float().cpu().numpy()

    # -------- selection --------

    def _select(self, node: Node) -> int:
        # PUCT over legal actions; illegal forced to -inf.
        n = node.N
        q = np.where(n > 0, node.W / np.maximum(n, 1), 0.0)
        u = self.c_puct * node.P * math.sqrt(node.total_N + 1.0) / (1.0 + n)
        score = q + u
        score[~node.mask] = -np.inf
        return int(np.argmax(score))

    def _add_root_noise(self, root: Node) -> None:
        legal = root.legal
        if len(legal) <= 1:
            return
        noise = self._np_rng.dirichlet([self.dir_alpha] * len(legal))
        f = self.dir_frac
        for i, a in enumerate(legal):
            root.P[a] = (1 - f) * root.P[a] + f * noise[i]

    # -------- one simulation --------

    def _simulate(self, root: Node) -> None:
        path: list[tuple[Node, int]] = []
        node = root
        while True:
            a = self._select(node)
            path.append((node, a))

            self.env.load_snapshot(node.snap)
            self.env.reseed(self.rng.getrandbits(64))
            _r, done = self.env.step(a)

            if done:
                value_vec = self._terminal_value()
                break

            sig = self._signature()
            bucket = node.children.setdefault(a, {})
            child = bucket.get(sig)
            if child is None:
                child = self._node_from_env()
                bucket[sig] = child
                value_vec = self._expand_current(child)
                break
            node = child

        # backup: each node gets its own to-move seat's value
        for nd, a in path:
            nd.N[a] += 1
            nd.W[a] += value_vec[nd.to_move]
            nd.total_N += 1

    # -------- public API --------

    def run(self, root_snapshot: bytes, add_root_noise: bool = True) -> Node:
        self.env.load_snapshot(root_snapshot)
        root = self._node_from_env()
        self._expand_current(root)
        if add_root_noise:
            self._add_root_noise(root)
        for _ in range(self.sims):
            self._simulate(root)
        return root

    def choose(
        self,
        root_snapshot: bytes,
        temperature: float = 1.0,
        add_root_noise: bool = True,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Run MCTS from a state snapshot and pick an action.

        Returns (action, pi, mask) where pi is the visit-count policy over all
        NUM_ACTIONS (0 on illegal) — the training target.
        """
        root = self.run(root_snapshot, add_root_noise=add_root_noise)
        counts = root.N.astype(np.float64)
        total = counts.sum()
        if total == 0:  # no sims expanded anything (shouldn't happen) -> uniform legal
            pi = root.mask.astype(np.float64)
            pi /= pi.sum()
        else:
            pi = counts / total

        if temperature <= 1e-6:
            action = int(np.argmax(counts))
        else:
            heated = counts ** (1.0 / temperature)
            hs = heated.sum()
            probs = heated / hs if hs > 0 else root.mask / root.mask.sum()
            probs = probs / probs.sum()          # guard float drift for np.choice
            action = int(self._np_rng.choice(NUM_ACTIONS, p=probs))
        return action, pi.astype(np.float32), root.mask.copy()
