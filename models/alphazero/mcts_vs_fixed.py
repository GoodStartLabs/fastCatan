"""Single-agent MCTS against a FIXED opponent policy (e.g. native AlphaBeta).

Contrast with mcts.py (max^n self-play, every seat searches via the net). Here only
seat 0 (the learner) makes tree decisions; opponent seats 1-3 and all chance are
folded into the environment transition by playing a fixed `opp_pick` policy and
reseeding. The tree therefore branches only on the learner's own moves, evaluated
against the opponent's ACTUAL behaviour.

Why: the net-modeled-opponent search (mcts.py used vs AB) flat-lined at 0% — the tree
assumed opponents play like the (weak) net, so its move evaluations were delusional
and the value/policy targets carried no signal. Modeling the opponent as the real AB
fixes that, and lets the learner's multi-ply search out-look-ahead AB(depth-1).

Cost: every macro-transition runs the opponent policy (ab_decide) for each opponent
turn until control returns to seat 0. With AB depth-1 (~free) this is affordable.

`choose(root_snapshot, temperature, add_root_noise) -> (action, pi, mask)` matches
mcts.MCTS so selfplay.play_one_game / evaluate.play_game drive it unchanged.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch

import fastcatan

from models.alphazero.mcts import (
    _unpack, p2p_trade_mask, p2p_banned_words, filter_p2p, Node,
    OBS_SIZE, NUM_ACTIONS, MASK_WORDS, NUM_PLAYERS, WIN_VP,
)
from models.alphazero.evaluate import make_alphabeta_pick

LEARNER = 0


class MCTSvsFixed:
    def __init__(
        self,
        net: torch.nn.Module,
        device: str = "cpu",
        sims: int = 64,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
        seed: int = 0,
        suppress_p2p: bool = True,
        value_mode: str = "vp_margin",
        ab_depth: int = 1,
        ab_prune: bool = False,
        leaf_eval: str = "net",
        ab_value_scale: float = 30.0,
        learner_seat: int = LEARNER,
    ):
        self.net = net
        self.device = device
        self.sims = sims
        self.c_puct = c_puct
        self.dir_alpha = dirichlet_alpha
        self.dir_frac = dirichlet_frac
        self.rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed ^ 0x71FED)
        self._p2p = p2p_trade_mask() if suppress_p2p else None
        self.value_mode = value_mode
        self.leaf_eval = leaf_eval
        self.ab_value_scale = ab_value_scale
        self.learner = learner_seat   # tree decisions + POV (seat rotation)
        self.opp = make_alphabeta_pick(
            self.rng, ab_depth, ab_prune,
            banned=p2p_banned_words() if suppress_p2p else None)

        self.env = fastcatan.Env()
        self._mask_buf = np.zeros(MASK_WORDS, dtype=np.uint64)
        self._obs = np.zeros(OBS_SIZE, dtype=np.float32)

    # -------- env reads --------

    def _legal(self):
        self.env.action_mask(self._mask_buf)
        mask, legal = _unpack(self._mask_buf)
        if self._p2p is not None:
            mask, legal = filter_p2p(mask, self._p2p)
        return mask, legal

    def _signature(self) -> tuple:
        e = self.env
        return (
            e.phase, e.flag, e.dice_roll,
            tuple(e.player_handsize(p) for p in range(NUM_PLAYERS)),
            tuple(e.player_vp(p) for p in range(NUM_PLAYERS)),
        )

    def _terminal_value(self) -> float:
        vps = [self.env.player_vp(p) for p in range(NUM_PLAYERS)]
        if self.value_mode == "vp_margin":
            best_other = max(vps[q] for q in range(NUM_PLAYERS)
                             if q != self.learner)
            return float(np.clip((vps[self.learner] - best_other) / 10.0,
                                 -1.0, 1.0))
        winner = next((p for p in range(NUM_PLAYERS) if vps[p] >= WIN_VP), -1)
        return 1.0 if winner == self.learner else -1.0

    def _advance_to_learner(self):
        """Play opponents + forced learner steps + chance from the env's CURRENT
        state until seat 0 faces a multi-legal choice or the game ends.

        Returns (done, mask, legal). Reseeds before each step to resample chance.
        """
        while True:
            mask, legal = self._legal()
            if not legal:
                return True, mask, legal
            cp = self.env.current_player
            if cp == self.learner and len(legal) > 1:
                return False, mask, legal
            action = (legal[0] if cp == self.learner
                      else self.opp(self.env, cp, legal))
            self.env.reseed(self.rng.getrandbits(64))
            _r, done = self.env.step(action)
            if done:
                return True, mask, legal

    # -------- net eval --------

    @torch.no_grad()
    def _expand(self, node: Node) -> float:
        """Expand from the env's CURRENT state (== node's state). Seat-0 POV only.

        leaf_eval='net'      -> the net's value head (default).
        leaf_eval='ab_value' -> HYBRID: the net still provides the PRIOR, but
        the leaf value is the native Catanatron heuristic — deterministic and
        fine-grained, attacking the leaf-noise saturation (a learned ±-ish
        value can't resolve AB-scale 1-3%-win-prob move differences; the
        hand value can, with zero variance). Normalized as a margin over the
        best opponent seat squashed by tanh(./ab_value_scale)."""
        self.env.write_obs(self.learner, self._obs)
        obs = torch.from_numpy(self._obs).unsqueeze(0).to(self.device)
        logits, value = self.net(obs)
        row = logits[0].float().cpu().numpy()
        row = np.where(node.mask, row, -np.inf)
        row -= row.max()
        p = np.exp(row)
        p[~node.mask] = 0.0
        s = p.sum()
        node.P = (p / s).astype(np.float32) if s > 0 else node.mask.astype(np.float32)
        node.expanded = True
        if self.leaf_eval == "ab_value":
            # Catanatron's value is LEXICOGRAPHIC: public VPs at 3e14 dwarf
            # the fine features (production/reach/etc at ~1e0-1e3). Decompose
            # and squash each scale separately so BOTH levels stay resolvable
            # in [-1,1] — a single tanh would quantize to coarse VP steps and
            # discard exactly the fine discrimination that makes AB strong.
            VP_W = 3e14
            v0 = self.env.ab_value(self.learner)
            vo = max(self.env.ab_value(q) for q in range(NUM_PLAYERS)
                     if q != self.learner)
            margin = v0 - vo
            vp_part = np.round(margin / VP_W)
            fine_part = margin - vp_part * VP_W
            return float(0.75 * np.tanh(vp_part / 3.0)
                         + 0.25 * np.tanh(fine_part / self.ab_value_scale))
        return float(value.item())

    # -------- selection --------

    def _select(self, node: Node) -> int:
        n = node.N
        q = np.where(n > 0, node.W / np.maximum(n, 1), 0.0)
        u = self.c_puct * node.P * math.sqrt(node.total_N + 1.0) / (1.0 + n)
        score = q + u
        score[~node.mask] = -np.inf
        return int(np.argmax(score))

    def _add_root_noise(self, root: Node) -> None:
        if len(root.legal) <= 1:
            return
        noise = self._np_rng.dirichlet([self.dir_alpha] * len(root.legal))
        f = self.dir_frac
        for i, a in enumerate(root.legal):
            root.P[a] = (1 - f) * root.P[a] + f * noise[i]

    # -------- one simulation --------

    def _simulate(self, root: Node) -> None:
        path = []
        node = root
        while True:
            a = self._select(node)
            path.append((node, a))

            self.env.load_snapshot(node.snap)
            self.env.reseed(self.rng.getrandbits(64))
            _r, done = self.env.step(a)
            if done:
                value = self._terminal_value()
                break

            done, mask, legal = self._advance_to_learner()
            if done:
                value = self._terminal_value()
                break

            sig = self._signature()
            bucket = node.children.setdefault(a, {})
            child = bucket.get(sig)
            if child is None:
                child = Node(self.env.snapshot(), self.learner, mask, legal)
                bucket[sig] = child
                value = self._expand(child)
                break
            node = child

        for nd, a in path:
            nd.N[a] += 1
            nd.W[a] += value
            nd.total_N += 1

    # -------- public API --------

    def choose(self, root_snapshot: bytes, temperature: float = 1.0,
               add_root_noise: bool = True):
        self.env.load_snapshot(root_snapshot)
        mask, legal = self._legal()
        root = Node(root_snapshot, self.learner, mask, legal)
        self._expand(root)
        if add_root_noise:
            self._add_root_noise(root)
        for _ in range(self.sims):
            self._simulate(root)

        counts = root.N.astype(np.float64)
        total = counts.sum()
        if total == 0:
            pi = root.mask.astype(np.float64); pi /= pi.sum()
        else:
            pi = counts / total

        if temperature <= 1e-6:
            action = int(np.argmax(counts))
        else:
            heated = counts ** (1.0 / temperature)
            hs = heated.sum()
            probs = heated / hs if hs > 0 else root.mask / root.mask.sum()
            probs = probs / probs.sum()
            action = int(self._np_rng.choice(NUM_ACTIONS, p=probs))
        return action, pi.astype(np.float32), root.mask.copy()
