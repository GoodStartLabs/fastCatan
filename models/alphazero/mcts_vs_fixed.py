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
        learner_trades: bool = False,
        trade_prior_frac: float = 0.05,
        trade_add_cap: int = 3,
        trade_step_cost: float = 0.01,
        value_mode: str = "vp_margin",
        ab_depth: int = 1,
        ab_prune: bool = False,
        leaf_eval: str = "net",
        ab_value_scale: float = 30.0,
        learner_seat: int = LEARNER,
        catanatron_chance: bool = False,
        opp_model: str = "alphabeta",
        judge: torch.nn.Module | None = None,
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
        # Learner-side P2P trading. When True, the p2p filter is NOT applied
        # at the LEARNER's own decision nodes (root included): the tree may
        # compose offers natively (ADD_GIVE/ADD_WANT -> OPEN -> responders ->
        # CONFIRM). The filter still applies to in-tree OPPONENT picks, so
        # the modeled opponents never *offer* — mirroring the real table,
        # where AlphaBeta cannot propose a trade. Opponent trade RESPONSES
        # survive the filter via never-strand (a responder's whole legal set
        # is trade ids), and ab_decide values ACCEPT vs DECLINE through its
        # own never-strand — exactly the 1-ply value response the trade-aware
        # table AB plays.
        self.learner_trades = learner_trades
        self.trade_prior_frac = trade_prior_frac
        self.trade_add_cap = trade_add_cap
        self.trade_step_cost = trade_step_cost
        self._trade_bool = p2p_trade_mask()
        _a = fastcatan.action
        self._add_give = list(range(_a.TRADE_ADD_GIVE_BASE,
                                    _a.TRADE_ADD_GIVE_BASE + 5))
        self._add_want = list(range(_a.TRADE_ADD_WANT_BASE,
                                    _a.TRADE_ADD_WANT_BASE + 5))
        # Compose-churn ids (ADD/OPEN/CANCEL): a step on these keeps the
        # clock on the learner with the board unchanged, so churn arms
        # inherit the pre-reply root value while real moves' Q carries the
        # opponents' replies — visits would sink into null negotiation.
        # _simulate charges trade_step_cost per churn step, REFUNDED when
        # the path contains a CONFIRM (completed trade): failed/aimless
        # negotiation pays, profitable trades are judged on board value.
        self._churn_bool = np.zeros_like(self._trade_bool)
        self._churn_bool[self._add_give + self._add_want
                         + [_a.TRADE_OPEN, _a.TRADE_CANCEL]] = True
        self._confirm_lo = _a.TRADE_CONFIRM_BASE
        self._confirm_hi = _a.TRADE_CONFIRM_BASE + 4
        self.value_mode = value_mode
        self.leaf_eval = leaf_eval
        self.ab_value_scale = ab_value_scale
        self.learner = learner_seat   # tree decisions + POV (seat rotation)
        if opp_model == "net":
            # STAGE-2 de-catanatronization: the in-tree opponent is the
            # net's own masked-argmax policy (the AB clone, 78-80% top-1 vs
            # the real AB) instead of native ab_decide. With leaf_eval='net'
            # the search is then FULLY SELF-CONTAINED — no ab_value/ab_decide
            # anywhere at inference. (The pre-IL net-modeled-opponent search
            # flat-lined at 0%, but that net didn't model AB at all; the
            # clone is a different regime.)
            self.opp = self._net_opp
        else:
            self.opp = make_alphabeta_pick(
                self.rng, ab_depth, ab_prune,
                banned=(p2p_banned_words() if (suppress_p2p or
                                               catanatron_chance) else None),
                chance_mode=1 if catanatron_chance else 0)

        self.env = fastcatan.Env()
        self._mask_buf = np.zeros(MASK_WORDS, dtype=np.uint64)
        # The PRIOR/OPP net may itself be full-obs (e.g. the XL single-net
        # config): size its buffers to the net's input width.
        net_dim = next(net.parameters()).shape[1]
        if net_dim not in (OBS_SIZE, fastcatan.OBS_FULL_SIZE):
            raise ValueError(f"net input dim {net_dim} unsupported")
        self._net_full = net_dim == fastcatan.OBS_FULL_SIZE
        self._obs = np.zeros(net_dim, dtype=np.float32)
        self._opp_obs = np.zeros(net_dim, dtype=np.float32)
        # JUDGE: a separate net that owns the leaf VALUE and the value-head
        # trade responses; `net` keeps the prior and the in-tree opponent
        # policy. Width-aware: a 1132 judge gets write_obs_full (POV prefix +
        # hidden-enemy appendix — the same information ab_value reads at
        # leaves), a 1084 judge gets plain POV obs (control configs).
        # Self-contained either way: no ab_value/ab_decide.
        self.judge = judge
        self._judge_full = False
        if judge is not None:
            jdim = next(judge.parameters()).shape[1]
            if jdim == fastcatan.OBS_FULL_SIZE:
                self._judge_full = True
            elif jdim != OBS_SIZE:
                raise ValueError(f"judge input dim {jdim} is neither "
                                 f"OBS_SIZE nor OBS_FULL_SIZE")
        self._full_obs = np.zeros(fastcatan.OBS_FULL_SIZE, dtype=np.float32)
        self._pov_obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self.last_root_value = 0.0   # backed-up V(root) from the last choose()

    @torch.no_grad()
    def _net_opp(self, game_env, cp: int, legal) -> int:
        """In-tree opponent pick: the net's argmax over the seat's legal set
        (cp's POV obs — the clone trained on all four seats). Deterministic,
        mirroring AlphaBeta's determinism; always lands in ``legal``.

        All-trade prompts (responder ACCEPT/DECLINE; proposer CONFIRM/CANCEL
        when the learner is the responder to a table offer) are OOD for the
        clone — AB games contain no p2p trades, so its logits carry no signal
        on those ids. They are answered by the VALUE head instead: 1-ply
        probe of each response (step, evaluate cp's POV, restore), argmax —
        the learned analogue of ab_decide's leaf-value response and of the
        table TradeAwareAlphaBeta._decide_trade. Ties (ACCEPT vs DECLINE
        before any swap executes) break on the lowest action id, the same
        class of tie behaviour as the native responders."""
        if len(legal) == 1:
            return int(legal[0])
        if all(self._trade_bool[a] for a in legal):
            return self._value_response(game_env, cp, legal)
        if self._net_full:
            game_env.write_obs_full(cp, self._opp_obs)
        else:
            game_env.write_obs(cp, self._opp_obs)
        obs = torch.from_numpy(self._opp_obs).unsqueeze(0).to(self.device)
        logits, _v = self.net(obs)
        row = logits[0].float().cpu().numpy()
        best = max(legal, key=lambda a: row[a])
        return int(best)

    @torch.no_grad()
    def _value_response(self, game_env, cp: int, legal) -> int:
        """Value-head argmax over an all-trade prompt (one batched forward).
        A CONFIRM probe executes the swap, so the proposer side genuinely
        prices the trade; ACCEPT/DECLINE probes tie pre-swap and fall to the
        lowest id, with the real pricing done at the learner's CONFIRM node.
        With a judge set, the probes are priced on full obs by the judge —
        same evaluator as the leaves."""
        vnet = self.judge if self.judge is not None else self.net
        judge_full = (self._judge_full if self.judge is not None
                      else self._net_full)
        obs_w = fastcatan.OBS_FULL_SIZE if judge_full else OBS_SIZE
        snap = game_env.snapshot()
        rows = np.empty((len(legal), obs_w), dtype=np.float32)
        for i, a in enumerate(legal):
            game_env.step(a)
            if judge_full:
                game_env.write_obs_full(cp, rows[i])
            else:
                game_env.write_obs(cp, rows[i])
            game_env.load_snapshot(snap)
        _logits, v = vnet(torch.from_numpy(rows).to(self.device))
        return int(legal[int(np.argmax(v.float().cpu().numpy()))])

    # -------- env reads --------

    def _legal(self, for_learner: bool = False):
        self.env.action_mask(self._mask_buf)
        mask, legal = _unpack(self._mask_buf)
        if self._p2p is not None and not (for_learner and self.learner_trades):
            mask, legal = filter_p2p(mask, self._p2p)
        elif for_learner and self.learner_trades and self.trade_add_cap > 0:
            mask, legal = self._cap_compose(mask, legal)
        return mask, legal

    def _cap_compose(self, mask, legal):
        """Bound the in-tree offer size: once a side of the in-flight offer
        holds `trade_add_cap` cards, that side's ADD ids are filtered.

        Why: an ADD step keeps the clock on the learner — no opponent has
        replied yet — so an uncapped ADD-churn arm inherits the pre-reply
        root value while every real move's Q already carries the opponents'
        replies. Visits sink into compose churn. Short arms force the
        subtree to the OPEN -> responder frontier within the sim budget,
        where trade arms get valued by actual responses."""
        e = self.env
        drop = []
        if sum(e.trade_give(r) for r in range(5)) >= self.trade_add_cap:
            drop += self._add_give
        if sum(e.trade_want(r) for r in range(5)) >= self.trade_add_cap:
            drop += self._add_want
        if not drop:
            return mask, legal
        m2 = mask.copy()
        m2[drop] = False
        l2 = [a for a in legal if m2[a]]
        if not l2:                     # never strand a decision node
            return mask, legal
        return m2, l2

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
            cp = self.env.current_player
            mask, legal = self._legal(for_learner=(cp == self.learner))
            if not legal:
                return True, mask, legal
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
        if self._net_full:
            self.env.write_obs_full(self.learner, self._obs)
        else:
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
        if self.learner_trades and self.trade_prior_frac > 0.0:
            # The IL prior is trade-blind (AB games contain no p2p trades →
            # ~zero softmax mass on compose ids), so PUCT would never explore
            # them. Floor: blend `trade_prior_frac` uniform mass over the
            # LEGAL trade ids whenever a non-trade alternative also exists
            # (pure-trade prompts like ACCEPT/DECLINE already carry all mass).
            tl = node.mask & self._trade_bool
            if tl.any() and bool((node.mask & ~self._trade_bool).any()):
                f = self.trade_prior_frac
                node.P *= np.float32(1.0 - f)
                node.P[tl] += np.float32(f / tl.sum())
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
        if self.judge is not None:
            if self._judge_full:
                self.env.write_obs_full(self.learner, self._full_obs)
                jin = self._full_obs
            else:
                self.env.write_obs(self.learner, self._pov_obs)
                jin = self._pov_obs
            _lg, jv = self.judge(torch.from_numpy(jin)
                                 .unsqueeze(0).to(self.device))
            return float(jv.item())
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

        if self.learner_trades and self.trade_step_cost > 0:
            n_churn = 0
            confirmed = False
            for _nd, a in path:
                if self._churn_bool[a]:
                    n_churn += 1
                elif self._confirm_lo <= a < self._confirm_hi:
                    confirmed = True
            if n_churn and not confirmed:
                value -= self.trade_step_cost * n_churn

        for nd, a in path:
            nd.N[a] += 1
            nd.W[a] += value
            nd.total_N += 1

    # -------- public API --------

    def choose(self, root_snapshot: bytes, temperature: float = 1.0,
               add_root_noise: bool = True):
        self.env.load_snapshot(root_snapshot)
        mask, legal = self._legal(for_learner=True)
        root = Node(root_snapshot, self.learner, mask, legal)
        self._expand(root)
        if add_root_noise:
            self._add_root_noise(root)
        for _ in range(self.sims):
            self._simulate(root)

        # Backed-up root value: mean over all sims of the leaf values that
        # flowed back to the root (sum W / total N). The stage-3 value target
        # (stage3_gen) — a denoised, lookahead-aggregated estimate vs a single
        # sparse outcome. Seat-0 POV, same [-1,1] scale as the leaf.
        self.last_root_value = (float(root.W.sum() / root.total_N)
                                if root.total_N > 0 else 0.0)

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
