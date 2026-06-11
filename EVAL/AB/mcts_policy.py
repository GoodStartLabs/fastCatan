"""State-aware MCTS policy for the catanatron bridge — the M4 instrument.

The bridge's plain PolicyFn is (obs, mask, rng) -> fast-id, which is enough
for a reactive net but not for search: MCTS needs the LIVE game state. This
policy reads the catanatron ``Game`` the bridge stashes on itself each
``decide()`` (``bridge._game``), injects it into a scratch fastcatan Env via
``bridge.state_inject.inject`` (the byte-validated mirror the AB-fidelity
tests use), and runs the hybrid MCTSvsFixed from that snapshot:

    learned prior (IL clone) proposes -> native ab_value judges the leaves
    (deterministic, two-scale lexicographic squash) -> 256-1024 stochastic
    sims out-search AB's fixed 1-2 ply.

This is the configuration that scored 29.0% [25.5-32.8] pooled (>=512 sims,
600 games) vs native AB-d1 — above 4-player parity — and 23.3% vs d2.

The chosen fast-id must land in the BRIDGE's mask (catanatron's
playable_actions reverse-mapped); the root mask of the injected state should
agree (mask-consistency tests), but sub-prompt shapes can differ, so we pick
the highest-visit action within the bridge mask and fall back to rng.choice
as a last resort (counted, reported at teardown).

Wiring (circular by construction):
    policy = MctsStatePolicy(net, seat=k, ...)
    bridge = CatanatronBridge(color, policy=policy, ...)
    policy.bridge = bridge
"""
from __future__ import annotations

import random

import numpy as np
import torch

import fastcatan

from bridge.state_inject import inject
from models.alphazero.mcts import MASK_WORDS
from models.alphazero.mcts_vs_fixed import MCTSvsFixed


class MctsStatePolicy:
    def __init__(
        self,
        net: torch.nn.Module,
        seat: int,
        sims: int = 512,
        leaf_eval: str = "ab_value",
        ab_value_scale: float = 86e6,
        model_ab_depth: int = 1,
        model_ab_prune: bool = False,
        model_catanatron_chance: bool = False,
        opp_model: str = "alphabeta",
        c_puct: float = 1.5,
        seed: int = 0,
        device: str = "cpu",
        enable_trades: bool = False,
        trade_prior_frac: float = 0.05,
        trade_add_cap: int = 3,
        judge: torch.nn.Module | None = None,
    ):
        self.bridge = None              # wired after bridge construction
        self.seat = seat
        self.env = fastcatan.Env()
        # enable_trades: the search composes p2p offers at its OWN nodes
        # (in-tree opponents still never offer — they only respond, like the
        # trade-aware table AB). Keep this in sync with the bridge's
        # enable_trades: trades on at the bridge but off here would leave the
        # compose ids permanently unvisited (zero prior mass).
        self.enable_trades = enable_trades
        self.mcts = MCTSvsFixed(
            net, device=device, sims=sims, c_puct=c_puct,
            dirichlet_frac=0.0, seed=seed, suppress_p2p=True,
            learner_trades=enable_trades, trade_prior_frac=trade_prior_frac,
            trade_add_cap=trade_add_cap,
            ab_depth=model_ab_depth, ab_prune=model_ab_prune,
            catanatron_chance=model_catanatron_chance,
            leaf_eval=leaf_eval, opp_model=opp_model,
            ab_value_scale=ab_value_scale, learner_seat=seat,
            judge=judge)
        self._mask_buf = np.zeros(MASK_WORDS, dtype=np.uint64)
        self.fallbacks = 0
        self.decisions = 0

    def _sync_seat(self, game) -> None:
        """catanatron SHUFFLES seating in State.__init__
        (`random.sample(players, ...)`), so the construction-time seat is
        fiction — derive the agent's TRUE seat from the live game every
        decision. (The g%4 assumption had the search optimizing some
        OPPONENT's position in ~75% of games across bridge runs v1-v5 —
        which pinned them all at ~0.25 x native ≈ 6%.)"""
        seat = int(game.state.color_to_index[self.bridge.color])
        if seat != self.seat:
            self.seat = seat
        self.mcts.learner = seat

    def _legal_bit(self, aid: int) -> bool:
        self.env.action_mask(self._mask_buf)
        return bool((int(self._mask_buf[aid >> 6]) >> (aid & 63)) & 1)

    def _replay_compose_scratch(self) -> bool:
        """Advance the injected root through the bridge's in-flight offer
        scratch (give/want adds stashed by _decide_compose) so the search
        continues the TRUE partial offer instead of restarting composition
        each prompt. Every replayed id is legality-checked against the native
        mask; any mismatch reverts to the fresh pre-compose root."""
        scratch = getattr(self.bridge, "_compose_scratch", None)
        if not scratch:
            return False
        give, want = scratch
        if not (sum(give) or sum(want)):
            return False
        _a = fastcatan.action
        ids = [_a.TRADE_ADD_GIVE_BASE + r for r in range(5)
               for _ in range(give[r])]
        ids += [_a.TRADE_ADD_WANT_BASE + r for r in range(5)
                for _ in range(want[r])]
        snap = self.env.snapshot()
        for aid in ids:
            if not self._legal_bit(aid):
                self.env.load_snapshot(snap)
                self.env.recompute_mask()
                return False
            self.env.step(aid)
        self.env.recompute_mask()
        return True

    def __call__(self, obs: np.ndarray, mask: "list[int]",
                 rng: random.Random) -> int:
        self.decisions += 1
        game = getattr(self.bridge, "_game", None)
        if game is None:                # defensive; should never happen
            self.fallbacks += 1
            return rng.choice(mask)
        self._sync_seat(game)
        inject(self.env, game)            # default actor = current_color ✓
        # state_inject fills the state but not the cached action mask; the
        # MCTS (unlike ab_decide) trusts the cache, and the root SNAPSHOT
        # embeds it — recompute before snapshotting or every root step is a
        # masked-off no-op (the 39/41-fallback smoke bug).
        self.env.recompute_mask()
        replayed = (self._replay_compose_scratch()
                    if self.enable_trades else False)
        action, pi, _root_mask = self.mcts.choose(
            self.env.snapshot(), temperature=0.0, add_root_noise=False)
        if action in mask:
            return int(action)
        if replayed:
            # The mid-compose continuation it chose (e.g. TRADE_CANCEL =
            # abandon the offer) has no bridge id at this prompt. Re-search
            # from the fresh pre-compose root, whose action set projects
            # into the bridge mask (regular moves included).
            inject(self.env, game)
            self.env.recompute_mask()
            action, pi, _root_mask = self.mcts.choose(
                self.env.snapshot(), temperature=0.0, add_root_noise=False)
            if action in mask:
                return int(action)
        # Sub-prompt shape mismatch: best searched action within the
        # bridge's mask.
        best = max(mask, key=lambda f: pi[f])
        if pi[best] > 0:
            return int(best)
        self.fallbacks += 1
        return rng.choice(mask)

    def decide_robber(self, game, move_robbers, color_to_seat, rng):
        """Own the composite (hex, victim) robber decision.

        The bridge's two-call protocol can't give a state-aware policy a
        valid root for the victim sub-pick (the live game still has the
        robber un-moved), which degraded victims to rng.choice — ~0.66
        contested steals per game decided by coin flip in bridge runs v1-v3.
        Here: search the hex on the mirrored state, STEP the mirror through
        the chosen hex, then search the steal from the true post-hex root.
        """
        import fastcatan as _fc
        from bridge import topology_map as _T
        _a = _fc.action

        hex_to_actions = {}
        for a in move_robbers:
            hex_to_actions.setdefault(a.value[0], []).append(a)
        hex_id_to_coord = {
            _a.MOVE_ROBBER_BASE + _T.COORD_TO_FAST_HEX[coord]: coord
            for coord in hex_to_actions
        }
        hex_mask = sorted(hex_id_to_coord.keys())

        self.decisions += 1
        self._sync_seat(game)
        inject(self.env, game)
        self.env.recompute_mask()
        snap = self.env.snapshot()
        action, pi, _m = self.mcts.choose(snap, temperature=0.0,
                                          add_root_noise=False)
        if action not in hex_id_to_coord:
            best = max(hex_mask, key=lambda f: pi[f])
            action = best if pi[best] > 0 else rng.choice(hex_mask)
            if pi[best] <= 0:
                self.fallbacks += 1
        coord = hex_id_to_coord[int(action)]

        candidates = hex_to_actions[coord]
        if len(candidates) == 1:
            return candidates[0]

        # Contested steal: advance the mirror through the chosen hex, then
        # search the victim from the genuine ROBBER_STEAL root.
        victim_to_action = {a.value[1]: a for a in candidates}
        steal_id_to_victim = {
            _a.STEAL_BASE + color_to_seat[v]: v for v in victim_to_action
        }
        steal_mask = sorted(steal_id_to_victim.keys())

        self.decisions += 1
        self.env.load_snapshot(snap)
        self.env.step(int(action))
        self.env.recompute_mask()
        s_action, s_pi, _sm = self.mcts.choose(self.env.snapshot(),
                                               temperature=0.0,
                                               add_root_noise=False)
        if s_action not in steal_id_to_victim:
            best = max(steal_mask, key=lambda f: s_pi[f])
            s_action = best if s_pi[best] > 0 else rng.choice(steal_mask)
            if s_pi[best] <= 0:
                self.fallbacks += 1
        return victim_to_action[steal_id_to_victim[int(s_action)]]
