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
        c_puct: float = 1.5,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.bridge = None              # wired after bridge construction
        self.seat = seat
        self.env = fastcatan.Env()
        self.mcts = MCTSvsFixed(
            net, device=device, sims=sims, c_puct=c_puct,
            dirichlet_frac=0.0, seed=seed, suppress_p2p=True,
            ab_depth=model_ab_depth, ab_prune=model_ab_prune,
            catanatron_chance=model_catanatron_chance,
            leaf_eval=leaf_eval,
            ab_value_scale=ab_value_scale, learner_seat=seat)
        self.fallbacks = 0
        self.decisions = 0

    def __call__(self, obs: np.ndarray, mask: "list[int]",
                 rng: random.Random) -> int:
        self.decisions += 1
        game = getattr(self.bridge, "_game", None)
        if game is None:                # defensive; should never happen
            self.fallbacks += 1
            return rng.choice(mask)
        inject(self.env, game)            # default actor = current_color ✓
        # state_inject fills the state but not the cached action mask; the
        # MCTS (unlike ab_decide) trusts the cache, and the root SNAPSHOT
        # embeds it — recompute before snapshotting or every root step is a
        # masked-off no-op (the 39/41-fallback smoke bug).
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
