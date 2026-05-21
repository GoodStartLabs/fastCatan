"""CatanatronBridge: wraps a fastcatan-trained policy as a catanatron Player.

Thesis eval path:
    - Train agent in fastcatan (fast C++ sim).
    - Plug agent's policy fn into this bridge.
    - Run inside catanatron's reference engine vs catanatron baselines
      (AlphaBetaPlayer, ValueFunctionPlayer, etc.).
    - Numbers comparable to Catanatron paper.

Policy signature:
    policy(obs, mask, rng) -> int
        obs:  np.ndarray[float32, (OBS_SIZE,)]   from encode_obs
        mask: list[int]                          sorted list of legal fast IDs
        rng:  random.Random                      seeded per bridge

Default policy is uniform-over-mask (smoke-test pipeline). Plug NN here once
trained.

Trade composition (OFFER_TRADE):
    Catanatron has no atomic-trade-list in playable_actions during PLAY_TURN
    — agents emit OFFER_TRADE freely (validated by is_valid_action). The
    bridge runs a compose sub-loop in PLAY_TURN: maintains a (give5, want5)
    scratch, presents ADD_GIVE/ADD_WANT/OPEN alongside regular PLAY_TURN
    actions on each iteration. When the policy picks OPEN, the assembled
    scratch becomes a single cat OFFER_TRADE Action; when it picks a
    non-trade action (build/end_turn/etc.) the scratch is dropped and that
    action is returned. The compose loop runs entirely inside one decide()
    call, so no cross-call scratch state is required.

Trade resolve:
    DECIDE_TRADE (responder): ACCEPT/REJECT, atomic IDs.
    DECIDE_ACCEPTEES (proposer): CANCEL or CONFIRM_p; bridge maps the
    partner color to fastcatan seat to land on TRADE_CONFIRM_BASE+seat.

Design note:
    No fastcatan.Env state-mirror — fastcatan's dice/dev/steal RNG can't be
    sync'd without a C++ force_dice patch, so the bridge sources obs from
    `encode_obs(game, color)` and the mask from catanatron's
    `playable_actions` reverse-mapped to fast IDs.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

import numpy as np

from catanatron import Color
from catanatron.game import Game
from catanatron.models.enums import Action, ActionPrompt, ActionType
from catanatron.models.player import Player
from catanatron.state_functions import player_has_rolled, player_key

import fastcatan
from bridge import topology_map as T
from bridge.action_codec import (
    RES_FAST_TO_CAT,
    encode_to_fast_ids,
    fast_freqdeck_to_cat,
)
from bridge.obs_encoder import encode_obs


_a = fastcatan.action

_STEAL_NONE_SENTINEL = _a.STEAL_BASE + 4

_COMPOSE_LOOP_CAP = 50


PolicyFn = Callable[[np.ndarray, "list[int]", random.Random], int]


def uniform_policy(obs: np.ndarray, mask: "list[int]", rng: random.Random) -> int:
    """Uniform random over the legal fast-ID mask. Default smoke-test policy."""
    return rng.choice(mask)


class CatanatronBridge(Player):
    """Wraps a fastcatan-style policy as a Catanatron Player.

    `policy` defaults to `uniform_policy` so the bridge plugs in immediately
    for smoke tests / pipeline checks. Swap in an NN-backed policy once the
    agent is trained.

    `enable_trades=True` exposes the OFFER_TRADE compose loop during
    PLAY_TURN. Set False to force the bridge to never propose trades
    (matches `--no-player-trading` training).
    """

    def __init__(self, color: Color, policy: Optional[PolicyFn] = None,
                 seed: int = 0, enable_trades: bool = True):
        super().__init__(color)
        self._rng = random.Random(seed)
        self._policy = policy if policy is not None else uniform_policy
        self._enable_trades = enable_trades

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        prompt = game.state.current_prompt

        # MOVE_ROBBER is a separate prompt — paired sub-decision.
        move_robbers = [a for a in playable_actions
                        if a.action_type == ActionType.MOVE_ROBBER]
        if move_robbers:
            obs = encode_obs(game, self.color)
            return self._decide_move_robber(
                obs, move_robbers, game.state.color_to_index)

        # Trade resolve prompts: responder ACCEPT/REJECT or proposer
        # CANCEL/CONFIRM_p. All atomic.
        if prompt in (ActionPrompt.DECIDE_TRADE, ActionPrompt.DECIDE_ACCEPTEES):
            return self._decide_trade_resolve(game, playable_actions)

        # PLAY_TURN post-roll: optional OFFER_TRADE composition.
        if (self._enable_trades
                and prompt == ActionPrompt.PLAY_TURN
                and player_has_rolled(game.state, self.color)):
            return self._decide_compose(game, playable_actions)

        # Regular path (initial placements, pre-roll, dev plays, discard, ...)
        return self._decide_regular(game, playable_actions)

    # ------------------------------------------------------------------
    # Regular path: 1:1 cat action <-> fast ID mapping.
    # ------------------------------------------------------------------

    def _decide_regular(self, game: Game, playable_actions):
        rep_map: dict[int, Action] = {}
        for a in playable_actions:
            try:
                fids = encode_to_fast_ids(a)
            except (KeyError, ValueError):
                continue
            if not fids:
                continue
            rep_map.setdefault(fids[0], a)

        if not rep_map:
            return self._rng.choice(playable_actions)

        obs = encode_obs(game, self.color)
        mask = sorted(rep_map.keys())
        chosen = self._policy(obs, mask, self._rng)
        if chosen not in rep_map:
            chosen = self._rng.choice(mask)
        return rep_map[chosen]

    # ------------------------------------------------------------------
    # MOVE_ROBBER: 2-step sub-policy (hex then victim).
    # ------------------------------------------------------------------

    def _decide_move_robber(self, obs: np.ndarray, move_robbers, color_to_seat):
        hex_to_actions: dict[tuple, list[Action]] = {}
        for a in move_robbers:
            coord = a.value[0]
            hex_to_actions.setdefault(coord, []).append(a)

        hex_id_to_coord: dict[int, tuple] = {}
        for coord in hex_to_actions:
            fid = _a.MOVE_ROBBER_BASE + T.COORD_TO_FAST_HEX[coord]
            hex_id_to_coord[fid] = coord
        hex_mask = sorted(hex_id_to_coord.keys())

        chosen_hex_id = self._policy(obs, hex_mask, self._rng)
        if chosen_hex_id not in hex_id_to_coord:
            chosen_hex_id = self._rng.choice(hex_mask)
        chosen_coord = hex_id_to_coord[chosen_hex_id]

        candidates = hex_to_actions[chosen_coord]
        if len(candidates) == 1:
            return candidates[0]

        victim_to_action: dict[Optional[Color], Action] = {}
        for a in candidates:
            victim_to_action[a.value[1]] = a

        steal_id_to_victim: dict[int, Optional[Color]] = {}
        for victim in victim_to_action:
            fid = (_STEAL_NONE_SENTINEL if victim is None
                   else _a.STEAL_BASE + color_to_seat[victim])
            steal_id_to_victim[fid] = victim
        steal_mask = sorted(steal_id_to_victim.keys())

        chosen_steal_id = self._policy(obs, steal_mask, self._rng)
        if chosen_steal_id not in steal_id_to_victim:
            chosen_steal_id = self._rng.choice(steal_mask)
        return victim_to_action[steal_id_to_victim[chosen_steal_id]]

    # ------------------------------------------------------------------
    # Trade resolve: DECIDE_TRADE / DECIDE_ACCEPTEES.
    # ------------------------------------------------------------------

    def _decide_trade_resolve(self, game: Game, playable_actions):
        color_to_seat = game.state.color_to_index
        rep_map: dict[int, Action] = {}
        for a in playable_actions:
            at = a.action_type
            if at == ActionType.ACCEPT_TRADE:
                rep_map.setdefault(_a.TRADE_ACCEPT, a)
            elif at == ActionType.REJECT_TRADE:
                rep_map.setdefault(_a.TRADE_DECLINE, a)
            elif at == ActionType.CANCEL_TRADE:
                rep_map.setdefault(_a.TRADE_CANCEL, a)
            elif at == ActionType.CONFIRM_TRADE:
                partner_color = a.value[10]
                partner_seat = color_to_seat[partner_color]
                rep_map.setdefault(_a.TRADE_CONFIRM_BASE + partner_seat, a)
            else:
                try:
                    fids = encode_to_fast_ids(a)
                except (KeyError, ValueError):
                    continue
                if fids:
                    rep_map.setdefault(fids[0], a)

        if not rep_map:
            return self._rng.choice(playable_actions)

        obs = encode_obs(game, self.color)
        mask = sorted(rep_map.keys())
        chosen = self._policy(obs, mask, self._rng)
        if chosen not in rep_map:
            chosen = self._rng.choice(mask)
        return rep_map[chosen]

    # ------------------------------------------------------------------
    # PLAY_TURN with OFFER_TRADE composition.
    # ------------------------------------------------------------------

    def _decide_compose(self, game: Game, playable_actions):
        """Optional OFFER_TRADE compose sub-loop. Each iteration picks one
        ID from {fast-ids of playable_actions} ∪ {ADD_GIVE/ADD_WANT/OPEN}.
        Non-trade pick → return that cat action. ADD_*: update scratch.
        OPEN: assemble cat OFFER_TRADE and return."""
        scratch_give = [0] * 5
        scratch_want = [0] * 5

        # Regular action reps once (don't depend on scratch).
        regular_reps: dict[int, Action] = {}
        for a in playable_actions:
            try:
                fids = encode_to_fast_ids(a)
            except (KeyError, ValueError):
                continue
            if not fids:
                continue
            regular_reps.setdefault(fids[0], a)

        # Player's own resource freqdeck (fastcatan order) — caps ADD_GIVE.
        ps = game.state.player_state
        self_key = player_key(game.state, self.color)
        own_resources_fast = [
            ps[f"{self_key}_{RES_FAST_TO_CAT[r]}_IN_HAND"] for r in range(5)
        ]

        for _ in range(_COMPOSE_LOOP_CAP):
            obs = encode_obs(game, self.color,
                             compose_scratch=(scratch_give, scratch_want))

            mask_set = set(regular_reps.keys())

            # ADD_GIVE legal: scratch.give[r] < own_resources_fast[r].
            for r in range(5):
                if scratch_give[r] < own_resources_fast[r]:
                    mask_set.add(_a.TRADE_ADD_GIVE_BASE + r)

            # ADD_WANT legal: always (bounded by reasonable cap).
            for r in range(5):
                if scratch_want[r] < 19:
                    mask_set.add(_a.TRADE_ADD_WANT_BASE + r)

            # OPEN legal: cat is_valid_trade — both sides non-empty AND
            # no resource appears on both sides.
            if (sum(scratch_give) > 0 and sum(scratch_want) > 0
                    and not any(scratch_give[r] > 0 and scratch_want[r] > 0
                                for r in range(5))):
                mask_set.add(_a.TRADE_OPEN)

            mask = sorted(mask_set)
            chosen = self._policy(obs, mask, self._rng)
            if chosen not in mask_set:
                chosen = self._rng.choice(mask)

            # Terminal: regular cat action.
            if chosen in regular_reps and not self._is_compose_id(chosen):
                return regular_reps[chosen]

            # ADD_GIVE
            if _a.TRADE_ADD_GIVE_BASE <= chosen < _a.TRADE_ADD_GIVE_BASE + 5:
                scratch_give[chosen - _a.TRADE_ADD_GIVE_BASE] += 1
                continue
            # ADD_WANT
            if _a.TRADE_ADD_WANT_BASE <= chosen < _a.TRADE_ADD_WANT_BASE + 5:
                scratch_want[chosen - _a.TRADE_ADD_WANT_BASE] += 1
                continue
            # OPEN -> emit OFFER_TRADE
            if chosen == _a.TRADE_OPEN:
                give_cat = fast_freqdeck_to_cat(scratch_give)
                want_cat = fast_freqdeck_to_cat(scratch_want)
                return Action(self.color, ActionType.OFFER_TRADE,
                              give_cat + want_cat)

            # Fallback: shouldn't reach here.
            return regular_reps.get(chosen, self._rng.choice(playable_actions))

        # Loop cap hit — drop scratch, fall back to regular pick.
        return self._rng.choice(playable_actions)

    @staticmethod
    def _is_compose_id(fid: int) -> bool:
        return (_a.TRADE_ADD_GIVE_BASE <= fid < _a.TRADE_ADD_GIVE_BASE + 5
                or _a.TRADE_ADD_WANT_BASE <= fid < _a.TRADE_ADD_WANT_BASE + 5
                or fid == _a.TRADE_OPEN)
