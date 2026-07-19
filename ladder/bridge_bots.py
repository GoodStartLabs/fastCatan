"""Native fastCatan ports of the three unmodified catanatron bridge bots."""

from __future__ import annotations

import numpy as np

import fastcatan
from examples.player_base import Player, build_p2p_trade_filter, legal_actions

P2P_FILTER = build_p2p_trade_filter()


def _is_forbidden(mask: np.ndarray, action: int) -> bool:
    return bool((int(mask[action >> 6]) >> (action & 63)) & 1)


class CatanatronWeightedRandom(Player):
    """Action-type weights from catanatron 3.3.0 WeightedRandomPlayer."""

    name = "catanatron-weighted-random"

    def bind_seat(self, seat: int) -> None:
        self.seat = int(seat)

    def act(self, env, mask: np.ndarray) -> int:
        actions = legal_actions(mask)
        if fastcatan.action.TRADE_DECLINE in actions:
            return fastcatan.action.TRADE_DECLINE
        actions = legal_actions(mask & ~P2P_FILTER) or actions
        weights = []
        for action in actions:
            if fastcatan.action.CITY_BASE <= action < fastcatan.action.CITY_BASE + 54:
                weights.append(10_000)
            elif fastcatan.action.SETTLE_BASE <= action < fastcatan.action.SETTLE_BASE + 54:
                weights.append(1_000)
            elif action == fastcatan.action.BUY_DEV:
                weights.append(100)
            else:
                weights.append(1)
        return self.rng.choices(actions, weights=weights, k=1)[0]


class CatanatronValueFunction(Player):
    """Catanatron ValueFunctionPlayer: one executed action then base_fn."""

    name = "catanatron-value"

    def bind_seat(self, seat: int) -> None:
        self.seat = int(seat)

    def act(self, env, mask: np.ndarray) -> int:
        actions = legal_actions(mask)
        if fastcatan.action.TRADE_DECLINE in actions:
            return fastcatan.action.TRADE_DECLINE
        actions = legal_actions(mask & ~P2P_FILTER) or actions
        if len(actions) == 1:
            return actions[0]
        pov = getattr(self, "seat", int(env.current_player))
        snapshot = env.snapshot()
        best_action = actions[0]
        best_value = float("-inf")
        for action in actions:
            env.step(action)
            value = float(env.ab_value(pov))
            env.load_snapshot(snapshot)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action


class CatanatronAlphaBetaD1(Player):
    """Native faithful AlphaBetaPlayer(depth=1, catanatron chance blur)."""

    name = "catanatron-alphabeta-d1"

    def __init__(self, seed: int = 0):
        super().__init__(seed=seed)
        self.banned = P2P_FILTER

    def bind_seat(self, seat: int) -> None:
        self.seat = int(seat)

    def act(self, env, mask: np.ndarray) -> int:
        actions = legal_actions(mask)
        if fastcatan.action.TRADE_DECLINE in actions:
            return fastcatan.action.TRADE_DECLINE
        pov = getattr(self, "seat", int(env.current_player))
        action = int(env.ab_decide(pov, 1, False, self.banned, 1))
        if action == int(fastcatan.SKIP_ACTION) or _is_forbidden(self.banned, action):
            allowed = legal_actions(mask & ~self.banned)
            return allowed[0] if allowed else actions[0]
        return action
