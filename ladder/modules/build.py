"""Build, discard, bank-trade, and forced resource choice policies."""

from __future__ import annotations

import fastcatan
from examples.player_base import legal_actions

from ladder.modules.trade import progress_metric
from ladder.view import PublicView


def _bank_ratio(env, seat: int, resource: int) -> int:
    ports = int(env.player_ports(seat))
    if ports & (1 << resource):
        return 2
    if ports & (1 << 5):
        return 3
    return 4


class BuildPolicy:
    TIERS = {"random-legal", "cheapest-first", "value-greedy", "weighted-random"}

    def __init__(self, tier: str):
        if tier not in self.TIERS:
            raise ValueError(f"unknown build tier: {tier}")
        self.tier = tier

    def discard(self, mask, view: PublicView, rng) -> int:
        actions = [
            action for action in legal_actions(mask)
            if fastcatan.action.DISCARD_BASE <= action < fastcatan.action.DISCARD_BASE + 5
        ]
        if self.tier == "random-legal":
            return rng.choice(actions)
        scored = []
        for action in actions:
            resource = action - fastcatan.action.DISCARD_BASE
            scarcity = 1.0 / (view.own_production[resource] + 0.5)
            scored.append((view.resources[resource] - scarcity, -action, action))
        return max(scored)[2]

    def choose_year_of_plenty(self, mask, view: PublicView) -> int | None:
        actions = [
            action for action in legal_actions(mask)
            if fastcatan.action.PLAY_YEAR_OF_PLENTY <= action < fastcatan.action.PLAY_YEAR_OF_PLENTY + 25
        ]
        if not actions:
            return None
        base = progress_metric(view.resources, view.own_production)
        scored = []
        for action in actions:
            offset = action - fastcatan.action.PLAY_YEAR_OF_PLENTY
            first, second = divmod(offset, 5)
            hand = list(view.resources)
            hand[first] += 1
            hand[second] += 1
            scored.append((progress_metric(hand, view.own_production) - base, -action, action))
        return max(scored)[2]

    def _score(self, action: int, env, view: PublicView) -> float:
        a = fastcatan.action
        if a.CITY_BASE <= action < a.CITY_BASE + 54:
            node = action - a.CITY_BASE
            return 120.0 + view.node_production_score(node, city_multiplier=1)
        if a.SETTLE_BASE <= action < a.SETTLE_BASE + 54:
            node = action - a.SETTLE_BASE
            return 80.0 + view.node_production_score(node) + view.port_synergy(node)
        if a.ROAD_BASE <= action < a.ROAD_BASE + 72:
            return 18.0 + 0.15 * int(env.player_road_length(view.seat))
        if action == a.BUY_DEV:
            return 42.0 + max(view.public_vp) - view.public_vp[view.seat]
        if a.TRADE_BASE <= action < a.TRADE_BASE + 25:
            give, receive = divmod(action - a.TRADE_BASE, 5)
            ratio = _bank_ratio(env, view.seat, give)
            hand = list(view.resources)
            hand[give] -= ratio
            hand[receive] += 1
            gain = (
                progress_metric(hand, view.own_production)
                - progress_metric(view.resources, view.own_production)
            )
            return 30.0 * gain - 2.0
        if action == a.END_TURN:
            return -1.0
        return 0.0

    def choose(self, mask, env, view: PublicView, rng) -> int:
        actions = list(legal_actions(mask))
        if not actions:
            raise RuntimeError("build module received an empty mask")
        if self.tier == "random-legal":
            return rng.choice(actions)
        if self.tier == "weighted-random":
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
            return rng.choices(actions, weights=weights, k=1)[0]
        if self.tier == "cheapest-first":
            def priority(action: int) -> tuple[int, int]:
                a = fastcatan.action
                if a.ROAD_BASE <= action < a.ROAD_BASE + 72:
                    return (5, -action)
                if action == a.BUY_DEV:
                    return (4, -action)
                if a.SETTLE_BASE <= action < a.SETTLE_BASE + 54:
                    return (3, -action)
                if a.CITY_BASE <= action < a.CITY_BASE + 54:
                    return (2, -action)
                if a.TRADE_BASE <= action < a.TRADE_BASE + 25:
                    return (1, -action)
                return (0, -action)
            return max(actions, key=priority)
        return max(actions, key=lambda action: (self._score(action, env, view), -action))
