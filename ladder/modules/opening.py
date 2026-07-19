"""Initial-settlement policies at three frozen strength tiers."""

from __future__ import annotations

import fastcatan
from examples.player_base import legal_actions

from ladder.view import PublicView


class OpeningPolicy:
    TIERS = {"random-legal", "production-weighted", "production-port-synergy"}

    def __init__(self, tier: str):
        if tier not in self.TIERS:
            raise ValueError(f"unknown opening tier: {tier}")
        self.tier = tier

    def choose(self, mask, view: PublicView, rng) -> int | None:
        settlements = [
            action for action in legal_actions(mask)
            if fastcatan.action.SETTLE_BASE <= action < fastcatan.action.SETTLE_BASE + 54
        ]
        if not settlements:
            return None
        if self.tier == "random-legal":
            return rng.choice(settlements)
        scored = []
        for action in settlements:
            node = action - fastcatan.action.SETTLE_BASE
            score = view.node_production_score(node)
            if self.tier == "production-port-synergy":
                score += view.port_synergy(node)
            scored.append((score, -action, action))
        return max(scored)[2]
