"""Development-card play timing policies."""

from __future__ import annotations

import fastcatan
from examples.player_base import legal_actions

from ladder.view import PublicView


DEV_PLAY_ACTIONS = {
    fastcatan.action.PLAY_KNIGHT,
    fastcatan.action.PLAY_ROAD_BUILDING,
    *range(fastcatan.action.PLAY_YEAR_OF_PLENTY, fastcatan.action.PLAY_YEAR_OF_PLENTY + 25),
    *range(fastcatan.action.PLAY_MONOPOLY, fastcatan.action.PLAY_MONOPOLY + 5),
}


class DevPolicy:
    TIERS = {"never", "hold-knights", "timed-play"}

    def __init__(self, tier: str):
        if tier not in self.TIERS:
            raise ValueError(f"unknown dev tier: {tier}")
        self.tier = tier

    def choose_main(self, mask, env, view: PublicView, build_policy) -> int | None:
        actions = set(legal_actions(mask)) & DEV_PLAY_ACTIONS
        if not actions or self.tier == "never":
            return None
        if self.tier == "hold-knights":
            if fastcatan.action.PLAY_KNIGHT in actions and (
                env.turn_count >= 24 or view.public_vp[view.seat] < max(view.public_vp)
            ):
                return fastcatan.action.PLAY_KNIGHT
            return None
        if fastcatan.action.PLAY_KNIGHT in actions and (
            view.public_vp[view.seat] <= max(view.public_vp)
        ):
            return fastcatan.action.PLAY_KNIGHT
        if fastcatan.action.PLAY_ROAD_BUILDING in actions:
            return fastcatan.action.PLAY_ROAD_BUILDING
        yop = build_policy.choose_year_of_plenty(mask, view)
        if yop is not None:
            return yop
        monopoly = [
            action for action in actions
            if fastcatan.action.PLAY_MONOPOLY <= action < fastcatan.action.PLAY_MONOPOLY + 5
        ]
        if monopoly and env.turn_count >= 16:
            return max(
                monopoly,
                key=lambda action: sum(
                    view.estimated_hand(player)[action - fastcatan.action.PLAY_MONOPOLY]
                    for player in range(4) if player != view.seat
                ),
            )
        return None
