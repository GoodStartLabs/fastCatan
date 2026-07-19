"""Robber destination and victim policies."""

from __future__ import annotations

import fastcatan
from examples.player_base import legal_actions

from ladder.topology import DICE_PIPS, HEX_TO_NODES
from ladder.view import PublicView


class RobberPolicy:
    TIERS = {"random", "richest-victim", "leader-blocker"}

    def __init__(self, tier: str):
        if tier not in self.TIERS:
            raise ValueError(f"unknown robber tier: {tier}")
        self.tier = tier

    def choose(self, mask, view: PublicView, rng) -> int:
        actions = legal_actions(mask)
        steals = [
            action for action in actions
            if fastcatan.action.STEAL_BASE <= action < fastcatan.action.STEAL_BASE + 4
        ]
        if steals:
            if self.tier == "random":
                return rng.choice(steals)
            def victim_score(action: int) -> tuple[float, int]:
                victim = action - fastcatan.action.STEAL_BASE
                score = float(view.hand_sizes[victim])
                if self.tier == "leader-blocker":
                    score += 6.0 * view.public_vp[victim]
                return score, -action
            return max(steals, key=victim_score)

        moves = [
            action for action in actions
            if fastcatan.action.MOVE_ROBBER_BASE <= action < fastcatan.action.MOVE_ROBBER_BASE + 19
        ]
        if not moves:
            return rng.choice(actions)
        if self.tier == "random":
            return rng.choice(moves)
        leader = max(
            (player for player in range(4) if player != view.seat),
            key=lambda player: (view.public_vp[player], view.hand_sizes[player], -player),
        )
        scored = []
        for action in moves:
            hex_id = action - fastcatan.action.MOVE_ROBBER_BASE
            resource = view.hex_resources[hex_id]
            pips = DICE_PIPS.get(view.hex_numbers[hex_id], 0)
            target = 0.0
            self_cost = 0.0
            for node in HEX_TO_NODES[hex_id]:
                levels = view.node_levels_by_relative_seat[node]
                for rel, level in enumerate(levels):
                    if level == 0:
                        continue
                    player = (view.seat + rel) & 3
                    if player == view.seat:
                        self_cost += level * pips
                    elif self.tier == "leader-blocker":
                        target += level * pips * (2.0 if player == leader else 0.25)
                    else:
                        target += level * pips * (1.0 + 0.05 * view.hand_sizes[player])
            scored.append((target - 1.5 * self_cost, -action, action))
        return max(scored)[2]
