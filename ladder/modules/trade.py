"""Legal-information p2p trade proposal, response, and confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import fastcatan
from examples.player_base import legal_actions

from ladder.view import PublicView

PURCHASE_COSTS = (
    (1, 1, 0, 0, 0),  # road
    (1, 1, 1, 1, 0),  # settlement
    (0, 0, 0, 2, 3),  # city
    (0, 0, 1, 1, 1),  # development card
)


def purchase_distance(hand: Sequence[float], production: Sequence[float]) -> float:
    """Production-weighted distance to the closest standard purchase."""
    return min(
        sum(max(float(cost[r]) - float(hand[r]), 0.0) / (float(production[r]) + 0.5)
            for r in range(5))
        for cost in PURCHASE_COSTS
    )


def progress_metric(hand: Sequence[float], production: Sequence[float]) -> float:
    return -purchase_distance(hand, production)


@dataclass(frozen=True)
class TradeScore:
    score: float
    own_delta: float
    proposer_delta: float


@dataclass(frozen=True)
class TradeEvaluator:
    leader_lambda: float = 0.75

    def score(
        self,
        *,
        receiver_hand: Sequence[float],
        receiver_production: Sequence[float],
        proposer_hand_estimate: Sequence[float],
        proposer_production: Sequence[float],
        give: Sequence[int],
        want: Sequence[int],
        leader_multiplier: float = 1.0,
    ) -> TradeScore:
        """Score an offer from the receiver's perspective.

        ``give`` is what the proposer transfers to the receiver; ``want`` is
        what the receiver transfers back.  The proposer hand is a public
        estimate, never its hidden composition.
        """
        receiver_after = [
            float(receiver_hand[r]) + int(give[r]) - int(want[r])
            for r in range(5)
        ]
        if min(receiver_after) < 0:
            return TradeScore(float("-inf"), float("-inf"), 0.0)
        proposer_after = [
            max(0.0, float(proposer_hand_estimate[r]) - int(give[r]) + int(want[r]))
            for r in range(5)
        ]
        own_delta = (
            progress_metric(receiver_after, receiver_production)
            - progress_metric(receiver_hand, receiver_production)
        )
        proposer_delta = (
            progress_metric(proposer_after, proposer_production)
            - progress_metric(proposer_hand_estimate, proposer_production)
        )
        score = own_delta - self.leader_lambda * leader_multiplier * proposer_delta
        return TradeScore(score, own_delta, proposer_delta)


class TradePolicy:
    """Stateful one-offer-per-turn controller for the engine's p2p FSM."""

    def __init__(
        self,
        propose_tier: str,
        respond_tier: str,
        *,
        evaluator: TradeEvaluator | None = None,
    ):
        self.propose_tier = propose_tier
        self.respond_tier = respond_tier
        self.evaluator = evaluator or TradeEvaluator()
        self._plan: tuple[int, int] | None = None
        self._plan_turn: int | None = None
        self._offered_turn: int | None = None

    def _new_turn(self, turn: int) -> None:
        if self._plan_turn != turn:
            self._plan = None
            self._plan_turn = turn

    def _choose_plan(self, view: PublicView) -> tuple[int, int] | None:
        hand = view.resources
        production = view.own_production
        best: tuple[float, int, int] | None = None
        for give_resource in range(5):
            reserve = 0 if self.propose_tier == "surplus-dump" else 1
            if hand[give_resource] <= reserve:
                continue
            for want_resource in range(5):
                if want_resource == give_resource:
                    continue
                after = list(hand)
                after[give_resource] -= 1
                after[want_resource] += 1
                gain = progress_metric(after, production) - progress_metric(hand, production)
                surplus = hand[give_resource] - min(cost[give_resource] for cost in PURCHASE_COSTS)
                score = gain + (0.03 * surplus if self.propose_tier == "surplus-dump" else 0.0)
                candidate = (score, -give_resource, -want_resource)
                if best is None or candidate > best:
                    best = candidate
        if best is None or (self.propose_tier == "targeted-need" and best[0] <= 0.0):
            return None
        return -best[1], -best[2]

    def compose_action(self, env, mask, view: PublicView) -> int | None:
        actions = set(legal_actions(mask))
        turn = int(env.turn_count)
        self._new_turn(turn)
        if self.propose_tier == "none" or self._offered_turn == turn:
            if fastcatan.action.TRADE_CANCEL in actions and any(env.trade_give(r) or env.trade_want(r) for r in range(5)):
                return fastcatan.action.TRADE_CANCEL
            return None
        if self._plan is None:
            self._plan = self._choose_plan(view)
        if self._plan is None:
            return None
        give_resource, want_resource = self._plan
        give = [int(env.trade_give(r)) for r in range(5)]
        want = [int(env.trade_want(r)) for r in range(5)]
        expected_give = [0] * 5
        expected_want = [0] * 5
        expected_give[give_resource] = 1
        expected_want[want_resource] = 1
        add_give = fastcatan.action.TRADE_ADD_GIVE_BASE + give_resource
        add_want = fastcatan.action.TRADE_ADD_WANT_BASE + want_resource
        if sum(give) == 0 and add_give in actions:
            return add_give
        if give == expected_give and sum(want) == 0 and add_want in actions:
            return add_want
        if give == expected_give and want == expected_want and fastcatan.action.TRADE_OPEN in actions:
            # The accepted Phase-0 engine auto-clears an all-declined offer and
            # returns here; remembering the OPEN prevents an assumed CANCEL ply.
            self._offered_turn = turn
            return fastcatan.action.TRADE_OPEN
        if fastcatan.action.TRADE_CANCEL in actions:
            self._offered_turn = turn
            return fastcatan.action.TRADE_CANCEL
        return None

    def response_action(self, env, mask, view: PublicView) -> int | None:
        actions = set(legal_actions(mask))
        if fastcatan.action.TRADE_DECLINE not in actions:
            return None
        if self.respond_tier == "decline-all" or fastcatan.action.TRADE_ACCEPT not in actions:
            return fastcatan.action.TRADE_DECLINE
        proposer = view.trade_proposer
        if proposer is None:
            return fastcatan.action.TRADE_DECLINE
        leader = max(view.public_vp)
        leader_multiplier = 1.75 if (
            self.respond_tier == "gain-minus-leader-boost"
            and view.public_vp[proposer] >= leader
        ) else 1.0
        scored = self.evaluator.score(
            receiver_hand=view.resources,
            receiver_production=view.own_production,
            proposer_hand_estimate=view.estimated_hand(proposer),
            proposer_production=view.production[proposer],
            give=[int(env.trade_give(r)) for r in range(5)],
            want=[int(env.trade_want(r)) for r in range(5)],
            leader_multiplier=leader_multiplier,
        )
        threshold = 0.02 if self.respond_tier == "own-gain-threshold" else 0.08
        return (
            fastcatan.action.TRADE_ACCEPT
            if scored.score >= threshold
            else fastcatan.action.TRADE_DECLINE
        )

    def confirm_action(self, mask, view: PublicView) -> int | None:
        confirms = [
            action for action in legal_actions(mask)
            if fastcatan.action.TRADE_CONFIRM_BASE <= action < fastcatan.action.TRADE_CONFIRM_BASE + 4
        ]
        if confirms:
            return min(
                confirms,
                key=lambda action: (
                    view.public_vp[action - fastcatan.action.TRADE_CONFIRM_BASE],
                    action,
                ),
            )
        if fastcatan.action.TRADE_CANCEL in legal_actions(mask):
            return fastcatan.action.TRADE_CANCEL
        return None
