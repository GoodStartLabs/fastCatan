"""One-switch modular legal-information ladder persona."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import fastcatan
from examples.player_base import Player, legal_actions

from .modules import BuildPolicy, DevPolicy, OpeningPolicy, RobberPolicy, TradePolicy
from .modules.dev import DEV_PLAY_ACTIONS
from .view import PublicView


@dataclass(frozen=True)
class PersonaConfig:
    name: str
    opening: str
    build: str
    robber: str
    dev: str
    trade_propose: str
    trade_respond: str
    trade_lambda: float = 0.75


class Persona(Player):
    """Dispatch solely on public phase/flag and the current legal mask."""

    def __init__(self, config: PersonaConfig, seed: int = 0):
        super().__init__(seed=seed)
        self.config = config
        self.name = config.name
        self.seat: int | None = None
        self.opening = OpeningPolicy(config.opening)
        self.build = BuildPolicy(config.build)
        self.robber = RobberPolicy(config.robber)
        self.dev = DevPolicy(config.dev)
        from .modules.trade import TradeEvaluator
        self.trade = TradePolicy(
            config.trade_propose,
            config.trade_respond,
            evaluator=TradeEvaluator(config.trade_lambda),
        )

    def bind_seat(self, seat: int) -> None:
        self.seat = int(seat)

    @staticmethod
    def _mask_without(mask: np.ndarray, actions: set[int]) -> np.ndarray:
        result = mask.copy()
        for action in actions:
            result[action >> 6] &= ~(np.uint64(1) << np.uint64(action & 63))
        return result

    def act(self, env, mask: np.ndarray) -> int:
        actions = legal_actions(mask)
        if not actions:
            raise RuntimeError(f"{self.name} received an empty mask")
        if len(actions) == 1:
            return actions[0]
        seat = int(env.current_player) if self.seat is None else self.seat
        view = PublicView.from_env(env, seat)
        phase = int(env.phase)
        flag = int(env.flag)

        # Initial placement: settlement module, then build module for the road.
        if phase in (0, 1):
            opening = self.opening.choose(mask, view, self.rng)
            return opening if opening is not None else self.build.choose(mask, env, view, self.rng)

        # Forced sub-decisions.
        if flag == 1:  # DISCARD_RESOURCES
            return self.build.discard(mask, view, self.rng)
        if flag in (2, 3):  # MOVE_ROBBER / ROBBER_STEAL
            return self.robber.choose(mask, view, self.rng)
        if flag == 4:  # YEAR_OF_PLENTY
            action = self.build.choose_year_of_plenty(mask, view)
            return action if action is not None else self.rng.choice(actions)
        if flag == 5:  # MONOPOLY
            return self.dev.choose_main(mask, env, view, self.build) or self.rng.choice(actions)
        if flag == 6:  # PLACE_ROAD
            return self.build.choose(mask, env, view, self.rng)
        if flag == 7:  # TRADE_PENDING
            response = self.trade.response_action(env, mask, view)
            if response is not None:
                return response
            confirm = self.trade.confirm_action(mask, view)
            if confirm is not None:
                return confirm
            return self.rng.choice(actions)

        # Normal main turn. Dice is mandatory before all strategic actions.
        if fastcatan.action.ROLL_DICE in actions:
            return fastcatan.action.ROLL_DICE

        confirm = self.trade.confirm_action(mask, view)
        if confirm is not None and any(
            fastcatan.action.TRADE_CONFIRM_BASE <= action < fastcatan.action.TRADE_CONFIRM_BASE + 4
            for action in actions
        ):
            return confirm
        compose = self.trade.compose_action(env, mask, view)
        if compose is not None:
            return compose
        dev_action = self.dev.choose_main(mask, env, view, self.build)
        if dev_action is not None:
            return dev_action

        excluded = set(DEV_PLAY_ACTIONS)
        excluded.update(range(fastcatan.action.TRADE_ADD_GIVE_BASE, fastcatan.action.TRADE_OPEN + 1))
        excluded.update({
            fastcatan.action.TRADE_ACCEPT,
            fastcatan.action.TRADE_DECLINE,
            *range(fastcatan.action.TRADE_CONFIRM_BASE, fastcatan.action.TRADE_CONFIRM_BASE + 4),
            fastcatan.action.TRADE_CANCEL,
        })
        build_mask = self._mask_without(mask, excluded)
        if legal_actions(build_mask):
            return self.build.choose(build_mask, env, view, self.rng)
        return self.rng.choice(actions)
