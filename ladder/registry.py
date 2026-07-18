"""Frozen ladder-v1 roster and construction registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from examples.player_base import Player

from .bridge_bots import (
    CatanatronAlphaBetaD1,
    CatanatronValueFunction,
    CatanatronWeightedRandom,
)
from .config import INCUMBENT
from .persona import Persona, PersonaConfig


@dataclass(frozen=True)
class AgentSpec:
    name: str
    band: str
    factory: Callable[[int], Player]
    promotion_eligible: bool
    anchor: bool = False
    incumbent: bool = False
    smoke: bool = False


PERSONA_CONFIGS = (
    PersonaConfig("random-legal", "random-legal", "random-legal", "random", "never", "none", "decline-all"),
    PersonaConfig("weighted-random", "production-weighted", "weighted-random", "random", "hold-knights", "none", "own-gain-threshold"),
    PersonaConfig("port-rusher", "production-port-synergy", "value-greedy", "richest-victim", "hold-knights", "surplus-dump", "own-gain-threshold"),
    PersonaConfig("builder-basic", "production-weighted", "cheapest-first", "richest-victim", "hold-knights", "none", "own-gain-threshold"),
    PersonaConfig("builder-strong", "production-port-synergy", "value-greedy", "leader-blocker", "timed-play", "targeted-need", "gain-minus-leader-boost", 1.25),
    PersonaConfig("trade-happy", "production-port-synergy", "value-greedy", "leader-blocker", "timed-play", "targeted-need", "own-gain-threshold", 0.20),
    PersonaConfig("trade-averse", "production-port-synergy", "value-greedy", "leader-blocker", "timed-play", "none", "gain-minus-leader-boost", 1.50),
    PersonaConfig("leader-blocker", "production-weighted", "value-greedy", "leader-blocker", "hold-knights", "surplus-dump", "gain-minus-leader-boost", 1.25),
    PersonaConfig("dev-rusher", "production-weighted", "value-greedy", "richest-victim", "timed-play", "none", "own-gain-threshold"),
    PersonaConfig("balanced-strong", "production-port-synergy", "value-greedy", "leader-blocker", "timed-play", "targeted-need", "gain-minus-leader-boost", 0.90),
)


def _persona_factory(config: PersonaConfig) -> Callable[[int], Player]:
    return lambda seed: Persona(config, seed=seed)


_SPECS = [
    AgentSpec(
        config.name,
        "legal-info",
        _persona_factory(config),
        True,
        incumbent=config.name == INCUMBENT,
        smoke=config.name in {
            "random-legal", "builder-basic", "builder-strong",
            "trade-happy", "trade-averse", "balanced-strong",
        },
    )
    for config in PERSONA_CONFIGS
]
_SPECS.extend([
    AgentSpec("catanatron-weighted-random", "bridge-bot", CatanatronWeightedRandom, False, anchor=True),
    AgentSpec("catanatron-value", "bridge-bot", CatanatronValueFunction, False, anchor=True, smoke=True),
    AgentSpec("catanatron-alphabeta-d1", "bridge-bot", CatanatronAlphaBetaD1, False, anchor=True),
])

REGISTRY: dict[str, AgentSpec] = {spec.name: spec for spec in _SPECS}


def register(spec: AgentSpec) -> None:
    if spec.name in REGISTRY:
        raise ValueError(f"duplicate ladder agent: {spec.name}")
    REGISTRY[spec.name] = spec


def candidate_spec(name: str) -> AgentSpec:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown candidate {name!r}; choices={sorted(REGISTRY)}") from exc


def build_agent(name: str, seed: int) -> Player:
    return candidate_spec(name).factory(seed)


def opponent_specs(*, tier: str, exclude: str | None = None) -> list[AgentSpec]:
    specs = [spec for spec in REGISTRY.values() if spec.name != exclude]
    if tier == "smoke":
        specs = [spec for spec in specs if spec.smoke]
    elif tier != "full":
        raise ValueError(f"unknown tier: {tier}")
    return specs
