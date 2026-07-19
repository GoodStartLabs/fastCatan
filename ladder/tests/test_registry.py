from ladder.registry import PERSONA_CONFIGS, REGISTRY, build_agent, opponent_specs


def test_v1_has_ten_personas_three_bridge_bots_and_four_oracles() -> None:
    assert len(PERSONA_CONFIGS) == 10
    assert sum(spec.band == "legal-info" for spec in REGISTRY.values()) == 10
    assert sum(spec.band == "bridge-bot" for spec in REGISTRY.values()) == 3
    assert sum(spec.band == "oracle" for spec in REGISTRY.values()) == 4
    assert all(spec.promotion_eligible for spec in REGISTRY.values() if spec.band == "legal-info")
    assert not any(spec.promotion_eligible for spec in REGISTRY.values() if spec.band == "bridge-bot")
    assert not any(spec.promotion_eligible for spec in REGISTRY.values() if spec.band == "oracle")


def test_smoke_roster_is_a_stratified_subset() -> None:
    names = {spec.name for spec in opponent_specs(tier="smoke")}
    assert {"random-legal", "builder-strong", "trade-happy", "trade-averse", "catanatron-value"} <= names
    assert len(names) < len(REGISTRY)


def test_every_registered_agent_constructs() -> None:
    for name in REGISTRY:
        assert build_agent(name, 7).name
