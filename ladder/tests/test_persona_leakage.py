from ladder.leakage_referee import run


def test_personas_ignore_hidden_opponent_perturbations() -> None:
    result = run(games=2, seed=19, sample_every=40)
    assert sum(result["perturbations"].values()) > 0
    assert result["persona_decisions_compared"] > 0
    assert result["findings"] == []
    assert result["n_findings"] == 0
