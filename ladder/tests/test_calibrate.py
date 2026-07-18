from ladder.calibrate import _inversions, _matrix_and_ranking


def _row(candidate, opponent, wins, mode="trading_on"):
    return {
        "candidate": candidate,
        "opponent": opponent,
        "mode": mode,
        "candidate_wins": wins,
        "games": 4,
        "no_winner": 0,
    }


def test_matrix_and_rating_keep_directed_matchups() -> None:
    agents = ["random-legal", "weighted-random"]
    rows = [
        _row("random-legal", "weighted-random", 0),
        _row("weighted-random", "random-legal", 3),
        _row("random-legal", "weighted-random", 1, "trading_off"),
        _row("weighted-random", "random-legal", 2, "trading_off"),
    ]
    result = _matrix_and_ranking(rows, agents)
    assert result["matrices"]["trading_on"]["random-legal"]["weighted-random"] == 0.0
    assert result["rankings"]["trading_on"][0]["agent"] == "weighted-random"
    assert _inversions(result, agents) == []
