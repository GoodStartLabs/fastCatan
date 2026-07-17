from ladder.stats import promotion_metric, rate


def test_no_winner_is_retained_as_loss() -> None:
    observed = rate(wins=2, games=4, no_winner=1)
    assert observed.rate == 0.5
    assert observed.no_winner == 1


def test_promotion_excludes_oracles_and_trading_off() -> None:
    rows = [
        {"candidate_wins": 2, "games": 4, "no_winner": 1,
         "promotion_eligible": True, "mode": "trading_on"},
        {"candidate_wins": 4, "games": 4, "no_winner": 0,
         "promotion_eligible": False, "mode": "trading_on"},
        {"candidate_wins": 4, "games": 4, "no_winner": 0,
         "promotion_eligible": True, "mode": "trading_off"},
    ]
    observed = promotion_metric(rows)
    assert observed.wins == 2
    assert observed.games == 4
    assert observed.no_winner == 1
    assert observed.rate == 0.5
