from ladder.stats import seat_rates


def test_seat_slice_keeps_no_winner_in_each_seat_denominator() -> None:
    rows = [{
        "candidate_seat0_games": 1, "candidate_seat0_wins": 0,
        "candidate_seat1_games": 1, "candidate_seat1_wins": 1,
        "candidate_seat2_games": 1, "candidate_seat2_wins": 0,
        "candidate_seat3_games": 1, "candidate_seat3_wins": 0,
    }]
    observed = seat_rates(rows)
    assert [observed[seat].games for seat in range(4)] == [1, 1, 1, 1]
    assert [observed[seat].rate for seat in range(4)] == [0.0, 1.0, 0.0, 0.0]
