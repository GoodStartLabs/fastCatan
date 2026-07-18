from ladder.modules.trade import TradeEvaluator


def test_clearly_good_trade_is_accepted_by_score() -> None:
    evaluator = TradeEvaluator(leader_lambda=0.25)
    scored = evaluator.score(
        receiver_hand=(2, 0, 1, 1, 0),
        receiver_production=(4, 2, 3, 3, 1),
        proposer_hand_estimate=(0, 2, 0, 1, 2),
        proposer_production=(2, 4, 1, 2, 4),
        give=(0, 0, 0, 0, 1),  # receiver gets ore to complete a dev card
        want=(1, 0, 0, 0, 0),  # receiver gives surplus brick
    )
    assert scored.own_delta > 0
    assert scored.score > 0.02


def test_clearly_bad_trade_is_refused_by_score() -> None:
    evaluator = TradeEvaluator(leader_lambda=0.75)
    scored = evaluator.score(
        receiver_hand=(1, 1, 5, 0, 0),
        receiver_production=(2, 1, 4, 2, 1),
        proposer_hand_estimate=(0, 0, 1, 1, 2),
        proposer_production=(1, 1, 3, 3, 4),
        give=(0, 0, 1, 0, 0),  # receiver gets redundant wool
        want=(0, 1, 0, 0, 0),  # loses lumber that completed a road
    )
    assert scored.own_delta < 0
    assert scored.score < 0


def test_impossible_receiver_payment_is_refused() -> None:
    scored = TradeEvaluator().score(
        receiver_hand=(0, 0, 0, 0, 0),
        receiver_production=(1, 1, 1, 1, 1),
        proposer_hand_estimate=(1, 1, 1, 1, 1),
        proposer_production=(1, 1, 1, 1, 1),
        give=(1, 0, 0, 0, 0),
        want=(0, 1, 0, 0, 0),
    )
    assert scored.score == float("-inf")
