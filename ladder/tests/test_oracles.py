import numpy as np

import fastcatan
from examples.player_base import legal_actions
from ladder.oracles import OracleAlphaBeta, OracleMCTS


def _mask(env):
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
    env.action_mask(mask)
    return mask


def test_ab_chance_twins_return_legal_opening_action() -> None:
    for chance_mode in (0, 1):
        env = fastcatan.Env()
        env.reset(123)
        player = OracleAlphaBeta(name="test", chance_mode=chance_mode, seed=1)
        player.bind_seat(0)
        action = player.act(env, _mask(env))
        assert action in legal_actions(_mask(env))


def test_mcts_abvalue_smoke_returns_legal_action() -> None:
    env = fastcatan.Env()
    env.reset(321)
    player = OracleMCTS(name="test-mcts", sims=4, seed=2)
    player.bind_seat(0)
    player.set_trading_mode(False)
    action = player.act(env, _mask(env))
    assert action in legal_actions(_mask(env))
