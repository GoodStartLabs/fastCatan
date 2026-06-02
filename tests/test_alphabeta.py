"""Native expectimax alpha-beta player (Env.ab_decide / Env.ab_value).

Pure-fastcatan checks (no catanatron): the value function is finite and
deterministic, the search returns legal moves without mutating the env, and the
player crushes random play. The *fidelity* of the port to Catanatron's
AlphaBetaPlayer (value function matches base_fn to machine precision, optimal
moves agree) is proven separately in EVAL/AB/test_native_ab_fidelity.py, which
needs catanatron + the bridge.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

import fastcatan
from tests.conftest import legal_actions


def _legal(env) -> list[int]:
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
    env.action_mask(mask)
    return legal_actions(mask)


def _advance(env, rng, n):
    """Step n random actions (resetting if a game ends), leaving a live state."""
    for _ in range(n):
        legal = _legal(env)
        if not legal:
            break
        env.step(rng.choice(legal))
        if env.phase == 3:
            env.reset(rng.getrandbits(64))


def test_ab_value_finite_and_deterministic():
    env = fastcatan.Env()
    env.reset(7)
    _advance(env, random.Random(1), 40)
    for pov in range(4):
        v = env.ab_value(pov)
        assert np.isfinite(v)
        assert env.ab_value(pov) == v  # pure function of state


def test_ab_value_does_not_mutate_state():
    env = fastcatan.Env()
    env.reset(3)
    _advance(env, random.Random(2), 30)
    before = env.snapshot()
    env.ab_value(0)
    env.ab_decide(env.current_player, 2, False)
    assert env.snapshot() == before  # search works on copies


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("prune", [False, True])
def test_ab_decide_returns_legal(depth, prune):
    env = fastcatan.Env()
    env.reset(11)
    rng = random.Random(5)
    for _ in range(400):
        legal = _legal(env)
        if not legal:
            break
        a = env.ab_decide(env.current_player, depth, prune)
        assert a != 0xFFFFFFFF, "ab_decide found no action on a non-empty state"
        assert a in legal, f"ab_decide returned illegal action {a}"
        # advance with a random move so we probe many distinct states
        env.step(rng.choice(legal))
        if env.phase == 3:
            env.reset(rng.getrandbits(64))


def test_ab_decide_single_action_shortcut():
    """A state with exactly one legal action returns it immediately."""
    env = fastcatan.Env()
    env.reset(0)
    # Right after reset the only legal move is the first settlement placements;
    # step until a single-action state appears (e.g. forced sub-phases) or bail.
    rng = random.Random(9)
    for _ in range(300):
        legal = _legal(env)
        if len(legal) == 1:
            assert env.ab_decide(env.current_player, 2, False) == legal[0]
            return
        env.step(rng.choice(legal))
        if env.phase == 3:
            env.reset(rng.getrandbits(64))
    pytest.skip("no single-action state encountered")


def _winrate_vs_random(depth, prune, n_games, seed0=1000):
    rng = random.Random(seed0)
    wins = 0
    for g in range(n_games):
        env = fastcatan.Env()
        env.reset(seed0 + g)
        for _ in range(4000):
            legal = _legal(env)
            if not legal:
                break
            if env.current_player == 0:
                a = env.ab_decide(0, depth, prune)
                if a == 0xFFFFFFFF or a not in legal:
                    a = rng.choice(legal)
            else:
                a = rng.choice(legal)
            env.step(a)
            if env.phase == 3:
                break
        if env.phase == 3 and env.player_vp(0) >= 10:
            wins += 1
    return wins / n_games


def test_ab_depth1_beats_random():
    # Depth 1 == Catanatron ValueFunctionPlayer; vs 3 random seats it should win
    # the large majority (chance baseline is 0.25). Conservative gate.
    wr = _winrate_vs_random(depth=1, prune=False, n_games=30)
    assert wr >= 0.6, f"depth-1 AB only won {wr:.2f} vs random"


def test_ab_depth2_strong():
    wr = _winrate_vs_random(depth=2, prune=False, n_games=20)
    assert wr >= 0.7, f"depth-2 AB only won {wr:.2f} vs random"
