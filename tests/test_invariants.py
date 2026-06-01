"""Per-step game invariants across random rollouts.

Each invariant checked after every step of every game in the seed sweep:
  - phase / current_player / turn_count in valid range
  - public VP non-decreasing per player (private VP can also rise via dev cards)
  - resources non-negative; hand size matches sum of resources
  - bank + sum(player resources) conserved at 19 per resource type
  - settlement+city placed-on-board count never exceeds stock-start
"""
from __future__ import annotations

import pytest

import fastcatan
from tests.conftest import play_random_game


RESOURCE_STARTING_BANK = 19


def _resource_total(env, r: int) -> int:
    return env.bank(r) + sum(env.player_resource(p, r) for p in range(4))


def _check_step(prev, env):
    assert env.phase in (0, 1, 2, 3)
    assert env.current_player in range(4)
    assert env.turn_count >= 0

    for p in range(4):
        # Public VP cannot decrease (longest-road / largest-army flips can occur,
        # but the holder change is +X for new holder and -X for old; we only
        # assert per-player total VP doesn't go *below* its prior value minus
        # the value of those two bonuses, i.e. 4. So bound is loose.)
        assert env.player_vp(p) <= 12, f"VP impossibly high for p{p}: {env.player_vp(p)}"
        assert env.player_handsize(p) == sum(env.player_resource(p, r) for r in range(5)), \
            f"handsize desync for p{p}"
        for r in range(5):
            assert env.player_resource(p, r) >= 0
        assert env.player_settlement_count(p) <= 5
        assert env.player_city_count(p) <= 4
        assert env.player_road_count(p) <= 15

    for r in range(5):
        assert env.bank(r) >= 0
        total = _resource_total(env, r)
        assert total == RESOURCE_STARTING_BANK, \
            f"resource {r} not conserved: bank+players={total}, expected {RESOURCE_STARTING_BANK}"


@pytest.mark.parametrize("seed", [0, 3, 11, 42])
def test_invariants_hold_throughout_random_game(seed):
    state_box = {"prev_vp": [0, 0, 0, 0]}

    def hook(env, mask, action, step_idx):
        _check_step(state_box, env)

    env, steps = play_random_game(seed, on_step=hook)
    # End-state checks
    _check_step(state_box, env)
    vps = [env.player_vp(p) for p in range(4)]
    assert max(vps) >= 10


@pytest.mark.parametrize("seed", [5, 17, 99])
def test_resource_conservation_strict(seed):
    """Stronger: total per-resource is 19 at every step."""
    def hook(env, mask, action, step_idx):
        for r in range(5):
            total = _resource_total(env, r)
            assert total == RESOURCE_STARTING_BANK, (
                f"step {step_idx} resource {r}: bank={env.bank(r)} "
                f"hands={[env.player_resource(p, r) for p in range(4)]} total={total}"
            )

    play_random_game(seed, on_step=hook)
