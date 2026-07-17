"""Native fastcatan probes for the two p2p-trade FSM sub-states the catanatron
differential cannot reach (spec 0.2 §B2).

catanatron models an OFFER atomically, so two fastcatan-internal compose
behaviours have no oracle analogue and are asserted here against fastcatan's own
documented mask semantics (rules.cpp:1755-1787, state.hpp MAX_TRADE_COMPOSE):

  1. the 50-compose churn cap — after MAX_TRADE_COMPOSE_PER_TURN compose actions
     in a turn, the whole compose block (ADD_GIVE / ADD_WANT / OPEN) is masked
     off while CANCEL stays legal, so the mask never empties and a turn cannot
     be stalled by unbounded compose churn;
  2. mid-turn resource invalidation — an assembled offer whose proposer no
     longer holds the offered ``give`` bundle (resources spent/stolen after
     composing) is un-OPEN-able: OPEN and the over-committed ADD_GIVE drop out
     of the mask the instant holdings fall below ``give``.

Run standalone:  PYTHONPATH=EVAL PYTHONHASHSEED=0 python -m bridge.tests.test_trade_fsm
"""
from __future__ import annotations

import numpy as np
import pytest

import fastcatan as fc
from bridge import state_mirror as M

_a = fc.action
GIVE_BASE = _a.TRADE_ADD_GIVE_BASE
WANT_BASE = _a.TRADE_ADD_WANT_BASE
OPEN = _a.TRADE_OPEN
CANCEL = _a.TRADE_CANCEL
MAX_COMPOSE = 50  # MAX_TRADE_COMPOSE_PER_TURN (state.hpp)


def _mask(env):
    m = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
    env.action_mask(m)
    return m


def _bit(m, i):
    return bool(int(m[i // 64]) >> (i % 64) & 1)


def _legal(m):
    out = []
    for w in range(fc.MASK_WORDS):
        bits = int(m[w])
        while bits:
            b = bits & -bits
            out.append(w * 64 + b.bit_length() - 1)
            bits ^= b
    return out


def _any_compose_bit(m):
    return (any(_bit(m, GIVE_BASE + r) for r in range(5))
            or any(_bit(m, WANT_BASE + r) for r in range(5))
            or _bit(m, OPEN))


def _drive_to_compose(seed, max_steps=4000):
    """Random-legal play until the current player can compose (ADD_WANT_0 legal
    == post-roll MAIN, flag NONE). Returns the Env parked in that state."""
    rng = np.random.default_rng(seed)
    env = fc.Env()
    env.reset(seed)
    for _ in range(max_steps):
        m = _mask(env)
        if _bit(m, WANT_BASE):  # compose available
            return env
        legals = _legal(m)
        if not legals:
            break
        env.step(int(legals[rng.integers(len(legals))]))
        if env.phase == 3:
            env.reset(seed + 1)
    raise RuntimeError(f"seed {seed}: never reached a compose-available state")


def _set_res(env, seat, r, n):
    snap = M.parse_snapshot(env.snapshot())
    snap.gs.player_resources[seat][r] = n
    # keep handsize consistent so nothing downstream trips
    hs = sum(snap.gs.player_resources[seat][k] for k in range(5))
    snap.gs.player_handsize[seat] = hs
    env.load_snapshot(M.to_bytes(snap))
    env.recompute_mask()


@pytest.mark.parametrize("seed", [1, 7, 21, 42, 100])
def test_compose_churn_cap_50(seed):
    env = _drive_to_compose(seed)
    m = _mask(env)
    assert _any_compose_bit(m), "baseline: compose block should be open"

    # Spam compose churn: cycle ADD_WANT across resources. Each step increments
    # trade_compose_count (even once want[r] saturates at 19 and the handler
    # no-ops), never rolls/ends the turn, so the cap must eventually bite.
    for k in range(MAX_COMPOSE):
        env.step(int(WANT_BASE + (k % 5)))

    m = _mask(env)
    assert not _any_compose_bit(m), (
        f"seed {seed}: compose block still open after {MAX_COMPOSE} churns "
        f"(legal={_legal(m)})")
    assert _bit(m, CANCEL), "CANCEL must stay legal so the mask never empties"
    assert int(m.sum()) != 0, "mask emptied — turn is stalled"


@pytest.mark.parametrize("seed", [1, 7, 21, 42, 100])
def test_mid_turn_resource_invalidation(seed):
    env = _drive_to_compose(seed)
    seat = env.current_player

    # Guarantee the proposer holds >=3 brick (fast res 0) so we can compose a
    # 2-brick give, and >=1 wool (res 2) for a non-empty want.
    _set_res(env, seat, 0, 3)
    _set_res(env, seat, 2, 1)

    env.step(int(GIVE_BASE + 0))   # give brick -> 1
    env.step(int(GIVE_BASE + 0))   # give brick -> 2
    env.step(int(WANT_BASE + 2))   # want wool -> 1

    snap = M.parse_snapshot(env.snapshot())
    assert snap.gs.trade_give[0] == 2, "compose did not assemble a 2-brick give"
    m = _mask(env)
    assert _bit(m, OPEN), "offer with give<=holdings should be OPEN-able"

    # Mid-turn: proposer's brick drops below the committed give (spent/stolen).
    _set_res(env, seat, 0, 1)      # now hold 1 brick, give still 2

    m = _mask(env)
    assert not _bit(m, OPEN), (
        f"seed {seed}: OPEN still legal with holdings(1) < give(2)")
    assert not _bit(m, GIVE_BASE + 0), (
        "ADD_GIVE(brick) still legal though holdings no longer exceed give")
    # Stepping the (masked) OPEN must be rejected: the offer must NOT go live.
    # (It still bumps trade_compose_count — OPEN is in the compose-count ID
    # range — but that bookkeeping tick is harmless; the trade itself is
    # refused, so flag stays NONE, no proposer is set, and no resources move.)
    pre = M.parse_snapshot(env.snapshot()).gs
    pre_res = [list(pre.player_resources[s]) for s in range(4)]
    env.step(int(OPEN))
    assert env.flag == 0, "invalid OPEN advanced flag to TRADE_PENDING"
    post = M.parse_snapshot(env.snapshot()).gs
    assert post.trade_proposer == 0xFF, "invalid OPEN registered a proposer"
    assert [list(post.player_resources[s]) for s in range(4)] == pre_res, (
        "invalid OPEN moved resources")


if __name__ == "__main__":
    for s in [1, 7, 21, 42, 100]:
        test_compose_churn_cap_50(s)
        test_mid_turn_resource_invalidation(s)
    print("trade FSM probes: churn-cap-50 and mid-turn-invalidation PASS "
          "(seeds 1,7,21,42,100)")
