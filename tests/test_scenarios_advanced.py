"""Deeper scenario coverage: bonus-card transitions, bank exhaustion,
trade flow, dev-VP win path."""
from __future__ import annotations

import random
import numpy as np
import pytest

import fastcatan
from tests.conftest import legal_actions


FLAG_NONE = 0
FLAG_TRADE_PENDING = 7

NO_PLAYER = 0xFF
LONGEST_ROAD_THRESHOLD = 5
LARGEST_ARMY_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Longest road: holder only set when length >= 5; transfer requires strict >.
# ---------------------------------------------------------------------------

def test_longest_road_threshold_and_strict_exceed():
    """Invariants every step:
       - holder != NO_PLAYER => player_road_length[holder] >= 5
       - holder != NO_PLAYER => no other player has length > holder
    """
    fires_holder = 0
    fires_transfer = 0
    seen_transfer_pairs = set()

    for seed in range(15):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
        prev_holder = NO_PLAYER
        prev_len = 0
        for _ in range(200_000):
            env.action_mask(mask)
            holder = env.longest_road_owner
            if holder != NO_PLAYER:
                fires_holder += 1
                hlen = env.player_road_length(holder)
                assert hlen >= LONGEST_ROAD_THRESHOLD, (
                    f"holder p{holder} has len={hlen} < 5"
                )
                for p in range(4):
                    if p == holder:
                        continue
                    other_len = env.player_road_length(p)
                    assert other_len <= hlen, (
                        f"p{p} len={other_len} > holder p{holder} len={hlen}"
                    )
            if holder != prev_holder and holder != NO_PLAYER and prev_holder != NO_PLAYER:
                # Transfer is valid if new holder strictly exceeds the prev
                # holder's CURRENT length (after this step's road updates),
                # OR prev was cut below threshold and new claimant >= 5.
                new_len = env.player_road_length(holder)
                prev_now = env.player_road_length(prev_holder)
                strict_exceed = new_len > prev_now
                prev_cut_below = prev_now < LONGEST_ROAD_THRESHOLD and new_len >= LONGEST_ROAD_THRESHOLD
                assert strict_exceed or prev_cut_below, (
                    f"invalid transfer to p{holder} len={new_len} from p{prev_holder} "
                    f"prev_len={prev_len} prev_now={prev_now}"
                )
                fires_transfer += 1
                seen_transfer_pairs.add((prev_holder, holder))
            prev_holder = holder
            prev_len = env.player_road_length(holder) if holder != NO_PLAYER else 0

            legals = legal_actions(mask)
            env.step(rng.choice(legals))
            if env.phase == 3:
                break
    assert fires_holder > 0, "no game reached longest-road threshold across 15 seeds"


# ---------------------------------------------------------------------------
# Largest army: same shape — threshold 3, strict exceed.
# ---------------------------------------------------------------------------

def test_largest_army_threshold_and_strict_exceed():
    fires_holder = 0
    for seed in range(40):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
        prev_holder = NO_PLAYER
        prev_n = 0
        for _ in range(200_000):
            env.action_mask(mask)
            holder = env.largest_army_owner
            if holder != NO_PLAYER:
                fires_holder += 1
                k = env.player_knights_played(holder)
                assert k >= LARGEST_ARMY_THRESHOLD, (
                    f"largest-army holder p{holder} knights={k} < 3"
                )
                for p in range(4):
                    if p == holder:
                        continue
                    assert env.player_knights_played(p) <= k, (
                        f"p{p} knights > holder p{holder} (no transfer on tie)"
                    )
            if holder != prev_holder and holder != NO_PLAYER and prev_holder != NO_PLAYER:
                new_k = env.player_knights_played(holder)
                assert new_k > prev_n, (
                    f"LA transfer to p{holder} knights={new_k} did not strictly exceed p{prev_holder}={prev_n}"
                )
            prev_holder = holder
            prev_n = env.player_knights_played(holder) if holder != NO_PLAYER else 0

            legals = legal_actions(mask)
            env.step(rng.choice(legals))
            if env.phase == 3:
                break
    # LA is rarer; accept zero fires (no fail), but log via skip if so
    if fires_holder == 0:
        pytest.skip("largest army never claimed in random play across 40 seeds")


# ---------------------------------------------------------------------------
# YOP mask respects bank stock (r1==r2 needs >=2, else >=1 each).
# ---------------------------------------------------------------------------

def test_yop_mask_respects_bank_stock():
    """When YOP flag active, every legal (r1, r2) combo satisfies bank constraints."""
    a = fastcatan.action

    checked = 0
    for seed in range(60):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
        for _ in range(200_000):
            env.action_mask(mask)
            legals = legal_actions(mask)
            if env.flag == 4:  # YEAR_OF_PLENTY
                bank = [env.bank(r) for r in range(5)]
                for aid in legals:
                    assert a.PLAY_YEAR_OF_PLENTY <= aid < a.PLAY_YEAR_OF_PLENTY + 25
                    rel = aid - a.PLAY_YEAR_OF_PLENTY
                    r1, r2 = rel // 5, rel % 5
                    if r1 == r2:
                        assert bank[r1] >= 2, (
                            f"YOP mask offers (r{r1},r{r2}) but bank[{r1}]={bank[r1]}"
                        )
                    else:
                        assert bank[r1] >= 1 and bank[r2] >= 1, (
                            f"YOP mask offers (r{r1},r{r2}) but bank={bank}"
                        )
                checked += 1
                if checked >= 3:
                    return
            env.step(rng.choice(legals))
            if env.phase == 3:
                break
    if checked == 0:
        pytest.skip("YOP never fired across 60 seeds")


# ---------------------------------------------------------------------------
# Bank-empty 4:1/port trade: cannot trade to receive a resource with bank==0.
# ---------------------------------------------------------------------------

def test_trade_to_bank_blocked_when_bank_empty():
    """When bank[get] == 0, no TRADE_BASE+give*4+get action with that `get` is legal."""
    a = fastcatan.action

    saw_empty = 0
    for seed in range(15):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
        for _ in range(200_000):
            env.action_mask(mask)
            legals = set(legal_actions(mask))
            for get in range(5):
                if env.bank(get) == 0:
                    saw_empty += 1
                    # TRADE_BASE = 210, layout: give*5 + get for 25 entries? confirm: 235-210=25
                    for give in range(5):
                        if give == get:
                            continue
                        aid = a.TRADE_BASE + give * 5 + get
                        assert aid not in legals, (
                            f"bank[{get}]=0 but trade action {aid} (give={give},get={get}) is legal"
                        )
            legals_list = list(legals)
            env.step(rng.choice(legals_list))
            if env.phase == 3:
                break
        if saw_empty > 50:
            return
    # Mid-game bank rarely hits 0 with random play; tolerate small fire count.


# ---------------------------------------------------------------------------
# Trade-pending flow: only response-related actions legal during TRADE_PENDING.
# ---------------------------------------------------------------------------

def test_trade_pending_mask_only_allows_response_actions():
    """When flag == TRADE_PENDING, legal actions ⊂ {ACCEPT, DECLINE, CONFIRM_*, CANCEL}."""
    a = fastcatan.action
    allowed = set([a.TRADE_ACCEPT, a.TRADE_DECLINE, a.TRADE_CANCEL])
    allowed.update(range(a.TRADE_CONFIRM_BASE, a.TRADE_CONFIRM_BASE + 4))

    fires = 0
    for seed in range(15):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
        for _ in range(200_000):
            env.action_mask(mask)
            legals = legal_actions(mask)
            if env.flag == FLAG_TRADE_PENDING:
                fires += 1
                for aid in legals:
                    assert aid in allowed, (
                        f"action {aid} legal during TRADE_PENDING (not in {sorted(allowed)})"
                    )
            env.step(rng.choice(legals))
            if env.phase == 3:
                break
        if fires >= 5:
            return
    # Trades may not fire under uniform random; skip rather than fail
    if fires == 0:
        pytest.skip("TRADE_PENDING never fired across 15 seeds")


# ---------------------------------------------------------------------------
# Win via dev-card VP: private VP triggers terminal even when public VP < 10.
# ---------------------------------------------------------------------------

def test_dev_vp_can_close_game():
    """Across many seeds, at least some winners have player_vp > player_vp_public,
    proving dev-VP cards contributed to closing. Verify in those games:
       - winner's private VP >= 10
       - public VP may be < 10
    """
    saw_dev_close = 0
    saw_pure_public = 0
    for seed in range(40):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
        for _ in range(200_000):
            env.action_mask(mask)
            legals = legal_actions(mask)
            env.step(rng.choice(legals))
            if env.phase == 3:
                break
        winner = next(p for p in range(4) if env.player_vp(p) >= 10)
        priv = env.player_vp(winner)
        pub = env.player_vp_public(winner)
        assert priv >= 10
        assert pub <= priv, "public VP should never exceed private VP"
        if pub < 10:
            saw_dev_close += 1
        else:
            saw_pure_public += 1

    assert saw_pure_public + saw_dev_close == 40
    # We don't strictly require a dev-close win in every run (statistical),
    # but across 40 seeds it should fire at least once.
    if saw_dev_close == 0:
        pytest.skip("no dev-VP win across 40 seeds — bump sample size or accept")


# ---------------------------------------------------------------------------
# VP card ownership: player_vp - player_vp_public equals count of revealed VP
# cards (which is == dev_VP holdings post-win since all hidden VPs reveal).
# Sanity: difference is in [0, 5] (5 VP cards total).
# ---------------------------------------------------------------------------

def test_private_vs_public_vp_difference_bounded():
    for seed in range(8):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
        for _ in range(200_000):
            env.action_mask(mask)
            for p in range(4):
                diff = env.player_vp(p) - env.player_vp_public(p)
                assert 0 <= diff <= 5, (
                    f"p{p} vp={env.player_vp(p)} public={env.player_vp_public(p)} diff={diff}"
                )
            env.step(rng.choice(legal_actions(mask)))
            if env.phase == 3:
                break
