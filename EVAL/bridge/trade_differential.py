"""Trade-heavy cross-engine differential driver (Phase-0 audit, spec 0.2 §B).

The differential harness in ``tests/test_differential.py`` proves fastcatan's
``step_one`` transitions identically to catanatron for random-play games. But
catanatron's ``generate_playable_actions`` never emits ``OFFER_TRADE`` in the
PLAY_TURN prompt (only maritime trades), so a ``RandomPlayer`` corpus contains
*zero* completed p2p trades — the trade FSM (compose -> open -> respond ->
confirm) is the one rule path with materially less differential mileage.

This driver closes that gap. It reuses the exact parity machinery from
``test_differential`` (``_translate`` / ``_field_diffs`` / ``_road_length_diffs``
/ ``_exempt_lr_cut_quirk``) but drives catanatron with ``TradeHappyPlayer``s
that aggressively OFFER / ACCEPT / CONFIRM / CANCEL. catanatron's
``is_valid_action`` explicitly admits an externally-supplied ``OFFER_TRADE`` in
PLAY_TURN after the roll (game.py:22), so the players can inject offers the
default move generator would never produce. Every ply — including the whole
trade FSM — is mirrored into fastcatan and asserted seat-absolute equal.

Coverage counters (spec §B2) prove which trade sub-states the corpus hit. Two
sub-states are fastcatan-FSM-internal and inexpressible as a single catanatron
action (catanatron offers atomically): the 50-compose churn cap and mid-turn
resource invalidation. Those are exercised by ``trade_fsm_probe`` against
fastcatan's own documented mask semantics, not through the oracle.

Run (box, 14 cores):
    PYTHONPATH=EVAL PYTHONHASHSEED=0 python -m bridge.trade_differential \
        --games 5000 --seed 0 --workers 14 --out /tmp/trade_diff.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from catanatron import Color
from catanatron.game import Game, is_valid_trade
from catanatron.models.enums import Action, ActionPrompt, ActionType, RESOURCES
from catanatron.models.player import Player, RandomPlayer
from catanatron.state_functions import get_player_freqdeck, player_has_rolled

import fastcatan as fc
from bridge import state_inject as SI
from bridge import state_mirror as M
# Reuse the *exact* parity logic the standing gate uses, so "zero unexempted
# divergences" means precisely what it means in test_differential.
from bridge.tests.test_differential import (
    FLAG_ROBBER_STEAL,
    _exempt_lr_cut_quirk,
    _field_diffs,
    _road_length_diffs,
    _translate,
)

_a = fc.action
COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]

# The four trade-scratch bookkeeping fields. Per-ply parity on these is NOT
# achievable against catanatron on the OFFER path: catanatron's apply_offer /
# apply_reject prompt the *proposer* to answer its own offer (a latent bug in a
# path catanatron's own move-gen never exercises), so its all-decline resolves
# one ply later than fastcatan's and its acceptees can include the proposer.
# Everything else — resources, bank, board, roads, dev, awards, VP, handsize,
# counts, ports — is rule-meaningful and MUST match exactly. We bucket
# divergences so "rule-meaningful parity" is measured independently of the
# catanatron trade-bookkeeping quirk.
SCRATCH_FIELDS = {"trade_proposer", "trade_give", "trade_want", "trade_response"}


def _bucket(diffs):
    """Split (name, a, b) diffs into (semantic, scratch_only)."""
    sem, scr = [], []
    for d in diffs:
        base = d[0].split("[")[0]
        (scr if base in SCRATCH_FIELDS else sem).append(d)
    return sem, scr


# ---------------------------------------------------------------------------
# Trade-happy catanatron player
# ---------------------------------------------------------------------------


class TradeHappyPlayer(Player):
    """Aggressively drives the p2p trade FSM so the differential exercises it.

    - PLAY_TURN (after roll): with prob ``p_offer`` inject a valid OFFER_TRADE
      built from *held* cards (giving 1-2 of a held resource, asking 1-2 of a
      different resource), capped at ``max_offers_per_turn`` to bound turn
      length. Otherwise pick a normal action (biased to keep the game moving).
    - DECIDE_TRADE (responder): ACCEPT if legal with prob ``p_accept`` else
      REJECT (so DECLINE-all is reachable).
    - DECIDE_ACCEPTEES (proposer): CONFIRM a random acceptee with prob
      ``p_confirm`` else CANCEL; CANCEL forced when nobody accepted.
    - All other prompts: uniform over playable_actions.
    """

    def __init__(self, color, seed=0, p_offer=0.6, p_accept=0.85,
                 p_confirm=0.85, max_offers_per_turn=3, multi_unit_prob=0.4):
        super().__init__(color)
        self.rng = random.Random((seed * 2654435761) ^ (hash(color) & 0xFFFFFFFF))
        self.p_offer = p_offer
        self.p_accept = p_accept
        self.p_confirm = p_confirm
        self.max_offers_per_turn = max_offers_per_turn
        self.multi_unit_prob = multi_unit_prob
        self._offers_this_turn = 0
        self._last_turn = -1

    def _maybe_offer(self, state):
        held = get_player_freqdeck(state, self.color)  # catanatron RESOURCES order
        have = [i for i, c in enumerate(held) if c > 0]
        if not have:
            return None
        give_i = self.rng.choice(have)
        give_n = 1
        if held[give_i] >= 2 and self.rng.random() < self.multi_unit_prob:
            give_n = 2
        want_choices = [i for i in range(5) if i != give_i]
        want_i = self.rng.choice(want_choices)
        want_n = 2 if self.rng.random() < self.multi_unit_prob else 1
        give = [0] * 5
        want = [0] * 5
        give[give_i] = give_n
        want[want_i] = want_n
        value = tuple(give) + tuple(want)
        if not is_valid_trade(value):
            return None
        return Action(self.color, ActionType.OFFER_TRADE, value)

    def decide(self, game, playable_actions):
        state = game.state
        prompt = state.current_prompt

        if prompt == ActionPrompt.PLAY_TURN and player_has_rolled(state, self.color):
            turn = state.num_turns
            if turn != self._last_turn:
                self._last_turn = turn
                self._offers_this_turn = 0
            if (self._offers_this_turn < self.max_offers_per_turn
                    and self.rng.random() < self.p_offer):
                offer = self._maybe_offer(state)
                if offer is not None:
                    self._offers_this_turn += 1
                    return offer

        if prompt == ActionPrompt.DECIDE_TRADE:
            accepts = [a for a in playable_actions
                       if a.action_type == ActionType.ACCEPT_TRADE]
            rejects = [a for a in playable_actions
                       if a.action_type == ActionType.REJECT_TRADE]
            # Dodge a catanatron bug: apply_offer_trade / apply_reject_trade only
            # exclude the *just-answered* color, not the proposer, so a proposer
            # seated above the responders is asked to DECIDE its own offer. If we
            # accept there, catanatron later emits a self-CONFIRM (proposer ==
            # partner) that fastcatan correctly refuses. A faithful trade actor
            # never answers its own offer — so when we are the proposer, reject.
            ct = state.current_trade
            proposer_seat = ct[10] if (ct and len(ct) >= 11) else None
            if proposer_seat is not None and proposer_seat == state.colors.index(self.color):
                if rejects:
                    return rejects[0]
            if accepts and self.rng.random() < self.p_accept:
                return accepts[0]
            if rejects:
                return rejects[0]
            return self.rng.choice(playable_actions)

        if prompt == ActionPrompt.DECIDE_ACCEPTEES:
            confirms = [a for a in playable_actions
                        if a.action_type == ActionType.CONFIRM_TRADE]
            cancels = [a for a in playable_actions
                       if a.action_type == ActionType.CANCEL_TRADE]
            if confirms and self.rng.random() < self.p_confirm:
                return self.rng.choice(confirms)
            if cancels:
                return cancels[0]
            return self.rng.choice(playable_actions)

        return self.rng.choice(playable_actions)


# ---------------------------------------------------------------------------
# Replay one game, mirror every ply, tally coverage
# ---------------------------------------------------------------------------


COVERAGE_KEYS = [
    "offer_open", "compose_add_give", "compose_add_want",
    "multi_unit_give", "multi_unit_want",
    "accept", "reject", "multi_accept", "decline_all",
    "confirm", "cancel", "completed_trade",
    "confirm_offset_0", "confirm_offset_1", "confirm_offset_2", "confirm_offset_3",
    "unaffordable_offer_seen", "proposer_self_prompted",
]


@dataclass
class GameResult:
    seed: int
    plies: int
    trades_completed: int
    coverage: Counter = field(default_factory=Counter)
    semantic_divergences: list = field(default_factory=list)
    n_scratch_divergent_plies: int = 0


def replay_trade_game(seed: int, max_ticks: int = 4000) -> GameResult:
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed & 0x7FFFFFFF)
    except Exception:
        pass

    players = [TradeHappyPlayer(c, seed=seed + i) for i, c in enumerate(COLORS)]
    game = Game(players, seed=seed)
    cti = game.state.color_to_index

    env = fc.Env()
    env.reset(0)

    cov = Counter()
    sem_divs = []
    n_scratch = 0
    plies = 0
    trades_completed = 0

    for _ in range(max_ticks):
        if game.winning_color() is not None:
            break

        state = game.state
        prompt = state.current_prompt
        pre_actor = cti[state.current_color()]

        # DECIDE_ACCEPTEES *before* the action: count how many accepted (multi
        # ACCEPT is a property of the pre-action state).
        pre_acceptees = None
        if prompt == ActionPrompt.DECIDE_ACCEPTEES:
            pre_acceptees = sum(1 for a in state.acceptees if a)

        # catanatron proposer-self-prompt bug: proposer asked to DECIDE its own
        # offer (finding, dodged by the player). Count how often it happens.
        if prompt == ActionPrompt.DECIDE_TRADE:
            ct = state.current_trade
            if ct and len(ct) >= 11 and ct[10] == pre_actor:
                cov["proposer_self_prompted"] += 1

        gs_pre, board_pre = SI.build_cgs(game, actor_seat=pre_actor)
        pre_res = [list(gs_pre.player_resources[s]) for s in range(4)]

        rec = game.play_tick()
        action = rec.action
        actor = cti[action.color]
        at = action.action_type

        # ---- coverage from the committed action ----
        if at == ActionType.OFFER_TRADE:
            give = action.value[:5]
            want = action.value[5:10]
            cov["offer_open"] += 1
            cov["compose_add_give"] += sum(give)
            cov["compose_add_want"] += sum(want)
            if any(c >= 2 for c in give):
                cov["multi_unit_give"] += 1
            if any(c >= 2 for c in want):
                cov["multi_unit_want"] += 1
            # does the proposer actually hold the offered cards?
            held = get_player_freqdeck(state, action.color)
            if any(held[i] < give[i] for i in range(5)):
                cov["unaffordable_offer_seen"] += 1
        elif at == ActionType.ACCEPT_TRADE:
            cov["accept"] += 1
        elif at == ActionType.REJECT_TRADE:
            cov["reject"] += 1
            # a REJECT that clears the trade == the terminal all-decline
            if not game.state.is_resolving_trade:
                cov["decline_all"] += 1
        elif at == ActionType.CONFIRM_TRADE:
            cov["confirm"] += 1
            partner = cti[action.value[-1]]
            cov[f"confirm_offset_{partner}"] += 1
        elif at == ActionType.CANCEL_TRADE:
            cov["cancel"] += 1

        if prompt == ActionPrompt.DECIDE_ACCEPTEES:
            if pre_acceptees and pre_acceptees >= 2:
                cov["multi_accept"] += 1

        gs_post, _ = SI.build_cgs(game)

        fids, rng_state, steal_victim = _translate(action, rec, gs_pre, gs_post, cti)

        gs_pre.current_player = actor
        gs_pre.discarding_player = actor
        if rng_state is not None:
            gs_pre.rng[:] = rng_state
        snap = M.CSnapshot()
        snap.gs = gs_pre
        snap.board = board_pre
        env.load_snapshot(M.to_bytes(snap))

        for fid in fids:
            env.step(int(fid))
        if steal_victim is not None and env.flag == FLAG_ROBBER_STEAL:
            env.step(int(_a.STEAL_BASE + steal_victim))

        fast_post = SI.read_fast(env)
        diffs = _field_diffs(fast_post, gs_post)
        diffs += _road_length_diffs(fast_post, gs_post, game)
        diffs = _exempt_lr_cut_quirk(diffs, action, fast_post, game)
        plies += 1

        # a completed trade == a CONFIRM that actually moved resources
        if at == ActionType.CONFIRM_TRADE:
            moved = any(pre_res[s] != list(gs_post.player_resources[s]) for s in range(4))
            if moved:
                trades_completed += 1
                cov["completed_trade"] += 1

        if diffs:
            sem, scr = _bucket(diffs)
            if scr and not sem:
                n_scratch += 1
            if sem:
                sem_divs.append({
                    "seed": seed, "ply": plies, "action": str(action),
                    "action_type": at.name, "diffs": sem[:40],
                    "scratch_diffs": scr[:20], "n_semantic": len(sem),
                })

    r = GameResult(seed=seed, plies=plies, trades_completed=trades_completed,
                   coverage=cov, semantic_divergences=sem_divs,
                   n_scratch_divergent_plies=n_scratch)
    return r


def _worker(seed):
    try:
        r = replay_trade_game(seed)
        return {"seed": r.seed, "plies": r.plies,
                "trades_completed": r.trades_completed,
                "coverage": dict(r.coverage),
                "semantic_divergences": r.semantic_divergences,
                "n_scratch_divergent_plies": r.n_scratch_divergent_plies,
                "error": None}
    except Exception as e:  # surface, don't crash the pool
        import traceback
        return {"seed": seed, "plies": 0, "trades_completed": 0,
                "coverage": {}, "semantic_divergences": [],
                "n_scratch_divergent_plies": 0,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=str, default="/tmp/trade_diff.json")
    args = ap.parse_args()

    t0 = time.time()
    seeds = list(range(args.seed, args.seed + args.games))
    agg = Counter()
    total_plies = 0
    games_with_trade = 0
    sem_divergences = []
    n_scratch_plies = 0
    errors = []
    n_done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_worker, seeds, chunksize=8):
            n_done += 1
            if res["error"]:
                errors.append({"seed": res["seed"], "error": res["error"]})
                continue
            total_plies += res["plies"]
            if res["trades_completed"] > 0:
                games_with_trade += 1
            for k, v in res["coverage"].items():
                agg[k] += v
            sem_divergences.extend(res["semantic_divergences"])
            n_scratch_plies += res["n_scratch_divergent_plies"]
            if n_done % 500 == 0:
                print(f"  {n_done}/{args.games} games, "
                      f"{games_with_trade} with a completed trade, "
                      f"SEMANTIC divergent plies={len(sem_divergences)}, "
                      f"scratch-only plies={n_scratch_plies}, "
                      f"{len(errors)} errors, {time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    summary = {
        "games": args.games,
        "games_with_completed_trade": games_with_trade,
        "total_plies": total_plies,
        "coverage": {k: agg.get(k, 0) for k in COVERAGE_KEYS},
        "n_semantic_divergent_plies": len(sem_divergences),
        "n_scratch_only_divergent_plies": n_scratch_plies,
        "semantic_divergences": sem_divergences[:50],
        "n_errors": len(errors),
        "errors": errors[:10],
        "elapsed_s": elapsed,
        "games_per_s": args.games / elapsed if elapsed else 0,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== TRADE DIFFERENTIAL SUMMARY ===")
    print(f"games:                       {args.games}")
    print(f"games w/ completed trade:    {games_with_trade}")
    print(f"total plies mirrored:        {total_plies}")
    print(f"SEMANTIC divergent plies:    {len(sem_divergences)}   (rule-meaningful; MUST be 0)")
    print(f"scratch-only divergent plies:{n_scratch_plies}   (catanatron proposer-self-prompt quirk)")
    print(f"errors:                      {len(errors)}")
    print(f"elapsed:                     {elapsed:.1f}s "
          f"({summary['games_per_s']:.1f} games/s, {args.workers} workers)")
    print("coverage:")
    for k in COVERAGE_KEYS:
        print(f"  {k:26s} {agg.get(k, 0)}")
    if errors:
        print("\nFIRST ERROR:\n", errors[0]["error"][:1500])
    if sem_divergences:
        print("\nFIRST SEMANTIC DIVERGENCE:\n",
              json.dumps(sem_divergences[0], indent=2)[:1500])
    else:
        print("\nNo rule-meaningful divergences: fastcatan's trade path moves "
              "resources/bank/board/dev/awards/VP identically to catanatron.")


if __name__ == "__main__":
    main()
