"""True cross-engine differential test: fastcatan vs catanatron.

For every ply of a catanatron game we:
  1. serialize catanatron's PRE-action state into fastcatan's full internal
     state and inject it (``state_inject``),
  2. translate the action catanatron just played into the fastcatan action
     ID(s) (``action_codec`` + robber/steal/confirm fixups),
  3. force fastcatan's RNG so stochastic draws (dice / dev / steal) reproduce
     catanatron's outcome (``rng_force``),
  4. step fastcatan and assert its resulting state equals catanatron's
     post-action state on every rule-meaningful, seat-absolute field.

Unlike ``test_parity_replay`` (which only checks the obs *encoder* reads
catanatron and never runs fastcatan's engine), this proves fastcatan's
``step_one`` *transitions* identically to catanatron — i.e. fastcatan plays
Catan by the same rules.

Turn-pointer bookkeeping (``current_player``, ``turn_count``) and the
forced-RNG ``dice_roll`` carrier are logged but not asserted as rule diffs;
everything else (board, roads, resources, dev cards, bank, deck, awards,
counts, ports, trade scratch) must match exactly.

Diffs are appended to ``bridge/tests/differential_diffs.jsonl``.

Bugs this test found and fixed in the C++ core (see git history):
  * Longest-road title off-by-one — the first player to reach exactly 5 roads
    never received the title (+2 VP).
  * Production bank-shortage — a sole recipient was paid ``min(demand, bank)``;
    catanatron pays nobody when demand exceeds the bank.
  * Road buildability — fast blocked building from a node carrying an enemy
    building even when the player had reached it first.
  * Longest-road history-dependent membership (``road_node_member``).
  * Initial-placement phase mislabel in the obs encoder (road count, not
    settlement count).

Known residuals (NOT fastcatan rule bugs; exercised only by random play, ~1-2%
of games, all on road cuts):
  * Longest-road TITLE tie-break: catanatron reassigns the title to the lowest
    seat index among players tied at the max length on a cut; fastcatan keeps
    it with the incumbent (standard rule). Exempted in ``_exempt_lr_tie_quirk``.
  * Longest-road LENGTH on a cut: catanatron's reported length can lag its own
    fresh recompute (incremental cache, build_settlement only recomputes on a
    2-edge "plow"), and its component membership for enemy-boundary nodes is
    itself history-dependent (incremental ``build_road`` vs ``dfs_walk``
    re-derivation on a cut). These are catanatron-side inconsistencies; the
    standard ``test_differential`` corpus (seeds 0..24) is green.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from catanatron import Color
from catanatron.game import Game
from catanatron.models.player import RandomPlayer
from catanatron.models.enums import ActionType

import fastcatan as fc
from bridge import state_inject as SI
from bridge import state_mirror as M
from bridge import topology_map as T
from bridge.action_codec import encode_to_fast_ids
from bridge.obs_encoder import DEV_TYPES_FAST
from bridge.rng_force import state_for_dice_sum, state_for_band

_a = fc.action
COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
FLAG_ROBBER_STEAL = 3

DIFFS_PATH = Path(__file__).parent / "differential_diffs.jsonl"

# Per-seat byte arrays (width 4) and global byte arrays compared each ply.
# player_road_length is handled separately (both engines cache it lazily; see
# _road_length_diffs) so it is intentionally absent here.
SEAT4_FIELDS = [
    "player_vp", "player_vp_without_dev", "player_handsize", "player_total_dev",
    "player_ports", "player_knights_played",
    "player_settlement_count", "player_city_count", "player_road_count",
    "player_discard_remaining",
]
SCALAR_FIELDS = ["robber_hex", "longest_road_owner", "largest_army_owner",
                 "phase"]


def _field_diffs(fast, cat) -> list[tuple]:
    """Rule-meaningful, seat-absolute field diffs between two CGameState."""
    out: list[tuple] = []

    def cmp(name, A, B, n):
        for i in range(n):
            if A[i] != B[i]:
                out.append((f"{name}[{i}]", int(A[i]), int(B[i])))

    cmp("node", fast.node, cat.node, 54)
    cmp("edge", fast.edge, cat.edge, 72)
    cmp("bank", fast.bank, cat.bank, 5)
    cmp("dev_deck", fast.dev_deck, cat.dev_deck, 5)
    for nm in SEAT4_FIELDS:
        cmp(nm, getattr(fast, nm), getattr(cat, nm), 4)
    for seat in range(4):
        cmp(f"res[{seat}]", fast.player_resources[seat],
            cat.player_resources[seat], 5)
        cmp(f"dev[{seat}]", fast.player_dev[seat], cat.player_dev[seat], 5)
        cmp(f"devpend[{seat}]", fast.player_dev_bought_this_turn[seat],
            cat.player_dev_bought_this_turn[seat], 5)
    for nm in SCALAR_FIELDS:
        if getattr(fast, nm) != getattr(cat, nm):
            out.append((nm, int(getattr(fast, nm)), int(getattr(cat, nm))))

    # Trade scratch only when either engine reports an active proposal.
    if fast.trade_proposer != 0xFF or cat.trade_proposer != 0xFF:
        if fast.trade_proposer != cat.trade_proposer:
            out.append(("trade_proposer", int(fast.trade_proposer),
                        int(cat.trade_proposer)))
        cmp("trade_give", fast.trade_give, cat.trade_give, 5)
        cmp("trade_want", fast.trade_want, cat.trade_want, 5)
        if fast.trade_response != cat.trade_response:
            out.append(("trade_response", int(fast.trade_response),
                        int(cat.trade_response)))
    return out


def _fresh_road_length(game, color) -> int:
    """Catanatron's TRUE longest road via a fresh recompute.

    catanatron's reported ``_LONGEST_ROAD_LENGTH`` is an incrementally cached
    value that is NOT refreshed when an opponent caps a road at a node where
    the player has only one incident road (build_settlement only recomputes on
    a 2-edge "plow"; board.py:121). In that case the cache stays stale while
    fastcatan — which recomputes every step — reports the correct (shorter)
    length. We compare against the fresh recompute so the differential tests
    rule correctness, not catanatron's cache-staleness artifact.
    """
    paths = game.state.board.continuous_roads_by_player(color)
    return max((len(p) for p in paths), default=0)


def _road_length_diffs(fast, gs_post, game) -> list[tuple]:
    """Road-length diffs that are genuine fastcatan bugs.

    Both engines cache longest-road length lazily, so a raw mismatch is not
    automatically a bug. A diff is real ONLY if fastcatan disagrees with BOTH
    catanatron's reported (cached) value AND a fresh recompute — i.e. fastcatan
    computed a length that is simply wrong. When fast == fresh but != cached,
    catanatron's cache is merely stale (board.py:121) and fastcatan is correct.
    """
    out: list[tuple] = []
    for seat in range(4):
        f = fast.player_road_length[seat]
        cached = gs_post.player_road_length[seat]
        if f == cached:
            continue
        fresh = _fresh_road_length(game, game.state.colors[seat])
        if f != fresh:
            out.append((f"player_road_length[{seat}]", int(f), int(cached)))
    return out


def _exempt_lr_tie_quirk(diffs, fast, game):
    """Drop longest-road title/VP diffs caused by catanatron's tie-handling.

    fastcatan keeps the longest-road title with the incumbent on a tie (the
    standard Catan rule). catanatron's build_settlement plow path instead
    reassigns it to the lowest seat index among players tied at the max length
    (board.py: road_color = max(road_lengths.items())). So after a road cut
    that leaves a tie, the two engines can credit the +2 VP to different (tied)
    players.

    When fastcatan's per-seat lengths all equal catanatron's FRESH recompute,
    fastcatan's longest-road computation is provably correct and the only
    difference is this tie attribution — catanatron's non-standard quirk, not a
    fastcatan bug. We drop the longest_road_owner and paired VP diffs in that
    case only. Any underlying length divergence (fast != fresh) is NOT exempted.
    (Project decision: keep fastcatan rule-correct; documented in PLAN.md.)
    """
    if not any(d[0] == "longest_road_owner" for d in diffs):
        return diffs
    lengths_ok = all(
        fast.player_road_length[s] == _fresh_road_length(game, game.state.colors[s])
        for s in range(4)
    )
    if not lengths_ok:
        return diffs
    return [d for d in diffs
            if d[0] != "longest_road_owner" and not d[0].startswith("player_vp")]


def _stolen_resource(gs_pre, gs_post, victim: int) -> int | None:
    """Fast resource index the victim lost between pre and post, or None."""
    for r in range(5):
        if gs_pre.player_resources[victim][r] - gs_post.player_resources[victim][r] == 1:
            return r
    return None


def _translate(action, rec, gs_pre, gs_post, color_to_index):
    """Return (fast_ids, rng_state, post_steal_victim).

    ``rng_state`` forces the upcoming stochastic draw; ``post_steal_victim`` is
    the seat to STEAL from after MOVE_ROBBER if fastcatan enters ROBBER_STEAL.
    """
    at = action.action_type

    if at == ActionType.ROLL:
        d1, d2 = rec.result
        return [_a.ROLL_DICE], state_for_dice_sum(d1 + d2), None

    if at == ActionType.BUY_DEVELOPMENT_CARD:
        drawn = rec.result  # dev type string
        d_idx = DEV_TYPES_FAST.index(drawn)
        lo = sum(gs_pre.dev_deck[0:d_idx])
        hi = lo + gs_pre.dev_deck[d_idx]
        total = sum(gs_pre.dev_deck[0:5])
        return [_a.BUY_DEV], state_for_band(total, lo, hi), None

    if at == ActionType.MOVE_ROBBER:
        coord, victim_color, *_ = action.value
        hexf = T.COORD_TO_FAST_HEX[coord]
        fids = [_a.MOVE_ROBBER_BASE + hexf]
        if victim_color is None:
            return fids, None, None
        victim = color_to_index[victim_color]
        stolen = _stolen_resource(gs_pre, gs_post, victim)
        rng = None
        if stolen is not None:
            total = sum(gs_pre.player_resources[victim][0:5])
            lo = sum(gs_pre.player_resources[victim][0:stolen])
            hi = lo + gs_pre.player_resources[victim][stolen]
            rng = state_for_band(total, lo, hi)
        return fids, rng, victim

    if at == ActionType.CONFIRM_TRADE:
        partner_color = action.value[-1]
        partner = color_to_index[partner_color]
        return [_a.TRADE_CONFIRM_BASE + partner], None, None

    # Everything else is deterministic and 1:1 (or expands) via the codec.
    return encode_to_fast_ids(action), None, None


def _replay_one(seed: int, max_ticks: int = 2000) -> dict:
    # catanatron's RandomPlayer / dice draw from both the stdlib and numpy
    # global RNGs; pin both so the corpus is reproducible across runs.
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    players = [RandomPlayer(c) for c in COLORS]
    game = Game(players, seed=seed)
    cti = game.state.color_to_index

    env = fc.Env()
    env.reset(0)

    records: list[dict] = []
    plies = 0

    for _ in range(max_ticks):
        if game.winning_color() is not None:
            break

        pre_actor = cti[game.state.current_color()]
        gs_pre, board_pre = SI.build_cgs(game, actor_seat=pre_actor)

        rec = game.play_tick()
        action = rec.action
        actor = cti[action.color]

        gs_post, _ = SI.build_cgs(game)  # catanatron POST state (cached lengths)

        fids, rng_state, steal_victim = _translate(
            action, rec, gs_pre, gs_post, cti
        )

        # Inject PRE-state (with the action's actor as current_player) + rng.
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
        diffs = _exempt_lr_tie_quirk(diffs, fast_post, game)
        plies += 1
        if diffs:
            records.append({
                "seed": seed,
                "ply": plies,
                "action": str(action),
                "action_type": action.action_type.name,
                "diffs": diffs[:40],
                "n_diffs": len(diffs),
            })

    return {"seed": seed, "plies": plies, "records": records}


def setup_module(_m):
    if DIFFS_PATH.exists():
        DIFFS_PATH.unlink()


@pytest.mark.parametrize("seed", list(range(25)))
def test_differential(seed):
    result = _replay_one(seed)
    if result["records"]:
        with DIFFS_PATH.open("a") as f:
            for r in result["records"]:
                f.write(json.dumps(r) + "\n")
    n_bad = sum(r["n_diffs"] for r in result["records"])
    n_plies_bad = len(result["records"])
    assert not result["records"], (
        f"seed {seed}: {n_plies_bad}/{result['plies']} plies diverged "
        f"({n_bad} field diffs). See {DIFFS_PATH}. "
        f"First: {result['records'][0]}"
    )
