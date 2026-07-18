"""Action-profile analyzer for goodSettler checkpoints and ladder agents.

Research surface (models/analysis/ is NOT frozen). Plays a *subject* agent for
N seat-balanced games against a fixed opponent set and logs, by game phase
(opening / mid / late), the subject's action-type distribution, trade behaviour
(proposals per game, confirm/cancel rates, give/want composition, a resource-EV
proxy), robber-steal targeting (does it hit the VP leader?), and where its
losses come from (final VP gaps, which opponent won, per-seat win share).

The subject may be:
  * ``il``            -> models/checkpoints/phase2_il_100k/il_policy.zip (the
                         incumbent, sampled like the eval harness), or
  * any ladder name   -> persona / oracle via ladder.registry.build_agent.

Every agent is driven through the ``examples.player_base.Player.act(env, mask)``
seam, so the driver is a faithful copy of ``ladder.match.play_one`` (including
its discard-seat routing) plus a per-decision recorder and terminal VP capture.
Board/seat/seed discipline mirrors ``ladder.match.play_rotation_block`` via
``ladder.seeds`` so profiles are comparable across subjects.

Usage (on the box, venv active, PYTHONHASHSEED=0):
    python -m models.analysis.action_profile --subject il --games 120 \
        --opponents balanced-strong,builder-strong,catanatron-value,oracle-ab-d2

Outputs a JSON profile to models/analysis/out/ and, unless --no-wandb, logs
tables to W&B project goodsettler-eval (entity good-start-labs).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import fastcatan
from examples.player_base import Player, build_p2p_trade_filter, legal_actions
from ladder.registry import build_agent
from ladder.seeds import derive_board_seed, derive_policy_seed
from ladder.config import MASTER_SEED
from models.eval import wilson_ci

A = fastcatan.action
NUM_ACTIONS = int(fastcatan.NUM_ACTIONS)
RES = ["brick", "lumber", "wool", "grain", "ore"]

# ---- action classification -------------------------------------------------
# Counts per block (from substrate §2 / topology constants).
_SETTLE, _CITY, _ROAD = int(A.SETTLE_BASE), int(A.CITY_BASE), int(A.ROAD_BASE)
_DISCARD, _MOVEROB, _STEAL = int(A.DISCARD_BASE), int(A.MOVE_ROBBER_BASE), int(A.STEAL_BASE)
_TRADEBANK = int(A.TRADE_BASE)
_ADDGIVE, _ADDWANT = int(A.TRADE_ADD_GIVE_BASE), int(A.TRADE_ADD_WANT_BASE)
_CONFIRM = int(A.TRADE_CONFIRM_BASE)
_YOP, _MONO = int(A.PLAY_YEAR_OF_PLENTY), int(A.PLAY_MONOPOLY)
_ROLL, _END, _BUYDEV = int(A.ROLL_DICE), int(A.END_TURN), int(A.BUY_DEV)
_KNIGHT, _ROADB = int(A.PLAY_KNIGHT), int(A.PLAY_ROAD_BUILDING)
_OPEN, _ACCEPT, _DECLINE, _CANCEL = (
    int(A.TRADE_OPEN), int(A.TRADE_ACCEPT), int(A.TRADE_DECLINE), int(A.TRADE_CANCEL),
)


def fine_type(aid: int) -> str:
    if _SETTLE <= aid < _SETTLE + 54:
        return "build-settle"
    if _CITY <= aid < _CITY + 54:
        return "build-city"
    if _ROAD <= aid < _ROAD + 72:
        return "build-road"
    if aid == _ROLL:
        return "roll"
    if aid == _END:
        return "end-turn"
    if _DISCARD <= aid < _DISCARD + 5:
        return "discard"
    if _MOVEROB <= aid < _MOVEROB + 19:
        return "robber-move"
    if _STEAL <= aid < _STEAL + 4:
        return "robber-steal"
    if _TRADEBANK <= aid < _TRADEBANK + 25:
        return "trade-bank"
    if aid == _BUYDEV:
        return "buy-dev"
    if aid == _KNIGHT:
        return "dev-knight"
    if aid == _ROADB:
        return "dev-road-building"
    if _YOP <= aid < _YOP + 25:
        return "dev-year-of-plenty"
    if _MONO <= aid < _MONO + 5:
        return "dev-monopoly"
    if _ADDGIVE <= aid < _ADDGIVE + 5:
        return "trade-add-give"
    if _ADDWANT <= aid < _ADDWANT + 5:
        return "trade-add-want"
    if aid == _OPEN:
        return "trade-open"
    if aid == _ACCEPT:
        return "trade-accept"
    if aid == _DECLINE:
        return "trade-decline"
    if _CONFIRM <= aid < _CONFIRM + 4:
        return "trade-confirm"
    if aid == _CANCEL:
        return "trade-cancel"
    return f"unknown-{aid}"


_COARSE = {
    "build-settle": "build", "build-city": "build", "build-road": "build",
    "roll": "roll", "end-turn": "end",
    "discard": "robber", "robber-move": "robber", "robber-steal": "robber",
    "trade-bank": "trade-bank",
    "buy-dev": "dev", "dev-knight": "dev", "dev-road-building": "dev",
    "dev-year-of-plenty": "dev", "dev-monopoly": "dev",
    "trade-add-give": "trade-p2p", "trade-add-want": "trade-p2p",
    "trade-open": "trade-p2p", "trade-accept": "trade-p2p",
    "trade-decline": "trade-p2p", "trade-confirm": "trade-p2p",
    "trade-cancel": "trade-p2p",
}


def coarse_type(aid: int) -> str:
    return _COARSE.get(fine_type(aid), "other")


def phase_bucket(env) -> str:
    ph = int(env.phase)
    if ph in (0, 1):  # INITIAL_PLACEMENT_1 / _2
        return "opening"
    leader_vp = max(int(env.player_vp_public(p)) for p in range(4))
    return "late" if leader_vp >= 8 else "mid"


# ---- per-subject accumulators ---------------------------------------------
@dataclass
class Profile:
    subject: str
    games: int = 0
    decided: int = 0
    wins: int = 0
    no_winner: int = 0
    seat_games: list = field(default_factory=lambda: [0, 0, 0, 0])
    seat_wins: list = field(default_factory=lambda: [0, 0, 0, 0])
    # action-type counts by phase bucket
    fine_by_phase: dict = field(default_factory=lambda: defaultdict(Counter))
    coarse_by_phase: dict = field(default_factory=lambda: defaultdict(Counter))
    subject_decisions: int = 0
    # trades
    opens: int = 0
    confirms: int = 0
    cancels: int = 0
    bank_trades: int = 0
    give_hist: Counter = field(default_factory=Counter)   # resource -> units offered
    want_hist: Counter = field(default_factory=Counter)   # resource -> units requested
    confirm_give_units: int = 0
    confirm_want_units: int = 0
    confirm_ev_uniform: float = 0.0  # sum over confirmed trades of (|want|-|give|)
    # robber
    steals: int = 0
    steal_on_leader: int = 0
    steal_victim_vp_rank: Counter = field(default_factory=Counter)  # 0=leader..3
    robber_hex_hist: Counter = field(default_factory=Counter)
    # outcomes / loss attribution
    per_opp: dict = field(default_factory=lambda: defaultdict(lambda: {"g": 0, "w": 0, "nw": 0}))
    win_final_vp: Counter = field(default_factory=Counter)
    loss_vp_gap: Counter = field(default_factory=Counter)  # winner_vp - subject_vp
    loss_subject_vp: Counter = field(default_factory=Counter)
    loser_to_winner_opp: Counter = field(default_factory=Counter)
    # incoherence probe: does a trade acquisition get spent on a build within N turns?
    coh_acquire_events: int = 0
    coh_covered_by_build: int = 0   # acquisitions followed by a build within N turns
    coh_builds_total: int = 0
    coh_end_hand_total: int = 0     # subject hand size at game end (hoarding signal)
    coh_games: int = 0


def record_decision(prof: Profile, env, seat: int, action: int) -> None:
    prof.subject_decisions += 1
    ph = phase_bucket(env)
    ft = fine_type(action)
    prof.fine_by_phase[ph][ft] += 1
    prof.coarse_by_phase[ph][coarse_type(action)] += 1

    if action == _OPEN or (_ADDGIVE <= action < _ADDGIVE + 5) or (_ADDWANT <= action < _ADDWANT + 5):
        pass
    if action == _OPEN:
        prof.opens += 1
        give = [int(env.trade_give(r)) for r in range(5)]
        want = [int(env.trade_want(r)) for r in range(5)]
        for r in range(5):
            prof.give_hist[RES[r]] += give[r]
            prof.want_hist[RES[r]] += want[r]
    elif _CONFIRM <= action < _CONFIRM + 4:
        prof.confirms += 1
        give = sum(int(env.trade_give(r)) for r in range(5))
        want = sum(int(env.trade_want(r)) for r in range(5))
        prof.confirm_give_units += give
        prof.confirm_want_units += want
        prof.confirm_ev_uniform += (want - give)
    elif action == _CANCEL:
        prof.cancels += 1
    elif _TRADEBANK <= action < _TRADEBANK + 25:
        prof.bank_trades += 1
    elif _STEAL <= action < _STEAL + 4:
        prof.steals += 1
        victim = action - _STEAL
        vps = [int(env.player_vp_public(p)) for p in range(4)]
        leader_vp = max(vps)
        if vps[victim] == leader_vp:
            prof.steal_on_leader += 1
        # rank of victim by VP (0 = highest)
        rank = sorted(range(4), key=lambda p: -vps[p]).index(victim)
        prof.steal_victim_vp_rank[rank] += 1
    elif _MOVEROB <= action < _MOVEROB + 19:
        prof.robber_hex_hist[action - _MOVEROB] += 1


# ---- instrumented driver (faithful copy of ladder.match.play_one) ----------
def profiled_play_one(env, seat_policies, subject_seat, prof, suppress_p2p, max_steps):
    mask_buf = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
    p2p = build_p2p_trade_filter() if suppress_p2p else None
    decisions = 0
    done = False
    discarding_seat = None
    obs_buf = np.zeros(fastcatan.OBS_SIZE, dtype=np.float32)
    # coherence (incoherence probe) per-game state for the subject
    COH_N = 2  # subject-turns window
    subj_turn = 0
    acquire_turns: list[int] = []
    build_turns: list[int] = []

    while not done and decisions < max_steps:
        turn_owner = int(env.current_player)
        if int(env.flag) == 1:  # DISCARD
            env.write_obs(turn_owner, obs_buf)
            remaining = [0, 0, 0, 0]
            for relative in range(4):
                absolute = (turn_owner + relative) & 3
                remaining[absolute] = int(round(float(obs_buf[relative * 16 + 14]) * 10))
            if discarding_seat is None:
                discarding_seat = next((p for p in range(4) if remaining[p] > 0), None)
            elif remaining[discarding_seat] == 0:
                discarding_seat = next(
                    ((turn_owner + off) & 3 for off in range(1, 5)
                     if remaining[(turn_owner + off) & 3] > 0), None)
            if discarding_seat is None:
                raise RuntimeError("discard flag has no public discarder")
            seat = discarding_seat
        else:
            discarding_seat = None
            seat = turn_owner
        env.action_mask(mask_buf)
        decision_mask = mask_buf.copy()
        if p2p is not None:
            decision_mask &= ~p2p
        if not legal_actions(decision_mask):
            raise RuntimeError(f"empty mask decisions={decisions} seat={seat}")
        action = int(seat_policies[seat].act(env, decision_mask.copy()))
        if seat == subject_seat:
            record_decision(prof, env, seat, action)
            # coherence tracking
            if action == _END:
                subj_turn += 1
            elif action == _ACCEPT or (_CONFIRM <= action < _CONFIRM + 4):
                acquire_turns.append(subj_turn)
            elif (_SETTLE <= action < _SETTLE + 54) or (_CITY <= action < _CITY + 54) \
                    or (_ROAD <= action < _ROAD + 72) or action == _BUYDEV:
                build_turns.append(subj_turn)
                prof.coh_builds_total += 1
        _, done = env.step(action)
        decisions += 1

    winner = -1
    vps = [int(env.player_vp(p)) for p in range(4)]
    for p in range(4):
        if vps[p] >= 10:
            winner = p
            break
    # fold coherence: an acquisition is "covered" if the subject builds within
    # COH_N of its own turns after acquiring (i.e., it spends what it traded for).
    prof.coh_games += 1
    prof.coh_acquire_events += len(acquire_turns)
    bt = sorted(build_turns)
    for at in acquire_turns:
        if any(at <= b <= at + COH_N for b in bt):
            prof.coh_covered_by_build += 1
    prof.coh_end_hand_total += sum(int(env.player_resource(subject_seat, r)) for r in range(5))
    return winner, vps


# ---- subject construction --------------------------------------------------
def make_subject_factory(subject: str, il_zip: str, device: str):
    if subject == "il":
        from sb3_contrib import MaskablePPO
        from models.baseline.evaluate import CheckpointPlayer
        model = MaskablePPO.load(il_zip, device=device)

        def factory(seed: int) -> Player:
            # deterministic=False: matches the eval harness (argmax trips the
            # documented TRADE_OPEN/CANCEL stall).
            return CheckpointPlayer(model, seed=seed, deterministic=False)

        return factory
    return lambda seed: build_agent(subject, seed)


def rotate(values, shift):
    n = len(values)
    shift %= n
    return list(values[-shift:] + values[:-shift]) if shift else list(values)


def run_profile(subject, opponents, games, master_seed, il_zip, device,
                suppress_p2p, max_steps=150_000):
    prof = Profile(subject=subject)
    subj_factory = make_subject_factory(subject, il_zip, device)
    env = fastcatan.Env()
    blocks_per_opp = max(1, games // 4)

    for opp in opponents:
        opp_factory = lambda seed: build_agent(opp, seed)
        for block in range(blocks_per_opp):
            board_seed = derive_board_seed(master_seed, block)
            base = [subj_factory, opp_factory, opp_factory, opp_factory]
            base_labels = [subject, opp, opp, opp]
            for rot in range(4):
                facs = rotate(base, rot)
                labels = rotate(base_labels, rot)
                subject_seat = labels.index(subject) if subject in labels else rot
                # subject is unique among the four labels only if name differs;
                # use rot directly (subject placed at seat rot by rotate()).
                subject_seat = rot
                policies = [
                    facs[s](derive_policy_seed(master_seed, block, rot, s))
                    for s in range(4)
                ]
                for s, pol in enumerate(policies):
                    b = getattr(pol, "bind_seat", None)
                    if b is not None:
                        b(s)
                    stm = getattr(pol, "set_trading_mode", None)
                    if stm is not None:
                        stm(not suppress_p2p)
                env.reset(board_seed)
                winner, vps = profiled_play_one(
                    env, policies, subject_seat, prof, suppress_p2p, max_steps)
                prof.games += 1
                prof.seat_games[subject_seat] += 1
                prof.per_opp[opp]["g"] += 1
                if winner < 0:
                    prof.no_winner += 1
                    continue
                prof.decided += 1
                if winner == subject_seat:
                    prof.wins += 1
                    prof.seat_wins[subject_seat] += 1
                    prof.per_opp[opp]["w"] += 1
                    prof.win_final_vp[vps[subject_seat]] += 1
                else:
                    prof.loss_vp_gap[vps[winner] - vps[subject_seat]] += 1
                    prof.loss_subject_vp[vps[subject_seat]] += 1
                    prof.loser_to_winner_opp[opp] += 1
    return prof


# ---- reporting -------------------------------------------------------------
def summarize(prof: Profile) -> dict:
    def dist(counter):
        tot = sum(counter.values()) or 1
        return {k: round(v / tot, 4) for k, v in counter.most_common()}

    lo, hi = wilson_ci(prof.wins, prof.decided) if prof.decided else (0.0, 0.0)
    out = {
        "subject": prof.subject,
        "games": prof.games,
        "decided": prof.decided,
        "no_winner": prof.no_winner,
        "win_rate": round(prof.wins / prof.decided, 4) if prof.decided else None,
        "win_rate_ci": [round(lo, 4), round(hi, 4)],
        "seat_win_rate": [
            round(prof.seat_wins[s] / prof.seat_games[s], 4) if prof.seat_games[s] else None
            for s in range(4)
        ],
        "subject_decisions": prof.subject_decisions,
        "coarse_by_phase": {ph: dist(c) for ph, c in prof.coarse_by_phase.items()},
        "fine_by_phase": {ph: dist(c) for ph, c in prof.fine_by_phase.items()},
        "trades": {
            "opens_per_game": round(prof.opens / prof.games, 3) if prof.games else 0,
            "confirms_per_game": round(prof.confirms / prof.games, 3) if prof.games else 0,
            "cancels_per_game": round(prof.cancels / prof.games, 3) if prof.games else 0,
            "confirm_rate": round(prof.confirms / prof.opens, 3) if prof.opens else None,
            "bank_trades_per_game": round(prof.bank_trades / prof.games, 3) if prof.games else 0,
            "give_units_per_open": round(sum(prof.give_hist.values()) / prof.opens, 3) if prof.opens else None,
            "want_units_per_open": round(sum(prof.want_hist.values()) / prof.opens, 3) if prof.opens else None,
            "give_resource_share": dist(prof.give_hist),
            "want_resource_share": dist(prof.want_hist),
            "confirmed_mean_give": round(prof.confirm_give_units / prof.confirms, 3) if prof.confirms else None,
            "confirmed_mean_want": round(prof.confirm_want_units / prof.confirms, 3) if prof.confirms else None,
            "confirmed_ev_uniform_mean": round(prof.confirm_ev_uniform / prof.confirms, 3) if prof.confirms else None,
        },
        "robber": {
            "steals": prof.steals,
            "steal_on_leader_rate": round(prof.steal_on_leader / prof.steals, 3) if prof.steals else None,
            "victim_vp_rank_share": dist(prof.steal_victim_vp_rank),
        },
        "coherence": {
            "trade_to_build_rate": round(prof.coh_covered_by_build / prof.coh_acquire_events, 3)
            if prof.coh_acquire_events else None,
            "acquire_events_per_game": round(prof.coh_acquire_events / prof.coh_games, 3)
            if prof.coh_games else None,
            "builds_per_game": round(prof.coh_builds_total / prof.coh_games, 3)
            if prof.coh_games else None,
            "end_hand_per_game": round(prof.coh_end_hand_total / prof.coh_games, 3)
            if prof.coh_games else None,
        },
        "outcomes": {
            "per_opponent": {k: {**v, "win_rate": round(v["w"] / v["g"], 4) if v["g"] else None}
                             for k, v in prof.per_opp.items()},
            "win_final_vp": dict(sorted(prof.win_final_vp.items())),
            "loss_vp_gap": dict(sorted(prof.loss_vp_gap.items())),
            "loss_subject_vp": dict(sorted(prof.loss_subject_vp.items())),
        },
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True,
                    help="'il' for the incumbent net, or a ladder agent name")
    ap.add_argument("--opponents",
                    default="balanced-strong,builder-strong,catanatron-value,oracle-ab-d2")
    ap.add_argument("--games", type=int, default=120,
                    help="games per opponent (rounded down to blocks of 4)")
    ap.add_argument("--seed", type=lambda v: int(v, 0), default=MASTER_SEED)
    ap.add_argument("--il-zip",
                    default="models/checkpoints/phase2_il_100k/il_policy.zip")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--suppress-p2p", action="store_true",
                    help="trades-off pass (default trades ON to profile trading)")
    ap.add_argument("--out-dir", default="models/analysis/out")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-group", default=None)
    args = ap.parse_args()

    opponents = [o for o in args.opponents.split(",") if o]
    t0 = time.time()
    prof = run_profile(
        args.subject, opponents, args.games, args.seed,
        args.il_zip, args.device, args.suppress_p2p,
    )
    summary = summarize(prof)
    summary["wall_seconds"] = round(time.time() - t0, 1)
    summary["opponents"] = opponents
    summary["mode"] = "trades_off" if args.suppress_p2p else "trades_on"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.subject}_{summary['mode']}"
    out_path = out_dir / f"profile_{tag}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n[written] {out_path}", flush=True)

    if not args.no_wandb and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            run = wandb.init(
                project="goodsettler-eval", entity="good-start-labs",
                group=args.wandb_group or "action-profile",
                name=f"profile-{tag}", config={"opponents": opponents, "games": args.games},
                reinit=True,
            )
            flat = {"win_rate": summary["win_rate"],
                    "opens_per_game": summary["trades"]["opens_per_game"],
                    "confirm_rate": summary["trades"]["confirm_rate"],
                    "steal_on_leader_rate": summary["robber"]["steal_on_leader_rate"]}
            wandb.log(flat)
            wandb.summary.update({"profile": summary})
            # coarse-by-phase table
            rows = []
            for ph, d in summary["coarse_by_phase"].items():
                for k, v in d.items():
                    rows.append([ph, k, v])
            tbl = wandb.Table(columns=["phase", "coarse_type", "share"], data=rows)
            wandb.log({"coarse_by_phase": tbl})
            run.finish()
            print("[wandb] logged", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[wandb] skipped: {ex!r}", flush=True)


if __name__ == "__main__":
    main()
