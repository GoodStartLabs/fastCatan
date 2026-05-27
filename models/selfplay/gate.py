"""M3 gate: does the latest model beat the N-rounds-ago snapshot?

**Balanced 2-vs-2** (recalibrated for 4 players). Two seats play `latest`, two
play `nago`; the assignment is rotated every game so each model occupies every
seat equally (cancels seat bias). The metric is the *latest team's* share of
decided games. Because each team holds 2 of 4 seats, equal policies → **0.50**,
so **>0.55 means meaningfully better** — the thesis's intended semantics.

(The earlier 1-vs-3 form had neutral 0.25, which made the >0.55 bar a "win >2×
fair share" dominance test rather than "better than". Per-seat diagnostics with
a single seat vs three opponents live in `eval_seats.py`.)

Sampling policy — argmax trips the trade-loop stall (root PLAN.md); the gate
samples. Use the SAME `--no-p2p-trade` setting that training used.

CLI:
    python -m models.selfplay.gate \
        --latest models/checkpoints/sp/snap_5000000.zip \
        --nago   models/checkpoints/sp/snap_3000000.zip \
        --games 1000 --no-p2p-trade
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

import fastcatan

from models.eval import wilson_ci
from models.selfplay.eval_seats import play_one
from models.selfplay.opponents import Opponent, PolicyOpponent
from models.selfplay.selfplay_env import _p2p_trade_mask_bool


def play_2v2(
    latest: Opponent,
    nago: Opponent,
    n_games: int,
    seed: int = 0,
    max_steps: int = 4000,
    suppress_p2p: bool = False,
) -> tuple[int, int, int]:
    """Balanced 2-vs-2, seat assignment rotated per game. Neutral (latest==nago)
    = 0.50. Returns (latest_wins, decided, no_winner); decided excludes stalls."""
    env = fastcatan.Env()
    seed_seq = random.Random(seed)
    p2p = _p2p_trade_mask_bool() if suppress_p2p else None
    obs_buf = np.zeros(fastcatan.OBS_SIZE, dtype=np.float32)
    mask_buf = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)

    latest_wins = nago_wins = no_winner = 0
    for g in range(n_games):
        # Interleaved complementary assignment, alternating per game, so `latest`
        # holds {0,2} half the games and {1,3} the other half (each seat equally).
        if g % 2 == 0:
            seat_policies = [latest, nago, latest, nago]
            latest_seats = (0, 2)
        else:
            seat_policies = [nago, latest, nago, latest]
            latest_seats = (1, 3)
        env.reset(seed_seq.getrandbits(64))
        winner = play_one(env, seat_policies, obs_buf, mask_buf, p2p, max_steps)
        if winner < 0:
            no_winner += 1
        elif winner in latest_seats:
            latest_wins += 1
        else:
            nago_wins += 1
    return latest_wins, latest_wins + nago_wins, no_winner


def gate_result(
    wins: int,
    decided: int,
    no_winner: int = 0,
    threshold: float = 0.55,
    min_decided_frac: float = 0.5,
) -> dict:
    """Latest-team win share over *decided* games + a `conclusive` guard.

    Neutral (equal policies) = 0.50 in the balanced 2-vs-2 gate, so `threshold`
    0.55 = meaningfully better. A gate where most games stall to no-winner is not
    evidence either way, so `pass` requires `conclusive` (>= `min_decided_frac`
    of games reached a winner) AND win share > threshold. `no_winner_rate` keeps
    the stall visible instead of hiding behind a 0/0 -> 0.0 'FAIL'.
    """
    games = decided + no_winner
    rate = wins / decided if decided else 0.0
    lo, hi = wilson_ci(wins, decided)
    no_winner_rate = no_winner / games if games else 0.0
    conclusive = games > 0 and decided >= min_decided_frac * games
    return {
        "win_rate": rate, "ci_low": lo, "ci_high": hi,
        "decided": decided, "no_winner": no_winner,
        "no_winner_rate": no_winner_rate,
        "conclusive": conclusive, "pass": bool(conclusive and rate > threshold),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--latest", required=True, help="latest snapshot")
    p.add_argument("--nago", required=True, help="N-ago snapshot")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--no-p2p-trade", action="store_true",
                   help="Forbid p2p trades (kills the trade-loop stall so games "
                        "terminate). Use the SAME setting training used.")
    args = p.parse_args()

    a = PolicyOpponent.load(Path(args.latest), name="latest")
    b = PolicyOpponent.load(Path(args.nago), name="nago")

    latest_wins, decided, no_winner = play_2v2(
        a, b, args.games, seed=args.seed, max_steps=args.max_steps,
        suppress_p2p=args.no_p2p_trade,
    )
    r = gate_result(latest_wins, decided, no_winner, args.threshold)

    print("\n=== M3 gate (balanced 2-vs-2, latest vs N-ago) ===")
    print(f"latest: {args.latest}")
    print(f"nago:   {args.nago}")
    print(f"games (decided): {decided}/{args.games} "
          f"(no-winner: {no_winner}, {r['no_winner_rate']:.1%})")
    print(f"latest team win share: {r['win_rate']:.4f}  "
          f"95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}]  (neutral 0.50)")
    if not r["conclusive"]:
        print("WARNING: too many no-winner games -> INCONCLUSIVE "
              "(consider --no-p2p-trade or the C++ TRADE_OPEN mask cap).")
    print(f"M3 gate (>{args.threshold:.2f}): {'PASS' if r['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
