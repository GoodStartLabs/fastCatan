"""Record a fastcatan game into the log format.

Drop-in replacement for the step loop in `examples/random_player_test.py`.

Usage:
    from ui.recorder import record_game
    from examples.random_player import RandomPlayer

    players = [RandomPlayer(seed=s) for s in (1, 2, 3, 4)]
    record_game(
        seed=42,
        players=players,
        out_path="DEBUG/logs/game_0000.jsonl.gz",
        snap_every=1,            # snapshot every step (default)
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

import fastcatan

# Local imports — avoid hard dependency on examples/ at import time.
from ui.log_format import (
    LOG_VERSION,
    FinalRecord,
    LogHeader,
    LogWriter,
    StepRecord,
    encode_snap,
)


def _player_names(players: Sequence) -> list[str]:
    return [getattr(p, "name", type(p).__name__) for p in players]


def record_game(
    *,
    seed: int,
    players: Sequence,
    out_path: str | Path,
    max_steps: int = 100_000,
    snap_every: int = 1,
    engine_tag: str | None = None,
) -> FinalRecord:
    """Play one game, write the log, return the final summary.

    `players`: sequence of 4 objects exposing `act(env, mask) -> int`.
    `snap_every`: write `env.snapshot()` on every Nth step (and the final).
                  1 = always; higher = smaller logs but slower seek.
    """
    if len(players) != 4:
        raise ValueError(f"need 4 players, got {len(players)}")
    if snap_every < 1:
        raise ValueError("snap_every must be >= 1")

    env = fastcatan.Env()
    env.reset(seed)
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)

    engine = engine_tag or f"fastcatan-{getattr(fastcatan, '__version__', '?')}"
    header = LogHeader(
        v=LOG_VERSION,
        engine=engine,
        obs=int(fastcatan.OBS_SIZE),
        mask=int(fastcatan.MASK_WORDS),
        nactions=int(fastcatan.NUM_ACTIONS),
        seed=seed,
        players=_player_names(players),
        snap_every=snap_every,
    )

    with LogWriter(out_path) as w:
        w.write_header(header)

        step_idx = 0
        done = False
        last_reward = 0.0
        for step_idx in range(max_steps):
            env.action_mask(mask)
            cp = int(env.current_player)
            ph = int(env.phase)
            fl = int(env.flag)
            t = int(env.turn_count)
            dr = int(env.dice_roll)

            action = int(players[cp].act(env, mask))
            reward, done = env.step(action)
            last_reward = float(reward)

            snap_b64: str | None = None
            if (step_idx % snap_every == 0) or done:
                snap_b64 = encode_snap(env.snapshot())

            w.write_step(StepRecord(
                i=step_idx, t=t, ph=ph, fl=fl, cp=cp, dr=dr,
                a=action, r=last_reward, d=int(done),
                snap=snap_b64,
            ))

            if done:
                break

        vps = [int(env.player_vp(p)) for p in range(4)]
        winner = next((p for p, v in enumerate(vps) if v >= 10), -1)
        final = FinalRecord(
            winner=winner,
            vps=vps,
            steps=step_idx + 1,
            turns=int(env.turn_count),
        )
        w.write_final(final)
        return final


# ---------------------------------------------------------------------------
# CLI: `PYTHONPATH=DEBUG python -m ui.recorder --seed 42 --out DEBUG/logs/game.jsonl.gz`
# ---------------------------------------------------------------------------

def _main(argv: list[str]) -> int:
    import argparse

    # examples/ is not a package — add to sys.path for the CLI use case.
    # recorder.py is DEBUG/ui/recorder.py, so repo root is parents[2].
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "examples"))

    parser = argparse.ArgumentParser(description="Record one fastcatan game.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True,
                        help="output path (.jsonl.gz)")
    parser.add_argument("--players", type=str, default="random",
                        help="1 name (all seats) or 4 comma-sep names")
    parser.add_argument("--snap-every", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--no-player-trading", action="store_true")
    args = parser.parse_args(argv)

    from player_base import build_p2p_trade_filter  # type: ignore
    from random_player import RandomPlayer  # type: ignore
    from alphabeta_player import AlphaBetaPlayer, NativeAlphaBetaPlayer  # type: ignore

    registry = {"random": RandomPlayer, "alphabeta": AlphaBetaPlayer,
                "alphabeta_native": NativeAlphaBetaPlayer}
    names = [n.strip() for n in args.players.split(",")]
    if len(names) == 1:
        names = names * 4
    if len(names) != 4:
        parser.error(f"--players must be 1 or 4 names, got {len(names)}")

    forbid = build_p2p_trade_filter() if args.no_player_trading else None
    players = [registry[n](seed=args.seed + seat * 1000, forbid=forbid)
               for seat, n in enumerate(names)]

    final = record_game(
        seed=args.seed,
        players=players,
        out_path=args.out,
        max_steps=args.max_steps,
        snap_every=args.snap_every,
    )
    print(f"wrote {args.out}")
    print(f"  steps={final.steps}  turns={final.turns}")
    print(f"  winner={final.winner}  vps={final.vps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
