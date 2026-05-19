"""4-player Catan test orchestrator. Players pluggable via --players flag."""

import argparse
import contextlib
import time
import numpy as np

import fastcatan
from player_base import build_p2p_trade_filter
from random_player import RandomPlayer
from alphabeta_player import AlphaBetaPlayer


PLAYER_REGISTRY = {
    "random": RandomPlayer,
    "alphabeta": AlphaBetaPlayer,
}

def make_players(spec: str, seed: int, forbid):
    names = [n.strip() for n in spec.split(",")]
    if len(names) == 1:
        names = names * 4
    if len(names) != 4:
        raise ValueError(f"--players must be 1 or 4 names, got {len(names)}")
    return [PLAYER_REGISTRY[n](seed=seed + seat * 1000, forbid=forbid)
            for seat, n in enumerate(names)]

def play_one(game_id, seed, max_steps, players, log_f):
    env = fastcatan.Env()
    env.reset(seed)
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)

    for step_idx in range(max_steps):
        env.action_mask(mask)
        player_idx = env.current_player
        phase = env.phase
        turn = env.turn_count
        action = players[player_idx].act(env, mask)
        reward, done = env.step(action)

        if log_f:
            log_f.write(
                f"{game_id},{seed},{step_idx},{turn},{phase},{player_idx},"
                f"{action},{reward:.0f},{int(done)}\n"
            )

        if done:
            vps = [env.player_vp(p) for p in range(4)]
            winner = next((p for p, v in enumerate(vps) if v >= 10), -1)
            return winner, vps, step_idx + 1, env.turn_count

    vps = [env.player_vp(p) for p in range(4)]
    return -1, vps, max_steps, env.turn_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--players", type=str, default="random",help="1 name (all seats) or 4 comma-sep names. " f"Choices: {list(PLAYER_REGISTRY)}")
    parser.add_argument("--no-player-trading", action="store_true", help="disable player-to-player trades (bank/port trades stay allowed)")
    parser.add_argument("--log", type=str, default="random_test.csv", help="CSV log path (pass empty string to disable)")
    args = parser.parse_args()

    forbid = build_p2p_trade_filter() if args.no_player_trading else None

    win_counts = [0, 0, 0, 0]
    no_winner = 0
    total_steps = 0
    total_turns = 0
    winner_vp_sum = 0
    loser_vp_sum = 0
    loser_n = 0
    max_steps = 100_000  # safety cap to prevent infinite games from hanging the test

    log_cm = open(args.log, "w") if args.log else contextlib.nullcontext()
    with log_cm as log_f:
        if log_f:
            log_f.write("game,seed,step,turn,phase,player,action,reward,done\n")

        t0 = time.perf_counter()
        for g in range(args.games):
            seed = args.seed + g
            players = make_players(args.players, seed, forbid)
            winner, vps, steps, turns = play_one(g, seed, max_steps, players, log_f)
            total_steps += steps
            total_turns += turns
            if winner < 0:
                no_winner += 1
            else:
                win_counts[winner] += 1
                winner_vp_sum += vps[winner]
                for p, v in enumerate(vps):
                    if p != winner:
                        loser_vp_sum += v
                        loser_n += 1
        elapsed = time.perf_counter() - t0

    n = args.games
    winners_n = n - no_winner
    print(f"players:         {args.players}")
    print(f"games:           {n}  (no-winner: {no_winner})")
    print(f"wins per seat:   {win_counts}")
    print(f"avg steps/game:  {total_steps / n:.1f}")
    print(f"avg turns/game:  {total_turns / n:.1f}")
    if winners_n:
        print(f"avg winner VP:   {winner_vp_sum / winners_n:.2f}")
    if loser_n:
        print(f"avg loser VP:    {loser_vp_sum / loser_n:.2f}")
    print(f"total time:      {elapsed:.3f}s")
    print(f"time/game:       {elapsed / n * 1000:.3f} ms")
    print(f"time/step:       {elapsed / total_steps * 1e6:.2f} us")
    print(f"throughput:      {n / elapsed:.1f} games/s  ({total_steps / elapsed:.0f} steps/s)")


if __name__ == "__main__":
    main()
