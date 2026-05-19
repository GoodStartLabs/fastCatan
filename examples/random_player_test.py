import argparse
import random
import time
import numpy as np

import fastcatan


def legal_actions(mask: np.ndarray) -> list[int]:
    """Return list of action IDs whose bit is set in the uint64 bitmask buffer."""
    out: list[int] = []
    for word_idx, word in enumerate(mask):
        w = int(word)
        base = word_idx * 64
        while w:
            bit = (w & -w).bit_length() - 1
            out.append(base + bit)
            w &= w - 1
    return out


def build_p2p_trade_filter() -> np.ndarray:
    """Mask with bits SET for every player-to-player trade action ID.

    Used as an AND-NOT filter to suppress p2p trading. Bank/port trades
    (TRADE_BASE..TRADE_BASE+25) stay enabled.
    """
    a = fastcatan.action
    ids = (
        list(range(a.TRADE_ADD_GIVE_BASE, a.TRADE_ADD_GIVE_BASE + 5))
        + list(range(a.TRADE_ADD_WANT_BASE, a.TRADE_ADD_WANT_BASE + 5))
        + [a.TRADE_OPEN, a.TRADE_ACCEPT, a.TRADE_DECLINE]
        + list(range(a.TRADE_CONFIRM_BASE, a.TRADE_CONFIRM_BASE + 4))
        + [a.TRADE_CANCEL]
    )
    m = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
    for aid in ids:
        m[aid // 64] |= np.uint64(1) << np.uint64(aid % 64)
    return m


def pick_random(rng: random.Random, mask: np.ndarray, forbid: np.ndarray | None) -> int:
    if forbid is not None:
        mask = mask & ~forbid
    legals = legal_actions(mask)
    if not legals:
        raise RuntimeError("no legal actions in mask")
    return rng.choice(legals)


def play_one(seed: int, max_steps: int, verbose: bool,
             forbid: np.ndarray | None) -> tuple[int, list[int], int, int]:
    env = fastcatan.Env()
    env.reset(seed)

    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
    rng = random.Random(seed)

    for step_idx in range(max_steps):
        env.action_mask(mask)
        action = pick_random(rng, mask, forbid)
        reward, done = env.step(action)

        if verbose and step_idx % 200 == 0:
            print(
                f"step={step_idx:5d} phase={env.phase} "
                f"cur={env.current_player} action={action} "
                f"vp={[env.player_vp(p) for p in range(4)]}"
            )

        if done:
            vps = [env.player_vp(p) for p in range(4)]
            winner = next((p for p, v in enumerate(vps) if v >= 10), -1)
            return winner, vps, step_idx + 1, env.turn_count

    vps = [env.player_vp(p) for p in range(4)]
    return -1, vps, max_steps, env.turn_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-player-trading", action="store_true",
                        help="disable player-to-player trades (bank/port trades still allowed)")
    args = parser.parse_args()

    forbid = build_p2p_trade_filter() if args.no_player_trading else None

    win_counts = [0, 0, 0, 0]
    no_winner = 0
    total_steps = 0
    total_turns = 0
    winner_vp_sum = 0
    loser_vp_sum = 0
    loser_n = 0

    t0 = time.perf_counter()
    for g in range(args.games):
        winner, vps, steps, turns = play_one(args.seed + g, args.max_steps, args.verbose, forbid)
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
