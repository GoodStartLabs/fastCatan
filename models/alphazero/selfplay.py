"""AlphaZero self-play + training for fastCatan.

All four seats are driven by the SAME current net via MCTS (self-play). Each move
records (obs from the to-move seat's POV, MCTS visit policy, legal mask, to-move seat);
at game end every record gets a value target z = +1 if its seat won else -1. Forced
single-legal-action states (roll dice, forced discards, etc.) are stepped directly and
NOT recorded — no decision, no learning signal, and it saves the bulk of MCTS calls.

The game env advances with its own RNG (genuine chance). Only the MCTS scratch env is
reseeded, internally, to sample chance during search.

Run (smoke):
    python -m models.alphazero.selfplay --total-games 20 --sims 24 --device cpu
"""
from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import fastcatan

from models.alphazero.net import PolicyValueNet, masked_log_softmax
from models.ckpt import write_stamp
from models.alphazero.mcts import (
    MCTS, _unpack, filter_p2p, p2p_trade_mask, MASK_WORDS, NUM_ACTIONS, OBS_SIZE,
)

CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
WIN_VP = 10


class Sample:
    __slots__ = ("obs", "pi", "mask", "to_move", "z")

    def __init__(self, obs, pi, mask, to_move):
        self.obs = obs
        self.pi = pi
        self.mask = mask
        self.to_move = to_move
        self.z = 0.0


def _legal(env, buf, p2p_bool) -> list[int]:
    env.action_mask(buf)
    mask, legal = _unpack(buf)
    if p2p_bool is not None:
        _, legal = filter_p2p(mask, p2p_bool)
    return legal


def play_one_game(game_env, mcts: MCTS, seed: int, temp_moves: int,
                  p2p_bool, opp_pick=None, value_mode: str = "sparse",
                  max_moves: int = 4000) -> tuple[list[Sample], int, int]:
    """Generate one training game.

    opp_pick=None  -> self-play: every seat plays via MCTS, all decisions recorded.
    opp_pick set   -> learner is seat 0 (MCTS, recorded); seats 1-3 use opp_pick
                      (e.g. native AlphaBeta) and are NOT recorded.

    value_mode 'sparse'    -> z = +1 if the record's seat won else -1.
               'vp_margin' -> z = clip((own_vp - best_other_vp)/10, -1, 1); a dense
                      target so losses against a dominant opponent still carry
                      gradient (sparse ±1 saturates at -1 and stops teaching).
    """
    game_env.reset(seed)
    mask_buf = np.zeros(MASK_WORDS, dtype=np.uint64)
    obs_buf = np.zeros(OBS_SIZE, dtype=np.float32)
    records: list[Sample] = []
    decision_moves = 0

    for _ in range(max_moves):
        legal = _legal(game_env, mask_buf, p2p_bool)
        if not legal:
            break
        cp = game_env.current_player

        if opp_pick is not None and cp != 0:       # opponent seat: act, don't record
            action = opp_pick(game_env, cp, legal) if len(legal) > 1 else legal[0]
            _r, done = game_env.step(action)
            if done:
                break
            continue

        if len(legal) == 1:                        # forced: step, don't record
            _r, done = game_env.step(legal[0])
            if done:
                break
            continue

        snap = game_env.snapshot()
        temp = 1.0 if decision_moves < temp_moves else 0.0
        action, pi, mask = mcts.choose(snap, temperature=temp, add_root_noise=True)

        game_env.write_obs(cp, obs_buf)
        records.append(Sample(obs_buf.copy(), pi, mask, cp))
        decision_moves += 1

        _r, done = game_env.step(action)
        if done:
            break

    vps = [game_env.player_vp(p) for p in range(fastcatan.NUM_PLAYERS)]
    winner = next((p for p in range(fastcatan.NUM_PLAYERS) if vps[p] >= WIN_VP), -1)
    for s in records:
        if value_mode == "vp_margin":
            best_other = max(vps[q] for q in range(fastcatan.NUM_PLAYERS)
                             if q != s.to_move)
            s.z = float(np.clip((vps[s.to_move] - best_other) / 10.0, -1.0, 1.0))
        else:
            s.z = 1.0 if s.to_move == winner else -1.0
    return records, winner, decision_moves


def train_steps(net, opt, buffer, n_steps, batch_size, value_coef, device) -> dict:
    net.train()
    last = {}
    for _ in range(n_steps):
        batch = random.sample(buffer, min(batch_size, len(buffer)))
        obs = torch.from_numpy(np.stack([s.obs for s in batch])).to(device)
        pi = torch.from_numpy(np.stack([s.pi for s in batch])).to(device)
        mask = torch.from_numpy(np.stack([s.mask for s in batch])).to(device)
        z = torch.tensor([s.z for s in batch], dtype=torch.float32, device=device)

        logits, value = net(obs)
        logp = masked_log_softmax(logits, mask)
        policy_loss = -(pi * logp).sum(dim=1).mean()
        value_loss = F.mse_loss(value, z)
        loss = policy_loss + value_coef * value_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        last = {"loss": float(loss), "policy": float(policy_loss),
                "value": float(value_loss)}
    return last


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-games", type=int, default=200)
    p.add_argument("--sims", type=int, default=50)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--temp-moves", type=int, default=20,
                   help="First N decision moves sample ~ visits (temp=1); then greedy.")
    p.add_argument("--buffer-size", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-steps-per-game", type=int, default=8)
    p.add_argument("--min-buffer", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--value-coef", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default=str(CKPT_DIR / "alphazero"))
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--allow-trades", action="store_true",
                   help="Allow p2p trades (default: suppressed, matching the "
                        "--no-trades M4 eval vs AlphaBeta and avoiding the "
                        "trade-compose stall).")
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    suppress = not args.allow_trades
    p2p_bool = p2p_trade_mask() if suppress else None
    net = PolicyValueNet().to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    mcts = MCTS(net, device=args.device, sims=args.sims, c_puct=args.c_puct,
                seed=args.seed, suppress_p2p=suppress)
    game_env = fastcatan.Env()
    buffer: deque = deque(maxlen=args.buffer_size)
    seed_seq = random.Random(args.seed ^ 0x5EED)

    for g in range(1, args.total_games + 1):
        recs, winner, dmoves = play_one_game(
            game_env, mcts, seed_seq.getrandbits(64), args.temp_moves, p2p_bool)
        buffer.extend(recs)
        msg = (f"[game {g:>4d}] winner={winner:+d} decisions={dmoves} "
               f"samples={len(recs)} buf={len(buffer)}")
        if len(buffer) >= args.min_buffer:
            stats = train_steps(net, opt, buffer, args.train_steps_per_game,
                                args.batch_size, args.value_coef, args.device)
            msg += (f"  loss={stats['loss']:.3f} p={stats['policy']:.3f} "
                    f"v={stats['value']:.3f}")
        print(msg, flush=True)

        if g % args.checkpoint_every == 0:
            ck = save_dir / f"az_{g}.pt"
            torch.save({"net_state": net.state_dict(), "args": vars(args)}, str(ck))
            write_stamp(ck)
            print(f"[ckpt] {ck}", flush=True)

    final = save_dir / "az_final.pt"
    torch.save({"net_state": net.state_dict(), "args": vars(args)}, str(final))
    write_stamp(final)
    print(f"[done] saved -> {final}", flush=True)


if __name__ == "__main__":
    main()
