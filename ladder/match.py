"""Seat-balanced, paired-board fastCatan match primitives.

The loop follows ``models.selfplay.eval_seats.play_one``: all four seats have an
explicit policy and every decision is made from the acting seat.  Ladder
personas use the richer ``examples.player_base.Player.act(env, mask)`` seam so
their legal own/private and public accessors are available directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Protocol, Sequence

import numpy as np

import fastcatan
from examples.player_base import Player, build_p2p_trade_filter, legal_actions

from .config import NO_WINNER_POLICY, ROTATIONS_PER_BLOCK
from .seeds import derive_board_seed, derive_policy_seed


class PlayerFactory(Protocol):
    def __call__(self, seed: int) -> Player: ...


@dataclass(frozen=True)
class GameResult:
    winner_seat: int
    decisions: int
    wall_seconds: float


@dataclass(frozen=True)
class RotationBlockResult:
    block_index: int
    board_seed: int
    games: int
    candidate_wins: int
    opponent_wins: int
    no_winner: int
    candidate_seat_wins: tuple[int, int, int, int]
    candidate_seat_games: tuple[int, int, int, int]
    winner_seats: tuple[int, int, int, int]
    decisions: int
    wall_seconds: float
    no_winner_policy: str = NO_WINNER_POLICY

    @property
    def win_share(self) -> float:
        # Truncations remain in the denominator and are therefore losses.
        return self.candidate_wins / self.games if self.games else 0.0

    @property
    def decisions_per_second(self) -> float:
        return self.decisions / self.wall_seconds if self.wall_seconds else 0.0


def _action_is_legal(mask: np.ndarray, action: int) -> bool:
    return 0 <= action < fastcatan.NUM_ACTIONS and bool(
        (int(mask[action >> 6]) >> (action & 63)) & 1
    )


def play_one(
    env,
    seat_policies: Sequence[Player],
    *,
    suppress_p2p: bool,
    max_steps: int,
) -> GameResult:
    """Drive one already-reset game and return winner ``-1`` on truncation.

    No-winner games include both the engine's MAX_TURNS backstop and the Python
    defensive ply cap.  Callers score either kind as a loss for every seat.
    """
    if len(seat_policies) != fastcatan.NUM_PLAYERS:
        raise ValueError("exactly four seat policies are required")

    mask_buf = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
    p2p = build_p2p_trade_filter() if suppress_p2p else None
    decisions = 0
    done = False
    started = perf_counter()

    while not done and decisions < max_steps:
        seat = int(env.current_player)
        env.action_mask(mask_buf)
        decision_mask = mask_buf.copy()
        if p2p is not None:
            decision_mask &= ~p2p
        if not legal_actions(decision_mask):
            raise RuntimeError(
                f"empty ladder mask at decision={decisions} seat={seat} "
                f"phase={env.phase} flag={env.flag}"
            )
        action = int(seat_policies[seat].act(env, decision_mask.copy()))
        if not _action_is_legal(decision_mask, action):
            raise ValueError(
                f"{seat_policies[seat].name} returned illegal action {action} "
                f"at seat {seat}"
            )
        _, done = env.step(action)
        decisions += 1

    elapsed = perf_counter() - started
    winner = -1
    for seat in range(fastcatan.NUM_PLAYERS):
        if env.player_vp(seat) >= 10:
            winner = seat
            break
    return GameResult(winner, decisions, elapsed)


def rotate(values: Sequence, shift: int) -> list:
    """Cyclically move base seat 0 to seat ``shift``."""
    n = len(values)
    if n == 0:
        return []
    shift %= n
    return list(values[-shift:] + values[:-shift]) if shift else list(values)


def play_rotation_block(
    *,
    env,
    candidate_name: str,
    opponent_name: str,
    candidate_factory: PlayerFactory,
    opponent_factory: PlayerFactory,
    master_seed: int,
    block_index: int,
    suppress_p2p: bool,
    max_steps: int,
) -> RotationBlockResult:
    """Play one board under all four cyclic candidate seat assignments."""
    board_seed = derive_board_seed(master_seed, block_index)
    base_labels = [candidate_name, opponent_name, opponent_name, opponent_name]
    base_factories: list[PlayerFactory] = [
        candidate_factory,
        opponent_factory,
        opponent_factory,
        opponent_factory,
    ]
    candidate_wins = 0
    opponent_wins = 0
    no_winner = 0
    candidate_seat_wins = [0, 0, 0, 0]
    candidate_seat_games = [0, 0, 0, 0]
    winner_seats: list[int] = []
    decisions = 0
    wall_seconds = 0.0

    for rotation_index in range(ROTATIONS_PER_BLOCK):
        labels = rotate(base_labels, rotation_index)
        factories = rotate(base_factories, rotation_index)
        candidate_seat = labels.index(candidate_name)
        candidate_seat_games[candidate_seat] += 1
        policies = [
            factory(
                derive_policy_seed(
                    master_seed, block_index, rotation_index, seat
                )
            )
            for seat, factory in enumerate(factories)
        ]
        env.reset(board_seed)
        result = play_one(
            env,
            policies,
            suppress_p2p=suppress_p2p,
            max_steps=max_steps,
        )
        winner_seats.append(result.winner_seat)
        decisions += result.decisions
        wall_seconds += result.wall_seconds
        if result.winner_seat < 0:
            no_winner += 1
        elif result.winner_seat == candidate_seat:
            candidate_wins += 1
            candidate_seat_wins[candidate_seat] += 1
        else:
            opponent_wins += 1

    return RotationBlockResult(
        block_index=block_index,
        board_seed=board_seed,
        games=ROTATIONS_PER_BLOCK,
        candidate_wins=candidate_wins,
        opponent_wins=opponent_wins,
        no_winner=no_winner,
        candidate_seat_wins=tuple(candidate_seat_wins),
        candidate_seat_games=tuple(candidate_seat_games),
        winner_seats=tuple(winner_seats),
        decisions=decisions,
        wall_seconds=wall_seconds,
    )


def run_matchup(
    *,
    candidate_name: str,
    opponent_name: str,
    candidate_factory: PlayerFactory,
    opponent_factory: PlayerFactory,
    games: int,
    master_seed: int,
    suppress_p2p: bool,
    max_steps: int = 150_000,
) -> list[RotationBlockResult]:
    """Run complete four-game rotation blocks for a pairwise matchup."""
    if games <= 0 or games % ROTATIONS_PER_BLOCK:
        raise ValueError("games must be a positive multiple of four")
    env = fastcatan.Env()
    return [
        play_rotation_block(
            env=env,
            candidate_name=candidate_name,
            opponent_name=opponent_name,
            candidate_factory=candidate_factory,
            opponent_factory=opponent_factory,
            master_seed=master_seed,
            block_index=block,
            suppress_p2p=suppress_p2p,
            max_steps=max_steps,
        )
        for block in range(games // ROTATIONS_PER_BLOCK)
    ]
