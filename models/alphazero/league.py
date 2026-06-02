"""Mixed opponent league for AlphaZero (the fix for the no-win collapse).

Training only against AlphaBeta flat-lines: a scratch/0.6 net never wins, value targets
have no variance, gradient dies (see [[alphazero-v1-built]]). The league keeps BEATABLE
opponents in the pool so some games are wins -> value-target variance -> gradient lives,
while AlphaBeta sits in the pool as the target whose sampling weight auto-ramps as the
learner improves.

Pool members (each: `.name`, `.act(env, seat, legal) -> action`):
  - RandomOpp        floor difficulty; guarantees winnable games at scratch start.
  - ABOpp(depth)     native Catanatron-AB port (ab_decide); d1 + d2 both in the pool.
  - SnapshotOpp      a frozen past-self PolicyValueNet (greedy policy head, no MCTS).

PFSP: weight_m = x_m*(1-x_m) + floor, x_m = learner win-rate vs m. Peaks at x=0.5
(challenging-but-beatable) and the floor keeps every member — including AB the learner
can't yet beat — in the sampling mix. Per game, each opponent seat draws a member
independently by weight.

The same per-seat dispatcher drives BOTH the real game and the in-tree opponent model
(MCTSvsFixed.opp), so the search models each seat's actual opponent.
"""
from __future__ import annotations

import numpy as np
import torch

import fastcatan

from models.alphazero.net import PolicyValueNet
from models.alphazero.mcts import NUM_ACTIONS, OBS_SIZE, NUM_PLAYERS

_NO_ACTION = 0xFFFFFFFF


class RandomOpp:
    name = "random"

    def __init__(self, rng):
        self.rng = rng

    def act(self, env, seat, legal):
        return self.rng.choice(legal)


class ABOpp:
    def __init__(self, depth, prune, rng):
        self.depth = depth
        self.prune = prune
        self.rng = rng
        self.name = f"ab_d{depth}"

    def act(self, env, seat, legal):
        a = env.ab_decide(seat, self.depth, self.prune)
        return a if (a != _NO_ACTION and a in legal) else self.rng.choice(legal)


class SnapshotOpp:
    """Frozen past-self: greedy argmax of the policy head over legal actions."""

    def __init__(self, name, state_dict, rng):
        self.name = name
        self.rng = rng
        self.net = PolicyValueNet()
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self._obs = np.zeros(OBS_SIZE, dtype=np.float32)

    @torch.no_grad()
    def act(self, env, seat, legal):
        env.write_obs(seat, self._obs)
        logits, _ = self.net(torch.from_numpy(self._obs).unsqueeze(0))
        row = logits[0].numpy()
        masked = np.full(NUM_ACTIONS, -np.inf, dtype=np.float32)
        masked[legal] = row[legal]
        a = int(np.argmax(masked))
        return a if a in legal else self.rng.choice(legal)


def pfsp_weights(winrates: dict, floor: float = 0.05) -> dict:
    """weight ∝ x(1-x)+floor — favors ~50% opponents, floor keeps all in the mix."""
    w = {n: x * (1.0 - x) + floor for n, x in winrates.items()}
    s = sum(w.values()) or 1.0
    return {n: v / s for n, v in w.items()}


def build_members(payload: dict, rng):
    """Reconstruct pool members in a worker from the broadcast payload."""
    members = []
    if payload.get("include_random", True):
        members.append(RandomOpp(rng))
    for d in payload.get("ab_depths", []):
        members.append(ABOpp(d, payload.get("ab_prune", False), rng))
    for name, sd in payload.get("snapshots", []):
        members.append(SnapshotOpp(name, sd, rng))
    return members


def sample_seat_assignment(members, weights: dict, rng):
    """Assign each opponent seat (1..NUM_PLAYERS-1) a member, sampled by PFSP weight.

    Returns (dispatch(env, seat, legal) -> action, set(member_names_at_table)).
    """
    names = [m.name for m in members]
    w = np.array([weights.get(n, 0.0) for n in names], dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(names)) / len(names)
    by_name = {m.name: m for m in members}

    seat_member = {}
    for seat in range(1, NUM_PLAYERS):
        pick = names[int(rng_choice(w, rng))]
        seat_member[seat] = by_name[pick]

    table_names = {m.name for m in seat_member.values()}

    def dispatch(env, seat, legal):
        return seat_member[seat].act(env, seat, legal)

    return dispatch, table_names


def rng_choice(weights: np.ndarray, rng) -> int:
    """Weighted index draw using a stdlib random.Random (workers seed these)."""
    r = rng.random() * float(weights.sum())
    acc = 0.0
    for i, wi in enumerate(weights):
        acc += float(wi)
        if r <= acc:
            return i
    return len(weights) - 1


class LeagueStats:
    """Main-process win-rate bookkeeping per member name (EMA-ish via decay)."""

    def __init__(self, decay: float = 0.99):
        self.games: dict = {}
        self.wins: dict = {}
        self.decay = decay

    def update(self, table_names, learner_won: bool):
        for n in table_names:
            self.games[n] = self.games.get(n, 0.0) * self.decay + 1.0
            self.wins[n] = self.wins.get(n, 0.0) * self.decay + (1.0 if learner_won else 0.0)

    def winrates(self, member_names, prior: float = 0.5) -> dict:
        out = {}
        for n in member_names:
            g = self.games.get(n, 0.0)
            out[n] = (self.wins.get(n, 0.0) / g) if g >= 1.0 else prior
        return out
