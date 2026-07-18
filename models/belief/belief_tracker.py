"""Legal per-seat belief tracker (card counting) over the PUBLIC event stream.

Leakage contract (DR-001): this tracker reads ONLY public information — the
board layout, the ordered public action stream, dice results, and each player's
PUBLIC hand SIZE — plus, for the recording seat, its own private hand. It NEVER
reads any opponent's private resource/dev fields nor the write_obs_full hidden
appendix. Steal outcomes are treated as distributions (a stolen card's identity
is never inspected). It is therefore legal actor input by construction; the true
appendix is used ONLY by validate() as an offline accuracy diagnostic.

Board channels are parsed from write_obs (verified offsets):
  hex_resource one-hot  obs[800:914].reshape(19,6).argmax(1)   (5 = desert)
  hex_number/12         obs[914:933]*12
  robber one-hot        obs[987:1006].argmax()
Per-player public hand size = obs[rel*16 + 1] * norm (we read exact via env).

Resource index order (fastcatan): [brick, lumber, wool, grain, ore].
"""
from __future__ import annotations

import numpy as np
import fastcatan as fc
from ladder.topology import HEX_TO_NODES

_NODE_HEXES: list[list[int]] = [[] for _ in range(fc.NUM_NODES)]
for _h, _nodes in enumerate(HEX_TO_NODES):
    for _n in _nodes:
        _NODE_HEXES[_n].append(_h)

A = fc.action
_SETTLE, _CITY, _ROAD = int(A.SETTLE_BASE), int(A.CITY_BASE), int(A.ROAD_BASE)
_DISCARD = int(A.DISCARD_BASE)
_ROLL, _BUYDEV = int(A.ROLL_DICE), int(A.BUY_DEV)
_MOVEROB, _STEAL = int(A.MOVE_ROBBER_BASE), int(A.STEAL_BASE)
_TRADEBANK = int(A.TRADE_BASE)
_YOP, _MONO = int(A.PLAY_YEAR_OF_PLENTY), int(A.PLAY_MONOPOLY)

# costs in [brick,lumber,wool,grain,ore]
_COST_SETTLE = np.array([1, 1, 1, 1, 0], float)
_COST_CITY = np.array([0, 0, 0, 2, 3], float)
_COST_ROAD = np.array([1, 1, 0, 0, 0], float)
_COST_DEV = np.array([0, 0, 1, 1, 1], float)

BELIEF_PER_OPP = 6  # 5 expected resources + 1 uncertainty scalar
N_FEATURES = BELIEF_PER_OPP * 3


class BeliefTracker:
    """One instance per game; call reset(env) then after_step(...) each ply."""

    def __init__(self) -> None:
        self.res = np.zeros((4, 5), float)   # expected composition
        self.unk = np.zeros(4, float)        # cards of uncertain type (steals)
        self.owner: dict[int, int] = {}      # node -> seat
        self.is_city: dict[int, int] = {}    # node -> 0/1
        self.hexnum = np.zeros(19, int)
        self.hexres = np.zeros(19, int)
        self.robber = 0
        self.prev_hs = np.zeros(4, float)    # public hand size after last step

    def reset(self, env) -> None:
        o = np.zeros(fc.OBS_SIZE, np.float32)
        env.write_obs(0, o)
        self.hexnum = np.rint(o[914:933] * 12).astype(int)
        self.hexres = o[800:914].reshape(19, 6).argmax(1)
        self.robber = int(o[987:1006].argmax())
        self.res[:] = 0.0
        self.unk[:] = 0.0
        self.owner.clear()
        self.is_city.clear()
        self.prev_hs = self._handsizes(env)

    def _produce(self, roll: int) -> None:
        if roll == 7:
            return
        for h in range(19):
            if self.hexnum[h] != roll or h == self.robber:
                continue
            r = self.hexres[h]
            if r >= 5:  # desert
                continue
            for n in HEX_TO_NODES[h]:
                s = self.owner.get(n)
                if s is not None:
                    self.res[s, r] += 2.0 if self.is_city.get(n) else 1.0

    def _handsizes(self, env) -> np.ndarray:
        return np.array([int(env.player_handsize(p)) for p in range(4)], float)

    def after_step(self, env, actor: int, action: int, phase_before: int) -> None:
        """Apply the effect of `action` (just stepped by `actor`) to beliefs.

        `phase_before` is env.phase BEFORE the step (0/1 = initial placement,
        where builds are free and the 2nd settlement grants adjacent resources).
        """
        initial = phase_before in (0, 1)
        if action == _ROLL:
            self._produce(int(env.dice_roll))
        elif _SETTLE <= action < _SETTLE + 54:
            n = action - _SETTLE
            self.owner[n] = actor
            self.is_city[n] = 0
            if phase_before == 1:  # 2nd placement: +1 per adjacent land hex
                for h in _NODE_HEXES[n]:
                    r = self.hexres[h]
                    if r < 5:
                        self.res[actor, r] += 1.0
            elif not initial:
                self.res[actor] -= _COST_SETTLE
        elif _CITY <= action < _CITY + 54:
            n = action - _CITY
            self.owner[n] = actor
            self.is_city[n] = 1
            if not initial:
                self.res[actor] -= _COST_CITY
        elif _ROAD <= action < _ROAD + 72:
            if not initial:
                self.res[actor] -= _COST_ROAD
        elif _MOVEROB <= action < _MOVEROB + 19:
            self.robber = action - _MOVEROB  # blocks that hex's production
        elif action == _BUYDEV:
            self.res[actor] -= _COST_DEV
        elif _DISCARD <= action < _DISCARD + 5:
            # discard action id encodes the exact resource discarded (public)
            self.res[actor, action - _DISCARD] -= 1.0
        elif _YOP <= action < _YOP + 25:
            off = action - _YOP
            self.res[actor, off // 5] += 1.0
            self.res[actor, off % 5] += 1.0
        elif _MONO <= action < _MONO + 5:
            r = action - _MONO
            taken = self.res[:, r].sum() - self.res[actor, r]
            self.res[:, r] = 0.0
            self.res[actor, r] += max(taken, 0.0)
        elif _TRADEBANK <= action < _TRADEBANK + 25:
            off = action - _TRADEBANK
            give, get = off // 5, off % 5
            # exact give count from the public hand-size delta (2:1/3:1/4:1)
            cur = int(env.player_handsize(actor))
            give_count = max(1, int(round(self.prev_hs[actor] - cur + 1.0)))
            self.res[actor, get] += 1.0
            self.res[actor, give] = max(self.res[actor, give] - give_count, 0.0)
        elif int(A.TRADE_CONFIRM_BASE) <= action < int(A.TRADE_CONFIRM_BASE) + 4:
            # p2p trade finalized (banned in the AB generator; matters at
            # trades-on deployment): proposer=actor gives `give`, gets `want`.
            partner = action - int(A.TRADE_CONFIRM_BASE)
            give = np.array([env.trade_give(r) for r in range(5)], float)
            want = np.array([env.trade_want(r) for r in range(5)], float)
            self.res[actor] += want - give
            self.res[partner] += give - want
        elif _STEAL <= action < _STEAL + 4:
            victim = action - _STEAL
            self.unk[actor] += 1.0
            tot = self.res[victim].sum() + self.unk[victim]
            if tot > 0:
                if self.unk[victim] > 0:
                    self.unk[victim] -= 1.0
                else:
                    self.res[victim] *= max(0.0, 1.0 - 1.0 / tot)
        # keep non-negative, then reconcile totals to the exact public handsize
        np.clip(self.res, 0.0, None, out=self.res)
        self._reconcile(env)
        self.prev_hs = self._handsizes(env)

    def _reconcile(self, env) -> None:
        """Scale each seat's belief so its total equals the public hand size.

        Public hand size is legal information; this absorbs approximation error
        from bank-trade ratios / steal distribution without peeking at hidden
        composition. Residual mass beyond the known-attributed part is carried
        as `unk` (uniform uncertainty)."""
        hs = self._handsizes(env)
        for s in range(4):
            known = self.res[s].sum()
            target = hs[s]
            if target <= 0:
                self.res[s] = 0.0
                self.unk[s] = 0.0
                continue
            if known > target and known > 0:
                self.res[s] *= target / known
                self.unk[s] = 0.0
            else:
                self.unk[s] = max(target - known, 0.0)

    def features(self, pov: int) -> np.ndarray:
        """POV-relative belief features for the 3 opponents (rel 1,2,3),
        matching the obs opponent ordering. Normalized by a soft hand scale."""
        out = np.zeros(N_FEATURES, np.float32)
        for rel in range(1, 4):
            s = (pov + rel) & 3
            base = (rel - 1) * BELIEF_PER_OPP
            out[base:base + 5] = self.res[s] / 8.0            # ~max hand scale
            tot = self.res[s].sum() + self.unk[s]
            out[base + 5] = (self.unk[s] / tot) if tot > 0 else 0.0
        return out

    def validate_error(self, env) -> tuple[float, float]:
        """Offline diagnostic ONLY: mean per-seat L1 error (in card units) of the
        expected composition vs the TRUE raw counts from env.player_resource.
        Returns (mean_abs_error, mean_true_total). Never used as actor input; the
        tracker itself never calls player_resource for opponents."""
        err = 0.0
        true_tot = 0.0
        for s in range(4):
            true_res = np.array([int(env.player_resource(s, r)) for r in range(5)], float)
            err += float(np.abs(self.res[s] - true_res).sum())
            true_tot += float(true_res.sum())
        return err / 4.0, true_tot / 4.0
