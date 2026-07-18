"""Full-information oracle-band players (never promotion eligible)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

import fastcatan
from examples.player_base import Player, build_p2p_trade_filter, legal_actions

P2P_FILTER = build_p2p_trade_filter()
ZERO_BANNED = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
DEFAULT_PRIOR = Path(__file__).resolve().parents[1] / "models/checkpoints/alphazero_vs_ab/az_final.pt"


@lru_cache(maxsize=2)
def _load_prior(path_text: str):
    from models.alphazero.net import load_policy_value_net
    import torch

    state = torch.load(path_text, map_location="cpu", weights_only=False)
    return load_policy_value_net(state, "cpu")


class OracleAlphaBeta(Player):
    def __init__(self, *, name: str, chance_mode: int, seed: int = 0):
        super().__init__(seed=seed)
        self.name = name
        self.chance_mode = int(chance_mode)
        self.seat: int | None = None
        self.trading_enabled = True

    def bind_seat(self, seat: int) -> None:
        self.seat = int(seat)

    def set_trading_mode(self, enabled: bool) -> None:
        self.trading_enabled = bool(enabled)

    def act(self, env, mask: np.ndarray) -> int:
        actions = legal_actions(mask)
        if len(actions) == 1:
            return actions[0]
        pov = int(env.current_player) if self.seat is None else self.seat
        banned = ZERO_BANNED if self.trading_enabled else P2P_FILTER
        action = int(env.ab_decide(pov, 2, False, banned, self.chance_mode))
        return action if action in actions else actions[0]


class OracleMCTS(Player):
    """Learned prior + symbolic ``ab_value`` leaves at a frozen sim count."""

    def __init__(
        self,
        *,
        name: str,
        sims: int,
        seed: int = 0,
        checkpoint: Path = DEFAULT_PRIOR,
    ):
        super().__init__(seed=seed)
        self.name = name
        self.sims = int(sims)
        self.checkpoint = Path(checkpoint)
        self.seat: int | None = None
        self.trading_enabled = True
        self._mcts = None

    def bind_seat(self, seat: int) -> None:
        self.seat = int(seat)

    def set_trading_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled != self.trading_enabled:
            self.trading_enabled = enabled
            self._mcts = None

    def _ensure_mcts(self):
        if self.seat is None:
            raise RuntimeError("OracleMCTS must be bound to a seat before acting")
        if self._mcts is None:
            if not self.checkpoint.exists():
                raise FileNotFoundError(self.checkpoint)
            from models.alphazero.mcts_vs_fixed import MCTSvsFixed

            self._mcts = MCTSvsFixed(
                _load_prior(str(self.checkpoint.resolve())),
                device="cpu",
                sims=self.sims,
                c_puct=1.5,
                dirichlet_frac=0.0,
                seed=self.rng.getrandbits(64),
                suppress_p2p=True,
                learner_trades=self.trading_enabled,
                trade_prior_frac=0.05,
                trade_add_cap=3,
                trade_step_cost=0.01,
                value_mode="vp_margin",
                ab_depth=2,
                ab_prune=True,
                leaf_eval="ab_value",
                ab_value_scale=86_000_000.0,
                learner_seat=self.seat,
                catanatron_chance=False,
                opp_model="alphabeta",
            )
        return self._mcts

    def act(self, env, mask: np.ndarray) -> int:
        actions = legal_actions(mask)
        if len(actions) == 1:
            return actions[0]
        # The existing MCTS macro-transition keys control by current_player;
        # during forced sub-phases (especially multi-seat discards) the engine
        # keeps the turn owner there. Native AB is the faithful full-info forced
        # resolver; MCTS owns every ordinary placement/main decision.
        if int(env.flag) != 0:
            pov = int(env.current_player) if self.seat is None else self.seat
            banned = ZERO_BANNED if self.trading_enabled else P2P_FILTER
            action = int(env.ab_decide(pov, 2, True, banned, 0))
            return action if action in actions else actions[0]
        mcts = self._ensure_mcts()
        mcts.learner = self.seat
        action, visits, _root_mask = mcts.choose(
            env.snapshot(), temperature=0.0, add_root_noise=False
        )
        if action in actions:
            return int(action)
        return max(actions, key=lambda candidate: (float(visits[candidate]), -candidate))


def oracle_ab_d2(seed: int) -> Player:
    return OracleAlphaBeta(name="oracle-ab-d2", chance_mode=0, seed=seed)


def oracle_ab_d2_blur(seed: int) -> Player:
    return OracleAlphaBeta(name="oracle-ab-d2-blur", chance_mode=1, seed=seed)


def oracle_mcts_256(seed: int) -> Player:
    return OracleMCTS(name="oracle-mcts-abvalue-256", sims=256, seed=seed)


def oracle_mcts_1024(seed: int) -> Player:
    return OracleMCTS(name="oracle-mcts-abvalue-1024", sims=1024, seed=seed)
