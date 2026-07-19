"""Perspective-pure view decoded from the frozen legal observation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import fastcatan

from .topology import DICE_PIPS, NODE_TO_HEXES, NODE_TO_PORT

PLAYER_OFFSET = 0
SELF_PRIVATE_OFFSET = 64
NODE_OFFSET = 80
EDGE_OFFSET = 512
HEX_RESOURCE_OFFSET = 800
HEX_NUMBER_OFFSET = 914
PORT_OFFSET = 933
ROBBER_OFFSET = 987
GAME_OFFSET = 1006
TRADE_OFFSET = 1057


@dataclass(frozen=True)
class PublicView:
    seat: int
    resources: tuple[int, int, int, int, int]
    hand_sizes: tuple[int, int, int, int]
    public_vp: tuple[int, int, int, int]
    production: tuple[tuple[float, float, float, float, float], ...]
    hex_resources: tuple[int, ...]
    hex_numbers: tuple[int, ...]
    port_types: tuple[int, ...]
    node_levels_by_relative_seat: tuple[tuple[int, int, int, int], ...]
    robber_hex: int
    trade_proposer: int | None

    @property
    def own_production(self) -> tuple[float, float, float, float, float]:
        return self.production[self.seat]

    @classmethod
    def from_env(cls, env, seat: int) -> "PublicView":
        obs = np.zeros(fastcatan.OBS_SIZE, dtype=np.float32)
        env.write_obs(seat, obs)
        resources = tuple(int(env.player_resource(seat, r)) for r in range(5))
        hand_sizes = tuple(int(env.player_handsize(p)) for p in range(4))
        public_vp = tuple(int(env.player_vp_public(p)) for p in range(4))

        node_raw = obs[NODE_OFFSET:EDGE_OFFSET].reshape(54, 8)
        node_levels: list[tuple[int, int, int, int]] = []
        for node in range(54):
            node_levels.append(tuple(
                2 if node_raw[node, 2 * rel + 1] > 0.5
                else 1 if node_raw[node, 2 * rel] > 0.5
                else 0
                for rel in range(4)
            ))
        hex_resources = tuple(
            int(value) for value in np.argmax(
                obs[HEX_RESOURCE_OFFSET:HEX_NUMBER_OFFSET].reshape(19, 6), axis=1
            )
        )
        hex_numbers = tuple(
            int(round(float(value) * 12))
            for value in obs[HEX_NUMBER_OFFSET:PORT_OFFSET]
        )
        port_types = tuple(
            int(value) for value in np.argmax(
                obs[PORT_OFFSET:ROBBER_OFFSET].reshape(9, 6), axis=1
            )
        )
        robber_hex = int(np.argmax(obs[ROBBER_OFFSET:GAME_OFFSET]))

        production = [[0.0] * 5 for _ in range(4)]
        for node, levels in enumerate(node_levels):
            for rel, level in enumerate(levels):
                if level == 0:
                    continue
                absolute = (seat + rel) & 3
                for hex_id in NODE_TO_HEXES[node]:
                    resource = hex_resources[hex_id]
                    if resource >= 5 or hex_id == robber_hex:
                        continue
                    production[absolute][resource] += (
                        level * DICE_PIPS.get(hex_numbers[hex_id], 0)
                    )

        proposer_rel = int(np.argmax(obs[TRADE_OFFSET:TRADE_OFFSET + 5]))
        trade_proposer = None if proposer_rel == 4 else (seat + proposer_rel) & 3
        return cls(
            seat=seat,
            resources=resources,
            hand_sizes=hand_sizes,
            public_vp=public_vp,
            production=tuple(tuple(row) for row in production),
            hex_resources=hex_resources,
            hex_numbers=hex_numbers,
            port_types=port_types,
            node_levels_by_relative_seat=tuple(node_levels),
            robber_hex=robber_hex,
            trade_proposer=trade_proposer,
        )

    def node_production_score(self, node: int, *, city_multiplier: int = 1) -> float:
        score = 0.0
        diversity: set[int] = set()
        for hex_id in NODE_TO_HEXES[node]:
            resource = self.hex_resources[hex_id]
            if resource >= 5:
                continue
            diversity.add(resource)
            score += city_multiplier * DICE_PIPS.get(self.hex_numbers[hex_id], 0)
        return score + 0.35 * len(diversity)

    def port_synergy(self, node: int) -> float:
        port = NODE_TO_PORT.get(node)
        if port is None:
            return 0.0
        port_type = self.port_types[port]
        if port_type == 5:
            return 1.5
        return 0.65 * self.own_production[port_type] + 0.5

    def estimated_hand(self, player: int) -> tuple[float, float, float, float, float]:
        """Legal public estimate from hand size and visible production only."""
        weights = [value + 0.5 for value in self.production[player]]
        total = sum(weights)
        hand = self.hand_sizes[player]
        if total <= 0:
            return (hand / 5.0,) * 5
        return tuple(hand * weight / total for weight in weights)
