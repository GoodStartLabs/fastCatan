"""DR-001 hidden-state perturbation referee for legal-information personas."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import fastcatan
from bridge import state_mirror as mirror
from bridge.leakage_referee import PERTURBATIONS

from .registry import PERSONA_CONFIGS, build_agent


def _legal(mask: np.ndarray) -> list[int]:
    actions: list[int] = []
    for word_index, word in enumerate(mask):
        bits = int(word)
        while bits:
            bit = bits & -bits
            actions.append(word_index * 64 + bit.bit_length() - 1)
            bits ^= bit
    return actions


def _decision(env, name: str, seed: int, seat: int) -> tuple[int, np.ndarray]:
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
    env.recompute_mask()
    env.action_mask(mask)
    agent = build_agent(name, seed)
    bind = getattr(agent, "bind_seat", None)
    if bind is not None:
        bind(seat)
    action = int(agent.act(env, mask.copy()))
    return action, mask


def run(*, games: int, seed: int, sample_every: int = 20) -> dict:
    rng = np.random.default_rng(seed ^ 0x9E3779B9)
    env = fastcatan.Env()
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
    counts = Counter()
    findings: list[dict] = []
    persona_names = [config.name for config in PERSONA_CONFIGS]

    for game_index in range(games):
        env.reset(seed + game_index)
        for step in range(8_000):
            env.action_mask(mask)
            actions = _legal(mask)
            if not actions or int(env.phase) == 3:
                break
            if step % sample_every == 0:
                base_bytes = env.snapshot()
                base_parsed = mirror.parse_snapshot(base_bytes)
                actor = (
                    int(base_parsed.gs.discarding_player)
                    if int(env.flag) == 1
                    else int(env.current_player)
                )
                opponents = [player for player in range(4) if player != actor]
                for left in range(len(opponents)):
                    for right in range(left + 1, len(opponents)):
                        a, b = opponents[left], opponents[right]
                        for tag, perturb in PERTURBATIONS.items():
                            parsed = mirror.parse_snapshot(base_bytes)
                            if not perturb(parsed.gs, a, b):
                                continue
                            mutated_bytes = mirror.to_bytes(parsed)
                            counts[f"perturb_{tag}"] += 1
                            for persona_index, name in enumerate(persona_names):
                                decision_seed = (
                                    seed + game_index * 1_000_003 + step * 101
                                    + persona_index
                                )
                                env.load_snapshot(base_bytes)
                                base_action, base_mask = _decision(env, name, decision_seed, actor)
                                env.load_snapshot(mutated_bytes)
                                mutated_action, mutated_mask = _decision(env, name, decision_seed, actor)
                                counts["persona_decisions_compared"] += 1
                                if not np.array_equal(base_mask, mutated_mask) or base_action != mutated_action:
                                    findings.append({
                                        "game": game_index,
                                        "step": step,
                                        "persona": name,
                                        "perturbation": tag,
                                        "actor": actor,
                                        "opponents": [a, b],
                                        "base_action": base_action,
                                        "mutated_action": mutated_action,
                                        "mask_equal": bool(np.array_equal(base_mask, mutated_mask)),
                                    })
                            env.load_snapshot(base_bytes)
                            env.recompute_mask()
                counts["states_checked"] += 1
                env.load_snapshot(base_bytes)
                env.recompute_mask()
                env.action_mask(mask)
                actions = _legal(mask)
            if not actions:
                break
            env.step(int(actions[rng.integers(len(actions))]))

    return {
        "games": games,
        "states_checked": counts["states_checked"],
        "persona_decisions_compared": counts["persona_decisions_compared"],
        "perturbations": {tag: counts[f"perturb_{tag}"] for tag in PERTURBATIONS},
        "n_findings": len(findings),
        "findings": findings[:50],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(games=args.games, seed=args.seed, sample_every=args.sample_every)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if result["n_findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
