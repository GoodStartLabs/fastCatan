"""Pool sharded eval-result JSONs (scripts/hpc/eval_array.sbatch) into one
aggregate with a fresh Wilson CI.

    python scripts/hpc/merge_results.py "EVAL/AB/results/hpc_12345/*/*.json"
    python scripts/hpc/merge_results.py shard1.json shard2.json ...

Handles both drivers:
  AB.tournament        — pools `bridge_wins`, evaluates the >25% thesis gate.
  AB.mixed_tournament  — pools `ab_wins`/`agent_wins` + trade tallies,
                         evaluates AB vs its fair share.
Shards must agree on the config keys (ckpt, policy, trades, depths, n_agents);
mismatches abort — pooling different experiments is meaningless.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

try:  # single source of truth when run from the repo root
    sys.path.insert(0, "EVAL")
    from models.eval import wilson_ci  # type: ignore
except Exception:  # standalone fallback (same math)
    def wilson_ci(wins: int, n: int, z: float = 1.96):
        if n == 0:
            return 0.0, 0.0
        p = wins / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return (c - h) / d, (c + h) / d

_CONFIG_KEYS = [
    "ckpt", "policy", "algo", "n_agents", "n_ab", "mcts_sims", "leaf_eval",
    "model_ab_depth", "model_ab_prune", "model_opp", "deterministic",
    "enable_trades", "ab_depth", "ab_prune", "opponent",
]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    paths: list[str] = []
    for arg in sys.argv[1:]:
        hits = glob.glob(arg)
        paths.extend(hits if hits else [arg])
    shards = [json.loads(Path(p).read_text()) for p in sorted(set(paths))]
    if not shards:
        sys.exit("no result JSONs matched")

    ref = {k: shards[0].get(k) for k in _CONFIG_KEYS}
    for i, s in enumerate(shards[1:], 1):
        for k in _CONFIG_KEYS:
            if s.get(k) != ref[k]:
                sys.exit(f"config mismatch in shard {i}: {k}="
                         f"{s.get(k)!r} != {ref[k]!r} — refusing to pool")

    games = sum(s["games"] for s in shards)
    decided = sum(s["decided"] for s in shards)
    no_winner = sum(s["no_winner"] for s in shards)
    elapsed = sum(s.get("elapsed_s", 0.0) for s in shards)
    pooled = {
        "pooled_from": len(shards),
        "shard_seeds": [s["seed"] for s in shards],
        "shard_games": [s["games"] for s in shards],
        **ref,
        "games": games, "decided": decided, "no_winner": no_winner,
        "elapsed_s": elapsed,
        "s_per_game": elapsed / games if games else 0.0,
    }

    if "bridge_wins" in shards[0]:                       # AB.tournament
        wins = sum(s["bridge_wins"] for s in shards)
        lo, hi = wilson_ci(wins, decided)
        rate = wins / decided if decided else 0.0
        gate = shards[0].get("gate_baseline", 0.25)
        pooled.update(bridge_wins=wins, win_rate=rate,
                      ci95_low=lo, ci95_high=hi,
                      gate_baseline=gate, gate_pass=lo > gate)
        print(f"pooled {len(shards)} shards: {wins}/{decided} = {rate:.4f} "
              f"[{lo:.4f}, {hi:.4f}]  GATE(CI-low>{gate}): "
              f"{'PASS' if lo > gate else 'FAIL'}")
    else:                                                # AB.mixed_tournament
        ab = sum(s["ab_wins"] for s in shards)
        agent_key = "agent_wins" if "agent_wins" in shards[0] else "ppo_wins"
        ag = sum(s[agent_key] for s in shards)
        n_agents = ref.get("n_agents") or 3
        ab_fair = shards[0].get("ab_fair_share", (4 - n_agents) / 4.0)
        ab_lo, ab_hi = wilson_ci(ab, decided)
        ag_lo, ag_hi = wilson_ci(ag, decided)
        ab_rate = ab / decided if decided else 0.0
        ag_rate = ag / decided if decided else 0.0
        trades = {}
        for s in shards:
            for k, v in (s.get("trades_total") or {}).items():
                trades[k] = trades.get(k, 0) + v
        pooled.update(ab_wins=ab, ab_win_rate=ab_rate, ab_ci95=[ab_lo, ab_hi],
                      ab_fair_share=ab_fair, agent_wins=ag,
                      agent_win_rate=ag_rate, agent_ci95=[ag_lo, ag_hi],
                      agent_per_seat_rate=ag_rate / n_agents,
                      trades_total=trades,
                      trades_per_game={k: v / games for k, v in trades.items()})
        verdict = ("AB ABOVE fair share" if ab_lo > ab_fair else
                   "AB BELOW fair share — suppressed" if ab_hi < ab_fair else
                   "AB ~ fair share")
        print(f"pooled {len(shards)} shards ({games} games, {decided} decided)")
        print(f"  AB block: {ab}/{decided} = {ab_rate:.4f} "
              f"[{ab_lo:.4f}, {ab_hi:.4f}]  fair={ab_fair}  -> {verdict}")
        print(f"  agents:   {ag}/{decided} = {ag_rate:.4f} "
              f"[{ag_lo:.4f}, {ag_hi:.4f}]  per-seat "
              f"{ag_rate / n_agents:.4f} vs 0.25")
        if trades.get("offers"):
            print(f"  trades:   {trades['offers'] / games:.2f} offers/game, "
                  f"{trades.get('confirms', 0) / games:.2f} confirmed/game")

    out = Path(paths[0]).parent.parent / "pooled.json"
    out.write_text(json.dumps(pooled, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
