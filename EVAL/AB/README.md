# EVAL/AB/ — M4: Alpha-Beta eval + final-model thesis gate

The thesis claim lives here: **the trained RL agent beats Catanatron's
Alpha-Beta with statistical significance — win rate > 25% over ≥1000 four-player
games, 95% CI** (`PLAN.md` M4, root). 0.25 is the 4-player chance baseline.

The agent plays through `EVAL/bridge/CatanatronBridge` *inside Catanatron's reference
engine*, so the numbers are directly comparable to Catanatron paper baselines.

## The gate, restructured (2026-06-07) — two tiers

Development iterates on the fast in-repo native AB; catanatron is the final
exam only:

1. **Dev gate (every iteration)** — native AB-d2 ladder, pure fastcatan:

   ```bash
   python -m models.alphazero.evaluate --ckpt <ckpt> \
       --opponent alphabeta --ab-depth 2 --sims 512 --games 200
   ```

   Wilson CI; promote on CI-low > 0.25. Calibration: the hybrid recipe
   scored 29.5% native-d2 vs 32.5% bridge-d2 — native is the conservative
   proxy.
2. **Final gate (ONCE, end of project)** — ≥1000 bridge games vs
   `AlphaBetaPlayer` d2 (shuffled seating, `PYTHONHASHSEED=0`):
   CI-low > 0.25. Run only on the final self-contained model.

**Self-containment requirement:** the thesis agent's search may not call
`ab_value` / `ab_decide` at inference — learned prior + learned leaf value +
learned opponent model only. AB-generated games/labels (IL, distillation)
are training-time-only and in-bounds. The hybrid configuration below
(`ab_value` leaves + AB in-tree opponent model) is therefore the
**reference recipe**, not the thesis agent. De-catanatronization order:
(1) learned leaf value (`il_pretrain --value-target ab_value` distillation
→ search with `--leaf-eval net`), (2) learned in-tree opponent model,
(3) re-train the prior with the learned value.

## Native AlphaBeta — a fast, faithful training opponent

The eval above runs Catanatron's **real** AlphaBeta through the bridge (~6.4 s/game,
crashes on P2P trades) — fine for the final gate, far too slow to *train* against.
So the same player is ported natively into the fastcatan C++ engine
(`src/catan/search.cpp`, `include/search.hpp`), exposed as:

```python
env = fastcatan.Env(); env.reset(seed)
env.ab_decide(pov, depth=2, prune=False)   # -> best flat action id (0xFFFFFFFF if none)
env.ab_value(pov)                          # -> Catanatron base_fn heuristic value
```

It is a faithful port of `catanatron.players.minimax.AlphaBetaPlayer`
(depth-2 expectimax over dice / dev-draw / robber-steal chance nodes, alpha-beta,
`list_prunned_actions`) + `value.base_fn` (`DEFAULT_WEIGHTS`). The engine refactor
that makes the chance forks possible is `rules.cpp::expand_action`
(forced-outcome cores split out of the RNG handlers; the RNG sim path stays
byte-identical — perft hash unchanged).

**Fidelity (validated, `test_native_ab_fidelity.py`, run via the bridge):**
- `ab_value` == `base_fn(DEFAULT_WEIGHTS)` to **machine precision** (worst rel
  error 1.9e-16 over 4800 state×seat pairs; exact in MAIN phase).
- On deterministic 1:1-action decisions, Catanatron's depth-1 pick achieves
  **exactly** fastcatan's best value (100 %) — every raw move difference is a
  pure value tie (different tie-break order).
- Two deliberate, documented deviations (both *more* correct than the
  reference): BUY_DEV forks the true remaining deck; robber-steal forks the
  victim's real hand (Catanatron uses an info-set blur / flat 1/5).

**Train against it** (`models/env.py`, `models/train_ppo.py`):

```bash
python -m models.train_ppo --opponent alphabeta --ab-depth 1 --num-envs 768 ...
```

Throughput (single env): `random` ~51k learner-steps/s, **depth-1 ~45k**
(≈ Catanatron `ValueFunctionPlayer`, nearly free and already crushes a random
learner), depth-2 ~5k (~10× slower). Depth-1 is the recommended training
opponent; bump to depth-2 / `--ab-prune` for a stronger curriculum. This is the
"opponent-in-pool" lever for the M4-blocked-on-M3 gap.

Pure-engine checks live in `tests/test_alphabeta.py`; the catanatron-fidelity
gate in `test_native_ab_fidelity.py` (this dir).

## Files

| File | Role |
|---|---|
| `policy.py` | wraps a trained checkpoint as a bridge `PolicyFn` (`obs, mask, rng -> int`). Registry mirrors `models/eval.py`; only `ppo` wired today. Raises on obs/action-dim mismatch. |
| `mcts_policy.py` | **state-aware** bridge policy: the bridge stashes the live `Game` each `decide()`; this injects it into a fastcatan `Env` (`bridge/state_inject`), calls `recompute_mask()` (injected states carry a stale cached mask — without this every root step is a masked no-op), runs the hybrid `MCTSvsFixed` (learned prior + `ab_value` leaves), and answers within the bridge's action mask (fallbacks counted in the result JSON). |
| `tournament.py` | the harness: policy-via-bridge vs `AlphaBetaPlayer`/`ValueFunctionPlayer`/`RandomPlayer`. Win rate + 95% Wilson CI + thesis gate → `results/*.json`. `--policy mcts` for search agents, `--rotate-seats` for seat-balanced runs (default is RED/seat-0 only), `--model-ab-depth/--model-ab-prune` to match the in-tree opponent model to the actual table. |
| `soak.py` | 10⁸-step stability soak (pure fastcatan): finite-obs + mask-integrity + leak checks. |
| `REPRODUCIBILITY.md` | toolchain, build flags, the **two-env** setup, **catanatron git pin**, seeds, train config. |
| `results/` | tournament result JSONs + `validation_1084.md` (pipeline validation). |

## Environment

The RL interface is **obs 1084 / actions 286**. The repo `.venv` carries
fastcatan + the **pinned catanatron** (3.3.0 @ git `41ba0db`, not PyPI —
newer builds move `models.tiles` → `models.map` and break the bridge import;
see `REPRODUCIBILITY.md`) and is what the current results were produced
under. `soak.py` needs only fastcatan.

```bash
# Smoke (seconds): any policy vs random through the bridge.
PYTHONPATH=.:EVAL python -m AB.tournament --games 20 --opponent random --ckpt <ckpt>

# 10^8 soak (~minutes at ~70k steps/s).
PYTHONPATH=.:EVAL python -m AB.soak --steps 100000000 --seed 7
```

The evaluated model is a `--ckpt` flag — reactive checkpoints (`--policy
ppo`, SB3 .zip) and search checkpoints (`--policy mcts`, AZ .pt) both work;
the interface must be 1084/286.

## State-aware hybrid search through the bridge (`--policy mcts`)

Reactive policies topped out at 0/200 here; the configuration that reached
parity on the native engine is a *search* agent, which needs the live game
state — wired 2026-06-06 (run under the repo `.venv`, which also carries the
pinned catanatron):

```bash
PYTHONPATH=.:EVAL python -m AB.tournament --policy mcts \
    --ckpt models/checkpoints/il_ab_d2_vpm/il_final.pt \
    --games 200 --mcts-sims 512 --model-ab-depth 2 --model-ab-prune \
    --opponent alphabeta --ab-depth 2 --ab-prune --no-trades --rotate-seats
```

**Reference results (2026-06-06), hybrid = IL-clone prior + `ab_value`
leaves (two-scale lexicographic squash, `--ab-value-scale 86e6`):**

| arena | result |
|---|---|
| native AB-d1, ≥512 sims (600 g) | **29.0% [25.5–32.8] — above 25% parity** |
| native AB-d2, 256–512 sims (600 g) | **23.3–23.75% — at parity** (was 0/200 for every reactive policy) |
| native AB-d2 *pruned* control (40 g) | 17.5% [8.8–32.0] — pruning ≈ no strength change |
| **bridge** AB-d2 pruned, rotated, 256 sims (100 g) | 5.0% [2.2–11.2] — pre-seat-fix (v2) |
| **bridge** AB-d2 pruned, shuffled seating, 512 sims (200 g, v6) | **32.5% [26.4–39.3] — GATE PASS** (hybrid reference) |

**The native→bridge transfer gap — RESOLVED (2026-06-06).** Ruled out by
experiment: injected value fidelity (machine precision), opponent pruning
strength, in-tree model depth. Real fixes: faithful in-tree chance model
(`--model-catanatron-chance`), policy-owned robber composite, and the
decisive one — **catanatron shuffles seating**, so the construction-time
seat had the search optimizing an opponent's position in ~75% of games
(pinning bridge runs at 0.25×native ≈ 6%); fixed by per-decision
`_sync_seat`. Details in the Status section below.

## Status (2026-06-06)

History, compressed: harness/soak/pin/pipeline validated 2026-05 (281/281
bridge tests, obs-identity 5/5, `results/validation_1084.md`,
`REPRODUCIBILITY.md`); every reactive policy — the 50M PPO seed (89.5% vs
`RandomPlayer` through this same bridge), 200M league self-play, arch-sweep
nets — scored **0/200–0/500 vs AlphaBeta** here, which is what motivated the
search campaign (root README).

- [x] **Native AB ladder beaten (2026-06-06):** hybrid search above parity vs
      d1 (29.0% [25.5–32.8]), at parity vs d2 (23.75% [19.8–28.2]) — see the
      campaign section in the root README for the design rules
      (dense value targets, neuro-symbolic leaves, sims scaling).
- [x] State-aware MCTS bridge policy + seat rotation wired (`--policy mcts`,
      `--rotate-seats`); `Env.recompute_mask()` added for injected states.
- [x] **Transfer gap CLOSED (2026-06-06): GATE PASS at 200 games — 65/200 =
      32.5% [26.4–39.3] vs catanatron AB-d2** (`results/tournament_mcts_
      alphabeta_20260606_152919.json`). The gap was three real fixes (faithful
      in-tree model `--model-catanatron-chance`, policy-owned robber composite,
      catanatron-line teacher data) plus the decisive one: **catanatron
      shuffles seating** (`State.__init__` `random.sample`) — the policy's
      list-position seat had the search optimizing an opponent in ~75% of
      games, pinning bridge runs at 0.25×native ≈ 6%. Fixed by per-decision
      `_sync_seat` (commit 19e2698). Full hunt: `model_divergence_*.json` +
      git history e6f5b3d→335833a.
- [x] **SUPERSEDED (2026-06-07): the official 1000-game run of the HYBRID
      recipe (interrupted at 150/1000, 67 wins = 44.7%) is retired
      un-resumed.** Gate restructure (see "The gate, restructured" above):
      the ≥1000-game bridge run happens ONCE, at the very end, on the final
      **self-contained** model — no `ab_value` leaves, no `ab_decide`
      in-tree opponent model at inference. The hybrid's 200-game PASS above
      stands as the reference result; development now iterates on the
      native dev gate. (For the record, the hybrid run was seeded and
      reproducible — config in `results/tournament_mcts_alphabeta_
      20260606_152919.json` at `--games 1000 --seed 2026`.)
- [x] **Full 10⁸-step soak PASS (2026-06-07):** 10⁸ steps / 99,244 episodes of
      random-legal play (seed 7), 0 no-winner, 0 per-step violations (finite
      obs, non-empty mask, action-in-mask), seat wins balanced
      [24636, 24909, 24703, 24996], 78.4k steps/s (21.3 min), RSS
      40.2 → 22.3 MiB (growth 0.55×, guard <1.5× OK).
      Log `DEBUG/logs/soak_1e8_20260607.log`.
- [ ] Record thesis-gate result (the 1000-game JSON) — **final self-contained
      model only, end of project** (two-tier gate, 2026-06-07).
