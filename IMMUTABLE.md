# Immutable layer — frozen at `substrate-v1`

The files and pins the research loop (Phase 3) may **never** edit. They encode
the rules, the legal-information boundary, the evaluator, and the tests that
prove all three. Changing anything here is a substrate change, not research:
it requires a human-reviewed PR, a regenerated freeze baseline, and — for any
edit that is eval-visible (rules, obs, evaluator, seeds, promotion) — a
**ladder-version bump** so results are never silently compared across substrates.

Enforced by `scripts/check_frozen.sh` (run in pre-commit / CI): it fails if a
diff versus the `substrate-v1` tag touches any frozen path. `tests/test_wiring.py`
and `tests/test_state_mirror_layout.py` are themselves frozen — the guards guard
the guards.

## Frozen paths

**Engine + rules (the simulator).**
- `src/catan/` — `rules.cpp` (transition), `obs.cpp` (encoders + `namespace norm`),
  `search.cpp` (native AB), `batched_env.cpp`.
- `include/` — `state.hpp` (`GameState`/`BoardLayout` layout; `MAX_TURNS=2000`,
  `MAX_TRADE_COMPOSE_PER_TURN`, `WIN_VP`), `rules.hpp`, `obs.hpp`, `topology.hpp`,
  `rng.hpp`, `mask.hpp`, `search.hpp`, `batched_env.hpp`.
- `bindings/` — the nanobind surface (`OBS_SIZE`, `NUM_ACTIONS`, the accessor API).

**Legal-information boundary.** `src/catan/obs.cpp` + `include/obs.hpp` (the 1084
perspective obs), and `EVAL/bridge/obs_encoder.py` (the catanatron→1084 mirror).
`write_obs_full`/`OBS_FULL` and `ab_value`/`ab_decide` stay judge/oracle-only
(DR-001). The leakage referee below is the frozen gate every obs function passes.

**Correctness + parity harnesses.**
- `tests/fuzz_invariants.cpp`, `tests/test_invariants.py`, `tests/test_determinism.py`.
- `EVAL/AB/test_native_ab_fidelity.py`.
- `EVAL/bridge/` — the bridge core (`state_mirror.py`, `state_inject.py`,
  `rng_force.py`, `action_codec.py`, `topology_map.py`, `obs_encoder.py`), the
  differential harness (`tests/test_differential.py` incl. `_exempt_lr_cut_quirk`),
  the trade driver (`trade_differential.py`), the FSM probes (`tests/test_trade_fsm.py`),
  and the leakage referee (`leakage_referee.py`).
- `tests/test_state_mirror_layout.py` — pins the full ctypes mirror layout (the
  384-B size assert alone is insufficient; §4.2).

**Evaluator + promotion.** `models/eval.py` (`wilson_ci` — the single promotion
statistic), `models/selfplay/eval_seats.py` (`play_one` per-seat driver),
`models/selfplay/gate.py` (2v2 seat-rotation), `EVAL/AB/tournament.py` (the final
catanatron bridge gate). The results schema `results/schema.py` + `results/SCHEMA.md`
(v0 contract). Benchmark seeds + seat-rotation logic and the frozen roster live in
`LADDER_VERSION.md` (added by spec 1.0). Generated result stores
(`results/ladder.parquet`, `results/ladder.md`) are NOT frozen — they are
append-only outputs.

**Entry points + wiring proofs.** `bin/` (`train_smoke.py` — one-command train
proof; `tournament.py` — one-command reproducible per-seat tournament),
`scripts/tournament.sh`, and `tests/test_wiring.py`.

**The freeze machinery itself.** `IMMUTABLE.md`, `IMMUTABLE.lock` (sha256 manifest),
`scripts/check_frozen.sh`, `scripts/gen_immutable_lock.sh`, `scripts/check_env.py`.

## Frozen pins

| Pin | Value | Where |
|---|---|---|
| catanatron commit | `d3f4ad0` (fork main; = `41ba0db` + rule fix #377) | `scripts/check_env.py` |
| Python | 3.12 | `environment.yml` |
| Hash seed | `PYTHONHASHSEED=0` (all test/eval runs) | — |
| Obs width | `OBS_SIZE=1084` (legal) / `OBS_FULL_SIZE=1132` (oracle) | `bindings/` |
| Action space | `NUM_ACTIONS=286` | `bindings/` |
| Length backstop | `MAX_TURNS=2000` → ENDED, no-winner (rule-correct truncation; ~110/10⁷) | `include/state.hpp` |
| Trade all-decline | auto-cancel to proposer's turn (B4, accepted 0.2) | `src/catan/rules.cpp` |

## Explicitly NOT frozen — the research surface (Phase 3 may edit)

`models/` nets, losses, optimization, opponent-sampling curriculum, teacher-data
usage (e.g. `models/train_ppo.py`, `models/alphazero/`, net architectures);
observation construction **above** the locked legal state (belief features per
DR-001); action representation **above** the locked engine API; approved search
components. These are where research happens and `check_frozen.sh` must stay
silent on them.

## Freeze bump history

- **v1 → v1.1 (2026-07-19):** `bin/train_smoke.sh` W&B project routing
  (`goodsettler` → `goodsettler-rl`) + guard re-baseline (`check_frozen.sh`
  default base `substrate-v1` → `substrate-v1.1`). Non-eval-visible
  (no rules/obs/evaluator/seeds/promotion change); no ladder-version bump.
