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
catanatron bridge gate). Benchmark seeds + seat-rotation logic and the results
schema are frozen with the ladder (`results/SCHEMA.md`, `LADDER_VERSION.md` — added
in spec 0.3 §E / 1.0).

**Training-loop wiring proofs.** `tests/test_wiring.py`, `bin/train_smoke.py`.

**The freeze machinery itself.** `IMMUTABLE.md`, `scripts/check_frozen.sh`,
`scripts/check_env.py`.

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
