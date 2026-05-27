# M4 Reproducibility

Everything needed to reproduce the M4 thesis numbers (final model vs Alpha-Beta,
the 10⁸-step soak) or to rebuild the environment on a fresh box. Captured /
updated 2026-05-27 on the development machine.

> Two things bite reproduction if missed:
> 1. **catanatron is NOT a PyPI release** — it is a pinned git commit / local
>    editable clone (§6).
> 2. **The RL interface is `1084 / 286`** (obs / actions), and the **thesis env
>    is the anaconda interpreter, not `.venv`** (§4, §5). `.venv` is a stale
>    724/296 build kept only for legacy M2 artifacts.

## 1. Hardware / OS

| | |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| Kernel | `6.17.0-23-generic` |
| CPU cores | 24 (`nproc`) |
| glibc | 2.39 (`Ubuntu GLIBC 2.39-0ubuntu8.7`) |

## 2. Toolchain

| Tool | Version |
|---|---|
| C++ compiler | GCC 13.3.0 (`Ubuntu 13.3.0-6ubuntu2~24.04.1`) |
| CMake | 4.3.2 |
| Python | 3.12.4 |

## 3. C++ build configuration

From `CMakeLists.txt` (build via `pip install -e . --no-build-isolation`, or
`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`):

- C++ standard: **C++23** (`CMAKE_CXX_STANDARD 23`, extensions OFF)
- Build type: **Release**
- Interprocedural optimization / LTO: **ON** (Release only)
- Core flags: `-O3 -march=native -fno-exceptions -fno-rtti -Wall -Wextra`

> ⚠️ **`-march=native` makes the binary CPU-specific.** To reproduce on a
> different microarchitecture you must rebuild (do not copy the `.so`). For a
> portable thesis artifact, rebuild with `-march=x86-64-v3` (or your target ISA).
>
> ⚠️ For the **final thesis runs** the debug `assert(incremental == recomputed)`
> mask check is OFF (Release). It is kept ON through M3 (root `PLAN.md` Risk 2).
> The soak (§8) re-establishes mask integrity at runtime over 10⁸ steps.

## 4. fastcatan interface — `1084 / 286` (current)

The committed source defines the interface, confirmed by compiling the headers:

| | |
|---|---|
| git commit | `f622d10` (branch `bridge`) |
| `OBS_SIZE` | **1084** (`include/obs.hpp` — 8ch×54 node + 4ch×72 edge ownership, …) |
| `NUM_ACTIONS` | **286** (`include/mask.hpp`) |
| `MASK_WORDS` | 5 (`uint64[5]`) |
| `NUM_PLAYERS` | 4 |
| reward | sparse ±1 terminal (+1 learner win, −1 any non-win terminal) |

`bridge/obs_encoder.py` and `ui/obs_decoder.py` mirror this 1084 layout;
`bridge/tests/test_obs_identity.py` passes against a 1084 build (encoder ↔ C++
`write_obs`, bit-for-bit — verified, `results/validation_1084.md`).

> ⚠️ **History / landmine.** Before 2026-05-27 the *built* extension and every
> checkpoint were a stale **724 / 296** build (May-6). The obs interface had been
> upgraded to 1084/286 in source and on the Python side, but fastcatan was never
> rebuilt and no model retrained. The build has since been brought to 1084/286
> (anaconda env, §5). **All 724/296 checkpoints are obsolete against the current
> build** — they will not load (wrong obs *and* action dim).

## 5. Python environments — TWO of them

There are two interpreters on this box. **Use anaconda for all M4 work.**

| | **anaconda3** (`/home/sinan/anaconda3/bin/python`) — THESIS ENV | `.venv` (`./.venv/bin/python`) — legacy |
|---|---|---|
| fastcatan | **1084 / 286** (editable → repo `build/`) | 724 / 296 (stale May-6 `.so`) |
| torch | 2.5.1+cu124 | 2.6.0+cu124 |
| stable-baselines3 | 2.7.0 | 2.8.0 |
| sb3-contrib | 2.7.1 | 2.8.0 |
| gymnasium | 0.29.1 | 1.2.3 |
| numpy | 1.26.4 | 2.4.4 |
| catanatron | 3.3.0 editable (§6) | none |

Train **and** eval M4 in the anaconda env (1084 fastcatan + catanatron + the
1084 model all line up there). `.venv` holds only the obsolete 724 M2 artifacts.

Build deps (both envs): `nanobind 2.12.0`, `scikit-build-core 0.12.2`, `cmake`,
`ninja`. Rebuild the extension from current source:

```bash
pip install --no-build-isolation -e . --config-settings=editable.rebuild=true
```

> **Use `editable.rebuild=true`.** A plain `pip install -e .` compiles the
> extension **once** at install time and never again — that is exactly how the
> May-6 build silently stayed at 724/296 while source moved to 1084/286 (§4
> landmine). With `editable.rebuild=true`, scikit-build-core re-runs the C++
> build automatically on the next `import fastcatan` after any source edit. The
> anaconda env was reinstalled this way on 2026-05-27 (verified `import
> fastcatan` → `1084 286`).

## 6. Catanatron (the Alpha-Beta baseline) — NOT PyPI

The thesis Alpha-Beta opponent is **Catanatron's** `AlphaBetaPlayer`, run inside
Catanatron's reference engine via `bridge/`. The required version is **catanatron
3.3.0**, a dev build that is **not on PyPI** (PyPI tops out at 3.2.1, a stripped
core with no `AlphaBetaPlayer`, no `ValueFunctionPlayer`, no domestic trades).

- **Pinned commit: `41ba0db`** ("deterministic discards", #362). This is the
  commit the local clone `/home/sinan/Desktop/msc/catanatron` sits on and the
  one the bridge is verified against — **281/281 `bridge/tests` pass** (2026-05-27,
  anaconda env, against the 1084/286 build). Installed editable:
  `pip install -e "/home/sinan/Desktop/msc/catanatron[gym]"`.
- The pin is recorded in the repo at **`requirements.txt`** (root):
  `catanatron[gym] @ git+https://github.com/bcollazo/catanatron.git@41ba0db`.
  Install with `pip install -r requirements.txt`.
- Canonical pin for a fresh box (full SHA):

```
pip install "catanatron @ git+https://github.com/bcollazo/catanatron.git@41ba0db5bc9d95b13a32e1ed7710094382a3b87a"
```

> ⚠️ Pin to the **commit**, not `main`. Newer catanatron builds (incl. the PyPI
> 3.3.0 wheel) folded `catanatron.models.tiles` (`LandTile`, `Port`) into
> `catanatron.models.map`; the bridge imports from `models.tiles`, so an
> unpinned install breaks bridge import. `41ba0db` still has `models.tiles`.
> (`38207ca` — the prior pin in earlier doc revisions — is an ancestor that also
> works but predates the deterministic-discards fix, which affects seeded RNG.)

Provides `catanatron.players.minimax.AlphaBetaPlayer`,
`catanatron.players.value.ValueFunctionPlayer`, and the domestic-trade action
types the bridge relies on (`CONFIRM_TRADE` value is an 11-tuple, partner color
at index 10 — matches `catanatron_bridge.py`). `AlphaBetaPlayer` has a 20s/move
internal deadline; default depth 2.

## 7. Seeds & determinism

- Tournament: `--seed S` → game `g` uses seed `S + g`; the bridge RNG is seeded
  per game with the same value. Default `S = 42`.
- Soak: `--seed` (default 7) seeds both the env game-seed sequence and the
  opponent/random-policy RNG.
- **`PYTHONHASHSEED` must be set in the environment before launch** for
  catanatron runs — `RandomPlayer` (`random.choice`) and set-iteration order
  depend on it (`bridge/PLAN.md`). The tournament warns if it is unset.

## 8. Commands (anaconda env)

```bash
AP=/home/sinan/anaconda3/bin/python

# Train the M4 model on the 1084/286 interface (768 envs = the winning config).
$AP -m models.train_ppo --num-envs 768 --total-steps 20_000_000 --run-name ppo_1084_20m
# (>=30M for gate margin; 20M is near-convergence. See training notes below.)

# Thesis gate: final model vs Alpha-Beta, >=1000 games, win rate + 95% CI.
PYTHONHASHSEED=0 $AP -m AB.tournament \
    --ckpt models/checkpoints/ppo_1084_20m/ppo_final.zip \
    --games 1000 --opponent alphabeta --ab-depth 2 --ab-prune --seed 42
# GATE: 95% Wilson CI lower bound > 0.25.  Result JSON -> AB/results/.
# Cost: AlphaBeta ~6.4 s/game unpruned (~1.8 h / 1000 games); --ab-prune cuts it.

# 10^8-step stability soak (no catanatron; runs in either env).
$AP -m AB.soak --steps 100000000 --seed 7
# PASS iff: no exception, all obs finite, every action legal, RSS growth < 1.5x.
```

## 9. Training config

PPO via `models/train_ppo.py` (`MaskablePPO` + `MaskableActorCriticPolicy`).
Defaults (the winning config — 768 envs clears the M2 >90%-vs-random gate):

| hyperparam | value |
|---|---|
| num_envs | 768 (DummyVecEnv; `--subproc` is slower here) |
| n_steps | 512 |
| batch_size | 2048 |
| n_epochs | 4 |
| learning_rate | 3e-4 |
| gamma | 0.999 |
| gae_lambda | 0.95 |
| ent_coef | 0.01 |
| clip_range | 0.2 |
| seed | 42 |

Trained **with** the trade-compose stall cap (`models/env.py`,
`MAX_TRADE_COMPOSE_PER_TURN = 20`); opponents during training are 3
uniform-random-legal seats.

- **M4 model:** `ppo_1084_20m` (or a longer run), trained on the **1084/286**
  interface in the anaconda env. This is the model the tournament evaluates.
- **Convergence:** 10M steps ≈ 88% vs random (fails the M2 CI bar); converges
  ~15–20M; ≥30M for margin. 20M is a usable first model; 50M gave the 724-era
  margin.
- **No checkpoints currently exist.** `models/checkpoints/` was emptied on
  2026-05-27 — all 724/296 artifacts (`ppo_capped_50m`, `sp_smoke_1m`,
  `sp_smoke_5m`, `sweep`, and earlier `ppo_random_*`) were deleted because they
  are unloadable against the 1084/286 build (wrong obs *and* action dim). The
  next training run starts from scratch on the 1084/286 interface; `models/
  selfplay/` (source) was untouched.
- **Historical reference (724/296, now deleted):** `ppo_capped_50m/ppo_final.zip`
  was the M2 gate-passing model on the *old* interface — 99.4% vs random
  (1000-game sampling, 95% CI [0.987, 0.997]). Recorded here for the number only;
  the weights no longer exist.

> The "final model" is a `--ckpt` flag — swap in any future (e.g. M3 self-play)
> 1084/286 checkpoint without code changes.

## 10. Known caveats

- **Interface match is mandatory.** Before any eval/train, check
  `fastcatan.OBS_SIZE` vs the checkpoint's `observation_space.shape[0]`.
  `AB/policy.py` raises on mismatch. A 724 checkpoint + a 1084 build (or vice
  versa) is the failure that blocked M4 mid-session.
- AlphaBetaPlayer has a **20s/move** deadline. At depth 2 with `--ab-prune` a
  1000-game run is tractable; unpruned is ~6.4 s/game. The JSON records
  `--ab-depth`/`--ab-prune` with every result.
- Eval uses **sampling** policy by default. Argmax (`--deterministic`) can stall
  in the within-turn trade-compose loop (`models/PLAN.md`); the bridge bounds it
  via `_COMPOSE_LOOP_CAP`, but sampling is the validated mode.
- Catanatron's longest-road is internally inconsistent on road cuts; fastcatan is
  rule-correct. The ≤1–2% differential residual is documented and exempted in
  `bridge/tests/test_differential.py` — it does not affect tournament outcomes.
