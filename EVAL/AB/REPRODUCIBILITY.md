# M4 Reproducibility

Everything needed to reproduce the M4 thesis numbers (final model vs Alpha-Beta,
the 10⁸-step soak) or to rebuild the environment on a fresh box. Captured /
updated 2026-05-27 on the development machine.

> Two things bite reproduction if missed:
> 1. **catanatron is NOT a PyPI release** — it is a pinned git commit / local
>    editable clone (§6).
> 2. **The RL interface is `1084 / 286`** (obs / actions), and the **thesis env
>    is the anaconda interpreter** (§4, §5).

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

## 4. fastcatan interface — `1084 / 286`

The committed source defines the interface, confirmed by compiling the headers:

| | |
|---|---|
| git commit | `f622d10` (branch `bridge`) |
| `OBS_SIZE` | **1084** (`include/obs.hpp` — 8ch×54 node + 4ch×72 edge ownership, …) |
| `NUM_ACTIONS` | **286** (`include/mask.hpp`) |
| `MASK_WORDS` | 5 (`uint64[5]`) |
| `NUM_PLAYERS` | 4 |
| reward | sparse ±1 terminal (+1 learner win, −1 any non-win terminal) |

`EVAL/bridge/obs_encoder.py` and `DEBUG/ui/obs_decoder.py` mirror this 1084 layout;
`EVAL/bridge/tests/test_obs_identity.py` passes (encoder ↔ C++ `write_obs`,
bit-for-bit — verified, `results/validation_1084.md`).

## 5. Python environment — anaconda

All M4 work runs in the **anaconda** interpreter
(`/home/sinan/anaconda3/bin/python`), where the 1084/286 fastcatan, catanatron,
and the trained model all line up.

| | **anaconda3** (`/home/sinan/anaconda3/bin/python`) — THESIS ENV |
|---|---|
| fastcatan | **1084 / 286** (editable → repo `build/`) |
| torch | 2.5.1+cu124 |
| stable-baselines3 | 2.7.0 |
| sb3-contrib | 2.7.1 |
| gymnasium | 0.29.1 |
| numpy | 1.26.4 |
| catanatron | 3.3.0 editable (§6) |

Build deps: `nanobind 2.12.0`, `scikit-build-core 0.12.2`, `cmake`, `ninja`.
Rebuild the extension from current source:

```bash
pip install --no-build-isolation -e . --config-settings=editable.rebuild=true
```

> **Use `editable.rebuild=true`.** A plain `pip install -e .` compiles the
> extension **once** at install time and never again, so the binary silently
> goes stale when `obs.hpp`/`mask.hpp` change. With `editable.rebuild=true`,
> scikit-build-core re-runs the C++ build automatically on the next `import
> fastcatan` after any source edit. The anaconda env is installed this way
> (verified `import fastcatan` → `1084 286`).

## 6. Catanatron (the Alpha-Beta baseline) — NOT PyPI

The thesis Alpha-Beta opponent is **Catanatron's** `AlphaBetaPlayer`, run inside
Catanatron's reference engine via `EVAL/bridge/`. The required version is **catanatron
3.3.0**, a dev build that is **not on PyPI** (PyPI tops out at 3.2.1, a stripped
core with no `AlphaBetaPlayer`, no `ValueFunctionPlayer`, no domestic trades).

- **Pinned commit: `41ba0db`** ("deterministic discards", #362). This is the
  commit the local clone `/home/sinan/Desktop/msc/catanatron` sits on and the
  one the bridge is verified against — **281/281 `EVAL/bridge/tests` pass** (2026-05-27,
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
  depend on it (`EVAL/bridge/PLAN.md`). The tournament warns if it is unset.

## 8. Commands (anaconda env)

```bash
AP=/home/sinan/anaconda3/bin/python

# Train the vs-random seed on the 1084/286 interface (768 envs = the winning config).
$AP -m models.train_ppo --num-envs 768 --total-steps 50_000_000 --run-name ppo_1084_50m
# (>=30M for gate margin; 50M is the verified M2/M3 seed. See training notes below.)

# Thesis gate: final model vs Alpha-Beta, >=1000 games, win rate + 95% CI.
PYTHONHASHSEED=0 PYTHONPATH=EVAL $AP -m AB.tournament \
    --ckpt models/checkpoints/ppo_1084_50m/ppo_final.zip \
    --games 1000 --opponent alphabeta --ab-depth 2 --ab-prune --seed 42 --no-trades
# GATE: 95% Wilson CI lower bound > 0.25.  Result JSON -> EVAL/AB/results/.
# Cost: AlphaBeta ~6.4 s/game unpruned (~1.8 h / 1000 games); --ab-prune cuts it.

# 10^8-step stability soak (no catanatron needed).
PYTHONPATH=EVAL $AP -m AB.soak --steps 100000000 --seed 7
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

Trained **with** the per-turn trade-compose stall cap (then
`MAX_TRADE_COMPOSE_PER_TURN = 20` in `models/env.py`; since moved into the C++
core, `include/state.hpp`, value 50, applied to every seat); opponents during
training are 3 uniform-random-legal seats.

- **Seed model:** `ppo_1084_50m`, trained on the **1084/286** interface in the
  anaconda env (768 envs, 50M steps, checkpoints every 5M). This is the verified
  M2 gate-passer and M3 self-play warm-start seed; the tournament evaluates it
  (or a later self-play checkpoint).
- **Convergence:** 10M steps ≈ 88% vs random (fails the M2 CI bar); converges
  ~15–20M; ≥30M for margin. `ppo_1084_50m` clears the M2 gate — **95.5%** vs
  random native (200g, sampling, CI-low 0.917, `models.eval`) and **89.5%** via
  the bridge vs `RandomPlayer` (200g, `--no-trades`, CI [0.845, 0.930]).

> The "final model" is a `--ckpt` flag — swap in any future (e.g. M3 self-play)
> 1084/286 checkpoint without code changes.

## 10. Known caveats

- **Interface match is mandatory.** Before any eval/train, check
  `fastcatan.OBS_SIZE` vs the checkpoint's `observation_space.shape[0]`.
  `EVAL/AB/policy.py` raises on mismatch — a checkpoint trained against a different
  obs/action dim than the live build will not load.
- AlphaBetaPlayer has a **20s/move** deadline. At depth 2 with `--ab-prune` a
  1000-game run is tractable; unpruned is ~6.4 s/game. The JSON records
  `--ab-depth`/`--ab-prune` with every result.
- Eval uses **sampling** policy by default. The C++ trade-compose cap
  (`MAX_TRADE_COMPOSE_PER_TURN`, `include/state.hpp`) prevents the within-turn
  trade-loop stall on the native path even under argmax (`--deterministic`); the
  bridge keeps its own `_COMPOSE_LOOP_CAP` per-call guard. Sampling is still the
  validated mode.
- Catanatron's longest-road is internally inconsistent on road cuts; fastcatan is
  rule-correct. The ≤1–2% differential residual is documented and exempted in
  `EVAL/bridge/tests/test_differential.py` — it does not affect tournament outcomes.
