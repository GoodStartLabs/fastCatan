# fastCatan

High-throughput simulator for 4-player Settlers of Catan, built for
reinforcement-learning research. C++23 core, nanobind bindings, batched
stepping with optional OpenMP, Gymnasium + PettingZoo wrappers.

> **Project context.** Companion simulator for an RL thesis on Catan.
> Goal: train an RL agent that beats classical Alpha-Beta with
> statistical significance over ≥1000 four-player games. The bottleneck
> the thesis explicitly calls out is simulator throughput — existing
> Python sims (Catanatron) can't generate self-play at the volume modern
> RL needs.

## Throughput

Measured with `bench/bench_throughput.py` (full per-component breakdown in
PLAN.md M1). Single node:

| Path | Steps/sec |
|---|---|
| Pure C++ batched `step_one` (1 thread) | ~10–50M (cache-bound; smaller batch faster) |
| Python batched hot path (mask + policy + obs + step) | ~1M |
| OpenMP across cores | near-linear (Linux/GCC; the macOS/clang build links no OpenMP) |

Far above the M1 target (5×10⁵) and ~7× Catanatron on equal footing
(games/s, random-vs-random). **The C++ `step_one` is not the bottleneck** — in
the batched hot path `write_obs` (the 1084-float encode) dominates; in the
single-env path the Python legal-action scan + interpreter glue dominate. So
the optimization target, if ever needed, is the obs, not the simulator (for a
GPU loop the cost shifts further to obs encode + CPU→GPU transfer).

## Quickstart

### Local (macOS / Linux)

```bash
git clone <repo> fastcatan
cd fastcatan
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install cmake ninja nanobind scikit-build-core    # build deps
pip install -e . --no-build-isolation --config-settings=editable.rebuild=true  # build + auto-rebuild on source edits
pip install torch sb3-contrib stable-baselines3         # for training
pip install -r requirements.txt                         # catanatron (pinned) for the bridge/eval path
```

> `editable.rebuild=true` makes scikit-build-core recompile the extension on the
> next `import fastcatan` after any C++ change. Without it, `pip install -e .`
> builds once and the binary silently goes stale when `obs.hpp`/`mask.hpp` change
> (this caused a 724/296-vs-1084/286 drift; see `AB/REPRODUCIBILITY.md` §4–5).

Verify:

```bash
python3 -c "import fastcatan as fc; print(fc.OBS_SIZE, fc.NUM_ACTIONS)"
# 1084 286
```

Run the correctness gates (after `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`):

```bash
ctest --test-dir build -R invariants            # 100k-game invariant fuzz smoke (~2s)
build/fuzz_invariants 10000000                  # full 10⁷-game gate (~3 min, OpenMP)
python3 -m pytest sim/tests/ bridge/tests/ -q   # unit + cross-engine differential
```

M1 gate result: 0 invariant violations over 10⁷ games / 4.04×10¹⁰ steps (see PLAN.md §M1).

## Hello world

### Single-env, raw API

```python
import fastcatan as fc
import numpy as np

env = fc.Env()
env.reset(seed=42)
print(f"phase={env.phase}, current_player={env.current_player}")

# Pick a legal action and step.
reward, done = env.step(fc.action.SETTLE_BASE + 5)   # build settlement at node 5
```

### Batched env (the hot path)

```python
import fastcatan as fc
import numpy as np

n = 4096
env = fc.BatchedEnv(num_envs=n, seed=42)
env.reset()

# Pre-allocate buffers; nanobind passes them through to C++ zero-copy.
actions = np.zeros(n, dtype=np.uint32)
rewards = np.zeros(n, dtype=np.float32)
dones   = np.zeros(n, dtype=np.uint8)
masks   = np.zeros((n, fc.MASK_WORDS), dtype=np.uint64)
obs     = np.zeros((n, fc.OBS_SIZE), dtype=np.float32)

env.write_masks(masks)
# ... your policy fills `actions` ...
env.step(actions, rewards, dones)
env.write_obs(obs)
```

### Gymnasium wrapper (single-agent vs random opponents)

```python
import fastcatan as fc
import numpy as np

rng = np.random.default_rng(0)
opp = fc.random_legal_policy(rng)
env = fc.GymEnv(seat=0, seed=42, opponent_fn=opp)

obs, info = env.reset(seed=42)
mask = info["action_mask"]              # bool[NUM_ACTIONS] for SB3
packed = info["action_mask_packed"]      # uint64[MASK_WORDS] raw

action = my_policy(obs, mask)
obs, reward, terminated, truncated, info = env.step(action)
```

### PettingZoo AEC (multi-agent / trading-net training)

```python
import fastcatan as fc

env = fc.CatanAECEnv(seed=42)
env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        env.step(None); continue
    action = my_policies[agent](obs["observation"], obs["action_mask"])
    env.step(action)
```

### Training with MaskablePPO

```bash
python3 -m models.train_ppo --num-envs 64 --total-steps 100_000
```

See [`models/PLAN.md`](models/PLAN.md) for the trainers (PPO + A2C/DQN/MuZero
references) and `models/env.py` for the single-agent Gym wrapper.

## Key concepts

### Action space

Flat `Discrete(NUM_ACTIONS=286)`:

```
0..53     SETTLE   build settlement at node N
54..107   CITY     build city at node N
108..179  ROAD     build road at edge E
180       ROLL_DICE
181       END_TURN
182..186  DISCARD  by resource (during DISCARD sub-phase)
187..205  MOVE_ROBBER  to hex H
206..209  STEAL    from player P
210..234  TRADE    bank/port: give*5 + get
235       BUY_DEV
236       PLAY_KNIGHT
237       PLAY_ROAD_BUILDING
238..262  PLAY_YEAR_OF_PLENTY  (give1*5 + give2)
263..267  PLAY_MONOPOLY  by resource
268..272  TRADE_ADD_GIVE_BASE
273..277  TRADE_REMOVE_GIVE_BASE
278..282  TRADE_ADD_WANT_BASE
283..287  TRADE_REMOVE_WANT_BASE
288       TRADE_OPEN
289       TRADE_ACCEPT
290       TRADE_DECLINE
291..294  TRADE_CONFIRM_BASE  by partner
295       TRADE_CANCEL
```

All action IDs are exposed as `fastcatan.action.<NAME>`.

### Action mask

Mask is a `uint64[5]` (320 bits, 296 used). Bit `i` set ⇔ action `i` is
legal. Read via `env.write_masks(buf)` (BatchedEnv) or
`info["action_mask_packed"]` (GymEnv). For SB3-style consumers, GymEnv
also unpacks to a `bool[NUM_ACTIONS]` in `info["action_mask"]`.

The mask is incrementally maintained inside `GameState` and refreshed on
every `step_one`. Sub-action types (trade compose) use a surgical updater
for ~21% throughput win.

### Observation

`obs` is a `float32[OBS_SIZE=1084]` from the current player's perspective.
Fields are POV-relative (self always at slot 0, opponents at +1, +2, +3).
Count fields (VP, hand size, resources, road length, bank, …) are **normalized**
by structural Catan maxima (see `src/catan/obs.cpp` `namespace norm`); one-hots
and bit flags stay 0/1. The bridge eval encoder (`bridge/obs_encoder.py`)
mirrors these divisors exactly — `bridge/tests/test_obs_identity.py` guards it.

### Reward

`+1` to the actor on the action that pushes their VP to 10. `-1` to the
actor if their action somehow triggered another player's win (rare).
Everything else is `0`. Loser perspective is reconstructed in the AEC
wrapper / training loop.

### RNG

xoshiro128++ per env, 16 bytes of state, seeded via SplitMix64 from a
master seed. Deterministic given the seed; fixed-seed reproducibility is
guarded by `sim/tests/test_determinism.py`.

## Repo layout

```
include/                 public headers (state, rules, mask, obs, batched_env, rng, topology)
src/catan/               core C++ implementation
bindings/pycatan/        nanobind module
python/fastcatan/        Python package (re-exports + Gym/PettingZoo wrappers)
sim/tests/               Python correctness tests (invariants, scenarios, determinism, mask)
sim/fuzz_invariants.cpp  10⁷-game C++ invariant fuzz gate (ctest -R invariants; see PLAN.md M1)
bridge/                  Catanatron interop + cross-engine differential (see bridge/PLAN.md)
models/                  RL trainers (PPO + A2C/DQN/MuZero) + Gym env (see models/PLAN.md)
ui/                      obs decoder / board render / replay (see ui/PLAN.md)
examples/                random + alpha-beta player references
bench/                   throughput benchmarks (bench_throughput.py + C++ bench_step/bench_batched)
CMakeLists.txt           build system
pyproject.toml           scikit-build-core editable install
PLAN.md                  thesis plan + milestone tracking
```