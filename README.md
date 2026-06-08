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
>
> **Status (2026-06-07):** the HYBRID search agent (imitation-learned prior
> + native `ab_value` leaves + ≥512-sim stochastic search +
> catanatron-faithful in-tree opponent model) **passed the official
> catanatron-bridge gate at 200 games: 65/200 = 32.5% [26.4–39.3] vs
> Alpha-Beta-d2** (CI-low > 25% = 4-player parity — the opponent that
> scored 0/200 against every reactive policy). Native ladders: d1 29.0%
> [25.5–32.8], d2 29.5% [23.6–36.2]. Because the hybrid calls
> catanatron-derived components (`ab_value`/`ab_decide`) at inference, the
> M4 gate is now **two-tier** (2026-06-07): a fast native AB-d2 dev ladder
> per iteration, and ONE ≥1000-game bridge run at the very end on a
> **self-contained** model (learned leaf value + learned opponent model —
> in progress). See [The Alpha-Beta campaign](#the-alpha-beta-campaign)
> and `EVAL/AB/README.md`.

## Throughput

Measured with `DEBUG/bench/bench_throughput.py` (full per-component breakdown in
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
bash scripts/setup.sh           # conda env `catan` (CPython 3.12) + all pinned deps + native build + verify
conda activate catan
# Linux CUDA box:  FASTCATAN_CUDA=cu124 bash scripts/setup.sh
```

This is the **canonical, identical-on-every-device** path: conda pins the interpreter
(`environment.yml`: python=3.12) so checkpoints + the native `cp312` build stay
compatible across macOS and Linux. **Do not** use a bare `python3 -m venv` — it
inherits whatever `python3` the box has (that is how a Mac drifts to 3.13/numpy-2.x
and breaks checkpoint pickles). Verify any device at any time:

```bash
python scripts/check_env.py     # asserts py3.12 + numpy 1.26.4 + fastcatan/catanatron import; non-zero on drift
```

> `editable.rebuild=true` makes scikit-build-core recompile the extension on the
> next `import fastcatan` after any C++ change. Without it, `pip install -e .`
> builds once and the binary silently goes stale when `obs.hpp`/`mask.hpp` change
> (see `EVAL/AB/REPRODUCIBILITY.md` §5).

Verify:

```bash
python3 -c "import fastcatan as fc; print(fc.OBS_SIZE, fc.NUM_ACTIONS)"
# 1084 286
```

Run the correctness gates (after `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`):

```bash
ctest --test-dir build -R invariants            # 100k-game invariant fuzz smoke (~2s)
build/fuzz_invariants 10000000                  # full 10⁷-game gate (~3 min, OpenMP)
python3 -m pytest tests/ EVAL/bridge/tests/ -q   # unit + cross-engine differential
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

# Native expectimax Alpha-Beta (a faithful Catanatron AlphaBetaPlayer port,
# in C++): pick the current player's best move, or score a seat's position.
best = env.ab_decide(env.current_player, depth=2, prune=False)
val  = env.ab_value(env.current_player)
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

`BatchedEnv` also exposes **tree-search primitives** (all OpenMP, GIL-free)
so a batched MCTS can drive the N slots as parallel scratch branches:
`save_snapshots`/`load_snapshots` ((N, `SNAPSHOT_BYTES`) buffers),
`reseed(seeds)` (resample chance per simulation), `step_raw` (no auto-reset;
`SKIP_ACTION` parks an env), `write_obs_pov_batch`/`write_obs_all4`,
`write_sigs` (chance signatures), and `ab_decide_batch` (native AB pick for
every env's current player — the batched opponent for training/searching vs
AB). On the single `Env`: `recompute_mask()` rebuilds the cached legal mask
after loading an *injected* state (bridge `state_inject` fills the state
fields but not the mask), and `ab_decide(..., banned_mask)` excludes an
action set at **every** node of the native search (e.g. p2p trades under
`--no-trades`) so its pick never needs a random fallback.

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

### Training (reactive baselines)

The PPO/A2C/DQN trainers (`models/train_ppo.py` etc., docs in
[`models/PLAN.md`](models/PLAN.md)) remain as the reactive-policy baselines —
historically the M2/M3 track. Note the campaign finding below: reactive
policies cap out near 0 vs Alpha-Beta regardless of steps/capacity; the live
track is the search stack in `models/alphazero/`.

## The Alpha-Beta campaign

The thesis question — *can a learned agent beat Alpha-Beta?* — was answered
by a sequence of controlled experiments (each gated by 200-game ladders vs
the fixed-hole native AB). Full forensics live in the git history; the
distilled findings:

### What failed (the falsification ledger)

Every result below is measured, not assumed. Don't re-try these:

| Lever | Verdict |
|---|---|
| More PPO steps (200M league self-play) | 0/200 vs AB — gains orthogonal |
| More capacity alone (512→2048 nets) | vs-random ↑, vs-AB flat |
| Reward shaping (VP potential) | policy-invariant, vs-AB ≈ 0 |
| Search on a **sparse-value** net (any net, any sims 64–512) | flat 2–7% — ±1 leaf noise can't resolve AB-scale (1–3% win-prob) move differences |
| Direct/anchored RL fine-tuning vs AB | value dilution (homogeneous loss margins flatten the value head); anchor stops the bleed but never climbs |
| Naive DAgger past round 1 | plateaus at the linear-error floor (~6–7.5% raw) |

### What worked (the design rules)

1. **Distribution beats optimization.** A supervised clone of AB-d1 (40k
   AB-vs-AB games — 42 s of generation, 78 s of training) reaches **0.975
   vs random**; PPO needed 30–50M steps for the same. The vs-random
   bottleneck was always the self-play distribution, never the optimizer.
2. **Value heads that feed search need DENSE targets** (`vp_margin`, not
   ±1). Single-variable test: same data, same net, value target ±1→margin
   flipped search from no-effect to 2× (7.5%→15.0% vs AB-d1 at 256 sims).
3. **The winning architecture is neuro-symbolic**: learned prior proposes
   (tames ~50-way branching), the native `ab_value` judges leaves
   (deterministic, zero variance — needs a *two-scale* tanh squash because
   catanatron's value is lexicographic: VP weight 3e14 over fine features
   ~1e7), and ≥512 stochastic sims out-search AB's fixed 1–2 ply. The
   hybrid is prior-invariant (40k/80k/160k clones all ≈23% at 256 sims) —
   **sims is the scaling axis** (23.0% → 28.25% → 30.5% at 256/512/1024 vs d1).
4. **Trainer laws** (from controlled pairs): replay ratio ≤ 2; lr scales
   *down* with net size (1024-net at the 512-net's lr = noisy plateau);
   track BOTH raw-policy and search-eval per checkpoint — their divergence
   is the diagnostic (raw↑/search↓ = value dilution; loss curves see nothing).

### The pipeline (all CPU-cheap; d2 data costs the same as d1)

```bash
# 1. Teacher dataset: AB-vs-AB games at ~1000 games/s (0 fallbacks = the
#    banned-mask ab_decide has no random hole to clone)
python -m models.alphazero.il_dataset --games 40000 --workers 8 \
    --ab-depth 1 --out-dir models/datasets/il_ab_d1

# 2. Supervised pretrain (masked-CE on teacher action + DENSE value)
python -m models.alphazero.il_pretrain --data-dir models/datasets/il_ab_d1 \
    --hidden 1024,1024,512 --value-target vp_margin --device cuda

# 3. The ladder (the campaign's unit of evidence — 200 games, report the CI)
python -m models.alphazero.evaluate --ckpt <ckpt> --opponent alphabeta \
    --ab-depth 2 --sims 512 --games 200 --leaf-eval ab_value \
    --ab-value-scale 86000000

# 4. The FINAL gate (catanatron engine; run ONCE at end of project, on the
#    self-contained model only — two-tier gate, 2026-06-07)
PYTHONPATH=.:EVAL python -m AB.tournament --policy mcts --ckpt <ckpt> \
    --mcts-sims 512 --opponent alphabeta --ab-depth 2 --ab-prune --no-trades
```

GPU-batched training infrastructure (used for the self-play track;
`models/alphazero/batched_mcts.py` + `batched_selfplay.py`): G game trees
searched in lockstep, one GPU forward per simulation over all pending
leaves — 9–10×/decision over per-game MCTS, flat G=64→1024.

### Open problem

The native→bridge transfer gap is **RESOLVED** (2026-06-06): faithful
in-tree chance model + policy-owned robber composite + the decisive fix —
catanatron **shuffles seating** in `State.__init__`, so the search had been
optimizing an opponent's position in ~75% of games (pinning bridge runs at
0.25×native ≈ 6%). Result: bridge v6 = **32.5% [26.4–39.3] vs AB-d2**, the
first GATE PASS.

The live open problem is **self-containment**: the thesis agent may not
call `ab_value`/`ab_decide` at inference. In order: (1) distill the
two-scale `ab_value` squash into the value head (`il_pretrain
--value-target ab_value` → search with `--leaf-eval net`), (2) replace the
in-tree opponent model with the net itself, (3) re-train the prior with
the learned value — each step gated on the native AB-d2 dev ladder
(`EVAL/AB/README.md`, "The gate, restructured").

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
268..272  TRADE_ADD_GIVE  by resource (p2p compose)
273..277  TRADE_ADD_WANT  by resource (p2p compose)
278       TRADE_OPEN
279       TRADE_ACCEPT
280       TRADE_DECLINE
281..284  TRADE_CONFIRM  by partner seat
285       TRADE_CANCEL
```

All action IDs are exposed as `fastcatan.action.<NAME>`.

### Action mask

Mask is a `uint64[5]` (320 bits, 286 used). Bit `i` set ⇔ action `i` is
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
and bit flags stay 0/1. The bridge eval encoder (`EVAL/bridge/obs_encoder.py`)
mirrors these divisors exactly — `EVAL/bridge/tests/test_obs_identity.py` guards it.

### Reward

`+1` to the actor on the action that pushes their VP to 10. `-1` to the
actor if their action somehow triggered another player's win (rare).
Everything else is `0`. Loser perspective is reconstructed in the AEC
wrapper / training loop.

### RNG

xoshiro128++ per env, 16 bytes of state, seeded via SplitMix64 from a
master seed. Deterministic given the seed; fixed-seed reproducibility is
guarded by `tests/test_determinism.py`.

## Repo layout

```
include/                 public headers (state, rules, mask, obs, batched_env, rng, topology, search)
src/catan/               core C++ implementation (rules, obs, batched_env, search=native Alpha-Beta)
bindings/pycatan/        nanobind module
python/fastcatan/        Python package (re-exports + Gym/PettingZoo wrappers)
tests/                   Python correctness tests (invariants, scenarios, determinism, mask, alphabeta)
tests/fuzz_invariants.cpp  10⁷-game C++ invariant fuzz gate (ctest -R invariants; see PLAN.md M1)
EVAL/bridge/                  Catanatron interop + cross-engine differential (see EVAL/bridge/PLAN.md)
EVAL/AB/                      M4 Alpha-Beta eval + native-AB fidelity gate (see EVAL/AB/README.md)
EVAL/AB/mcts_policy.py        state-aware hybrid-search bridge policy (--policy mcts)
models/                  RL trainers (PPO + A2C/DQN/MuZero) + Gym env (see models/PLAN.md)
models/alphazero/        the AB campaign: mcts + batched_mcts (lockstep GPU search),
                         batched_selfplay (single-process trainer, anchor-mixing),
                         il_dataset (AB-vs-AB generator + DAgger mode), il_pretrain
                         (masked-CE + dense-value), mcts_vs_fixed (--leaf-eval ab_value
                         hybrid), evaluate (the 200-game ladder)
models/datasets/         IL shards + memmap caches (gitignored)
DEBUG/ui/                      obs decoder / board render / replay (see DEBUG/ui/PLAN.md)
examples/                random + alpha-beta player references
DEBUG/bench/                   throughput benchmarks (bench_throughput.py + C++ bench_step/bench_batched)
CMakeLists.txt           build system
pyproject.toml           scikit-build-core editable install
PLAN.md                  thesis plan + milestone tracking
```