# Architecture & file guide

A walking tour of every file in this repo, grouped by concern. New
contributors should read this top-to-bottom; everyone else can skim to
the section they need.

## Bird's-eye view

```
┌─────────────────────────────────────────────────────────────┐
│                       Python layer                          │
│  fastcatan.{Env, BatchedEnv}        nanobind bindings       │
│  fastcatan.GymEnv                   single-agent Gymnasium  │
│  fastcatan.CatanAECEnv              multi-agent PettingZoo  │
│  fastcatan.{play, AlphaBetaPlayer}  evaluation harness      │
│  fastcatan.{policy_from_sb3, ...}   self-play wrappers      │
└──────────────┬──────────────────────────────────────────────┘
               │ nanobind / numpy ndarray (zero-copy)
┌──────────────▼──────────────────────────────────────────────┐
│                       C++ core                              │
│   GameState (384 B) ──▶ step_one  ──▶ recompute / surgical  │
│                          │                action_mask       │
│   reset_one ─▶ BoardLayout │                                │
│                          │                                  │
│   BatchedEnv  ──▶ N envs in contiguous memory + auto-reset  │
└─────────────────────────────────────────────────────────────┘
```

The C++ core is single-threaded per env and stateless above the API
boundary. Python drives it through nanobind, exchanging numpy arrays
zero-copy (no pickling, no per-step copies).

---

## Directory layout

```
fastCatan/
├── README.md              project intro + quickstart
├── PLAN.md                thesis plan + milestone tracking
├── HPC.md                 HPC build / SLURM setup
├── ARCHITECTURE.md        ← you are here
├── LICENSE                MIT
├── CMakeLists.txt         build system (HPC-ready)
├── pyproject.toml         scikit-build-core editable install
├── .gitignore             ignore build outputs and caches
├── .clangd                clangd config for IDEs
│
├── include/               public C++ headers
├── src/catan/             core C++ implementations
├── bindings/pycatan/      nanobind module bridging C++ ↔ Python
├── python/fastcatan/      pure-Python package (wrappers, agents)
├── bench/                 standalone C++ throughput benchmarks
└── tools/                 build scripts, tests, profilers
```

---

## C++ core — `include/` and `src/catan/`

The "engine." All game rules, RNG, and batched stepping. No Python or
external dependencies — links cleanly with GCC 14.2 / clang 17 / etc.

### `include/state.hpp`
Defines two POD structs:

- **`GameState`** (384 bytes, 6 cache lines, alignas(64)) — everything
  that mutates during a game: nodes, edges, per-player resources, dev
  cards, awards, sub-phase flags, RNG state, and the cached
  `action_mask[5]`. Trivially copyable (`memcpy` clones a state, which
  is what `Env.snapshot()` and the alpha-beta search rely on).
- **`BoardLayout`** (48 bytes) — static-per-episode board state: hex
  resources, hex numbers, port types, port-pattern selector. Set once
  during `reset_one`, read often during `step_one`.

Also defines:
- `Phase` and `Flag` enums (game phase, sub-phase override).
- Small helpers: `node_pack/level/owner` for the bit-packed
  `node[]` encoding.
- Constants: `NO_PLAYER = 0xFF`, `NODE_EMPTY/SETTLEMENT/CITY`.

### `include/topology.hpp`
Compile-time-constant adjacency tables for the standard 19-hex Catan
board: `hex_to_node`, `node_to_node`, `edge_to_node`, etc., plus port
node placements (`port_to_node_A`, `port_to_node_B`). 11 tables in all,
all `inline constexpr`. The board is fixed; only the resource/number
randomization changes per episode.

### `include/rng.hpp`
xoshiro128++ PRNG (16 B state, BigCrush-clean) plus a SplitMix64 helper
to derive per-env seeds from a master seed. xoshiro is the default for
all stochasticity (dice, dev-card draws, robber-steal target).

### `include/rules.hpp`
Public C++ API:

```cpp
void reset_one(GameState&, BoardLayout&, uint64_t seed);
void reset_with_layout(GameState&, const BoardLayout&, uint64_t seed,
                       uint8_t start_player_override = 0xFF);
void step_one(GameState&, const BoardLayout&, uint32_t action,
              float& reward, uint8_t& done);
void recompute_awards(GameState&);
void refresh_mask(GameState&, const BoardLayout&);
```

Plus the action-ID layout in `namespace catan::action`: 296 flat IDs
for builds, dice, sub-phase actions, trades, dev-card plays, and the
PvP-trade compose protocol.

### `include/mask.hpp`
Declares `compute_mask(state, board, mask_out)` — full recompute for
debugging / cross-checking. The "live" mask is maintained inside
`GameState::action_mask` (see incremental updates in `rules.cpp`).
Constants `MASK_WORDS = 5` and `NUM_ACTIONS = 296`.

### `include/obs.hpp`
Declares `write_obs(state, board, pov, out)` — encodes a 724-element
float32 observation from a chosen player's perspective (POV-flipped
seat indexing, so the agent always sees its own slot at index 0).
Constant `OBS_SIZE = 724`.

### `include/batched_env.hpp`
Declares `BatchedEnv` — N envs in one contiguous buffer for hot-path
RL. Provides `init / destroy / reset / step / write_obs / write_masks`,
each parallelizable via OpenMP (auto-detected at CMake time). Exposes
`last_winner[]` so wrappers can read who won the just-completed game
*before* the auto-reset wipes the state.

### `src/catan/rules.cpp`
The bulk of the engine — ~1600 lines implementing every rule:

- **Initial placement**: snake order, distance rule, second-settlement
  payout, port grants.
- **Production payout**: hex-by-hex sweep on dice rolls, with the
  bank-shortage rule (single recipient gets `min(demand, bank)`;
  multiple recipients only paid if bank covers everyone).
- **Building**: settlement / city / road including connectivity rules
  (own road meets, no-cross-opponent), cost deductions, port grants.
- **Robber sub-phases**: forced discards (half-rounded-down, snake
  order), robber move with valid hex check, victim selection (auto if
  one candidate, manual if multiple).
- **Trades**: bank/port (best-ratio resolution: 2:1 if specific port,
  3:1 if generic, 4:1 default), and PvP trade as a 3-phase compose →
  respond → confirm protocol.
- **Dev cards**: weighted random draw from the deck, one-turn
  cooldown, knight-played-this-turn check, special handling for
  Year-of-Plenty, Monopoly, Road Building, and the hidden VP card.
- **Award logic**: `check_largest_army` and `check_longest_road` with
  strict-exceed transfer (incumbent keeps title on ties), cut-below-
  threshold loss for longest road.
- **Win check**: `check_game_ended` triggered after any VP change.
- **Reward signal**: `+1` to the actor on the action that hits 10 VP;
  `-1` if their action somehow triggered another player's win.
- **Incremental mask**: every successful action calls
  `refresh_action_mask` (full recompute) or
  `refresh_compose_mask_bits` (surgical update for trade compose, the
  highest-frequency action class). Debug builds assert
  `surgical == full_recompute` after every step to catch drift.

The longest-road algorithm is at the end of the file: a per-player DFS
with mark-on-enter / clear-on-exit backtracking, respecting opponent
settlement blocks. ~12 hand-built test positions guard it
(`test_longest_road.py`).

### `src/catan/obs.cpp`
The observation encoder. Walks every per-player block, then the board
features (nodes/edges/hexes/ports/robber), then game state and trade
scratch. ~150 lines.

### `src/catan/batched_env.cpp`
Implements the `BatchedEnv` declarations. Uses `std::aligned_alloc`
for cache-aligned arrays, OpenMP parallel-for around the per-env step
loop (gated on `FCATAN_HAVE_OPENMP`), and captures `last_winner`
before the auto-reset on `done`.

---

## Standalone benchmarks — `bench/`

Pure C++ binaries built by CMake; useful for measuring the engine
without Python overhead.

### `bench/bench_step.cpp`
Single-env throughput. Random-legal-action loop driven by a separate
xoshiro picker. Reports steps/sec, ns/step, games/sec.

### `bench/bench_batched.cpp`
Batched throughput. Same loop but over N envs at once. The fairest
measure of the C++ core's ceiling on a given machine. With OpenMP on
HPC this scales near-linearly with cores.

---

## Python bindings — `bindings/pycatan/`

### `bindings/pycatan/bindings.cpp`
The nanobind module (`_fastcatan.so`). Exposes:

- **`Env`** — single-env handle. Useful for tests, debugging, and the
  alpha-beta scratch env that holds search state.
- **`BatchedEnv`** — hot-path N-env handle.
- **`action`** — submodule with every action-ID constant.
- Module-level shape constants (`OBS_SIZE`, `NUM_ACTIONS`, etc).

Key design choice: every method that takes a numpy array uses
`nb::ndarray<...>` so the buffer is passed through to C++ zero-copy
(no Python-side allocation, no per-call memcpy). Long-running calls
release the GIL via `nb::gil_scoped_release`.

State serialization (`Env.snapshot()` / `Env.load_snapshot()`,
`BatchedEnv.snapshot(idx)`) is exposed for search algorithms (alpha-
beta, MCTS) that need to branch state without committing.

---

## Python package — `python/fastcatan/`

The user-facing API. Importing `fastcatan` re-exports everything from
`_fastcatan` plus the wrappers below.

### `python/fastcatan/__init__.py`
Single import surface. Soft-imports for optional dependencies:
- `gym_env` (Gymnasium) — only present if `gymnasium` installed
- `pettingzoo_env` — only if `pettingzoo` installed
- `tournament` / `alphabeta` / `selfplay` — always available (just numpy)

### `python/fastcatan/gym_env.py`
**`GymEnv`** — single-agent Gymnasium wrapper. Wraps a
`BatchedEnv(num_envs=1)` and an `opponent_fn` callback that drives the
3 non-learner seats. Compatible with sb3-contrib MaskablePPO out of
the box: `info["action_mask"]` is a `bool[NUM_ACTIONS]` array.
`info["action_mask_packed"]` keeps the raw `uint64[5]` for code that
prefers bitmask form.

Also exposes `random_legal_policy(rng)` and `lowest_legal_policy` —
small built-in opponents — and `unpack_mask(packed)` for converting
between mask formats.

### `python/fastcatan/pettingzoo_env.py`
**`CatanAECEnv`** — PettingZoo Agent-Environment-Cycle wrapper. Each
of the 4 seats is its own agent. Used for trading-net training where
different policies handle different sub-decisions (build vs trade).

Per-agent observations are POV-flipped (each agent sees its own slot
at obs index 0). On terminal step, rewards are broadcast: winner
`+1`, all losers `-1`. Cumulative rewards retrieved via `last()` per
PettingZoo convention.

### `python/fastcatan/tournament.py`
**`play(agent_a, agent_b, n_games, ...)`** — tournament harness.
Runs N games batched through `BatchedEnv`, with per-game seat-plan
control (default: A in seats 0+2, B in seats 1+3), Wilson 95%
confidence intervals on win rates, and truncation guards.

Defines the canonical `Policy` signature used by all evaluation
agents:

```python
policy(obs, mask_packed, env_idx, seat, env) -> action_id
```

The trailing `env` parameter is the live `BatchedEnv` — search-based
agents (alpha-beta, MCTS) use it to snapshot state and explore
branches.

Built-in baselines: `random_legal_policy_for_eval(rng)` and
`lowest_legal_policy_for_eval()`.

### `python/fastcatan/alphabeta.py`
**`AlphaBetaPlayer`** — depth-limited minimax with alpha-beta pruning,
inspired by Catanatron's AlphaBetaPlayer. Implements the `Policy`
signature. Snapshots the live env into a scratch `Env`, runs search
to the configured depth, returns the best legal action.

Heuristic: VP-weighted multi-feature score (VP, public VP, pieces on
board, knights played, road length, ports, longest-road / largest-
army titles). Random tiebreaker among equal-scored actions to avoid
systematic low-id bias.

Pruning: trade-compose actions (`TRADE_ADD_GIVE/_REMOVE_*`) are
filtered out by default — they explode the branching factor without
changing game-state value. Set `prune_compose=False` to keep them.
`action_limit=k` adds top-k 1-step pruning at each node when the
legal set is huge.

Performance:
- Depth 1: ~55% win rate vs random (par).
- Depth 2 with `action_limit=12`: ~75% win rate vs random.

### `python/fastcatan/selfplay.py`
Self-play adapters for SB3-style agents:

- **`policy_from_sb3(model, deterministic)`** — wraps a MaskablePPO
  model in the tournament `Policy` signature for evaluation (current
  champion vs older snapshot, etc).
- **`FrozenSelfPlayOpponent(model, deterministic)`** — adapts an SB3
  model to `GymEnv(opponent_fn=...)`, so the agent's training
  opponent is a frozen snapshot of itself (or any other model).

Used to bootstrap iterative self-play training without writing an
RL loop from scratch.

---

## Build infrastructure

### `CMakeLists.txt`
The canonical build. Targets:

- **`fastcatan_core`** (static lib) — compiles `rules.cpp`,
  `obs.cpp`, `batched_env.cpp`. Used by every other target.
- **`fastcatan`** (shared lib) — `tools/c_api.cpp` for the ctypes
  shim. Used by the Python test scripts that pre-date nanobind.
- **`bench_step` / `bench_batched`** — standalone benchmarks.
- **`_fastcatan`** (Python extension, gated on `SKBUILD`) — the
  nanobind module, built only when invoked through scikit-build-core.

Release flags: `-O3 -march=native -fno-exceptions -fno-rtti` + LTO via
`CMAKE_INTERPROCEDURAL_OPTIMIZATION`. The nanobind target gets fresh
flags (it needs RTTI + exceptions). OpenMP detected via
`find_package(OpenMP)` and linked into `fastcatan_core` when
available.

### `pyproject.toml`
scikit-build-core editable-install config. `pip install -e .` builds
the C++ extension and installs `fastcatan` into the venv. Wheel
includes the entire `python/fastcatan/` package alongside the compiled
`_fastcatan.so`.

### `tools/build_lib.sh`
Shell script that compiles `tools/c_api.cpp` directly with clang++
into `build/libfastcatan.{dylib,so}` for the ctypes-based test scripts.
Faster iteration than rebuilding through CMake when only the ctypes
shim changes.

### `tools/build_bench.sh`
Same idea for the standalone benchmarks.

### `tools/c_api.cpp`
The ctypes shim used by the older test scripts (`test_step1.py`
through `test_perft.py`). Exposes a flat C ABI around `GameState`,
`BatchedEnv`, and a few mutators (`fcatan_set_node`,
`fcatan_set_edge`, `fcatan_give_resources`, etc.) that tests use to
poke state directly.

The shim and the nanobind module are independent paths to the same
core; both link against `fastcatan_core`. The nanobind module is
preferred for production use (faster, type-safe, zero-copy) — the
ctypes shim exists because the early tests were written against it.

---

## Tools — `tools/`

### `tools/profile_train.py`
The profiler. Times every layer of the training pipeline so you can
see exactly where wall time goes. Sections:

- `engine` — raw env ops (step / mask / obs / reset)
- `gym` — single-env Gym wrapper overhead
- `ppo` — full SB3 MaskablePPO learn() loop
- `cprofile` — function-level breakdown via cProfile

Use this on HPC to decide whether to invest in a custom BatchedEnv-
driven PPO loop or stick with SB3.

### `tools/train_smoke.py`
Minimal MaskablePPO trainer: random opponents, MlpPolicy, a few
hundred timesteps. Verifies the full RL stack end-to-end. Good first
thing to run after a fresh install.

### `tools/viz_topology.py`
Renders the standard Catan board with all node IDs, edge IDs, hex
IDs, and both port patterns visualized. Useful for debugging
coordinate questions ("is edge 47 the same as catanatron's frozenset
{31, 32}?") — though the coordinate-translation tooling that needs it
is not currently checked in.

### `tools/perft_hashes.json`
Pinned trajectory hashes generated by `tools/test_perft.py --pin`.
Each entry: `(seed, n_steps) → final-state FNV-1a hash`. CI compares
against these on every change; any code change that affects the
trajectory shows up as a hash mismatch.

### `tools/test_*.py` — the test suite
19 suites, ~120 test cases total. All pass on the current code.

| File | What it covers |
|---|---|
| `test_step1.py` | initial-placement rules (slice 1 of `step_one`) |
| `test_step2.py` | dice rolls + production payout (slice 2) |
| `test_step3.py` | building in MAIN phase (slice 3) |
| `test_step4.py` | robber sub-phases: discard, move, steal (slice 4) |
| `test_step5.py` | bank/port trades (slice 5) |
| `test_step6.py` | dev cards: buy + play knight + largest army (slice 6) |
| `test_trade.py` | PvP trade 3-phase protocol |
| `test_longest_road.py` | hand-built corpus for the LR algorithm (12 cases) |
| `test_mask.py` | `compute_mask` consistency vs simulation |
| `test_obs.py` | obs encoder shape, determinism, POV correctness |
| `test_batched.py` | BatchedEnv lifecycle + auto-reset |
| `test_perft.py` | pinned trajectory-hash regression test |
| `test_nanobind.py` | nanobind module sanity (constants, zero-copy buffers) |
| `test_gym.py` | `GymEnv` wrapper |
| `test_pettingzoo.py` | `CatanAECEnv` AEC wrapper |
| `test_tournament.py` | `play()` harness + Wilson CI |
| `test_alphabeta.py` | AB player + snapshot round-trip + win rate vs random |
| `test_selfplay.py` | SB3 → tournament-Policy adapter |

Run all in one shot:

```bash
for t in tools/test_*.py; do
    echo "=== $t ===";
    python3 "$t" 2>&1 | tail -1;
done
```

Expected output: every line ends with `ALL TESTS PASS` or
`ALL PERFT HASHES MATCH`.

---

## Documentation

### `README.md`
Top-level intro: what fastCatan is, throughput numbers, quickstart
install, the core code examples (`Env`, `BatchedEnv`, `GymEnv`,
`CatanAECEnv`, MaskablePPO training), key concepts (action space, mask,
obs, reward, RNG), and project status.

### `PLAN.md`
The thesis-side roadmap. Five milestones (M1–M5) with dated
deliverables, throughput targets, and risk register. Updated as
work progresses.

### `HPC.md`
Linux/HPC build guide: module loads (GCC 14.2 + cmake 3.27 + python
3.12 + CUDA 12.4), venv creation, `pip install`, the standalone
bench targets, the RL training stack (`torch sb3-contrib stable-
baselines3`), a SLURM job template, common pitfalls + fixes, and
performance expectations per scale.

### `ARCHITECTURE.md`
This file.

### `LICENSE`
MIT.

### `.clangd`
Tells clangd which compile flags to use for the headers when you
open them in an editor. Avoids spurious "include not found" diagnostics.

---

## Reading order for a new contributor

1. **`README.md`** — what the project does, hello-world examples.
2. **`include/state.hpp` + `include/rules.hpp`** — the data model and
   public C++ API. Two short headers.
3. **`src/catan/rules.cpp`** — the engine. Long but well-sectioned by
   "slice" comments matching the test files.
4. **`bindings/pycatan/bindings.cpp`** — how C++ types reach Python.
5. **`python/fastcatan/__init__.py`** — what the Python package
   actually exposes.
6. **`python/fastcatan/tournament.py`** — the canonical Policy
   signature. Other agents (`alphabeta.py`, `selfplay.py`) build on it.
7. **`tools/test_step1.py`** — example of how the engine is exercised
   from Python (via the ctypes shim — older but instructive).
8. **`tools/train_smoke.py`** — the full training loop, end-to-end.
9. **`HPC.md`** — what changes when you move from Mac/dev to HPC.
10. **`PLAN.md`** — where the project is heading.

Once you've read those, the rest of the repo should fit into context.
