# fastCatan — High-Throughput Catan Simulator

## Context

Author is writing on RL for 4-player Settlers of Catan. Thesis goal: an RL agent that beats Alpha-Beta with statistical significance (>25% win rate over ≥1000 four-player games). The bottleneck the thesis explicitly calls out is simulator throughput — existing Python sim (Catanatron) can't generate self-play at the volume modern RL needs.

This plan designs `fastCatan`: a greenfield C++23 simulator with nanobind Python bindings, Gymnasium interface, targeting **5×10⁷ steps/sec/node** to make M3 self-play feasible, with full 4-player rules including player-to-player trading intact. Correctness is enforced by invariant fuzzing, hand-built rule unit tests, and perft-style determinism hashes.

Decisions confirmed by user: greenfield C++, Linux HPC only, full trading from M0 foundations.

## Architecture Summary

| Decision | Choice | Reason |
|---|---|---|
| Language | C++23 | GCC 14.2 on dev + HPC frontend → full `std::mdspan`, deducing-this, `std::print`, `std::stacktrace`, multidim subscript. Mature perf tooling (PGO, VTune, SIMD, OpenMP); 26-week solo clock favors known toolchain |
| Binding | nanobind | 2–10× less per-call overhead than pybind11; native DLPack; step() crosses boundary constantly |
| State layout | AoS per game, SoA across batch; ~512 B/env | Five cache lines/env; 1M envs fits RAM easily |
| Action space | Flat discrete ~300, incremental legal-action mask | Masking is the longest branch-heavy path; incremental removes it |
| Trading | Sub-phase compositional encoding | Avoids combinatorial blowup; matches thesis's separate trading net cleanly |
| RNG | xoshiro128+, per-env, 16 B state | 0.5 ns/call, BigCrush-clean, reproducible |
| Parallelism | Single-process `BatchedEnv` + OpenMP parallel-for, auto-reset; multi-process across NUMA | Amortizes OMP launch across thousands of envs; auto-reset kills the slow-env-stalls-batch bug |
| Tensor handoff | DLPack zero-copy via `nanobind::ndarray` | No memcpy per step; PyTorch views same C++ buffer |
| Gym API | Single-agent Gymnasium + perspective flip for self-play; PettingZoo AEC (done in M0) | 99% of RL libs target Gym; AEC wrapper needed only for trading-net training |
| Build | CMake 3.27+ with scikit-build-core; GoogleTest + Google Benchmark via FetchContent | One build system; `pip install -e .` works for dev |
| Platform | Linux x86_64 only; `-std=c++23 -O3 -march=native -flto -fno-exceptions -fno-rtti`, PGO in final mile. Toolchain: GCC 14.2 (Debian) confirmed on dev + HPC frontend. | HPC target; skip macOS/Windows CI complexity |

## Repository Layout

```
fastCatan/
├── CMakeLists.txt
├── pyproject.toml                     # scikit-build-core + nanobind
├── include/catan/
│   ├── topology.hpp                   # static board adjacency — codegen'd
│   ├── state.hpp                      # GameState struct
│   ├── action.hpp                     # action IDs, decoding tables
│   ├── rules.hpp                      # reset_one, step_one, write_obs
│   ├── mask.hpp                       # incremental legal-action mask
│   ├── rng.hpp                        # xoshiro128+
│   └── batched_env.hpp
├── src/catan/
│   ├── rules.cpp
│   ├── mask.cpp
│   ├── longest_road.cpp               # isolated — trickiest algorithm
│   ├── obs.cpp                        # GameState -> observation tensor
│   ├── batched_env.cpp
│   └── topology_gen.py                # build-time codegen
├── bindings/pycatan/
│   └── bindings.cpp                   # thin nanobind layer
├── python/fastcatan/
│   ├── __init__.py
│   ├── gym_env.py                     # single-agent wrapper + opponent hook
│   ├── pettingzoo_env.py              # M3
│   └── benchmarks/throughput.py
├── tests/
│   ├── cpp/{test_rules,test_mask,test_longest_road,test_determinism}.cpp
│   └── python/{test_gym_api,test_invariants}.py
└── bench/bench_step.cpp               # Google Benchmark, pure-C++ numbers
```

### Public C++ API (kept minimal)

```cpp
namespace catan {
    void reset_one(GameState& s, uint64_t seed) noexcept;
    void step_one(GameState& s, uint32_t action_id,
                  float& reward_out, uint8_t& done_out) noexcept;
    void write_obs(const GameState& s, uint8_t player_pov, uint8_t* out) noexcept;
}
```

### Public Python API

```python
import fastcatan
env = fastcatan.BatchedEnv(num_envs=4096, seed=42)
obs, mask = env.reset()                              # zero-copy tensors
obs, reward, done, mask, info = env.step(actions)

single = fastcatan.GymEnv(opponent_fn=alpha_beta_policy)  # single-agent wrapper
```

## Key Design Details

### GameState (`include/catan/state.hpp`)

```cpp
struct alignas(64) GameState {
    // Nodes (54): high nibble = owner (0..3, 0xF empty), low nibble = level (0 empty, 1 settlement, 2 city).
    // One byte per slot carries both fields; one load per mask update after a build action.
    uint8_t node[54];

    // Edges (72): 0..3 owner, 0xFF empty. Byte-per-slot; scattered writes make bit-packing a net loss.
    uint8_t edge[72];

    // Board layout (randomized per episode; adjacency is static in topology.hpp).
    uint8_t hex_resource[19];           // 0=brick 1=lumber 2=wool 3=grain 4=ore 5=desert
    uint8_t hex_number[19];             // 0 for desert; else 2..12 skipping 7
    uint8_t port_type[9];               // 0..4 = 2:1 specific, 5 = 3:1 generic; indexed by port slot
    uint8_t robber_hex;

    // Per-player (AoS within env).
    uint8_t  resources[4][5];           // brick, lumber, wool, grain, ore
    uint8_t  dev_cards_playable[4][5];
    uint8_t  dev_cards_pending[4][5];   // bought this turn, not playable yet
    uint8_t  knights_played[4];
    uint8_t  vp_public[4];
    uint8_t  vp_hidden[4];              // VP dev cards
    uint8_t  longest_road_len[4];
    uint8_t  played_dev_this_turn[4];
    uint16_t ports_mask[4];             // bitstring: which ports player can trade through

    // Global.
    uint8_t  dev_deck_remaining[5];
    uint8_t  bank[5];
    uint8_t  longest_road_holder;       // 0xFF if none
    uint8_t  largest_army_holder;
    uint8_t  current_player;
    uint8_t  phase;                     // initial_placement / roll / main / discard / robber / steal / trade_propose / trade_respond
    uint8_t  dice_roll;
    uint8_t  turn_number;

    uint64_t rng_state[2];              // xoshiro128+
    uint64_t action_mask[5];            // incrementally maintained 320-bit bitstring
};
```

~512 bytes/env including alignment. Eight cache lines. 1M envs ≈ 500 MB.

**Encoding choices:**
- `node[54]` merges `owner` + `level` into one byte (nibble-packed). Every build action mutates owner and level together — one load, one store.
- `hex_resource` / `hex_number` stay `uint8_t` (not 3/4-bit packed) because production rolls SIMD-sweep all 19 hexes with `_mm256_cmpeq_epi8` against the rolled number. Bit-packing kills vectorization.
- `action_mask[5]` and `ports_mask[4]` are bitstrings by design — bitwise set ops (`&`, `|`, `popcount`, `tzcnt`) are 1-cycle and dominate action-sampling and port-trade checks.
- Everything else kept byte-wide — scattered read-modify-write cost of bit-packing outweighs cache savings since a whole env already fits in L1 many times over.

Topology (adjacency tables for 19 hexes / 54 nodes / 72 edges / 9 ports) is compile-time constants in `topology.hpp`, generated by `topology_gen.py` at build time.

### Action Space (~300 flat actions)

- Turn management: roll, end turn
- Build: 54 settlement + 54 city + 72 road
- Dev: buy, play {knight, YoP, RB, monopoly}; VP dev is passive
- YoP: 25 resource-pair choices (dedup via mask)
- Monopoly: 5
- Robber: 19 hex × steal-target (masked to ≤4)
- Discard sub-phase: 5 "discard 1 of R" actions, repeated until half shed
- Trade sub-phase: "add 1 of R to give", "add 1 of R to want", finalize, accept, reject, counter — composes instead of enumerating combinations

### Incremental Legal-Action Mask

Every state mutation updates affected mask bits (typically <10 bit flips). `step_one` never scans the board to determine legality. In **debug builds**, after every step, recompute the mask from scratch and `assert(incremental == recomputed)` — leave this on until M4.

### Parallelism

- Layer 1: single env step is ~100–200 ns of work — never thread internally.
- Layer 2: `BatchedEnv` holds N states in one contiguous buffer. `step()` is one `#pragma omp parallel for` over `step_one` calls. Auto-reset when `done_out[i]==1` — no stalls.
- Layer 3: output tensors (`obs_out`, `reward_out`, `done_out`, `mask_out`) are exposed to Python as zero-copy `nb::ndarray` views, wrapped as PyTorch tensors via DLPack. Same tensor object returned each step.
- Layer 4: `step()` releases the GIL (`nb::call_guard<nb::gil_scoped_release>`).
- Layer 5: NUMA-aware multi-process via `torch.multiprocessing` in M4 — one BatchedEnv per socket, shared-memory aggregation. **Do not use `SubprocVecEnv`** — pickling per step is the slow path we're replacing.
- Escape hatch: `BatchedEnv::step_n(num_steps, policy_fn)` runs N steps entirely in C++ with a C-callable policy. Used for random-policy benchmarks and frozen-opponent rollouts, bypassing Python entirely.

### Gymnasium Wrapper

Single-agent API returns obs from the perspective of whichever player is currently to-move. The env internally calls `opponent_fn(obs) -> action` for the 3 frozen opponents until control returns to the learner. `info["current_player_id"]` is always populated.

For trading-net training (M3 self-play+): a PettingZoo AEC wrapper yields control to Python for every player's every decision. Same C++ core, different Python surface.

### Correctness

TODO

## Milestones

### M0 — Foundations (DONE)

- C++23 simulator + nanobind bindings; Linux-x86_64 CI.
- Full rules: building, dev cards, robber, bank/port + player-to-player trading.
- Gymnasium 4-player env (single-agent wrapper) + PettingZoo AEC.
- Incremental legal-action mask (296 used / 320 bits); surgical trade updater.

### M1 — Baselines

- Random baseline (Python).
- Alpha-Beta baseline (Python): depth-limited minimax + multi-feature heuristic.
- Game-log capture for both Python baselines (action stream, board state, seed).
- Random baseline (C++).
- Alpha-Beta baseline (C++).
- Parity test: C++ vs Python baselines on shared seed-replay corpus (state/action equivalence + steps/sec comparison).
- Parity test: C++ baseline vs Catanatron on shared seed-replay corpus (rule equivalence). Important. Catan rules are notoriously subtle; even a single mis-implemented corner case can silently break the agent's learning.
- Throughput + log dashboard: steps/sec, episode length distribution, win rates per seat.
- **Gate: C++ baselines match Python and Catanatron on log replay (zero divergence) and beat Python throughput by ≥100×.**

### M2 — Initial RL Agent

- MaskablePPO training pipeline driven by `BatchedEnv` (`tools/train_smoke.py` hardened).
- Lock observation encoder + reward shaping.
- Train first model vs random opponents.
- **Gate: >90% win rate vs random baseline over 1000 four-player games.**

### M3 — Self-Play Training

- Iterative self-play schedule (frozen snapshot rotation).
- Hyperparam sweep over learning rate, entropy, snapshot interval.
- **Gate: latest model beats N-step-ago snapshot >55%.**

### M4 — Alpha-Beta Eval + Final Model

- Tournament harness `play(agent_a, agent_b, n_games) -> (win_rate, 95% CI)`.
- Final model vs Alpha-Beta over ≥1000 four-player games.
- 10⁸-step soak test for stability.
- Reproducibility doc: CMake args, GCC version, glibc, seed schedule, training config.
- **Thesis gate: >25% win rate vs Alpha-Beta with 95% CI.**

## Top 3 Risks

### Risk 1: Longest Road is a graph problem with subtle corner cases

Road length on a graph with opponent-cuts is hamiltonian-path-ish. Getting it wrong is silent in short games and hard to debug in long ones.

**Mitigation.** Isolate `longest_road.cpp` from day one. Write its tests first. Hand-build a corpus of ~200 positions, each with the known correct length per player, covering ties, cuts by opponent settlement, branching, and cycle cases. Run on every commit. Never write this from memory.

### Risk 2: Incremental mask goes out of sync with state

A stale mask bit lets the agent pick an illegal action; the sim either crashes or silently executes it. Training data corrupts invisibly.

**Mitigation.** Debug builds `assert(incremental == recomputed)` after every `step_one`. Keep on through M3 self-play; only turn off for the M4 final thesis runs. ~2× debug slowdown is cheap insurance. Seed-reproducible fuzz tests surface drift before it reaches real training.

### Risk 3: Python↔C++ boundary dominates step time at high N

At 5×10⁷ steps/sec with N=4096 envs, a `step()` call is ~80 µs of native work; nanobind dispatch + tensor lifecycle is ~5–20 µs. 6–25% overhead is tolerable but invisible in C++ microbenches.

**Mitigation.** Benchmark from Python from M1 baselines, not just from C++. Release GIL in `step()`. Same tensors reused across calls, zero per-step Python work. If still binding at M3 self-play, ship the `step_n(num_steps, policy_fn)` C-callable escape hatch for benchmarks and frozen-opponent rollouts.

## Critical Files (M0 build order, retained for reference)

1. `CMakeLists.txt`, `pyproject.toml` — scaffold
2. `src/catan/topology_gen.py` + `include/catan/topology.hpp` — board constants
3. `include/catan/rng.hpp`, `include/catan/state.hpp` — core types
4. `include/catan/rules.hpp` + `src/catan/rules.cpp` — step_one, reset_one
5. `src/catan/longest_road.cpp` — isolated, test-first
6. `include/catan/mask.hpp` + `src/catan/mask.cpp` — incremental legal-action mask
7. `src/catan/obs.cpp` — observation tensor
8. `include/catan/batched_env.hpp` + `src/catan/batched_env.cpp` — batched stepping
9. `bindings/pycatan/bindings.cpp` — nanobind exposure
10. `python/fastcatan/gym_env.py` — single-agent wrapper
11. `bench/bench_step.cpp` + `python/fastcatan/benchmarks/throughput.py` — benchmarks

## Verification

- **Invariants:** `ctest -R invariants` runs the 10⁷-game fuzz loop. Must pass green before any throughput claim is made.
- **Determinism:** `ctest -R perft` runs fixed-seed/fixed-policy 100k-step trajectory, compares hash to the committed value.
- **Throughput:** `python -m fastcatan.benchmarks.throughput --n-envs 4096 --steps 100000 --warmup 1000` reports steps/sec. Run 3× under `numactl --cpubind=0 --membind=0`; take median. Gate on per-milestone target.
- **RL smoke test (M2 onward):** `python scripts/ppo_smoke.py` — 1M-step PPO run vs random opponent, must reach >90% win rate.
- **Final (M4):** `python scripts/tournament.py --agent-a ppo_final --agent-b alphabeta --n-games 1000` reports win rate + 95% CI.
