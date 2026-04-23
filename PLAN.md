# fastCatan — High-Throughput Catan Simulator

## Context

Author is writing on RL for 4-player Settlers of Catan. Thesis goal: an RL agent that beats Alpha-Beta with statistical significance (>25% win rate over ≥1000 four-player games). The bottleneck the thesis explicitly calls out is simulator throughput — existing Python sim (Catanatron) can't generate self-play at the volume modern RL needs.

This plan designs `fastCatan`: a greenfield C++20 simulator with nanobind Python bindings, Gymnasium interface, targeting **5×10⁷ steps/sec/node** by milestone 4, with full 4-player rules including player-to-player trading intact. Correctness is continuously cross-validated against Catanatron as ground truth.

Decisions confirmed by user: greenfield C++, Linux HPC only, full trading from M2.

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
| Gym API | Single-agent Gymnasium + perspective flip for self-play; PettingZoo AEC added in M3 | 99% of RL libs target Gym; AEC wrapper needed only for trading-net training |
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
│   ├── catanatron_diff.py             # differential test harness
│   └── benchmarks/throughput.py
├── tests/
│   ├── cpp/{test_rules,test_mask,test_longest_road,test_determinism}.cpp
│   └── python/{test_gym_api,test_differential,test_invariants}.py
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

For trading-net training (thesis M3+): a PettingZoo AEC wrapper yields control to Python for every player's every decision. Same C++ core, different Python surface.

### Correctness

Three-pillar strategy, ordered by ROI:

1. **Differential testing against Catanatron.** Generate 10k random-legal sequences via fastCatan's own mask, replay in Catanatron, assert equality on resources, VP, occupancy, phase, dice, dev deck after every step. Commit disagreement list as a pinned test suite. Catanatron is the ground truth.
2. **Property/invariant fuzz.** Run 10⁷ random games, assert after every step: `sum(resources_in_play) + sum(bank) == 95`; `sum(dev_deck) + sum(held) + sum(played_knights) == 25`; `vp_public[i] + vp_hidden[i] ≤ 12`; settlement distance-rule; road connectivity; mask nonempty unless terminal. `assert()` macros — on in debug/CI, off in release.
3. **Perft-style determinism.** Fixed seed + fixed random policy hash → 100k steps → hash of final state. Committed. Any refactor that changes the hash breaks CI.
4. **Unit tests.** ~50 handwritten rule corner cases (longest-road ties, road-cut-by-settlement, largest-army transfers, port 2:1 without resources, trade offered to player at 7-card limit, YoP with empty bank, etc.). Mirror Catanatron's own test corpus where possible.

## Milestones (aligned with the thesis's 5 × 4-week blocks)

Principle: **RL work is unblocked by end of M1**; every later milestone is additive.

### M1 — Weeks 1–4 (May 2026): Usable Vertical Slice

- CMake + nanobind + pyproject skeleton; Linux-x86_64 CI. `CMAKE_CXX_STANDARD 23`, `CMAKE_CXX_STANDARD_REQUIRED ON`, `CMAKE_CXX_EXTENSIONS OFF`. Pin GCC ≥14 in CI. Document HPC compiler version in README.
- `topology.hpp` codegen, standard 19-hex board only.
- `GameState`, `reset_one`, `step_one`, `write_obs`. No incremental mask — recompute legality on demand.
- Rules: initial placement, rolling, all building, buy-dev, play-knight + robber, bank/port trading. **Other dev cards and player-to-player trading deferred to M2** (scope cut to hit date).
- Single-env Gymnasium wrapper.
- Differential test harness vs Catanatron (green on implemented subset).
- First throughput benchmark.
- **Target: 5×10⁵ steps/sec.** RL researcher can start PPO vs random opponent on a crippled-but-correct Catan.

### M2 — Weeks 5–8 (June): Full Rules + Batching

- All dev cards (Year-of-Plenty, Road-Builder, Monopoly). VP dev-card hidden count.
- Longest-road algorithm in `src/catan/longest_road.cpp` with its own test corpus (hand-built positions with known-correct lengths, ported from Catanatron's test cases).
- Player-to-player trading as a sub-phase (compositional encoding).
- `BatchedEnv` with OpenMP parallel-for.
- DLPack zero-copy obs/mask/reward/done tensors via nanobind.
- Fuzzer passes 10⁷ random games with zero invariant violations.
- Differential test: 10k-step sequences match Catanatron exactly.
- **Target: 5×10⁶ steps/sec ("millions of steps/sec" threshold hit).**

### M3 — Weeks 9–12 (July): Incremental Mask + Correctness Lockdown

- Incremental legal-action mask. Debug builds assert `incremental == recomputed` every step.
- PettingZoo AEC wrapper (needed for trading-net training).
- Perft-style determinism test pinned in CI.
- PGO pipeline set up.
- CI regression detection: throughput must not drop >5% between commits.
- **Target: 2×10⁷ steps/sec.**

### M4 — Weeks 13–20 (Aug–Sept): Performance Endgame

- SIMD batch-mask computation where profiling shows wins (AVX2/AVX-512 for batched affordability checks).
- NUMA-aware multi-process sharding via `torch.multiprocessing`.
- Final PGO + LTO build.
- `perf stat` audit of L1/L2/LLC miss rates; `perf record` for hot functions.
- Observation encoder microbench — at 5×10⁷ steps/sec, writing obs may need SIMD scatter.
- Turn off debug assertions for final runs.
- **Target: 5×10⁷ steps/sec.** This is the long-run training regime for thesis results.

### M5 — Weeks 21–26 (Oct–Nov): Harden for Final Evaluation

- Freeze `step_one` semantics. No more API changes.
- 10⁸-step soak test — no crashes, no leaks, no accumulation drift.
- Tournament harness: `play(agent_a, agent_b, n_games) -> (win_rate, confidence_interval)`. This is what validates the "beat AlphaBeta over ≥1000 games" claim.
- Reproducibility doc: exact CMake args, compiler version, glibc, seed schedule.

## Top 3 Risks

### Risk 1: Longest Road is a graph problem with subtle corner cases

Road length on a graph with opponent-cuts is hamiltonian-path-ish. Getting it wrong fails differential tests only in long games — hard to debug.

**Mitigation.** Isolate `longest_road.cpp` from day one. Write its tests first. Port Catanatron's test positions into our corpus (~200 boards, each with known length per player). Run on every commit. Never write this from memory.

### Risk 2: Incremental mask goes out of sync with state

A stale mask bit lets the agent pick an illegal action; the sim either crashes or silently executes it. Training data corrupts invisibly.

**Mitigation.** Debug builds `assert(incremental == recomputed)` after every `step_one`. Keep on through M4; only turn off for the final thesis runs. ~2× debug slowdown is cheap insurance. Seed-reproducible fuzz tests surface drift before it reaches real training.

### Risk 3: Python↔C++ boundary dominates step time at high N

At 5×10⁷ steps/sec with N=4096 envs, a `step()` call is ~80 µs of native work; nanobind dispatch + tensor lifecycle is ~5–20 µs. 6–25% overhead is tolerable but invisible in C++ microbenches.

**Mitigation.** Benchmark from Python from M1, not just from C++. Release GIL in `step()`. Same tensors reused across calls, zero per-step Python work. If still binding at M4, ship the `step_n(num_steps, policy_fn)` C-callable escape hatch for benchmarks and frozen-opponent rollouts.

## Critical Files to Create (in order)

1. `CMakeLists.txt`, `pyproject.toml` — scaffold
2. `src/catan/topology_gen.py` + `include/catan/topology.hpp` — board constants
3. `include/catan/rng.hpp`, `include/catan/state.hpp` — core types
4. `include/catan/rules.hpp` + `src/catan/rules.cpp` — step_one, reset_one
5. `src/catan/longest_road.cpp` — isolated, test-first
6. `include/catan/mask.hpp` + `src/catan/mask.cpp` — legal-action mask (recompute version first, incremental in M3)
7. `src/catan/obs.cpp` — observation tensor
8. `include/catan/batched_env.hpp` + `src/catan/batched_env.cpp` — batched stepping (M2)
9. `bindings/pycatan/bindings.cpp` — nanobind exposure
10. `python/fastcatan/gym_env.py` — single-agent wrapper
11. `python/fastcatan/catanatron_diff.py` — differential test harness
12. `bench/bench_step.cpp` + `python/fastcatan/benchmarks/throughput.py` — benchmarks

## Verification

- **Correctness:** `pytest tests/python/test_differential.py` runs 10k random sequences through both fastCatan and Catanatron, diffs every step. Must pass green before any throughput claim is made.
- **Invariants:** `ctest -R invariants` runs the 10⁷-game fuzz loop.
- **Determinism:** `ctest -R perft` runs fixed-seed/fixed-policy 100k-step trajectory, compares hash to the committed value.
- **Throughput:** `python -m fastcatan.benchmarks.throughput --n-envs 4096 --steps 100000 --warmup 1000` reports steps/sec. Run 3× under `numactl --cpubind=0 --membind=0`; take median. Gate on per-milestone target.
- **RL smoke test (M2 onward):** `python scripts/ppo_smoke.py` — 1M-step PPO run vs random opponent, must reach >90% win rate.
- **Final (M5):** `python scripts/tournament.py --agent-a ppo_final --agent-b alphabeta --n-games 1000` reports win rate + 95% CI.
