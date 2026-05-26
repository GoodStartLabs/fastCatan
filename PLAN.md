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

Enforced at three levels:

1. **Internal invariants & scenarios** (`sim/tests/`, pytest): resource
   conservation, mask legality, phase/flag transitions, determinism
   (fixed-seed trajectory), terminal reward.

2. **Cross-engine differential vs Catanatron** (the ground-truth oracle) —
   the M1 gate. A true co-stepping harness drives BOTH engines through the
   same action stream and asserts full state parity every ply. Pieces in
   `bridge/` (see `bridge/PLAN.md`):
   - `state_mirror.py` — byte-exact ctypes mirror of the 384-B C++
     `GameState`+`BoardLayout` (validated vs live snapshots).
   - `state_inject.py` — serialize any Catanatron state into fastcatan
     (`load_snapshot`).
   - `rng_force.py` — force fastcatan's xoshiro so dice / dev-draw / steal
     reproduce Catanatron's outcome (verified against the C++ engine).
   - `tests/test_differential.py` — co-stepping state parity.
   - `tests/test_obs_identity.py` — C++ `write_obs` vs bridge `encode_obs`
     bit-for-bit (obs layout + normalization parity).

   This is stronger than `test_parity_replay.py` (which only checks the obs
   encoder and never runs fastcatan's engine). It caught and fixed **5 real
   sim bugs** (git log): longest-road title off-by-one (first player to reach
   exactly 5 roads got no title +2 VP); production bank-shortage (all-or-
   nothing per resource, no partial to a sole recipient — matches Catanatron
   `yield_resources`); road buildability through an enemy-occupied component
   node; history-dependent longest-road membership (`road_node_member` field
   added to `GameState`); obs trade-response no-trade mismatch. Plus an
   initial-placement phase mislabel in the bridge encoder.

3. **Known parity bound — Catanatron's longest road is internally
   inconsistent.** It caches lengths (recomputes only the "plowed" player on a
   cut); its connected-component membership for enemy-boundary nodes is
   history-dependent (`build_road` excludes them, `dfs_walk` on a cut re-adds
   them); and it reassigns the title to the lowest seat index among tied
   players on a cut. fastcatan is rule-correct (incumbent keeps the title on a
   tie) and matches Catanatron on ~99% of positions; the residual (≤1–2% of
   random games, road cuts only) is Catanatron's own inconsistency, exempted
   in `test_differential`. Bit-parity there would need porting Catanatron's
   component state machine — not worth it (fastcatan is the correct one).

**Reproducibility** of the differential: pin `random.seed`, `np.random.seed`,
AND `PYTHONHASHSEED` — Catanatron's `RandomPlayer` + set-iteration depend on
all three; without `PYTHONHASHSEED` the corpus is not reproducible run-to-run.

## Milestones

### M0 — Foundations (DONE)

- C++23 simulator + nanobind bindings; Linux-x86_64 CI.
- Full rules: building, dev cards, robber, bank/port + player-to-player trading.
- Gymnasium 4-player env (single-agent wrapper) + PettingZoo AEC.
- Incremental legal-action mask (296 used / 320 bits); surgical trade updater.

### M1 — Correctness, Baselines, Bridge

Goal: validate the C++ sim plays Catan correctly before any RL touches it. Self-play means only the learned policy needs to be fast — baselines stay in Python.

- [x] Random baseline (Python).
- [x] Alpha-Beta baseline (Python): depth-limited minimax + multi-feature heuristic.
- [x] Log capture: random baseline games (action stream, board state, seed). `ui/recorder.py` + `logs/game.jsonl.gz`.
- [x] Log capture: Alpha-Beta baseline games.
- [x] Catanatron bridge: topology map, action codec, obs encoder, run_eval. 251 bridge tests green (`bridge/tests/`).
- [x] Replay parity tests: `bridge/tests/test_parity_replay.py` walks shared-seed games action-by-action against Catanatron.
- [x] **Cross-engine differential** (`bridge/tests/test_differential.py` + `test_obs_identity.py`): co-steps fastcatan *and* Catanatron on identical action streams, asserts full state + obs parity every ply (via the `state_mirror`/`state_inject`/`rng_force` harness). Found and fixed **5 real sim bugs** (see Correctness). 25-seed corpus green; residual ≤1–2% is Catanatron's own longest-road inconsistency (documented, exempted).
- [x] Throughput + bottleneck dashboard: `bench/bench_throughput.py` (Python-path breakdown) + `bench/bench_step.cpp` & `bench/bench_batched.cpp` (pure-C++ floor, standalone CMake targets). `bench_throughput.py` gives the single-env per-component µs breakdown, batched kernel scaling with nanobind-dispatch isolation, and fastcatan-vs-Catanatron on equal footing (games/s, turns/s; steps/s flagged non-comparable). `bench/bench_comprehensive.py` covers the distribution half (episode length percentiles, win rates per seat, VP). **Bottlenecks named:** single-env baseline is *Python-bound* — the legal-action bit-scan (`legal_actions`) is the largest kernel (~410 ns, ~36% of per-step) and total Python overhead (scan + interpreter glue + policy) is ~80%; the pure-C++ `step_one` is only ~47 ns (Release, `bench_step` replay) ≈ 4% of the per-step budget, with another ~76 ns of nanobind dispatch + return-tuple alloc on top in the bound path. Batched hot path is *obs-encode-bound* — `write_obs` (~80 ns/env, 1084 floats) dominates `step_one` (~40 ns/env) once dispatch amortizes (dispatch floor ~40 ns/call). Note: `step_one` fuses rules+mask-update+RNG and is not separable from Python; combined it is ~47 ns so not the bottleneck. Single-env 0.75M steps/s (Python) vs batched 24M steps/s = the ~32× amortization that justifies `BatchedEnv`; pure-C++ floors are ~21M (single-env `step_one`) and ~11M/core (`bench_batched 4096`, single-thread, near-linear with OpenMP on HPC). Equal footing vs Catanatron random4: ~7× games/s and ~7× turns/s.
- [ ] 10⁷-game invariant fuzz run (scaffolding in `sim/tests/test_invariants.py`, needs scaled execution).
- **Gate: zero rule divergence vs Catanatron on replay corpus; per-component bottleneck named with measured µs in dashboard.**

### M2 — Initial RL Agent

- [x] Shared Gym env over `fastcatan.Env` (`models/env.py`, seat 0 = learner, seats 1–3 random).
- [x] MaskablePPO trainer (`models/train_ppo.py`) + checkpoint at `models/checkpoints/ppo_random/ppo_final.zip`.
- [x] Reference trainers in separate files for thesis breadth: A2C, DQN, MuZero scaffold (`models/train_{a2c,dqn,muzero}.py`) + checkpoints.
- [x] Eval harness `models/eval.py` (win rate + 95% Wilson CI vs random over N games).
- [x] Lock observation encoder + reward. **Obs frozen**: 1084 floats, count fields normalized by structural Catan maxima (divisors in `src/catan/obs.cpp` `namespace norm`, mirrored in `bridge/obs_encoder.py` + `ui/obs_decoder.py` — keep in sync; verified by `test_obs_identity`). **Reward frozen**: sparse ±1 terminal (+1 on the action reaching 10 VP, −1 if it lets an opponent win), no shaping; `models/env.py` treats a stalled game (`turn_count >= MAX_TURNS=1000`) and a no-winner terminal as a **loss** (−1). NOTE: changing obs/reward invalidates trained checkpoints — the old `models/checkpoints/*` were deleted after this freeze.
- [ ] ⚠️ **Stall cap is ineffective as written.** `models/env.py` caps on `turn_count >= MAX_TURNS`, but `turn_count` only increments on END_TURN (`src/catan/rules.cpp:540`). The real degenerate stall is a *within-turn* `TRADE_OPEN`/`TRADE_CANCEL` loop (greedy/argmax policy never ends its turn), so `turn_count` freezes and the −1 cap never fires. Empirically, a trained greedy policy stalled ~85% of games to the step limit. **Fix: cap on a per-episode step counter (count `env.step` calls), and/or cap `TRADE_OPEN` re-opens per turn in the mask.** The −1 *values* are correct everywhere; only the trigger is broken.
- [ ] Gate run: ≥1000 games of PPO checkpoint vs random with win rate ≥0.90 recorded. (Retrain on the frozen interface first — pre-freeze checkpoints are gone.) NOTE: eval with **sampling**, not `--deterministic` (argmax triggers the trade-loop stall above). A real 10M-step run (~8 min, DummyVecEnv ~21.6k fps) reached ~90% sampled win rate before the checkpoints were cleared.
- **Gate: >90% win rate vs random baseline over 1000 four-player games.**

### M3 — Self-Play Training

- Iterative self-play schedule (frozen snapshot rotation).
- Hyperparam sweep over learning rate, entropy, snapshot interval.
- **Gate: latest model beats N-step-ago snapshot >55%.**

### M4 — Alpha-Beta Eval + Final Model

- [~] Tournament harness `play(agent_a, agent_b, n_games) -> (win_rate, 95% CI)` (eval.py covers PPO-vs-random shape; needs PPO-vs-AB wiring).
- [ ] Final model vs Alpha-Beta over ≥1000 four-player games.
- [ ] 10⁸-step soak test for stability.
- [ ] Reproducibility doc: CMake args, GCC version, glibc, seed schedule, training config.
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
