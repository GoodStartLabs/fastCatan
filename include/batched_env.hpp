#pragma once
#include <cstddef>
#include <cstdint>
#include "state.hpp"

namespace catan {

    // BatchedEnv — N independent (GameState, BoardLayout) pairs in one
    // contiguous buffer. Single-threaded loop for now; add OpenMP after the
    // benchmark says it pays.
    //
    // Auto-reset semantics: when step() observes done=1 for env i, the
    // next reset_one is invoked immediately so the next step() starts on
    // a fresh game. This avoids "slow-env-stalls-batch" bugs in vectorized
    // training.
    struct BatchedEnv {
        uint32_t n;                  // number of envs
        GameState*  states;          // aligned array, length n
        BoardLayout* layouts;        // length n
        uint64_t    seed_counter;    // monotonic per-env seed derivation
        uint8_t*    last_winner;     // last game's winner per env, NO_PLAYER if none
    };

    // Allocate buffers, zero state, seed each env from `master_seed` via
    // SplitMix64. Does NOT call reset_one — caller drives that via
    // batched_env_reset.
    void batched_env_init(BatchedEnv& env, uint32_t n_envs, uint64_t master_seed) noexcept;

    // Free buffers. Safe to call on a zero-initialized BatchedEnv.
    void batched_env_destroy(BatchedEnv& env) noexcept;

    // Reset all envs to fresh starting positions, each with a unique seed.
    void batched_env_reset(BatchedEnv& env) noexcept;

    // Step every env by one action. `actions` length n.
    // Writes per-env reward + done. On done, env is auto-reset.
    void batched_env_step(BatchedEnv& env,
                           const uint32_t* actions,
                           float* rewards_out,
                           uint8_t* dones_out) noexcept;

    // Write obs from each env's current_player POV. `out` length n*OBS_SIZE.
    void batched_env_write_obs(const BatchedEnv& env, float* out) noexcept;

    // Write legal-action mask for every env. `out` length n*MASK_WORDS.
    void batched_env_write_masks(const BatchedEnv& env, uint64_t* out) noexcept;

    // ------------------------------------------------------------------
    // Batched tree-search primitives (GPU-batched MCTS support).
    //
    // Additive: the rollout API above is unchanged. These let a search drive
    // the N slots as parallel scratch branches — load a frontier of
    // snapshots, reseed chance per simulation, step WITHOUT auto-reset, and
    // read obs/signatures in one OpenMP pass instead of N Python calls.
    // ------------------------------------------------------------------

    // Bytes per env snapshot (GameState + BoardLayout) — the row stride of
    // save/load buffers. Matches Env.snapshot()'s byte length.
    inline constexpr std::size_t SNAPSHOT_BYTES =
        sizeof(GameState) + sizeof(BoardLayout);

    // step_raw: this action id leaves the env untouched (finished walkers
    // idle while the rest of the batch keeps descending).
    inline constexpr uint32_t SKIP_ACTION = 0xFFFFFFFFu;

    // Ints per row of batched_env_write_sigs:
    // [current_player, phase, flag, dice_roll, handsize0..3, vp0..3].
    inline constexpr int SIG_INTS = 12;

    // Serialize every env into `out` (n * SNAPSHOT_BYTES, row i = env i).
    void batched_env_save(const BatchedEnv& env, uint8_t* out) noexcept;

    // Restore every env from `buf` (n * SNAPSHOT_BYTES, row i = env i).
    void batched_env_load(BatchedEnv& env, const uint8_t* buf) noexcept;

    // Reseed env i's in-state RNG with seeds[i]. Like Env.reseed: required
    // after a load so each simulation resamples chance instead of replaying
    // the snapshot's predetermined dice/draws.
    void batched_env_reseed(BatchedEnv& env, const uint64_t* seeds) noexcept;

    // Step env i by actions[i] WITHOUT auto-reset: terminal states stay
    // readable and loaded branches aren't wiped mid-search. SKIP_ACTION
    // leaves env i untouched (rewards_out[i]=0, dones_out[i]=0). Does not
    // advance seed_counter.
    void batched_env_step_raw(BatchedEnv& env, const uint32_t* actions,
                              float* rewards_out, uint8_t* dones_out) noexcept;

    // Obs for env i from povs[i]'s POV (not just current_player).
    // `out` length n*OBS_SIZE.
    void batched_env_write_obs_pov(const BatchedEnv& env, const uint8_t* povs,
                                   float* out) noexcept;

    // Obs for env i from ALL 4 POVs -> `out` is (n, 4, OBS_SIZE), pov-major
    // within each env. One pass for max^n leaf evaluation.
    void batched_env_write_obs_all4(const BatchedEnv& env, float* out) noexcept;

    // OBS_FULL_SIZE variants (POV prefix + hidden-enemy appendix) for the
    // learned judge. `out` rows are OBS_FULL_SIZE wide.
    void batched_env_write_obs_full_pov(const BatchedEnv& env, const uint8_t* povs,
                                        float* out) noexcept;
    void batched_env_write_obs_full_all4(const BatchedEnv& env, float* out) noexcept;

    // Decision/chance signature per env -> `out` is (n, SIG_INTS) int32.
    // Row layout documented at SIG_INTS. Distinguishes chance outcomes the
    // same way the Python MCTS _signature does (dice totals, robber steals
    // via handsizes, VP dev-draws via vps).
    void batched_env_write_sigs(const BatchedEnv& env, int32_t* out) noexcept;

    // Native AlphaBeta pick for every env's CURRENT player, one OpenMP pass:
    // out[i] = ab_decide(states[i], layouts[i], states[i].current_player,
    // depth, prune, nullptr, banned). 0xFFFFFFFF where no legal action.
    // `banned` optional uint64[MASK_WORDS] (see search.hpp). This is the
    // batched opponent/in-tree-advance primitive for training vs AB — the
    // per-env Python Env round-trip was the throughput killer.
    void batched_env_ab_decide(const BatchedEnv& env, int depth, bool prune,
                               const uint64_t* banned, uint32_t* out,
                               int chance_mode = 0) noexcept;

}  // namespace catan
