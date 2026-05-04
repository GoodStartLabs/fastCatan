#pragma once
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

}  // namespace catan
