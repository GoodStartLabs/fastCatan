#include "batched_env.hpp"
#include "rules.hpp"
#include "mask.hpp"
#include "obs.hpp"
#include "rng.hpp"

#include <cstdlib>
#include <cstring>
#include <new>

namespace catan {

namespace {

// Stable per-env seed derivation. SplitMix64 of (master, env_id).
inline uint64_t derive_seed(uint64_t master, uint32_t i) noexcept {
    uint64_t x = master ^ (uint64_t(i) * 0x9E3779B97F4A7C15ULL);
    return splitmix64(x);
}

template <typename T>
T* aligned_array(uint32_t n) noexcept {
    constexpr std::size_t ALIGN = 64;
    std::size_t bytes = sizeof(T) * std::size_t(n);
    bytes = (bytes + ALIGN - 1) & ~(ALIGN - 1);
    void* p = std::aligned_alloc(ALIGN, bytes);
    return reinterpret_cast<T*>(p);
}

}  // namespace

void batched_env_init(BatchedEnv& env, uint32_t n_envs, uint64_t master_seed) noexcept {
    env.n            = n_envs;
    env.states       = aligned_array<GameState>(n_envs);
    env.layouts      = aligned_array<BoardLayout>(n_envs);
    env.last_winner  = aligned_array<uint8_t>(n_envs);
    env.seed_counter = master_seed;

    for (uint32_t i = 0; i < n_envs; ++i) {
        new (&env.states[i])  GameState{};
        new (&env.layouts[i]) BoardLayout{};
        env.last_winner[i] = NO_PLAYER;
    }
}

void batched_env_destroy(BatchedEnv& env) noexcept {
    if (env.states)      std::free(env.states);
    if (env.layouts)     std::free(env.layouts);
    if (env.last_winner) std::free(env.last_winner);
    env.states      = nullptr;
    env.layouts     = nullptr;
    env.last_winner = nullptr;
    env.n           = 0;
}

void batched_env_reset(BatchedEnv& env) noexcept {
#if FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i) {
        uint64_t seed = derive_seed(env.seed_counter, uint32_t(i));
        reset_one(env.states[i], env.layouts[i], seed);
    }
    env.seed_counter += env.n;
}

void batched_env_step(BatchedEnv& env,
                       const uint32_t* actions,
                       float* rewards_out,
                       uint8_t* dones_out) noexcept {
#if FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i) {
        float reward = 0.0f;
        uint8_t done = 0;
        step_one(env.states[i], env.layouts[i], actions[i], reward, done);

        rewards_out[i] = reward;
        dones_out[i]   = done;

        if (done) {
            // Capture winner before auto-reset wipes the state.
            uint8_t winner = NO_PLAYER;
            for (uint8_t p = 0; p < 4; ++p) {
                if (env.states[i].player_vp[p] >= 10) {
                    winner = p;
                    break;
                }
            }
            env.last_winner[i] = winner;

            uint64_t seed = derive_seed(env.seed_counter, uint32_t(i));
            reset_one(env.states[i], env.layouts[i], seed);
        }
    }
    env.seed_counter += env.n;
}

void batched_env_write_obs(const BatchedEnv& env, float* out) noexcept {
#if FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i) {
        write_obs(env.states[i], env.layouts[i],
                   env.states[i].current_player,
                   out + std::size_t(i) * OBS_SIZE);
    }
}

void batched_env_write_masks(const BatchedEnv& env, uint64_t* out) noexcept {
    // Read from the incrementally-maintained s.action_mask field. step_one
    // and reset_one keep it current. ~free vs the previous full recompute.
#if FCATAN_HAVE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < int32_t(env.n); ++i) {
        std::memcpy(out + std::size_t(i) * MASK_WORDS,
                    env.states[i].action_mask,
                    sizeof(uint64_t) * MASK_WORDS);
    }
}

}  // namespace catan
