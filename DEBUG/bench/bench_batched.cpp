// bench/bench_batched.cpp — batched pure-C++ throughput floor.
//
// Drives BatchedEnv with a uniform-random legal policy. The per-pass loop is
// write_masks -> (serial picker) -> step -> repeat, which mirrors the real
// hot path (policy serial, step parallel). With OpenMP compiled in, the
// per-env step is parallel; the picker stays serial, so at low core counts
// the picker can dominate — that's why bench_throughput.py isolates the C++
// step kernel separately. This binary reports the realistic end-to-end number.
//
// Usage:  bench_batched [n_envs=4096] [passes=5000] [seed=42]
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "batched_env.hpp"
#include "mask.hpp"   // MASK_WORDS
#include "rng.hpp"
#include "bench_common.hpp"

using namespace catan;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const uint32_t n_envs = (argc > 1) ? uint32_t(strtoul(argv[1], nullptr, 10)) : 4096u;
    const uint32_t passes = (argc > 2) ? uint32_t(strtoul(argv[2], nullptr, 10)) : 5000u;
    const uint64_t seed   = (argc > 3) ? strtoull(argv[3], nullptr, 10) : 42ull;

    BatchedEnv env{};
    batched_env_init(env, n_envs, seed);
    batched_env_reset(env);

    std::vector<uint32_t> actions(n_envs);
    std::vector<float>     rewards(n_envs);
    std::vector<uint8_t>   dones(n_envs);
    std::vector<uint64_t>  masks(size_t(n_envs) * MASK_WORDS);

    Xoshiro128 rng;
    xoshiro_seed(rng, seed ^ 0xdeadbeefcafef00dull);

    uint64_t total_steps = 0, games_done = 0;

    const auto t0 = clk::now();
    for (uint32_t p = 0; p < passes; ++p) {
        batched_env_write_masks(env, masks.data());
        for (uint32_t i = 0; i < n_envs; ++i)
            actions[i] = bench::pick_random_legal(&masks[size_t(i) * MASK_WORDS], rng);
        batched_env_step(env, actions.data(), rewards.data(), dones.data());
        for (uint32_t i = 0; i < n_envs; ++i) games_done += dones[i];
        total_steps += n_envs;
    }
    const double elapsed = std::chrono::duration<double>(clk::now() - t0).count();

    batched_env_destroy(env);

    printf("=== bench_batched (pure C++%s) ===\n",
#ifdef FCATAN_HAVE_OPENMP
           ", OpenMP"
#else
           ", single-thread"
#endif
    );
    printf("n_envs:        %u\n", n_envs);
    printf("step passes:   %u\n", passes);
    printf("total steps:   %llu\n", (unsigned long long)total_steps);
    printf("games done:    %llu\n", (unsigned long long)games_done);
    printf("elapsed:       %.4f s\n", elapsed);
    printf("steps/sec:     %.0f\n", total_steps / elapsed);
    printf("ns/step:       %.1f\n", elapsed * 1e9 / total_steps);
    printf("games/sec:     %.0f\n", games_done / elapsed);
    return 0;
}
