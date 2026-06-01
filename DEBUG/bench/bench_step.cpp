// bench/bench_step.cpp — single-env pure-C++ throughput floor.
//
// Reports two numbers:
//   * step_one only (replay)  — pure C++ kernel cost, no action selection in
//     the timed loop. Directly comparable to the Python dashboard's
//     `env.step` figure minus the nanobind dispatch floor.
//   * random play (end-to-end) — realistic loop incl. reading the maintained
//     mask + a uniform-random legal picker, like the orphaned binary reported.
//
// Usage:  bench_step [target_steps=1000000] [seed=42]
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "rules.hpp"
#include "state.hpp"
#include "rng.hpp"
#include "bench_common.hpp"

using namespace catan;
using clk = std::chrono::steady_clock;

static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
    const uint64_t target = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 1'000'000ull;
    const uint64_t seed   = (argc > 2) ? strtoull(argv[2], nullptr, 10) : 42ull;

    Xoshiro128 picker;
    xoshiro_seed(picker, seed ^ 0x9e3779b97f4a7c15ull);

    // --- Phase 1: generate an action stream by random legal play (untimed). ---
    // Record the per-game reset seeds so the replay reproduces the same dones,
    // keeping every replayed action legal.
    std::vector<uint32_t> stream;
    std::vector<uint64_t> reset_seeds;
    stream.reserve(target);
    {
        GameState s{}; BoardLayout b{};
        uint64_t gseed = seed;
        reset_one(s, b, gseed);
        reset_seeds.push_back(gseed);
        float reward; uint8_t done;
        while (stream.size() < target) {
            const uint32_t a = bench::pick_random_legal(s.action_mask, picker);
            stream.push_back(a);
            step_one(s, b, a, reward, done);
            if (done) {
                gseed = seed + reset_seeds.size();
                reset_one(s, b, gseed);
                reset_seeds.push_back(gseed);
            }
        }
    }

    // --- Phase 2a: timed replay — pure step_one, no picker/mask in the loop. ---
    uint64_t replay_games = 0;
    double t_replay;
    {
        GameState s{}; BoardLayout b{};
        size_t seg = 0;
        reset_one(s, b, reset_seeds[seg++]);
        float reward; uint8_t done;
        const auto t0 = clk::now();
        for (const uint32_t a : stream) {
            step_one(s, b, a, reward, done);
            if (done) {
                ++replay_games;
                if (seg < reset_seeds.size()) reset_one(s, b, reset_seeds[seg++]);
            }
        }
        t_replay = secs(t0, clk::now());
    }

    // --- Phase 2b: timed random play — end-to-end (mask read + picker + step). ---
    uint64_t play_steps = 0, play_games = 0;
    double t_play;
    {
        GameState s{}; BoardLayout b{};
        uint64_t gseed = seed;
        reset_one(s, b, gseed);
        Xoshiro128 rng;
        xoshiro_seed(rng, seed ^ 0xdeadbeefcafef00dull);
        float reward; uint8_t done;
        const auto t0 = clk::now();
        while (play_steps < target) {
            const uint32_t a = bench::pick_random_legal(s.action_mask, rng);
            step_one(s, b, a, reward, done);
            ++play_steps;
            if (done) {
                ++play_games;
                gseed = seed + play_games;
                reset_one(s, b, gseed);
            }
        }
        t_play = secs(t0, clk::now());
    }

    printf("=== bench_step (single-env, pure C++) ===\n");
    printf("target steps:        %llu\n", (unsigned long long)target);
    printf("\n");
    printf("[step_one only, replay]\n");
    printf("  games:             %llu\n", (unsigned long long)replay_games);
    printf("  elapsed:           %.4f s\n", t_replay);
    printf("  steps/sec:         %.0f\n", target / t_replay);
    printf("  ns/step:           %.1f\n", t_replay * 1e9 / target);
    printf("\n");
    printf("[random play, end-to-end: mask read + picker + step_one]\n");
    printf("  games:             %llu\n", (unsigned long long)play_games);
    printf("  elapsed:           %.4f s\n", t_play);
    printf("  steps/sec:         %.0f\n", play_steps / t_play);
    printf("  games/sec:         %.0f\n", play_games / t_play);
    printf("  ns/step:           %.1f\n", t_play * 1e9 / play_steps);
    if (play_games)
        printf("  steps/game:        %.1f\n", double(play_steps) / play_games);
    return 0;
}
