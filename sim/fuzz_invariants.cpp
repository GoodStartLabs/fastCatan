// sim/fuzz_invariants.cpp — 10^7-game invariant fuzz (M1 correctness gate).
//
// Plays random-legal games entirely in C++ and checks per-step game invariants:
//   * phase / current_player in valid range
//   * per-player VP <= 12, public VP <= total VP
//   * hand size == sum of held resources
//   * settlement/city/road stock never exceeds starting stock (5/4/15) — the
//     fields count *remaining* pieces, so this also catches uint8 underflow
//   * bank + sum(player resources) conserved at 19 per resource type
//   * a non-empty legal-action mask at every step
//   * terminal state has a winner (some player VP >= 10)
//
// These INVARIANT VIOLATIONS are the correctness gate; any one is a rule bug
// and exits nonzero. Distinct from that: a game may exceed the per-game step
// cap without terminating. Under *uniform-random* play this is NOT a rule
// defect — it is one of two rule-correct situations, and every per-step
// invariant still holds up to the cap:
//   1. Heavy tail — a finite but very long game (the trade sub-phase lets
//      random play wander). Replaying with a larger cap terminates normally
//      (e.g. seed 1366915 ends at 194018 steps, 0 violations).
//   2. Deadlock — the board is built out AND the dev deck is exhausted, so the
//      last 1-2 VP are unreachable for every player (no affordable/legal build
//      remains; longest-road and largest-army are already assigned). Max VP
//      across all players stays < 10 forever, so there is no winner (e.g. seed
//      2446268: 10^8 steps, no player's VP ever exceeds 9). step_one has no
//      "no-progress" terminal, so such a game never ends on its own; at the RL
//      level this is handled by the episode step cap in models/env.py
//      (MAX_EPISODE_STEPS), which truncates a no-winner game and scores it -1.
// Capped games are counted and the longest is reported; they do NOT fail the
// gate — only true invariant violations do.
//
// Mirrors sim/tests/test_invariants.py (the readable spec) but runs the full
// 10^7-game sweep that pure Python cannot: Python ~35 games/s/core (per-step
// accessor calls dominate); this is ~5x10^4 games/s, OpenMP-scaled.
//
// Each game is fully determined by its seed (engine RNG + action picker both
// seeded from it), so any reported game reproduces from the printed seed.
//
// Usage:  fuzz_invariants [games=1000000] [base_seed=0] [max_steps_per_game=1000000]
// Gate:   build/fuzz_invariants 10000000     # exit 0 == zero invariant violations
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "rules.hpp"
#include "state.hpp"
#include "rng.hpp"
#include "mask.hpp"   // MASK_WORDS

using namespace catan;
using clk = std::chrono::steady_clock;

namespace {

constexpr int     STARTING_BANK = 19;
constexpr uint8_t SETTLE_STOCK  = 5;
constexpr uint8_t CITY_STOCK    = 4;
constexpr uint8_t ROAD_STOCK    = 15;

// Uniformly pick one legal action from the maintained mask; sets `empty` if no
// bit is set (an invariant violation in itself — roll/end_turn must always keep
// at least one legal action).
inline uint32_t pick_random_legal(const uint64_t mask[MASK_WORDS],
                                  Xoshiro128& rng, bool& empty) noexcept {
    uint32_t buf[MASK_WORDS * 64];
    uint32_t n = 0;
    for (uint32_t w = 0; w < MASK_WORDS; ++w) {
        uint64_t bits = mask[w];
        const uint32_t base = w * 64u;
        while (bits) {
            buf[n++] = base + uint32_t(__builtin_ctzll(bits));
            bits &= bits - 1;
        }
    }
    empty = (n == 0);
    return n ? buf[rng.bounded(n)] : 0u;
}

// Returns a static description of the first violated invariant, or nullptr if
// the state is consistent. Cheap: ~40 byte loads, all-OK path branchless-ish.
inline const char* check_invariants(const GameState& s) noexcept {
    if (static_cast<uint8_t>(s.phase) > 3) return "phase out of range";
    if (s.current_player >= NUM_PLAYERS)   return "current_player out of range";

    for (int p = 0; p < NUM_PLAYERS; ++p) {
        if (s.player_vp[p] > 12)                         return "player VP impossibly high (>12)";
        if (s.player_vp_without_dev[p] > s.player_vp[p]) return "public VP exceeds total VP";
        int hand = 0;
        for (int r = 0; r < NUM_RESOURCES; ++r) hand += s.player_resources[p][r];
        if (int(s.player_handsize[p]) != hand)           return "handsize desync vs resource sum";
        if (s.player_settlement_count[p] > SETTLE_STOCK) return "settlement stock overflow/underflow";
        if (s.player_city_count[p] > CITY_STOCK)         return "city stock overflow/underflow";
        if (s.player_road_count[p] > ROAD_STOCK)         return "road stock overflow/underflow";
    }

    for (int r = 0; r < NUM_RESOURCES; ++r) {
        int total = s.bank[r];
        for (int p = 0; p < NUM_PLAYERS; ++p) total += s.player_resources[p][r];
        if (total != STARTING_BANK)                      return "resource not conserved (!= 19)";
    }
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    const uint64_t games   = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 1'000'000ull;
    const uint64_t base    = (argc > 2) ? strtoull(argv[2], nullptr, 10) : 0ull;
    const uint64_t maxstep = (argc > 3) ? strtoull(argv[3], nullptr, 10) : 1'000'000ull;

    std::atomic<uint64_t> games_done{0};   // reached a valid terminal (winner)
    std::atomic<uint64_t> total_steps{0};
    std::atomic<uint64_t> violations{0};   // true invariant breaches — the gate
    std::atomic<uint64_t> capped{0};       // exceeded step cap without terminating
    std::atomic<bool>     have_violation{false};

    // First-violation + global-longest-game detail (guarded by critical).
    uint64_t    vio_seed = 0, vio_step = 0;
    const char* vio_msg  = nullptr;
    uint64_t    longest_steps = 0, longest_seed = 0;

    auto record_violation = [&](uint64_t seed, uint64_t step, const char* msg) {
        violations.fetch_add(1, std::memory_order_relaxed);
        #pragma omp critical(fuzz_report)
        {
            if (!have_violation.load(std::memory_order_relaxed)) {
                vio_seed = seed; vio_step = step; vio_msg = msg;
                have_violation.store(true, std::memory_order_relaxed);
            }
        }
    };

    const auto t0 = clk::now();
    #pragma omp parallel
    {
        GameState s; BoardLayout b;
        Xoshiro128 picker;
        uint64_t local_steps = 0;
        uint64_t t_longest_steps = 0, t_longest_seed = 0;  // thread-local longest game

        #pragma omp for schedule(dynamic, 1024)
        for (long long g = 0; g < (long long)games; ++g) {
            if (have_violation.load(std::memory_order_relaxed)) continue;  // drain fast on a real bug

            const uint64_t gseed = base + uint64_t(g);
            reset_one(s, b, gseed);
            xoshiro_seed(picker, gseed ^ 0x9e3779b97f4a7c15ull);

            if (const char* msg = check_invariants(s)) { record_violation(gseed, 0, msg); continue; }

            float    reward = 0.0f;
            uint8_t  done   = 0;
            uint64_t steps  = 0;
            bool     bad    = false;

            while (steps < maxstep) {
                bool empty = false;
                const uint32_t a = pick_random_legal(s.action_mask, picker, empty);
                if (empty) { record_violation(gseed, steps, "empty legal-action mask"); bad = true; break; }
                step_one(s, b, a, reward, done);
                ++steps;
                if (const char* msg = check_invariants(s)) { record_violation(gseed, steps, msg); bad = true; break; }
                if (done) break;
            }
            local_steps += steps;
            if (steps > t_longest_steps) { t_longest_steps = steps; t_longest_seed = gseed; }

            if (bad) continue;
            if (!done) {
                // Hit the cap without terminating: heavy-tail random game, NOT a
                // rule violation. Count it; do not fail the gate.
                capped.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            uint8_t mx = 0;
            for (int p = 0; p < NUM_PLAYERS; ++p) mx = std::max(mx, s.player_vp[p]);
            if (mx < WIN_VP) { record_violation(gseed, steps, "terminated without a winner (max VP < 10)"); continue; }

            games_done.fetch_add(1, std::memory_order_relaxed);
        }

        total_steps.fetch_add(local_steps, std::memory_order_relaxed);
        #pragma omp critical(fuzz_report)
        {
            if (t_longest_steps > longest_steps) { longest_steps = t_longest_steps; longest_seed = t_longest_seed; }
        }
    }
    const double elapsed = std::chrono::duration<double>(clk::now() - t0).count();

    const uint64_t nvio   = violations.load();
    const uint64_t nsteps = total_steps.load();
    const uint64_t ndone  = games_done.load();
    const uint64_t ncap   = capped.load();

    printf("=== fuzz_invariants (pure C++%s) ===\n",
#ifdef _OPENMP
           ", OpenMP"
#else
           ", single-thread"
#endif
    );
    printf("games requested:      %llu\n", (unsigned long long)games);
    printf("base seed:            %llu\n", (unsigned long long)base);
    printf("max steps/game:       %llu\n", (unsigned long long)maxstep);
    printf("games terminated:     %llu\n", (unsigned long long)ndone);
    printf("games hit step cap:   %llu\n", (unsigned long long)ncap);
    printf("total steps:          %llu\n", (unsigned long long)nsteps);
    printf("elapsed:              %.3f s\n", elapsed);
    if (elapsed > 0) {
        printf("games/sec:            %.0f\n", (ndone + ncap) / elapsed);
        printf("steps/sec:            %.0f\n", nsteps / elapsed);
    }
    if (ndone + ncap) printf("steps/game (avg):     %.1f\n", double(nsteps) / double(ndone + ncap));
    printf("longest game:         %llu steps (seed %llu)\n",
           (unsigned long long)longest_steps, (unsigned long long)longest_seed);
    if (ncap) {
        printf("note: %llu game(s) hit the %llu-step cap without a winner. Under random\n"
               "      play this is rule-correct non-termination (a heavy-tail long game,\n"
               "      or a built-out/dev-exhausted deadlock where the last VP is\n"
               "      unreachable for all players), NOT an invariant breach. Longest:\n"
               "      seed %llu (replay: build/fuzz_invariants 1 %llu 100000000).\n",
               (unsigned long long)ncap, (unsigned long long)maxstep,
               (unsigned long long)longest_seed, (unsigned long long)longest_seed);
    }
    printf("invariant failures: %llu\n", (unsigned long long)nvio);
    if (nvio) {
        printf("FIRST VIOLATION: seed=%llu step=%llu : %s\n",
               (unsigned long long)vio_seed, (unsigned long long)vio_step,
               vio_msg ? vio_msg : "(unknown)");
        printf("  reproduce: build/fuzz_invariants 1 %llu\n", (unsigned long long)vio_seed);
        return 1;
    }
    printf("resource-conservation OK; zero invariant violations over %llu games "
           "(%llu terminated, %llu capped).\n",
           (unsigned long long)(ndone + ncap), (unsigned long long)ndone, (unsigned long long)ncap);
    return 0;
}
