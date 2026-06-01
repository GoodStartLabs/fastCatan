// bench/bench_common.hpp — shared helpers for the standalone C++ benchmarks.
//
// These executables measure the PURE-C++ throughput floor (no Python, no
// nanobind). The Python dashboard (bench/bench_throughput.py) subtracts
// nanobind dispatch against these numbers.
#pragma once
#include <cstdint>

#include "mask.hpp"   // MASK_WORDS (pulls state.hpp transitively)
#include "rng.hpp"    // Xoshiro128

namespace bench {

// Collect every set bit of a 5-word mask into out[], return the count.
// out[] must hold MASK_WORDS*64 ids. We size to the full bit width (not
// NUM_ACTIONS) so a stray high bit can never overflow the buffer.
inline uint32_t legal_actions(const uint64_t mask[catan::MASK_WORDS],
                              uint32_t out[catan::MASK_WORDS * 64]) noexcept {
    uint32_t n = 0;
    for (uint32_t w = 0; w < catan::MASK_WORDS; ++w) {
        uint64_t bits = mask[w];
        const uint32_t base = w * 64u;
        while (bits) {
            out[n++] = base + uint32_t(__builtin_ctzll(bits));
            bits &= bits - 1;            // clear lowest set bit
        }
    }
    return n;
}

// Uniformly pick one legal action from the maintained mask. Returns 0 if the
// mask is empty (should never happen — roll/end_turn keep ≥1 legal action).
inline uint32_t pick_random_legal(const uint64_t mask[catan::MASK_WORDS],
                                  catan::Xoshiro128& rng) noexcept {
    uint32_t buf[catan::MASK_WORDS * 64];
    const uint32_t n = legal_actions(mask, buf);
    return n ? buf[rng.bounded(n)] : 0u;
}

}  // namespace bench
