#pragma once
#include <cstdint>
#include "state.hpp"

namespace catan {

    // Initialize one episode. Randomizes BoardLayout (port pattern, port
    // types, hex resources, hex numbers with 6/8-adjacency rejection),
    // seeds the per-env RNG from `seed`, and resets GameState to the
    // start of INITIAL_PLACEMENT_1.
    //
    // Same `seed` always produces identical (BoardLayout, GameState).
    void reset_one(GameState& s, BoardLayout& b, uint64_t seed) noexcept;

    // Advance one env by one action. Implementation lands in M1 scope
    // step-by-step; signature stable.
    //
    // void step_one(GameState& s, const BoardLayout& b,
    //               uint32_t action, float& reward, uint8_t& done) noexcept;

}  // namespace catan
