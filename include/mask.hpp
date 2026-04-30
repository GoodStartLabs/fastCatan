#pragma once
#include <cstdint>
#include "state.hpp"
#include "rules.hpp"

namespace catan {

    // Flat action space upper bound (exclusive). Matches action IDs in rules.hpp.
    // [0,54)   settlement build at node
    // [54,108) city build at node
    // [108,180) road build at edge
    // 180      ROLL_DICE
    // 181      END_TURN
    // [182,187) DISCARD by resource
    // [187,206) MOVE_ROBBER by hex
    // [206,210) STEAL by player
    // [210,235) TRADE (give*5 + get)
    // 235      BUY_DEV
    // 236      PLAY_KNIGHT
    inline constexpr uint32_t NUM_ACTIONS = 237;

    // 5 × uint64 = 320 bits. Bit i corresponds to action ID i.
    inline constexpr uint32_t MASK_WORDS = 5;

    // Recompute the legal-action bitmask from the current state.
    // M1: full recompute every call. M3 will replace this with an
    // incrementally maintained version, gated on the same predicate set.
    void compute_mask(const GameState& s, const BoardLayout& b,
                      uint64_t mask[MASK_WORDS]) noexcept;

}  // namespace catan
