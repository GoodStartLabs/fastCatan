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

    // Advance one env by one action. Illegal actions are no-ops (state
    // unchanged, reward 0, done unchanged). Caller is responsible for
    // passing only mask-legal actions in production; the safety net is
    // for fuzz tests and partial mask correctness during development.
    //
    // M1 slice 1 implements: INITIAL_PLACEMENT_1 / _2 (settlement + road).
    // Other phases land in subsequent slices.
    void step_one(GameState& s, const BoardLayout& b,
                  uint32_t action, float& reward, uint8_t& done) noexcept;

    // Action ID layout (flat encoding; PLAN.md §Action Space).
    // Slice 1 only uses the build ranges.
    namespace action {
        inline constexpr uint32_t SETTLE_BASE      = 0;    // [0, 54)
        inline constexpr uint32_t CITY_BASE        = 54;   // [54, 108)
        inline constexpr uint32_t ROAD_BASE        = 108;  // [108, 180)
        inline constexpr uint32_t ROLL_DICE        = 180;
        inline constexpr uint32_t END_TURN         = 181;
        inline constexpr uint32_t DISCARD_BASE     = 182;  // [182, 187) discard 1 of resource r
        inline constexpr uint32_t MOVE_ROBBER_BASE = 187;  // [187, 206) move robber to hex h
        inline constexpr uint32_t STEAL_BASE       = 206;  // [206, 210) steal from player p
        inline constexpr uint32_t TRADE_BASE       = 210;  // [210, 235) bank/port trade: give*5 + get
        inline constexpr uint32_t TRADE_END        = 235;  // exclusive upper bound
        inline constexpr uint32_t BUY_DEV          = 235;
        inline constexpr uint32_t PLAY_KNIGHT      = 236;
        // M2 reserves 237..240 for VP/RoadBuilding/YearOfPlenty/Monopoly play actions.
    }

}  // namespace catan
