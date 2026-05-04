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

    // Recompute and apply award titles (longest road + largest army) given
    // current node/edge/knight state. Test-only entry — normally driven
    // implicitly by step_one after relevant actions.
    void recompute_awards(GameState& s) noexcept;

    // Refresh the GameState::action_mask field by full recompute. Call
    // after any direct state mutation that bypasses step_one (e.g. test
    // setters that poke resources or pieces). step_one + reset_one already
    // refresh internally.
    void refresh_mask(GameState& s, const BoardLayout& b) noexcept;

    // Initialize an episode using a CALLER-PROVIDED BoardLayout. Skips the
    // hex/port randomization that reset_one performs. RNG is seeded from
    // `seed`; start_player is set from `start_player_override` if < 4,
    // otherwise drawn from the RNG. Used by the differential test to
    // share board state with another simulator (e.g. Catanatron).
    void reset_with_layout(GameState& s, const BoardLayout& b,
                            uint64_t seed, uint8_t start_player_override = 0xFF) noexcept;

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
        inline constexpr uint32_t BUY_DEV              = 235;
        inline constexpr uint32_t PLAY_KNIGHT          = 236;
        inline constexpr uint32_t PLAY_ROAD_BUILDING   = 237;
        inline constexpr uint32_t PLAY_YEAR_OF_PLENTY  = 238;  // [238, 263) base + give1*5 + give2
        inline constexpr uint32_t YOP_END              = 263;  // exclusive
        inline constexpr uint32_t PLAY_MONOPOLY        = 263;  // [263, 268) base + resource
        inline constexpr uint32_t MONOPOLY_END         = 268;  // exclusive

        // Player-to-player trade (compositional). Compose only valid post-roll
        // with no active flag; OPEN/ACCEPT/DECLINE/CONFIRM/CANCEL drive the flow.
        inline constexpr uint32_t TRADE_ADD_GIVE_BASE    = 268;  // [268, 273) +1 of give resource r
        inline constexpr uint32_t TRADE_REMOVE_GIVE_BASE = 273;  // [273, 278) -1 of give resource r
        inline constexpr uint32_t TRADE_ADD_WANT_BASE    = 278;  // [278, 283) +1 of want resource r
        inline constexpr uint32_t TRADE_REMOVE_WANT_BASE = 283;  // [283, 288) -1 of want resource r
        inline constexpr uint32_t TRADE_OPEN             = 288;  // finalize proposal, broadcast to opponents
        inline constexpr uint32_t TRADE_ACCEPT           = 289;  // responder accepts the open proposal
        inline constexpr uint32_t TRADE_DECLINE          = 290;  // responder declines
        inline constexpr uint32_t TRADE_CONFIRM_BASE     = 291;  // [291, 295) proposer picks accepting partner p
        inline constexpr uint32_t TRADE_CANCEL           = 295;  // proposer cancels (compose or post-response)
    }

}  // namespace catan
