#pragma once
#include <cstdint>
#include <type_traits>
#include "rng.hpp"

namespace catan {

    // player-id fields when no player owns the slot
    inline constexpr uint8_t NO_PLAYER = 0xFF;

    // Game-wide constants.
    inline constexpr uint8_t NUM_RESOURCES = 5;
    inline constexpr uint8_t NUM_PLAYERS   = 4;
    inline constexpr uint8_t WIN_VP        = 10;

    // Length backstop: end any game that reaches MAX_TURNS turns with no winner
    // (turn_count is uint16_t, ++ per END_TURN at rules.cpp). This is the single
    // length authority for the game; the per-turn trade-compose cap (Python
    // ComposeCapper / FastCatanEnv) is the liveness guard that guarantees turns
    // end, so this always fires for a non-terminating game. 2000 ~= 2x the
    // observed random-game max of 945 turns -> ample headroom for longer
    // strong-vs-strong self-play, so it only bites genuine non-termination.
    inline constexpr uint16_t MAX_TURNS    = 2000;

    // node[] encoding: bits 0-1 = level, bits 2-4 = owner.
    inline constexpr uint8_t NODE_EMPTY      = 0;
    inline constexpr uint8_t NODE_SETTLEMENT = 1;
    inline constexpr uint8_t NODE_CITY       = 2;

    inline constexpr uint8_t node_level(uint8_t n) noexcept { return n & 0x03; }
    inline constexpr uint8_t node_owner(uint8_t n) noexcept { return (n >> 2) & 0x03; }
    inline constexpr uint8_t node_pack(uint8_t lvl, uint8_t owner) noexcept {
        return uint8_t((lvl & 0x03) | ((owner & 0x03) << 2));
    }

    // ---------------------------------------------------------------------
    // Phase: top-level game phase.
    // ---------------------------------------------------------------------
    enum class Phase : uint8_t {
        INITIAL_PLACEMENT_1 = 0,
        INITIAL_PLACEMENT_2,
        MAIN,
        ENDED,
    };

    // ---------------------------------------------------------------------
    // Flag: forced-action override. NONE during normal play; set when a
    // player (or all players) must resolve a specific action before the
    // game can continue.
    // ---------------------------------------------------------------------
    enum class Flag : uint8_t {
        NONE = 0,
        DISCARD_RESOURCES, // players with >7 cards must discard half (after a 7 roll)
        MOVE_ROBBER,       // current player must move robber
        ROBBER_STEAL,      // current player must pick victim to steal from
        YEAR_OF_PLENTY,    // current player picks 2 resources from bank
        MONOPOLY,          // current player picks a resource type
        PLACE_ROAD,        // road building dev card: place up to 2 free roads
        TRADE_PENDING,     // current player proposed trade, awaiting responses
    };

    // ---------------------------------------------------------------------
    // BoardLayout: static after random initialisation at episode start.
    // Topology / adjacency lives in topology.hpp.
    // ---------------------------------------------------------------------
    struct BoardLayout {
        uint8_t hex_resource[19]; // 0=brick 1=lumber 2=wool 3=grain 4=ore 5=desert
        uint8_t hex_number[19];   // 0 for desert; else 2..12 skipping 7
        uint8_t port_type[9];     // 0..4 = 2:1 specific resource, 5 = 3:1 generic
    };

    // ---------------------------------------------------------------------
    // GameState: dynamic per-step state. 64-byte aligned, 256 bytes total
    // ---------------------------------------------------------------------
    struct alignas(64) GameState {
        // --- Board state ---
        uint8_t node[54];  
        uint8_t edge[72];
        uint8_t robber_hex;

        // --- Turn / phase state ---
        uint8_t  dice_roll;    // 0 if not yet rolled this turn, else 2..12
        uint16_t turn_count;
        Phase    phase;        // INITIAL_PLACEMENT_1 / _2 / MAIN / ENDED
        Flag     flag;         // forced-action override; NONE during normal play
        uint8_t  start_player; // who began initial placement; phase 1 clockwise from here, phase 2 counter-clockwise back to here

        // --- Per-player private state (index 0..3) ---
        uint8_t player_resources[4][5];            // brick, lumber, wool, grain, ore
        uint8_t player_dev[4][5];                  // dev cards: knight, VP, road building, year of plenty, monopoly
        uint8_t player_dev_bought_this_turn[4][5]; // one-turn cooldown on newly bought dev cards (VP exempt)
        uint8_t player_vp[4];                      // total victory points
        uint8_t player_ports[4];                   // port access bitmask: bit 0=brick, 1=lumber, 2=wool, 3=grain, 4=ore, 5=3:1 generic; bits 6-7 unused

        // --- Per-player counters ---
        uint8_t player_knights_played[4];   // contributes to largest army
        uint8_t player_road_length[4];      // longest contiguous road
        uint8_t player_settlement_count[4]; // settlements left to place (starts at 5)
        uint8_t player_city_count[4];       // cities left to place      (starts at 4)
        uint8_t player_road_count[4];       // roads left to place       (starts at 15)

        // --- Awards / turn flags ---
        uint8_t longest_road_owner; // player id, or sentinel if none
        uint8_t largest_army_owner; // player id, or sentinel if none
        bool    dev_card_played;    // current player already played a dev card this turn
        uint8_t current_player;     // whose turn it is; preserved across sub-phases
        uint8_t discarding_player;  // active discarder during DISCARD_RESOURCES sub-phase
        uint8_t free_roads_remaining; // 0..2 during PLACE_ROAD sub-phase from Road Building

        // --- Per-player public state (visible to all players) ---
        uint8_t player_handsize[4];        // total resource cards held
        uint8_t player_total_dev[4];       // total hidden dev cards held
        uint8_t player_vp_without_dev[4];  // public VP = total VP minus hidden dev-card VPs
        uint8_t player_discard_remaining[4]; // cards still owed for DISCARD sub-phase; 0 if none

        // --- Bank ---
        uint8_t bank[5];     // resource supply: brick, lumber, wool, grain, ore
        uint8_t dev_deck[5]; // dev cards remaining: knight, VP, road building, year of plenty, monopoly

        // --- Player-to-player trade scratch (compose -> respond -> confirm) ---
        uint8_t trade_give[5];   // proposer's offered bundle, accumulated via TRADE_ADD_GIVE
        uint8_t trade_want[5];   // proposer's requested bundle
        uint8_t trade_response;  // 2 bits per player (LSB-first): 0=PENDING 1=ACCEPT 2=DECLINE 3=N/A(proposer)
        uint8_t trade_proposer;  // player id who opened the proposal; NO_PLAYER outside trade

        // --- RNG (per-env xoshiro128++ state) ---
        Xoshiro128 rng;      // 16 B; seeded in reset_one, advances on dice/dev/steal

        // --- Incrementally maintained legal-action mask (320 bits over 5 words) ---
        // Updated after every step_one. Consumers read this directly instead
        // of paying for a full recompute. PLAN.md M3 deliverable.
        uint64_t action_mask[5];

        // --- Longest-road component membership (one 54-bit set per player) ---
        // Bit n set => node n is a valid longest-road path endpoint for the
        // player. A node joins the player's set when the player builds an
        // incident road (or settlement) while the node is NOT enemy-occupied,
        // and is never removed thereafter. This mirrors catanatron's
        // connected-component bookkeeping (board.py build_road/build_settlement)
        // and is what makes a road segment terminating at an opponent building
        // count toward longest road IFF the road was built before the opponent
        // settled there. Without it the longest-road length diverges from
        // catanatron (the ground-truth engine) in ~65% of random games.
        uint64_t road_node_member[4];
    };

    // Just for Checking.
    static_assert(std::is_trivially_copyable_v<GameState>,
                  "GameState must be trivially copyable for memcpy cloning");
    static_assert(sizeof(GameState) == 384,
                 "GameState must be exactly 6 cache lines; update if layout changes intentionally");
    static_assert(std::is_trivially_copyable_v<BoardLayout>,
                  "BoardLayout must be trivially copyable");

}