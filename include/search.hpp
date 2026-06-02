#pragma once
#include <cstdint>
#include "state.hpp"

// Native depth-limited expectimax alpha-beta player, a faithful port of
// Catanatron's catanatron.players.minimax.AlphaBetaPlayer (+ value.base_fn
// heuristic and tree_search_utils chance expansion). Built on the fastCatan
// engine (rules.cpp expand_action) so it is fast enough to use as a training
// opponent — unlike running the reference engine through the eval bridge.
//
// Two documented deviations from Catanatron's chance handling (both more
// correct than the reference, see expand_action in rules.cpp):
//   * BUY_DEV forks the true remaining deck (not the info-set blur).
//   * Robber steal forks the victim's actual hand (not a flat 1/5).

namespace catan {

    // Number of heuristic weights; order matches Catanatron's DEFAULT_WEIGHTS.
    inline constexpr int AB_NUM_WEIGHTS = 13;

    // Catanatron's DEFAULT_WEIGHTS (value.py), in the AbWeight order below.
    extern const double AB_DEFAULT_WEIGHTS[AB_NUM_WEIGHTS];

    // Indices into a weights array (mirrors the DEFAULT_WEIGHTS dict order).
    enum AbWeight : int {
        AB_W_PUBLIC_VPS = 0,
        AB_W_PRODUCTION,
        AB_W_ENEMY_PRODUCTION,
        AB_W_NUM_TILES,
        AB_W_REACH0,
        AB_W_REACH1,
        AB_W_BUILDABLE,
        AB_W_LONGEST_ROAD,
        AB_W_HAND_SYNERGY,
        AB_W_HAND_RESOURCES,
        AB_W_DISCARD_PENALTY,
        AB_W_HAND_DEVS,
        AB_W_ARMY_SIZE,
    };

    // Leaf heuristic: Catanatron base_fn value of state (s,b) from pov's seat.
    // `weights` may be null to use AB_DEFAULT_WEIGHTS, else an AB_NUM_WEIGHTS
    // array. Pure function of the state (no search, no RNG).
    double ab_value(const GameState& s, const BoardLayout& b, uint8_t pov,
                    const double* weights) noexcept;

    // Pick pov's best action via depth-limited expectimax alpha-beta.
    //   depth  : search depth (Catanatron default 2).
    //   prune  : if true, apply list_prunned_actions (initial 1-tile settles,
    //            most-impactful robber); if false, search all legal actions
    //            (Catanatron AlphaBetaPlayer default).
    //   weights: null -> AB_DEFAULT_WEIGHTS.
    // Returns the chosen flat action ID, or 0xFFFFFFFF if no legal action
    // exists (terminal / not pov's decision).
    uint32_t ab_decide(const GameState& s, const BoardLayout& b, uint8_t pov,
                       int depth, bool prune, const double* weights) noexcept;

}  // namespace catan
