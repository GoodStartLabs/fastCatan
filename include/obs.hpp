#pragma once
#include <cstdint>
#include "state.hpp"
#include "topology.hpp"

namespace catan {

    // Per-player slot count (used both for self and opponents in the obs).
    // [vp, handsize, total_dev, knights_played, road_length,
    //  settle_left, city_left, road_left,
    //  ports(6), discard_remaining, is_current]
    inline constexpr uint32_t OBS_PER_PLAYER = 16;

    // Self-only private fields (in addition to the per-player block).
    // [resources(5), dev_playable(5), dev_bought_pending(5), dev_card_played]
    inline constexpr uint32_t OBS_SELF_PRIVATE = 16;

    // Board (static+dynamic).
    // node ownership 8ch × 54 (per-player settle/city in relseat order),
    // edge ownership 4ch × 72 (per-player road in relseat order),
    // hex_resource one-hot 6ch × 19, hex_number/12 × 19,
    // port_type one-hot 6ch × 9, robber one-hot × 19
    inline constexpr uint32_t OBS_BOARD =
        8 * topology::NUM_NODES +
        4 * topology::NUM_EDGES +
        6 * topology::NUM_HEXES + topology::NUM_HEXES +
        6 * topology::NUM_PORTS +
        topology::NUM_HEXES;

    // Game-state fields. phase(4) flag(8) dice_roll(13) turn(1) bank(5)
    // dev_deck(5) longest(5) army(5) start_player(4) free_roads(1)
    inline constexpr uint32_t OBS_GAME = 4 + 8 + 13 + 1 + 5 + 5 + 5 + 5 + 4 + 1;

    // Trade scratch fields. proposer(5) give(5) want(5) response(3*4)
    inline constexpr uint32_t OBS_TRADE = 5 + 5 + 5 + 3 * 4;

    inline constexpr uint32_t OBS_SIZE =
        4 * OBS_PER_PLAYER + OBS_SELF_PRIVATE + OBS_BOARD + OBS_GAME + OBS_TRADE;

    // Full-state appendix: the hidden enemy state the POV obs masks out.
    // Per opponent (relseat +1, +2, +3), 16 floats:
    // [resources(5), dev_playable(5), dev_bought_pending(5), hidden_dev_vp(1)]
    // Consumers: the learned JUDGE (leaf value) only — the same information
    // ab_value reads at leaves. POV policy nets keep the OBS_SIZE prefix.
    inline constexpr uint32_t OBS_FULL_APPENDIX = 3 * 16;
    inline constexpr uint32_t OBS_FULL_SIZE = OBS_SIZE + OBS_FULL_APPENDIX;

    // Encode the env state into a flat float tensor from `player_pov`'s
    // perspective. Count fields are normalized by structural Catan maxima
    // (see namespace norm in obs.cpp; mirrored in bridge/obs_encoder.py);
    // booleans/one-hot as 0.0 / 1.0. The encoding is fixed and stable;
    // changes here require RL agents to retrain.
    void write_obs(const GameState& s, const BoardLayout& b,
                   uint8_t player_pov, float* out) noexcept;

    // OBS_FULL_SIZE variant: byte-identical OBS_SIZE prefix (write_obs), then
    // the full-state appendix. Same normalization constants as the prefix.
    void write_obs_full(const GameState& s, const BoardLayout& b,
                        uint8_t player_pov, float* out) noexcept;

}  // namespace catan
