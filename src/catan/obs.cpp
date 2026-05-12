// Observation encoder: GameState -> flat float tensor from current-player POV.
//
// Layout (offsets are documented in comments below; OBS_SIZE in obs.hpp).
//   Per-player block (4 players, in relative seat order: self, +1, +2, +3)
//     16 floats each = 64 floats
//   Self private (resources, dev cards, dev_card_played flag) = 16 floats
//   Board (node, edge, hex, port, robber)
//   Game state (phase, flag, dice, bank, dev_deck, awards, ...)
//   Trade scratch (proposer, give, want, responses)
#include "obs.hpp"
#include "topology.hpp"

namespace catan {

namespace {

inline uint8_t relseat(uint8_t self, uint8_t player) noexcept {
    return uint8_t((player + NUM_PLAYERS - self) & 0x3);
}

struct Writer {
    float* p;
    inline void put(float v) noexcept { *p++ = v; }
    inline void zero(int n) noexcept { for (int i = 0; i < n; ++i) *p++ = 0.0f; }
    // One-hot of size `n`. idx in [0, n) sets that slot to 1; others 0.
    // If idx >= n (e.g., sentinel), all slots stay 0.
    inline void onehot(int idx, int n) noexcept {
        for (int i = 0; i < n; ++i) *p++ = (i == idx) ? 1.0f : 0.0f;
    }
};

// Per-player slot layout (16 floats):
// [vp, handsize, total_dev, knights_played, road_length,
//  settle_left, city_left, road_left,
//  ports[0..5] (6 floats),
//  discard_remaining, is_current]
inline void encode_player(Writer& w, const GameState& s, uint8_t pl,
                           bool is_self) noexcept {
    // VP: total for self (private), public-only for opponents.
    w.put(float(is_self ? s.player_vp[pl] : s.player_vp_without_dev[pl]));
    w.put(float(s.player_handsize[pl]));
    w.put(float(s.player_total_dev[pl]));
    w.put(float(s.player_knights_played[pl]));
    w.put(float(s.player_road_length[pl]));
    w.put(float(s.player_settlement_count[pl]));
    w.put(float(s.player_city_count[pl]));
    w.put(float(s.player_road_count[pl]));
    uint8_t ports = s.player_ports[pl];
    for (int b = 0; b < 6; ++b) w.put(float((ports >> b) & 1));
    w.put(float(s.player_discard_remaining[pl]));
    w.put(float(s.current_player == pl ? 1 : 0));
}

}  // namespace

void write_obs(const GameState& s, const BoardLayout& b,
               uint8_t self, float* out) noexcept {
    Writer w{out};

    // ----- Per-player blocks in relative-seat order: self, +1, +2, +3 -----
    for (uint8_t rel = 0; rel < NUM_PLAYERS; ++rel) {
        uint8_t pl = uint8_t((self + rel) & 0x3);
        encode_player(w, s, pl, /*is_self=*/(rel == 0));
    }

    // ----- Self private -----
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) w.put(float(s.player_resources[self][r]));
    for (uint8_t d = 0; d < 5; ++d)             w.put(float(s.player_dev[self][d]));
    for (uint8_t d = 0; d < 5; ++d)             w.put(float(s.player_dev_bought_this_turn[self][d]));
    w.put(float(s.dev_card_played ? 1 : 0));

    // ----- Board: nodes (4 channels per node) -----
    // [own_settle, own_city, opp_settle, opp_city]
    for (uint8_t n = 0; n < topology::NUM_NODES; ++n) {
        uint8_t nb = s.node[n];
        uint8_t lvl = node_level(nb);
        uint8_t own = node_owner(nb);
        bool is_self_owned = (lvl != NODE_EMPTY) && (own == self);
        bool is_opp_owned  = (lvl != NODE_EMPTY) && (own != self);
        w.put((is_self_owned && lvl == NODE_SETTLEMENT) ? 1.0f : 0.0f);
        w.put((is_self_owned && lvl == NODE_CITY)       ? 1.0f : 0.0f);
        w.put((is_opp_owned  && lvl == NODE_SETTLEMENT) ? 1.0f : 0.0f);
        w.put((is_opp_owned  && lvl == NODE_CITY)       ? 1.0f : 0.0f);
    }

    // ----- Board: edges (2 channels per edge) -----
    for (uint8_t e = 0; e < topology::NUM_EDGES; ++e) {
        uint8_t owner = s.edge[e];
        w.put(owner == self                          ? 1.0f : 0.0f);
        w.put((owner != self && owner != NO_PLAYER)  ? 1.0f : 0.0f);
    }

    // ----- Hex resource (one-hot 6 per hex) -----
    for (uint8_t h = 0; h < topology::NUM_HEXES; ++h) {
        w.onehot(b.hex_resource[h], 6);
    }

    // ----- Hex number (normalized by 12) -----
    for (uint8_t h = 0; h < topology::NUM_HEXES; ++h) {
        w.put(float(b.hex_number[h]) / 12.0f);
    }

    // ----- Port types (one-hot 6 per port) -----
    for (uint8_t pt = 0; pt < topology::NUM_PORTS; ++pt) {
        w.onehot(b.port_type[pt], 6);
    }

    // ----- Robber hex one-hot -----
    w.onehot(int(s.robber_hex), int(topology::NUM_HEXES));

    // ----- Game state -----
    w.onehot(int(s.phase), 4);                    // 4 phase values
    w.onehot(int(s.flag),  8);                    // 8 flag values
    // dice_roll one-hot over 13 slots (0 = not rolled, 2..12 valid)
    w.onehot(int(s.dice_roll), 13);
    w.put(float(s.turn_count) / 400.0f);          // normalized turn count
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) w.put(float(s.bank[r]));
    for (uint8_t d = 0; d < 5; ++d)             w.put(float(s.dev_deck[d]));

    // longest_road_owner: 5 slots [self, +1, +2, +3, none]
    if (s.longest_road_owner == NO_PLAYER) w.onehot(4, 5);
    else                                   w.onehot(int(relseat(self, s.longest_road_owner)), 5);

    // largest_army_owner: same 5-slot encoding
    if (s.largest_army_owner == NO_PLAYER) w.onehot(4, 5);
    else                                   w.onehot(int(relseat(self, s.largest_army_owner)), 5);

    // start_player relative
    w.onehot(int(relseat(self, s.start_player)), 4);

    // free_roads_remaining
    w.put(float(s.free_roads_remaining));

    // ----- Trade scratch -----
    // trade_proposer: 5 slots [self, +1, +2, +3, none]
    if (s.trade_proposer == NO_PLAYER) w.onehot(4, 5);
    else                                w.onehot(int(relseat(self, s.trade_proposer)), 5);

    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) w.put(float(s.trade_give[r]));
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) w.put(float(s.trade_want[r]));

    // Per-opponent response: 4-slot one-hot [PENDING, ACCEPT, DECLINE, N/A]
    for (uint8_t rel = 1; rel < NUM_PLAYERS; ++rel) {
        uint8_t pl = uint8_t((self + rel) & 0x3);
        uint8_t v = uint8_t((s.trade_response >> (2 * pl)) & 0x3);
        w.onehot(int(v), 4);
    }
}

}  // namespace catan
