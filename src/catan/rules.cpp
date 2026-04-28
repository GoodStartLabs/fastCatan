#include "rules.hpp"
#include "topology.hpp"

#include <cstring>
#include <utility>

namespace catan {

namespace {

// 19 hex tiles: 3 brick, 4 lumber, 4 wool, 4 grain, 3 ore, 1 desert.
// Resource codes match BoardLayout::hex_resource semantics in state.hpp.
constexpr uint8_t HEX_RESOURCE_BAG[19] = {
    0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4,
    5,
};

// 18 number tokens placed on non-desert hexes. Skips 7 (the robber roll).
constexpr uint8_t NUMBER_BAG[18] = {
    2,
    3, 3,
    4, 4,
    5, 5,
    6, 6,
    8, 8,
    9, 9,
    10, 10,
    11, 11,
    12,
};

// 9 port types: one 2:1 per resource (codes 0..4) plus four 3:1 generic (5).
constexpr uint8_t PORT_BAG[9] = { 0, 1, 2, 3, 4, 5, 5, 5, 5 };

// Initial dev deck: 14 knight, 5 VP, 2 road building, 2 year of plenty, 2 monopoly.
constexpr uint8_t INITIAL_DEV_DECK[5] = { 14, 5, 2, 2, 2 };

constexpr uint8_t DESERT_RESOURCE = 5;

inline void shuffle(uint8_t* arr, std::size_t n, Xoshiro128& r) noexcept {
    for (std::size_t i = n - 1; i > 0; --i) {
        uint32_t j = r.bounded(uint32_t(i + 1));
        std::swap(arr[i], arr[j]);
    }
}

// True iff some 6 or 8 hex is adjacent to another 6 or 8.
// Standard Catan rule: red numbers (the high-pip 6 and 8) cannot touch.
inline bool red_numbers_touch(const BoardLayout& b) noexcept {
    for (uint8_t h = 0; h < topology::NUM_HEXES; ++h) {
        uint8_t n = b.hex_number[h];
        if (n != 6 && n != 8) continue;
        for (uint8_t k = 0; k < topology::MAX_NEIGHBORS_PER_HEX; ++k) {
            uint8_t nb = topology::hex_to_hex[h][k];
            if (nb == topology::NO_HEX) break;
            uint8_t nn = b.hex_number[nb];
            if (nn == 6 || nn == 8) return true;
        }
    }
    return false;
}

// Place number tokens on non-desert hexes. Reshuffle until no 6 touches a 6/8
// and no 8 touches a 6/8. Capped retry; in practice converges in <10 tries.
inline void place_numbers(BoardLayout& b, Xoshiro128& r) noexcept {
    uint8_t bag[18];
    for (int attempt = 0; attempt < 256; ++attempt) {
        std::memcpy(bag, NUMBER_BAG, sizeof(NUMBER_BAG));
        shuffle(bag, 18, r);

        std::size_t bi = 0;
        for (uint8_t h = 0; h < topology::NUM_HEXES; ++h) {
            if (b.hex_resource[h] == DESERT_RESOURCE) {
                b.hex_number[h] = 0;
            } else {
                b.hex_number[h] = bag[bi++];
            }
        }

        if (!red_numbers_touch(b)) return;
    }
    // Pathological seed: bail out with last attempt. Rules tests should
    // never observe this since the rejection rate is ~few percent at most.
}

inline uint8_t find_desert(const BoardLayout& b) noexcept {
    for (uint8_t h = 0; h < topology::NUM_HEXES; ++h) {
        if (b.hex_resource[h] == DESERT_RESOURCE) return h;
    }
    return 0;  // unreachable: bag always contains exactly one desert.
}

}  // namespace

void reset_one(GameState& s, BoardLayout& b, uint64_t seed) noexcept {
    // Zero everything; sets every counter / flag / array entry to its
    // "fresh game" value except the few exceptions handled explicitly below.
    std::memset(&s, 0, sizeof(s));
    std::memset(&b, 0, sizeof(b));

    // RNG comes online before any randomized field is touched.
    xoshiro_seed(s.rng, seed);

    // ----- BoardLayout -----
    b.port_layout = uint8_t(s.rng.bounded(2));

    std::memcpy(b.port_type, PORT_BAG, sizeof(PORT_BAG));
    shuffle(b.port_type, 9, s.rng);

    std::memcpy(b.hex_resource, HEX_RESOURCE_BAG, sizeof(HEX_RESOURCE_BAG));
    shuffle(b.hex_resource, 19, s.rng);

    place_numbers(b, s.rng);

    // ----- GameState -----
    // Edges use NO_PLAYER (0xFF) as the empty marker; memset(0) above is wrong for them.
    std::memset(s.edge, NO_PLAYER, sizeof(s.edge));

    // Robber starts on the desert.
    s.robber_hex = find_desert(b);

    s.phase = Phase::INITIAL_PLACEMENT_1;
    s.flag  = Flag::NONE;
    s.start_player   = uint8_t(s.rng.bounded(4));
    s.current_player = s.start_player;

    // Per-player initial pieces.
    for (uint8_t p = 0; p < 4; ++p) {
        s.player_settlement_count[p] = 5;
        s.player_city_count[p]       = 4;
        s.player_road_count[p]       = 15;
    }

    // Awards have no holder yet.
    s.longest_road_owner = NO_PLAYER;
    s.largest_army_owner = NO_PLAYER;

    // Bank starts with 19 of each resource.
    for (uint8_t r = 0; r < 5; ++r) s.bank[r] = 19;

    // Dev deck stocked.
    std::memcpy(s.dev_deck, INITIAL_DEV_DECK, sizeof(INITIAL_DEV_DECK));
}

}  // namespace catan
