#include "rules.hpp"
#include "topology.hpp"
#include "mask.hpp"   // for catan::compute_mask declaration (debug assert in step_one)

#include <cassert>
#include <cstring>
#include <utility>

namespace catan {

// Forward decl: full-recompute mask body lives at the bottom of the file
// (after all sub-phase helpers). Both reset_one and step_one call this
// to refresh GameState::action_mask after any state mutation.
namespace { inline void refresh_action_mask(GameState& s, const BoardLayout& b) noexcept; }

// Forward decl for surgical update covering ONLY the trade-compose bit
// range [268, 296). Used after ADD/REMOVE GIVE/WANT compose actions —
// those don't touch flag/dice/resources/bank, so other mask bits are
// unaffected and we can skip the full recompute.
namespace { inline void refresh_compose_mask_bits(GameState& s, const BoardLayout& b) noexcept; }

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

    // Trade scratch: no proposer until TRADE_OPEN.
    s.trade_proposer = NO_PLAYER;

    // Bank starts with 19 of each resource.
    for (uint8_t r = 0; r < 5; ++r) s.bank[r] = 19;

    // Dev deck stocked.
    std::memcpy(s.dev_deck, INITIAL_DEV_DECK, sizeof(INITIAL_DEV_DECK));

    // Initialize incremental action_mask field.
    refresh_action_mask(s, b);
}

// =====================================================================
// step_one — slice 1: initial placement (phases _1 and _2)
// =====================================================================

namespace {

// In initial placement, current player must place settlement first, then
// road. Derive sub-step from settlement_count (no extra state needed):
//   _1: count==5 -> settlement; count==4 -> road
//   _2: count==4 -> settlement; count==3 -> road
inline bool need_settlement(const GameState& s) noexcept {
    uint8_t cnt = s.player_settlement_count[s.current_player];
    return (s.phase == Phase::INITIAL_PLACEMENT_1) ? (cnt == 5) : (cnt == 4);
}

// Distance rule: node empty + all neighbor nodes empty.
inline bool can_settle_initial(const GameState& s, uint8_t node_id) noexcept {
    if (node_level(s.node[node_id]) != NODE_EMPTY) return false;
    for (uint8_t k = 0; k < topology::MAX_EDGES_PER_NODE; ++k) {
        uint8_t nb = topology::node_to_node[node_id][k];
        if (nb == topology::NO_NODE) break;
        if (node_level(s.node[nb]) != NODE_EMPTY) return false;
    }
    return true;
}

inline bool any_player_road_at(const GameState& s, uint8_t node_id, uint8_t player) noexcept {
    for (uint8_t k = 0; k < topology::MAX_EDGES_PER_NODE; ++k) {
        uint8_t e = topology::node_to_edge[node_id][k];
        if (e == topology::NO_EDGE) break;
        if (s.edge[e] == player) return true;
    }
    return false;
}

// In _2, the just-placed (second) settlement is the only player-owned
// settlement that has no incident player road yet.
inline uint8_t find_unroaded_settlement(const GameState& s, uint8_t player) noexcept {
    for (uint8_t n = 0; n < topology::NUM_NODES; ++n) {
        uint8_t b = s.node[n];
        if (node_level(b) != NODE_SETTLEMENT) continue;
        if (node_owner(b) != player) continue;
        if (!any_player_road_at(s, n, player)) return n;
    }
    return topology::NO_NODE;
}

inline bool can_road_initial(const GameState& s, uint8_t edge_id, uint8_t player) noexcept {
    if (s.edge[edge_id] != NO_PLAYER) return false;
    uint8_t n0 = topology::edge_to_node[edge_id][0];
    uint8_t n1 = topology::edge_to_node[edge_id][1];

    if (s.phase == Phase::INITIAL_PLACEMENT_1) {
        // Road incident to player's only settlement so far.
        auto owns_settle = [&](uint8_t n) {
            return node_level(s.node[n]) == NODE_SETTLEMENT
                && node_owner(s.node[n]) == player;
        };
        return owns_settle(n0) || owns_settle(n1);
    }
    // _2: road must touch the freshly placed second settlement.
    uint8_t target = find_unroaded_settlement(s, player);
    if (target == topology::NO_NODE) return false;
    return n0 == target || n1 == target;
}

inline void grant_port(GameState& s, const BoardLayout& b,
                        uint8_t node_id, uint8_t player) noexcept {
    const auto& tbl = (b.port_layout == 0)
        ? topology::node_to_port_A
        : topology::node_to_port_B;
    uint8_t slot = tbl[node_id];
    if (slot == topology::NO_PORT) return;
    uint8_t ptype = b.port_type[slot];   // 0..4 specific, 5 generic
    s.player_ports[player] |= uint8_t(1u << ptype);
}

// Phase _2 second settlement grants 1 of each adjacent non-desert resource.
inline void payout_second_placement(GameState& s, const BoardLayout& b,
                                     uint8_t node_id, uint8_t player) noexcept {
    for (uint8_t k = 0; k < topology::MAX_HEXES_PER_NODE; ++k) {
        uint8_t h = topology::node_to_hex[node_id][k];
        if (h == topology::NO_HEX) break;
        uint8_t r = b.hex_resource[h];
        if (r == 5) continue;  // desert pays nothing
        s.player_resources[player][r] += 1;
        s.bank[r] -= 1;
        s.player_handsize[player] += 1;
    }
}

inline void place_settlement(GameState& s, const BoardLayout& b,
                              uint8_t node_id, uint8_t player, bool payout) noexcept {
    s.node[node_id] = node_pack(NODE_SETTLEMENT, player);
    s.player_settlement_count[player] -= 1;
    s.player_vp[player] += 1;
    s.player_vp_without_dev[player] += 1;
    grant_port(s, b, node_id, player);
    if (payout) payout_second_placement(s, b, node_id, player);
}

inline void place_road(GameState& s, uint8_t edge_id, uint8_t player) noexcept {
    s.edge[edge_id] = player;
    s.player_road_count[player] -= 1;
}

// After placing road, advance to next initial-placement turn or transition phase.
//   _1: clockwise start_player ... start_player+3, then snake to _2 same player.
//   _2: counter-clockwise back to start_player, then transition to MAIN.
inline void advance_initial_turn(GameState& s) noexcept {
    uint8_t last_in_p1 = uint8_t((s.start_player + 3) & 0x03);

    if (s.phase == Phase::INITIAL_PLACEMENT_1) {
        if (s.current_player == last_in_p1) {
            s.phase = Phase::INITIAL_PLACEMENT_2;
            // current_player stays — same player gets immediate second placement
        } else {
            s.current_player = uint8_t((s.current_player + 1) & 0x03);
        }
    } else {  // INITIAL_PLACEMENT_2
        if (s.current_player == s.start_player) {
            s.phase = Phase::MAIN;
            // current_player stays at start_player; he rolls first MAIN turn
        } else {
            s.current_player = uint8_t((s.current_player + 3) & 0x03);  // -1 mod 4
        }
    }
}

inline void handle_initial_placement(GameState& s, const BoardLayout& b,
                                      uint32_t action) noexcept {
    if (need_settlement(s)) {
        if (action >= action::SETTLE_BASE + topology::NUM_NODES) return;
        uint8_t node_id = uint8_t(action - action::SETTLE_BASE);
        if (!can_settle_initial(s, node_id)) return;
        place_settlement(s, b, node_id, s.current_player,
                         /*payout=*/(s.phase == Phase::INITIAL_PLACEMENT_2));
    } else {
        if (action < action::ROAD_BASE
            || action >= action::ROAD_BASE + topology::NUM_EDGES) return;
        uint8_t edge_id = uint8_t(action - action::ROAD_BASE);
        if (!can_road_initial(s, edge_id, s.current_player)) return;
        place_road(s, edge_id, s.current_player);
        advance_initial_turn(s);
    }
}

}  // namespace

// =====================================================================
// step_one — slice 2: roll dice + production payout + end turn
// =====================================================================

namespace {

constexpr uint8_t NUM_RESOURCES = 5;
constexpr uint8_t NUM_PLAYERS = 4;
constexpr uint8_t WIN_VP = 10;

// Forward decls — defined later in this anon namespace.
inline void check_game_ended(GameState& s) noexcept;
inline void check_longest_road(GameState& s) noexcept;
inline void clear_trade_scratch(GameState& s) noexcept;

// Build costs, indexed [brick, lumber, wool, grain, ore].
constexpr uint8_t COST_SETTLEMENT[5] = { 1, 1, 1, 1, 0 };
constexpr uint8_t COST_CITY[5]       = { 0, 0, 0, 2, 3 };
constexpr uint8_t COST_ROAD[5]       = { 1, 1, 0, 0, 0 };

inline bool can_pay(const GameState& s, uint8_t player, const uint8_t cost[5]) noexcept {
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) {
        if (s.player_resources[player][r] < cost[r]) return false;
    }
    return true;
}

inline void pay_to_bank(GameState& s, uint8_t player, const uint8_t cost[5]) noexcept {
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) {
        s.player_resources[player][r] -= cost[r];
        s.bank[r]                     += cost[r];
        s.player_handsize[player]     -= cost[r];
    }
}

// MAIN-phase road legality. Edge must be empty and connect to player's
// existing pieces at one endpoint without crossing an opponent's
// settlement/city at that endpoint.
inline bool road_connects(const GameState& s, uint8_t edge_id, uint8_t player) noexcept {
    for (uint8_t side = 0; side < 2; ++side) {
        uint8_t v = topology::edge_to_node[edge_id][side];
        uint8_t n = s.node[v];
        uint8_t lvl = node_level(n);

        // Opponent settlement/city at this endpoint blocks the chain.
        if (lvl != NODE_EMPTY && node_owner(n) != player) continue;

        // Own settlement/city at v -> connects.
        if (lvl != NODE_EMPTY && node_owner(n) == player) return true;

        // Otherwise: a player road meeting at v (other than this edge).
        for (uint8_t k = 0; k < topology::MAX_EDGES_PER_NODE; ++k) {
            uint8_t e2 = topology::node_to_edge[v][k];
            if (e2 == topology::NO_EDGE) break;
            if (e2 == edge_id) continue;
            if (s.edge[e2] == player) return true;
        }
    }
    return false;
}

inline void handle_build_settle(GameState& s, const BoardLayout& b, uint32_t action) noexcept {
    if (action - action::SETTLE_BASE >= topology::NUM_NODES) return;
    uint8_t pl = s.current_player;
    uint8_t node_id = uint8_t(action - action::SETTLE_BASE);

    if (s.player_settlement_count[pl] == 0)        return;
    if (!can_settle_initial(s, node_id))           return;  // distance rule
    if (!any_player_road_at(s, node_id, pl))       return;  // road adjacency
    if (!can_pay(s, pl, COST_SETTLEMENT))          return;

    pay_to_bank(s, pl, COST_SETTLEMENT);
    place_settlement(s, b, node_id, pl, /*payout=*/false);
    // Settlement might split an opponent's road graph at this node.
    check_longest_road(s);
    check_game_ended(s);
}

inline void handle_build_city(GameState& s, uint32_t action) noexcept {
    if (action - action::CITY_BASE >= topology::NUM_NODES) return;
    uint8_t pl = s.current_player;
    uint8_t node_id = uint8_t(action - action::CITY_BASE);

    uint8_t n = s.node[node_id];
    if (node_level(n) != NODE_SETTLEMENT)  return;
    if (node_owner(n) != pl)                return;
    if (s.player_city_count[pl] == 0)      return;
    if (!can_pay(s, pl, COST_CITY))         return;

    pay_to_bank(s, pl, COST_CITY);
    s.node[node_id] = node_pack(NODE_CITY, pl);
    s.player_city_count[pl]       -= 1;
    s.player_settlement_count[pl] += 1;  // settlement piece returns to player's stock
    s.player_vp[pl]               += 1;
    s.player_vp_without_dev[pl]   += 1;
    check_game_ended(s);
}

inline void handle_build_road(GameState& s, uint32_t action) noexcept {
    if (action - action::ROAD_BASE >= topology::NUM_EDGES) return;
    uint8_t pl = s.current_player;
    uint8_t edge_id = uint8_t(action - action::ROAD_BASE);

    if (s.player_road_count[pl] == 0)   return;
    if (s.edge[edge_id] != NO_PLAYER)   return;
    if (!road_connects(s, edge_id, pl)) return;
    if (!can_pay(s, pl, COST_ROAD))     return;

    pay_to_bank(s, pl, COST_ROAD);
    place_road(s, edge_id, pl);
    check_longest_road(s);
    check_game_ended(s);
}

inline void check_game_ended(GameState& s) noexcept {
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        if (s.player_vp[p] >= WIN_VP) {
            s.phase = Phase::ENDED;
            return;
        }
    }
}

// Roll 2d6, set dice_roll, run production payout (or trigger 7-handling).
inline void handle_roll_dice(GameState& s, const BoardLayout& b) noexcept {
    if (s.dice_roll != 0) return;  // already rolled this turn

    uint32_t pair = s.rng.bounded(36);
    uint8_t d1 = uint8_t(pair / 6) + 1;
    uint8_t d2 = uint8_t(pair % 6) + 1;
    s.dice_roll = uint8_t(d1 + d2);

    if (s.dice_roll == 7) {
        // Snapshot pre-discard state. Each player with >7 cards must
        // discard floor(handsize/2) cards.
        s.rolling_player = s.current_player;

        bool anyone_over = false;
        for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
            uint8_t hs = s.player_handsize[p];
            s.player_discard_remaining[p] = (hs > 7) ? uint8_t(hs / 2) : uint8_t(0);
            if (hs > 7) anyone_over = true;
        }

        if (anyone_over) {
            // Hand control to first discarder, clockwise from rolling player.
            for (uint8_t i = 0; i < NUM_PLAYERS; ++i) {
                uint8_t p = uint8_t((s.rolling_player + i) & 0x03);
                if (s.player_discard_remaining[p] > 0) {
                    s.current_player = p;
                    break;
                }
            }
            s.flag = Flag::DISCARD_RESOURCES;
        } else {
            s.flag = Flag::MOVE_ROBBER;
        }
        return;
    }

    // Compute demand[player][resource] from hexes matching the roll
    // (excluding the robber hex). Settlement = 1, city = 2.
    uint8_t demand[NUM_PLAYERS][NUM_RESOURCES];
    std::memset(demand, 0, sizeof(demand));

    for (uint8_t h = 0; h < topology::NUM_HEXES; ++h) {
        if (b.hex_number[h] != s.dice_roll) continue;
        if (h == s.robber_hex) continue;
        uint8_t res = b.hex_resource[h];
        if (res >= NUM_RESOURCES) continue;  // desert defensive (no number)

        for (uint8_t k = 0; k < topology::MAX_NODES_PER_HEX; ++k) {
            uint8_t v = topology::hex_to_node[h][k];
            uint8_t n = s.node[v];
            uint8_t lvl = node_level(n);
            if (lvl == NODE_EMPTY) continue;
            uint8_t owner = node_owner(n);
            demand[owner][res] += (lvl == NODE_SETTLEMENT) ? 1 : 2;
        }
    }

    // Apply per resource. Bank-shortage rule:
    //   - Single recipient: gets min(demand, bank).
    //   - Multiple recipients: all get full demand iff bank covers total;
    //     otherwise nobody gets that resource.
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) {
        uint8_t total = 0;
        uint8_t recipients = 0;
        uint8_t solo = 0;
        for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
            if (demand[p][r] == 0) continue;
            total += demand[p][r];
            ++recipients;
            solo = p;
        }
        if (recipients == 0) continue;

        if (recipients == 1) {
            uint8_t give = (total <= s.bank[r]) ? total : s.bank[r];
            s.player_resources[solo][r] += give;
            s.player_handsize[solo]     += give;
            s.bank[r]                   -= give;
        } else if (total <= s.bank[r]) {
            for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
                if (demand[p][r] == 0) continue;
                s.player_resources[p][r] += demand[p][r];
                s.player_handsize[p]     += demand[p][r];
                s.bank[r]                -= demand[p][r];
            }
        }
        // else: contested + insufficient → nobody (skip).
    }
}

inline void handle_end_turn(GameState& s) noexcept {
    if (s.dice_roll == 0) return;        // must roll first
    if (s.flag != Flag::NONE) return;    // unresolved sub-phase

    // Move freshly bought dev cards into the playable pile.
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        for (uint8_t d = 0; d < 5; ++d) {
            s.player_dev[p][d]                  += s.player_dev_bought_this_turn[p][d];
            s.player_dev_bought_this_turn[p][d]  = 0;
        }
    }

    // Discard any unfinalized trade scratch from this turn.
    clear_trade_scratch(s);

    s.dice_roll       = 0;
    s.dev_card_played = false;
    s.current_player  = uint8_t((s.current_player + 1) & 0x03);
    s.turn_count     += 1;

    check_game_ended(s);
}

// =====================================================================
// Slice 4: robber sub-phases (discard / move robber / steal)
// =====================================================================

// Steal a random resource card from `victim` and give it to current_player.
// Caller must have validated that victim has at least 1 card.
inline void do_steal(GameState& s, uint8_t victim) noexcept {
    uint8_t total = s.player_handsize[victim];
    if (total == 0) return;

    uint32_t pick = s.rng.bounded(total);
    uint32_t cum = 0;
    uint8_t r = 0;
    for (r = 0; r < NUM_RESOURCES; ++r) {
        cum += s.player_resources[victim][r];
        if (pick < cum) break;
    }

    s.player_resources[victim][r]            -= 1;
    s.player_handsize[victim]                -= 1;
    s.player_resources[s.current_player][r]  += 1;
    s.player_handsize[s.current_player]      += 1;
}

// Apply robber move: update robber_hex, then resolve to STEAL flag /
// auto-steal / NONE depending on victim count on the new hex.
inline void resolve_post_robber(GameState& s) noexcept {
    uint8_t candidates[4];
    uint8_t n_cand = 0;

    for (uint8_t k = 0; k < topology::MAX_NODES_PER_HEX; ++k) {
        uint8_t v = topology::hex_to_node[s.robber_hex][k];
        uint8_t n = s.node[v];
        uint8_t lvl = node_level(n);
        if (lvl == NODE_EMPTY) continue;
        uint8_t owner = node_owner(n);
        if (owner == s.current_player) continue;
        if (s.player_handsize[owner] == 0) continue;

        bool seen = false;
        for (uint8_t j = 0; j < n_cand; ++j) {
            if (candidates[j] == owner) { seen = true; break; }
        }
        if (!seen) candidates[n_cand++] = owner;
    }

    if (n_cand == 0) {
        s.flag = Flag::NONE;
    } else if (n_cand == 1) {
        do_steal(s, candidates[0]);
        s.flag = Flag::NONE;
    } else {
        s.flag = Flag::ROBBER_STEAL;
    }
}

inline void handle_discard(GameState& s, uint32_t action) noexcept {
    if (action - action::DISCARD_BASE >= NUM_RESOURCES) return;
    uint8_t r  = uint8_t(action - action::DISCARD_BASE);
    uint8_t pl = s.current_player;

    if (s.player_discard_remaining[pl] == 0)   return;
    if (s.player_resources[pl][r] == 0)        return;  // can't discard what you don't have

    s.player_resources[pl][r]      -= 1;
    s.bank[r]                      += 1;
    s.player_handsize[pl]          -= 1;
    s.player_discard_remaining[pl] -= 1;

    if (s.player_discard_remaining[pl] == 0) {
        // Hand off to next discarder clockwise from rolling player.
        for (uint8_t i = 1; i <= NUM_PLAYERS; ++i) {
            uint8_t p = uint8_t((s.rolling_player + i) & 0x03);
            if (s.player_discard_remaining[p] > 0) {
                s.current_player = p;
                return;
            }
        }
        // All discards done — robber move next, by rolling player.
        s.current_player = s.rolling_player;
        s.flag = Flag::MOVE_ROBBER;
    }
}

inline void handle_move_robber(GameState& s, uint32_t action) noexcept {
    if (action - action::MOVE_ROBBER_BASE >= topology::NUM_HEXES) return;
    uint8_t hex = uint8_t(action - action::MOVE_ROBBER_BASE);
    if (hex == s.robber_hex) return;  // must move to a different hex

    s.robber_hex = hex;
    resolve_post_robber(s);
}

inline void handle_steal(GameState& s, uint32_t action) noexcept {
    if (action - action::STEAL_BASE >= NUM_PLAYERS) return;
    uint8_t victim = uint8_t(action - action::STEAL_BASE);
    if (victim == s.current_player) return;
    if (s.player_handsize[victim] == 0) return;

    // Victim must own a settle/city on the robber's hex.
    bool valid = false;
    for (uint8_t k = 0; k < topology::MAX_NODES_PER_HEX; ++k) {
        uint8_t v = topology::hex_to_node[s.robber_hex][k];
        uint8_t n = s.node[v];
        if (node_level(n) != NODE_EMPTY && node_owner(n) == victim) {
            valid = true;
            break;
        }
    }
    if (!valid) return;

    do_steal(s, victim);
    s.flag = Flag::NONE;
}

// =====================================================================
// Slice 6: dev cards (buy + play knight)
// =====================================================================

// Dev card type indices (match player_dev[] / dev_deck[] layout).
constexpr uint8_t DEV_KNIGHT          = 0;
constexpr uint8_t DEV_VP              = 1;
constexpr uint8_t DEV_ROAD_BUILDING   = 2;
constexpr uint8_t DEV_YEAR_OF_PLENTY  = 3;
constexpr uint8_t DEV_MONOPOLY        = 4;

constexpr uint8_t COST_DEV[5] = { 0, 0, 1, 1, 1 };  // wool + grain + ore

constexpr uint8_t LARGEST_ARMY_THRESHOLD = 3;
constexpr uint8_t LARGEST_ARMY_VP        = 2;

// After a knight is played, check whether largest_army_owner changes.
// Title goes to first player to play `LARGEST_ARMY_THRESHOLD` knights and
// only transfers when another player STRICTLY exceeds the holder's count.
inline void check_largest_army(GameState& s) noexcept {
    uint8_t holder   = s.largest_army_owner;
    uint8_t holder_n = (holder != NO_PLAYER) ? s.player_knights_played[holder] : 0;
    uint8_t threshold = (holder == NO_PLAYER) ? LARGEST_ARMY_THRESHOLD
                                              : uint8_t(holder_n + 1);

    uint8_t winner   = NO_PLAYER;
    uint8_t winner_n = uint8_t(threshold - 1);
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        uint8_t k = s.player_knights_played[p];
        if (k >= threshold && k > winner_n) {
            winner   = p;
            winner_n = k;
        }
    }

    if (winner == NO_PLAYER || winner == holder) return;

    if (holder != NO_PLAYER) {
        s.player_vp[holder]              -= LARGEST_ARMY_VP;
        s.player_vp_without_dev[holder]  -= LARGEST_ARMY_VP;
    }
    s.largest_army_owner = winner;
    s.player_vp[winner]              += LARGEST_ARMY_VP;
    s.player_vp_without_dev[winner]  += LARGEST_ARMY_VP;
}

inline void handle_buy_dev(GameState& s) noexcept {
    uint8_t pl = s.current_player;
    if (!can_pay(s, pl, COST_DEV)) return;

    uint16_t total = 0;
    for (uint8_t d = 0; d < 5; ++d) total += s.dev_deck[d];
    if (total == 0) return;

    pay_to_bank(s, pl, COST_DEV);

    uint32_t pick = s.rng.bounded(total);
    uint32_t cum = 0;
    uint8_t card = 0;
    for (card = 0; card < 5; ++card) {
        cum += s.dev_deck[card];
        if (pick < cum) break;
    }

    s.dev_deck[card] -= 1;
    s.player_total_dev[pl] += 1;

    if (card == DEV_VP) {
        // VP cards skip the cooldown (always counted, even on buy turn).
        s.player_dev[pl][card] += 1;
        s.player_vp[pl]        += 1;
        // public VP does NOT change — VP cards stay hidden until win reveal.
        check_game_ended(s);
    } else {
        s.player_dev_bought_this_turn[pl][card] += 1;
    }
}

inline void handle_play_knight(GameState& s) noexcept {
    if (s.flag != Flag::NONE) return;     // not during a sub-phase
    if (s.dev_card_played)    return;     // one dev card per turn

    uint8_t pl = s.current_player;
    if (s.player_dev[pl][DEV_KNIGHT] == 0) return;

    s.player_dev[pl][DEV_KNIGHT]    -= 1;
    s.player_total_dev[pl]          -= 1;
    s.player_knights_played[pl]     += 1;
    s.dev_card_played                = true;
    s.rolling_player                 = pl;
    s.flag                           = Flag::MOVE_ROBBER;

    check_largest_army(s);
    check_game_ended(s);
}

// True if player has at least one legal road placement on the current board.
inline bool has_any_legal_road(const GameState& s, uint8_t player) noexcept {
    if (s.player_road_count[player] == 0) return false;
    for (uint8_t e = 0; e < topology::NUM_EDGES; ++e) {
        if (s.edge[e] != NO_PLAYER) continue;
        if (road_connects(s, e, player)) return true;
    }
    return false;
}

// =====================================================================
// Longest road algorithm (graph DFS, no edge revisit, opponent-blocked nodes)
// =====================================================================

constexpr uint8_t LONGEST_ROAD_THRESHOLD = 5;
constexpr uint8_t LONGEST_ROAD_VP        = 2;

// Recursive DFS body. Marks `edge` as visited; explores onward edges that
// share `exit_node = the other endpoint`. Path length = edges traversed.
// Opponent-occupied nodes block continuation through them.
static uint8_t lr_dfs(const GameState& s, uint8_t player,
                       uint8_t edge, uint8_t entry_node,
                       bool* visited) noexcept {
    visited[edge] = true;
    uint8_t exit_node = (topology::edge_to_node[edge][0] == entry_node)
                      ? topology::edge_to_node[edge][1]
                      : topology::edge_to_node[edge][0];

    uint8_t max_branch = 0;
    uint8_t n = s.node[exit_node];
    uint8_t lvl = node_level(n);
    bool blocked = (lvl != NODE_EMPTY) && (node_owner(n) != player);

    if (!blocked) {
        for (uint8_t k = 0; k < topology::MAX_EDGES_PER_NODE; ++k) {
            uint8_t next_e = topology::node_to_edge[exit_node][k];
            if (next_e == topology::NO_EDGE) break;
            if (next_e == edge)            continue;
            if (visited[next_e])           continue;
            if (s.edge[next_e] != player)  continue;
            uint8_t branch = lr_dfs(s, player, next_e, exit_node, visited);
            if (branch > max_branch) max_branch = branch;
        }
    }

    visited[edge] = false;
    return uint8_t(1 + max_branch);
}

inline uint8_t longest_road_for(const GameState& s, uint8_t player) noexcept {
    uint8_t best = 0;
    for (uint8_t v = 0; v < topology::NUM_NODES; ++v) {
        uint8_t n = s.node[v];
        uint8_t lvl = node_level(n);
        if (lvl != NODE_EMPTY && node_owner(n) != player) continue;

        for (uint8_t k = 0; k < topology::MAX_EDGES_PER_NODE; ++k) {
            uint8_t e = topology::node_to_edge[v][k];
            if (e == topology::NO_EDGE) break;
            if (s.edge[e] != player) continue;

            bool visited[topology::NUM_EDGES] = {};
            uint8_t len = lr_dfs(s, player, e, v, visited);
            if (len > best) best = len;
        }
    }
    return best;
}

// Recompute every player's longest-road length and update the title.
// Called after any action that could mutate the road graph (own road build,
// free road via RB, opponent settlement that may split a road).
//
// Title rules:
//   - Need >=5 to claim.
//   - Holder loses title if their length drops below 5 (e.g., cut by an opponent).
//   - Transfer only on STRICT exceed (ties keep incumbent).
inline void check_longest_road(GameState& s) noexcept {
    uint8_t lens[NUM_PLAYERS];
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        lens[p] = longest_road_for(s, p);
        s.player_road_length[p] = lens[p];
    }

    uint8_t holder = s.longest_road_owner;
    uint8_t holder_len = (holder != NO_PLAYER) ? lens[holder] : 0;

    // Cut-below-threshold: holder loses title.
    if (holder != NO_PLAYER && holder_len < LONGEST_ROAD_THRESHOLD) {
        s.player_vp[holder]              -= LONGEST_ROAD_VP;
        s.player_vp_without_dev[holder]  -= LONGEST_ROAD_VP;
        s.longest_road_owner             = NO_PLAYER;
        holder = NO_PLAYER;
        holder_len = 0;
    }

    uint8_t threshold = (holder == NO_PLAYER) ? LONGEST_ROAD_THRESHOLD
                                              : uint8_t(holder_len + 1);
    uint8_t winner   = NO_PLAYER;
    uint8_t winner_n = uint8_t(threshold - 1);
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        if (lens[p] >= threshold && lens[p] > winner_n) {
            winner   = p;
            winner_n = lens[p];
        }
    }

    if (winner == NO_PLAYER || winner == holder) return;

    if (holder != NO_PLAYER) {
        s.player_vp[holder]              -= LONGEST_ROAD_VP;
        s.player_vp_without_dev[holder]  -= LONGEST_ROAD_VP;
    }
    s.longest_road_owner = winner;
    s.player_vp[winner]              += LONGEST_ROAD_VP;
    s.player_vp_without_dev[winner]  += LONGEST_ROAD_VP;
}

inline void handle_play_road_building(GameState& s) noexcept {
    if (s.flag != Flag::NONE) return;
    if (s.dev_card_played)    return;
    uint8_t pl = s.current_player;
    if (s.player_dev[pl][DEV_ROAD_BUILDING] == 0) return;

    s.player_dev[pl][DEV_ROAD_BUILDING] -= 1;
    s.player_total_dev[pl]              -= 1;
    s.dev_card_played                    = true;
    s.free_roads_remaining               = 2;
    s.flag                               = Flag::PLACE_ROAD;

    // No legal roads at all — clear immediately so the game doesn't stall.
    if (!has_any_legal_road(s, pl)) {
        s.free_roads_remaining = 0;
        s.flag                 = Flag::NONE;
    }
}

// PLACE_ROAD sub-phase: free road placement, no resource cost.
// Reuses ROAD_BASE+edge action IDs.
inline void handle_place_road(GameState& s, uint32_t action) noexcept {
    if (action - action::ROAD_BASE >= topology::NUM_EDGES) return;
    uint8_t pl = s.current_player;
    uint8_t edge_id = uint8_t(action - action::ROAD_BASE);

    if (s.player_road_count[pl] == 0)   return;
    if (s.edge[edge_id] != NO_PLAYER)   return;
    if (!road_connects(s, edge_id, pl)) return;

    s.edge[edge_id] = pl;
    s.player_road_count[pl] -= 1;
    s.free_roads_remaining  -= 1;
    check_longest_road(s);

    if (s.free_roads_remaining == 0 || !has_any_legal_road(s, pl)) {
        s.free_roads_remaining = 0;
        s.flag                 = Flag::NONE;
    }
    check_game_ended(s);
}

inline void handle_play_yop(GameState& s, uint32_t action) noexcept {
    if (s.flag != Flag::NONE) return;
    if (s.dev_card_played)    return;
    uint32_t off = action - action::PLAY_YEAR_OF_PLENTY;
    if (off >= 25) return;
    uint8_t r1 = uint8_t(off / NUM_RESOURCES);
    uint8_t r2 = uint8_t(off % NUM_RESOURCES);

    uint8_t pl = s.current_player;
    if (s.player_dev[pl][DEV_YEAR_OF_PLENTY] == 0) return;

    // Bank must cover the requested pair.
    if (r1 == r2) {
        if (s.bank[r1] < 2) return;
    } else {
        if (s.bank[r1] < 1 || s.bank[r2] < 1) return;
    }

    s.player_dev[pl][DEV_YEAR_OF_PLENTY] -= 1;
    s.player_total_dev[pl]               -= 1;
    s.dev_card_played                     = true;

    s.player_resources[pl][r1] += 1;
    s.bank[r1]                 -= 1;
    s.player_handsize[pl]      += 1;

    s.player_resources[pl][r2] += 1;
    s.bank[r2]                 -= 1;
    s.player_handsize[pl]      += 1;
}

inline void handle_play_monopoly(GameState& s, uint32_t action) noexcept {
    if (s.flag != Flag::NONE) return;
    if (s.dev_card_played)    return;
    uint32_t off = action - action::PLAY_MONOPOLY;
    if (off >= NUM_RESOURCES) return;
    uint8_t r = uint8_t(off);

    uint8_t pl = s.current_player;
    if (s.player_dev[pl][DEV_MONOPOLY] == 0) return;

    s.player_dev[pl][DEV_MONOPOLY] -= 1;
    s.player_total_dev[pl]         -= 1;
    s.dev_card_played               = true;

    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        if (p == pl) continue;
        uint8_t take = s.player_resources[p][r];
        if (take == 0) continue;
        s.player_resources[p][r]   -= take;
        s.player_handsize[p]       -= take;
        s.player_resources[pl][r]  += take;
        s.player_handsize[pl]      += take;
    }
}

// =====================================================================
// Player-to-player trade (3-phase: compose -> respond -> confirm)
// =====================================================================

constexpr uint8_t TR_PENDING = 0;
constexpr uint8_t TR_ACCEPT  = 1;
constexpr uint8_t TR_DECLINE = 2;
constexpr uint8_t TR_NA      = 3;  // proposer's own slot

inline uint8_t get_tr(uint8_t resp, uint8_t p) noexcept {
    return uint8_t((resp >> (2 * p)) & 0x3);
}
inline uint8_t set_tr(uint8_t resp, uint8_t p, uint8_t v) noexcept {
    return uint8_t((resp & ~uint8_t(0x3 << (2 * p))) | uint8_t((v & 0x3) << (2 * p)));
}

inline bool trade_scratch_valid(const GameState& s) noexcept {
    bool give_any = false, want_any = false;
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) {
        if (s.trade_give[r] > 0) give_any = true;
        if (s.trade_want[r] > 0) want_any = true;
    }
    return give_any && want_any;
}

inline bool trade_scratch_empty(const GameState& s) noexcept {
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) {
        if (s.trade_give[r] > 0 || s.trade_want[r] > 0) return false;
    }
    return true;
}

inline void clear_trade_scratch(GameState& s) noexcept {
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) {
        s.trade_give[r] = 0;
        s.trade_want[r] = 0;
    }
    s.trade_response = 0;
    s.trade_proposer = NO_PLAYER;
}

// Compose helpers — only valid post-roll with no active flag.
inline void handle_trade_add_give(GameState& s, uint32_t action) noexcept {
    uint32_t r = action - action::TRADE_ADD_GIVE_BASE;
    if (r >= NUM_RESOURCES) return;
    uint8_t pl = s.current_player;
    if (s.player_resources[pl][r] <= s.trade_give[r]) return;  // can't give what you don't own
    s.trade_give[r] += 1;
}
inline void handle_trade_remove_give(GameState& s, uint32_t action) noexcept {
    uint32_t r = action - action::TRADE_REMOVE_GIVE_BASE;
    if (r >= NUM_RESOURCES) return;
    if (s.trade_give[r] == 0) return;
    s.trade_give[r] -= 1;
}
inline void handle_trade_add_want(GameState& s, uint32_t action) noexcept {
    uint32_t r = action - action::TRADE_ADD_WANT_BASE;
    if (r >= NUM_RESOURCES) return;
    if (s.trade_want[r] >= 19) return;  // cap to bank max
    s.trade_want[r] += 1;
}
inline void handle_trade_remove_want(GameState& s, uint32_t action) noexcept {
    uint32_t r = action - action::TRADE_REMOVE_WANT_BASE;
    if (r >= NUM_RESOURCES) return;
    if (s.trade_want[r] == 0) return;
    s.trade_want[r] -= 1;
}

// Open the proposal: validate, set TRADE_PENDING flag, hand control to first
// responder (clockwise from proposer).
inline void handle_trade_open(GameState& s) noexcept {
    if (!trade_scratch_valid(s)) return;
    uint8_t pl = s.current_player;
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) {
        if (s.player_resources[pl][r] < s.trade_give[r]) return;
    }

    s.trade_proposer = pl;
    s.trade_response = 0;
    s.trade_response = set_tr(s.trade_response, pl, TR_NA);
    s.flag = Flag::TRADE_PENDING;

    // First responder (clockwise from proposer)
    uint8_t first_resp = uint8_t((pl + 1) & 0x3);
    s.current_player = first_resp;
}

// Responder accept/decline. If they auto-fail validation (insufficient resources
// for the requested bundle), force a decline to keep the protocol clean.
inline void handle_trade_response(GameState& s, bool wanted_accept) noexcept {
    if (s.flag != Flag::TRADE_PENDING) return;
    if (s.current_player == s.trade_proposer) return;  // confirm phase
    uint8_t pl = s.current_player;
    if (get_tr(s.trade_response, pl) != TR_PENDING) return;

    bool can_accept = wanted_accept;
    if (can_accept) {
        for (uint8_t r = 0; r < NUM_RESOURCES; ++r) {
            if (s.player_resources[pl][r] < s.trade_want[r]) {
                can_accept = false;
                break;
            }
        }
    }
    s.trade_response = set_tr(s.trade_response, pl,
                               can_accept ? TR_ACCEPT : TR_DECLINE);

    // Advance to next pending responder (clockwise from proposer).
    for (uint8_t i = 1; i < NUM_PLAYERS; ++i) {
        uint8_t p = uint8_t((s.trade_proposer + i) & 0x3);
        if (get_tr(s.trade_response, p) == TR_PENDING) {
            s.current_player = p;
            return;
        }
    }
    // All responded — back to proposer for confirm/cancel.
    s.current_player = s.trade_proposer;
}

inline void handle_trade_confirm(GameState& s, uint32_t action) noexcept {
    if (s.flag != Flag::TRADE_PENDING) return;
    if (s.current_player != s.trade_proposer) return;  // wait for confirm phase
    uint32_t off = action - action::TRADE_CONFIRM_BASE;
    if (off >= NUM_PLAYERS) return;
    uint8_t partner = uint8_t(off);
    if (partner == s.trade_proposer) return;
    if (get_tr(s.trade_response, partner) != TR_ACCEPT) return;

    uint8_t pl = s.trade_proposer;
    // Defense in depth: validate both sides still hold the bundles.
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) {
        if (s.player_resources[pl][r]      < s.trade_give[r]) return;
        if (s.player_resources[partner][r] < s.trade_want[r]) return;
    }

    int delta_pl = 0;
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) {
        s.player_resources[pl][r]      -= s.trade_give[r];
        s.player_resources[partner][r] += s.trade_give[r];
        s.player_resources[partner][r] -= s.trade_want[r];
        s.player_resources[pl][r]      += s.trade_want[r];
        delta_pl += int(s.trade_want[r]) - int(s.trade_give[r]);
    }
    s.player_handsize[pl]      = uint8_t(int(s.player_handsize[pl]) + delta_pl);
    s.player_handsize[partner] = uint8_t(int(s.player_handsize[partner]) - delta_pl);

    clear_trade_scratch(s);
    s.flag = Flag::NONE;
    // current_player remains proposer — they keep their turn
}

// Cancel works during compose (no flag) AND during TRADE_PENDING (proposer only).
inline void handle_trade_cancel(GameState& s) noexcept {
    if (s.flag == Flag::NONE) {
        clear_trade_scratch(s);
        return;
    }
    if (s.flag != Flag::TRADE_PENDING) return;
    if (s.current_player != s.trade_proposer) return;
    clear_trade_scratch(s);
    s.flag = Flag::NONE;
}

// =====================================================================
// Slice 5: bank / port trade (4:1, 3:1, 2:1)
// =====================================================================

// Ratio at which `player` can convert `give` resource through best
// available channel: 2 if they own a 2:1 specific port for `give`,
// 3 if any 3:1 generic port, else 4 (bank-only).
inline uint8_t trade_ratio(const GameState& s, uint8_t player, uint8_t give) noexcept {
    uint8_t mask = s.player_ports[player];
    if (mask & uint8_t(1u << give)) return 2;
    if (mask & uint8_t(1u << 5))    return 3;
    return 4;
}

// Action offset = give*5 + get. Same-resource trade is illegal.
inline void handle_trade(GameState& s, uint32_t action) noexcept {
    uint32_t off = action - action::TRADE_BASE;
    if (off >= uint32_t(NUM_RESOURCES) * NUM_RESOURCES) return;
    uint8_t give = uint8_t(off / NUM_RESOURCES);
    uint8_t get  = uint8_t(off % NUM_RESOURCES);
    if (give == get) return;

    uint8_t pl = s.current_player;
    uint8_t ratio = trade_ratio(s, pl, give);

    if (s.player_resources[pl][give] < ratio) return;
    if (s.bank[get] == 0)                     return;

    s.player_resources[pl][give] -= ratio;
    s.bank[give]                 += ratio;
    s.player_resources[pl][get]  += 1;
    s.bank[get]                  -= 1;
    s.player_handsize[pl]        -= uint8_t(ratio - 1);
}

// Dispatches dev-play actions (knight / road building / yop / monopoly).
// Returns true iff an action was matched and routed.
inline bool dispatch_dev_play(GameState& s, uint32_t action) noexcept {
    if (action == action::PLAY_KNIGHT)         { handle_play_knight(s); return true; }
    if (action == action::PLAY_ROAD_BUILDING)  { handle_play_road_building(s); return true; }
    if (action >= action::PLAY_YEAR_OF_PLENTY
        && action <  action::YOP_END)          { handle_play_yop(s, action); return true; }
    if (action >= action::PLAY_MONOPOLY
        && action <  action::MONOPOLY_END)     { handle_play_monopoly(s, action); return true; }
    return false;
}

// Dispatches the compose-phase trade actions (only legal post-roll, no flag).
// Returns true iff matched.
inline bool dispatch_trade_compose(GameState& s, uint32_t action) noexcept {
    if (action >= action::TRADE_ADD_GIVE_BASE
        && action <  action::TRADE_ADD_GIVE_BASE + NUM_RESOURCES) {
        handle_trade_add_give(s, action); return true;
    }
    if (action >= action::TRADE_REMOVE_GIVE_BASE
        && action <  action::TRADE_REMOVE_GIVE_BASE + NUM_RESOURCES) {
        handle_trade_remove_give(s, action); return true;
    }
    if (action >= action::TRADE_ADD_WANT_BASE
        && action <  action::TRADE_ADD_WANT_BASE + NUM_RESOURCES) {
        handle_trade_add_want(s, action); return true;
    }
    if (action >= action::TRADE_REMOVE_WANT_BASE
        && action <  action::TRADE_REMOVE_WANT_BASE + NUM_RESOURCES) {
        handle_trade_remove_want(s, action); return true;
    }
    if (action == action::TRADE_OPEN)   { handle_trade_open(s);   return true; }
    if (action == action::TRADE_CANCEL) { handle_trade_cancel(s); return true; }
    return false;
}

inline void handle_main(GameState& s, const BoardLayout& b, uint32_t action) noexcept {
    if (s.flag != Flag::NONE) {
        // Sub-phase active — only the matching sub-phase action is legal.
        switch (s.flag) {
            case Flag::DISCARD_RESOURCES:
                if (action >= action::DISCARD_BASE
                    && action <  action::DISCARD_BASE + NUM_RESOURCES) {
                    handle_discard(s, action);
                }
                break;
            case Flag::MOVE_ROBBER:
                if (action >= action::MOVE_ROBBER_BASE
                    && action <  action::MOVE_ROBBER_BASE + topology::NUM_HEXES) {
                    handle_move_robber(s, action);
                }
                break;
            case Flag::ROBBER_STEAL:
                if (action >= action::STEAL_BASE
                    && action <  action::STEAL_BASE + NUM_PLAYERS) {
                    handle_steal(s, action);
                }
                break;
            case Flag::PLACE_ROAD:
                if (action >= action::ROAD_BASE
                    && action <  action::ROAD_BASE + topology::NUM_EDGES) {
                    handle_place_road(s, action);
                }
                break;
            case Flag::TRADE_PENDING:
                if (s.current_player == s.trade_proposer) {
                    // Confirm phase: pick accepting partner or cancel.
                    if (action >= action::TRADE_CONFIRM_BASE
                        && action <  action::TRADE_CONFIRM_BASE + NUM_PLAYERS) {
                        handle_trade_confirm(s, action);
                    } else if (action == action::TRADE_CANCEL) {
                        handle_trade_cancel(s);
                    }
                } else {
                    // Respond phase
                    if      (action == action::TRADE_ACCEPT)  handle_trade_response(s, true);
                    else if (action == action::TRADE_DECLINE) handle_trade_response(s, false);
                }
                break;
            default:
                break;
        }
        return;
    }

    if (s.dice_roll == 0) {
        // Pre-roll: ROLL_DICE or any non-VP dev play.
        if (action == action::ROLL_DICE) handle_roll_dice(s, b);
        else dispatch_dev_play(s, action);
        return;
    }
    // Post-roll: build / bank-trade / dev / pvp-trade compose / end turn.
    if      (action == action::END_TURN)                                       handle_end_turn(s);
    else if (action <  action::CITY_BASE)                                       handle_build_settle(s, b, action);
    else if (action <  action::ROAD_BASE)                                       handle_build_city(s, action);
    else if (action <  action::ROLL_DICE)                                       handle_build_road(s, action);
    else if (action >= action::TRADE_BASE && action < action::TRADE_END)        handle_trade(s, action);
    else if (action == action::BUY_DEV)                                         handle_buy_dev(s);
    else if (dispatch_dev_play(s, action))                                      ;  // handled
    else                                                                        dispatch_trade_compose(s, action);
}

}  // namespace

void recompute_awards(GameState& s) noexcept {
    check_longest_road(s);
    check_largest_army(s);
}

void refresh_mask(GameState& s, const BoardLayout& b) noexcept {
    refresh_action_mask(s, b);
}

void reset_with_layout(GameState& s, const BoardLayout& b,
                        uint64_t seed, uint8_t start_player_override) noexcept {
    std::memset(&s, 0, sizeof(s));
    xoshiro_seed(s.rng, seed);

    std::memset(s.edge, NO_PLAYER, sizeof(s.edge));
    s.robber_hex = find_desert(b);

    s.phase = Phase::INITIAL_PLACEMENT_1;
    s.flag  = Flag::NONE;
    s.start_player = (start_player_override < 4)
        ? start_player_override
        : uint8_t(s.rng.bounded(4));
    s.current_player = s.start_player;

    for (uint8_t p = 0; p < 4; ++p) {
        s.player_settlement_count[p] = 5;
        s.player_city_count[p]       = 4;
        s.player_road_count[p]       = 15;
    }

    s.longest_road_owner = NO_PLAYER;
    s.largest_army_owner = NO_PLAYER;
    s.trade_proposer     = NO_PLAYER;

    for (uint8_t r = 0; r < 5; ++r) s.bank[r] = 19;
    std::memcpy(s.dev_deck, INITIAL_DEV_DECK, sizeof(INITIAL_DEV_DECK));

    refresh_action_mask(s, b);
}

void step_one(GameState& s, const BoardLayout& b, uint32_t action,
              float& reward, uint8_t& done) noexcept {
    reward = 0.0f;
    done = 0;

    Phase phase_before = s.phase;
    uint8_t actor = s.current_player;

    switch (s.phase) {
        case Phase::INITIAL_PLACEMENT_1:
        case Phase::INITIAL_PLACEMENT_2:
            handle_initial_placement(s, b, action);
            break;
        case Phase::MAIN:
            handle_main(s, b, action);
            break;
        case Phase::ENDED:
            break;
    }

    if (s.phase == Phase::ENDED) {
        done = 1;
        // If this action triggered the terminal transition, the actor (the
        // player whose action it was) is the winner. Reward is from actor's
        // perspective: +1 win, 0 if game was already over.
        if (phase_before != Phase::ENDED) {
            // Find winner; should equal `actor` since only actor's action
            // pushed VP over the threshold.
            for (uint8_t p = 0; p < 4; ++p) {
                if (s.player_vp[p] >= 10) {
                    reward = (p == actor) ? 1.0f : -1.0f;
                    break;
                }
            }
        }
    }

    // Refresh the incremental action_mask field. Surgical fast path for
    // trade ADD/REMOVE compose actions; full recompute for everything else.
    if (action >= action::TRADE_ADD_GIVE_BASE && action < action::TRADE_OPEN) {
        refresh_compose_mask_bits(s, b);
    } else {
        refresh_action_mask(s, b);
    }
#ifndef NDEBUG
    {
        uint64_t expected[5];
        compute_mask(s, b, expected);
        for (uint32_t _i = 0; _i < 5; ++_i) {
            assert(s.action_mask[_i] == expected[_i] && "incremental mask drift");
        }
    }
#endif
}

}  // namespace catan

// =====================================================================
// Mask: compute legal actions for the current state.
// =====================================================================
#include "mask.hpp"

namespace catan {

// Internal full-recompute body. Writes into the provided 5-word buffer.
// Used by both the incremental-mask field updater (refresh_action_mask)
// and the public compute_mask wrapper.
static inline void recompute_full(const GameState& s,
                                   [[maybe_unused]] const BoardLayout& b,
                                   uint64_t mask[MASK_WORDS]) noexcept {
    for (uint32_t i = 0; i < MASK_WORDS; ++i) mask[i] = 0;

    auto set_bit = [&](uint32_t a) noexcept {
        mask[a >> 6] |= (uint64_t(1) << (a & 63));
    };

    if (s.phase == Phase::ENDED) return;

    if (s.phase == Phase::INITIAL_PLACEMENT_1
        || s.phase == Phase::INITIAL_PLACEMENT_2) {
        if (need_settlement(s)) {
            for (uint8_t n = 0; n < topology::NUM_NODES; ++n) {
                if (can_settle_initial(s, n)) {
                    set_bit(action::SETTLE_BASE + n);
                }
            }
        } else {
            for (uint8_t e = 0; e < topology::NUM_EDGES; ++e) {
                if (can_road_initial(s, e, s.current_player)) {
                    set_bit(action::ROAD_BASE + e);
                }
            }
        }
        return;
    }

    // MAIN phase
    if (s.flag != Flag::NONE) {
        switch (s.flag) {
            case Flag::DISCARD_RESOURCES: {
                uint8_t pl = s.current_player;
                if (s.player_discard_remaining[pl] > 0) {
                    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) {
                        if (s.player_resources[pl][r] > 0) {
                            set_bit(action::DISCARD_BASE + r);
                        }
                    }
                }
                break;
            }
            case Flag::MOVE_ROBBER: {
                for (uint8_t h = 0; h < topology::NUM_HEXES; ++h) {
                    if (h != s.robber_hex) set_bit(action::MOVE_ROBBER_BASE + h);
                }
                break;
            }
            case Flag::ROBBER_STEAL: {
                uint8_t pl = s.current_player;
                bool seen[NUM_PLAYERS] = { false, false, false, false };
                for (uint8_t k = 0; k < topology::MAX_NODES_PER_HEX; ++k) {
                    uint8_t v = topology::hex_to_node[s.robber_hex][k];
                    uint8_t n = s.node[v];
                    if (node_level(n) == NODE_EMPTY) continue;
                    uint8_t owner = node_owner(n);
                    if (owner == pl || seen[owner]) continue;
                    if (s.player_handsize[owner] == 0) continue;
                    seen[owner] = true;
                    set_bit(action::STEAL_BASE + owner);
                }
                break;
            }
            case Flag::PLACE_ROAD: {
                uint8_t pl = s.current_player;
                if (s.player_road_count[pl] > 0) {
                    for (uint8_t e = 0; e < topology::NUM_EDGES; ++e) {
                        if (s.edge[e] != NO_PLAYER) continue;
                        if (road_connects(s, e, pl)) set_bit(action::ROAD_BASE + e);
                    }
                }
                break;
            }
            case Flag::TRADE_PENDING: {
                if (s.current_player == s.trade_proposer) {
                    // Confirm phase. CANCEL always legal; CONFIRM_p for accepters.
                    set_bit(action::TRADE_CANCEL);
                    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
                        if (p == s.trade_proposer) continue;
                        if (get_tr(s.trade_response, p) == TR_ACCEPT) {
                            set_bit(action::TRADE_CONFIRM_BASE + p);
                        }
                    }
                } else {
                    // Respond phase.
                    set_bit(action::TRADE_ACCEPT);
                    set_bit(action::TRADE_DECLINE);
                }
                break;
            }
            default:
                break;
        }
        return;
    }

    uint8_t pl = s.current_player;

    // Helper closure for dev-play legality (shared pre/post roll).
    auto add_dev_play_bits = [&]() noexcept {
        if (s.dev_card_played) return;
        if (s.player_dev[pl][DEV_KNIGHT] > 0) set_bit(action::PLAY_KNIGHT);
        if (s.player_dev[pl][DEV_ROAD_BUILDING] > 0
            && has_any_legal_road(s, pl))     set_bit(action::PLAY_ROAD_BUILDING);
        if (s.player_dev[pl][DEV_YEAR_OF_PLENTY] > 0) {
            for (uint8_t r1 = 0; r1 < NUM_RESOURCES; ++r1) {
                for (uint8_t r2 = 0; r2 < NUM_RESOURCES; ++r2) {
                    bool ok = (r1 == r2) ? (s.bank[r1] >= 2)
                                          : (s.bank[r1] >= 1 && s.bank[r2] >= 1);
                    if (ok) set_bit(action::PLAY_YEAR_OF_PLENTY + uint32_t(r1) * NUM_RESOURCES + r2);
                }
            }
        }
        if (s.player_dev[pl][DEV_MONOPOLY] > 0) {
            for (uint8_t r = 0; r < NUM_RESOURCES; ++r) {
                set_bit(action::PLAY_MONOPOLY + r);
            }
        }
    };

    if (s.dice_roll == 0) {
        // Pre-roll: ROLL_DICE always legal; any non-VP dev card playable.
        set_bit(action::ROLL_DICE);
        add_dev_play_bits();
        return;
    }

    // Post-roll
    set_bit(action::END_TURN);
    add_dev_play_bits();

    // BUY_DEV
    if (can_pay(s, pl, COST_DEV)) {
        uint16_t total = 0;
        for (uint8_t d = 0; d < 5; ++d) total += s.dev_deck[d];
        if (total > 0) set_bit(action::BUY_DEV);
    }

    // Build settlement
    if (s.player_settlement_count[pl] > 0 && can_pay(s, pl, COST_SETTLEMENT)) {
        for (uint8_t n = 0; n < topology::NUM_NODES; ++n) {
            if (can_settle_initial(s, n) && any_player_road_at(s, n, pl)) {
                set_bit(action::SETTLE_BASE + n);
            }
        }
    }

    // Build city
    if (s.player_city_count[pl] > 0 && can_pay(s, pl, COST_CITY)) {
        for (uint8_t n = 0; n < topology::NUM_NODES; ++n) {
            uint8_t nb = s.node[n];
            if (node_level(nb) == NODE_SETTLEMENT && node_owner(nb) == pl) {
                set_bit(action::CITY_BASE + n);
            }
        }
    }

    // Build road
    if (s.player_road_count[pl] > 0 && can_pay(s, pl, COST_ROAD)) {
        for (uint8_t e = 0; e < topology::NUM_EDGES; ++e) {
            if (s.edge[e] != NO_PLAYER) continue;
            if (road_connects(s, e, pl)) set_bit(action::ROAD_BASE + e);
        }
    }

    // Bank/port trades
    for (uint8_t give = 0; give < NUM_RESOURCES; ++give) {
        uint8_t ratio = trade_ratio(s, pl, give);
        if (s.player_resources[pl][give] < ratio) continue;
        for (uint8_t get = 0; get < NUM_RESOURCES; ++get) {
            if (give == get) continue;
            if (s.bank[get] == 0) continue;
            set_bit(action::TRADE_BASE + uint32_t(give) * NUM_RESOURCES + get);
        }
    }

    // Player-to-player trade compose actions
    for (uint8_t r = 0; r < NUM_RESOURCES; ++r) {
        if (s.player_resources[pl][r] > s.trade_give[r]) {
            set_bit(action::TRADE_ADD_GIVE_BASE + r);
        }
        if (s.trade_give[r] > 0) {
            set_bit(action::TRADE_REMOVE_GIVE_BASE + r);
        }
        if (s.trade_want[r] < 19) {
            set_bit(action::TRADE_ADD_WANT_BASE + r);
        }
        if (s.trade_want[r] > 0) {
            set_bit(action::TRADE_REMOVE_WANT_BASE + r);
        }
    }
    if (trade_scratch_valid(s)) {
        // Proposer must own all give resources to open
        bool ok = true;
        for (uint8_t r = 0; r < NUM_RESOURCES; ++r) {
            if (s.player_resources[pl][r] < s.trade_give[r]) { ok = false; break; }
        }
        if (ok) set_bit(action::TRADE_OPEN);
    }
    if (!trade_scratch_empty(s)) {
        set_bit(action::TRADE_CANCEL);
    }
}

// Public API. Writes recomputed mask into the caller's buffer. Independent
// of the incremental field on GameState.
void compute_mask(const GameState& s, const BoardLayout& b,
                  uint64_t mask[MASK_WORDS]) noexcept {
    recompute_full(s, b, mask);
}

// Definition for the top-of-file forward decl. Called by reset_one /
// step_one after any state mutation to keep s.action_mask current.
namespace {
inline void refresh_action_mask(GameState& s, const BoardLayout& b) noexcept {
    recompute_full(s, b, s.action_mask);
}

// Surgical updater for trade-compose actions. Only the compose bit range
// [268, 296) can be affected because ADD/REMOVE GIVE/WANT mutate only
// trade scratch — not flag, dice_roll, phase, resources, or bank.
inline void refresh_compose_mask_bits(GameState& s, const BoardLayout& b) noexcept {
    // Compose bits all sit in word 4: bit 268 = position 12, bit 295 = position 39.
    // Range is positions 12..39 inclusive (28 bits).
    constexpr uint64_t COMPOSE_BIT_RANGE = ((uint64_t(1) << 28) - 1) << 12;
    s.action_mask[4] &= ~COMPOSE_BIT_RANGE;

    // Compose bits are only legal when MAIN, post-roll, no flag.
    if (s.phase != Phase::MAIN || s.flag != Flag::NONE || s.dice_roll == 0) {
        (void)b;
        return;
    }

    auto set_bit = [&](uint32_t a) noexcept {
        s.action_mask[a >> 6] |= (uint64_t(1) << (a & 63));
    };

    uint8_t pl = s.current_player;
    for (uint8_t r = 0; r < 5; ++r) {
        if (s.player_resources[pl][r] > s.trade_give[r])
            set_bit(action::TRADE_ADD_GIVE_BASE + r);
        if (s.trade_give[r] > 0)
            set_bit(action::TRADE_REMOVE_GIVE_BASE + r);
        if (s.trade_want[r] < 19)
            set_bit(action::TRADE_ADD_WANT_BASE + r);
        if (s.trade_want[r] > 0)
            set_bit(action::TRADE_REMOVE_WANT_BASE + r);
    }
    // TRADE_OPEN: scratch valid AND proposer holds the give bundle.
    bool give_any = false, want_any = false;
    for (uint8_t r = 0; r < 5; ++r) {
        if (s.trade_give[r] > 0) give_any = true;
        if (s.trade_want[r] > 0) want_any = true;
    }
    if (give_any && want_any) {
        bool ok = true;
        for (uint8_t r = 0; r < 5; ++r) {
            if (s.player_resources[pl][r] < s.trade_give[r]) { ok = false; break; }
        }
        if (ok) set_bit(action::TRADE_OPEN);
    }
    // TRADE_CANCEL: scratch nonempty.
    if (give_any || want_any) {
        set_bit(action::TRADE_CANCEL);
    }
    (void)b;
}
}  // namespace

}  // namespace catan
