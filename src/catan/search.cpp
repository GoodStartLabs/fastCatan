// Native expectimax alpha-beta player — faithful port of Catanatron's
// AlphaBetaPlayer (catanatron/players/minimax.py) + base_fn value function
// (players/value.py) + chance expansion (players/tree_search_utils.py).
//
// The leaf heuristic is computed directly from GameState + BoardLayout +
// topology (no Catanatron objects); the search branches via the engine's
// expand_action (rules.cpp), which forks the three stochastic action classes
// for the expectimax averaging.
#include "search.hpp"

#include <algorithm>
#include <cmath>

#include "rules.hpp"
#include "mask.hpp"
#include "topology.hpp"

namespace catan {

const double AB_DEFAULT_WEIGHTS[AB_NUM_WEIGHTS] = {
    3e14,    // AB_W_PUBLIC_VPS       — winning dominates all else
    1e8,     // AB_W_PRODUCTION
    -1e8,    // AB_W_ENEMY_PRODUCTION
    1.0,     // AB_W_NUM_TILES
    0.0,     // AB_W_REACH0           — reachable_production_0 (unused by default)
    1e4,     // AB_W_REACH1           — reachable_production_1
    1e3,     // AB_W_BUILDABLE        — buildable_nodes
    10.0,    // AB_W_LONGEST_ROAD     — factor when built-out, else 0.1
    1e2,     // AB_W_HAND_SYNERGY
    1.0,     // AB_W_HAND_RESOURCES
    -5.0,    // AB_W_DISCARD_PENALTY  — applied as a flat penalty if hand > 7
    10.0,    // AB_W_HAND_DEVS
    10.1,    // AB_W_ARMY_SIZE
};

namespace {

// fastCatan resource indices: 0=brick 1=lumber(wood) 2=wool(sheep) 3=grain(wheat) 4=ore
constexpr int R_BRICK = 0, R_WOOD = 1, R_SHEEP = 2, R_WHEAT = 3, R_ORE = 4;

constexpr double TRANSLATE_VARIETY = 4.0;       // each new resource ~ 4 prod pts
constexpr double PROBA_POINT       = 2.778 / 100.0;

// 2d6 sum probability, sum in [2,12]: (6 - |7-sum|)/36. 0 for desert (num 0).
inline double number_probability(uint8_t number) noexcept {
    if (number < 2 || number > 12) return 0.0;
    int d = 7 - int(number);
    if (d < 0) d = -d;
    return double(6 - d) / 36.0;
}

// EFFECTIVE per-resource production for player p (robber hex excluded), i.e.
// Catanatron build_production_features(consider_robber=True) for one player.
void player_production(const GameState& s, const BoardLayout& b, uint8_t p,
                       double prod[5]) noexcept {
    for (int r = 0; r < 5; ++r) prod[r] = 0.0;
    for (uint8_t node = 0; node < topology::NUM_NODES; ++node) {
        uint8_t nb = s.node[node];
        uint8_t lvl = node_level(nb);
        if (lvl == NODE_EMPTY || node_owner(nb) != p) continue;
        double mult = (lvl == NODE_SETTLEMENT) ? 1.0 : 2.0;  // city counts double
        for (uint8_t k = 0; k < topology::MAX_HEXES_PER_NODE; ++k) {
            uint8_t h = topology::node_to_hex[node][k];
            if (h == topology::NO_HEX || h == s.robber_hex) continue;
            uint8_t res = b.hex_resource[h];
            if (res >= NUM_RESOURCES) continue;              // desert
            prod[res] += mult * number_probability(b.hex_number[h]);
        }
    }
}

// Catanatron value_production: sum of effective production, plus (for our own)
// a variety bonus rewarding diversity of produced resources.
double value_production_sum(const double prod[5], bool include_variety) noexcept {
    double sum = 0.0;
    int nonzero = 0;
    for (int r = 0; r < 5; ++r) {
        sum += prod[r];
        if (prod[r] != 0.0) ++nonzero;
    }
    if (include_variety) sum += nonzero * TRANSLATE_VARIETY * PROBA_POINT;
    return sum;
}

// Robber-independent total production probability of a node (Catanatron
// map.node_production summed over resources) — used by count_production for
// the reachability features.
double node_total_production(const BoardLayout& b, uint8_t node) noexcept {
    double t = 0.0;
    for (uint8_t k = 0; k < topology::MAX_HEXES_PER_NODE; ++k) {
        uint8_t h = topology::node_to_hex[node][k];
        if (h == topology::NO_HEX) continue;
        t += number_probability(b.hex_number[h]);
    }
    return t;
}

// Globally buildable (Catanatron board_buildable_ids): node empty and no
// adjacent node carries a building (distance rule).
bool node_board_buildable(const GameState& s, uint8_t node) noexcept {
    if (node_level(s.node[node]) != NODE_EMPTY) return false;
    for (uint8_t k = 0; k < topology::MAX_EDGES_PER_NODE; ++k) {
        uint8_t adj = topology::node_to_node[node][k];
        if (adj == topology::NO_NODE) continue;
        if (node_level(s.node[adj]) != NODE_EMPTY) return false;
    }
    return true;
}

// zero_nodes: union of pov's road-connected components = nodes incident to a
// pov road, plus pov's own buildings (Catanatron connected_components[pov]).
void pov_zero_nodes(const GameState& s, uint8_t pov, bool zero[topology::NUM_NODES]) noexcept {
    for (uint8_t i = 0; i < topology::NUM_NODES; ++i) zero[i] = false;
    for (uint8_t node = 0; node < topology::NUM_NODES; ++node) {
        uint8_t nb = s.node[node];
        if (node_level(nb) != NODE_EMPTY && node_owner(nb) == pov) zero[node] = true;
    }
    for (uint8_t e = 0; e < topology::NUM_EDGES; ++e) {
        if (s.edge[e] != pov) continue;
        zero[topology::edge_to_node[e][0]] = true;
        zero[topology::edge_to_node[e][1]] = true;
    }
}

// level-1 reachable nodes (Catanatron iter_level_nodes, 1 road out): zero_nodes
// plus every neighbor reachable from a non-enemy zero node across an edge that
// is not an enemy road.
void pov_level1_nodes(const GameState& s, uint8_t pov,
                      const bool zero[topology::NUM_NODES],
                      bool lvl1[topology::NUM_NODES]) noexcept {
    for (uint8_t i = 0; i < topology::NUM_NODES; ++i) lvl1[i] = zero[i];
    for (uint8_t node = 0; node < topology::NUM_NODES; ++node) {
        if (!zero[node]) continue;
        uint8_t nb = s.node[node];
        if (node_level(nb) != NODE_EMPTY && node_owner(nb) != pov) continue;  // can't expand through enemy node
        for (uint8_t k = 0; k < topology::MAX_EDGES_PER_NODE; ++k) {
            uint8_t neigh = topology::node_to_node[node][k];
            if (neigh == topology::NO_NODE) continue;
            uint8_t eo = s.edge[topology::node_to_edge[node][k]];
            if (eo != NO_PLAYER && eo != pov) continue;       // enemy road blocks
            lvl1[neigh] = true;
        }
    }
}

}  // namespace

double ab_value(const GameState& s, const BoardLayout& b, uint8_t pov,
                const double* weights) noexcept {
    const double* W = weights ? weights : AB_DEFAULT_WEIGHTS;
    uint8_t enemy = uint8_t((pov + 1) & 0x3);  // Catanatron P1 = next seat

    double our_prod[5], enemy_prod[5];
    player_production(s, b, pov, our_prod);
    player_production(s, b, enemy, enemy_prod);
    double production       = value_production_sum(our_prod, /*variety=*/true);
    double enemy_production = value_production_sum(enemy_prod, /*variety=*/false);

    bool zero[topology::NUM_NODES], lvl1[topology::NUM_NODES];
    pov_zero_nodes(s, pov, zero);
    pov_level1_nodes(s, pov, zero, lvl1);

    double reach0 = 0.0, reach1 = 0.0;
    int num_buildable = 0;
    for (uint8_t node = 0; node < topology::NUM_NODES; ++node) {
        uint8_t nb = s.node[node];
        bool pov_building = (node_level(nb) != NODE_EMPTY && node_owner(nb) == pov);
        bool buildable    = node_board_buildable(s, node);
        bool owned_or_buildable = pov_building || buildable;  // Catanatron get_owned_or_buildable
        if (owned_or_buildable && zero[node]) reach0 += node_total_production(b, node);
        if (owned_or_buildable && lvl1[node]) reach1 += node_total_production(b, node);
        if (zero[node] && buildable) ++num_buildable;          // buildable_node_ids(pov)
    }

    // num_tiles: distinct hexes touched by pov's settlements/cities.
    bool tile_seen[topology::NUM_HEXES] = {false};
    for (uint8_t node = 0; node < topology::NUM_NODES; ++node) {
        uint8_t nb = s.node[node];
        if (node_level(nb) == NODE_EMPTY || node_owner(nb) != pov) continue;
        for (uint8_t k = 0; k < topology::MAX_HEXES_PER_NODE; ++k) {
            uint8_t h = topology::node_to_hex[node][k];
            if (h != topology::NO_HEX) tile_seen[h] = true;
        }
    }
    int num_tiles = 0;
    for (uint8_t h = 0; h < topology::NUM_HEXES; ++h) if (tile_seen[h]) ++num_tiles;

    // hand synergy: closeness to affording a city and a settlement.
    double wheat = s.player_resources[pov][R_WHEAT];
    double ore   = s.player_resources[pov][R_ORE];
    double sheep = s.player_resources[pov][R_SHEEP];
    double brick = s.player_resources[pov][R_BRICK];
    double wood  = s.player_resources[pov][R_WOOD];
    double dist_city = (std::max(2.0 - wheat, 0.0) + std::max(3.0 - ore, 0.0)) / 5.0;
    double dist_settle = (std::max(1.0 - wheat, 0.0) + std::max(1.0 - sheep, 0.0)
                          + std::max(1.0 - brick, 0.0) + std::max(1.0 - wood, 0.0)) / 4.0;
    double hand_synergy = (2.0 - dist_city - dist_settle) / 2.0;

    int num_in_hand = s.player_handsize[pov];
    double discard_penalty = (num_in_hand > 7) ? W[AB_W_DISCARD_PENALTY] : 0.0;

    double longest_road_factor = (num_buildable == 0) ? W[AB_W_LONGEST_ROAD] : 0.1;
    double longest_road_length = double(s.player_road_length[pov]);

    return double(s.player_vp_without_dev[pov]) * W[AB_W_PUBLIC_VPS]   // public VP
         + production       * W[AB_W_PRODUCTION]
         + enemy_production  * W[AB_W_ENEMY_PRODUCTION]
         + reach0            * W[AB_W_REACH0]
         + reach1            * W[AB_W_REACH1]
         + hand_synergy      * W[AB_W_HAND_SYNERGY]
         + double(num_buildable) * W[AB_W_BUILDABLE]
         + double(num_tiles)     * W[AB_W_NUM_TILES]
         + double(num_in_hand)   * W[AB_W_HAND_RESOURCES]
         + discard_penalty
         + longest_road_length * longest_road_factor
         + double(s.player_total_dev[pov])      * W[AB_W_HAND_DEVS]
         + double(s.player_knights_played[pov]) * W[AB_W_ARMY_SIZE];
}

namespace {

constexpr double INF = 1e300;

// Terminal: someone reached the win threshold (actual VP, like Catanatron's
// winning_color()) or the engine ended the game (incl. MAX_TURNS).
inline bool is_terminal(const GameState& s) noexcept {
    if (s.phase == Phase::ENDED) return true;
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p)
        if (s.player_vp[p] >= WIN_VP) return true;
    return false;
}

// Flatten the legal-action bitmask to a list of action IDs.
int legal_actions(const GameState& s, uint32_t* out) noexcept {
    int n = 0;
    for (int w = 0; w < MASK_WORDS; ++w) {
        uint64_t bits = s.action_mask[w];
        int base = w * 64;
        while (bits) {
            int bit = __builtin_ctzll(bits);
            uint32_t aid = uint32_t(base + bit);
            if (aid < NUM_ACTIONS) out[n++] = aid;
            bits &= bits - 1;
        }
    }
    return n;
}

// Catanatron list_prunned_actions, adapted to fastCatan's action space:
//   * initial placement: drop settlements on 1-tile (single-hex) nodes;
//   * robber: keep only the move maximizing (enemy production - our
//     production) among hexes adjacent to an enemy building.
// (Catanatron's maritime-trade prune does not map: fastCatan auto-resolves
// the best port ratio so there are no dominated 4:1 duplicates to drop.)
int prune_actions(const GameState& s, const BoardLayout& b,
                  const uint32_t* in, int nin, uint32_t* out) noexcept {
    bool initial = (s.phase == Phase::INITIAL_PLACEMENT_1
                    || s.phase == Phase::INITIAL_PLACEMENT_2);

    // Robber: identify the single most-impactful move (only if any candidate
    // move targets an enemy-adjacent hex), mirroring prune_robber_actions.
    bool have_robber = false;
    for (int i = 0; i < nin; ++i)
        if (in[i] >= action::MOVE_ROBBER_BASE
            && in[i] < action::MOVE_ROBBER_BASE + topology::NUM_HEXES) { have_robber = true; break; }

    uint32_t best_robber = 0xFFFFFFFFu;
    if (have_robber) {
        // Hexes adjacent to any enemy building.
        bool enemy_hex[topology::NUM_HEXES] = {false};
        for (uint8_t node = 0; node < topology::NUM_NODES; ++node) {
            uint8_t nb = s.node[node];
            if (node_level(nb) == NODE_EMPTY || node_owner(nb) == s.current_player) continue;
            for (uint8_t k = 0; k < topology::MAX_HEXES_PER_NODE; ++k) {
                uint8_t h = topology::node_to_hex[node][k];
                if (h != topology::NO_HEX) enemy_hex[h] = true;
            }
        }
        uint8_t enemy = uint8_t((s.current_player + 1) & 0x3);
        double best_impact = -INF;
        for (int i = 0; i < nin; ++i) {
            if (in[i] < action::MOVE_ROBBER_BASE
                || in[i] >= action::MOVE_ROBBER_BASE + topology::NUM_HEXES) continue;
            uint8_t hex = uint8_t(in[i] - action::MOVE_ROBBER_BASE);
            if (hex >= topology::NUM_HEXES || !enemy_hex[hex]) continue;
            GameState cs = s;
            cs.robber_hex = hex;
            double our[5], opp[5];
            player_production(cs, b, s.current_player, our);
            player_production(cs, b, enemy, opp);
            double impact = value_production_sum(opp, true) - value_production_sum(our, true);
            if (impact > best_impact) { best_impact = impact; best_robber = in[i]; }
        }
    }

    int n = 0;
    for (int i = 0; i < nin; ++i) {
        uint32_t a = in[i];
        bool is_robber = (a >= action::MOVE_ROBBER_BASE
                          && a < action::MOVE_ROBBER_BASE + topology::NUM_HEXES);
        if (is_robber && best_robber != 0xFFFFFFFFu) {
            if (a != best_robber) continue;     // keep only the most impactful
        }
        if (initial && a >= action::SETTLE_BASE
            && a < action::SETTLE_BASE + topology::NUM_NODES) {
            uint8_t node = uint8_t(a - action::SETTLE_BASE);
            int n_hex = 0;
            for (uint8_t k = 0; k < topology::MAX_HEXES_PER_NODE; ++k)
                if (topology::node_to_hex[node][k] != topology::NO_HEX) ++n_hex;
            if (n_hex == 1) continue;           // drop 1-tile corner settlements
        }
        out[n++] = a;
    }
    return n;
}

int get_actions(const GameState& s, const BoardLayout& b, bool prune,
                uint32_t* out) noexcept {
    uint32_t legal[NUM_ACTIONS];
    int n = legal_actions(s, legal);
    if (!prune || n == 0) {
        for (int i = 0; i < n; ++i) out[i] = legal[i];
        return n;
    }
    int pn = prune_actions(s, b, legal, n, out);
    if (pn == 0) {  // defensive: never hand the search an empty set
        for (int i = 0; i < n; ++i) out[i] = legal[i];
        return n;
    }
    return pn;
}

// Expectimax alpha-beta. Maximizes at pov's nodes, minimizes elsewhere; chance
// is averaged via expand_action. alpha/beta are threaded through outcomes
// exactly as Catanatron's alphabeta does (action-level cutoffs only).
double alphabeta(const GameState& s, const BoardLayout& b, uint8_t pov,
                 int depth, double alpha, double beta,
                 const double* W, bool prune) noexcept {
    if (depth == 0 || is_terminal(s)) return ab_value(s, b, pov, W);

    uint32_t actions[NUM_ACTIONS];
    int na = get_actions(s, b, prune, actions);
    if (na == 0) return ab_value(s, b, pov, W);

    bool maximizing = (s.current_player == pov);
    GameState children[MAX_EXPAND_OUTCOMES];
    double probas[MAX_EXPAND_OUTCOMES];

    if (maximizing) {
        double best = -INF;
        for (int i = 0; i < na; ++i) {
            int nc = expand_action(s, b, actions[i], children, probas);
            double ev = 0.0;
            for (int j = 0; j < nc; ++j)
                ev += probas[j] * alphabeta(children[j], b, pov, depth - 1, alpha, beta, W, prune);
            if (ev > best) best = ev;
            if (best > alpha) alpha = best;
            if (alpha >= beta) break;           // beta cutoff
        }
        return best;
    } else {
        double best = INF;
        for (int i = 0; i < na; ++i) {
            int nc = expand_action(s, b, actions[i], children, probas);
            double ev = 0.0;
            for (int j = 0; j < nc; ++j)
                ev += probas[j] * alphabeta(children[j], b, pov, depth - 1, alpha, beta, W, prune);
            if (ev < best) best = ev;
            if (best < beta) beta = best;
            if (beta <= alpha) break;           // alpha cutoff
        }
        return best;
    }
}

}  // namespace

uint32_t ab_decide(const GameState& s_in, const BoardLayout& b, uint8_t pov,
                   int depth, bool prune, const double* weights) noexcept {
    const double* W = weights ? weights : AB_DEFAULT_WEIGHTS;

    // Recompute the root legal-action mask so the search is self-contained and
    // robust to a stale/empty mask (e.g. a freshly injected state). Child masks
    // are rebuilt by expand_action. Equals the maintained mask for a live env.
    GameState s = s_in;
    compute_mask(s, b, s.action_mask);

    uint32_t actions[NUM_ACTIONS];
    int na = get_actions(s, b, prune, actions);
    if (na == 0) return 0xFFFFFFFFu;
    if (na == 1) return actions[0];             // Catanatron decide() shortcut

    double alpha = -INF, beta = INF;
    double best_value = -INF;
    uint32_t best_action = actions[0];
    GameState children[MAX_EXPAND_OUTCOMES];
    double probas[MAX_EXPAND_OUTCOMES];

    for (int i = 0; i < na; ++i) {
        int nc = expand_action(s, b, actions[i], children, probas);
        double ev = 0.0;
        for (int j = 0; j < nc; ++j)
            ev += probas[j] * alphabeta(children[j], b, pov, depth - 1, alpha, beta, W, prune);
        if (ev > best_value) { best_value = ev; best_action = actions[i]; }
        if (best_value > alpha) alpha = best_value;  // thread alpha (root never cuts; beta=+inf)
    }
    return best_action;
}

}  // namespace catan
