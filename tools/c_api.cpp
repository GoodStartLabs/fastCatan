// Minimal C ABI shim for ctypes-based Python tests.
// Not the eventual nanobind binding — that lives in bindings/pycatan/.
#include "rules.hpp"
#include "mask.hpp"
#include "obs.hpp"
#include "batched_env.hpp"
#include "topology.hpp"
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <new>

namespace {
struct CatanEnv {
    catan::GameState  s;
    catan::BoardLayout b;
};
}

extern "C" {

void* fcatan_create() noexcept {
    void* mem = std::aligned_alloc(64, sizeof(CatanEnv));
    if (!mem) return nullptr;
    return new (mem) CatanEnv{};
}

void fcatan_destroy(void* p) noexcept {
    if (!p) return;
    static_cast<CatanEnv*>(p)->~CatanEnv();
    std::free(p);
}

void fcatan_reset(void* p, uint64_t seed) noexcept {
    auto* e = static_cast<CatanEnv*>(p);
    catan::reset_one(e->s, e->b, seed);
}

uint8_t fcatan_step(void* p, uint32_t action) noexcept {
    auto* e = static_cast<CatanEnv*>(p);
    float reward;
    uint8_t done;
    catan::step_one(e->s, e->b, action, reward, done);
    return done;
}

#define ENV(p) (static_cast<CatanEnv*>(p))

// Top-level state
uint8_t  fcatan_phase(void* p)            noexcept { return uint8_t(ENV(p)->s.phase); }
uint8_t  fcatan_flag(void* p)             noexcept { return uint8_t(ENV(p)->s.flag); }
uint8_t  fcatan_current_player(void* p)   noexcept { return ENV(p)->s.current_player; }
uint8_t  fcatan_start_player(void* p)     noexcept { return ENV(p)->s.start_player; }
uint8_t  fcatan_rolling_player(void* p)   noexcept { return ENV(p)->s.rolling_player; }
uint8_t  fcatan_robber_hex(void* p)       noexcept { return ENV(p)->s.robber_hex; }
uint8_t  fcatan_dice_roll(void* p)        noexcept { return ENV(p)->s.dice_roll; }
uint16_t fcatan_turn_count(void* p)       noexcept { return ENV(p)->s.turn_count; }
uint8_t  fcatan_longest_road_owner(void* p)  noexcept { return ENV(p)->s.longest_road_owner; }
uint8_t  fcatan_largest_army_owner(void* p)  noexcept { return ENV(p)->s.largest_army_owner; }
uint8_t  fcatan_player_discard_remaining(void* p, int pl) noexcept { return ENV(p)->s.player_discard_remaining[pl]; }
uint8_t  fcatan_free_roads_remaining(void* p)       noexcept { return ENV(p)->s.free_roads_remaining; }
uint8_t  fcatan_trade_give(void* p, int r)          noexcept { return ENV(p)->s.trade_give[r]; }
uint8_t  fcatan_trade_want(void* p, int r)          noexcept { return ENV(p)->s.trade_want[r]; }
uint8_t  fcatan_trade_response_byte(void* p)        noexcept { return ENV(p)->s.trade_response; }
uint8_t  fcatan_trade_proposer(void* p)             noexcept { return ENV(p)->s.trade_proposer; }

// Board
uint8_t fcatan_node(void* p, int i)       noexcept { return ENV(p)->s.node[i]; }
uint8_t fcatan_edge(void* p, int i)       noexcept { return ENV(p)->s.edge[i]; }
uint8_t fcatan_hex_resource(void* p, int h) noexcept { return ENV(p)->b.hex_resource[h]; }
uint8_t fcatan_hex_number(void* p, int h)   noexcept { return ENV(p)->b.hex_number[h]; }
uint8_t fcatan_port_type(void* p, int i)    noexcept { return ENV(p)->b.port_type[i]; }
uint8_t fcatan_port_layout(void* p)         noexcept { return ENV(p)->b.port_layout; }

// Per-player
uint8_t fcatan_player_vp(void* p, int pl)         noexcept { return ENV(p)->s.player_vp[pl]; }
uint8_t fcatan_player_vp_public(void* p, int pl)  noexcept { return ENV(p)->s.player_vp_without_dev[pl]; }
uint8_t fcatan_player_handsize(void* p, int pl)   noexcept { return ENV(p)->s.player_handsize[pl]; }
uint8_t fcatan_player_resource(void* p, int pl, int r) noexcept { return ENV(p)->s.player_resources[pl][r]; }
uint8_t fcatan_player_settlement_count(void* p, int pl) noexcept { return ENV(p)->s.player_settlement_count[pl]; }
uint8_t fcatan_player_city_count(void* p, int pl) noexcept { return ENV(p)->s.player_city_count[pl]; }
uint8_t fcatan_player_road_count(void* p, int pl) noexcept { return ENV(p)->s.player_road_count[pl]; }
uint8_t fcatan_player_ports(void* p, int pl)      noexcept { return ENV(p)->s.player_ports[pl]; }
uint8_t fcatan_player_knights_played(void* p, int pl) noexcept { return ENV(p)->s.player_knights_played[pl]; }
uint8_t fcatan_player_dev(void* p, int pl, int d)  noexcept { return ENV(p)->s.player_dev[pl][d]; }
uint8_t fcatan_player_dev_bought(void* p, int pl, int d) noexcept { return ENV(p)->s.player_dev_bought_this_turn[pl][d]; }
uint8_t fcatan_player_total_dev(void* p, int pl)   noexcept { return ENV(p)->s.player_total_dev[pl]; }
uint8_t fcatan_dev_card_played(void* p)            noexcept { return ENV(p)->s.dev_card_played ? 1 : 0; }

uint8_t fcatan_bank(void* p, int r)       noexcept { return ENV(p)->s.bank[r]; }
uint8_t fcatan_dev_deck(void* p, int d)   noexcept { return ENV(p)->s.dev_deck[d]; }

// --- Test-only mutators (manipulate state for unit tests) ---
// All setters refresh the incremental action_mask field after mutating
// state, so subsequent compute_mask calls reflect the change.

// Add `n` of resource `r` to player `pl`, taken from the bank.
// Keeps player_handsize and bank consistent.
void fcatan_give_resources(void* p, int pl, int r, uint8_t n) noexcept {
    auto* e = ENV(p);
    e->s.player_resources[pl][r] += n;
    e->s.player_handsize[pl]     += n;
    e->s.bank[r]                 -= n;
    catan::refresh_mask(e->s, e->b);
}

// Force a player's VP (and public VP). For testing end-of-game triggers.
void fcatan_set_player_vp(void* p, int pl, uint8_t vp) noexcept {
    auto* e = ENV(p);
    e->s.player_vp[pl]              = vp;
    e->s.player_vp_without_dev[pl]  = vp;
    catan::refresh_mask(e->s, e->b);
}

// Force a player's playable dev card count. Updates player_total_dev.
void fcatan_set_player_dev(void* p, int pl, int type, uint8_t n) noexcept {
    auto* e = ENV(p);
    e->s.player_dev[pl][type] = n;
    uint8_t total = 0;
    for (int d = 0; d < 5; ++d) total += e->s.player_dev[pl][d];
    e->s.player_total_dev[pl] = total;
    catan::refresh_mask(e->s, e->b);
}

// Force a player's knights played count (largest-army setup).
void fcatan_set_player_knights_played(void* p, int pl, uint8_t n) noexcept {
    auto* e = ENV(p);
    e->s.player_knights_played[pl] = n;
    catan::refresh_mask(e->s, e->b);
}

// Place an arbitrary node value (level + owner) directly. Test-only —
// bypasses placement rules. Use with care.
void fcatan_set_node(void* p, int node_id, uint8_t level, uint8_t owner) noexcept {
    auto* e = ENV(p);
    e->s.node[node_id] = catan::node_pack(level, owner);
    catan::refresh_mask(e->s, e->b);
}

// Place an edge owner directly. NO_PLAYER (0xFF) clears the edge.
void fcatan_set_edge(void* p, int edge_id, uint8_t owner) noexcept {
    auto* e = ENV(p);
    e->s.edge[edge_id] = owner;
    catan::refresh_mask(e->s, e->b);
}

// Read computed longest-road length (set by check_longest_road).
uint8_t fcatan_player_road_length(void* p, int pl) noexcept {
    return ENV(p)->s.player_road_length[pl];
}

// Re-run check_longest_road + check_largest_army on the current state.
// Useful after test setters mutate node/edge/knight counts directly.
void fcatan_recompute_awards(void* p) noexcept {
    auto* e = ENV(p);
    catan::recompute_awards(e->s);
    catan::refresh_mask(e->s, e->b);
}

// Reset using a caller-provided BoardLayout. For differential testing.
//   hex_resource: 19 bytes (0=brick, 1=lumber, 2=wool, 3=grain, 4=ore, 5=desert)
//   hex_number:   19 bytes (0 for desert, 2..12 skipping 7 elsewhere)
//   port_type:    9 bytes (0..4 = 2:1 specific, 5 = 3:1 generic)
//   port_layout:  0 = pattern A (Catanatron-aligned), 1 = pattern B
//   seed:         per-env RNG seed
//   start_player_override: 0..3 to force, 0xFF to randomize
void fcatan_reset_with_layout(void* p,
                                const uint8_t* hex_resource,
                                const uint8_t* hex_number,
                                const uint8_t* port_type,
                                uint8_t port_layout,
                                uint64_t seed,
                                uint8_t start_player_override) noexcept {
    auto* e = ENV(p);
    std::memcpy(e->b.hex_resource, hex_resource, 19);
    std::memcpy(e->b.hex_number,   hex_number,   19);
    std::memcpy(e->b.port_type,    port_type,    9);
    e->b.port_layout = port_layout;
    catan::reset_with_layout(e->s, e->b, seed, start_player_override);
}

// --- Mask + state cloning helpers ---
void fcatan_compute_mask(void* p, uint64_t* out_mask) noexcept {
    auto* e = ENV(p);
    catan::compute_mask(e->s, e->b, out_mask);
}

uint32_t fcatan_obs_size() noexcept {
    return catan::OBS_SIZE;
}

void fcatan_write_obs(void* p, uint8_t player_pov, float* out) noexcept {
    auto* e = ENV(p);
    catan::write_obs(e->s, e->b, player_pov, out);
}

uint64_t fcatan_state_size() noexcept {
    return uint64_t(sizeof(CatanEnv));
}

void fcatan_copy_state(const void* src, void* dst) noexcept {
    *static_cast<CatanEnv*>(dst) = *static_cast<const CatanEnv*>(src);
}

uint8_t fcatan_state_equal(const void* a, const void* b) noexcept {
    return std::memcmp(a, b, sizeof(CatanEnv)) == 0 ? 1 : 0;
}

// FNV-1a 64-bit hash over the entire CatanEnv byte image. Used for perft
// determinism regression tests — same seed + same action sequence must
// produce the same hash on any compliant build.
uint64_t fcatan_state_hash(const void* p) noexcept {
    constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    constexpr uint64_t FNV_PRIME  = 0x100000001b3ULL;
    const uint8_t* bytes = static_cast<const uint8_t*>(p);
    uint64_t h = FNV_OFFSET;
    for (size_t i = 0; i < sizeof(CatanEnv); ++i) {
        h ^= bytes[i];
        h *= FNV_PRIME;
    }
    return h;
}

// ============================================================
// BatchedEnv API
// ============================================================
void* fbatched_create(uint32_t n_envs, uint64_t master_seed) noexcept {
    auto* be = new catan::BatchedEnv{};
    catan::batched_env_init(*be, n_envs, master_seed);
    return be;
}

void fbatched_destroy(void* p) noexcept {
    if (!p) return;
    auto* be = static_cast<catan::BatchedEnv*>(p);
    catan::batched_env_destroy(*be);
    delete be;
}

uint32_t fbatched_num_envs(const void* p) noexcept {
    return static_cast<const catan::BatchedEnv*>(p)->n;
}

void fbatched_reset(void* p) noexcept {
    catan::batched_env_reset(*static_cast<catan::BatchedEnv*>(p));
}

void fbatched_step(void* p, const uint32_t* actions,
                    float* rewards, uint8_t* dones) noexcept {
    catan::batched_env_step(*static_cast<catan::BatchedEnv*>(p),
                             actions, rewards, dones);
}

void fbatched_write_obs(const void* p, float* out) noexcept {
    catan::batched_env_write_obs(*static_cast<const catan::BatchedEnv*>(p), out);
}

void fbatched_write_masks(const void* p, uint64_t* out) noexcept {
    catan::batched_env_write_masks(*static_cast<const catan::BatchedEnv*>(p), out);
}

// Read-only state probes for testing.
uint8_t fbatched_phase(const void* p, uint32_t i) noexcept {
    return uint8_t(static_cast<const catan::BatchedEnv*>(p)->states[i].phase);
}
uint8_t fbatched_current_player(const void* p, uint32_t i) noexcept {
    return static_cast<const catan::BatchedEnv*>(p)->states[i].current_player;
}
uint8_t fbatched_player_handsize(const void* p, uint32_t i, int pl) noexcept {
    return static_cast<const catan::BatchedEnv*>(p)->states[i].player_handsize[pl];
}
uint8_t fbatched_bank(const void* p, uint32_t i, int r) noexcept {
    return static_cast<const catan::BatchedEnv*>(p)->states[i].bank[r];
}
uint8_t fbatched_player_resource(const void* p, uint32_t i, int pl, int r) noexcept {
    return static_cast<const catan::BatchedEnv*>(p)->states[i].player_resources[pl][r];
}

#undef ENV

}  // extern "C"
