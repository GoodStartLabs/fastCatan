// Minimal C ABI shim for ctypes-based Python tests.
// Not the eventual nanobind binding — that lives in bindings/pycatan/.
#include "rules.hpp"
#include "mask.hpp"
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
// Add `n` of resource `r` to player `pl`, taken from the bank.
// Keeps player_handsize and bank consistent.
void fcatan_give_resources(void* p, int pl, int r, uint8_t n) noexcept {
    auto* e = ENV(p);
    e->s.player_resources[pl][r] += n;
    e->s.player_handsize[pl]     += n;
    e->s.bank[r]                 -= n;
}

// Force a player's VP (and public VP). For testing end-of-game triggers.
void fcatan_set_player_vp(void* p, int pl, uint8_t vp) noexcept {
    auto* e = ENV(p);
    e->s.player_vp[pl]              = vp;
    e->s.player_vp_without_dev[pl]  = vp;
}

// Force a player's playable dev card count. Updates player_total_dev.
void fcatan_set_player_dev(void* p, int pl, int type, uint8_t n) noexcept {
    auto* e = ENV(p);
    e->s.player_dev[pl][type] = n;
    uint8_t total = 0;
    for (int d = 0; d < 5; ++d) total += e->s.player_dev[pl][d];
    e->s.player_total_dev[pl] = total;
}

// Force a player's knights played count (largest-army setup).
void fcatan_set_player_knights_played(void* p, int pl, uint8_t n) noexcept {
    ENV(p)->s.player_knights_played[pl] = n;
}

// --- Mask + state cloning helpers ---
void fcatan_compute_mask(void* p, uint64_t* out_mask) noexcept {
    auto* e = ENV(p);
    catan::compute_mask(e->s, e->b, out_mask);
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

#undef ENV

}  // extern "C"
