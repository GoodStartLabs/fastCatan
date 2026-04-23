#pragma once
#include <cstdint>

namespace catan {

// Node encoding: high nibble = owner (0..3, 0xF = empty), low nibble = level (0 empty, 1 settlement, 2 city).
inline constexpr uint8_t NODE_EMPTY = 0xF0;

constexpr uint8_t node_owner(uint8_t cell) { return cell >> 4; }
constexpr uint8_t node_level(uint8_t cell) { return cell & 0x0F; }
constexpr uint8_t make_node(uint8_t owner, uint8_t level) {
    return static_cast<uint8_t>((owner << 4) | (level & 0x0F));
}

// Edge encoding: 0..3 owner, 0xFF empty. Byte-per-slot; scattered writes make bit-packing a net loss here.
inline constexpr uint8_t EDGE_EMPTY = 0xFF;

struct alignas(64) GameState {
    // Nodes (54): owner + level packed into one byte each.
    uint8_t node[54];

    // Edges (72): one byte per edge, owner id or EDGE_EMPTY.
    uint8_t edge[72];

    // Hexes (19): board layout, randomized per episode. Adjacency is static in topology.hpp.
    uint8_t hex_resource[19];   // 0=brick 1=lumber 2=wool 3=grain 4=ore 5=desert
    uint8_t hex_number[19];     // 0 for desert; else 2..12 skipping 7
    uint8_t port_type[9];       // 0..4 = 2:1 specific resource, 5 = 3:1 generic; indexed by port slot
    uint8_t robber_hex;

    // Per player (index 0..3).
    uint8_t player_resources[4][5];   // brick, lumber, wool, grain, ore
};

} // namespace catan
