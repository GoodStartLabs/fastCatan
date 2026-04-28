#pragma once
#include <cstdint>

namespace catan {

    // xoshiro128++ — 32-bit output, 16 B state, BigCrush-clean.
    // Reference: https://prng.di.unimi.it/xoshiro128plusplus.c
    //   (Blackman & Vigna, public domain)

    struct Xoshiro128 {
        uint32_t s[4];

        // Advance state, return next 32-bit value.
        inline uint32_t next() noexcept {
            auto rotl = [](uint32_t x, int k) -> uint32_t {
                return (x << k) | (x >> (32 - k));
            };
            const uint32_t result = rotl(s[0] + s[3], 7) + s[0];
            const uint32_t t = s[1] << 9;
            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];
            s[2] ^= t;
            s[3] = rotl(s[3], 11);
            return result;
        }

        // Unbiased uniform integer in [0, bound). Lemire's method.
        // The rejection branch almost never fires for small bounds.
        inline uint32_t bounded(uint32_t bound) noexcept {
            uint64_t product = uint64_t(next()) * bound;
            uint32_t lo = uint32_t(product);
            if (lo < bound) {
                uint32_t threshold = uint32_t(0u - bound) % bound;
                while (lo < threshold) {
                    product = uint64_t(next()) * bound;
                    lo = uint32_t(product);
                }
            }
            return uint32_t(product >> 32);
        }
    };

    // SplitMix64 — mixes a 64-bit input. Used to expand the user-facing
    // master seed into the 128 bits of xoshiro state, and to derive
    // per-env seeds from a master seed at batch creation.
    inline uint64_t splitmix64(uint64_t& x) noexcept {
        uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    // Seed an Xoshiro128 from a single 64-bit key. Falls back to a fixed
    // non-zero state if key==0 (xoshiro is undefined when all bits are zero).
    inline void xoshiro_seed(Xoshiro128& r, uint64_t key) noexcept {
        uint64_t x = key ? key : 0x123456789abcdef0ULL;
        uint64_t a = splitmix64(x);
        uint64_t b = splitmix64(x);
        r.s[0] = uint32_t(a);
        r.s[1] = uint32_t(a >> 32);
        r.s[2] = uint32_t(b);
        r.s[3] = uint32_t(b >> 32);
    }

    static_assert(sizeof(Xoshiro128) == 16, "Xoshiro128 must be 16 bytes");

}  // namespace catan
