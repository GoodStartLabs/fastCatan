// nanobind bindings for fastCatan.
// Exposes BatchedEnv + a single-env convenience class + action constants.
//
// The hot path uses zero-copy ndarray buffers — pass numpy arrays in/out
// and the C++ side writes through their .data() pointers directly.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <new>

#include "rules.hpp"
#include "mask.hpp"
#include "obs.hpp"
#include "batched_env.hpp"

namespace nb = nanobind;
using namespace catan;

namespace {

// Owning wrapper so the Python object manages BatchedEnv lifetime.
struct PyBatchedEnv {
    BatchedEnv inner{};

    PyBatchedEnv(uint32_t n_envs, uint64_t seed) {
        batched_env_init(inner, n_envs, seed);
    }
    ~PyBatchedEnv() { batched_env_destroy(inner); }
    PyBatchedEnv(const PyBatchedEnv&) = delete;
    PyBatchedEnv& operator=(const PyBatchedEnv&) = delete;
};

// Single-env convenience wrapper for the sub-stepwise / debugging API.
struct PyEnv {
    GameState s{};
    BoardLayout b{};

    void reset(uint64_t seed) noexcept {
        reset_one(s, b, seed);
    }
    nb::tuple step(uint32_t action) noexcept {
        float reward = 0.0f;
        uint8_t done = 0;
        step_one(s, b, action, reward, done);
        return nb::make_tuple(reward, done);
    }
    uint8_t phase() const noexcept { return uint8_t(s.phase); }
    uint8_t flag()  const noexcept { return uint8_t(s.flag); }
    uint8_t current_player() const noexcept { return s.current_player; }
    uint8_t dice_roll() const noexcept { return s.dice_roll; }
    uint16_t turn_count() const noexcept { return s.turn_count; }
};

}  // namespace

NB_MODULE(_fastcatan, m) {
    m.doc() = "High-throughput Catan simulator (C++ core via nanobind)";

    // --- Sizes / constants ---
    m.attr("OBS_SIZE")    = OBS_SIZE;
    m.attr("MASK_WORDS")  = MASK_WORDS;
    m.attr("NUM_ACTIONS") = NUM_ACTIONS;
    m.attr("NUM_PLAYERS") = uint32_t(4);
    m.attr("NUM_NODES")   = uint32_t(topology::NUM_NODES);
    m.attr("NUM_EDGES")   = uint32_t(topology::NUM_EDGES);
    m.attr("NUM_HEXES")   = uint32_t(topology::NUM_HEXES);
    m.attr("NUM_PORTS")   = uint32_t(topology::NUM_PORTS);

    // --- Action ID constants (under fastcatan.action) ---
    nb::module_ act = m.def_submodule("action", "Flat action ID layout.");
    act.attr("SETTLE_BASE")            = action::SETTLE_BASE;
    act.attr("CITY_BASE")              = action::CITY_BASE;
    act.attr("ROAD_BASE")              = action::ROAD_BASE;
    act.attr("ROLL_DICE")              = action::ROLL_DICE;
    act.attr("END_TURN")               = action::END_TURN;
    act.attr("DISCARD_BASE")           = action::DISCARD_BASE;
    act.attr("MOVE_ROBBER_BASE")       = action::MOVE_ROBBER_BASE;
    act.attr("STEAL_BASE")             = action::STEAL_BASE;
    act.attr("TRADE_BASE")             = action::TRADE_BASE;
    act.attr("TRADE_END")              = action::TRADE_END;
    act.attr("BUY_DEV")                = action::BUY_DEV;
    act.attr("PLAY_KNIGHT")            = action::PLAY_KNIGHT;
    act.attr("PLAY_ROAD_BUILDING")     = action::PLAY_ROAD_BUILDING;
    act.attr("PLAY_YEAR_OF_PLENTY")    = action::PLAY_YEAR_OF_PLENTY;
    act.attr("YOP_END")                = action::YOP_END;
    act.attr("PLAY_MONOPOLY")          = action::PLAY_MONOPOLY;
    act.attr("MONOPOLY_END")           = action::MONOPOLY_END;
    act.attr("TRADE_ADD_GIVE_BASE")    = action::TRADE_ADD_GIVE_BASE;
    act.attr("TRADE_REMOVE_GIVE_BASE") = action::TRADE_REMOVE_GIVE_BASE;
    act.attr("TRADE_ADD_WANT_BASE")    = action::TRADE_ADD_WANT_BASE;
    act.attr("TRADE_REMOVE_WANT_BASE") = action::TRADE_REMOVE_WANT_BASE;
    act.attr("TRADE_OPEN")             = action::TRADE_OPEN;
    act.attr("TRADE_ACCEPT")           = action::TRADE_ACCEPT;
    act.attr("TRADE_DECLINE")          = action::TRADE_DECLINE;
    act.attr("TRADE_CONFIRM_BASE")     = action::TRADE_CONFIRM_BASE;
    act.attr("TRADE_CANCEL")           = action::TRADE_CANCEL;

    // --- Single-env API (mostly for tests + debugging) ---
    nb::class_<PyEnv>(m, "Env",
        "Single env handle. For batched throughput use BatchedEnv.")
        .def(nb::init<>())
        .def("reset", &PyEnv::reset, nb::arg("seed"))
        .def("step",  &PyEnv::step,  nb::arg("action"),
             "Apply action; returns (reward, done).")
        .def_prop_ro("phase",          &PyEnv::phase)
        .def_prop_ro("flag",           &PyEnv::flag)
        .def_prop_ro("current_player", &PyEnv::current_player)
        .def_prop_ro("dice_roll",      &PyEnv::dice_roll)
        .def_prop_ro("turn_count",     &PyEnv::turn_count);

    // --- BatchedEnv: hot path ---
    using ArrU32 = nb::ndarray<uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using ArrU8  = nb::ndarray<uint8_t,  nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using ArrF32 = nb::ndarray<float,    nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using ArrU64 = nb::ndarray<uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using ArrF32_2D = nb::ndarray<float,    nb::ndim<2>, nb::c_contig, nb::device::cpu>;
    using ArrU64_2D = nb::ndarray<uint64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

    nb::class_<PyBatchedEnv>(m, "BatchedEnv",
        "Owns N independent envs in a contiguous buffer. "
        "step() runs all envs in lockstep; auto-reset on done.")
        .def(nb::init<uint32_t, uint64_t>(),
             nb::arg("num_envs"), nb::arg("seed") = 42)
        .def_prop_ro("num_envs", [](const PyBatchedEnv& e) { return e.inner.n; })
        .def("reset",
             [](PyBatchedEnv& e) {
                 nb::gil_scoped_release release;
                 batched_env_reset(e.inner);
             },
             "Reset all envs to fresh starting positions.")
        .def("step",
             [](PyBatchedEnv& e, ArrU32 actions, ArrF32 rewards, ArrU8 dones) {
                 if (actions.shape(0) != e.inner.n
                     || rewards.shape(0) != e.inner.n
                     || dones.shape(0)   != e.inner.n) {
                     throw std::runtime_error("step buffer length mismatch");
                 }
                 nb::gil_scoped_release release;
                 batched_env_step(e.inner, actions.data(), rewards.data(), dones.data());
             },
             nb::arg("actions"), nb::arg("rewards_out"), nb::arg("dones_out"))
        .def("write_obs",
             [](PyBatchedEnv& e, ArrF32_2D out) {
                 if (out.shape(0) != e.inner.n || out.shape(1) != OBS_SIZE) {
                     throw std::runtime_error("obs buffer shape mismatch");
                 }
                 nb::gil_scoped_release release;
                 batched_env_write_obs(e.inner, out.data());
             },
             nb::arg("out"),
             "Fill (num_envs, OBS_SIZE) float buffer with each env's POV obs.")
        .def("write_obs_pov",
             [](PyBatchedEnv& e, uint32_t env_idx, uint8_t pov, ArrF32 out) {
                 if (env_idx >= e.inner.n) throw std::runtime_error("env_idx out of range");
                 if (out.shape(0) != OBS_SIZE) throw std::runtime_error("obs buffer length mismatch");
                 nb::gil_scoped_release release;
                 write_obs(e.inner.states[env_idx], e.inner.layouts[env_idx], pov, out.data());
             },
             nb::arg("env_idx"), nb::arg("pov"), nb::arg("out"),
             "Single-env obs from a chosen POV. For PettingZoo-style AEC.")
        .def("write_masks",
             [](PyBatchedEnv& e, ArrU64_2D out) {
                 if (out.shape(0) != e.inner.n || out.shape(1) != MASK_WORDS) {
                     throw std::runtime_error("mask buffer shape mismatch");
                 }
                 nb::gil_scoped_release release;
                 batched_env_write_masks(e.inner, out.data());
             },
             nb::arg("out"),
             "Fill (num_envs, MASK_WORDS) uint64 buffer with legal-action bits.")
        // Read-only state probes (for tests; not the hot path).
        .def("phase",
             [](const PyBatchedEnv& e, uint32_t i) -> uint8_t {
                 return uint8_t(e.inner.states[i].phase);
             }, nb::arg("env_idx"))
        .def("current_player",
             [](const PyBatchedEnv& e, uint32_t i) -> uint8_t {
                 return e.inner.states[i].current_player;
             }, nb::arg("env_idx"))
        .def("player_vp",
             [](const PyBatchedEnv& e, uint32_t i, uint32_t pl) -> uint8_t {
                 return e.inner.states[i].player_vp[pl];
             }, nb::arg("env_idx"), nb::arg("player"))
        .def("last_winner",
             [](const PyBatchedEnv& e, uint32_t i) -> uint8_t {
                 return e.inner.last_winner[i];
             }, nb::arg("env_idx"),
             "Winner of the most recently completed game in env_idx, or "
             "NO_PLAYER (255) if no game has finished yet. Set on done; "
             "preserved across the auto-reset.");
}
