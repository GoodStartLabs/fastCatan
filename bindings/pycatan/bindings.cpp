// nanobind bindings for fastCatan.
// Exposes BatchedEnv + a single-env convenience class + action constants.
//
// The hot path uses zero-copy ndarray buffers — pass numpy arrays in/out
// and the C++ side writes through their .data() pointers directly.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <cstring>

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

    uint8_t player_vp(uint8_t seat) const noexcept { return s.player_vp[seat]; }
    uint8_t player_vp_public(uint8_t seat) const noexcept { return s.player_vp_without_dev[seat]; }
    uint8_t player_handsize(uint8_t seat) const noexcept { return s.player_handsize[seat]; }
    uint8_t player_settlement_count(uint8_t seat) const noexcept { return s.player_settlement_count[seat]; }
    uint8_t player_city_count(uint8_t seat) const noexcept { return s.player_city_count[seat]; }
    uint8_t player_road_count(uint8_t seat) const noexcept { return s.player_road_count[seat]; }
    uint8_t player_knights_played(uint8_t seat) const noexcept { return s.player_knights_played[seat]; }
    uint8_t player_road_length(uint8_t seat) const noexcept { return s.player_road_length[seat]; }
    uint8_t player_ports(uint8_t seat) const noexcept { return s.player_ports[seat]; }
    uint8_t player_resource(uint8_t seat, uint8_t r) const noexcept { return s.player_resources[seat][r]; }
    uint8_t bank(uint8_t r) const noexcept { return s.bank[r]; }
    uint8_t longest_road_owner() const noexcept { return s.longest_road_owner; }
    uint8_t largest_army_owner() const noexcept { return s.largest_army_owner; }

    // Snapshot/restore as Python bytes. Used by alpha-beta search to
    // branch state without committing to one path.
    nb::bytes snapshot() const {
        char buf[sizeof(GameState) + sizeof(BoardLayout)];
        std::memcpy(buf, &s, sizeof(GameState));
        std::memcpy(buf + sizeof(GameState), &b, sizeof(BoardLayout));
        return nb::bytes(buf, sizeof(buf));
    }
    void load_snapshot(nb::bytes data) {
        const std::size_t want = sizeof(GameState) + sizeof(BoardLayout);
        if (data.size() != want)
            throw std::runtime_error("snapshot size mismatch");
        std::memcpy(&s, data.c_str(), sizeof(GameState));
        std::memcpy(&b, data.c_str() + sizeof(GameState), sizeof(BoardLayout));
    }

    // Read the cached action_mask bits into a uint64 buffer of length MASK_WORDS.
    void action_mask(nb::ndarray<uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> out) const {
        if (out.shape(0) != MASK_WORDS)
            throw std::runtime_error("action_mask buffer length mismatch");
        for (uint32_t i = 0; i < MASK_WORDS; ++i) out.data()[i] = s.action_mask[i];
    }

    void write_obs(uint8_t pov, nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> out) const {
        if (out.shape(0) != OBS_SIZE)
            throw std::runtime_error("obs buffer length mismatch");
        ::catan::write_obs(s, b, pov, out.data());
    }
};

}  // namespace

NB_MODULE(_fastcatan, m) {
    m.doc() = R"(High-throughput Catan simulator (C++ core via nanobind).

This module exposes:
  - ``Env``        single-env handle for tests / debugging.
  - ``BatchedEnv`` N-env handle for hot-path RL (auto-reset on done).
  - ``action``     namespace with all action ID constants.
  - Module-level shape constants: ``OBS_SIZE``, ``MASK_WORDS``,
    ``NUM_ACTIONS``, ``NUM_PLAYERS``, ``NUM_NODES``, ``NUM_EDGES``,
    ``NUM_HEXES``, ``NUM_PORTS``.

Action space: flat Discrete(NUM_ACTIONS). See README for the full
ID layout. Use ``compute_mask`` / ``BatchedEnv.write_masks`` to get
the legal-action bitmask before sampling.

Observation: ``OBS_SIZE`` float32 features per env, POV-relative
(learner at slot 0, opponents at +1, +2, +3).

Reward: +1 on the action that wins; -1 on actions that trigger
another player's win (rare); 0 everywhere else.

Determinism: same seed → same trajectory. Perft hashes pinned.
)";

    // --- Sizes / constants ---
    m.attr("OBS_SIZE")    = OBS_SIZE;       // float32 features per env
    m.attr("MASK_WORDS")  = MASK_WORDS;     // uint64 words in legal-action mask
    m.attr("NUM_ACTIONS") = NUM_ACTIONS;    // total flat action IDs
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
    act.attr("BUY_DEV")                = action::BUY_DEV;
    act.attr("PLAY_KNIGHT")            = action::PLAY_KNIGHT;
    act.attr("PLAY_ROAD_BUILDING")     = action::PLAY_ROAD_BUILDING;
    act.attr("PLAY_YEAR_OF_PLENTY")    = action::PLAY_YEAR_OF_PLENTY;
    act.attr("PLAY_MONOPOLY")          = action::PLAY_MONOPOLY;
    act.attr("TRADE_ADD_GIVE_BASE")    = action::TRADE_ADD_GIVE_BASE;
    act.attr("TRADE_ADD_WANT_BASE")    = action::TRADE_ADD_WANT_BASE;
    act.attr("TRADE_OPEN")             = action::TRADE_OPEN;
    act.attr("TRADE_ACCEPT")           = action::TRADE_ACCEPT;
    act.attr("TRADE_DECLINE")          = action::TRADE_DECLINE;
    act.attr("TRADE_CONFIRM_BASE")     = action::TRADE_CONFIRM_BASE;
    act.attr("TRADE_CANCEL")           = action::TRADE_CANCEL;

    // --- Single-env API (mostly for tests + debugging) ---
    nb::class_<PyEnv>(m, "Env",
        "Single env handle. For batched throughput use ``BatchedEnv``. "
        "This API is intended for unit tests, debugging, and small experiments — "
        "every method round-trips through Python so it's not the hot path.")
        .def(nb::init<>(), "Construct an empty env (call ``reset`` next).")
        .def("reset", &PyEnv::reset, nb::arg("seed"),
             "Reset to a fresh game with the given uint64 seed.")
        .def("step",  &PyEnv::step,  nb::arg("action"),
             "Apply ``action`` (a flat action ID; see ``fastcatan.action``). "
             "Returns ``(reward: float, done: bool)``. Illegal actions are "
             "no-ops — use the mask to avoid them.")
        .def_prop_ro("phase",          &PyEnv::phase,
             "Current phase: 0=INITIAL_PLACEMENT_1, 1=INITIAL_PLACEMENT_2, "
             "2=MAIN, 3=ENDED.")
        .def_prop_ro("flag",           &PyEnv::flag,
             "Active sub-phase flag (0=NONE, 1=DISCARD, 2=MOVE_ROBBER, "
             "3=ROBBER_STEAL, 4=YEAR_OF_PLENTY, 5=MONOPOLY, 6=PLACE_ROAD, "
             "7=TRADE_PENDING).")
        .def_prop_ro("current_player", &PyEnv::current_player,
             "Active player index 0..3.")
        .def_prop_ro("dice_roll",      &PyEnv::dice_roll,
             "Last dice roll (2..12) or 0 if not yet rolled this turn.")
        .def_prop_ro("turn_count",     &PyEnv::turn_count,
             "Monotonic turn counter (increments on END_TURN).")
        // State accessors for heuristic policies / alpha-beta evaluation.
        .def("player_vp", &PyEnv::player_vp, nb::arg("seat"),
             "Total VP for ``seat`` (incl. hidden VP cards).")
        .def("player_vp_public", &PyEnv::player_vp_public, nb::arg("seat"),
             "Publicly-visible VP for ``seat`` (excludes hidden VP cards).")
        .def("player_handsize", &PyEnv::player_handsize, nb::arg("seat"))
        .def("player_settlement_count", &PyEnv::player_settlement_count, nb::arg("seat"),
             "Settlements REMAINING in ``seat``'s stock (5 - placed-on-board).")
        .def("player_city_count", &PyEnv::player_city_count, nb::arg("seat"),
             "Cities REMAINING in stock.")
        .def("player_road_count", &PyEnv::player_road_count, nb::arg("seat"),
             "Roads REMAINING in stock.")
        .def("player_knights_played", &PyEnv::player_knights_played, nb::arg("seat"))
        .def("player_road_length", &PyEnv::player_road_length, nb::arg("seat"),
             "Computed longest-road length for ``seat``.")
        .def("player_ports", &PyEnv::player_ports, nb::arg("seat"),
             "Bitmask of ports the seat has access to (bits 0..4 = 2:1 by "
             "resource, bit 5 = 3:1 generic).")
        .def("player_resource", &PyEnv::player_resource,
             nb::arg("seat"), nb::arg("resource"),
             "Resource count in ``seat``'s hand. Resources: "
             "0=brick 1=lumber 2=wool 3=grain 4=ore.")
        .def("bank", &PyEnv::bank, nb::arg("resource"),
             "Bank stock for the given resource.")
        .def_prop_ro("longest_road_owner", &PyEnv::longest_road_owner,
             "Player holding longest road or 255 if none.")
        .def_prop_ro("largest_army_owner", &PyEnv::largest_army_owner,
             "Player holding largest army or 255 if none.")
        // Snapshot/restore for state branching (alpha-beta search etc).
        .def("snapshot", &PyEnv::snapshot,
             "Serialize the env's GameState + BoardLayout into a bytes "
             "object. Round-trip with ``load_snapshot``.")
        .def("load_snapshot", &PyEnv::load_snapshot, nb::arg("data"),
             "Restore from bytes produced by ``snapshot`` or "
             "``BatchedEnv.snapshot``.")
        .def("action_mask", &PyEnv::action_mask, nb::arg("out"),
             "Read the incrementally-maintained action mask into the "
             "provided uint64 buffer of length MASK_WORDS.")
        .def("write_obs", &PyEnv::write_obs,
             nb::arg("pov"), nb::arg("out"),
             "Write obs from ``pov`` (player 0..3) into the provided "
             "float32 buffer of length OBS_SIZE.");

    // --- BatchedEnv: hot path ---
    using ArrU32 = nb::ndarray<uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using ArrU8  = nb::ndarray<uint8_t,  nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using ArrF32 = nb::ndarray<float,    nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using ArrU64 = nb::ndarray<uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using ArrF32_2D = nb::ndarray<float,    nb::ndim<2>, nb::c_contig, nb::device::cpu>;
    using ArrU64_2D = nb::ndarray<uint64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

    nb::class_<PyBatchedEnv>(m, "BatchedEnv",
        R"(Owns N independent Catan envs in one contiguous buffer.

This is the hot path for RL training. Per-env state is laid out for
cache-friendly batched stepping. ``step`` advances every env by one
action in lockstep; envs that hit a terminal state are auto-reset
with a fresh per-env seed (their ``last_winner`` is preserved).

Pre-allocate numpy buffers and pass them to ``step`` / ``write_obs`` /
``write_masks`` — nanobind hands the buffer pointers straight through to
the C++ side, no copies.

Throughput: ~12M steps/sec single-thread on a 2024 Mac, scaling near-
linear with cores via OpenMP on Linux + GCC.

Example::

    env = fastcatan.BatchedEnv(num_envs=4096, seed=42)
    env.reset()
    actions = np.zeros(4096, dtype=np.uint32)
    rewards = np.zeros(4096, dtype=np.float32)
    dones   = np.zeros(4096, dtype=np.uint8)
    masks   = np.zeros((4096, fastcatan.MASK_WORDS), dtype=np.uint64)
    obs     = np.zeros((4096, fastcatan.OBS_SIZE), dtype=np.float32)

    for _ in range(rollout_len):
        env.write_masks(masks)
        # ... policy fills `actions` ...
        env.step(actions, rewards, dones)
        env.write_obs(obs)
)")
        .def(nb::init<uint32_t, uint64_t>(),
             nb::arg("num_envs"), nb::arg("seed") = 42,
             "Allocate ``num_envs`` envs. Per-env seeds derived from "
             "``seed`` via SplitMix64 (independent streams). Call "
             "``reset`` before stepping.")
        .def_prop_ro("num_envs", [](const PyBatchedEnv& e) { return e.inner.n; },
             "Number of envs in this batch.")
        .def("reset",
             [](PyBatchedEnv& e) {
                 nb::gil_scoped_release release;
                 batched_env_reset(e.inner);
             },
             "Reset all envs to fresh starting positions, advancing the "
             "internal seed counter so each call produces a different batch.")
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
             nb::arg("actions"), nb::arg("rewards_out"), nb::arg("dones_out"),
             R"(Advance every env by one action.

Args:
    actions:     uint32 array of length ``num_envs``. Action ID per env.
    rewards_out: float32 array of length ``num_envs``. Filled in place.
    dones_out:   uint8 array of length ``num_envs``. Filled in place.

On done, the env auto-resets with the next per-env seed. Use
``last_winner(i)`` immediately after ``step`` to read who won the
just-completed game (preserved across the reset).

GIL is released during execution.
)")
        .def("write_obs",
             [](PyBatchedEnv& e, ArrF32_2D out) {
                 if (out.shape(0) != e.inner.n || out.shape(1) != OBS_SIZE) {
                     throw std::runtime_error("obs buffer shape mismatch");
                 }
                 nb::gil_scoped_release release;
                 batched_env_write_obs(e.inner, out.data());
             },
             nb::arg("out"),
             R"(Fill (num_envs, OBS_SIZE) float32 buffer in place.

Each row is the obs from that env's CURRENT player's POV (POV-relative
encoding: own slot at 0, opponents at +1/+2/+3 in seat order).
)")
        .def("write_obs_pov",
             [](PyBatchedEnv& e, uint32_t env_idx, uint8_t pov, ArrF32 out) {
                 if (env_idx >= e.inner.n) throw std::runtime_error("env_idx out of range");
                 if (out.shape(0) != OBS_SIZE) throw std::runtime_error("obs buffer length mismatch");
                 nb::gil_scoped_release release;
                 write_obs(e.inner.states[env_idx], e.inner.layouts[env_idx], pov, out.data());
             },
             nb::arg("env_idx"), nb::arg("pov"), nb::arg("out"),
             "Single-env obs from a chosen POV (player 0..3). Used by "
             "PettingZoo AEC to render each agent's view.")
        .def("write_masks",
             [](PyBatchedEnv& e, ArrU64_2D out) {
                 if (out.shape(0) != e.inner.n || out.shape(1) != MASK_WORDS) {
                     throw std::runtime_error("mask buffer shape mismatch");
                 }
                 nb::gil_scoped_release release;
                 batched_env_write_masks(e.inner, out.data());
             },
             nb::arg("out"),
             R"(Fill (num_envs, MASK_WORDS) uint64 buffer in place.

Bit ``i`` of word ``w`` corresponds to action ID ``w*64 + i``. Action
IDs that are illegal in the current state have bit 0; legal actions
have bit 1.

This is a memcpy from the maintained-on-step ``s.action_mask`` field —
roughly free vs full recompute.
)")
        // Read-only state probes (for tests; not the hot path).
        .def("phase",
             [](const PyBatchedEnv& e, uint32_t i) -> uint8_t {
                 return uint8_t(e.inner.states[i].phase);
             }, nb::arg("env_idx"),
             "Current phase for env ``env_idx`` (0..3).")
        .def("current_player",
             [](const PyBatchedEnv& e, uint32_t i) -> uint8_t {
                 return e.inner.states[i].current_player;
             }, nb::arg("env_idx"),
             "Active player (0..3) for env ``env_idx``.")
        .def("player_vp",
             [](const PyBatchedEnv& e, uint32_t i, uint32_t pl) -> uint8_t {
                 return e.inner.states[i].player_vp[pl];
             }, nb::arg("env_idx"), nb::arg("player"),
             "Total VP for ``player`` in env ``env_idx`` (incl. hidden VP cards).")
        .def("last_winner",
             [](const PyBatchedEnv& e, uint32_t i) -> uint8_t {
                 return e.inner.last_winner[i];
             }, nb::arg("env_idx"),
             "Winner of the most recently completed game in ``env_idx`` "
             "(0..3), or 255 (NO_PLAYER) if no game has finished. Set on "
             "the terminating step; preserved across the auto-reset so "
             "wrappers can read it after ``step`` returns ``done=1``.")
        .def("snapshot",
             [](const PyBatchedEnv& e, uint32_t i) {
                 if (i >= e.inner.n)
                     throw std::runtime_error("env_idx out of range");
                 char buf[sizeof(GameState) + sizeof(BoardLayout)];
                 std::memcpy(buf, &e.inner.states[i], sizeof(GameState));
                 std::memcpy(buf + sizeof(GameState),
                              &e.inner.layouts[i], sizeof(BoardLayout));
                 return nb::bytes(buf, sizeof(buf));
             }, nb::arg("env_idx"),
             "Serialize env ``env_idx`` (GameState + BoardLayout) into a "
             "bytes object. Used by alpha-beta search to take a baseline "
             "snapshot, then explore branches via a scratch ``Env``.")
        .def("player_handsize",
             [](const PyBatchedEnv& e, uint32_t i, uint32_t pl) -> uint8_t {
                 return e.inner.states[i].player_handsize[pl];
             }, nb::arg("env_idx"), nb::arg("player"));
}
