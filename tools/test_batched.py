#!/usr/bin/env python3
"""Tests for BatchedEnv."""

from __future__ import annotations
import ctypes, sys, time
from pathlib import Path

NUM_RESOURCES = 5
NUM_PLAYERS = 4
PHASE_INITIAL_1 = 0
PHASE_MAIN = 2

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "build" / ("libfastcatan.dylib" if sys.platform == "darwin" else "libfastcatan.so")
if not LIB.exists(): print(f"missing {LIB}; build first"); sys.exit(1)
lib = ctypes.CDLL(str(LIB))

VP, U8, U16, U32, U64, I = (ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint16,
                             ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int)
F = ctypes.c_float
def _b(name, restype, *argtypes):
    f = getattr(lib, name); f.restype = restype; f.argtypes = list(argtypes); return f

obs_size = _b("fcatan_obs_size", U32)
MASK_WORDS = 5
NUM_ACTIONS = 286

bcreate  = _b("fbatched_create",  VP, U32, U64)
bdestroy = _b("fbatched_destroy", None, VP)
bnum     = _b("fbatched_num_envs", U32, VP)
breset   = _b("fbatched_reset",    None, VP)
bstep    = _b("fbatched_step",     None, VP, ctypes.POINTER(U32), ctypes.POINTER(F), ctypes.POINTER(U8))
bobs     = _b("fbatched_write_obs", None, VP, ctypes.POINTER(F))
bmask    = _b("fbatched_write_masks", None, VP, ctypes.POINTER(U64))

bphase   = _b("fbatched_phase", U8, VP, U32)
bcp      = _b("fbatched_current_player", U8, VP, U32)
bhand    = _b("fbatched_player_handsize", U8, VP, U32, I)
bbank    = _b("fbatched_bank", U8, VP, U32, I)
bres     = _b("fbatched_player_resource", U8, VP, U32, I, I)


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


def total_resources(env, i):
    return (sum(bbank(env, i, r) for r in range(5)) +
            sum(bres(env, i, p, r) for p in range(4) for r in range(5)))


def pick_random_legal(mask_arr, i, py_rng):
    """Find random legal action for env i from a flat (n × MASK_WORDS) buffer."""
    base = i * MASK_WORDS
    bits = []
    for w in range(MASK_WORDS):
        v = mask_arr[base + w]
        while v:
            lsb = v & (-v)
            idx = lsb.bit_length() - 1
            bits.append(w * 64 + idx)
            v ^= lsb
    return py_rng.choice(bits) if bits else 0


def test_create_destroy():
    e = bcreate(8, 42)
    fails = fail(bnum(e) == 8, f"num_envs = {bnum(e)}")
    bdestroy(e)
    return fails


def test_reset_creates_initial_phase():
    e = bcreate(16, 42)
    breset(e)
    fails = 0
    for i in range(16):
        fails += fail(bphase(e, i) == PHASE_INITIAL_1,
                      f"env {i} phase = {bphase(e, i)}")
        fails += fail(total_resources(e, i) == 95,
                      f"env {i} total != 95")
    bdestroy(e)
    return fails


def test_seed_independence_across_envs():
    """Different envs should have different boards (different seeds)."""
    e = bcreate(4, 99)
    breset(e)
    obs = (F * (4 * obs_size()))()
    bobs(e, obs)
    fails = 0
    obs_per_env = []
    for i in range(4):
        obs_per_env.append(tuple(obs[i*obs_size():(i+1)*obs_size()]))
    distinct = len(set(obs_per_env))
    fails += fail(distinct == 4, f"only {distinct}/4 envs distinct")
    bdestroy(e)
    return fails


def test_step_advances_all_envs():
    e = bcreate(8, 42)
    breset(e)
    masks = (U64 * (8 * MASK_WORDS))()
    bmask(e, masks)

    import random
    py_rng = random.Random(0)
    actions = (U32 * 8)()
    rewards = (F * 8)()
    dones = (U8 * 8)()
    for i in range(8):
        actions[i] = pick_random_legal(masks, i, py_rng)
    bstep(e, actions, rewards, dones)
    # All envs should have moved off phase 0 in their settle step (or stayed if action was illegal).
    fails = 0
    for i in range(8):
        fails += fail(total_resources(e, i) == 95, f"env {i} resource conservation broke")
    bdestroy(e)
    return fails


def test_auto_reset_on_done():
    """After step that triggers done, the env should be auto-reset to phase 0."""
    e = bcreate(4, 42)
    breset(e)
    # Run random play until any env hits done; verify auto-reset in subsequent step.
    import random
    py_rng = random.Random(7)
    masks = (U64 * (4 * MASK_WORDS))()
    actions = (U32 * 4)()
    rewards = (F * 4)()
    dones = (U8 * 4)()
    fails = 0
    seen_done = False
    for step in range(8000):
        bmask(e, masks)
        for i in range(4):
            actions[i] = pick_random_legal(masks, i, py_rng)
        bstep(e, actions, rewards, dones)
        for i in range(4):
            if dones[i]:
                seen_done = True
                # Next step's mask should reflect a fresh game (phase 0)
                fails += fail(bphase(e, i) == PHASE_INITIAL_1,
                              f"env {i} phase={bphase(e, i)} after done (want 0)")
                fails += fail(total_resources(e, i) == 95,
                              f"env {i} resources broken after auto-reset")
        if seen_done and step > 4000:
            break
    fails += fail(seen_done, "no env ever finished in 8000 steps")
    bdestroy(e)
    return fails


def test_mask_per_env_independent():
    e = bcreate(4, 42)
    breset(e)
    masks = (U64 * (4 * MASK_WORDS))()
    bmask(e, masks)
    # All envs are in initial-settle phase. Mask should have 54 set bits per env (all empty nodes legal)
    fails = 0
    for i in range(4):
        bits = 0
        for w in range(MASK_WORDS):
            bits += bin(masks[i * MASK_WORDS + w]).count('1')
        fails += fail(bits == 54, f"env {i} mask bits = {bits}")
    bdestroy(e)
    return fails


def test_throughput():
    """Rough throughput sanity check across batch sizes."""
    import random
    py_rng = random.Random(42)
    for n in (16, 256, 1024):
        e = bcreate(n, 42)
        breset(e)
        masks = (U64 * (n * MASK_WORDS))()
        actions = (U32 * n)()
        rewards = (F * n)()
        dones = (U8 * n)()

        # warmup
        for _ in range(10):
            bmask(e, masks)
            for i in range(n):
                actions[i] = pick_random_legal(masks, i, py_rng)
            bstep(e, actions, rewards, dones)

        n_steps = 200
        t0 = time.perf_counter()
        for _ in range(n_steps):
            bmask(e, masks)
            for i in range(n):
                actions[i] = pick_random_legal(masks, i, py_rng)
            bstep(e, actions, rewards, dones)
        dt = time.perf_counter() - t0
        steps_per_sec = n_steps * n / dt
        print(f"  n={n:4d}: {steps_per_sec:>11,.0f} steps/sec  ({dt*1000:.1f} ms for {n_steps} batches)")
        bdestroy(e)
    return 0


def main():
    total = 0
    print("== test_create_destroy ==");                 total += test_create_destroy()
    print("== test_reset_creates_initial_phase ==");    total += test_reset_creates_initial_phase()
    print("== test_seed_independence_across_envs =="); total += test_seed_independence_across_envs()
    print("== test_step_advances_all_envs ==");        total += test_step_advances_all_envs()
    print("== test_mask_per_env_independent ==");      total += test_mask_per_env_independent()
    print("== test_auto_reset_on_done ==");            total += test_auto_reset_on_done()
    print("== test_throughput ==");                    total += test_throughput()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} failures")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
