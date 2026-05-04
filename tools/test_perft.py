#!/usr/bin/env python3
"""Perft-style determinism regression test.

For a fixed (seed, n_steps), running the env with a deterministic action
picker (lowest-id legal action) must always produce the same final state
hash. Pinned hashes are stored below; CI fails on any drift.

Two failure modes this catches:
  1. Same code, two runs → different hash : nondeterminism / TSan-class bug
  2. Same code today, code change later → different hash from pinned :
     trajectory-affecting regression (RNG sequence, rules drift, struct layout)

Usage:
    python3 tools/test_perft.py             # verify against pinned hashes
    python3 tools/test_perft.py --pin       # recompute and rewrite the pin file

Hashes are platform-portable as long as struct layout is stable across
compilers. They MAY differ between macOS/Linux if compilers add different
padding to GameState — but our struct uses alignas(64) and has no implicit
padding, so it should be portable.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "build" / ("libfastcatan.dylib" if sys.platform == "darwin" else "libfastcatan.so")
PIN_FILE = REPO / "tools" / "perft_hashes.json"

if not LIB.exists():
    print(f"missing {LIB}; run `bash tools/build_lib.sh`")
    sys.exit(1)
lib = ctypes.CDLL(str(LIB))

VP, U8, U16, U32, U64, I = (ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint16,
                             ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int)
def _b(name, restype, *argtypes):
    f = getattr(lib, name); f.restype = restype; f.argtypes = list(argtypes); return f

create     = _b("fcatan_create",     VP)
destroy    = _b("fcatan_destroy",    None, VP)
reset_     = _b("fcatan_reset",      None, VP, U64)
step_      = _b("fcatan_step",       U8,   VP, U32)
phase      = _b("fcatan_phase",      U8,   VP)
state_hash = _b("fcatan_state_hash", U64,  VP)
compute_mask = _b("fcatan_compute_mask", None, VP, ctypes.POINTER(U64))

MASK_WORDS = 5
PHASE_ENDED = 3


def lowest_legal_action(env) -> int:
    """Always pick the action with the lowest ID. Fully deterministic."""
    arr = (U64 * MASK_WORDS)()
    compute_mask(env, arr)
    for w in range(MASK_WORDS):
        v = int(arr[w])
        if v:
            return w * 64 + (v & -v).bit_length() - 1
    return -1


def run_perft(seed: int, n_steps: int) -> tuple[int, int, int]:
    """Run trajectory; return (final_hash, steps_taken, terminated_early)."""
    env = create()
    reset_(env, seed)
    steps = 0
    terminated_early = 0
    for _ in range(n_steps):
        if phase(env) == PHASE_ENDED:
            terminated_early = 1
            break
        a = lowest_legal_action(env)
        if a < 0:
            terminated_early = 1
            break
        step_(env, a)
        steps += 1
    h = int(state_hash(env))
    destroy(env)
    return h, steps, terminated_early


# (seed, n_steps) — varied to cover initial placement, MAIN, sub-phases.
TRAJECTORIES = [
    (42, 50),
    (42, 200),
    (42, 1000),
    (7, 100),
    (7, 500),
    (123, 1000),
    (999, 2000),
    (0, 100),
    (1, 100),
    (2, 100),
]


def cmd_pin():
    pins = {}
    for seed, n_steps in TRAJECTORIES:
        h, steps, term = run_perft(seed, n_steps)
        key = f"seed={seed} steps={n_steps}"
        pins[key] = {"hash": h, "steps_taken": steps, "terminated_early": term}
        print(f"  {key}: hash=0x{h:016x} steps={steps} term={term}")
    PIN_FILE.write_text(json.dumps(pins, indent=2))
    print(f"\nwrote {PIN_FILE}")


def cmd_verify():
    if not PIN_FILE.exists():
        print(f"no pin file at {PIN_FILE}; run with --pin first")
        sys.exit(2)
    pins = json.loads(PIN_FILE.read_text())
    fails = 0
    for seed, n_steps in TRAJECTORIES:
        key = f"seed={seed} steps={n_steps}"
        if key not in pins:
            print(f"  SKIP: no pin for {key}")
            continue
        expected = pins[key]
        h, steps, term = run_perft(seed, n_steps)
        ok = (h == expected["hash"]
              and steps == expected["steps_taken"]
              and term == expected["terminated_early"])
        if ok:
            print(f"  ok  {key}: 0x{h:016x}")
        else:
            fails += 1
            print(f"  FAIL {key}:")
            print(f"     pinned: hash=0x{expected['hash']:016x} "
                  f"steps={expected['steps_taken']} term={expected['terminated_early']}")
            print(f"     actual: hash=0x{h:016x} steps={steps} term={term}")

    # Also verify replay-twice determinism for each trajectory (cheap cross-check).
    print("\n--- replay-twice determinism ---")
    for seed, n_steps in TRAJECTORIES:
        h1, _, _ = run_perft(seed, n_steps)
        h2, _, _ = run_perft(seed, n_steps)
        if h1 != h2:
            fails += 1
            print(f"  FAIL determinism seed={seed} steps={n_steps}: "
                  f"first=0x{h1:016x} second=0x{h2:016x}")
    print(f"\n{'ALL PERFT HASHES MATCH' if fails == 0 else f'FAIL — {fails} mismatches'}")
    sys.exit(0 if fails == 0 else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", action="store_true", help="recompute + rewrite the pin file")
    args = ap.parse_args()
    if args.pin:
        cmd_pin()
    else:
        cmd_verify()


if __name__ == "__main__":
    main()
