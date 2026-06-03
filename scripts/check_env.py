#!/usr/bin/env python3
"""Assert this interpreter matches the canonical fastCatan env on EVERY device.

Run after setup (scripts/setup.sh does, last step) and any time a checkpoint won't
load or a device "feels off". Prints every component, exits non-zero on any drift.
The pins here MIRROR requirements.txt — the WHY (cp312 native build + numpy 1.x
pickle compat) lives there. If you bump a pin, bump it in BOTH places.
"""
from __future__ import annotations

import sys

EXPECT = {
    "python": (3, 12),
    "numpy": "1.26.4",
    "torch": "2.5.1",
    "stable_baselines3": "2.7.0",
    "sb3_contrib": "2.7.1",
    "gymnasium": "0.29.1",
    "cloudpickle": "2.2.1",
}

problems: list[str] = []


def report(name: str, got, want, ok: bool) -> None:
    print(f"  [{'OK' if ok else 'XX'}] {name:<20} {got}   (want {want})")
    if not ok:
        problems.append(f"{name}: {got} != {want}")


# interpreter — only major.minor matters (cp312 native build / SB3 pickle compat).
py = sys.version_info[:2]
report("python", ".".join(map(str, py)), ".".join(map(str, EXPECT["python"])), py == EXPECT["python"])

# torch may carry a +cu124 / +cpu local tag; compare the public version only.
for mod in ("numpy", "torch", "stable_baselines3", "sb3_contrib", "gymnasium", "cloudpickle"):
    want = EXPECT[mod]
    try:
        got = getattr(__import__(mod), "__version__", "?")
        report(mod, got, want, got.split("+", 1)[0] == want)
    except Exception as e:  # noqa: BLE001
        problems.append(f"{mod}: import failed ({e})")
        print(f"  [XX] {mod:<20} IMPORT FAILED: {e}")

# native extension present + built for THIS interpreter (cp312).
try:
    import fastcatan

    print(f"  [OK] {'fastcatan':<20} {fastcatan.OBS_SIZE} obs / {fastcatan.NUM_ACTIONS} actions")
except Exception as e:  # noqa: BLE001
    problems.append(f"fastcatan: import failed ({e}) — rebuild: pip install -e . --no-build-isolation")
    print(f"  [XX] {'fastcatan':<20} IMPORT FAILED: {e}")

# catanatron (M4 eval opponent).
try:
    import catanatron  # noqa: F401

    print(f"  [OK] {'catanatron':<20} import ok")
except Exception as e:  # noqa: BLE001
    problems.append(f"catanatron: import failed ({e})")
    print(f"  [XX] {'catanatron':<20} IMPORT FAILED: {e}")

if problems:
    print(f"\nENV DRIFT — {len(problems)} mismatch(es):", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    print("\nFix:  bash scripts/setup.sh   (canonical pins live in requirements.txt)", file=sys.stderr)
    sys.exit(1)

print("\nENV OK — matches canonical fastCatan spec (conda `catan`: py3.12 + numpy 1.26.4).")
