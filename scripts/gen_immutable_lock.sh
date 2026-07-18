#!/usr/bin/env bash
# Regenerate IMMUTABLE.lock — a sha256 manifest of every frozen file (IMMUTABLE.md).
# Run this in the SAME reviewed commit as any intentional edit to a frozen file, so
# the content-hash guard in check_frozen.sh stays green only for deliberate changes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Keep this list identical to check_frozen.sh's FROZEN_RE.
FROZEN_RE='^(src/catan/|include/|bindings/|EVAL/bridge/|tests/fuzz_invariants\.cpp$|tests/test_invariants\.py$|tests/test_determinism\.py$|tests/test_wiring\.py$|tests/test_state_mirror_layout\.py$|EVAL/AB/test_native_ab_fidelity\.py$|EVAL/AB/tournament\.py$|models/eval\.py$|models/selfplay/eval_seats\.py$|models/selfplay/gate\.py$|bin/|results/schema\.py$|results/SCHEMA\.md$|IMMUTABLE\.md$|scripts/check_frozen\.sh$|scripts/gen_immutable_lock\.sh$|scripts/check_env\.py$|scripts/tournament\.sh$)'

git ls-files | grep -E "$FROZEN_RE" | LC_ALL=C sort | while read -r f; do
  printf '%s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$f"
done > IMMUTABLE.lock
echo "wrote IMMUTABLE.lock ($(wc -l < IMMUTABLE.lock) frozen files)"
