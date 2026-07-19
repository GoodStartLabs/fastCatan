#!/usr/bin/env bash
# Freeze guard (spec 0.3 Task B/4). Two independent checks; either failing blocks:
#   (1) git-diff vs the frozen substrate tag touches a frozen path;
#   (2) any frozen file's content differs from IMMUTABLE.lock (sha256 manifest).
# The lock catches drift even with no git history (exported tree); the tag-diff
# catches new files added under a frozen prefix. See IMMUTABLE.md.
#
#   scripts/check_frozen.sh [base-ref]     # default base: substrate-v1.1
set -euo pipefail
BASE="${1:-substrate-v1.1}"
cd "$(git rev-parse --show-toplevel)"

# Immutable path prefixes / exact files. The research surface under models/ is
# deliberately NOT listed (only the evaluator/promotion/schema files are). Keep
# identical to gen_immutable_lock.sh.
FROZEN_RE='^(src/catan/|include/|bindings/|EVAL/bridge/|tests/fuzz_invariants\.cpp$|tests/test_invariants\.py$|tests/test_determinism\.py$|tests/test_wiring\.py$|tests/test_state_mirror_layout\.py$|EVAL/AB/test_native_ab_fidelity\.py$|EVAL/AB/tournament\.py$|models/eval\.py$|models/selfplay/eval_seats\.py$|models/selfplay/gate\.py$|bin/|results/schema\.py$|results/SCHEMA\.md$|IMMUTABLE\.md$|scripts/check_frozen\.sh$|scripts/gen_immutable_lock\.sh$|scripts/check_env\.py$|scripts/tournament\.sh$)'

fail=0

# (1) tag-diff
if git rev-parse -q --verify "$BASE" >/dev/null; then
  changed="$(
    { git diff --name-only "$BASE"...HEAD
      git diff --name-only
      git diff --name-only --cached
      git ls-files --others --exclude-standard
    } | sort -u
  )"
  hits="$(printf '%s\n' "$changed" | grep -E "$FROZEN_RE" || true)"
  if [ -n "$hits" ]; then
    echo "FROZEN-LAYER VIOLATION vs $BASE — immutable paths changed:" >&2
    printf '  %s\n' $hits >&2
    fail=1
  fi
else
  echo "check_frozen: base ref '$BASE' not found — skipping tag-diff (lock check still runs)." >&2
fi

# (2) content-hash vs IMMUTABLE.lock
if [ -f IMMUTABLE.lock ]; then
  if ! sha256sum -c IMMUTABLE.lock --quiet 2>/dev/null; then
    echo "FROZEN-LAYER VIOLATION — contents differ from IMMUTABLE.lock:" >&2
    sha256sum -c IMMUTABLE.lock 2>/dev/null | grep -v ': OK$' >&2 || true
    fail=1
  fi
else
  echo "check_frozen: IMMUTABLE.lock missing — run scripts/gen_immutable_lock.sh." >&2
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "Edits to frozen paths require a human-reviewed PR, a regenerated IMMUTABLE.lock," >&2
  echo "and (if eval-visible) a ladder-version bump. See IMMUTABLE.md." >&2
  exit 1
fi
echo "check_frozen: OK — no immutable-layer drift vs $BASE and IMMUTABLE.lock verified."
