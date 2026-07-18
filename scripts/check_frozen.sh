#!/usr/bin/env bash
# Freeze guard (spec 0.3 Task 4). Fails if any change vs the frozen substrate tag
# touches an immutable path (see IMMUTABLE.md). Wire into pre-commit / CI.
#
#   scripts/check_frozen.sh [base-ref]     # default base: substrate-v1
#
# Catches committed drift since the tag AND uncommitted / untracked working-tree
# edits, so a frozen file cannot be changed without it being seen.
set -euo pipefail
BASE="${1:-substrate-v1}"
cd "$(git rev-parse --show-toplevel)"

if ! git rev-parse -q --verify "$BASE" >/dev/null; then
  echo "check_frozen: base ref '$BASE' not found (tag the frozen baseline first)." >&2
  exit 2
fi

# Immutable path prefixes / exact files. The research surface under models/ is
# deliberately NOT listed (only the evaluator/promotion files are).
FROZEN_RE='^(src/catan/|include/|bindings/|EVAL/bridge/|tests/fuzz_invariants\.cpp$|tests/test_invariants\.py$|tests/test_determinism\.py$|tests/test_wiring\.py$|tests/test_state_mirror_layout\.py$|EVAL/AB/test_native_ab_fidelity\.py$|EVAL/AB/tournament\.py$|models/eval\.py$|models/selfplay/eval_seats\.py$|models/selfplay/gate\.py$|bin/train_smoke\.py$|IMMUTABLE\.md$|scripts/check_frozen\.sh$|scripts/check_env\.py$|results/SCHEMA\.md$)'

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
  echo "Edits to frozen paths require a human-reviewed PR and (if eval-visible) a" >&2
  echo "ladder-version bump. See IMMUTABLE.md." >&2
  exit 1
fi
echo "check_frozen: OK — no immutable-layer drift vs $BASE"
