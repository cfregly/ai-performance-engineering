#!/usr/bin/env bash
# Run warp specialization comparison tests in CI mode with longer iterations.
# This script ensures CI-style settings regardless of the calling environment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Force CI mode so tests/test_warp_specialization.py picks high-iteration loops
export CI=1

# Set warp-specialization specific knobs for CI mode via CLI flags
# Note: WARP_SPEC_CI_* env vars are no longer supported - use CLI flags instead
# Allow override via env vars for backward compatibility, but prefer CLI flags
ITERATIONS="${WARP_SPEC_CI_ITERATIONS:-50}"
WARMUP="${WARP_SPEC_CI_WARMUP:-5}"

"${PYTHON:-python3}" "${REPO_ROOT}/tests/test_warp_specialization.py" \
    --iterations "${ITERATIONS}" \
    --warmup "${WARMUP}" \
    "$@"
