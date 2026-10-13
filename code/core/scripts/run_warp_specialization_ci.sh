#!/usr/bin/env bash
# Run the warp specialization benchmark comparison with CI loop counts.
# WARP_SPEC_CI_ITERATIONS and WARP_SPEC_CI_WARMUP set the counts. Explicit
# --iterations or --warmup arguments override them because argparse keeps the
# last value it sees.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ITERATIONS="${WARP_SPEC_CI_ITERATIONS:-50}"
WARMUP="${WARP_SPEC_CI_WARMUP:-5}"

# Run as a module from code/ so the chapter and core packages import.
cd "${REPO_ROOT}"
"${PYTHON:-python3}" -m tests.test_warp_specialization \
    --iterations "${ITERATIONS}" \
    --warmup "${WARMUP}" \
    "$@"
