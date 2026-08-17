#!/usr/bin/env bash
# Run dual-architecture builds across CUDA chapters.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPARE_BUILD_JOBS="${COMPARE_BUILD_JOBS:-2}"
if [[ ! "${COMPARE_BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "COMPARE_BUILD_JOBS must be a positive integer" >&2
  exit 2
fi

CHAPTERS=(
  ch01
  ch02
  ch04
  ch06
  ch07
  ch08
  ch09
  ch10
  ch11
  ch12
)

echo "=== Dual-architecture compare builds ==="
for chapter in "${CHAPTERS[@]}"; do
  echo ""
  echo ">>> ${chapter}: make compare"
  (cd "${REPO_ROOT}/${chapter}" && make --jobs="${COMPARE_BUILD_JOBS}" compare)
done

echo ""
echo "All dual-architecture builds completed successfully."
