#!/usr/bin/env bash
# Run dual-architecture builds across CUDA chapters.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
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
  (cd "${REPO_ROOT}/${chapter}" && make compare)
done

echo ""
echo "All dual-architecture builds completed successfully."
