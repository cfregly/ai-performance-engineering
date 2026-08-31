#!/usr/bin/env bash
#
# Build helper for the CUDA 13 Blackwell TMA demonstrations.
# This script compiles:
#   - ch07/async_prefetch_2d_demo (2D TMA copy)
#   - ch10/tma_2d_pipeline_blackwell (2D TMA tile pipeline)
#
# Usage:
#   ./core/scripts/build_tma_demos.sh [--arch sm_100] [--dry-run]
#
# Requires CUDA 13.0+ with nvcc on PATH.

set -euo pipefail

ARCH="sm_100"
MAKE_ARGS=()
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      ARCH="${2:?--arch requires an argument (e.g. sm_100)}"
      shift 2
      ;;
    --dry-run)
      MAKE_ARGS+=(-n)
      DRY_RUN=1
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      echo "Unexpected positional argument: $1" >&2
      exit 1
      ;;
  esac
done

case "${ARCH}" in
  sm_100|sm_103|sm_120|sm_121) ;;
  *) echo "Unsupported ARCH=${ARCH}; expected sm_100, sm_103, sm_120, or sm_121" >&2; exit 2 ;;
esac
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
suffix="_${ARCH/_/}"

echo ">>> Building Blackwell TMA demos (ARCH=${ARCH})"

echo "-> ch07/async_prefetch_2d_demo${suffix}"
make "${MAKE_ARGS[@]}" -C "${repo_root}/ch07" ARCH="${ARCH}" USE_ARCH_SUFFIX=1 "async_prefetch_2d_demo${suffix}"

echo "-> ch10/tma_2d_pipeline_blackwell${suffix}"
make "${MAKE_ARGS[@]}" -C "${repo_root}/ch10" ARCH="${ARCH}" USE_ARCH_SUFFIX=1 "tma_2d_pipeline_blackwell${suffix}"

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "Dry run complete; no binaries were built. Planned output paths:"
else
  echo "Builds complete. Binaries:"
fi
echo "  - ${repo_root}/ch07/async_prefetch_2d_demo${suffix}"
echo "  - ${repo_root}/ch10/tma_2d_pipeline_blackwell${suffix}"
