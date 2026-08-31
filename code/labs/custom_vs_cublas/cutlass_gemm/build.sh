#!/bin/bash
# Build the CUTLASS Blackwell GEMM library
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-100a;103a}"
BUILD_JOBS="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN)}"
PYTHON_EXECUTABLE="$(command -v "${PYTHON:-python3}")"

echo "Building CUTLASS Blackwell GEMM..."

mkdir -p "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
    "-DPython_EXECUTABLE=${PYTHON_EXECUTABLE}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    "$@"

cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"

echo ""
echo "Build complete! Library: ${BUILD_DIR}/cutlass_blackwell_gemm.so"
echo ""
echo "To use in Python:"
echo "  import torch"
echo "  import sys"
echo "  sys.path.insert(0, '${BUILD_DIR}')"
echo "  import cutlass_blackwell_gemm"
