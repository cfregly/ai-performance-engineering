#pragma once

#include <cstddef>

namespace aisp {

// Offset for cuBLASLt's CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 layout.
// Scale factors are tiled as 128 operand rows by four K-axis scale columns.
// Callers must provide a reduction dimension divisible by 64 and a row count
// divisible by 128 so the resulting offsets cover the scale buffer exactly.
inline constexpr std::size_t nvfp4_vec16_scale_offset(
    int row,
    int scale_column,
    int reduction_dim) {
    const int scale_tiles_k = (reduction_dim / 16) / 4;
    return static_cast<std::size_t>(row / 128) * 512 * scale_tiles_k
        + static_cast<std::size_t>(scale_column / 4) * 512
        + static_cast<std::size_t>(row % 32) * 16
        + static_cast<std::size_t>((row % 128) / 32) * 4
        + static_cast<std::size_t>(scale_column % 4);
}

}  // namespace aisp
