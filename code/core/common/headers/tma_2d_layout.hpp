#pragma once

#include <cstddef>
#include <cstdint>

namespace cuda_tma {

// Host-side metadata passed directly to cuTensorMapEncodeTiled. Keeping this
// CUDA-free makes the row-major descriptor contract testable without a GPU.
struct TensorMap2DLayout {
    std::uint64_t dimensions[2];
    std::uint64_t strides_bytes[1];
    std::uint32_t box[2];
    std::uint32_t element_strides[2];
};

inline TensorMap2DLayout make_2d_tensor_map_layout(
    int width, int height, int ld, int box_width, int box_height) {
    // Driver dimension 0 is contiguous (columns); dimension 1 advances by ld.
    // The transfer box and device coordinates use the same (column, row) order.
    return {
        {static_cast<std::uint64_t>(width), static_cast<std::uint64_t>(height)},
        {static_cast<std::uint64_t>(ld) * sizeof(float)},
        {static_cast<std::uint32_t>(box_width), static_cast<std::uint32_t>(box_height)},
        {1, 1},
    };
}

// Partial output tiles need element-sized stores: a TMA store may write a
// complete 16-byte granule at the right edge, clobbering row padding.
template <typename T, int Rows, int Cols>
#if defined(__CUDACC__)
__host__ __device__
#endif
inline void store_partial_2d_tile(
    const T (&tile)[Rows][Cols], T* output, int output_ld,
    int row_origin, int col_origin, int valid_rows, int valid_cols,
    int thread_index, int thread_count) {
    for (int index = thread_index; index < valid_rows * valid_cols; index += thread_count) {
        const int row = index / valid_cols;
        const int col = index % valid_cols;
        output[static_cast<std::size_t>(row_origin + row) * output_ld + col_origin + col] =
            tile[row][col];
    }
}

}  // namespace cuda_tma
