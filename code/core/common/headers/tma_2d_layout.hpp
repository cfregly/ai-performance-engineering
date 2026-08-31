#pragma once

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

}  // namespace cuda_tma
