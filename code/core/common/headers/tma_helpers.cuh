#pragma once

/**
 * tma_helpers.cuh - Tensor Memory Accelerator (TMA) utility functions
 * 
 * Provides helpers for TMA bulk async operations (introduced in Hopper SM 9.0,
 * enhanced in Blackwell SM 10.0 and Grace-Blackwell SM 12.1).
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "arch_detection.cuh"
#include "tma_2d_layout.hpp"

namespace cuda_tma {

// Error checking helpers
inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        std::abort();
    }
}

inline void check_cu(CUresult res, const char* what) {
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(res, &err_str);
        std::fprintf(stderr, "CUDA driver error (%s): %s\n", what, err_str ? err_str : "unknown");
        std::abort();
    }
}

inline bool device_supports_tma() {
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return false;
    }
    if (prop.major < 9) {
        return false;
    }

    int tensor_map_supported = 0;
    CUresult cu_res = cuDeviceGetAttribute(
        &tensor_map_supported,
        CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED,
        device);
    if (cu_res != CUDA_SUCCESS || tensor_map_supported == 0) {
        return false;
    }

    return true;  // Hopper (SM 90) and newer introduce TMA
}

inline PFN_cuTensorMapEncodeTiled_v12000 load_cuTensorMapEncodeTiled() {
    void* func_ptr = nullptr;
    cudaDriverEntryPointQueryResult query_result{};

    cudaError_t err = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled",
        &func_ptr,
        13000,  // Prefer CUDA 13 implementation when available
        cudaEnableDefault,
        &query_result);

    if (err != cudaSuccess || query_result != cudaDriverEntryPointSuccess) {
        // Fallback to CUDA 12.x entry point for Hopper if CUDA 13 is unavailable.
        err = cudaGetDriverEntryPointByVersion(
            "cuTensorMapEncodeTiled",
            &func_ptr,
            12000,
            cudaEnableDefault,
            &query_result);
    }

    if (err != cudaSuccess || query_result != cudaDriverEntryPointSuccess || func_ptr == nullptr) {
        std::fprintf(stderr, "cuTensorMapEncodeTiled unavailable on this runtime.\n");
        return nullptr;
    }

    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(func_ptr);
}

inline bool make_2d_tensor_map(
    CUtensorMap& desc,
    PFN_cuTensorMapEncodeTiled_v12000 encode,
    void* base,
    int width,
    int height,
    int ld,
    int box_width,
    int box_height,
    CUtensorMapSwizzle swizzle_mode) {
    // Row-major float matrix: dimension 0 is columns, dimension 1 is rows.
    // Device TMA coordinates must be (column, row), matching box_width/box_height.
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
    constexpr uint32_t rank = 2;
    const auto layout = make_2d_tensor_map_layout(width, height, ld, box_width, box_height);
    const auto& dims = layout.dimensions;
    const auto& stride = layout.strides_bytes;
    const auto& box = layout.box;
    const auto& elem_stride = layout.element_strides;

    constexpr auto interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr auto promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr auto oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    std::printf(
        "[TMA] Encoding 2D tensor map: base=%p dims={%llu,%llu} stride_bytes={%llu} box={%u,%u} "
        "elem_stride={%u,%u} interleave=%d swizzle=%d l2=%d oob=%d\n",
        base,
        static_cast<unsigned long long>(dims[0]),
        static_cast<unsigned long long>(dims[1]),
        static_cast<unsigned long long>(stride[0]),
        box[0],
        box[1],
        elem_stride[0],
        elem_stride[1],
        static_cast<int>(interleave),
        static_cast<int>(swizzle_mode),
        static_cast<int>(promotion),
        static_cast<int>(oob_fill));

    auto fn = encode ? encode : cuTensorMapEncodeTiled;
    CUresult res = fn(
        &desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        rank,
        base,
        dims,
        stride,
        box,
        elem_stride,
        interleave,
        swizzle_mode,
        promotion,
        oob_fill);
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        const char* err_name = nullptr;
        cuGetErrorString(res, &err_str);
        cuGetErrorName(res, &err_name);
        std::fprintf(stderr,
                     "[TMA] cuTensorMapEncodeTiled (2D) failed: %s (%s, %d) "
                     "(dataType=%d rank=%u base=%p dims={%llu,%llu} stride={%llu} box={%u,%u} "
                     "elem_stride={%u,%u} interleave=%d swizzle=%d l2=%d oob=%d)\n",
                     err_str ? err_str : "unknown",
                     err_name ? err_name : "unknown",
                     static_cast<int>(res),
                     CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                     rank,
                     base,
                     static_cast<unsigned long long>(dims[0]),
                     static_cast<unsigned long long>(dims[1]),
                     static_cast<unsigned long long>(stride[0]),
                     box[0],
                     box[1],
                     elem_stride[0],
                     elem_stride[1],
                     static_cast<int>(interleave),
                     static_cast<int>(swizzle_mode),
                     static_cast<int>(promotion),
                     static_cast<int>(oob_fill));
        return false;
    }
    std::printf("[TMA] 2D descriptor ok (res=%d): base=%p dims={%llu,%llu} stride_bytes={%llu} box={%u,%u} "
                "elem_stride={%u,%u} interleave=%d swizzle=%d l2=%d oob=%d\n",
                static_cast<int>(res),
                base,
                static_cast<unsigned long long>(dims[0]),
                static_cast<unsigned long long>(dims[1]),
                static_cast<unsigned long long>(stride[0]),
                box[0],
                box[1],
                elem_stride[0],
                elem_stride[1],
                static_cast<int>(interleave),
                static_cast<int>(swizzle_mode),
                static_cast<int>(promotion),
                static_cast<int>(oob_fill));
    return true;
}

inline bool make_1d_tensor_map(
    CUtensorMap& desc,
    PFN_cuTensorMapEncodeTiled_v12000 encode,
    void* base,
    int elements,
    int box_elements,
    CUtensorMapSwizzle swizzle_mode = CU_TENSOR_MAP_SWIZZLE_NONE) {
    constexpr uint32_t rank = 1;
    std::uint64_t dims[rank] = {static_cast<std::uint64_t>(elements)};
    std::uint64_t stride_bytes[rank] = {static_cast<std::uint64_t>(sizeof(float))};
    // Query architecture-specific TMA limits
    cuda_arch::TMALimits limits = cuda_arch::get_tma_limits();
    std::uint32_t box[rank] = {std::min(static_cast<std::uint32_t>(box_elements), limits.max_1d_box_size)};
    std::uint32_t elem_stride[rank] = {1};

    constexpr auto interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr auto promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    constexpr auto oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    std::printf(
        "[TMA] Encoding 1D tensor map: base=%p elements=%d stride_bytes=%llu box=%u elem_stride=%u interleave=%d "
        "swizzle=%d l2=%d oob=%d\n",
        base,
        elements,
        static_cast<unsigned long long>(stride_bytes[0]),
        box[0],
        elem_stride[0],
        static_cast<int>(interleave),
        static_cast<int>(swizzle_mode),
        static_cast<int>(promotion),
        static_cast<int>(oob_fill));

    auto fn = encode ? encode : cuTensorMapEncodeTiled;
    CUresult res = fn(
        &desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        rank,
        base,
        dims,
        stride_bytes,
        box,
        elem_stride,
        interleave,
        swizzle_mode,
        promotion,
        oob_fill);
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        const char* err_name = nullptr;
        cuGetErrorString(res, &err_str);
        cuGetErrorName(res, &err_name);
        std::fprintf(stderr,
                     "[TMA] cuTensorMapEncodeTiled (1D) failed: %s (%s, %d) (dataType=%d rank=%u base=%p elements=%d "
                     "stride_bytes=%llu box=%u elem_stride=%u interleave=%d swizzle=%d l2=%d oob=%d)\n",
                     err_str ? err_str : "unknown",
                     err_name ? err_name : "unknown",
                     static_cast<int>(res),
                     CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                     rank,
                     base,
                     elements,
                     static_cast<unsigned long long>(stride_bytes[0]),
                     box[0],
                     elem_stride[0],
                     static_cast<int>(interleave),
                     static_cast<int>(swizzle_mode),
                     static_cast<int>(promotion),
                     static_cast<int>(oob_fill));
        return false;
    }
    std::printf("[TMA] 1D descriptor ok (res=%d): base=%p elements=%d stride_bytes=%llu box=%u elem_stride=%u "
                "interleave=%d swizzle=%d l2=%d oob=%d\n",
                static_cast<int>(res),
                base,
                elements,
                static_cast<unsigned long long>(stride_bytes[0]),
                box[0],
                elem_stride[0],
                static_cast<int>(interleave),
                static_cast<int>(swizzle_mode),
                static_cast<int>(promotion),
                static_cast<int>(oob_fill));
    return true;
}

}  // namespace cuda_tma
