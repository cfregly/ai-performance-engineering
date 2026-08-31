// Real CUDA correctness gate for each production shared-2D-descriptor caller.
// Compile one TMA_VALIDATION_CASE at a time via run_tma_2d_layout_validation.py.
// Including the sample gives this gate the actual production kernel; no copied
// kernel or CPU emulation is used. CPU code computes independent expected values.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "../../core/common/headers/tma_helpers.cuh"

#ifndef TMA_VALIDATION_CASE
#error "Select a production caller with -DTMA_VALIDATION_CASE=0 through 6"
#endif
#if CUDART_VERSION < 13000
#error "This production-caller correctness gate requires CUDA 13.0 or newer"
#endif
#if !defined(TMA_MULTICAST_TARGET) || (TMA_MULTICAST_TARGET != 100 && TMA_MULTICAST_TARGET != 103)
#error "Select a supported architecture through run_tma_2d_layout_validation.py"
#endif

#define main tma_sample_main
#if TMA_VALIDATION_CASE == 0
#include "../../ch07/async_prefetch_2d_demo.cu"
#elif TMA_VALIDATION_CASE == 1
#include "../../ch07/optimized_tma_bulk_tensor_2d.cu"
#elif TMA_VALIDATION_CASE == 2
#include "../../ch07/optimized_tma_copy.cu"
#elif TMA_VALIDATION_CASE == 3
#include "../../ch10/tma_2d_pipeline_blackwell.cu"
#elif TMA_VALIDATION_CASE == 4
#include "../../core/common/headers/cuda13_demos.cuh"
#elif TMA_VALIDATION_CASE == 5
#include "../../ch10/tma_multicast_cluster.cu"
#elif TMA_VALIDATION_CASE == 6
#include "../../ch10/tma_multicast_baseline.cu"
#else
#error "TMA_VALIDATION_CASE must select a production caller (0 through 6)"
#endif
#undef main

namespace {

constexpr int kGuardElements = 64;  // Preserve 256-byte base alignment.
constexpr float kCanary = -9876.5f;

struct DeviceFloats {
    float* data = nullptr;
    explicit DeviceFloats(const std::vector<float>& values) {
        cuda_tma::check_cuda(cudaMalloc(&data, values.size() * sizeof(float)), "validation allocation");
        cuda_tma::check_cuda(cudaMemcpy(data, values.data(), values.size() * sizeof(float),
                                       cudaMemcpyHostToDevice), "validation upload");
    }
    ~DeviceFloats() { cudaFree(data); }
    DeviceFloats(const DeviceFloats&) = delete;
    DeviceFloats& operator=(const DeviceFloats&) = delete;
};

bool compare_every_element(const DeviceFloats& actual, const std::vector<float>& expected) {
    std::vector<float> observed(expected.size());
    cuda_tma::check_cuda(cudaGetLastError(), "validation kernel launch");
    cuda_tma::check_cuda(cudaDeviceSynchronize(), "validation kernel completion");
    cuda_tma::check_cuda(cudaMemcpy(observed.data(), actual.data, observed.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost), "validation readback");
    std::size_t mismatches = 0;
    double max_abs_error = 0.0;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        const double error = std::abs(static_cast<double>(observed[i]) - expected[i]);
        const double tolerance = expected[i] == kCanary ? 0.0 : 1e-5 + 1e-6 * std::abs(expected[i]);
        if (!std::isfinite(observed[i]) || error > tolerance) {
            if (mismatches < 8) {
                std::fprintf(stderr, "Mismatch at storage[%zu]: actual=%.9g expected=%.9g\n",
                             i, observed[i], expected[i]);
            }
            ++mismatches;
        }
        max_abs_error = std::max(max_abs_error, error);
    }
    std::printf("TMA_LAYOUT_COMPARE case=%d elements=%zu mismatches=%zu max_abs_error=%.9g\n",
                TMA_VALIDATION_CASE, observed.size(), mismatches, max_abs_error);
    return mismatches == 0;
}

#if TMA_VALIDATION_CASE <= 4
template <int BoxCols, int BoxRows, int Stages = 1>
bool verify_copy_case(int height, int width, int ld) {
    const std::size_t storage_size = static_cast<std::size_t>(height) * ld + 2 * kGuardElements;
    std::vector<float> input(storage_size, kCanary);
    std::vector<float> expected(storage_size, kCanary);
    auto input_at = [&](int row, int col) {
        return input[kGuardElements + static_cast<std::size_t>(row) * ld + col];
    };
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            // Unequal row/column contributions reveal transposes and origin swaps.
            input[kGuardElements + static_cast<std::size_t>(row) * ld + col] =
                static_cast<float>(row * 1009 + col * 3) + 0.25f;
        }
    }

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            float value = input_at(row, col);
#if TMA_VALIDATION_CASE == 2
            const int row_origin = (row / kTile2D_M) * kTile2D_M;
            const int col_origin = (col / kTile2D_N) * kTile2D_N;
            const int rows = std::min(kTile2D_M, height - row_origin);
            const int cols = std::min(kTile2D_N, width - col_origin);
            const int local = (row - row_origin) * cols + col - col_origin;
            const int near = std::min(local + 1, rows * cols - 1);
            const int far = std::min(local + kLookahead, rows * cols - 1);
            value = std::fma(input_at(row_origin + far / cols, col_origin + far % cols), 0.125f,
                            std::fma(input_at(row_origin + near / cols, col_origin + near % cols),
                                     0.25f, value * 0.75f));
#elif TMA_VALIDATION_CASE == 3
            value = std::fma(value, 1.0001f, 0.0001f);
#elif TMA_VALIDATION_CASE == 4
            value *= 1.5f;
#endif
            expected[kGuardElements + static_cast<std::size_t>(row) * ld + col] = value;
        }
    }

    DeviceFloats source(input);
    DeviceFloats destination(std::vector<float>(storage_size, kCanary));
    float* src = source.data + kGuardElements;
    float* dst = destination.data + kGuardElements;
    CUtensorMap in_desc{}, out_desc{};
    const auto encode = cuda_tma::load_cuTensorMapEncodeTiled();
    if (!encode || !cuda_tma::make_2d_tensor_map(in_desc, encode, src, width, height, ld,
                                               BoxCols, BoxRows, CU_TENSOR_MAP_SWIZZLE_NONE) ||
        !cuda_tma::make_2d_tensor_map(out_desc, encode, dst, width, height, ld,
                                    BoxCols, BoxRows, CU_TENSOR_MAP_SWIZZLE_NONE)) {
        return false;
    }

    const dim3 grid((width + BoxCols - 1) / BoxCols, (height + BoxRows - 1) / BoxRows);
#if TMA_VALIDATION_CASE == 0
    tma_copy_2d_kernel<128, 64><<<grid, dim3(16, 8)>>>(in_desc, out_desc, height, width);
#elif TMA_VALIDATION_CASE == 1
    tma_bulk_copy_kernel<128, 64><<<grid, dim3(32)>>>(in_desc, out_desc, width, height);
#elif TMA_VALIDATION_CASE == 2
    descriptor_tma_2d_copy_kernel<kTile2D_M, kTile2D_N><<<grid, dim3(16, 16)>>>(
        in_desc, out_desc, height, width);
#elif TMA_VALIDATION_CASE == 3
    const dim3 pipeline_grid((width + BoxCols - 1) / BoxCols, (height + TILE_M - 1) / TILE_M);
    tma_2d_pipeline_kernel<BoxCols, BoxRows, Stages><<<pipeline_grid, dim3(32, 4)>>>(
        in_desc, out_desc, dst, height, width, ld);
#elif TMA_VALIDATION_CASE == 4
    cuda13_demos::tma_copy_kernel<BoxRows, BoxCols><<<grid, dim3(32, 4)>>>(
        in_desc, out_desc, dst, width, height, ld);
#endif
    std::printf("TMA_LAYOUT_SHAPE case=%d height=%d width=%d ld=%d box_cols=%d box_rows=%d stages=%d\n",
                TMA_VALIDATION_CASE, height, width, ld, BoxCols, BoxRows, Stages);
    return compare_every_element(destination, expected);
}

template <int BoxCols, int BoxRows, int Stages = 1>
bool verify_copy_shapes() {
    return verify_copy_case<BoxCols, BoxRows, Stages>(256, 384, 384) &&
           verify_copy_case<BoxCols, BoxRows, Stages>(259, 196, 208) &&
           verify_copy_case<BoxCols, BoxRows, Stages>(129, 385, 400);
}
#else
bool verify_gemm_case(int m, int n, int k) {
    std::vector<float> a(static_cast<std::size_t>(m) * k);
    std::vector<float> b(static_cast<std::size_t>(k) * n);
    std::vector<float> expected(static_cast<std::size_t>(m) * n + 2 * kGuardElements, kCanary);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            a[static_cast<std::size_t>(row) * k + col] = ((row * 17 + col * 3) % 29 - 14) * 0.0625f;
        }
    }
    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            b[static_cast<std::size_t>(row) * n + col] = ((row * 31 + col * 7) % 37 - 18) * 0.03125f;
        }
    }
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double value = 0.0;
            for (int inner = 0; inner < k; ++inner) {
                value += static_cast<double>(a[static_cast<std::size_t>(row) * k + inner]) *
                         b[static_cast<std::size_t>(inner) * n + col];
            }
            expected[kGuardElements + static_cast<std::size_t>(row) * n + col] = static_cast<float>(value);
        }
    }
    DeviceFloats d_a(a), d_b(b), d_c(std::vector<float>(expected.size(), kCanary));
    CUtensorMap b_desc{};
#if TMA_VALIDATION_CASE == 5
    const auto kernel = tma_multicast_gemm_kernel;
    const auto encode = cuda_tma::load_cuTensorMapEncodeTiled();
    if (!encode || !cuda_tma::make_2d_tensor_map(b_desc, encode, d_b.data, n, k, n,
                                               TILE_N, TILE_K, CU_TENSOR_MAP_SWIZZLE_NONE)) {
        return false;
    }
#else
    const auto kernel = tma_nomulticast_gemm_kernel;
#endif
    cuda_tma::check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             DYNAMIC_SMEM_BYTES), "validation dynamic shared memory");
    cuda_tma::check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
                        "validation cluster size");
    const int m_tiles = (m + TILE_M - 1) / TILE_M;
    cudaLaunchAttribute attributes[1]{};
    attributes[0].id = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim = {CLUSTER_M, CLUSTER_N, 1};
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(((m_tiles + CLUSTER_M - 1) / CLUSTER_M) * CLUSTER_M,
                          (n + TILE_N - 1) / TILE_N);
    config.blockDim = dim3(BLOCK_SIZE);
    config.dynamicSmemBytes = DYNAMIC_SMEM_BYTES;
    config.attrs = attributes;
    config.numAttrs = 1;
#if TMA_VALIDATION_CASE == 5
    cuda_tma::check_cuda(cudaLaunchKernelEx(&config, kernel, b_desc, d_a.data, d_b.data,
                                           d_c.data + kGuardElements, m, n, k), "validation multicast GEMM");
#else
    cuda_tma::check_cuda(cudaLaunchKernelEx(&config, kernel, d_a.data, d_b.data,
                                           d_c.data + kGuardElements, m, n, k), "validation baseline GEMM");
#endif
    std::printf("TMA_LAYOUT_SHAPE case=%d m=%d n=%d k=%d\n", TMA_VALIDATION_CASE, m, n, k);
    return compare_every_element(d_c, expected);
}
#endif

}  // namespace

int main() {
    cuda_tma::check_cuda(cudaSetDevice(0), "validation device selection");
    cudaDeviceProp device{};
    cuda_tma::check_cuda(cudaGetDeviceProperties(&device, 0), "validation device properties");
    if (device.major * 10 + device.minor != TMA_MULTICAST_TARGET || !cuda_tma::device_supports_tma()) {
        std::fprintf(stderr, "SKIPPED: gate requires the selected SM%d TMA target; observed SM%d%d\n",
                     TMA_MULTICAST_TARGET, device.major, device.minor);
        return 3;
    }
    cuda_tma::check_cu(cuInit(0), "validation driver initialization");
#if TMA_VALIDATION_CASE <= 1
    constexpr int shape_count = 3;
    const bool ok = verify_copy_shapes<64, 128>();
#elif TMA_VALIDATION_CASE == 2
    constexpr int shape_count = 3;
    const bool ok = verify_copy_shapes<kTile2D_N, kTile2D_M>();
#elif TMA_VALIDATION_CASE == 3
    constexpr int shape_count = 18;
    const bool ok = verify_copy_shapes<128, 64, 1>() &&
                    verify_copy_shapes<128, 32, 2>() &&
                    verify_copy_shapes<128, 32, 1>() &&
                    verify_copy_shapes<64, 64, 1>() &&
                    verify_copy_shapes<64, 32, 2>() &&
                    verify_copy_shapes<64, 32, 1>();
#elif TMA_VALIDATION_CASE == 4
    constexpr int shape_count = 9;
    const bool ok = verify_copy_shapes<128, 64>() &&
                    verify_copy_shapes<64, 64>() &&
                    verify_copy_shapes<64, 32>();
#else
    constexpr int shape_count = 3;
    const bool ok = verify_gemm_case(64, 256, 384) &&
                    verify_gemm_case(67, 196, 259) &&
                    verify_gemm_case(129, 388, 131);
#endif
    if (!ok) return 1;
    std::printf("TMA_LAYOUT_PASS case=%d shapes=%d full_output_and_canaries=checked\n",
                TMA_VALIDATION_CASE, shape_count);
    return 0;
}
