// baseline_tma_copy.cu -- Scalar tiled-neighbor transform (baseline).
//
// Times the same transform as optimized_tma_copy.cu: a 4096x4096 float32
// matrix split into 64x64 tiles. Each output blends its element with the
// elements 1 and 64 places later in the tile's row-major order, clamped to
// the tile's last element, so every tap stays inside the tile.
//
// The kernel launches the same grid and block shape as the TMA arm and runs
// the same per-thread loops. Every tap is a scalar load from global memory and
// every output is a scalar store. Nothing is staged in shared memory, so the
// pair isolates the cost of the descriptor-backed TMA staging.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

namespace {

constexpr int kLookahead = 64;
constexpr int kTile2D_M = 64;   // Tile height (rows)
constexpr int kTile2D_N = 64;   // Tile width (cols)

__host__ __device__ __forceinline__ float combine_values(float center, float near_val, float far_val) {
    return fmaf(far_val, 0.125f, fmaf(near_val, 0.25f, center * 0.75f));
}

// CPU oracle. It is a copy of tiled_neighbor_reference_at in
// optimized_tma_copy.cu, so both arms check their output against the same
// reference. test_audit_fix_o13_w2011.py fails if the two copies differ.
float tiled_neighbor_reference_at(
    const std::vector<float>& input,
    int rows,
    int columns,
    std::size_t index) {
    const int row = static_cast<int>(index / static_cast<std::size_t>(columns));
    const int column = static_cast<int>(index % static_cast<std::size_t>(columns));
    const int row_origin = (row / kTile2D_M) * kTile2D_M;
    const int column_origin = (column / kTile2D_N) * kTile2D_N;
    const int tile_rows = std::min(kTile2D_M, rows - row_origin);
    const int tile_columns = std::min(kTile2D_N, columns - column_origin);
    const int tile_elements = tile_rows * tile_columns;
    const int local = (row - row_origin) * tile_columns + column - column_origin;
    const int near = std::min(local + 1, tile_elements - 1);
    const int far = std::min(local + kLookahead, tile_elements - 1);
    const auto global_index = [&](int tile_index) {
        const int source_row = row_origin + tile_index / tile_columns;
        const int source_column = column_origin + tile_index % tile_columns;
        return static_cast<std::size_t>(source_row) * columns + source_column;
    };
    return combine_values(
        input[index],
        input[global_index(near)],
        input[global_index(far)]);
}

// Same grid, block shape and per-thread loops as descriptor_tma_2d_copy_kernel
// in optimized_tma_copy.cu. Where that kernel reads tile[r][c] from shared
// memory, this one reads tile_src[r * N + c] from global memory.
template <int TILE_M, int TILE_N>
__global__ void scalar_tiled_neighbor_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int M,
    int N) {
    const int tile_m = blockIdx.y * TILE_M;
    const int tile_n = blockIdx.x * TILE_N;
    if (tile_m >= M || tile_n >= N) {
        return;
    }

    const int tile_rows = min(TILE_M, M - tile_m);
    const int tile_cols = min(TILE_N, N - tile_n);
    const int tile_elems = tile_rows * tile_cols;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;
    const std::size_t tile_offset = static_cast<std::size_t>(tile_m) * N + tile_n;
    const float* __restrict__ tile_src = src + tile_offset;
    float* __restrict__ tile_dst = dst + tile_offset;

    if (tile_rows == TILE_M && tile_cols == TILE_N) {
        // Full tiles divide by the compile-time TILE_N, as the TMA kernel does,
        // so runtime integer division does not slow this baseline.
        constexpr int kFull = TILE_M * TILE_N;
        for (int i = tid; i < kFull; i += threads) {
            const unsigned ui = static_cast<unsigned>(i);
            const unsigned un = static_cast<unsigned>(min(i + 1, kFull - 1));
            const unsigned uf = static_cast<unsigned>(min(i + kLookahead, kFull - 1));
            tile_dst[(ui / TILE_N) * N + ui % TILE_N] = combine_values(
                tile_src[(ui / TILE_N) * N + ui % TILE_N],
                tile_src[(un / TILE_N) * N + un % TILE_N],
                tile_src[(uf / TILE_N) * N + uf % TILE_N]);
        }
    } else {
        for (int i = tid; i < tile_elems; i += threads) {
            const int local_row = i / tile_cols;
            const int local_col = i % tile_cols;
            const int near_linear = min(i + 1, tile_elems - 1);
            const int far_linear = min(i + kLookahead, tile_elems - 1);
            tile_dst[local_row * N + local_col] = combine_values(
                tile_src[local_row * N + local_col],
                tile_src[(near_linear / tile_cols) * N + near_linear % tile_cols],
                tile_src[(far_linear / tile_cols) * N + far_linear % tile_cols]);
        }
    }
}

}  // namespace

int main() {
    NVTX_RANGE("main");
    const int M = 4096;  // Matrix rows
    const int N = 4096;  // Matrix cols
    const size_t matrix_bytes = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<float> h_matrix(static_cast<std::size_t>(M) * N);
    {
        NVTX_RANGE("setup");
        for (int i = 0; i < M * N; ++i) {
            h_matrix[i] = static_cast<float>(i % 1000) / 1000.0f;
        }
    }

    float *d_mat_src = nullptr, *d_mat_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mat_src, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_mat_dst, matrix_bytes));
    CUDA_CHECK(cudaMemcpy(d_mat_src, h_matrix.data(), matrix_bytes, cudaMemcpyHostToDevice));
    // All-ones bytes are a float NaN, so any element the kernel skips fails the check below.
    CUDA_CHECK(cudaMemset(d_mat_dst, 0xFF, matrix_bytes));

    const dim3 block2d(16, 16);  // 256 threads per block, as in the TMA arm
    const dim3 grid2d((N + kTile2D_N - 1) / kTile2D_N, (M + kTile2D_M - 1) / kTile2D_M);

    {
        NVTX_RANGE("warmup");
        scalar_tiled_neighbor_kernel<kTile2D_M, kTile2D_N><<<grid2d, block2d>>>(d_mat_src, d_mat_dst, M, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr int kIterations = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < kIterations; ++iter) {
        NVTX_RANGE("compute_kernel:scalar_tiled_neighbor_kernel");
        scalar_tiled_neighbor_kernel<kTile2D_M, kTile2D_N><<<grid2d, block2d>>>(d_mat_src, d_mat_dst, M, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / kIterations;

    std::vector<float> h_matrix_out(static_cast<std::size_t>(M) * N);
    CUDA_CHECK(cudaMemcpy(
        h_matrix_out.data(), d_mat_dst, matrix_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_mat_src));
    CUDA_CHECK(cudaFree(d_mat_dst));

    bool copy_ok = true;
    {
        NVTX_RANGE("verify");
        for (std::size_t idx = 0; idx < h_matrix_out.size(); ++idx) {
            const float expected = tiled_neighbor_reference_at(h_matrix, M, N, idx);
            if (h_matrix_out[idx] != expected) {
                std::fprintf(stderr,
                             "Scalar tiled-neighbor mismatch at %zu (row %zu, column %zu): "
                             "got %.9g expected %.9g\n",
                             idx,
                             idx / static_cast<std::size_t>(N),
                             idx % static_cast<std::size_t>(N),
                             static_cast<double>(h_matrix_out[idx]),
                             static_cast<double>(expected));
                copy_ok = false;
                break;
            }
        }
    }
    if (!copy_ok) {
        std::fprintf(stderr, "Scalar tiled-neighbor validation failed.\n");
        return EXIT_FAILURE;
    }

    std::printf("Scalar tiled-neighbor transform (%dx%d, tile=%dx%d): %.3f ms\n",
                M, N, kTile2D_M, kTile2D_N, avg_ms);
    std::printf("TIME_MS: %.6f\n", avg_ms);
#ifdef VERIFY
    float verify_checksum = 0.0f;
    VERIFY_CHECKSUM(h_matrix_out.data(), M * N, &verify_checksum);
    VERIFY_PRINT_CHECKSUM(verify_checksum);
#endif
    return 0;
}
