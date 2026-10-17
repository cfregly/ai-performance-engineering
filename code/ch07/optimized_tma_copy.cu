// optimized_tma_copy.cu -- Descriptor-backed 2D TMA tiled-neighbor transform.
//
// Times descriptor_tma_2d_copy_kernel on a 4096x4096 float32 matrix split into
// 64x64 tiles. Each output blends its element with the elements 1 and 64 places
// later in the tile's row-major order, clamped to the tile's last element.
// baseline_tma_copy.cu times the same transform with scalar global loads.
//
// BEFORE (baseline_tma_copy.cu):
//     each thread issues three scalar global loads per output
//     and one scalar global store
//
// AFTER (this file):
//     one TMA load copies the whole tile into shared memory
//     the neighbor taps read shared memory
//     one TMA store writes the finished tile back
//
// The binary requires real CUtensorMap descriptor support and exits with
// SKIPPED if tensor maps are not usable on the current runtime.

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>

// For TMA 2D tensor descriptors (SM90+)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#include <cuda/barrier>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/headers/tma_helpers.cuh"
#include "../core/common/nvtx_utils.cuh"

#if CUDART_VERSION >= 13000
#define TMA_CUDA13_AVAILABLE 1
namespace cde = cuda::device::experimental;
#else
#define TMA_CUDA13_AVAILABLE 0
#endif

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

namespace cg = cooperative_groups;

namespace {

constexpr int kLookahead = 64;

__host__ __device__ __forceinline__ float combine_values(float center, float near_val, float far_val) {
    return fmaf(far_val, 0.125f, fmaf(near_val, 0.25f, center * 0.75f));
}

constexpr int kTile2D_M = 64;   // Tile height (rows)
constexpr int kTile2D_N = 64;   // Tile width (cols)  

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

// Not launched. The benchmark times only descriptor_tma_2d_copy_kernel.
template <int TILE_M, int TILE_N>
__global__ void async_pipeline_2d_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int M,
    int N
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // Tile coordinates
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int row_offset = tile_m * TILE_M;
    const int col_offset = tile_n * TILE_N;
    
    __shared__ alignas(128) float smem[TILE_M][TILE_N + 4];
    
    // Cooperative loading using the thread block
    cg::thread_block block = cg::this_thread_block();
    
    // Pipeline for async loads
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 1> pipe_state;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        new (&pipe_state) cuda::pipeline_shared_state<cuda::thread_scope_block, 1>();
    }
    block.sync();
    auto pipe = cuda::make_pipeline(block, &pipe_state);
    
    pipe.producer_acquire();
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;
    const int elems_per_tile = TILE_M * TILE_N;
    
    for (int i = tid; i < elems_per_tile; i += threads) {
        const int local_row = i / TILE_N;
        const int local_col = i % TILE_N;
        const int global_row = row_offset + local_row;
        const int global_col = col_offset + local_col;
        
        if (global_row < M && global_col < N) {
            cuda::memcpy_async(&smem[local_row][local_col],
                              &src[global_row * N + global_col],
                              sizeof(float), pipe);
        }
    }
    
    pipe.producer_commit();
    pipe.consumer_wait();
    block.sync();

    const int tile_rows = min(TILE_M, max(M - row_offset, 0));
    const int tile_cols = min(TILE_N, max(N - col_offset, 0));
    const int tile_elems = tile_rows * tile_cols;
    
    for (int i = tid; i < tile_elems; i += threads) {
        const int local_row = i / tile_cols;
        const int local_col = i % tile_cols;
        const int global_row = row_offset + local_row;
        const int global_col = col_offset + local_col;

        const int near_linear = min(i + 1, tile_elems - 1);
        const int far_linear = min(i + kLookahead, tile_elems - 1);
        const int near_row = near_linear / tile_cols;
        const int near_col = near_linear % tile_cols;
        const int far_row = far_linear / tile_cols;
        const int far_col = far_linear % tile_cols;

        dst[global_row * N + global_col] = combine_values(
            smem[local_row][local_col],
            smem[near_row][near_col],
            smem[far_row][far_col]);
    }
    
    pipe.consumer_release();
#else
    // Fallback for older architectures
    const int row_offset = blockIdx.y * TILE_M;
    const int col_offset = blockIdx.x * TILE_N;
    const int local_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int tile_rows = min(TILE_M, max(M - row_offset, 0));
    const int tile_cols = min(TILE_N, max(N - col_offset, 0));
    const int tile_elems = tile_rows * tile_cols;
    const int threads = blockDim.x * blockDim.y;

    for (int i = local_tid; i < tile_elems; i += threads) {
        const int local_row = i / tile_cols;
        const int local_col = i % tile_cols;
        const int near_linear = min(i + 1, tile_elems - 1);
        const int far_linear = min(i + kLookahead, tile_elems - 1);
        const int near_row = near_linear / tile_cols;
        const int near_col = near_linear % tile_cols;
        const int far_row = far_linear / tile_cols;
        const int far_col = far_linear % tile_cols;
        const int global_row = row_offset + local_row;
        const int global_col = col_offset + local_col;

        dst[global_row * N + global_col] = combine_values(
            src[(row_offset + local_row) * N + (col_offset + local_col)],
            src[(row_offset + near_row) * N + (col_offset + near_col)],
            src[(row_offset + far_row) * N + (col_offset + far_col)]);
    }
#endif
}

#if TMA_CUDA13_AVAILABLE
template <int TILE_M, int TILE_N>
__global__ void descriptor_tma_2d_copy_kernel(
    const __grid_constant__ CUtensorMap in_desc,
    const __grid_constant__ CUtensorMap out_desc,
    float* output,
    int M,
    int N,
    int output_ld) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr std::size_t kTileBytes = static_cast<std::size_t>(TILE_M) * TILE_N * sizeof(float);
    __shared__ alignas(128) float tile[TILE_M][TILE_N];
    __shared__ alignas(128) float output_tile[TILE_M][TILE_N];
    using block_barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(block_barrier) unsigned char barrier_storage[sizeof(block_barrier)];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        init(reinterpret_cast<block_barrier*>(barrier_storage), blockDim.x * blockDim.y);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    auto* bar_ptr = reinterpret_cast<block_barrier*>(barrier_storage);
    auto& bar = *bar_ptr;

    const int tile_m = blockIdx.y * TILE_M;
    const int tile_n = blockIdx.x * TILE_N;
    if (tile_m >= M || tile_n >= N) {
        return;
    }

    cuda::barrier<cuda::thread_scope_block>::arrival_token token;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&tile, &in_desc, tile_n, tile_m, bar);
        token = cuda::device::barrier_arrive_tx(bar, 1, kTileBytes);
    } else {
        token = bar.arrive();
    }
    bar.wait(std::move(token));
    __syncthreads();

    const int tile_rows = min(TILE_M, max(M - tile_m, 0));
    const int tile_cols = min(TILE_N, max(N - tile_n, 0));
    const int tile_elems = tile_rows * tile_cols;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;

    if (tile_cols == TILE_N && tile_rows == TILE_M) {
        // Full-tile fast path. The generic loop divides by the RUNTIME tile_cols, which the
        // compiler cannot strength-reduce, so this copy kernel was compute-bound on integer
        // division (ncu: SM 71% / DRAM 25%) instead of memory-bound. TILE_N is a compile-time
        // power of two, so these / and % fold to shift/mask; the indices are identical.
        constexpr int kFull = TILE_M * TILE_N;
        for (int i = tid; i < kFull; i += threads) {
            const unsigned ui = static_cast<unsigned>(i);
            const unsigned un = static_cast<unsigned>(min(i + 1, kFull - 1));
            const unsigned uf = static_cast<unsigned>(min(i + kLookahead, kFull - 1));
            output_tile[ui / TILE_N][ui % TILE_N] = combine_values(
                tile[ui / TILE_N][ui % TILE_N],
                tile[un / TILE_N][un % TILE_N],
                tile[uf / TILE_N][uf % TILE_N]);
        }
    } else {
        for (int i = tid; i < tile_elems; i += threads) {
            const int local_row = i / tile_cols;
            const int local_col = i % tile_cols;
            const int near_linear = min(i + 1, tile_elems - 1);
            const int far_linear = min(i + kLookahead, tile_elems - 1);
            const int near_row = near_linear / tile_cols;
            const int near_col = near_linear % tile_cols;
            const int far_row = far_linear / tile_cols;
            const int far_col = far_linear % tile_cols;

            output_tile[local_row][local_col] = combine_values(
                tile[local_row][local_col],
                tile[near_row][near_col],
                tile[far_row][far_col]);
        }
    }
    __syncthreads();

    cde::fence_proxy_async_shared_cta();
    __syncthreads();

    if (tile_rows == TILE_M && tile_cols == TILE_N) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            cde::cp_async_bulk_tensor_2d_shared_to_global(&out_desc, tile_n, tile_m, &output_tile);
            cde::cp_async_bulk_commit_group();
            cde::cp_async_bulk_wait_group_read<0>();
        }
    } else {
        cuda_tma::store_partial_2d_tile(
            output_tile, output, output_ld, tile_m, tile_n, tile_rows, tile_cols, tid, threads);
    }
#else
    (void)in_desc;
    (void)out_desc;
    (void)output;
    (void)output_ld;
    (void)M;
    (void)N;
#endif
}
#endif

#if TMA_CUDA13_AVAILABLE
// The timed 4096x4096 shape is square and tile-aligned, so its full-output check
// cannot see a width/height swap in the tensor-map helper or a partial-tile bug.
// This shape has more columns than rows and partial tiles on both edges.
constexpr int kValidationRows2D = 200;
constexpr int kValidationColumns2D = 328;

float tma_2d_validation_input(int row, int column) {
    return static_cast<float>((row * 131 + column * 17) % 1024) / 128.0f;
}

bool validate_tma_2d_non_square(PFN_cuTensorMapEncodeTiled_v12000 encode) {
    NVTX_RANGE("verify");
    const int rows = kValidationRows2D;
    const int columns = kValidationColumns2D;
    const std::size_t elements = static_cast<std::size_t>(rows) * columns;
    const std::size_t bytes = elements * sizeof(float);

    std::vector<float> input(elements);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            input[static_cast<std::size_t>(row) * columns + column] =
                tma_2d_validation_input(row, column);
        }
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
    // All-ones bytes are a float NaN, so any element the kernel skips fails the comparison.
    CUDA_CHECK(cudaMemset(d_output, 0xFF, bytes));

    CUtensorMap input_desc{};
    CUtensorMap output_desc{};
    const bool encoded =
        cuda_tma::make_2d_tensor_map(
            input_desc, encode, d_input, columns, rows, columns,
            kTile2D_N, kTile2D_M, CU_TENSOR_MAP_SWIZZLE_NONE) &&
        cuda_tma::make_2d_tensor_map(
            output_desc, encode, d_output, columns, rows, columns,
            kTile2D_N, kTile2D_M, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (!encoded) {
        std::fprintf(stderr,
                     "2D TMA non-square validation could not encode %dx%d tensor maps.\n",
                     rows,
                     columns);
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        return false;
    }

    const dim3 block(16, 16);
    const dim3 grid((columns + kTile2D_N - 1) / kTile2D_N, (rows + kTile2D_M - 1) / kTile2D_M);
    descriptor_tma_2d_copy_kernel<kTile2D_M, kTile2D_N><<<grid, block>>>(
        input_desc, output_desc, d_output, rows, columns, columns);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> output(elements);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    for (std::size_t index = 0; index < elements; ++index) {
        const float reference = tiled_neighbor_reference_at(input, rows, columns, index);
        if (output[index] != reference) {
            std::fprintf(stderr,
                         "2D TMA non-square (%dx%d) mismatch at row %zu column %zu: "
                         "got %.9g expected %.9g\n",
                         rows,
                         columns,
                         index / static_cast<std::size_t>(columns),
                         index % static_cast<std::size_t>(columns),
                         static_cast<double>(output[index]),
                         static_cast<double>(reference));
            return false;
        }
    }
    std::printf("2D TMA non-square validation passed (%dx%d, tile=%dx%d)\n",
                rows, columns, kTile2D_M, kTile2D_N);
    return true;
}
#endif

}  // namespace

bool benchmark_tma_2d(cudaDeviceProp& prop) {
    std::printf("\n--- 2D TMA Tiled-Neighbor Transform Benchmark ---\n");
#if !TMA_CUDA13_AVAILABLE
    (void)prop;
    std::fprintf(stderr, "SKIPPED: optimized_tma_copy requires usable tensor-map/TMA support\n");
    return false;
#endif
    
    const int M = 4096;  // Matrix rows
    const int N = 4096;  // Matrix cols  
    const size_t matrix_bytes = static_cast<size_t>(M) * N * sizeof(float);
    
    float *d_mat_src = nullptr, *d_mat_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mat_src, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_mat_dst, matrix_bytes));
    
    // Initialize
    std::vector<float> h_matrix(M * N);
    for (int i = 0; i < M * N; ++i) {
        NVTX_RANGE("setup");
        h_matrix[i] = static_cast<float>(i % 1000) / 1000.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_mat_src, h_matrix.data(), matrix_bytes, cudaMemcpyHostToDevice));
    // All-ones bytes are a float NaN, so any element the kernel skips fails the full-output check.
    CUDA_CHECK(cudaMemset(d_mat_dst, 0xFF, matrix_bytes));

    dim3 block2d(16, 16);  // 256 threads per block
    dim3 grid2d((N + kTile2D_N - 1) / kTile2D_N,
                (M + kTile2D_M - 1) / kTile2D_M);

    bool use_tensor_map = false;
    CUtensorMap in_desc{};
    CUtensorMap out_desc{};
    auto encode = cuda_tma::load_cuTensorMapEncodeTiled();
    use_tensor_map =
        cuda_tma::device_supports_tma() &&
        encode &&
        cuda_tma::make_2d_tensor_map(
            in_desc,
            encode,
            d_mat_src,
            N,
            M,
            N,
            kTile2D_N,
            kTile2D_M,
            CU_TENSOR_MAP_SWIZZLE_NONE) &&
        cuda_tma::make_2d_tensor_map(
            out_desc,
            encode,
            d_mat_dst,
            N,
            M,
            N,
            kTile2D_N,
            kTile2D_M,
            CU_TENSOR_MAP_SWIZZLE_NONE);

    if (!use_tensor_map) {
        std::fprintf(stderr, "SKIPPED: optimized_tma_copy requires usable tensor-map/TMA support\n");
        CUDA_CHECK(cudaFree(d_mat_src));
        CUDA_CHECK(cudaFree(d_mat_dst));
        return false;
    }
    if (!validate_tma_2d_non_square(encode)) {
        std::fprintf(stderr, "2D TMA validation failed.\n");
        CUDA_CHECK(cudaFree(d_mat_src));
        CUDA_CHECK(cudaFree(d_mat_dst));
        std::exit(EXIT_FAILURE);
    }
    descriptor_tma_2d_copy_kernel<kTile2D_M, kTile2D_N><<<grid2d, block2d>>>(in_desc, out_desc, d_mat_dst, M, N, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start2d, stop2d;
    CUDA_CHECK(cudaEventCreate(&start2d));
    CUDA_CHECK(cudaEventCreate(&stop2d));
    
    constexpr int kIterations2D = 20;
    CUDA_CHECK(cudaEventRecord(start2d));
    for (int iter = 0; iter < kIterations2D; ++iter) {
        {
            NVTX_RANGE("compute_kernel:descriptor_tma_2d_copy_kernel");
            descriptor_tma_2d_copy_kernel<kTile2D_M, kTile2D_N><<<grid2d, block2d>>>(in_desc, out_desc, d_mat_dst, M, N, N);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop2d));
    CUDA_CHECK(cudaEventSynchronize(stop2d));

    std::vector<float> h_matrix_out(static_cast<std::size_t>(M) * N);
    CUDA_CHECK(cudaMemcpy(
        h_matrix_out.data(), d_mat_dst, matrix_bytes, cudaMemcpyDeviceToHost));
    bool copy_ok = true;
    for (std::size_t idx = 0; idx < h_matrix_out.size(); ++idx) {
        const float expected = tiled_neighbor_reference_at(h_matrix, M, N, idx);
        if (h_matrix_out[idx] != expected) {
            std::fprintf(stderr,
                         "2D TMA tiled-neighbor mismatch at %zu: got %.9g expected %.9g\n",
                         idx,
                         static_cast<double>(h_matrix_out[idx]),
                         static_cast<double>(expected));
            copy_ok = false;
            break;
        }
    }
    
    float tma2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&tma2d_ms, start2d, stop2d));
    const float avg_tma2d_ms = tma2d_ms / kIterations2D;

    CUDA_CHECK(cudaEventDestroy(start2d));
    CUDA_CHECK(cudaEventDestroy(stop2d));
    CUDA_CHECK(cudaFree(d_mat_src));
    CUDA_CHECK(cudaFree(d_mat_dst));
    if (!copy_ok) {
        std::fprintf(stderr, "2D TMA validation failed.\n");
        std::exit(EXIT_FAILURE);
    }

    // Calculate bandwidth
    const double bytes_transferred = 2.0 * matrix_bytes;  // Read + Write
    const double bandwidth_gbps = (bytes_transferred / 1e9) / (avg_tma2d_ms / 1000.0);
    
    std::printf("%s (%dx%d, tile=%dx%d): %.3f ms\n",
                "Descriptor-backed 2D TMA tiled-neighbor transform",
                M, N, kTile2D_M, kTile2D_N, avg_tma2d_ms);
    std::printf("  Effective requested read-plus-write bandwidth: %.1f GB/s\n",
                bandwidth_gbps);
    std::printf("TIME_MS: %.6f\n", avg_tma2d_ms);
#ifdef VERIFY
    float verify_checksum = 0.0f;
    VERIFY_CHECKSUM(h_matrix_out.data(), M * N, &verify_checksum);
    VERIFY_PRINT_CHECKSUM(verify_checksum);
#endif
    return true;
}

int main() {
    NVTX_RANGE("main");
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 9) {
        std::fprintf(stderr, "SKIPPED: optimized_tma_copy requires SM 90+\n");
        return 3;
    }

    std::printf("=== Descriptor-Backed 2D TMA Benchmark (SM %d.%d) ===\n", prop.major, prop.minor);
    if (!benchmark_tma_2d(prop)) {
        return 3;
    }
    return 0;
}
