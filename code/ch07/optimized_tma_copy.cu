// optimized_tma_copy.cu -- Pipeline-staged neighbor copy with a real 2D TMA path.
//
// This file demonstrates:
// 1. Pipeline-staged 1D neighbor copies with cuda::pipeline
// 2. A descriptor-backed 2D tensor-map copy when CUDA 13+/TMA support is present
//
// BEFORE (manual tiling):
//     for (tile_y) for (tile_x):
//         for (i in tile): load element[y+i][x+j]
//     Many small, potentially uncoalesced loads
//
// AFTER (pipeline-staged copy):
//     cuda::pipeline overlaps global-memory fetch with the neighbor combine step
//     Shared-memory staging reduces redundant reads and amortizes latency
//
// The 1D path stays focused on async pipeline staging. The 2D path now requires
// real CUtensorMap descriptor support and fails fast if tensor maps are not
// usable on the current runtime.

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

constexpr int kThreadsPerBlock = 256;
constexpr int kValuesPerThread = 8;
constexpr int kTileElems = kThreadsPerBlock * kValuesPerThread;  // 2048 elements
constexpr int kLookahead = 64;
constexpr int kStages = 2;
constexpr int kStageSpan = kTileElems + kLookahead;
constexpr int kElements = 1 << 25;
constexpr bool kValidateOutput = false;

__host__ __device__ __forceinline__ float combine_values(float center, float near_val, float far_val) {
    return fmaf(far_val, 0.125f, fmaf(near_val, 0.25f, center * 0.75f));
}

constexpr int kTile2D_M = 64;   // Tile height (rows)
constexpr int kTile2D_N = 64;   // Tile width (cols)  

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
    int M,
    int N) {
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

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(&out_desc, tile_n, tile_m, &output_tile);
        cde::cp_async_bulk_commit_group();
        cde::cp_async_bulk_wait_group_read<0>();
    }
#else
    (void)in_desc;
    (void)out_desc;
    (void)M;
    (void)N;
#endif
}
#endif

__global__ void tma_neighbor_copy_kernel(const float* __restrict__ src,
                                         float* __restrict__ dst,
                                         int n,
                                         int total_tiles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const int tiles_per_block = (total_tiles + gridDim.x - 1) / gridDim.x;
    const int first_tile = blockIdx.x * tiles_per_block;
    const int tiles_to_process = min(tiles_per_block, max(total_tiles - first_tile, 0));
    if (tiles_to_process <= 0) {
        return;
    }

    extern __shared__ float shared[];
    float* stage_buffers[kStages];
    for (int stage = 0; stage < kStages; ++stage) {
        stage_buffers[stage] = shared + stage * kStageSpan;
    }

    cg::thread_block block = cg::this_thread_block();
    __shared__ alignas(cuda::pipeline_shared_state<cuda::thread_scope_block, kStages>)
        unsigned char pipeline_storage[sizeof(cuda::pipeline_shared_state<cuda::thread_scope_block, kStages>)];
    auto* pipeline_state =
        reinterpret_cast<cuda::pipeline_shared_state<cuda::thread_scope_block, kStages>*>(pipeline_storage);
    if (threadIdx.x == 0) {
        new (pipeline_state) cuda::pipeline_shared_state<cuda::thread_scope_block, kStages>();
    }
    block.sync();
    auto pipe = cuda::make_pipeline(block, pipeline_state);

    auto enqueue_tile = [&](int stage, int tile_idx) -> bool {
        if (tile_idx >= first_tile + tiles_to_process) {
            return false;
        }
        const int global_offset = tile_idx * kTileElems;
        if (global_offset >= n) {
            return false;
        }
        const int remaining = n - global_offset;
        const int stage_elems = remaining > kStageSpan ? kStageSpan : remaining;
        pipe.producer_acquire();
        cuda::memcpy_async(
            block,
            stage_buffers[stage],
            src + global_offset,
            static_cast<size_t>(stage_elems) * sizeof(float),
            pipe);
        pipe.producer_commit();
        return true;
    };

    int stage_tile[kStages];
    bool stage_ready[kStages] = {false, false};
    int next_tile = first_tile;
    for (int stage = 0; stage < kStages; ++stage) {
        stage_tile[stage] = next_tile;
        stage_ready[stage] = enqueue_tile(stage, next_tile);
        if (stage_ready[stage]) {
            ++next_tile;
        }
    }

    int tiles_processed = 0;
    int current_stage = 0;
    while (tiles_processed < tiles_to_process) {
        if (!stage_ready[current_stage]) {
            current_stage = (current_stage + 1) % kStages;
            continue;
        }

        pipe.consumer_wait();
        block.sync();

        const int tile_idx = stage_tile[current_stage];
        const int global_offset = tile_idx * kTileElems;
        const int stage_valid = min(kStageSpan, n - global_offset);
        if (stage_valid > 0) {
            const int max_elem = min(kTileElems, n - global_offset);
            float* tile_ptr = stage_buffers[current_stage];
            const int stage_limit = stage_valid - 1;

            for (int base = threadIdx.x * kValuesPerThread;
                 base < max_elem;
                 base += blockDim.x * kValuesPerThread) {
#pragma unroll
                for (int i = 0; i < kValuesPerThread; ++i) {
                    const int local_idx = base + i;
                    if (local_idx >= max_elem) {
                        break;
                    }
                    const int global_idx = global_offset + local_idx;
                    if (global_idx >= n) {
                        continue;
                    }
                    const float center = tile_ptr[local_idx];
                    const int near_local = (local_idx + 1 <= stage_limit) ? (local_idx + 1) : stage_limit;
                    int far_local = local_idx + kLookahead;
                    if (far_local > stage_limit) {
                        far_local = stage_limit;
                    }
                    const float near_val = tile_ptr[near_local];
                    const float far_val = tile_ptr[far_local];
                    dst[global_idx] = combine_values(center, near_val, far_val);
                }
            }
        }

        pipe.consumer_release();
        stage_ready[current_stage] = false;
        ++tiles_processed;

        if (next_tile < first_tile + tiles_to_process) {
            stage_tile[current_stage] = next_tile;
            stage_ready[current_stage] = enqueue_tile(current_stage, next_tile);
            if (stage_ready[current_stage]) {
                ++next_tile;
            }
        }

        current_stage = (current_stage + 1) % kStages;
    }
#else
    (void)src;
    (void)dst;
    (void)n;
    (void)total_tiles;
#endif
}

float checksum(const std::vector<float>& data) {
    double sum = 0.0;
    for (float v : data) {
        NVTX_RANGE("verify");
        sum += static_cast<double>(v);
    }
    return static_cast<float>(sum / static_cast<double>(data.size()));
}

}  // namespace

bool benchmark_tma_2d(cudaDeviceProp& prop) {
    std::printf("\n--- 2D Copy Benchmark ---\n");
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
    descriptor_tma_2d_copy_kernel<kTile2D_M, kTile2D_N><<<grid2d, block2d>>>(in_desc, out_desc, M, N);
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
            descriptor_tma_2d_copy_kernel<kTile2D_M, kTile2D_N><<<grid2d, block2d>>>(in_desc, out_desc, M, N);
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
        if (h_matrix_out[idx] != h_matrix[idx]) {
            std::fprintf(stderr,
                         "2D TMA mismatch at %zu: got %.9g expected %.9g\n",
                         idx,
                         static_cast<double>(h_matrix_out[idx]),
                         static_cast<double>(h_matrix[idx]));
            copy_ok = false;
            break;
        }
    }
    
    float tma2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&tma2d_ms, start2d, stop2d));
    const float avg_tma2d_ms = tma2d_ms / kIterations2D;
    
    // Calculate bandwidth
    const double bytes_transferred = 2.0 * matrix_bytes;  // Read + Write
    const double bandwidth_gbps = (bytes_transferred / 1e9) / (avg_tma2d_ms / 1000.0);
    // Peak bandwidth estimation: B200 = ~8 TB/s, use conservative estimate
    // memoryClockRate is deprecated in newer CUDA, use known peak for Blackwell
    double peak_bandwidth = 8000.0;  // B200 theoretical peak (8 TB/s)
#if __CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 6)
    // Older CUDA versions have memoryClockRate
    peak_bandwidth = static_cast<double>(prop.memoryClockRate) * 1e3 * 
                     (prop.memoryBusWidth / 8) * 2 / 1e9;  // HBM is DDR
#endif
    const double efficiency = 100.0 * bandwidth_gbps / peak_bandwidth;
    
    std::printf("%s (%dx%d, tile=%dx%d): %.3f ms\n",
                "Descriptor-backed 2D TMA copy",
                M, N, kTile2D_M, kTile2D_N, avg_tma2d_ms);
    std::printf("  Achieved bandwidth: %.1f GB/s (%.1f%% of peak)\n",
                bandwidth_gbps, efficiency);
    
    CUDA_CHECK(cudaEventDestroy(start2d));
    CUDA_CHECK(cudaEventDestroy(stop2d));
    CUDA_CHECK(cudaFree(d_mat_src));
    CUDA_CHECK(cudaFree(d_mat_dst));
    if (!copy_ok) {
        std::fprintf(stderr, "2D TMA validation failed.\n");
        std::exit(EXIT_FAILURE);
    }
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

    std::printf("=== Pipeline-Staged Copy Benchmarks (SM %d.%d) ===\n\n", prop.major, prop.minor);
    std::printf("--- Pipeline-Staged Neighbor Copy Benchmark ---\n");

    const size_t bytes = static_cast<size_t>(kElements) * sizeof(float);

    float *d_src = nullptr, *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));

    std::vector<float> h_input(kElements);
    for (int i = 0; i < kElements; ++i) {
        NVTX_RANGE("setup");
        h_input[i] = static_cast<float>((i % 1024) - 512) / 128.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_src, h_input.data(), bytes, cudaMemcpyHostToDevice));

    const int total_tiles = (kElements + kTileElems - 1) / kTileElems;
    const int max_blocks = 2 * prop.multiProcessorCount;
    const int grid = std::min(total_tiles, max_blocks);
    const size_t shared_bytes = static_cast<size_t>(kStages) * kStageSpan * sizeof(float);

    tma_neighbor_copy_kernel<<<grid, kThreadsPerBlock, shared_bytes>>>(
        d_src, d_dst, kElements, total_tiles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr int kIterations = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < kIterations; ++iter) {
        NVTX_RANGE("compute_kernel:tma_neighbor_copy_kernel:smem");
        tma_neighbor_copy_kernel<<<grid, kThreadsPerBlock, shared_bytes>>>(
            d_src, d_dst, kElements, total_tiles);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / kIterations;
    std::printf("Pipeline-staged neighbor copy (optimized, non-TMA): %.3f ms\n", avg_ms);

    std::vector<float> h_output(kElements);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_dst, bytes, cudaMemcpyDeviceToHost));

    if (kValidateOutput) {
        std::vector<float> h_reference(kElements);
        for (int i = 0; i < kElements; ++i) {
            NVTX_RANGE("verify");
            const int near_idx = (i + 1 < kElements) ? (i + 1) : (kElements - 1);
            const int far_idx = (i + kLookahead < kElements) ? (i + kLookahead) : (kElements - 1);
            h_reference[i] = combine_values(h_input[i], h_input[near_idx], h_input[far_idx]);
        }

        float max_error = 0.0f;
        for (int i = 0; i < kElements; ++i) {
            NVTX_RANGE("verify");
            max_error = std::max(max_error, std::abs(h_reference[i] - h_output[i]));
        }
        std::printf("Output checksum: %.6f (max error %.6f)\n", checksum(h_output), max_error);
    } else {
        std::printf("Output checksum: %.6f\n", checksum(h_output));
    }
#ifdef VERIFY
    float verify_checksum = 0.0f;
    VERIFY_CHECKSUM(h_output.data(), kElements, &verify_checksum);
    VERIFY_PRINT_CHECKSUM(verify_checksum);
#endif

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));

    // Run TMA 2D benchmark (from PERFORMANCE_OPTIMIZATION_ANALYSIS.md)
    if (!benchmark_tma_2d(prop)) {
        return 3;
    }

    std::printf("\n=== Summary ===\n");
    std::printf("TMA 1D: Good for streaming linear data\n");
    std::printf("TMA 2D: Better for tiled access patterns (attention, GEMM)\n");
    
    return 0;
}
