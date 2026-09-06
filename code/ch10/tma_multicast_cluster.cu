// tma_multicast_cluster.cu - TMA Multicast for CTA Clusters (Ch10)
//
// This benchmark demonstrates cluster multicast for a tiled FP32 GEMM:
// - Blocks are launched in 16x1 clusters along M (sixteen CTAs share the same B tile).
// - Cluster rank 0 issues a single TMA bulk tensor load for the B tile and
//   multicasts it to all CTAs in the cluster.
// - Each CTA loads its own A tile and computes its C tile.
//
// Note: This is an educational example; achieving a speedup depends on the
// workload regime and the overhead of cluster synchronization.

#include <cooperative_groups.h>
#include <cuda/__ptx/instructions/cp_async_bulk_tensor.h>
#include <cuda/barrier>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/headers/tma_helpers.cuh"
#include "../core/common/nvtx_utils.cuh"

namespace cg = cooperative_groups;
namespace cptx = cuda::ptx;
namespace cde = cuda::device::experimental;

#ifndef TMA_MULTICAST_TARGET
#error "Build tma_multicast_cluster.cu through its Makefile target"
#endif

#if TMA_MULTICAST_TARGET != 0 && TMA_MULTICAST_TARGET != 100 && \
    TMA_MULTICAST_TARGET != 103
#error "TMA_MULTICAST_TARGET must be 0, 100, or 103"
#endif

#if defined(__CUDA_ARCH__) && TMA_MULTICAST_TARGET == 100 && \
    (!defined(__CUDA_ARCH_SPECIFIC__) || __CUDA_ARCH_SPECIFIC__ != 1000)
#error "TMA_MULTICAST_TARGET=100 requires compute_100a"
#endif

#if defined(__CUDA_ARCH__) && TMA_MULTICAST_TARGET == 103 && \
    (!defined(__CUDA_ARCH_SPECIFIC__) || __CUDA_ARCH_SPECIFIC__ != 1030)
#error "TMA_MULTICAST_TARGET=103 requires compute_103a"
#endif

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// Tile dimensions
// NOTE: This example is designed to be *bandwidth sensitive* so that cluster
// multicast has a clear win: keep TILE_M small (low reuse of each B element)
// while keeping TILE_N/TILE_K large (large B tile).
constexpr int TILE_M = 4;
constexpr int TILE_N = 128;
constexpr int TILE_K = 128;
constexpr int BLOCK_SIZE = 128;
constexpr size_t B_SMEM_BYTES =
    static_cast<size_t>(TILE_K) * TILE_N * sizeof(float);
constexpr size_t A_SMEM_BYTES =
    static_cast<size_t>(TILE_M) * TILE_K * sizeof(float);

using block_barrier = cuda::barrier<cuda::thread_scope_block>;

// Keep every shared object in one aligned dynamic allocation so the TMA
// destination and multicast barrier have fixed, aligned CTA-relative offsets in
// every block.
struct alignas(128) TmaMulticastSharedStorage {
    alignas(128) float B[TILE_K][TILE_N];
    alignas(128) float A[TILE_M][TILE_K];
    alignas(block_barrier) unsigned char barrier[sizeof(block_barrier)];
};

constexpr size_t DYNAMIC_SMEM_BYTES = sizeof(TmaMulticastSharedStorage);
static_assert(B_SMEM_BYTES == 65536);
static_assert(A_SMEM_BYTES == 2048);
static_assert(offsetof(TmaMulticastSharedStorage, B) % 128 == 0);
static_assert(offsetof(TmaMulticastSharedStorage, A) % 128 == 0);
static_assert(
    offsetof(TmaMulticastSharedStorage, barrier) % alignof(block_barrier) == 0);
static_assert(DYNAMIC_SMEM_BYTES % 128 == 0);

// Cluster configuration: 16x1 cluster along M (shares B tiles).
constexpr int CLUSTER_M = 16;
constexpr int CLUSTER_N = 1;

#if TMA_MULTICAST_TARGET == 103
// CUDA 13.0 bundles CCCL 3.0.1, whose generated wrapper omitted SM103a.
// NVIDIA added this exact architecture guard and PTX instruction in CCCL 3.1.
// https://github.com/NVIDIA/cccl/commit/f32097d47a1a65527bcb3062277c791655408b30
__device__ __forceinline__ void issue_tma_multicast_sm103(
    void* dst,
    const CUtensorMap* tensor_map,
    const int (&coords)[2],
    std::uint64_t* barrier,
    std::uint16_t cta_mask) {
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1030
    const auto dst_smem =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(dst));
    const auto barrier_smem =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(dst_smem),
          "l"(tensor_map),
          "r"(coords[0]),
          "r"(coords[1]),
          "r"(barrier_smem),
          "h"(cta_mask)
        : "memory");
#endif
}
#endif

//============================================================================
// TMA Multicast Kernel
//============================================================================

__global__ __launch_bounds__(BLOCK_SIZE, 1)
void tma_multicast_gemm_kernel(
    const __grid_constant__ CUtensorMap b_desc,
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [K, N]
    float* __restrict__ C,        // [M, N]
    int M, int N, int K
) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto& shared = *reinterpret_cast<TmaMulticastSharedStorage*>(smem_raw);
    auto& B_smem = shared.B;
    auto& A_smem = shared.A;
#if TMA_MULTICAST_TARGET == 100 || TMA_MULTICAST_TARGET == 103
    cg::cluster_group cluster = cg::this_cluster();
    const int cluster_rank = cluster.block_rank();

    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    const bool tile_valid = (tile_m * TILE_M < M) && (tile_n * TILE_N < N);

    const int tid = threadIdx.x;
    // 128 threads × (1×4 outputs/thread) = 512 outputs = 4×128 tile.
    constexpr int COLS_PER_THREAD = 4;
    constexpr int THREADS_PER_ROW = TILE_N / COLS_PER_THREAD;  // 32
    const int thread_m = tid / THREADS_PER_ROW;                // 0..3
    const int thread_n = (tid % THREADS_PER_ROW) * COLS_PER_THREAD;  // 0..124

    auto* bar = reinterpret_cast<block_barrier*>(shared.barrier);
    if (tid == 0) {
        init(bar, static_cast<int>(blockDim.x));
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    float acc[COLS_PER_THREAD] = {0.0f};
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_base = k_tile * TILE_K;

        block_barrier::arrival_token token;
        if (tid == 0) {
            token = cuda::device::barrier_arrive_tx(*bar, 1, B_SMEM_BYTES);
        } else {
            token = bar->arrive();
        }

        // Ensure every CTA has joined the barrier generation before the
        // multicast is issued (avoids missed completions on large clusters).
        __syncthreads();
        cluster.sync();

        // Cluster multicast: issue one B tile load per cluster.
        if (cluster_rank == 0 && tid == 0) {
            // B[K,N] is row-major: contiguous column first, then the K row.
            const int coords[2] = {tile_n * TILE_N, k_base};
            const uint16_t cta_mask = static_cast<uint16_t>((1u << (CLUSTER_M * CLUSTER_N)) - 1u);
#if TMA_MULTICAST_TARGET == 103
            issue_tma_multicast_sm103(
                &B_smem[0][0],
                &b_desc,
                coords,
                cuda::device::barrier_native_handle(*bar),
                cta_mask);
#else
            cptx::cp_async_bulk_tensor(
                cptx::space_cluster,
                cptx::space_global,
                &B_smem[0][0],
                &b_desc,
                coords,
                cuda::device::barrier_native_handle(*bar),
                cta_mask);
#endif
        }

        // Each CTA loads its own A tile while the multicast B load is in flight.
        for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            int mm = i / TILE_K;
            int kk = i % TILE_K;
            int global_m = tile_m * TILE_M + mm;
            int global_k = k_base + kk;
            A_smem[mm][kk] = (global_m < M && global_k < K) ? A[global_m * K + global_k] : 0.0f;
        }
        __syncthreads();

        bar->wait(std::move(token));
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a_val = A_smem[thread_m][kk];
            #pragma unroll
            for (int j = 0; j < COLS_PER_THREAD; ++j) {
                acc[j] += a_val * B_smem[kk][thread_n + j];
            }
        }

        __syncthreads();
        cluster.sync();  // Avoid overwriting B_smem before all CTAs finish.
    }

    #pragma unroll
    for (int j = 0; j < COLS_PER_THREAD; ++j) {
        int global_m = tile_m * TILE_M + thread_m;
        int global_n = tile_n * TILE_N + thread_n + j;
        if (tile_valid && global_m < M && global_n < N) {
            C[global_m * N + global_n] = acc[j];
        }
    }
#else
    // Fallback (no clusters/TMA): standard tiled GEMM
    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    const int tid = threadIdx.x;

    constexpr int COLS_PER_THREAD = 4;
    constexpr int THREADS_PER_ROW = TILE_N / COLS_PER_THREAD;
    float acc[COLS_PER_THREAD] = {0.0f};

    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; ++k_tile) {
        const int k_base = k_tile * TILE_K;
        for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            int mm = i / TILE_K;
            int kk = i % TILE_K;
            int global_m = tile_m * TILE_M + mm;
            int global_k = k_base + kk;
            A_smem[mm][kk] = (global_m < M && global_k < K) ? A[global_m * K + global_k] : 0.0f;
        }
        for (int i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            int kk = i / TILE_N;
            int nn = i % TILE_N;
            int global_k = k_base + kk;
            int global_n = tile_n * TILE_N + nn;
            B_smem[kk][nn] = (global_k < K && global_n < N) ? B[global_k * N + global_n] : 0.0f;
        }
        __syncthreads();

        int tm = tid / THREADS_PER_ROW;
        int tn = (tid % THREADS_PER_ROW) * COLS_PER_THREAD;
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a_val = A_smem[tm][kk];
            for (int j = 0; j < COLS_PER_THREAD; ++j) {
                acc[j] += a_val * B_smem[kk][tn + j];
            }
        }
        __syncthreads();
    }

    int tm = tid / THREADS_PER_ROW;
    int tn = (tid % THREADS_PER_ROW) * COLS_PER_THREAD;
    for (int j = 0; j < COLS_PER_THREAD; ++j) {
        int global_m = tile_m * TILE_M + tm;
        int global_n = tile_n * TILE_N + tn + j;
        if (global_m < M && global_n < N) {
            C[global_m * N + global_n] = acc[j];
        }
    }
#endif
}

//============================================================================
// Benchmark harness
//============================================================================

int main(int argc, char** argv) {
    NVTX_RANGE("main");
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::printf("TMA Multicast GEMM Example\n");
    std::printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

#if TMA_MULTICAST_TARGET == 0
    std::printf(
        "SKIPPED: compile target does not provide CTA cluster TMA multicast\n");
    return 3;
#endif

    if (prop.major < 9) {
        std::printf("SKIPPED: requires SM90+ for TMA/cluster multicast\n");
        return 3;
    }

    cudaFuncAttributes kernel_attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(
        &kernel_attributes,
        tma_multicast_gemm_kernel));
    int max_shared_bytes = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &max_shared_bytes,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        0));
    const size_t required_shared_bytes =
        DYNAMIC_SMEM_BYTES + kernel_attributes.sharedSizeBytes;
    if (required_shared_bytes > static_cast<size_t>(max_shared_bytes)) {
        std::printf(
            "SKIPPED: multicast kernel requires %zu bytes of shared memory, device permits %d\n",
            required_shared_bytes,
            max_shared_bytes);
        return 3;
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        tma_multicast_gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(DYNAMIC_SMEM_BYTES)));

    int M = 2048;
    int N = 2048;
    int K = 2048;
    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    } else if (argc != 1) {
        std::fprintf(stderr, "Usage: %s [M N K]\n", argv[0]);
        return 1;
    }

    std::printf("Matrix: [%d, %d] x [%d, %d] = [%d, %d]\n", M, K, K, N, M, N);
    std::printf("Tile: %dx%dx%d, Cluster: %dx%d\n\n", TILE_M, TILE_N, TILE_K, CLUSTER_M, CLUSTER_N);

    size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    std::vector<float> h_A(static_cast<size_t>(M) * K);
    std::vector<float> h_B(static_cast<size_t>(K) * N);
    {
        NVTX_RANGE("setup");
        for (size_t i = 0; i < h_A.size(); ++i) {
            h_A[i] = static_cast<float>(rand() % 100) / 100.0f;
        }
        for (size_t i = 0; i < h_B.size(); ++i) {
            h_B[i] = static_cast<float>(rand() % 100) / 100.0f;
        }
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    CUtensorMap b_desc{};
    {
        NVTX_RANGE("tile");
        cuda_tma::check_cu(cuInit(0), "cuInit");
        auto encode = cuda_tma::load_cuTensorMapEncodeTiled();
        if (!encode) {
            std::fprintf(stderr, "cuTensorMapEncodeTiled unavailable on this runtime.\n");
            return 1;
        }
        const bool ok = cuda_tma::make_2d_tensor_map(
            b_desc,
            encode,
            d_B,
            /*width=*/N,
            /*height=*/K,
            /*ld=*/N,
            /*box_width=*/TILE_N,
            /*box_height=*/TILE_K,
            CU_TENSOR_MAP_SWIZZLE_NONE);
        if (!ok) {
            return 1;
        }
    }

    dim3 block(BLOCK_SIZE);
    // Every cluster must be complete, including a final partial M tile. Padded
    // CTAs participate in multicast/barriers but tile_valid suppresses stores.
    const int m_tiles = (M + TILE_M - 1) / TILE_M;
    dim3 grid(((m_tiles + CLUSTER_M - 1) / CLUSTER_M) * CLUSTER_M,
              (N + TILE_N - 1) / TILE_N);

    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_M;
    attrs[0].val.clusterDim.y = CLUSTER_N;
    attrs[0].val.clusterDim.z = 1;

    CUDA_CHECK(cudaFuncSetAttribute(
        tma_multicast_gemm_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed,
        1));

    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = DYNAMIC_SMEM_BYTES;
    config.stream = 0;
    config.attrs = attrs;
    config.numAttrs = 1;

    // Warmup
    CUDA_CHECK(cudaLaunchKernelEx(&config, tma_multicast_gemm_kernel, b_desc, d_A, d_B, d_C, M, N, K));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        NVTX_RANGE("iteration");
        CUDA_CHECK(cudaLaunchKernelEx(&config, tma_multicast_gemm_kernel, b_desc, d_A, d_B, d_C, M, N, K));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;

    double flops = 2.0 * static_cast<double>(M) * N * K;
    double tflops = (flops / 1e12) / (avg_ms / 1000.0);

    std::printf("Results:\n");
    std::printf("  Avg time: %.3f ms\n", avg_ms);
    std::printf("  TFLOPS: %.2f\n", tflops);

#ifdef VERIFY
    std::vector<float> h_C(static_cast<size_t>(M) * N);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (float v : h_C) {
        NVTX_RANGE("verify");
        checksum += static_cast<double>(v);
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
