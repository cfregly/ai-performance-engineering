// tma_multicast_baseline.cu - Cluster GEMM without TMA Multicast (Ch10)
//
// Baseline for the cluster multicast example:
// - Blocks are launched in clusters (same shape as optimized).
// - Each CTA loads the B tile into its own SMEM via standard global loads (no multicast).
//   This causes redundant global loads for the same tile across the cluster.
//
// COMPARE WITH: tma_multicast_cluster.cu
//   - Optimized uses TMA multicast so a single load feeds all CTAs in the cluster.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

#ifndef TMA_MULTICAST_TARGET
#error "Build tma_multicast_baseline.cu through its Makefile target"
#endif

#if TMA_MULTICAST_TARGET != 0 && TMA_MULTICAST_TARGET != 100 && \
    TMA_MULTICAST_TARGET != 103
#error "TMA_MULTICAST_TARGET must be 0, 100, or 103"
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

// Tile dimensions (same as optimized for fair comparison)
// NOTE: This example is designed to be *bandwidth sensitive* so that cluster
// multicast has a clear win: keep TILE_M small (low reuse of each B element)
// while keeping TILE_N/TILE_K large (large B tile).
constexpr int TILE_M = 4;
constexpr int TILE_N = 128;
constexpr int TILE_K = 128;
constexpr int BLOCK_SIZE = 128;
constexpr size_t B_SMEM_BYTES =
    static_cast<size_t>(TILE_K) * TILE_N * sizeof(float);
constexpr size_t DYNAMIC_SMEM_BYTES = B_SMEM_BYTES;
static_assert(DYNAMIC_SMEM_BYTES == 65536);

// Cluster configuration: 16x1 cluster along M (shares B tiles).
constexpr int CLUSTER_M = 16;
constexpr int CLUSTER_N = 1;

__global__ __launch_bounds__(BLOCK_SIZE, 1)
void tma_nomulticast_gemm_kernel(
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [K, N]
    float* __restrict__ C,        // [M, N]
    int M, int N, int K
	) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto* B_smem = reinterpret_cast<float (*)[TILE_N]>(smem_raw);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const int tile_m = blockIdx.x;
	    const int tile_n = blockIdx.y;
	    const bool tile_valid = (tile_m * TILE_M < M) && (tile_n * TILE_N < N);

    const int tid = threadIdx.x;
    // 128 threads × (1×4 outputs/thread) = 512 outputs = 4×128 tile.
    constexpr int COLS_PER_THREAD = 4;
    constexpr int THREADS_PER_ROW = TILE_N / COLS_PER_THREAD;  // 32
    const int thread_m = tid / THREADS_PER_ROW;                // 0..3
    const int thread_n = (tid % THREADS_PER_ROW) * COLS_PER_THREAD;  // 0..124

    __shared__ alignas(128) float A_smem[TILE_M][TILE_K];

	    float acc[COLS_PER_THREAD] = {0.0f};
	    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
	
	    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
	        const int k_base = k_tile * TILE_K;
	
	        // Load A tile while the rank-0 B load is in flight.
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
	
        const float* b_tile = &B_smem[0][0];
	
	        #pragma unroll
	        for (int kk = 0; kk < TILE_K; ++kk) {
	            float a_val = A_smem[thread_m][kk];
	            #pragma unroll
	            for (int j = 0; j < COLS_PER_THREAD; ++j) {
                acc[j] += a_val * b_tile[kk * TILE_N + thread_n + j];
            }
        }
	
	        __syncthreads();
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

    __shared__ float A_smem[TILE_M][TILE_K];
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

int main(int argc, char** argv) {
    NVTX_RANGE("main");
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::printf("TMA Cluster GEMM Baseline (No Multicast)\n");
    std::printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

#if TMA_MULTICAST_TARGET == 0
    std::printf(
        "SKIPPED: compile target does not provide a comparable CTA cluster TMA multicast pair\n");
    return 3;
#endif

    if (prop.major < 9) {
        std::printf("SKIPPED: requires SM90+ for TMA/cluster launch\n");
        return 3;
    }

    cudaFuncAttributes kernel_attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(
        &kernel_attributes,
        tma_nomulticast_gemm_kernel));
    int max_shared_bytes = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &max_shared_bytes,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        0));
    const size_t required_shared_bytes =
        DYNAMIC_SMEM_BYTES + kernel_attributes.sharedSizeBytes;
    if (required_shared_bytes > static_cast<size_t>(max_shared_bytes)) {
        std::printf(
            "SKIPPED: baseline requires %zu bytes of shared memory, device permits %d\n",
            required_shared_bytes,
            max_shared_bytes);
        return 3;
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        tma_nomulticast_gemm_kernel,
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

    dim3 block(BLOCK_SIZE);
    dim3 grid((M + TILE_M - 1) / TILE_M,
              (N + TILE_N - 1) / TILE_N);

    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_M;
    attrs[0].val.clusterDim.y = CLUSTER_N;
    attrs[0].val.clusterDim.z = 1;

    CUDA_CHECK(cudaFuncSetAttribute(
        tma_nomulticast_gemm_kernel,
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
    CUDA_CHECK(cudaLaunchKernelEx(&config, tma_nomulticast_gemm_kernel, d_A, d_B, d_C, M, N, K));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        NVTX_RANGE("iteration");
        CUDA_CHECK(cudaLaunchKernelEx(&config, tma_nomulticast_gemm_kernel, d_A, d_B, d_C, M, N, K));
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
