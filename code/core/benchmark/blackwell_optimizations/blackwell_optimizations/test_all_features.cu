// Cluster-launch scalar GEMM control with FP8 storage.
// No TMA, TMEM, tensor-core arithmetic, warp specialization or remote DSMEM access.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do { const cudaError_t err = (call); if (err != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); std::exit(1); } } while (0)

namespace cg = cooperative_groups;

// FP8 storage, local shared-memory buffers and explicit cluster synchronization.
template<int TILE_M, int TILE_N, int TILE_K, int NUM_STAGES>
__global__ void __cluster_dims__(2, 2, 1) blackwell_ultra_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    static_assert(TILE_M * TILE_N <= 1024, "block exceeds CUDA thread limit");
    static_assert(TILE_K <= TILE_M && TILE_K <= TILE_N, "tile loads require enough x/y threads");
    static_assert(NUM_STAGES > 0, "at least one buffer is required");
    cg::cluster_group cluster = cg::this_cluster();
    __shared__ __nv_fp8_e4m3 smem_A[NUM_STAGES][TILE_M][TILE_K];
    __shared__ __nv_fp8_e4m3 smem_B[NUM_STAGES][TILE_K][TILE_N];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;
    
    float sum = 0.0f;
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    // Fill the initial buffers with ordinary thread-issued loads.
    for (int s = 0; s < min(NUM_STAGES, num_k_tiles); ++s) {
        
        // Load FP8 tiles
        if (ty < TILE_M && tx < TILE_K) {
            int a_row = by * TILE_M + ty;
            int a_col = s * TILE_K + tx;
            if (a_row < M && a_col < K) {
                smem_A[s][ty][tx] = A[a_row * K + a_col];
            } else {
                smem_A[s][ty][tx] = __nv_fp8_e4m3(0.0f);
            }
        }
        
        if (ty < TILE_K && tx < TILE_N) {
            int b_row = s * TILE_K + ty;
            int b_col = bx * TILE_N + tx;
            if (b_row < K && b_col < N) {
                smem_B[s][ty][tx] = B[b_row * N + b_col];
            } else {
                smem_B[s][ty][tx] = __nv_fp8_e4m3(0.0f);
            }
        }
        
        __syncthreads(); // publish all initial loads before any thread reads them
    }
    
    // Main compute loop with pipelining
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int stage = kt % NUM_STAGES;
        
        // Cluster synchronization only; there is no remote shared-memory access.
        cluster.sync();
        
        // Compute with FP8 -> FP32 accumulation
        if (ty < TILE_M && tx < TILE_N) {
            #pragma unroll
            for (int k = 0; k < TILE_K; ++k) {
                float a_val = (float)smem_A[stage][ty][k];
                float b_val = (float)smem_B[stage][k][tx];
                sum += a_val * b_val;
            }
        }
        
        // Prefetch next stage
        int next_kt = kt + NUM_STAGES;
        if (next_kt < num_k_tiles) {
            __syncthreads(); // all readers finish before this buffer is overwritten
            
            if (ty < TILE_M && tx < TILE_K) {
                int a_row = by * TILE_M + ty;
                int a_col = next_kt * TILE_K + tx;
                if (a_row < M && a_col < K) {
                    smem_A[stage][ty][tx] = A[a_row * K + a_col];
                } else {
                    smem_A[stage][ty][tx] = __nv_fp8_e4m3(0.0f);
                }
            }
            
            if (ty < TILE_K && tx < TILE_N) {
                int b_row = next_kt * TILE_K + ty;
                int b_col = bx * TILE_N + tx;
                if (b_row < K && b_col < N) {
                    smem_B[stage][ty][tx] = B[b_row * N + b_col];
                } else {
                    smem_B[stage][ty][tx] = __nv_fp8_e4m3(0.0f);
                }
            }
            
            __syncthreads(); // publish replacement values before the next read
        }
    }
    
    // Write result
    if (row < M && col < N && ty < TILE_M && tx < TILE_N) {
        C[row * N + col] = sum;
    }
}

// Feature detection and reporting
void report_blackwell_features() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("=== Selected CUDA device ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Streaming Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared Memory Per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("L2 Cache Size: %.2f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));
    printf("\n");
    
    printf("=== Mechanisms exercised by this control ===\n");
    printf("  - 2x2 CTA cluster launch and cluster synchronization\n");
    printf("  - FP8 E4M3 storage converted to scalar FP32 arithmetic\n");
    printf("  - Synchronous loads into local shared-memory buffers\n");
    printf("Not exercised: TMA, TMEM, tensor-core MMA, warp specialization or remote DSMEM.\n\n");
}

void float_to_fp8_e4m3(const float* input, __nv_fp8_e4m3* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = __nv_fp8_e4m3(input[i]);
    }
}

bool verify_result(const float* C, int M, int N, float expected, float tolerance) {
    int errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (!std::isfinite(C[i]) || fabs(C[i] - expected) > tolerance) {
            if (errors < 5) {
                printf("  Error at %d: expected %f, got %f\n", i, expected, C[i]);
            }
            errors++;
        }
    }
    if (errors > 0) {
        printf("  Total errors: %d / %d (%.2f%%)\n", errors, M * N, 100.0 * errors / (M * N));
        return false;
    }
    return true;
}

int main() {
    printf("========================================\n");
    printf("  CLUSTER / FP8 STORAGE CONTROL\n");
    printf("========================================\n\n");
    
    // Check device
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    int cluster_supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch, 0));
    if (!cluster_supported) {
        printf("UNSUPPORTED: this control requires thread-block cluster launch (Hopper or newer).\n");
        return 3;
    }

    report_blackwell_features();
    
    // Run comprehensive test
    printf("========================================\n");
    printf("  SCALAR CLUSTER CONTROL\n");
    printf("========================================\n\n");
    
    printf("Testing: cluster launch + FP8 storage + scalar shared-memory GEMM\n\n");
    
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    const int TILE_M = 32;
    const int TILE_N = 32;
    const int TILE_K = 32;
    const int NUM_STAGES = 4;
    
    // Allocate host memory
    float *h_A_fp32 = new float[M * K];
    float *h_B_fp32 = new float[K * N];
    float *h_C = new float[M * N];
    
    // Initialize
    for (int i = 0; i < M * K; ++i) h_A_fp32[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B_fp32[i] = 1.0f;
    
    // Convert to FP8
    __nv_fp8_e4m3 *h_A_fp8 = new __nv_fp8_e4m3[M * K];
    __nv_fp8_e4m3 *h_B_fp8 = new __nv_fp8_e4m3[K * N];
    
    float_to_fp8_e4m3(h_A_fp32, h_A_fp8, M * K);
    float_to_fp8_e4m3(h_B_fp32, h_B_fp8, K * N);
    
    // Allocate device memory
    __nv_fp8_e4m3 *d_A, *d_B;
    float *d_C;
    
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A_fp8, M * K * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_fp8, K * N * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    
    // Setup cluster launch
    dim3 block(TILE_N, TILE_M);
    dim3 grid(2 * (((N + TILE_N - 1) / TILE_N + 1) / 2),
              2 * (((M + TILE_M - 1) / TILE_M + 1) / 2));
    
    cudaLaunchConfig_t config = {0};
    config.gridDim = grid;
    config.blockDim = block;
    
    // __cluster_dims__ fixes the 2x2 cluster at compile time. No runtime override.

    // Warmup
    printf("Warming up...\n");
    void* kernel_args[] = {(void*)&d_A, (void*)&d_B, (void*)&d_C, (void*)&M, (void*)&N, (void*)&K};
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaLaunchKernelExC(
            &config,
            (void*)blackwell_ultra_gemm_kernel<TILE_M, TILE_N, TILE_K, NUM_STAGES>,
            kernel_args
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    printf("Running benchmark...\n");
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int num_iters = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; ++i) {
        cudaError_t err = cudaLaunchKernelExC(
            &config,
            (void*)blackwell_ultra_gemm_kernel<TILE_M, TILE_N, TILE_K, NUM_STAGES>,
            kernel_args
        );
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / num_iters;
    
    // Verify
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    float expected = (float)K;
    bool passed = verify_result(h_C, M, N, expected, 1e-3f);
    
    printf("\n========================================\n");
    printf("  RESULTS\n");
    printf("========================================\n\n");
    
    printf("Matrix Size: %dx%dx%d\n", M, N, K);
    printf("Tile Size: %dx%dx%d\n", TILE_M, TILE_N, TILE_K);
    printf("Pipeline Stages: %d\n", NUM_STAGES);
    printf("Cluster Dimensions: 2x2\n");
    printf("Precision: FP8 (E4M3) → FP32\n\n");
    
    if (!passed || !std::isfinite(avg_ms) || avg_ms <= 0) return 1;
    printf("Time (average): %.3f ms\n", avg_ms);
    
    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e9);
    printf("Performance: %.2f TFLOPS\n", tflops);
    
    size_t fp32_bytes = (M * K + K * N + M * N) * sizeof(float);
    size_t fp8_bytes = (M * K + K * N) * sizeof(__nv_fp8_e4m3) + M * N * sizeof(float);
    printf("Memory Savings: %.1f%% vs FP32\n", 100.0 * (1.0 - (double)fp8_bytes / fp32_bytes));
    
    double bandwidth = fp8_bytes / (avg_ms * 1e6);
    printf("Ideal minimum-traffic rate (not measured bandwidth): %.2f GB/s\n", bandwidth);
    
    printf("\nVerification: %s\n", passed ? "✓ PASSED" : "✗ FAILED");
    
    printf("\nMechanisms: cluster launch/sync, FP8 storage, scalar FP32 arithmetic and local shared memory.\n");
    printf("No TMA/TMEM/tensor-core/warp-specialized/remote-DSMEM qualification is implied.\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A_fp32;
    delete[] h_B_fp32;
    delete[] h_C;
    delete[] h_A_fp8;
    delete[] h_B_fp8;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("\n========================================\n");
    printf("  CLUSTER CONTROL COMPLETE\n");
    printf("========================================\n");
    
    return passed ? 0 : 1;
}

