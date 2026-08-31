// Shared-memory scalar GEMM controls (legacy test_tma file/target name).
// No TMA, asynchronous copy, tensor-core, or TMEM instruction is issued.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do { const cudaError_t err = (call); if (err != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); std::exit(1); } } while (0)

// Single-stage scalar shared-memory tiling.
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void tma_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    static_assert(TILE_M * TILE_N <= 1024, "block exceeds CUDA thread limit");
    static_assert(TILE_K <= TILE_M && TILE_K <= TILE_N, "tile loads require enough x/y threads");
    
    // Shared memory for tiled scalar computation
    __shared__ float smem_A[TILE_M][TILE_K];
    __shared__ float smem_B[TILE_K][TILE_N];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;
    
    float sum = 0.0f;
    
    // Number of tiles along K dimension
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        // Ordinary thread-issued global loads into shared memory.
        
        // Load A tile
        if (ty < TILE_M && tx < TILE_K) {
            int a_row = by * TILE_M + ty;
            int a_col = kt * TILE_K + tx;
            if (a_row < M && a_col < K) {
                smem_A[ty][tx] = A[a_row * K + a_col];
            } else {
                smem_A[ty][tx] = 0.0f;
            }
        }
        
        // Load B tile
        if (ty < TILE_K && tx < TILE_N) {
            int b_row = kt * TILE_K + ty;
            int b_col = bx * TILE_N + tx;
            if (b_row < K && b_col < N) {
                smem_B[ty][tx] = B[b_row * N + b_col];
            } else {
                smem_B[ty][tx] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute on tile
        if (ty < TILE_M && tx < TILE_N) {
            #pragma unroll
            for (int k = 0; k < TILE_K; ++k) {
                sum += smem_A[ty][k] * smem_B[k][tx];
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N && ty < TILE_M && tx < TILE_N) {
        C[row * N + col] = sum;
    }
}

// Multi-buffer scalar control; loads are synchronous, not TMA.
template<int TILE_M, int TILE_N, int TILE_K, int NUM_STAGES>
__global__ void tma_pipelined_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    static_assert(TILE_M * TILE_N <= 1024, "block exceeds CUDA thread limit");
    static_assert(TILE_K <= TILE_M && TILE_K <= TILE_N, "tile loads require enough x/y threads");
    static_assert(NUM_STAGES > 0, "at least one buffer is required");
    __shared__ float smem_A[NUM_STAGES][TILE_M][TILE_K];
    __shared__ float smem_B[NUM_STAGES][TILE_K][TILE_N];
    const int ACTUAL_STAGES = NUM_STAGES;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;
    
    float sum = 0.0f;
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    // Prefetch first stages
    for (int s = 0; s < min(ACTUAL_STAGES, num_k_tiles); ++s) {
        int stage = s % ACTUAL_STAGES;
        
        // Load A tile for stage
        if (ty < TILE_M && tx < TILE_K) {
            int a_row = by * TILE_M + ty;
            int a_col = stage * TILE_K + tx;
            if (a_row < M && a_col < K) {
                smem_A[stage][ty][tx] = A[a_row * K + a_col];
            } else {
                smem_A[stage][ty][tx] = 0.0f;
            }
        }
        
        // Load B tile for stage
        if (ty < TILE_K && tx < TILE_N) {
            int b_row = stage * TILE_K + ty;
            int b_col = bx * TILE_N + tx;
            if (b_row < K && b_col < N) {
                smem_B[stage][ty][tx] = B[b_row * N + b_col];
            } else {
                smem_B[stage][ty][tx] = 0.0f;
            }
        }
        
        __syncthreads();
    }
    
    // Main loop with pipelining
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int stage = kt % ACTUAL_STAGES;
        
        // Compute on current stage
        if (ty < TILE_M && tx < TILE_N) {
            #pragma unroll
            for (int k = 0; k < TILE_K; ++k) {
                sum += smem_A[stage][ty][k] * smem_B[stage][k][tx];
            }
        }
        
        // Prefetch next stage if available
        int next_kt = kt + ACTUAL_STAGES;
        if (next_kt < num_k_tiles) {
            __syncthreads();
            
            // Load A tile for next stage
            if (ty < TILE_M && tx < TILE_K) {
                int a_row = by * TILE_M + ty;
                int a_col = next_kt * TILE_K + tx;
                if (a_row < M && a_col < K) {
                    smem_A[stage][ty][tx] = A[a_row * K + a_col];
                } else {
                    smem_A[stage][ty][tx] = 0.0f;
                }
            }
            
            // Load B tile for next stage
            if (ty < TILE_K && tx < TILE_N) {
                int b_row = next_kt * TILE_K + ty;
                int b_col = bx * TILE_N + tx;
                if (b_row < K && b_col < N) {
                    smem_B[stage][ty][tx] = B[b_row * N + b_col];
                } else {
                    smem_B[stage][ty][tx] = 0.0f;
                }
            }
            
            __syncthreads();
        }
    }
    
    // Write result
    if (row < M && col < N && ty < TILE_M && tx < TILE_N) {
        C[row * N + col] = sum;
    }
}

bool verify_result(const float* C, int M, int N, float expected) {
    for (int i = 0; i < M * N; ++i) {
        if (!std::isfinite(C[i]) || fabs(C[i] - expected) > 1e-3) {
            printf("Verification failed at index %d: expected %f, got %f\n", 
                   i, expected, C[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("=== Shared-memory scalar GEMM controls (legacy test_tma target) ===\n\n");
    
    // Check compute capability
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    printf("This control uses ordinary shared-memory loads and scalar FP32 arithmetic.\n");
    printf("It does not exercise TMA; TMA was introduced with Hopper (SM90).\n\n");

    // Test configuration
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    const int TILE_M = 32;
    const int TILE_N = 32;
    const int TILE_K = 32;
    const int NUM_STAGES = 4;
    
    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    
    // Initialize with simple values
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Test 1: Single-buffer scalar GEMM
    printf("Test 1: Single-buffer scalar GEMM\n");
    dim3 block1(TILE_N, TILE_M);
    dim3 grid1((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    tma_gemm_kernel<TILE_M, TILE_N, TILE_K><<<grid1, block1>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    float ms1 = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms1, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    float expected = (float)K;
    bool passed1 = verify_result(h_C, M, N, expected);
    printf("  Time: %.3f ms\n", ms1);
    printf("  Result: %s\n\n", passed1 ? "PASSED" : "FAILED");
    
    // Test 2: Scalar GEMM with multiple buffers
    printf("Test 2: Scalar GEMM with %d buffers\n", NUM_STAGES);
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaEventRecord(start));
    tma_pipelined_gemm_kernel<TILE_M, TILE_N, TILE_K, NUM_STAGES><<<grid1, block1>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    float ms2 = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms2, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool passed2 = verify_result(h_C, M, N, expected);
    printf("  Time: %.3f ms\n", ms2);
    printf("  Result: %s\n", passed2 ? "PASSED" : "FAILED");
    if (!passed1 || !passed2 || !std::isfinite(ms1) || !std::isfinite(ms2) || ms1 <= 0 || ms2 <= 0) return 1;
    if (ms2 > 0) printf("  Verified control time ratio: %.2fx\n\n", ms1 / ms2);
    
    // Performance metrics
    double flops = 2.0 * M * N * K;
    double tflops1 = flops / (ms1 * 1e9);
    double tflops2 = flops / (ms2 * 1e9);
    
    printf("Performance:\n");
    printf("  Single-buffer scalar: %.2f TFLOPS\n", tflops1);
    printf("  Multi-buffer scalar: %.2f TFLOPS\n", tflops2);
    
    // Bandwidth estimation
    double bytes_transferred = (M * K + K * N + M * N) * sizeof(float);
    double bw1 = bytes_transferred / (ms1 * 1e6); // GB/s
    double bw2 = bytes_transferred / (ms2 * 1e6);
    
    printf("\nIdeal minimum-traffic rate (not measured memory bandwidth):\n");
    printf("  Single-buffer scalar: %.2f GB/s\n", bw1);
    printf("  Multi-buffer scalar: %.2f GB/s\n", bw2);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("\n=== Shared-memory control complete ===\n");
    printf("Status: %s\n", (passed1 && passed2) ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    
    return (passed1 && passed2) ? 0 : 1;
}

