// baseline_dsmem_reduction.cu - Standard Two-Pass Reduction (Ch10)
//
// WHAT: Traditional reduction that requires:
//   1. Per-block reduction to shared memory
//   2. Global memory write of partial sums
//   3. Second kernel for final reduction
//
// WHY THIS IS SLOWER:
//   - Requires global memory round-trip between passes
//   - Two kernel launches = more overhead
//   - No cross-CTA communication within a pass
//
// COMPARE WITH: optimized_dsmem_reduction.cu
//   - Optimized uses DSMEM for cross-CTA reduction within cluster
//   - Single kernel, single global memory write per cluster
//   - Eliminates intermediate global memory traffic

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

constexpr int BLOCK_SIZE = 256;
constexpr int ELEMENTS_PER_BLOCK = 4096;

namespace {
int env_or_default(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return fallback;
    }
    char* end = nullptr;
    long parsed = std::strtol(raw, &end, 10);
    if (end == raw || (end && *end != '\0') || parsed < 0 || parsed > 1000000) {
        return fallback;
    }
    return static_cast<int>(parsed);
}
}  // namespace

//============================================================================
// Warp-level reduction
//============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

//============================================================================
// Block-level reduction
//============================================================================

__device__ float block_reduce_sum(float val, float* smem) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;
    
    // Warp reduction
    val = warp_reduce_sum(val);
    
    // Write warp results to shared memory
    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    
    // First warp reduces all warp results
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

//============================================================================
// Pass 1: Block-level reduction to partial sums
//============================================================================

__global__ void baseline_block_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ partial_sums,
    int N
) {
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;
    
    __shared__ float smem_reduce[32];
    
    // Each thread accumulates multiple elements
    float local_sum = 0.0f;
    for (int i = tid; i < ELEMENTS_PER_BLOCK; i += BLOCK_SIZE) {
        int global_idx = block_offset + i;
        if (global_idx < N) {
            local_sum += input[global_idx];
        }
    }
    
    // Block-level reduction
    float block_sum = block_reduce_sum(local_sum, smem_reduce);
    
    // Write partial sum to global memory (THIS IS THE BOTTLENECK)
    if (tid == 0) {
        partial_sums[blockIdx.x] = block_sum;
    }
}

//============================================================================
// Pass 2: Final reduction of partial sums
//============================================================================

__global__ void baseline_final_reduction_kernel(
    const float* __restrict__ partial_sums,
    float* __restrict__ output,
    int num_blocks,
    int blocks_per_output
) {
    const int output_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int start_block = output_idx * blocks_per_output;
    
    __shared__ float smem_reduce[32];
    
    // Each thread accumulates multiple partial sums
    float local_sum = 0.0f;
    for (int i = tid; i < blocks_per_output; i += BLOCK_SIZE) {
        int block_idx = start_block + i;
        if (block_idx < num_blocks) {
            local_sum += partial_sums[block_idx];
        }
    }
    
    // Block-level reduction
    float block_sum = block_reduce_sum(local_sum, smem_reduce);
    
    // Write final result
    if (tid == 0) {
        output[output_idx] = block_sum;
    }
}

//============================================================================
// Benchmark
//============================================================================

int main() {
    NVTX_RANGE("main");
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Baseline Two-Pass Reduction\n");
    printf("===========================\n");
    printf("Device: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);
    
    // Problem size
    const int N = 64 * 1024 * 1024;  // 64M elements
    const int CLUSTER_SIZE = 4;  // Match optimized for fair comparison
    const int elements_per_cluster = ELEMENTS_PER_BLOCK * CLUSTER_SIZE;
    const int num_clusters = (N + elements_per_cluster - 1) / elements_per_cluster;
    const int num_blocks = num_clusters * CLUSTER_SIZE;
    
    printf("Problem Size:\n");
    printf("  Elements: %d (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("  Blocks: %d\n", num_blocks);
    printf("  Output clusters: %d\n\n", num_clusters);
    
    // Allocate
    float *d_input, *d_output, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_clusters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));
    
    // Initialize with known pattern
    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i) {
        NVTX_RANGE("warmup");
        h_input[i] = 1.0f;  // Sum should equal N
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int warmup = env_or_default("AISP_CUDA_BINARY_PROFILE_WARMUP", 5);
    const int iterations = env_or_default("AISP_CUDA_BINARY_PROFILE_ITERATIONS", 50);
    
    //========================================================================
    // Benchmark Two-Pass Reduction
    //========================================================================
    CUDA_CHECK(cudaMemset(d_output, 0, num_clusters * sizeof(float)));
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        NVTX_RANGE("warmup");
        baseline_block_reduction_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
        baseline_final_reduction_kernel<<<num_clusters, BLOCK_SIZE>>>(d_partial, d_output, num_blocks, CLUSTER_SIZE);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        NVTX_RANGE("compute_kernel:baseline_block_reduction_kernel");
        baseline_block_reduction_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
        baseline_final_reduction_kernel<<<num_clusters, BLOCK_SIZE>>>(d_partial, d_output, num_blocks, CLUSTER_SIZE);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    
    // Verify
    std::vector<float> h_output(num_clusters);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_clusters * sizeof(float), cudaMemcpyDeviceToHost));
    float total = std::accumulate(h_output.begin(), h_output.end(), 0.0f);
    
    printf("Results:\n");
    printf("  Time: %.3f ms\n", avg_ms);
    printf("  Sum: %.0f (expected: %d)\n", total, N);
    printf("\nNote: Two-pass reduction requires global memory round-trip.\n");
    printf("Compare with optimized_dsmem_reduction for single-pass cluster reduction.\n");

    VERIFY_PRINT_CHECKSUM(total);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_partial));
    
    return 0;
}
