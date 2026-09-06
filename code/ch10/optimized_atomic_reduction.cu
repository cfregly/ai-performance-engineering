// optimized_atomic_reduction.cu - Single-Pass Atomic Reduction (Ch10)
//
// CHAPTER 10 CONTEXT: "Tensor Core Pipelines & Cluster Features"
// This is the DSMEM-FREE optimized version for cross-block reduction
// Compare with baseline_atomic_reduction.cu (two-pass)
//
// APPROACH:
//   Single kernel: Each block reduces locally then atomicAdd to output
//
// WHY FASTER than two-pass:
//   - Single kernel launch (no launch overhead)
//   - No global memory round-trip for partial sums
//   - Atomic operations are highly optimized on modern GPUs
//
// TRADE-OFF:
//   - Atomic contention at high block counts
//   - Not as fast as DSMEM (when available) due to global memory atomics

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
constexpr int GROUPS_PER_OUTPUT = 4;  // Match DSMEM cluster size for comparison

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
    
    val = warp_reduce_sum(val);
    
    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

//============================================================================
// Single-Pass Atomic Reduction Kernel
//============================================================================

__global__ void single_pass_atomic_reduction(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int blocks_per_output
) {
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;
    const int output_idx = blockIdx.x / blocks_per_output;
    
    __shared__ float smem[32];
    
    // Local reduction
    float local_sum = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < ELEMENTS_PER_BLOCK; i += BLOCK_SIZE) {
        int global_idx = block_offset + i;
        if (global_idx < N) {
            local_sum += input[global_idx];
        }
    }
    
    // Block-level reduction
    float block_sum = block_reduce_sum(local_sum, smem);
    
    // Single atomic add directly to output (no intermediate storage)
    if (tid == 0) {
        atomicAdd(&output[output_idx], block_sum);
    }
}

//============================================================================
// Main
//============================================================================

int main() {
    NVTX_RANGE("main");
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Single-Pass Atomic Reduction (DSMEM-Free Optimized)\n");
    printf("===================================================\n");
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Note: This works on ANY CUDA device (no cluster required)\n\n");
    
    // Problem size
    const int N = 16 * 1024 * 1024;
    const int elements_per_group = ELEMENTS_PER_BLOCK * GROUPS_PER_OUTPUT;
    const int num_groups = (N + elements_per_group - 1) / elements_per_group;
    const int num_blocks = num_groups * GROUPS_PER_OUTPUT;
    
    printf("Problem: %d elements (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("Blocks: %d, Output groups: %d\n\n", num_blocks, num_groups);
    
    // Allocate
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_groups * sizeof(float)));
    
    // Initialize
    std::vector<float> h_input(N, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        NVTX_RANGE("warmup");
        CUDA_CHECK(cudaMemset(d_output, 0, num_groups * sizeof(float)));
        single_pass_atomic_reduction<<<num_blocks, BLOCK_SIZE>>>(
            d_input, d_output, N, GROUPS_PER_OUTPUT);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    const int iterations = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        NVTX_RANGE("compute_kernel:single_pass_atomic_reduction");
        CUDA_CHECK(cudaMemset(d_output, 0, num_groups * sizeof(float)));
        // SINGLE KERNEL LAUNCH per iteration
        single_pass_atomic_reduction<<<num_blocks, BLOCK_SIZE>>>(
            d_input, d_output, N, GROUPS_PER_OUTPUT);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    // Verify
    std::vector<float> h_output(num_groups);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_groups * sizeof(float), cudaMemcpyDeviceToHost));
    float total = std::accumulate(h_output.begin(), h_output.end(), 0.0f);
    
    printf("Single-Pass Atomic Reduction (Optimized):\n");
    printf("  Time: %.3f ms\n", ms / iterations);
    printf("  Timing includes cudaMemset(d_output, 0, ...) before each iteration\n");
    printf("  Sum: %.0f (expected: %d) - %s\n", total, N,
           (abs(total - N) < 1000) ? "PASS" : "FAIL");
    printf("\nAdvantages over two-pass:\n");
    printf("  - Single kernel launch\n");
    printf("  - No intermediate global memory buffer\n");
    printf("  - Modern GPUs have fast atomics\n");

    VERIFY_PRINT_CHECKSUM(total);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}
