// baseline_atomic_reduction.cu - Two-Pass Block Reduction Baseline (Ch10)
//
// CHAPTER 10 CONTEXT: "Tensor Core Pipelines & Cluster Features"
// This is the DSMEM-FREE baseline for cross-block reduction
// Compare with optimized_atomic_reduction.cu (single-pass atomic)
//
// APPROACH:
//   Pass 1: Each block reduces to a partial sum in global memory
//   Pass 2: Second kernel reduces partial sums to final output
//
// WHY SLOWER:
//   - Requires TWO kernel launches (launch overhead)
//   - Global memory round-trip between passes (bandwidth cost)
//   - No cross-block communication within a single pass

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
// Pass 1: Per-block reduction to partial sums
//============================================================================

__global__ void pass1_block_reduction(
    const float* __restrict__ input,
    float* __restrict__ partial_sums,
    int N
) {
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;
    
    __shared__ float smem[32];
    
    float local_sum = 0.0f;
    for (int i = tid; i < ELEMENTS_PER_BLOCK; i += BLOCK_SIZE) {
        int global_idx = block_offset + i;
        if (global_idx < N) {
            local_sum += input[global_idx];
        }
    }
    
    float block_sum = block_reduce_sum(local_sum, smem);
    
    // GLOBAL MEMORY WRITE - this is the bottleneck
    if (tid == 0) {
        partial_sums[blockIdx.x] = block_sum;
    }
}

//============================================================================
// Pass 2: Reduce partial sums to final output
//============================================================================

__global__ void pass2_final_reduction(
    const float* __restrict__ partial_sums,
    float* __restrict__ output,
    int num_blocks,
    int blocks_per_output
) {
    const int output_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int start_block = output_idx * blocks_per_output;
    
    __shared__ float smem[32];
    
    float local_sum = 0.0f;
    for (int i = tid; i < blocks_per_output; i += BLOCK_SIZE) {
        int block_idx = start_block + i;
        if (block_idx < num_blocks) {
            // GLOBAL MEMORY READ of partial sums
            local_sum += partial_sums[block_idx];
        }
    }
    
    float block_sum = block_reduce_sum(local_sum, smem);
    
    if (tid == 0) {
        output[output_idx] = block_sum;
    }
}

//============================================================================
// Main
//============================================================================

int main() {
    NVTX_RANGE("main");
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Two-Pass Block Reduction (DSMEM-Free Baseline)\n");
    printf("==============================================\n");
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Note: This works on ANY CUDA device (no cluster required)\n\n");
    
    // Problem size (match DSMEM version for fair comparison)
    const int N = 16 * 1024 * 1024;
    const int elements_per_group = ELEMENTS_PER_BLOCK * GROUPS_PER_OUTPUT;
    const int num_groups = (N + elements_per_group - 1) / elements_per_group;
    const int num_blocks = num_groups * GROUPS_PER_OUTPUT;
    
    printf("Problem: %d elements (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("Blocks: %d, Groups: %d\n\n", num_blocks, num_groups);
    
    // Allocate
    float *d_input, *d_output, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_groups * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));
    
    // Initialize
    std::vector<float> h_input(N, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        NVTX_RANGE("warmup");
        pass1_block_reduction<<<num_blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
        pass2_final_reduction<<<num_groups, BLOCK_SIZE>>>(d_partial, d_output, num_blocks, GROUPS_PER_OUTPUT);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    const int iterations = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        NVTX_RANGE("compute_kernel:pass1_block_reduction");
        // TWO KERNEL LAUNCHES per iteration
        pass1_block_reduction<<<num_blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
        pass2_final_reduction<<<num_groups, BLOCK_SIZE>>>(d_partial, d_output, num_blocks, GROUPS_PER_OUTPUT);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    // Verify
    std::vector<float> h_output(num_groups);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_groups * sizeof(float), cudaMemcpyDeviceToHost));
    float total = std::accumulate(h_output.begin(), h_output.end(), 0.0f);
    
    printf("Two-Pass Reduction (Baseline):\n");
    printf("  Time: %.3f ms\n", ms / iterations);
    printf("  Sum: %.0f (expected: %d) - %s\n", total, N,
           (abs(total - N) < 1000) ? "PASS" : "FAIL");
    printf("\nBottlenecks:\n");
    printf("  - 2 kernel launches per iteration\n");
    printf("  - Global memory round-trip for partial sums\n");

    VERIFY_PRINT_CHECKSUM(total);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_partial));
    
    return 0;
}
