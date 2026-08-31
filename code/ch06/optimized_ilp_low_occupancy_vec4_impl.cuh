#pragma once
// CUDA 13 + Blackwell: Uses Float8 (32-byte aligned) for 256-bit loads

#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../core/common/headers/profiling_helpers.cuh"

// CUDA 13 + Blackwell: 32-byte aligned type for 256-bit loads
struct alignas(32) Float8 {
    float elems[8];
};
static_assert(sizeof(Float8) == 32, "Float8 must be 32 bytes");
static_assert(alignof(Float8) == 32, "Float8 must be 32-byte aligned");

namespace ilp_low_occ_vec4 {

__global__ void independent_ops_kernel(float* __restrict__ output,
                                       const float* __restrict__ input,
                                       int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        float val = input[i];
        float val2 = val * 2.0f;
        float val3 = val + 1.0f;
        float val4 = val * 3.0f;
        float val5 = val - 5.0f;
        output[i] = val2 + val3 + val4 + val5;
    }
}

// ILP kernel using Float8 (256-bit loads, 8-way ILP)
__global__ void unrolled_ilp_kernel(float* __restrict__ output,
                                    const float* __restrict__ input,
                                    int N) {
    constexpr int VEC_WIDTH = 8;
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int idx = vec_idx; idx * VEC_WIDTH < N; idx += stride) {
        int base_idx = idx * VEC_WIDTH;
        if (base_idx + (VEC_WIDTH - 1) < N) {
            const Float8* input8 = reinterpret_cast<const Float8*>(input);
            Float8 vals = input8[idx];  // 256-bit load
            Float8 res;
            // 8-way ILP - all independent operations
            res.elems[0] = vals.elems[0] * 2.0f + 1.0f;
            res.elems[1] = vals.elems[1] * 3.0f - 5.0f;
            res.elems[2] = vals.elems[2] * 4.0f + 2.0f;
            res.elems[3] = vals.elems[3] * 5.0f - 3.0f;
            res.elems[4] = vals.elems[4] * 2.5f + 0.5f;
            res.elems[5] = vals.elems[5] * 3.5f - 2.0f;
            res.elems[6] = vals.elems[6] * 4.5f + 1.5f;
            res.elems[7] = vals.elems[7] * 5.5f - 4.0f;
            Float8* output8 = reinterpret_cast<Float8*>(output);
            output8[idx] = res;  // 256-bit store
        } else {
            // Handle remainder
            for (int i = 0; i < VEC_WIDTH && base_idx + i < N; ++i) {
                float val = input[base_idx + i];
                switch (i) {
                    case 0: output[base_idx + i] = val * 2.0f + 1.0f; break;
                    case 1: output[base_idx + i] = val * 3.0f - 5.0f; break;
                    case 2: output[base_idx + i] = val * 4.0f + 2.0f; break;
                    case 3: output[base_idx + i] = val * 5.0f - 3.0f; break;
                    case 4: output[base_idx + i] = val * 2.5f + 0.5f; break;
                    case 5: output[base_idx + i] = val * 3.5f - 2.0f; break;
                    case 6: output[base_idx + i] = val * 4.5f + 1.5f; break;
                    default: output[base_idx + i] = val * 5.5f - 4.0f; break;
                }
            }
        }
    }
}

float measure_kernel(
    void (*kernel)(float*, const float*, int),
    float* d_output,
    const float* d_input,
    int N,
    int threads_per_block,
    int blocks,
    cudaStream_t stream,
    const char* name
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    if (blocks <= 0) {
        blocks = (N + threads_per_block - 1) / threads_per_block;
    }
    float best_ms = FLT_MAX;
    const int iterations = 200;
    for (int rep = 0; rep < 3; ++rep) {
        kernel<<<blocks, threads_per_block, 0, stream>>>(d_output, d_input, N);
        cudaStreamSynchronize(stream);
        cudaEventRecord(start, stream);
        {
            PROFILE_KERNEL_LAUNCH(name);
            for (int i = 0; i < iterations; ++i) {
                kernel<<<blocks, threads_per_block, 0, stream>>>(d_output, d_input, N);
            }
        }
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        best_ms = std::min(best_ms, ms / iterations);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return best_ms;
}

inline int run_ilp_low_occupancy_vec4(const char* title, int max_active_blocks_override) {
    const int N = 10'000'000;
    const int threads_per_block = 256;
    int total_blocks = (N + threads_per_block - 1) / threads_per_block;
    int active_blocks = total_blocks;
    if (max_active_blocks_override > 0) {
        active_blocks = std::min(total_blocks, max_active_blocks_override);
    }

    printf("========================================\n");
    printf("%s\n", title);
    printf("========================================\n");
    printf("Problem size: %d elements\n", N);
    printf("Threads per block: %d\n", threads_per_block);
    if (max_active_blocks_override > 0) {
        printf("Active blocks per kernel: %d (capped)\n", active_blocks);
    } else {
        printf("Active blocks per kernel: %d (full occupancy)\n", active_blocks);
    }
    printf("Iterations per measurement: 200 (best-of-3)\n\n");

    float* h_input = nullptr;
    float* h_output_indep = nullptr;
    float* h_output_unrolled = nullptr;
    cudaMallocHost(&h_input, N * sizeof(float));
    cudaMallocHost(&h_output_indep, N * sizeof(float));
    cudaMallocHost(&h_output_unrolled, N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaMallocAsync(&d_input, N * sizeof(float), stream);
    cudaMallocAsync(&d_output, N * sizeof(float), stream);
    {
        PROFILE_MEMORY_COPY("H2D copy");
        cudaMemcpyAsync(d_input, h_input, N * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
    }
    cudaStreamSynchronize(stream);

    printf("1. Independent operations (grid-stride scalar):\n");
    float indep_time = measure_kernel(
        independent_ops_kernel, d_output, d_input, N,
        threads_per_block, active_blocks, stream, "independent_ops");
    printf("   Time: %.3f ms\n", indep_time);
    cudaMemcpyAsync(h_output_indep, d_output, N * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    printf("\n2. Vectorized ILP (Float8 loads/stores, 8-way ILP):\n");
    float unrolled_time = measure_kernel(
        unrolled_ilp_kernel, d_output, d_input, N,
        threads_per_block, active_blocks, stream, "unrolled_ilp");
    printf("   Time: %.3f ms\n", unrolled_time);
    cudaMemcpyAsync(h_output_unrolled, d_output, N * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    bool indep_correct = true;
    bool unrolled_correct = true;
    for (int i = 0; i < N; ++i) {
        const float expected_indep = h_input[i] * 7.0f - 4.0f;
        if (!std::isfinite(h_output_indep[i]) ||
            fabsf(h_output_indep[i] - expected_indep) > 1e-5f * std::max(1.0f, fabsf(expected_indep))) {
            indep_correct = false;
        }
        float expected_unrolled = 0.0f;
        switch (i % 8) {
            case 0: expected_unrolled = h_input[i] * 2.0f + 1.0f; break;
            case 1: expected_unrolled = h_input[i] * 3.0f - 5.0f; break;
            case 2: expected_unrolled = h_input[i] * 4.0f + 2.0f; break;
            case 3: expected_unrolled = h_input[i] * 5.0f - 3.0f; break;
            case 4: expected_unrolled = h_input[i] * 2.5f + 0.5f; break;
            case 5: expected_unrolled = h_input[i] * 3.5f - 2.0f; break;
            case 6: expected_unrolled = h_input[i] * 4.5f + 1.5f; break;
            case 7: expected_unrolled = h_input[i] * 5.5f - 4.0f; break;
        }
        if (!std::isfinite(h_output_unrolled[i]) ||
            fabsf(h_output_unrolled[i] - expected_unrolled) > 1e-5f * std::max(1.0f, fabsf(expected_unrolled))) {
            unrolled_correct = false;
        }
    }

    printf("\n========================================\n");
    printf("Results:\n");
    printf("  Independent:     %s\n", indep_correct ? "✓ Correct" : "✗ Incorrect");
    printf("  Vectorized ILP:  %s\n", unrolled_correct ? "✓ Correct" : "✗ Incorrect");
    if (indep_time > 0 && unrolled_time > 0) {
        printf("  Time ratio:      %.2fx (different arithmetic workloads, not a speedup comparison)\n", indep_time / unrolled_time);
    }
    printf("\nKey insight: Float8 (256-bit) vectorization + 8-way ILP maximize memory bandwidth.\n");
    printf("========================================\n");

    cudaFreeAsync(d_output, stream);
    cudaFreeAsync(d_input, stream);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_output_unrolled);
    cudaFreeHost(h_output_indep);
    cudaFreeHost(h_input);

    return (indep_correct && unrolled_correct) ? 0 : 1;
}

}  // namespace ilp_low_occ_vec4

inline int run_ilp_low_occupancy_vec4(const char* title, int max_active_blocks_override) {
    return ilp_low_occ_vec4::run_ilp_low_occupancy_vec4(title, max_active_blocks_override);
}
