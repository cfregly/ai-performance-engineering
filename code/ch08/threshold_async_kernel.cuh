#pragma once

#include "threshold_common.cuh"

namespace ch08 {

enum class ThresholdAsyncLaunchResult {
    kSuccess,
    kUnsupported,
    kFailed,
};

#if CUDART_VERSION >= 12000
__global__ void threshold_predicated_async_kernel(
    const float* __restrict__ inputs,
    float* __restrict__ output,
    float threshold,
    int count) {
    extern __shared__ __align__(16) float shmem[];

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int stages = 2;
    constexpr int values_per_thread = 4;
    constexpr int tiles_per_block = 4;
    const int tile_span = blockDim.x * values_per_thread;
    const int total_tiles = (count + tile_span - 1) / tile_span;
    const int block_tile_base = blockIdx.x * tiles_per_block;
    if (block_tile_base >= total_tiles) {
        return;
    }
    int tiles_for_block = tiles_per_block;
    const int tiles_remaining = total_tiles - block_tile_base;
    if (tiles_for_block > tiles_remaining) {
        tiles_for_block = tiles_remaining;
    }
    int stage = 0;

    auto stage_ptr = [&](int idx) {
        return shmem + idx * tile_span;
    };

    auto issue_cp_async = [&](int stage_idx, int tile_offset) {
        const int vec = threadIdx.x * values_per_thread;
        float* dst = stage_ptr(stage_idx) + vec;
        const int global_idx = tile_offset + vec;
        if (global_idx + (values_per_thread - 1) < count) {
            const unsigned dst_addr =
                static_cast<unsigned>(__cvta_generic_to_shared(dst));
            const unsigned long long src_addr =
                reinterpret_cast<unsigned long long>(inputs + global_idx);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :
                         : "r"(dst_addr), "l"(src_addr));
        } else {
            #pragma unroll
            for (int i = 0; i < values_per_thread; ++i) {
                const int idx = global_idx + i;
                dst[i] = (idx >= tile_offset && idx < count) ? inputs[idx] : 0.0f;
            }
        }
    };

    auto wait_for_stage = [&](bool allow_overlap) {
        if (allow_overlap) {
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();
    };

    auto tile_offset = [&](int local_tile) {
        return (block_tile_base + local_tile) * tile_span;
    };

    int tiles_enqueued = 0;
    int initial_tiles = stages;
    if (initial_tiles > tiles_for_block) {
        initial_tiles = tiles_for_block;
    }
    for (; tiles_enqueued < initial_tiles; ++tiles_enqueued) {
        issue_cp_async(tiles_enqueued % stages, tile_offset(tiles_enqueued));
        asm volatile("cp.async.commit_group;\n");
    }
    wait_for_stage(tiles_for_block > 1);

    int tiles_processed = 0;
    while (tiles_processed < tiles_for_block) {
        const int current_tile_offset = tile_offset(tiles_processed);
        const int base_idx = current_tile_offset + threadIdx.x * values_per_thread;
        float* tile_ptr = stage_ptr(stage) + threadIdx.x * values_per_thread;

        float results[values_per_thread];
        #pragma unroll
        for (int i = 0; i < values_per_thread; ++i) {
            const int out_idx = base_idx + i;
            float transformed = transform_with_scale(tile_ptr[i], threshold);
            results[i] = (out_idx < count) ? transformed : 0.0f;
        }

        #pragma unroll
        for (int i = 0; i < values_per_thread; i += 4) {
            const int idx = base_idx + i;
            if (idx + 3 < count) {
                float4 vec = make_float4(
                    results[i + 0],
                    results[i + 1],
                    results[i + 2],
                    results[i + 3]);
                reinterpret_cast<float4*>(output + idx)[0] = vec;
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int tail = idx + j;
                    if (tail < count) {
                        output[tail] = results[i + j];
                    }
                }
            }
        }

        ++tiles_processed;

        if (tiles_enqueued < tiles_for_block) {
            const int refill_stage = stage;
            issue_cp_async(refill_stage, tile_offset(tiles_enqueued));
            asm volatile("cp.async.commit_group;\n");
            ++tiles_enqueued;
        }

        stage ^= 1;
        if (tiles_processed < tiles_for_block) {
            // One pending group is the next tile itself: it must be complete.
            // Only leave a group pending when a later tile is also enqueued.
            const bool keep_inflight = (tiles_enqueued - tiles_processed) > 1;
            wait_for_stage(keep_inflight);
        }
    }
#else
    (void)inputs;
    (void)output;
    (void)threshold;
    (void)count;
#endif
}
#endif  // CUDART_VERSION >= 12000

inline ThresholdAsyncLaunchResult launch_threshold_predicated_async(
    const float* inputs,
    float* output,
    float threshold,
    int count,
    cudaStream_t stream,
    cudaError_t* launch_error = nullptr) {
#if CUDART_VERSION >= 12000
    const dim3 block(kThresholdOptimizedThreads);
    constexpr int values_per_thread = 4;
    constexpr int tiles_per_block = 4;
    const int tile_span = block.x * values_per_thread;
    const int total_tiles = (count + tile_span - 1) / tile_span;
    int grid_x = (total_tiles + tiles_per_block - 1) / tiles_per_block;
    if (grid_x < 1) {
        grid_x = 1;
    }
    const dim3 grid(grid_x);
    const size_t shared_bytes = 2 * tile_span * sizeof(float);
    threshold_predicated_async_kernel<<<grid, block, shared_bytes, stream>>>(
        inputs,
        output,
        threshold,
        count);
    cudaError_t err = cudaGetLastError();
    if (launch_error != nullptr) {
        *launch_error = err;
    }
    if (err == cudaSuccess) {
        return ThresholdAsyncLaunchResult::kSuccess;
    }
    if (err == cudaErrorInvalidDeviceFunction ||
        err == cudaErrorNoKernelImageForDevice ||
        err == cudaErrorNotSupported) {
        return ThresholdAsyncLaunchResult::kUnsupported;
    }
    return ThresholdAsyncLaunchResult::kFailed;
#else
    (void)inputs;
    (void)output;
    (void)threshold;
    (void)count;
    (void)stream;
    if (launch_error != nullptr) {
        *launch_error = cudaErrorNotSupported;
    }
    return ThresholdAsyncLaunchResult::kUnsupported;
#endif
}

}  // namespace ch08
