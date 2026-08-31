#pragma once

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <limits>
#include <cstdint>
#include <mutex>

#include "threshold_common.cuh"

namespace ch08 {

namespace cg = cooperative_groups;

#if CUDART_VERSION >= 12000
template <int ValuesPerThread>
__global__ void threshold_tma_pipeline_kernel(
    const float* __restrict__ inputs,
    float* __restrict__ output,
    float threshold,
    int count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    constexpr int stages = 2;
    constexpr int values_per_thread = ValuesPerThread;
    const int tile_span = blockDim.x * values_per_thread;
    const int stride = gridDim.x * tile_span;
    int tile_start = blockIdx.x * tile_span;
    if (tile_start >= count) {
        return;
    }

    extern __shared__ __align__(16) float shmem[];
    auto stage_ptr = [&](int stage) {
        return shmem + stage * tile_span;
    };

    cg::thread_block block = cg::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, stages> pipeline_state;
    auto pipe = cuda::make_pipeline(block, &pipeline_state);

    auto enqueue_tile = [&](int stage, int offset) -> bool {
        if (offset >= count) {
            return false;
        }
        pipe.producer_acquire();
        const size_t elems = min(tile_span, count - offset);
        const size_t copy_bytes = static_cast<size_t>(elems) * sizeof(float);
        if (copy_bytes % 16 == 0 &&
            reinterpret_cast<uintptr_t>(inputs + offset) % 16 == 0) {
            cuda::memcpy_async(block, stage_ptr(stage), inputs + offset,
                               cuda::aligned_size_t<16>(copy_bytes), pipe);
        } else {
            // Ragged counts or offset input views cannot promise 16-byte alignment.
            cuda::memcpy_async(block, stage_ptr(stage), inputs + offset, copy_bytes, pipe);
        }
        pipe.producer_commit();
        return true;
    };

    bool stage_ready[stages] = {false, false};
    int stage_offsets[stages] = {tile_start, tile_start + stride};
    int next_tile = tile_start;
    for (int s = 0; s < stages; ++s) {
        stage_ready[s] = enqueue_tile(s, stage_offsets[s]);
        if (stage_ready[s]) {
            next_tile += stride;
        }
    }

    int current_stage = 0;
    while (stage_ready[current_stage]) {
        const int current_tile = stage_offsets[current_stage];
        if (current_tile >= count) {
            break;
        }

        pipe.consumer_wait();
        block.sync();

        const int valid_elems = min(tile_span, count - current_tile);
        const int base_idx = threadIdx.x * values_per_thread;
        float* tile_ptr = stage_ptr(current_stage) + base_idx;

        float values[values_per_thread];
        #pragma unroll
        for (int i = 0; i < values_per_thread; ++i) {
            const int elem = base_idx + i;
            values[i] = elem < valid_elems ? tile_ptr[i] : 0.0f;
        }

        block.sync();
        pipe.consumer_release();

        #pragma unroll
        for (int i = 0; i < values_per_thread; ++i) {
            const int out_idx = current_tile + base_idx + i;
            if (out_idx < count) {
                output[out_idx] = transform_with_scale(values[i], threshold);
            }
        }

        stage_ready[current_stage] = false;

        const int refill_stage = current_stage;
        current_stage ^= 1;
        if (next_tile < count) {
            stage_offsets[refill_stage] = next_tile;
            stage_ready[refill_stage] = enqueue_tile(refill_stage, next_tile);
            if (stage_ready[refill_stage]) {
                next_tile += stride;
            }
        }

        if (!stage_ready[current_stage]) {
            break;
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

template <int ValuesPerThread>
inline cudaError_t launch_threshold_tma_pipeline_variant(
    const float* inputs,
    float* output,
    float threshold,
    int count,
    cudaStream_t stream) {
#if CUDART_VERSION >= 12000
    const dim3 block(kThresholdOptimizedThreads);
    constexpr int values_per_thread = ValuesPerThread;
    constexpr int stages = 2;
    const int tile_span = block.x * values_per_thread;
    int grid_x = (count + tile_span - 1) / tile_span;
    constexpr int kMaxPipelineBlocks = 2048;
    if (grid_x > kMaxPipelineBlocks) {
        grid_x = kMaxPipelineBlocks;
    }
    if (grid_x < 1) {
        grid_x = 1;
    }
    const dim3 grid(grid_x);
    const size_t shared_bytes = stages * tile_span * sizeof(float);
    threshold_tma_pipeline_kernel<ValuesPerThread><<<grid, block, shared_bytes, stream>>>(
        inputs,
        output,
        threshold,
        count);
    return cudaGetLastError();
#else
    (void)inputs;
    (void)output;
    (void)threshold;
    (void)count;
    (void)stream;
    return cudaErrorNotSupported;
#endif
}

inline cudaError_t launch_threshold_tma_pipeline(
    const float* inputs,
    float* output,
    float threshold,
    int count,
    cudaStream_t stream) {
#if CUDART_VERSION >= 12000
    constexpr int candidates[] = {4, 6, 8};
    static std::mutex cache_mutex;
    static int cached_device = -1;
    static long long cached_count = -1;
    static int cached_variant = 0;

    int device_id = 0;
    cudaGetDevice(&device_id);

    auto get_cached_variant = [&]() -> int {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (cached_variant != 0 && cached_device == device_id && cached_count == static_cast<long long>(count)) {
            return cached_variant;
        }
        return 0;
    };

    auto store_cached_variant = [&](int variant) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cached_variant = variant;
        cached_device = device_id;
        cached_count = static_cast<long long>(count);
    };

    int selected_variant = get_cached_variant();
    cudaError_t last_error = cudaErrorUnknown;

    if (selected_variant == 0) {
        float best_time_ms = std::numeric_limits<float>::max();
        int best_index = -1;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        auto measure_variant = [&](auto launcher, float* elapsed_ms) -> cudaError_t {
            cudaEventRecord(start, stream);
            cudaError_t err = launcher(inputs, output, threshold, count, stream);
            if (err != cudaSuccess) {
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                return err;
            }
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(elapsed_ms, start, stop);
            return cudaSuccess;
        };

        for (int idx = 0; idx < static_cast<int>(sizeof(candidates) / sizeof(int)); ++idx) {
            int vpt = candidates[idx];
            float elapsed = 0.0f;
            cudaError_t err = cudaSuccess;
            switch (vpt) {
                case 4: {
                    err = measure_variant(launch_threshold_tma_pipeline_variant<4>, &elapsed);
                    break;
                }
                case 6: {
                    err = measure_variant(launch_threshold_tma_pipeline_variant<6>, &elapsed);
                    break;
                }
                case 8: {
                    err = measure_variant(launch_threshold_tma_pipeline_variant<8>, &elapsed);
                    break;
                }
                default:
                    continue;
            }
            if (err != cudaSuccess) {
                last_error = err;
                continue;
            }
            if (elapsed < best_time_ms) {
                best_time_ms = elapsed;
                best_index = idx;
            }
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        if (best_index < 0) {
            return last_error;
        }

        selected_variant = candidates[best_index];
        store_cached_variant(selected_variant);
    }

    switch (selected_variant) {
        case 4:
            return launch_threshold_tma_pipeline_variant<4>(inputs, output, threshold, count, stream);
        case 6:
            return launch_threshold_tma_pipeline_variant<6>(inputs, output, threshold, count, stream);
        default:
            return launch_threshold_tma_pipeline_variant<8>(inputs, output, threshold, count, stream);
    }
#else
    (void)inputs;
    (void)output;
    (void)threshold;
    (void)count;
    (void)stream;
    return cudaErrorNotSupported;
#endif
}

}  // namespace ch08
