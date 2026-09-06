// optimized_warp_spec_pingpong.cu - Warp-role ping-pong pipeline (Ch10)
//
// WHAT:
//   Three dedicated warps cooperate on a two-stage ring buffer:
//   - Warp 0 is the producer and keeps the next tile prefetched
//   - Warp 1 and Warp 2 alternate between compute and store duties
//
// WHY THIS IS FASTER:
//   The two consumer warps "ping-pong" their roles across iterations. While one
//   consumer warp computes tile N, the other stores tile N-1, and the producer
//   warp refills the freed stage for tile N+1. That overlap is the actual
//   chapter pattern described in the book.
//
// COMPARE WITH:
//   baseline_warp_spec_pingpong.cu
//   - Same math and the same three-warp footprint
//   - Only one shared-memory stage, so load/compute/store serialize

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

namespace cg = cooperative_groups;

namespace {
constexpr int TILE_SIZE = 64;
constexpr int TILE_ELEMS = TILE_SIZE * TILE_SIZE;
constexpr int PIPELINE_STAGES = 2;
constexpr int WARPS_PER_BLOCK = 3;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int TOTAL_TILES = 4096;
constexpr int ITERATIONS = 10;

constexpr int PRODUCER_WARP = 0;
constexpr int CONSUMER_A_WARP = 1;
constexpr int CONSUMER_B_WARP = 2;

enum StageState : int {
    STAGE_EMPTY = 0,
    STAGE_LOADED = 1,
    STAGE_COMPUTED = 2,
};

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t status_ = (call);                                          \
        if (status_ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(status_));                         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__device__ void compute_tile(const float* __restrict__ a_tile,
                             const float* __restrict__ b_tile,
                             float* __restrict__ c_tile,
                             int lane_id) {
    for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
        float x = a_tile[idx];
        float y = b_tile[idx];
        c_tile[idx] = sqrtf(x * x + y * y);
    }
}

__device__ __forceinline__ void load_tile(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ stage_a,
                                          float* __restrict__ stage_b,
                                          size_t offset,
                                          int lane_id) {
    for (int idx = lane_id; idx < TILE_ELEMS / 4; idx += warpSize) {
        float4 a4 = *reinterpret_cast<const float4*>(&A[offset + idx * 4]);
        float4 b4 = *reinterpret_cast<const float4*>(&B[offset + idx * 4]);
        *reinterpret_cast<float4*>(&stage_a[idx * 4]) = a4;
        *reinterpret_cast<float4*>(&stage_b[idx * 4]) = b4;
    }
}

__device__ __forceinline__ void store_tile(const float* __restrict__ stage_c,
                                           float* __restrict__ C,
                                           size_t offset,
                                           int lane_id) {
    for (int idx = lane_id; idx < TILE_ELEMS / 4; idx += warpSize) {
        float4 c4 = *reinterpret_cast<const float4*>(&stage_c[idx * 4]);
        *reinterpret_cast<float4*>(&C[offset + idx * 4]) = c4;
    }
}

__device__ __forceinline__ void wait_for_stage(volatile int* stage_state,
                                               volatile int* stage_tile,
                                               int stage,
                                               int tile,
                                               int target_state) {
    while (stage_state[stage] != target_state || stage_tile[stage] != tile) {
        __nanosleep(64);
    }
}

__device__ __forceinline__ void publish_stage(volatile int* stage_state,
                                              volatile int* stage_tile,
                                              int stage,
                                              int tile,
                                              int state,
                                              int lane_id) {
    __syncwarp();
    if (lane_id == 0) {
        stage_tile[stage] = tile;
        __threadfence_block();
        stage_state[stage] = state;
    }
    __syncwarp();
}

__global__ void warp_specialized_pingpong(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int total_tiles) {
    extern __shared__ float shared_mem[];
    float* a_stages = shared_mem;
    float* b_stages = a_stages + PIPELINE_STAGES * TILE_ELEMS;
    float* c_stages = b_stages + PIPELINE_STAGES * TILE_ELEMS;
    __shared__ volatile int stage_state[PIPELINE_STAGES];
    __shared__ volatile int stage_tile[PIPELINE_STAGES];

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    const bool is_producer = warp_id == PRODUCER_WARP;
    const bool is_consumer_a = warp_id == CONSUMER_A_WARP;
    const bool is_consumer_b = warp_id == CONSUMER_B_WARP;

    if (threadIdx.x < PIPELINE_STAGES) {
        stage_state[threadIdx.x] = STAGE_EMPTY;
        stage_tile[threadIdx.x] = -1;
    }
    cg::this_thread_block().sync();

    if (is_producer) {
        for (int preload = 0; preload < PIPELINE_STAGES; ++preload) {
            const int tile = blockIdx.x + preload * gridDim.x;
            if (tile >= total_tiles) {
                break;
            }
            const size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;
            float* stage_a = a_stages + preload * TILE_ELEMS;
            float* stage_b = b_stages + preload * TILE_ELEMS;
            load_tile(A, B, stage_a, stage_b, offset, lane_id);
            publish_stage(stage_state, stage_tile, preload, tile, STAGE_LOADED, lane_id);
        }
    }

    int tiles_for_block = 0;
    if (blockIdx.x < total_tiles) {
        tiles_for_block = 1 + (total_tiles - 1 - blockIdx.x) / gridDim.x;
    }

    for (int tile_iter = 0; tile_iter < tiles_for_block; ++tile_iter) {
        const int tile = blockIdx.x + tile_iter * gridDim.x;
        const int stage = tile_iter % PIPELINE_STAGES;
        const int prev_stage = stage ^ 1;
        float* stage_a = a_stages + stage * TILE_ELEMS;
        float* stage_b = b_stages + stage * TILE_ELEMS;
        float* stage_c = c_stages + stage * TILE_ELEMS;

        const bool consumer_a_computes = (tile_iter % PIPELINE_STAGES) == 0;
        const bool is_compute_warp =
            consumer_a_computes ? is_consumer_a : is_consumer_b;
        const bool is_store_warp =
            tile_iter > 0 && (consumer_a_computes ? is_consumer_b : is_consumer_a);

        if (is_compute_warp) {
            wait_for_stage(stage_state, stage_tile, stage, tile, STAGE_LOADED);
            compute_tile(stage_a, stage_b, stage_c, lane_id);
            publish_stage(stage_state, stage_tile, stage, tile, STAGE_COMPUTED, lane_id);
        }

        if (is_store_warp) {
            const int prev_tile = blockIdx.x + (tile_iter - 1) * gridDim.x;
            const size_t prev_offset = static_cast<size_t>(prev_tile) * TILE_ELEMS;
            float* prev_c = c_stages + prev_stage * TILE_ELEMS;
            wait_for_stage(stage_state, stage_tile, prev_stage, prev_tile, STAGE_COMPUTED);
            store_tile(prev_c, C, prev_offset, lane_id);
            publish_stage(stage_state, stage_tile, prev_stage, -1, STAGE_EMPTY, lane_id);
        }

        if (is_producer) {
            const int next_tile = blockIdx.x + (tile_iter + 1) * gridDim.x;
            if (tile_iter > 0 && next_tile < total_tiles) {
                wait_for_stage(stage_state, stage_tile, prev_stage, -1, STAGE_EMPTY);
                const size_t next_offset = static_cast<size_t>(next_tile) * TILE_ELEMS;
                float* next_a = a_stages + prev_stage * TILE_ELEMS;
                float* next_b = b_stages + prev_stage * TILE_ELEMS;
                load_tile(A, B, next_a, next_b, next_offset, lane_id);
                publish_stage(stage_state, stage_tile, prev_stage, next_tile, STAGE_LOADED, lane_id);
            }
        }
    }

    if (tiles_for_block > 0) {
        const int final_iter = tiles_for_block - 1;
        const int final_tile = blockIdx.x + final_iter * gridDim.x;
        const int final_stage = final_iter % PIPELINE_STAGES;
        const size_t final_offset = static_cast<size_t>(final_tile) * TILE_ELEMS;
        float* final_c = c_stages + final_stage * TILE_ELEMS;
        const bool consumer_a_finishes = (final_iter % PIPELINE_STAGES) == 0;
        const bool is_final_store_warp =
            consumer_a_finishes ? is_consumer_a : is_consumer_b;
        if (is_final_store_warp) {
            wait_for_stage(stage_state, stage_tile, final_stage, final_tile, STAGE_COMPUTED);
            store_tile(final_c, C, final_offset, lane_id);
            publish_stage(stage_state, stage_tile, final_stage, -1, STAGE_EMPTY, lane_id);
        }
    }
}

double checksum(const std::vector<float>& data) {
    double acc = 0.0;
    for (float v : data) {
        NVTX_RANGE("verify");
        acc += static_cast<double>(v);
    }
    return acc / static_cast<double>(data.size());
}
}  // namespace

int main() {
    NVTX_RANGE("main");

    const size_t elems = static_cast<size_t>(TOTAL_TILES) * TILE_ELEMS;
    const size_t bytes = elems * sizeof(float);

    std::vector<float> h_A(elems), h_B(elems), h_C(elems);
    for (size_t i = 0; i < elems; ++i) {
        NVTX_RANGE("setup");
        h_A[i] = static_cast<float>(i % 251) * 0.01f;
        h_B[i] = static_cast<float>((i + 17) % 127) * 0.02f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const dim3 block(THREADS_PER_BLOCK);
    const dim3 grid(std::min(TOTAL_TILES, std::max(1, prop.multiProcessorCount * 2)));
    const size_t shared_bytes = 3 * PIPELINE_STAGES * TILE_ELEMS * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(
        warp_specialized_pingpong,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));

    warp_specialized_pingpong<<<grid, block, shared_bytes>>>(d_A, d_B, d_C, TOTAL_TILES);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        NVTX_RANGE("compute_kernel:warp_specialized_pingpong");
        warp_specialized_pingpong<<<grid, block, shared_bytes>>>(d_A, d_B, d_C, TOTAL_TILES);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(ITERATIONS);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    const double chk = checksum(h_C);

    std::printf("Optimized warp-role ping-pong pipeline: %.3f ms\n", avg_ms);
    std::printf("TIME_MS: %.6f\n", avg_ms);
    std::printf("Checksum: %.6f\n", chk);
    VERIFY_PRINT_CHECKSUM(static_cast<float>(chk));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
