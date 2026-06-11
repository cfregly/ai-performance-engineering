// PyTorch extension wrapper for warp_specialized_pipeline_enhanced.cu
// Based on Chapter 10's enhanced warp specialization pattern

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <algorithm>
#include "../core/common/nvtx_utils.cuh"

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
            throw std::runtime_error(cudaGetErrorString(status)); \
        } \
    } while(0)

constexpr int TILE = 64;
constexpr int TILE_ELEMS = TILE * TILE;
constexpr int PIPELINE_DEPTH = 2;

// Enhanced compute with more work to showcase pipeline benefits.
// Compute warps shard the tile to avoid overlapping writes.
__device__ void compute_tile_enhanced(const float* a,
                                      const float* b,
                                      float* c,
                                      int compute_warp_id,
                                      int lane,
                                      int num_compute_warps) {
    int thread_in_compute_group = compute_warp_id * warpSize + lane;
    int compute_group_threads = num_compute_warps * warpSize;
    for (int idx = thread_in_compute_group; idx < TILE_ELEMS; idx += compute_group_threads) {
        float x = a[idx];
        float y = b[idx];

        // More compute to show pipeline overlap benefit
        float result = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            result += sqrtf(x * x + y * y) * 0.125f;
        }
        c[idx] = result;
    }
}

// Enhanced warp-specialized kernel with:
// - Double-buffer pipeline (2 stages)
// - 1 producer warp, 6 compute warps, 1 consumer warp
// - Block-uniform pipeline collectives for correctness
__global__ void warp_specialized_enhanced_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int total_tiles
) {
    cg::thread_block block = cg::this_thread_block();
    
    // Double buffering for the pipeline
    extern __shared__ float smem[];
    float* A_tiles = smem;
    float* B_tiles = smem + PIPELINE_DEPTH * TILE_ELEMS;
    float* C_tiles = smem + 2 * PIPELINE_DEPTH * TILE_ELEMS;

    using pipe_state = cuda::pipeline_shared_state<cuda::thread_scope_block, PIPELINE_DEPTH>;
    __shared__ alignas(pipe_state) unsigned char state_storage[sizeof(pipe_state)];
    auto* state = reinterpret_cast<pipe_state*>(state_storage);
    if (threadIdx.x == 0) {
        new (state) pipe_state();
    }
    block.sync();
    auto pipe = cuda::make_pipeline(block, state);

    int warp_id = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;

    // 8 warps total: 1 producer, 6 compute, 1 consumer
    constexpr int PRODUCER_WARP = 0;
    constexpr int CONSUMER_WARP = 7;
    constexpr int COMPUTE_WARPS = 6;

    // Warp roles
    bool is_producer = (warp_id == PRODUCER_WARP);
    bool is_consumer = (warp_id == CONSUMER_WARP);
    bool is_compute = (warp_id >= 1 && warp_id <= COMPUTE_WARPS);

    // Prime the pipeline.
    for (int stage = 0; stage < PIPELINE_DEPTH; ++stage) {
        int tile = blockIdx.x + stage * gridDim.x;
        if (tile >= total_tiles) {
            break;
        }
        size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;
        int buf_idx = stage;

        // Block-scoped pipeline collectives must be executed uniformly.
        pipe.producer_acquire();
        if (is_producer) {
            for (int idx = lane; idx < TILE_ELEMS / 4; idx += warpSize) {
                float4 a4 = *reinterpret_cast<const float4*>(&A[offset + idx * 4]);
                float4 b4 = *reinterpret_cast<const float4*>(&B[offset + idx * 4]);
                *reinterpret_cast<float4*>(&A_tiles[buf_idx * TILE_ELEMS + idx * 4]) = a4;
                *reinterpret_cast<float4*>(&B_tiles[buf_idx * TILE_ELEMS + idx * 4]) = b4;
            }
        }
        pipe.producer_commit();
    }

    block.sync();

    // Process tiles with pipeline reuse.
    int tile_iter = 0;
    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x, ++tile_iter) {
        int buf_idx = tile_iter % PIPELINE_DEPTH;
        size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;

        pipe.consumer_wait();
        block.sync();

        if (is_compute) {
            compute_tile_enhanced(
                &A_tiles[buf_idx * TILE_ELEMS],
                &B_tiles[buf_idx * TILE_ELEMS],
                &C_tiles[buf_idx * TILE_ELEMS],
                warp_id - 1,
                lane,
                COMPUTE_WARPS
            );
        }
        block.sync();

        if (is_consumer) {
            // Vectorized stores
            for (int idx = lane; idx < TILE_ELEMS / 4; idx += warpSize) {
                float4 c4 = *reinterpret_cast<const float4*>(&C_tiles[buf_idx * TILE_ELEMS + idx * 4]);
                *reinterpret_cast<float4*>(&C[offset + idx * 4]) = c4;
            }
        }
        block.sync();

        pipe.consumer_release();
        block.sync();

        int next_tile = tile + PIPELINE_DEPTH * gridDim.x;
        if (next_tile < total_tiles) {
            int next_stage = (tile_iter + PIPELINE_DEPTH) % PIPELINE_DEPTH;
            size_t next_offset = static_cast<size_t>(next_tile) * TILE_ELEMS;
            pipe.producer_acquire();
            if (is_producer) {
                for (int idx = lane; idx < TILE_ELEMS / 4; idx += warpSize) {
                    float4 a4 = *reinterpret_cast<const float4*>(&A[next_offset + idx * 4]);
                    float4 b4 = *reinterpret_cast<const float4*>(&B[next_offset + idx * 4]);
                    *reinterpret_cast<float4*>(&A_tiles[next_stage * TILE_ELEMS + idx * 4]) = a4;
                    *reinterpret_cast<float4*>(&B_tiles[next_stage * TILE_ELEMS + idx * 4]) = b4;
                }
            }
            pipe.producer_commit();
        }
        block.sync();
    }
}

torch::Tensor warp_specialized_pipeline_enhanced_forward(
    const torch::Tensor& A,
    const torch::Tensor& B
) {
    TORCH_CHECK(A.is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(A.sizes() == B.sizes(), "Inputs must have same shape");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Inputs must be float32");

    int n_elements = A.numel();
    TORCH_CHECK(n_elements > 0, "Inputs must be non-empty");
    TORCH_CHECK(
        n_elements % TILE_ELEMS == 0,
        "Input element count must be a multiple of ",
        TILE_ELEMS,
        " for warp_specialized_pipeline_enhanced_forward");
    int total_tiles = (n_elements + TILE_ELEMS - 1) / TILE_ELEMS;
    
    auto C = torch::empty_like(A);
    
    // Launch configuration: 8 warps per block (1 producer + 6 compute + 1 consumer)
    dim3 block(8 * 32);  // 8 warps * 32 threads
    dim3 grid(std::min(total_tiles, 256));
    size_t shared_bytes = (3 * PIPELINE_DEPTH * TILE_ELEMS) * sizeof(float);
    
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int max_dynamic_smem = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &max_dynamic_smem,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device));
    TORCH_CHECK(shared_bytes <= static_cast<size_t>(max_dynamic_smem),
                "Requested shared memory (", shared_bytes,
                " bytes) exceeds device limit (", max_dynamic_smem, " bytes)");
    CUDA_CHECK(cudaFuncSetAttribute(
        warp_specialized_enhanced_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    
    warp_specialized_enhanced_kernel<<<grid, block, shared_bytes, c10::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        total_tiles
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("warp_specialized_pipeline_enhanced_forward", &warp_specialized_pipeline_enhanced_forward,
          "Enhanced warp-specialized pipeline forward (Chapter 10)");
}
