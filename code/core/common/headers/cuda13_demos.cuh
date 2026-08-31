#pragma once

/**
 * cuda13_demos.cuh - Standalone demos of CUDA 13.0 features
 * 
 * Self-contained demonstrations of stream-ordered memory allocation
 * and TMA bulk async operations.
 */

#include <cuda.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <type_traits>

#include "arch_detection.cuh"
#include "tma_helpers.cuh"

namespace cuda_device = cuda::device::experimental;

namespace cuda13_demos {

constexpr int MAX_TMA_TILE_N = 128;
constexpr int DEFAULT_TILE_M = 64;

// -----------------------------------------------------------------------------
// Stream-ordered memory allocation demo
// -----------------------------------------------------------------------------

static __global__ void fill_sequence(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = static_cast<float>(idx);
    }
}

static __global__ void saxpy_kernel(float* out, const float* in, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = alpha * in[idx] + out[idx];
    }
}

inline void run_stream_ordered_memory_demo() {
    std::printf("\n[CUDA 13] Stream-ordered memory allocation demo\n");
    cudaStream_t stream;
    cuda_tma::check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");

    // Configure the default mempool for aggressive reuse so allocations
    // happen entirely on the GPU timeline.
    cudaMemPool_t pool{};
    cuda_tma::check_cuda(cudaDeviceGetDefaultMemPool(&pool, /*device=*/0), "get default mempool");
    std::uint64_t threshold = 0;
    cuda_tma::check_cuda(cudaMemPoolSetAttribute(
                   pool,
                   cudaMemPoolAttrReleaseThreshold,
                   &threshold),
               "set mempool threshold");

    constexpr int N = 1 << 15;
    constexpr size_t BYTES = sizeof(float) * N;
    float* a = nullptr;
    float* b = nullptr;

    cuda_tma::check_cuda(cudaMallocAsync(&a, BYTES, stream), "cudaMallocAsync(a)");
    cuda_tma::check_cuda(cudaMallocAsync(&b, BYTES, stream), "cudaMallocAsync(b)");

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    fill_sequence<<<grid, block, 0, stream>>>(a, N);
    fill_sequence<<<grid, block, 0, stream>>>(b, N);
    saxpy_kernel<<<grid, block, 0, stream>>>(b, a, 2.0f, N);

    cuda_tma::check_cuda(cudaFreeAsync(a, stream), "cudaFreeAsync(a)");
    cuda_tma::check_cuda(cudaFreeAsync(b, stream), "cudaFreeAsync(b)");

    cuda_tma::check_cuda(cudaStreamSynchronize(stream), "stream sync");
    cuda_tma::check_cuda(cudaStreamDestroy(stream), "destroy stream");
    std::printf("  ✓ Allocations executed entirely on the GPU stream timeline\n");
}

// -----------------------------------------------------------------------------
// Simple 2D TMA demo
// -----------------------------------------------------------------------------

template <int TILE_M, int TILE_N>
static __global__ void tma_copy_kernel(
    const __grid_constant__ CUtensorMap in_desc,
    const __grid_constant__ CUtensorMap out_desc,
    float* out_fallback,
    int width,
    int height,
    int ld_out) {
    constexpr int participants = 128;
    if (blockDim.x * blockDim.y != participants) {
        return;
    }

    __shared__ alignas(128) float smem[TILE_M][TILE_N];
    using block_barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(block_barrier) unsigned char bar_storage[sizeof(block_barrier)];
    auto* bar = reinterpret_cast<block_barrier*>(bar_storage);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        init(bar, participants);
        cuda_device::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    int tile_x = blockIdx.x * TILE_N;
    int tile_y = blockIdx.y * TILE_M;

    bool in_bounds = (tile_x + TILE_N) <= width && (tile_y + TILE_M) <= height;

    block_barrier::arrival_token token;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
            &smem,
            &in_desc,
            tile_x,
            tile_y,
            *bar);
        token = cuda::device::barrier_arrive_tx(*bar, 1, sizeof(smem));
    } else {
        token = bar->arrive();
    }
    bar->wait(std::move(token));

    // Simple transform: scale by 1.5x
    for (int row = threadIdx.y; row < TILE_M; row += blockDim.y) {
        for (int col = threadIdx.x; col < TILE_N; col += blockDim.x) {
            smem[row][col] *= 1.5f;
        }
    }
    cuda_device::fence_proxy_async_shared_cta();
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        cuda_device::cp_async_bulk_tensor_2d_shared_to_global(
            &out_desc,
            tile_x,
            tile_y,
            &smem);
        cuda_device::cp_async_bulk_commit_group();
        cuda_device::cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();

    if (!in_bounds) {
        for (int row = threadIdx.y; row < TILE_M; row += blockDim.y) {
            int global_row = tile_y + row;
            if (global_row >= height) {
                continue;
            }
            for (int col = threadIdx.x; col < TILE_N; col += blockDim.x) {
                int global_col = tile_x + col;
                if (global_col >= width) {
                    continue;
                }
                out_fallback[global_row * ld_out + global_col] = smem[row][col];
            }
        }
    }
}

inline void run_simple_tma_demo() {
    std::printf("\n[CUDA 13] Tensor Memory Accelerator 2D copy demo\n");
    if (!cuda_tma::device_supports_tma()) {
        std::printf("  ⚠️  Device does not support TMA (requires SM 90 or newer)\n");
        return;
    }

    auto encode = cuda_tma::load_cuTensorMapEncodeTiled();
    if (!encode) {
        std::printf("  ⚠️  cuTensorMapEncodeTiled unavailable on this runtime\n");
        return;
    }

    cuda_arch::TMALimits limits = cuda_arch::get_tma_limits();
    int descriptor_width = std::min<int>(limits.max_2d_box_width, MAX_TMA_TILE_N);
    int descriptor_height = std::min<int>(limits.max_2d_box_height, DEFAULT_TILE_M);
    if (descriptor_width >= 128) {
        descriptor_width = 128;
    } else if (descriptor_width >= 64) {
        descriptor_width = 64;
    } else {
        descriptor_width = 32;
    }
    if (descriptor_height >= 64) {
        descriptor_height = 64;
    } else {
        descriptor_height = 32;
    }

    if (descriptor_width <= 0 || descriptor_height <= 0) {
        std::printf("  ⚠️  Unable to determine valid TMA box dimensions; skipping demo\n");
        return;
    }

    const int WIDTH = descriptor_width;
    const int HEIGHT = descriptor_height;
    const size_t BYTES = static_cast<size_t>(WIDTH) * HEIGHT * sizeof(float);

    std::printf("  → Using descriptor box %dx%d (width × height)\n", WIDTH, HEIGHT);

    std::vector<float> h_in(WIDTH * HEIGHT);
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            h_in[y * WIDTH + x] = static_cast<float>((y + 1) * (x + 1));
        }
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    cuda_tma::check_cuda(cudaMalloc(&d_in, BYTES), "cudaMalloc d_in");
    cuda_tma::check_cuda(cudaMalloc(&d_out, BYTES), "cudaMalloc d_out");
    cuda_tma::check_cuda(cudaMemcpy(d_in, h_in.data(), BYTES, cudaMemcpyHostToDevice), "copy input");
    cuda_tma::check_cuda(cudaMemset(d_out, 0, BYTES), "memset output");

    CUtensorMap in_desc{};
    CUtensorMap out_desc{};
    bool ok_in = cuda_tma::make_2d_tensor_map(in_desc, encode, d_in, WIDTH, HEIGHT, WIDTH, WIDTH, HEIGHT, CU_TENSOR_MAP_SWIZZLE_NONE);
    bool ok_out = cuda_tma::make_2d_tensor_map(out_desc, encode, d_out, WIDTH, HEIGHT, WIDTH, WIDTH, HEIGHT, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (!ok_in || !ok_out) {
        std::printf("  ⚠️  Failed to encode tensor maps; skipping TMA demo\n");
        cudaFree(d_in);
        cudaFree(d_out);
        return;
    }

    dim3 block(32, 4, 1);
    dim3 grid(1, 1, 1);

    auto launch_demo = [&](auto width_tag, auto height_tag) {
        constexpr int TILE_N = decltype(width_tag)::value;
        constexpr int TILE_M = decltype(height_tag)::value;
        tma_copy_kernel<TILE_M, TILE_N><<<grid, block>>>(in_desc, out_desc, d_out, WIDTH, HEIGHT, WIDTH);
    };

    if (HEIGHT == 64) {
        if (WIDTH == 128) {
            launch_demo(std::integral_constant<int, 128>{}, std::integral_constant<int, 64>{});
        } else if (WIDTH == 64) {
            launch_demo(std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{});
        } else {
            launch_demo(std::integral_constant<int, 32>{}, std::integral_constant<int, 64>{});
        }
    } else {
        if (WIDTH == 128) {
            launch_demo(std::integral_constant<int, 128>{}, std::integral_constant<int, 32>{});
        } else if (WIDTH == 64) {
            launch_demo(std::integral_constant<int, 64>{}, std::integral_constant<int, 32>{});
        } else {
            launch_demo(std::integral_constant<int, 32>{}, std::integral_constant<int, 32>{});
        }
    }

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err == cudaErrorNotSupported ||
        launch_err == cudaErrorInvalidDeviceFunction ||
        launch_err == cudaErrorInvalidValue) {
        std::printf("  ⚠️  TMA kernel unavailable on this GPU (%s); skipping demo\n",
                    cudaGetErrorString(launch_err));
        cudaFree(d_in);
        cudaFree(d_out);
        return;
    }
    cuda_tma::check_cuda(launch_err, "launch tma_copy_kernel");

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err == cudaErrorNotSupported ||
        sync_err == cudaErrorInvalidDeviceFunction ||
        sync_err == cudaErrorInvalidValue) {
        std::printf("  ⚠️  TMA execution not supported on this GPU (%s); skipping demo\n",
                    cudaGetErrorString(sync_err));
        cudaFree(d_in);
        cudaFree(d_out);
        return;
    }
    cuda_tma::check_cuda(sync_err, "sync tma_copy_kernel");

    std::vector<float> h_out(WIDTH * HEIGHT);
    cuda_tma::check_cuda(cudaMemcpy(h_out.data(), d_out, BYTES, cudaMemcpyDeviceToHost), "copy result");

    std::printf("  ✓ TMA copied %dx%d tile (swizzle=none, L2 promotion=none)\n", HEIGHT, WIDTH);
    std::printf("  Sample output element: %.2f -> %.2f\n", h_in[0], h_out[0]);

    cudaFree(d_in);
    cudaFree(d_out);
}

}  // namespace cuda13_demos
