// async_prefetch_2d_demo.cu -- Minimal 2D TMA copy to drive bandwidth.
// Uses cp.async.bulk.tensor.2d with a single stage to keep the kernel memory-bound.

#include <algorithm>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#include "../core/common/headers/tma_helpers.cuh"

#if CUDART_VERSION >= 13000
#include <cuda.h>
#include "../core/common/nvtx_utils.cuh"
#define TMA_CUDA13_AVAILABLE 1
#else
#define TMA_CUDA13_AVAILABLE 0
#endif

namespace cde = cuda::device::experimental;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#if TMA_CUDA13_AVAILABLE

template <int TILE_M, int TILE_N>
__global__ void tma_copy_2d_kernel(const __grid_constant__ CUtensorMap in_desc,
                                   const __grid_constant__ CUtensorMap out_desc,
                                   int M,
                                   int N) {
    constexpr std::size_t BYTES_PER_TILE =
        static_cast<std::size_t>(TILE_M) * TILE_N * sizeof(float);
    __shared__ alignas(128) float tile[TILE_M][TILE_N];
    using block_barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(block_barrier) unsigned char barrier_storage[sizeof(block_barrier)];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        init(reinterpret_cast<block_barrier*>(barrier_storage), blockDim.x * blockDim.y);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    auto* bar_ptr = reinterpret_cast<block_barrier*>(barrier_storage);
    auto& bar = *bar_ptr;

    const int tile_m = blockIdx.y * TILE_M;
    const int tile_n = blockIdx.x * TILE_N;
    if (tile_m >= M || tile_n >= N) {
        return;
    }

    const int rows = min(TILE_M, M - tile_m);
    const int cols = min(TILE_N, N - tile_n);

    cuda::barrier<cuda::thread_scope_block>::arrival_token token;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &tile,
            &in_desc,
            tile_n,  // contiguous column offset (dimension 0)
            tile_m,  // row offset (dimension 1)
            bar);
        token = cuda::device::barrier_arrive_tx(bar, 1, BYTES_PER_TILE);
    }
    if (!(threadIdx.x == 0 && threadIdx.y == 0)) {
        token = bar.arrive();
    }
    bar.wait(std::move(token));
    __syncthreads();

    // No compute: pure copy.
    cde::fence_proxy_async_shared_cta();
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(
            &out_desc,
            tile_n,  // contiguous column offset (dimension 0)
            tile_m,  // row offset (dimension 1)
            &tile);
        cde::cp_async_bulk_commit_group();
        cde::cp_async_bulk_wait_group_read<0>();
    }
}
#endif

int main() {
    NVTX_RANGE("main");
#if !TMA_CUDA13_AVAILABLE
    std::printf("SKIPPED: Requires CUDA 13.0+.\n");
    return 3;
#else
    if (!cuda_tma::device_supports_tma()) {
        std::printf("SKIPPED: TMA hardware/runtime support not detected.\n");
        return 3;
    }

    constexpr int TILE_M = 128;
    constexpr int TILE_N = 64;  // 128x64 tile stays well under 48KB shared
    constexpr int M = 4096;
    constexpr int N = 4096;

    const auto limits = cuda_arch::get_tma_limits();
    if (TILE_N > static_cast<int>(limits.max_2d_box_width) ||
        TILE_M > static_cast<int>(limits.max_2d_box_height)) {
        std::printf("SKIPPED: TILE exceeds TMA 2D box limits (w=%u h=%u)\n",
                    limits.max_2d_box_width,
                    limits.max_2d_box_height);
        return 3;
    }

    const std::size_t bytes = static_cast<std::size_t>(M) * N * sizeof(float);
    std::vector<float> h_in(static_cast<std::size_t>(M) * N);
    std::vector<float> h_out(static_cast<std::size_t>(M) * N, 0.0f);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            const std::size_t idx = static_cast<std::size_t>(row) * N + col;
            h_in[idx] = static_cast<float>((row * 131 + col * 17) % 8191) / 8191.0f;
        }
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    CUtensorMap in_desc{};
    CUtensorMap out_desc{};
    auto encode = cuda_tma::load_cuTensorMapEncodeTiled();
    if (!encode ||
        !cuda_tma::make_2d_tensor_map(in_desc, encode, d_in, N, M, N, TILE_N, TILE_M, CU_TENSOR_MAP_SWIZZLE_NONE) ||
        !cuda_tma::make_2d_tensor_map(out_desc, encode, d_out, N, M, N, TILE_N, TILE_M, CU_TENSOR_MAP_SWIZZLE_NONE)) {
        std::printf("SKIPPED: cuTensorMapEncodeTiled unavailable\n");
        cudaFree(d_in);
        cudaFree(d_out);
        return 3;
    }

    dim3 block(16, 8, 1);  // 128 threads
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, 1);

    tma_copy_2d_kernel<TILE_M, TILE_N><<<grid, block>>>(in_desc, out_desc, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr int kIters = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kIters; ++i) {
        NVTX_RANGE("compute_kernel");
        tma_copy_2d_kernel<TILE_M, TILE_N><<<grid, block>>>(in_desc, out_desc, M, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const float avg_ms = elapsed_ms / static_cast<float>(kIters);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (std::size_t idx = 0; idx < h_out.size(); ++idx) {
        NVTX_RANGE("iteration");
        if (h_out[idx] != h_in[idx]) {
            std::printf("Mismatch at %zu: got %f expected %f\n", idx, h_out[idx], h_in[idx]);
            ok = false;
            break;
        }
    }

    std::printf("2D TMA copy: %.3f ms (avg over %d iters) [%s]\n",
                avg_ms,
                kIters,
                ok ? "OK" : "FAIL");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    return ok ? 0 : 1;
#endif
}
