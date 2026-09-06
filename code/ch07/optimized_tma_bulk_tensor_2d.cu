// optimized_tma_bulk_tensor_2d.cu
// TMA-backed 2D global->shared->global copy using cp.async.bulk.tensor.2d
// with mbarrier completion for loads and bulk-group completion for stores.
//
// This uses CUDA 13 device wrappers (cuda::device::experimental) and targets
// Blackwell/Grace-Blackwell (sm_100+/sm_103+/sm_121). When TMA is unavailable,
// it falls back to a manual copy so the sample still runs.

#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <utility>
#include <vector>

#include "../core/common/headers/tma_helpers.cuh"
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

#if CUDART_VERSION < 13000
int main() {
    NVTX_RANGE("main");
    std::printf("SKIP: optimized_tma_bulk_tensor_2d requires CUDA 13.0+ for cp.async.bulk.tensor\n");
    return 0;
}
#else

namespace cde = cuda::device::experimental;
using block_barrier = cuda::barrier<cuda::thread_scope_block>;
using cuda_tma::check_cuda;
using cuda_tma::load_cuTensorMapEncodeTiled;
using cuda_tma::make_2d_tensor_map;

namespace {

constexpr int TILE_M = 128;
constexpr int TILE_N = 64;   // Keep shared tile under 48 KB
constexpr int TMA_THREADS = 32;  // Single warp for the TMA path to minimize barrier overhead
constexpr int ITERATIONS = 10;
constexpr std::size_t TILE_BYTES =
    static_cast<std::size_t>(TILE_M) * TILE_N * sizeof(float);

static_assert((TILE_BYTES % 16) == 0,
              "TMA 2D copies require sizeBytes to be a multiple of 16 bytes");
static_assert(((TILE_N * sizeof(float)) % 16) == 0,
              "TMA 2D copies require a 16-byte aligned leading dimension");

template <int TILE_M_VALUE, int TILE_N_VALUE>
__global__ void tma_bulk_copy_kernel(const __grid_constant__ CUtensorMap in_desc,
                                     const __grid_constant__ CUtensorMap out_desc,
                                     float* output,
                                     int width,
                                     int height,
                                     int output_ld) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const int tile_row = blockIdx.y * TILE_M_VALUE;
    const int tile_col = blockIdx.x * TILE_N_VALUE;
    if (tile_row >= height || tile_col >= width) {
        return;
    }

    // 128-byte alignment is required for ≥2D TMA destinations in shared memory.
    __shared__ alignas(128) float tile[TILE_M_VALUE][TILE_N_VALUE];
    __shared__ alignas(block_barrier) unsigned char barrier_storage[sizeof(block_barrier)];
    auto* bar = reinterpret_cast<block_barrier*>(barrier_storage);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        init(bar, blockDim.x * blockDim.y);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    constexpr std::size_t kTileBytes =
        static_cast<std::size_t>(TILE_M_VALUE) * TILE_N_VALUE * sizeof(float);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            tile, &in_desc, tile_col, tile_row, *bar);
        cde::cp_async_bulk_commit_group();
    }
    cuda::barrier<cuda::thread_scope_block>::arrival_token token;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        token = cuda::device::barrier_arrive_tx(*bar, 1, kTileBytes);
    } else {
        token = bar->arrive();
    }
    bar->wait(std::move(token));
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        cde::cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();

    const int rows = min(TILE_M_VALUE, height - tile_row);
    const int cols = min(TILE_N_VALUE, width - tile_col);
    if (rows == TILE_M_VALUE && cols == TILE_N_VALUE) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            cde::cp_async_bulk_tensor_2d_shared_to_global(
                &out_desc, tile_col, tile_row, tile);
            cde::cp_async_bulk_commit_group();
            cde::cp_async_bulk_wait_group_read<0>();
        }
    } else {
        cuda_tma::store_partial_2d_tile(
            tile, output, output_ld, tile_row, tile_col, rows, cols,
            threadIdx.y * blockDim.x + threadIdx.x, blockDim.x * blockDim.y);
    }
#else
    (void)in_desc;
    (void)out_desc;
    (void)output;
    (void)output_ld;
    (void)width;
    (void)height;
#endif
}

float checksum(const std::vector<float>& data) {
    double sum = 0.0;
    for (float v : data) {
        NVTX_RANGE("verify");
        sum += static_cast<double>(v);
    }
    return static_cast<float>(sum / static_cast<double>(data.size()));
}

}  // namespace

int main() {
    const int width = 2048;
    const int height = 2048;
    const int ld = width;
    const std::size_t bytes = static_cast<std::size_t>(width) * height * sizeof(float);
    const std::size_t ld_bytes = static_cast<std::size_t>(ld) * sizeof(float);
    const bool stride_aligned_16 = (ld_bytes % 16) == 0;
    const bool tile_bytes_aligned_16 = (TILE_BYTES % 16) == 0;

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    const int sm_version = prop.major * 10 + prop.minor;
    const bool arch_ok = sm_version >= 90;

    bool tma_capable = arch_ok && stride_aligned_16 && tile_bytes_aligned_16 &&
                       cuda_tma::device_supports_tma();
    PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
    CUtensorMap in_desc{};
    CUtensorMap out_desc{};

    std::vector<float> h_src(width * height);
    for (int i = 0; i < width * height; ++i) {
        NVTX_RANGE("setup");
        h_src[i] = static_cast<float>((i % 127) - 63) * 0.01f;
    }

    float* d_src = nullptr;
    float* d_dst = nullptr;
    check_cuda(cudaMalloc(&d_src, bytes), "cudaMalloc d_src");
    check_cuda(cudaMalloc(&d_dst, bytes), "cudaMalloc d_dst");
    check_cuda(cudaMemcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice), "copy input");
    check_cuda(cudaMemset(d_dst, 0, bytes), "zero output");

    if (tma_capable) {
        encode = load_cuTensorMapEncodeTiled();
        tma_capable = encode != nullptr &&
                      make_2d_tensor_map(
                          in_desc,
                          encode,
                          d_src,
                          width,
                          height,
                          ld,
                          TILE_N,
                          TILE_M,
                          CU_TENSOR_MAP_SWIZZLE_NONE) &&
                      make_2d_tensor_map(
                          out_desc,
                          encode,
                          d_dst,
                          width,
                          height,
                          ld,
                          TILE_N,
                          TILE_M,
                          CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    dim3 block_tma(TMA_THREADS, 1, 1);
    dim3 grid((width + TILE_N - 1) / TILE_N, (height + TILE_M - 1) / TILE_M, 1);

    if (!tma_capable) {
    std::printf(
        "❌  TMA unavailable (sm=%d, stride16=%s, size16=%s, tensor map encode=%s).\n",
        sm_version,
        stride_aligned_16 ? "ok" : "no",
        tile_bytes_aligned_16 ? "ok" : "no",
        encode ? "ok" : "missing");
        return 1;
    }

    // Warmup
    tma_bulk_copy_kernel<TILE_M, TILE_N><<<grid, block_tma>>>(in_desc, out_desc, d_dst, width, height, ld);
    check_cuda(cudaGetLastError(), "warmup launch");
    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreate(&start), "event create start");
    check_cuda(cudaEventCreate(&stop), "event create stop");

    check_cuda(cudaEventRecord(start), "event record start");
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        NVTX_RANGE("compute_kernel");
        tma_bulk_copy_kernel<TILE_M, TILE_N><<<grid, block_tma>>>(
            in_desc, out_desc, d_dst, width, height, ld);
        check_cuda(cudaGetLastError(), "iteration launch");
    }
    check_cuda(cudaEventRecord(stop), "event record stop");
    check_cuda(cudaEventSynchronize(stop), "event sync stop");

    float total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "elapsed time");
    const float avg_ms = total_ms / static_cast<float>(ITERATIONS);

    const char* path_label = tma_capable ? "TMA" : "fallback-manual";
    std::printf("TMA 2D bulk tensor copy: %.3f ms (path: %s)\n", avg_ms, path_label);

    // Compare the full nonuniform tensor so transposes and partial copies fail.
    std::vector<float> h_dst(width * height);
    check_cuda(cudaMemcpy(h_dst.data(), d_dst, bytes, cudaMemcpyDeviceToHost), "copy output");
    bool copy_ok = true;
    for (std::size_t idx = 0; idx < h_dst.size(); ++idx) {
        if (h_dst[idx] != h_src[idx]) {
            std::fprintf(stderr,
                         "Mismatch at %zu: got %.9g expected %.9g\n",
                         idx,
                         static_cast<double>(h_dst[idx]),
                         static_cast<double>(h_src[idx]));
            copy_ok = false;
            break;
        }
    }
    std::printf("Output checksum: %.6f\n", checksum(h_dst));
#ifdef VERIFY
    float verify_checksum = 0.0f;
    VERIFY_CHECKSUM(h_dst.data(), static_cast<int>(h_dst.size()), &verify_checksum);
    VERIFY_PRINT_CHECKSUM(verify_checksum);
#endif

    check_cuda(cudaEventDestroy(start), "destroy start");
    check_cuda(cudaEventDestroy(stop), "destroy stop");
    check_cuda(cudaFree(d_src), "free d_src");
    check_cuda(cudaFree(d_dst), "free d_dst");
    return copy_ok ? 0 : 2;
}

#endif  // CUDART_VERSION >= 13000
