// optimized_memory_access.cu -- 256-bit (32B) vectorized copy with register blocking.
// CUDA 13 + Blackwell: Uses Float8 (32-byte aligned) for 256-bit loads/stores.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "../core/common/headers/cuda_helpers.cuh"
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

// CUDA 13 + Blackwell: 32-byte aligned type for 256-bit loads
struct alignas(32) Float8 {
    float elems[8];
};
static_assert(sizeof(Float8) == 32, "Float8 must be 32 bytes");
static_assert(alignof(Float8) == 32, "Float8 must be 32-byte aligned");

constexpr int N = 1 << 24;
constexpr int REPEAT = 50;
constexpr int BLOCK_THREADS = 256;
constexpr int VECTORS_PER_THREAD = 4;

__device__ __forceinline__ Float8 load_float8(const Float8* ptr) {
  Float8 v;
#if __CUDA_ARCH__ >= 800
  // Use two 128-bit LDGs to fetch the 256-bit struct without custom intrinsics
  const float4* src4 = reinterpret_cast<const float4*>(ptr);
  reinterpret_cast<float4*>(&v)[0] = __ldg(src4);
  reinterpret_cast<float4*>(&v)[1] = __ldg(src4 + 1);
#else
  v = *ptr;
#endif
  return v;
}

// Coalesced copy with register blocking and Float8 (256-bit loads)
__global__ void coalesced_copy(const Float8* __restrict__ src,
                               Float8* __restrict__ dst,
                               int n_vec) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int base = tid; base < n_vec; base += stride * VECTORS_PER_THREAD) {
    Float8 lane_buffer[VECTORS_PER_THREAD];
    #pragma unroll
    for (int item = 0; item < VECTORS_PER_THREAD; ++item) {
      const int idx = base + item * stride;
      if (idx < n_vec) {
        lane_buffer[item] = load_float8(src + idx);  // 256-bit load
      }
    }
    #pragma unroll
    for (int item = 0; item < VECTORS_PER_THREAD; ++item) {
      const int idx = base + item * stride;
      if (idx < n_vec) {
        dst[idx] = lane_buffer[item];  // 256-bit store
      }
    }
  }
}

float checksum(const std::vector<float>& data) {
  double acc = 0.0;
  for (float v : data) {
      NVTX_RANGE("verify");
      acc += static_cast<double>(v);
  }
  return static_cast<float>(acc / static_cast<double>(data.size()));
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float max_err = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    max_err = fmaxf(max_err, fabsf(a[i] - b[i]));
  }
  return max_err;
}

int main() {
    NVTX_RANGE("main");
  static_assert(N % 8 == 0, "N must be divisible by 8 for Float8");
  const int n_vec = N / 8;

  std::vector<float> h_src(N), h_dst(N, 0.0f);
  for (int i = 0; i < N; ++i) {
      NVTX_RANGE("setup");
    h_src[i] = static_cast<float>((i % 2048) - 1024) / 256.0f;
  }

  float *d_src = nullptr, *d_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(BLOCK_THREADS);
  dim3 grid((n_vec + block.x * VECTORS_PER_THREAD - 1) / (block.x * VECTORS_PER_THREAD));

  // Warmup
  coalesced_copy<<<grid, block>>>(
      reinterpret_cast<const Float8*>(d_src),
      reinterpret_cast<Float8*>(d_dst),
      n_vec);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < REPEAT; ++iter) {
      NVTX_RANGE("compute_kernel:coalesced_copy");
    coalesced_copy<<<grid, block>>>(
        reinterpret_cast<const Float8*>(d_src),
        reinterpret_cast<Float8*>(d_dst),
        n_vec);
  }
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / static_cast<float>(REPEAT);
  std::printf("Coalesced copy (Float8, 256-bit): %.3f ms\n", avg_ms);
  std::printf("TIME_MS: %.6f\n", avg_ms);

  CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("Checksum: %.6f, Max diff: %.6e\n", checksum(h_dst), max_abs_diff(h_src, h_dst));
#ifdef VERIFY
  float verify_checksum = 0.0f;
  VERIFY_CHECKSUM(h_dst.data(), N, &verify_checksum);
  VERIFY_PRINT_CHECKSUM(verify_checksum);
#endif

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));
  return 0;
}
