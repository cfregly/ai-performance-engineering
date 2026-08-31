// basic_streams.cu -- CUDA 13.0 stream overlap demo with error handling.

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int WORK_ITERS = 4;
constexpr float BIAS_ADD = 0.001f;
constexpr float BIAS_SUB = 0.0002f;
constexpr float DECAY = 0.9f;

// Optimized kernel with vectorized loads and async copy support
// Launch bounds removed to let compiler auto-tune for different architectures (B200 vs GB10)
__global__ void scale_kernel(float* data, int n, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = data[idx];
#pragma unroll 4
    for (int iter = 0; iter < WORK_ITERS; ++iter) {
      val = val * scale + BIAS_ADD;
      val = val * DECAY - BIAS_SUB;
    }
    data[idx] = val;
  }
}

// CUDA 13 + Blackwell: 32-byte aligned type for 256-bit loads
struct alignas(32) Float8 {
    float elems[8];
};
static_assert(sizeof(Float8) == 32, "Float8 must be 32 bytes");
static_assert(alignof(Float8) == 32, "Float8 must be 32-byte aligned");

// Blackwell-optimized version using Float8 for 256-bit loads
// Launch bounds removed to let compiler auto-tune for different architectures (B200 vs GB10)
__global__ void scale_kernel_vectorized_float8(float* data, int n, float scale) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  
  if (idx + 7 < n) {
    // Load 8 floats at once (256-bit transaction on Blackwell)
    Float8 vec = *reinterpret_cast<Float8*>(&data[idx]);
    
    // Process all 8 elements
#pragma unroll 4
    for (int iter = 0; iter < WORK_ITERS; ++iter) {
      vec.elems[0] = vec.elems[0] * scale + BIAS_ADD;
      vec.elems[1] = vec.elems[1] * scale + BIAS_ADD;
      vec.elems[2] = vec.elems[2] * scale + BIAS_ADD;
      vec.elems[3] = vec.elems[3] * scale + BIAS_ADD;
      vec.elems[4] = vec.elems[4] * scale + BIAS_ADD;
      vec.elems[5] = vec.elems[5] * scale + BIAS_ADD;
      vec.elems[6] = vec.elems[6] * scale + BIAS_ADD;
      vec.elems[7] = vec.elems[7] * scale + BIAS_ADD;
      vec.elems[0] = vec.elems[0] * DECAY - BIAS_SUB;
      vec.elems[1] = vec.elems[1] * DECAY - BIAS_SUB;
      vec.elems[2] = vec.elems[2] * DECAY - BIAS_SUB;
      vec.elems[3] = vec.elems[3] * DECAY - BIAS_SUB;
      vec.elems[4] = vec.elems[4] * DECAY - BIAS_SUB;
      vec.elems[5] = vec.elems[5] * DECAY - BIAS_SUB;
      vec.elems[6] = vec.elems[6] * DECAY - BIAS_SUB;
      vec.elems[7] = vec.elems[7] * DECAY - BIAS_SUB;
    }
    
    // Store 8 floats at once (256-bit store on Blackwell)
    *reinterpret_cast<Float8*>(&data[idx]) = vec;
  } else {
    // Handle remaining elements
    for (int i = idx; i < n; i++) {
      float val = data[i];
#pragma unroll 4
      for (int iter = 0; iter < WORK_ITERS; ++iter) {
        val = val * scale + BIAS_ADD;
        val = val * DECAY - BIAS_SUB;
      }
      data[i] = val;
    }
  }
}

// Vectorized version using float4 for better memory throughput (pre-Blackwell)
// Launch bounds removed to let compiler auto-tune for different architectures (B200 vs GB10)
__global__ void scale_kernel_vectorized(float* data, int n, float scale) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  
  if (idx + 3 < n) {
    // Load 4 floats at once (128-bit transaction)
    float4 vec = *reinterpret_cast<float4*>(&data[idx]);
    
    // Process all 4 elements
#pragma unroll 4
    for (int iter = 0; iter < WORK_ITERS; ++iter) {
      vec.x = vec.x * scale + BIAS_ADD;
      vec.y = vec.y * scale + BIAS_ADD;
      vec.z = vec.z * scale + BIAS_ADD;
      vec.w = vec.w * scale + BIAS_ADD;
      vec.x = vec.x * DECAY - BIAS_SUB;
      vec.y = vec.y * DECAY - BIAS_SUB;
      vec.z = vec.z * DECAY - BIAS_SUB;
      vec.w = vec.w * DECAY - BIAS_SUB;
    }
    
    // Store 4 floats at once
    *reinterpret_cast<float4*>(&data[idx]) = vec;
  } else {
    // Handle remaining elements
    for (int i = idx; i < n; i++) {
      float val = data[i];
#pragma unroll 4
      for (int iter = 0; iter < WORK_ITERS; ++iter) {
        val = val * scale + BIAS_ADD;
        val = val * DECAY - BIAS_SUB;
      }
      data[i] = val;
    }
  }
}

// Shared-memory staging; this code does not issue an asynchronous copy or TMA.
// Launch with at most 256 threads. All threads participate in the barrier.
__global__ void scale_kernel_async(float* __restrict__ data, int n, float scale) {
  __shared__ float4 smem[256];
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 4;
  float4 vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (idx + 3 < n) vec = *reinterpret_cast<const float4*>(data + idx);
  else {
    if (idx < n) vec.x = data[idx];
    if (idx + 1 < n) vec.y = data[idx + 1];
    if (idx + 2 < n) vec.z = data[idx + 2];
  }
  smem[tid] = vec;
  __syncthreads();
  vec = smem[tid];
#pragma unroll
  for (int iter = 0; iter < WORK_ITERS; ++iter) {
    vec.x = (vec.x * scale + BIAS_ADD) * DECAY - BIAS_SUB;
    vec.y = (vec.y * scale + BIAS_ADD) * DECAY - BIAS_SUB;
    vec.z = (vec.z * scale + BIAS_ADD) * DECAY - BIAS_SUB;
    vec.w = (vec.w * scale + BIAS_ADD) * DECAY - BIAS_SUB;
  }
  if (idx + 3 < n) *reinterpret_cast<float4*>(data + idx) = vec;
  else {
    if (idx < n) data[idx] = vec.x;
    if (idx + 1 < n) data[idx + 1] = vec.y;
    if (idx + 2 < n) data[idx + 2] = vec.z;
  }
}

float stream_input(int index, int buffer) {
  return static_cast<float>((index % 257) - 128) / 256.0f + 0.25f * buffer;
}

float reference_scale(float value, float scale, int launches) {
  for (int launch = 0; launch < launches; ++launch) {
    for (int iter = 0; iter < WORK_ITERS; ++iter) {
      value = value * scale + BIAS_ADD;
      value = value * DECAY - BIAS_SUB;
    }
  }
  return value;
}

bool verify_stream_output(const float* output, int n, int buffer, float scale, int launches) {
  float reference[257];
  for (int i = 0; i < 257; ++i) reference[i] = reference_scale(stream_input(i, buffer), scale, launches);
  for (int i = 0; i < n; ++i) {
    const float expected = reference[i % 257];
    if (!std::isfinite(output[i]) || std::abs(output[i] - expected) > 2e-5f * std::max(1.0f, std::abs(expected))) {
      std::fprintf(stderr, "overlap mismatch at %d: %.9g versus %.9g\n", i, output[i], expected);
      return false;
    }
  }
  return true;
}

int run_stream_overlap_demo(int N, int ITERS, int PIPELINE_BATCHES) {
  if (N <= 0 || ITERS <= 0 || PIPELINE_BATCHES <= 0) return 1;
  const size_t BYTES = static_cast<size_t>(N) * sizeof(float);
  constexpr int WARMUP = 5;
  float *h_a = nullptr, *h_b = nullptr, *d_a = nullptr, *d_b = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_a, BYTES));
  CUDA_CHECK(cudaMallocHost(&h_b, BYTES));
  CUDA_CHECK(cudaMalloc(&d_a, BYTES));
  CUDA_CHECK(cudaMalloc(&d_b, BYTES));
  cudaStream_t stream1, stream2, d2h_stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&d2h_stream, cudaStreamNonBlocking));
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  cudaDeviceProp prop;
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const bool use_float8 = prop.major == 10 || prop.major == 12;
  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  dim3 grid_vec_float4((N + block.x * 4 - 1) / (block.x * 4));
  dim3 grid_vec_float8((N + block.x * 8 - 1) / (block.x * 8));
  auto launch_vector = [&](float* data, float scale, cudaStream_t stream) {
    if (use_float8) scale_kernel_vectorized_float8<<<grid_vec_float8, block, 0, stream>>>(data, N, scale);
    else scale_kernel_vectorized<<<grid_vec_float4, block, 0, stream>>>(data, N, scale);
  };
  auto reset_host = [&] {
    for (int i = 0; i < N; ++i) {
      h_a[i] = stream_input(i, 0);
      h_b[i] = stream_input(i, 1);
    }
  };
  auto launch_variant = [&](int kind) {
    if (kind == 0) scale_kernel<<<grid, block, 0, stream1>>>(d_a, N, 1.1f);
    else if (kind == 1) launch_vector(d_a, 1.1f, stream1);
    else scale_kernel_async<<<grid_vec_float4, block, 0, stream1>>>(d_a, N, 1.1f);
  };
  bool correct = true;
  float kernel_ms[3];
  for (int kind = 0; kind < 3; ++kind) {
    reset_host();
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, BYTES, cudaMemcpyHostToDevice, stream1));
    for (int i = 0; i < WARMUP; ++i) launch_variant(kind);
    // Reset after warmup so every variant measures identical input and work.
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, BYTES, cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaEventRecord(start, stream1));
    for (int i = 0; i < ITERS; ++i) launch_variant(kind);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, stream1));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms[kind], start, stop));
    CUDA_CHECK(cudaMemcpyAsync(h_a, d_a, BYTES, cudaMemcpyDeviceToHost, stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    correct = verify_stream_output(h_a, N, 0, 1.1f, ITERS) && correct;
  }

  cudaEvent_t h2d_done[2], compute_done[2], d2h_done[2];
  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaEventCreateWithFlags(&h2d_done[i], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&compute_done[i], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&d2h_done[i], cudaEventDisableTiming));
  }
  auto measure_pipeline = [&](bool overlap) {
    reset_host();
    for (int i = 0; i < 2; ++i) CUDA_CHECK(cudaEventRecord(d2h_done[i], d2h_stream));
    CUDA_CHECK(cudaStreamSynchronize(d2h_stream));
    const auto t0 = std::chrono::steady_clock::now();
    // Both modes process each buffer PIPELINE_BATCHES times, with feedback.
    for (int job = 0; job < 2 * PIPELINE_BATCHES; ++job) {
      const int buf = job & 1;
      float* d_buf = buf == 0 ? d_a : d_b;
      float* h_buf = buf == 0 ? h_a : h_b;
      const float scale = buf == 0 ? 1.05f : 0.95f;
      if (!overlap) {
        CUDA_CHECK(cudaMemcpyAsync(d_buf, h_buf, BYTES, cudaMemcpyHostToDevice, stream1));
        launch_vector(d_buf, scale, stream1);
        CUDA_CHECK(cudaMemcpyAsync(h_buf, d_buf, BYTES, cudaMemcpyDeviceToHost, stream1));
        CUDA_CHECK(cudaStreamSynchronize(stream1));
      } else {
        CUDA_CHECK(cudaStreamWaitEvent(stream1, d2h_done[buf], 0));
        CUDA_CHECK(cudaMemcpyAsync(d_buf, h_buf, BYTES, cudaMemcpyHostToDevice, stream1));
        CUDA_CHECK(cudaEventRecord(h2d_done[buf], stream1));
        CUDA_CHECK(cudaStreamWaitEvent(stream2, h2d_done[buf], 0));
        launch_vector(d_buf, scale, stream2);
        CUDA_CHECK(cudaEventRecord(compute_done[buf], stream2));
        CUDA_CHECK(cudaStreamWaitEvent(d2h_stream, compute_done[buf], 0));
        CUDA_CHECK(cudaMemcpyAsync(h_buf, d_buf, BYTES, cudaMemcpyDeviceToHost, d2h_stream));
        CUDA_CHECK(cudaEventRecord(d2h_done[buf], d2h_stream));
      }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    CUDA_CHECK(cudaStreamSynchronize(d2h_stream));
    const auto t1 = std::chrono::steady_clock::now();
    correct = verify_stream_output(h_a, N, 0, 1.05f, PIPELINE_BATCHES) && correct;
    correct = verify_stream_output(h_b, N, 1, 0.95f, PIPELINE_BATCHES) && correct;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };
  const double sequential_ms = measure_pipeline(false);
  const double overlap_ms = measure_pipeline(true);
  std::printf("Overlap full-output verification: %s (%d elements per buffer)\n", correct ? "PASS" : "FAIL", N);
  if (correct) {
    std::printf("GPU: %s; vector width: %d floats\n", prop.name, use_float8 ? 8 : 4);
    const char* names[] = {"Scalar", "Vectorized", "Shared-memory staging"};
    for (int kind = 0; kind < 3; ++kind) {
      const double ms = kernel_ms[kind] / ITERS;
      std::printf("%s kernel: %.6f ms (%.2f GB/s effective read+write)\n", names[kind], ms, 2.0 * BYTES / (ms * 1e6));
    }
    std::printf("Sequential pipeline: %.3f ms; overlapped pipeline: %.3f ms; identical %d jobs\n", sequential_ms, overlap_ms, 2 * PIPELINE_BATCHES);
    if (overlap_ms > 0) std::printf("Stream overlap speedup: %.2fx\n", sequential_ms / overlap_ms);
  }
  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaEventDestroy(h2d_done[i]));
    CUDA_CHECK(cudaEventDestroy(compute_done[i]));
    CUDA_CHECK(cudaEventDestroy(d2h_done[i]));
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream1));
  CUDA_CHECK(cudaStreamDestroy(stream2));
  CUDA_CHECK(cudaStreamDestroy(d2h_stream));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFreeHost(h_a));
  CUDA_CHECK(cudaFreeHost(h_b));
  return correct ? 0 : 1;
}

int main(int argc, char** argv) {
  NVTX_RANGE("main");
  int n = 1 << 22, iterations = 100, batches = 4;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--elements") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--batches") == 0 && i + 1 < argc) batches = std::atoi(argv[++i]);
    else return 1;
  }
  return run_stream_overlap_demo(n, iterations, batches);
}
