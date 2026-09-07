// baseline_memory_transfer_multigpu.cu - GPU-to-GPU transfer via host staging.
// Compile: nvcc -O3 -std=c++17 -arch=sm_121
// baseline_memory_transfer_multigpu.cu -o
// baseline_memory_transfer_multigpu_sm121

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(status));                                \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

constexpr size_t kElementCount = 100 * 1024 * 1024;
constexpr int kIterations = 100;
constexpr size_t kPatternPeriod = 4093;

__global__ void initialize_source(float *data, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const int centered = static_cast<int>(index % kPatternPeriod) - 2046;
    data[index] = static_cast<float>(centered) * (1.0f / 256.0f);
  }
}

static float expected_value(size_t index) {
  const int centered = static_cast<int>(index % kPatternPeriod) - 2046;
  return static_cast<float>(centered) * (1.0f / 256.0f);
}

static const char *parse_dump_path(int argc, char **argv) {
  const char *dump_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dump-output") == 0) {
      if (++i >= argc) {
        std::fprintf(stderr, "--dump-output requires a path argument\n");
        std::exit(2);
      }
      dump_path = argv[i];
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      std::exit(2);
    }
  }
  return dump_path;
}

static void validate_and_dump_destination(float *d_dst, int dst_device,
                                          size_t count, const char *dump_path) {
  NVTX_RANGE("verification");
  std::vector<float> output(count);
  CUDA_CHECK(cudaSetDevice(dst_device));
  CUDA_CHECK(cudaMemcpy(output.data(), d_dst, count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < count; ++i) {
    const float expected = expected_value(i);
    if (output[i] != expected) {
      std::fprintf(
          stderr,
          "Destination mismatch at element %zu: expected %.9g, got %.9g\n", i,
          expected, output[i]);
      std::exit(EXIT_FAILURE);
    }
  }
  std::printf("OUTPUT_VALIDATED: %zu\n", count);

  if (dump_path != nullptr) {
    std::FILE *file = std::fopen(dump_path, "wb");
    if (file == nullptr) {
      std::fprintf(stderr, "Failed to open dump path: %s\n", dump_path);
      std::exit(2);
    }
    const size_t written =
        std::fwrite(output.data(), sizeof(float), count, file);
    const int close_status = std::fclose(file);
    if (written != count || close_status != 0) {
      std::fprintf(stderr, "Failed to write complete destination dump: %s\n",
                   dump_path);
      std::exit(2);
    }
  }
}

int main(int argc, char **argv) {
  NVTX_RANGE("main");
  const char *dump_path = parse_dump_path(argc, argv);
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    std::printf("SKIPPED: requires >=2 GPUs\n");
    return 3;
  }

  const int src_device = 0;
  const int dst_device = 1;
  const size_t bytes = kElementCount * sizeof(float);

  std::printf("=== Baseline: GPU-to-GPU via Host Staging (PCIe) ===\n");
  std::printf("Devices: %d -> %d\n", src_device, dst_device);
  std::printf("Array size: %zu elements (%.1f MB)\n\n", kElementCount,
              bytes / 1e6);

  float *d_src = nullptr;
  float *d_dst = nullptr;
  float *h_buffer = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_buffer, bytes));

  CUDA_CHECK(cudaSetDevice(src_device));
  CUDA_CHECK(cudaMalloc(&d_src, bytes));
  initialize_source<<<(kElementCount + 255) / 256, 256>>>(d_src, kElementCount);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaSetDevice(dst_device));
  CUDA_CHECK(cudaMalloc(&d_dst, bytes));
  CUDA_CHECK(cudaMemset(d_dst, 0, bytes));
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaSetDevice(src_device));
  CUDA_CHECK(cudaMemcpy(h_buffer, d_src, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaSetDevice(dst_device));
  CUDA_CHECK(cudaMemcpy(d_dst, h_buffer, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  const auto start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < kIterations; ++iter) {
    NVTX_RANGE("transfer_sync:h2d");
    CUDA_CHECK(cudaSetDevice(src_device));
    CUDA_CHECK(cudaMemcpy(h_buffer, d_src, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaSetDevice(dst_device));
    CUDA_CHECK(cudaMemcpy(d_dst, h_buffer, bytes, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto end = std::chrono::high_resolution_clock::now();

  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  const double avg_ms = elapsed_ms / kIterations;
  const double bandwidth_gbs = (2.0 * bytes / 1e9) / (avg_ms / 1000.0);
  std::printf("Average time per iteration: %.3f ms\n", avg_ms);
  std::printf("Bandwidth: %.2f GB/s (host-staged)\n", bandwidth_gbs);
  std::printf("TIME_MS: %.9f\n", avg_ms);

  validate_and_dump_destination(d_dst, dst_device, kElementCount, dump_path);

  CUDA_CHECK(cudaSetDevice(src_device));
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaSetDevice(dst_device));
  CUDA_CHECK(cudaFree(d_dst));
  CUDA_CHECK(cudaFreeHost(h_buffer));
  return 0;
}
