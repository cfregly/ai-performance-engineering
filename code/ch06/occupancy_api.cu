// occupancy_api.cu -- demonstrate cudaOccupancyMaxPotentialBlockSize (CUDA 13.0).

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
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

__global__ void sampleKernel(float* data, int n) {
  extern __shared__ float tile[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  tile[threadIdx.x] = idx < n ? data[idx] : 0.0f;
  __syncthreads();
  if (idx < n) {
    data[idx] = sqrtf(tile[threadIdx.x] * tile[threadIdx.x] + 1.0f);
  }
}

size_t sample_shared_bytes(int block_size) {
  return static_cast<size_t>(block_size) * sizeof(float);
}

int main() {
    NVTX_RANGE("main");
  constexpr int N = 1 << 20;
  float* h_data = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_data, N * sizeof(float)));
  for (int i = 0; i < N; ++i) {
      NVTX_RANGE("setup");
    h_data[i] = static_cast<float>(i % 1000) / 1000.0f;
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  float* d_data = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_data, N * sizeof(float), stream));
  CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, N * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  int min_grid = 0;
  int block_size = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
      &min_grid,
      &block_size,
      sampleKernel,
      sample_shared_bytes,
      0));

  std::printf("Suggested block size: %d\n", block_size);
  std::printf("Minimum grid for full occupancy: %d\n", min_grid);

  int grid = (N + block_size - 1) / block_size;
  if (grid < min_grid) grid = min_grid;

  size_t shared_mem = sample_shared_bytes(block_size);
  sampleKernel<<<grid, block_size, shared_mem, stream>>>(d_data, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(h_data, d_data, N * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::printf("First result = %.3f\n", h_data[0]);

  CUDA_CHECK(cudaFreeAsync(d_data, stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  cudaFreeHost(h_data);
  return 0;
}
