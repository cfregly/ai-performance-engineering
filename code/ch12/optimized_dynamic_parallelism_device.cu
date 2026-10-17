// optimized_dynamic_parallelism_device.cu
// Device-initiated launches with minor tweaks for better throughput.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include <vector>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t status = (call);                                              \
    if (status != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status));                               \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

__device__ __forceinline__ float fuse_op(float x) {
    // Same math as baseline; faster because we launch fewer, larger grids
    #pragma unroll 4
    for (int i = 0; i < 8; ++i) {
        x = fmaf(x, 1.0002f, 0.001f * (i + 1));
        x = tanhf(x);
    }
    return x;
}

// Child kernel with wider blocks covering the full buffer in one launch
__global__ void childKernelOpt(float* data, int start, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int global_idx = start + idx;
        float v = data[global_idx];
        data[global_idx] = fuse_op(v);
    }
}

__global__ void parentKernelOpt(float* data, int N, int* launch_count) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Optimized: single, wide child grid (reduces launch overhead and improves occupancy)
        dim3 child_block(512);
        dim3 child_grid((N + child_block.x - 1) / child_block.x);
        childKernelOpt<<<child_grid, child_block>>>(data, 0, N);
        atomicAdd(launch_count, 1);
    }
}

int main() {
    NVTX_RANGE("main");
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("Device: %s (CC %d.%d)\\n", prop.name, prop.major, prop.minor);

    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 2048));

    // Allocate data
    const int N = 262144;  // match baseline size for apples-to-apples
    float* d_data = nullptr;
    int* d_launch_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_launch_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_launch_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

    dim3 parent_grid(1);
    dim3 parent_block(256);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    parentKernelOpt<<<parent_grid, parent_block>>>(d_data, N, d_launch_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    int launches = 0;
    CUDA_CHECK(cudaMemcpy(&launches, d_launch_count, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("Device child launches (optimized): %d\n", launches);
    std::printf("Elapsed_ms: %.6f ms\n", elapsed_ms);

#ifdef VERIFY
    std::vector<float> h_data(N);
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (float v : h_data) {
        NVTX_RANGE("verify");
        checksum += static_cast<double>(v);
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_launch_count));
    return 0;
}
