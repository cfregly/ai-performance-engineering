// baseline_cuda_graphs_conditional.cu -- Standard approach with dynamic dispatch (baseline).

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                              \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int N = 1 << 16;
constexpr int KERNEL_ITERS = 1024;

__global__ void expensive_kernel(float* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        #pragma unroll 4
        for (int i = 0; i < KERNEL_ITERS; ++i) {
            val = sqrtf(val * val + scale) * 0.99f;
        }
        data[idx] = val;
    }
}

__global__ void cheap_kernel(float* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

__global__ void predicate_kernel(int* condition, float* data, int n, float threshold) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *condition = (data[0] > threshold) ? 1 : 0;
    }
}

static void write_output_dump(const char* path, const std::vector<float>& output) {
    std::FILE* file = std::fopen(path, "wb");
    if (file == nullptr) {
        std::fprintf(stderr, "Failed to open dump path: %s\n", path);
        std::exit(2);
    }
    const size_t written = std::fwrite(output.data(), sizeof(float), output.size(), file);
    const int close_status = std::fclose(file);
    if (written != output.size() || close_status != 0) {
        std::fprintf(stderr, "Failed to write complete output dump: %s\n", path);
        std::exit(2);
    }
    std::printf("OUTPUT_DUMPED: %zu\n", output.size());
}

int main(int argc, char** argv) {
    NVTX_RANGE("main");
    const char* dump_path = nullptr;
    if (argc != 1) {
        if (argc != 3 || std::strcmp(argv[1], "--dump-output") != 0) {
            std::fprintf(stderr, "Usage: %s [--dump-output <path>]\n", argv[0]);
            return 2;
        }
        dump_path = argv[2];
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Baseline Conditional Graphs (standard dynamic dispatch) on %s (SM %d.%d)\n", 
                prop.name, prop.major, prop.minor);
    
    size_t bytes = N * sizeof(float);
    float *d_data = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    
    std::vector<float> h_data(N);
    for (int i = 0; i < N; ++i) {
        NVTX_RANGE("setup");
        h_data[i] = 1.0f + (i % 100) * 0.01f;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    constexpr int ITERS = 5000;
    constexpr float THRESHOLD = 0.5f;
    
    int *d_condition = nullptr;
    CUDA_CHECK(cudaMalloc(&d_condition, sizeof(int)));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Baseline: host-side conditional dispatch with D2H synchronization
    // This demonstrates the cost of host-device synchronization
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        NVTX_RANGE("transfer_sync:d2h");
        // Evaluate condition on device
        predicate_kernel<<<1, 1>>>(d_condition, d_data, N, THRESHOLD);
        
        // D2H copy forces synchronization (this is the bottleneck!)
        int h_condition = 0;
        CUDA_CHECK(cudaMemcpy(&h_condition, d_condition, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Host-side conditional dispatch
        if (h_condition) {
            expensive_kernel<<<grid, block>>>(d_data, N, 1.01f);
        } else {
            cheap_kernel<<<grid, block>>>(d_data, N, 0.99f);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    std::printf("Baseline (individual kernel launches): %.2f ms (%.3f ms/iter)\n", ms, ms / ITERS);

    if (dump_path != nullptr) {
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));
        write_output_dump(dump_path, h_data);
    }

#ifdef VERIFY
    if (dump_path == nullptr) {
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));
    }
    double checksum = 0.0;
    for (float v : h_data) {
        checksum += std::abs(v);
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_condition));
    CUDA_CHECK(cudaFree(d_data));
    
    return 0;
}
