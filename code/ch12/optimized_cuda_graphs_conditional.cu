// optimized_cuda_graphs_conditional.cu -- CUDA graph replay with device-side branching.

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
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                              \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int N = 1 << 16;
constexpr int KERNEL_ITERS = 1024;

__device__ __forceinline__ float expensive_path(float val, float scale) {
    #pragma unroll 4
    for (int i = 0; i < KERNEL_ITERS; ++i) {
        val = sqrtf(val * val + scale) * 0.99f;
    }
    return val;
}

__device__ __forceinline__ float cheap_path(float val, float scale) {
    return val * scale;
}

__global__ void predicate_kernel(int* condition, const float* data, float threshold) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *condition = (data[0] > threshold) ? 1 : 0;
    }
}

__global__ void conditional_dispatch_kernel(
    const int* condition,
    float* data,
    int n,
    float expensive_scale,
    float cheap_scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const bool run_expensive = (*condition != 0);
    const float val = data[idx];
    data[idx] = run_expensive
        ? expensive_path(val, expensive_scale)
        : cheap_path(val, cheap_scale);
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
    std::printf(
        "Optimized Conditional Graphs (CUDA graph + device-side branching) on %s (SM %d.%d)\n",
        prop.name,
        prop.major,
        prop.minor);

    const bool supports_graphs = (prop.major >= 7 && prop.minor >= 5) || prop.major >= 8;
    if (!supports_graphs) {
        std::printf("CUDA Graphs require compute capability 7.5 or newer.\n");
        return 0;
    }

    const size_t bytes = N * sizeof(float);
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    std::vector<float> h_data(N);
    for (int i = 0; i < N; ++i) {
        NVTX_RANGE("setup");
        h_data[i] = 1.0f + (i % 100) * 0.01f;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    constexpr float THRESHOLD = 0.5f;

    int* d_condition = nullptr;
    CUDA_CHECK(cudaMalloc(&d_condition, sizeof(int)));

    cudaStream_t graph_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&graph_stream, cudaStreamNonBlocking));

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    CUDA_CHECK(cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal));
    predicate_kernel<<<1, 1, 0, graph_stream>>>(d_condition, d_data, THRESHOLD);
    conditional_dispatch_kernel<<<grid, block, 0, graph_stream>>>(
        d_condition,
        d_data,
        N,
        1.01f,
        0.99f);
    CUDA_CHECK(cudaStreamEndCapture(graph_stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    constexpr int ITERS = 5000;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, graph_stream));
    for (int i = 0; i < ITERS; ++i) {
        NVTX_RANGE("compute_graph:launch");
        CUDA_CHECK(cudaGraphLaunch(graph_exec, graph_stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, graph_stream));
    CUDA_CHECK(cudaStreamSynchronize(graph_stream));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::printf(
        "Optimized (graph replay with device-side branch): %.2f ms (%.3f ms/iter)\n",
        ms,
        ms / ITERS);

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

    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(graph_stream));
    CUDA_CHECK(cudaFree(d_condition));
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
