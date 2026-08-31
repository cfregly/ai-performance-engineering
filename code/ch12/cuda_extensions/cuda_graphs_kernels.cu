// cuda_graphs_kernels.cu - CUDA kernels for CUDA graphs benchmarks
// Can be loaded as PyTorch CUDA extension

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include "profiling_helpers.cuh"

#ifndef cudaGraphInstantiateFlagAutoFreeOnLaunch
#define cudaGraphInstantiateFlagAutoFreeOnLaunch 0
#endif

struct GraphCache {
    cudaGraphExec_t exec = nullptr;
    cudaStream_t stream = nullptr;
    int device = -1;
    int64_t num_elements = -1;
    int iterations = -1;
    float* data_ptr = nullptr;

    void reset() {
        if (exec != nullptr) {
            cudaGraphExecDestroy(exec);
            exec = nullptr;
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        device = -1;
        num_elements = -1;
        iterations = -1;
        data_ptr = nullptr;
    }
};

static GraphCache g_graph_cache;

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t _status = (call);                                          \
        TORCH_CHECK(_status == cudaSuccess,                                    \
                    "CUDA error at ", __FILE__, ":", __LINE__, " - ",          \
                    cudaGetErrorString(_status));                              \
    } while (0)

__global__ void kernel_a_kernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = data[idx] * 1.1f + 0.1f;
}

__global__ void kernel_b_kernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
}

__global__ void kernel_c_kernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = sinf(data[idx] * 0.1f);
}

void separate_kernel_launches(torch::Tensor data, int iterations) {
    TORCH_CHECK(data.is_cuda(), "data must be CUDA tensor");
    TORCH_CHECK(data.dtype() == torch::kFloat32, "data must be float32");
    
    // Ensure we're on the correct device
    int device_id = data.device().index();
    CHECK_CUDA(cudaSetDevice(device_id));
    
    int n = data.size(0);
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Use default stream (nullptr) - this is the legacy default stream
    // PyTorch operations on default stream will be properly synchronized
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("separate_kernel_launches");
        for (int i = 0; i < iterations; ++i) {
            kernel_a_kernel<<<num_blocks, threads_per_block, 0, stream>>>(data.data_ptr<float>(), n);
            kernel_b_kernel<<<num_blocks, threads_per_block, 0, stream>>>(data.data_ptr<float>(), n);
            kernel_c_kernel<<<num_blocks, threads_per_block, 0, stream>>>(data.data_ptr<float>(), n);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    // Note: No explicit synchronization here - PyTorch benchmark harness handles synchronization
}

void graph_replay(torch::Tensor data, int iterations) {
    TORCH_CHECK(data.is_cuda(), "data must be CUDA tensor");
    TORCH_CHECK(data.dtype() == torch::kFloat32, "data must be float32");
    
    // Ensure we're on the correct device
    int device_id = data.device().index();
    CHECK_CUDA(cudaSetDevice(device_id));
    
    int n = data.size(0);
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    float* const data_ptr = data.data_ptr<float>();

    bool needs_capture = (g_graph_cache.exec == nullptr) ||
                         (g_graph_cache.device != device_id) ||
                         (g_graph_cache.num_elements != n) ||
                         (g_graph_cache.iterations != iterations) ||
                         (g_graph_cache.data_ptr != data_ptr);

    if (needs_capture) {
        // Reset any previous cached graph/stream
        g_graph_cache.reset();

        CHECK_CUDA(cudaStreamCreateWithFlags(&g_graph_cache.stream, cudaStreamNonBlocking));
        cudaGraph_t graph = nullptr;

        try {
            CHECK_CUDA(cudaStreamBeginCapture(g_graph_cache.stream, cudaStreamCaptureModeGlobal));
            for (int i = 0; i < iterations; ++i) {
                kernel_a_kernel<<<num_blocks, threads_per_block, 0, g_graph_cache.stream>>>(data_ptr, n);
                kernel_b_kernel<<<num_blocks, threads_per_block, 0, g_graph_cache.stream>>>(data_ptr, n);
                kernel_c_kernel<<<num_blocks, threads_per_block, 0, g_graph_cache.stream>>>(data_ptr, n);
            }
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamEndCapture(g_graph_cache.stream, &graph));
#if CUDART_VERSION >= 12000
            CHECK_CUDA(cudaGraphInstantiateWithFlags(
                &g_graph_cache.exec,
                graph,
                cudaGraphInstantiateFlagAutoFreeOnLaunch));
#else
            CHECK_CUDA(cudaGraphInstantiate(&g_graph_cache.exec, graph, nullptr, nullptr, 0));
#endif
            CHECK_CUDA(cudaGraphUpload(g_graph_cache.exec, g_graph_cache.stream));
            CHECK_CUDA(cudaGraphDestroy(graph));
            g_graph_cache.device = device_id;
            g_graph_cache.num_elements = n;
            g_graph_cache.iterations = iterations;
            g_graph_cache.data_ptr = data_ptr;
        } catch (...) {
            if (graph != nullptr) {
                cudaGraphDestroy(graph);
            }
            g_graph_cache.reset();
            throw;
        }
    }

    if (g_graph_cache.exec == nullptr || g_graph_cache.stream == nullptr) {
        TORCH_CHECK(false, "CUDA graph cache not initialized");
    }

    {
        PROFILE_KERNEL_LAUNCH("graph_replay");
        CHECK_CUDA(cudaGraphLaunch(g_graph_cache.exec, g_graph_cache.stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(g_graph_cache.stream));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("separate_kernel_launches", &separate_kernel_launches, "Separate kernel launches (baseline)");
    m.def("graph_replay", &graph_replay, "CUDA graph replay (optimized)");
}
