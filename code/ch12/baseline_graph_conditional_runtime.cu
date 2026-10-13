// baseline_graph_conditional_runtime.cu
//
// Baseline: Runtime conditional execution WITHOUT CUDA graph conditional nodes.
// Uses host-side decision making with graph switching.
//
// Limitations:
// - Requires host synchronization to read condition
// - Cannot branch within a single graph
// - Higher latency due to host roundtrip
// - Must maintain multiple graph instantiations
//
// This baseline demonstrates the traditional approach before CUDA 12.4's
// conditional graph nodes (cudaGraphConditionalHandle).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
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

constexpr int N = 1 << 16;  // 64K elements
constexpr int THREADS = 256;

// Expensive computation kernel (e.g., full attention)
__global__ void expensive_kernel(float* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        // Simulate expensive computation
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            val = sqrtf(val * val + scale) * 0.99f;
        }
        data[idx] = val;
    }
}

// Cheap computation kernel (e.g., cached lookup)
__global__ void cheap_kernel(float* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// Condition evaluation kernel (writes result to device memory)
__global__ void evaluate_condition(float* data, int n, int* condition, float threshold) {
    // Simple condition: check if mean > threshold
    // In real use: speculative decode acceptance, KV cache hit, etc.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        int sample_count = min(n, 1024);
        for (int i = 0; i < sample_count; ++i) {
            sum += data[i];
        }
        float mean = sum / sample_count;
        *condition = (mean > threshold) ? 1 : 0;
    }
}

int main() {
    NVTX_RANGE("main");
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    std::printf("======================================================================\n");
    std::printf("Baseline: Graph Conditional Runtime (Host-Side Switching)\n");
    std::printf("======================================================================\n");
    std::printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("\n");
    
    // Check graph support
    bool supports_graphs = (prop.major >= 7 && prop.minor >= 5) || prop.major >= 8;
    if (!supports_graphs) {
        // Exit code 3 plus a SKIPPED: line is the harness skip contract, as in the
        // optimized binary. Never report a 0.0 ms time for work that did not run.
        std::printf("SKIPPED: CUDA Graphs require compute capability 7.5+\n");
        return 3;
    }
    
    // Allocate memory
    size_t bytes = N * sizeof(float);
    float *d_data = nullptr;
    int *d_condition = nullptr;
    int h_condition = 0;
    
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_condition, sizeof(int)));
    
    // Initialize data
    std::vector<float> h_data(N);
    for (int i = 0; i < N; ++i) {
        NVTX_RANGE("setup");
        h_data[i] = 1.0f + (i % 100) * 0.01f;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
    
    dim3 block(THREADS);
    dim3 grid((N + block.x - 1) / block.x);
    
    // Create streams for graph capture
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    
    // ========================================
    // Baseline approach: Capture TWO separate graphs
    // (one for expensive path, one for cheap path)
    // Host must decide which to launch
    // ========================================
    
    cudaGraph_t graph_expensive, graph_cheap;
    cudaGraphExec_t exec_expensive, exec_cheap;
    
    // Capture expensive path graph
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    expensive_kernel<<<grid, block, 0, stream>>>(d_data, N, 1.01f);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph_expensive));
    CUDA_CHECK(cudaGraphInstantiate(&exec_expensive, graph_expensive, nullptr, nullptr, 0));
    
    // Capture cheap path graph
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    cheap_kernel<<<grid, block, 0, stream>>>(d_data, N, 1.001f);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph_cheap));
    CUDA_CHECK(cudaGraphInstantiate(&exec_cheap, graph_cheap, nullptr, nullptr, 0));
    
    // ========================================
    // Benchmark: Host-side conditional switching
    // ========================================
    constexpr int WARMUP = 10;
    constexpr int ITERS = 5000;
    
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        NVTX_RANGE("compute_graph:launch");
        evaluate_condition<<<1, 1, 0, stream>>>(d_data, N, d_condition, 0.5f);
        CUDA_CHECK(cudaStreamSynchronize(stream));  // HOST SYNC - slow!
        CUDA_CHECK(cudaMemcpy(&h_condition, d_condition, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (h_condition) {
            CUDA_CHECK(cudaGraphLaunch(exec_expensive, stream));
        } else {
            CUDA_CHECK(cudaGraphLaunch(exec_cheap, stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Timed iterations
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start, stream));
    
    for (int i = 0; i < ITERS; ++i) {
        NVTX_RANGE("transfer_sync:d2h");
        // Evaluate condition on device
        evaluate_condition<<<1, 1, 0, stream>>>(d_data, N, d_condition, 0.5f);
        
        // HOST ROUNDTRIP - the slow part!
        // Must synchronize to read condition before deciding which graph to launch
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(&h_condition, d_condition, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Host-side decision
        if (h_condition) {
            CUDA_CHECK(cudaGraphLaunch(exec_expensive, stream));
        } else {
            CUDA_CHECK(cudaGraphLaunch(exec_cheap, stream));
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / ITERS;
    
    std::printf("Results:\n");
    std::printf("  Total time: %.2f ms (%d iterations)\n", total_ms, ITERS);
    std::printf("  Average per iteration: %.3f ms\n", avg_ms);
    std::printf("\n");
    std::printf("Baseline limitations:\n");
    std::printf("  - Host synchronization per decision (%.3f ms overhead)\n", avg_ms * 0.3);
    std::printf("  - Cannot branch within single graph\n");
    std::printf("  - Requires multiple graph instantiations\n");
    std::printf("\n");
    std::printf("See optimized version for cudaGraphConditionalHandle approach.\n");
    std::printf("\n");
    std::printf("TIME_MS: %.6f\n", avg_ms);
    
    // Cleanup
    CUDA_CHECK(cudaGraphExecDestroy(exec_expensive));
    CUDA_CHECK(cudaGraphExecDestroy(exec_cheap));
    CUDA_CHECK(cudaGraphDestroy(graph_expensive));
    CUDA_CHECK(cudaGraphDestroy(graph_cheap));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_condition));
    
    return 0;
}


