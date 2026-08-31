// memory_transfer_zero_copy_demo.cu -- GPU access to GB10 coherent system memory
// DGX Spark's published peak unified-memory bandwidth is 273 GB/s. This
// benchmark reports observed effective traffic and does not equate a link rate
// with sustained application bandwidth.
// Compile: nvcc -O3 -std=c++17 -arch=sm_121 memory_transfer_zero_copy_demo.cu -o memory_transfer_zero_copy_demo

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>
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

// Zero-copy kernel: Direct CPU memory access from GPU
// On GB10, this reads its coherent LPDDR5X unified system memory.
__global__ void zero_copy_process_kernel(
    const float* __restrict__ cpu_input,   // CPU memory (via NVLink-C2C)
    float* __restrict__ gpu_output,         // GPU-accessible output allocation
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Direct load from CPU memory - compiler generates coherent access
        // The measured rate is bounded by the platform's 273 GB/s peak unified
        // system-memory bandwidth and by this kernel's compute and write traffic.
        float val = cpu_input[idx];
        
        // Compute
        float result = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            result += sqrtf(val * val + float(i)) * 0.125f;
        }
        
        // Write to GPU memory
        gpu_output[idx] = result;
    }
}

// Bidirectional zero-copy: GPU reads CPU memory, writes back to CPU memory
__global__ void bidirectional_zero_copy_kernel(
    const float* __restrict__ cpu_input,
    float* __restrict__ cpu_output,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Read from CPU memory
        float val = cpu_input[idx];
        
        // Compute
        float result = val * val + val * 0.5f + 1.0f;
        
        // Write directly to CPU memory (no explicit D2H transfer!)
        cpu_output[idx] = result;
    }
}

// Traditional approach: Explicit copy for comparison
__global__ void traditional_process_kernel(
    const float* __restrict__ gpu_input,
    float* __restrict__ gpu_output,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = gpu_input[idx];
        
        float result = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            result += sqrtf(val * val + float(i)) * 0.125f;
        }
        
        gpu_output[idx] = result;
    }
}

int main() {
    NVTX_RANGE("main");
    // Detect architecture
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    bool is_gb10 = (prop.major == 12 && prop.minor == 1);
    
    std::printf("=== GB10 Zero-Copy Coherent Memory Benchmark ===\n");
    std::printf("Architecture: %s (SM %d.%d)\n", 
                is_gb10 ? "Grace-Blackwell GB10" : "Other",
                prop.major, prop.minor);

    if (!is_gb10) {
        std::fprintf(
            stderr,
            "SKIPPED: this coherent unified-memory benchmark requires "
            "Grace-Blackwell GB10 (SM 12.1); observed SM %d.%d.\n",
            prop.major,
            prop.minor);
        return EXIT_FAILURE;
    }
    
    // Test with moderately large array
    constexpr size_t N = 64 * 1024 * 1024;  // 64M elements = 256 MB
    constexpr size_t BYTES = N * sizeof(float);
    
    std::printf("\nTest configuration:\n");
    std::printf("  Array size: %zu MB\n", BYTES / (1024 * 1024));
    std::printf("  Elements: %zu\n\n", N);
    
    // Allocate CPU memory (pinned for GPU access)
    float *h_input = nullptr, *h_output = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_input, BYTES));  // Pinned, GPU-accessible
    CUDA_CHECK(cudaMallocHost(&h_output, BYTES));
    
    // Allocate GPU memory for comparison
    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, BYTES));
    CUDA_CHECK(cudaMalloc(&d_output, BYTES));
    
    // Initialize
    for (size_t i = 0; i < N; ++i) {
        NVTX_RANGE("setup");
        h_input[i] = static_cast<float>(i % 1000) / 1000.0f;
    }
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    constexpr int WARMUP = 10;
    constexpr int ITERS = 50;
    
    // ============================================================
    // Test 1: Traditional approach (H2D + kernel + D2H)
    // ============================================================
    std::printf("Test 1: Traditional (H2D copy + kernel + D2H copy)\n");
    
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        NVTX_RANGE("warmup");
        CUDA_CHECK(cudaMemcpy(d_input, h_input, BYTES, cudaMemcpyHostToDevice));
        traditional_process_kernel<<<grid, block>>>(d_input, d_output, N);
        CUDA_CHECK(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        NVTX_RANGE("transfer_sync:h2d");
        CUDA_CHECK(cudaMemcpy(d_input, h_input, BYTES, cudaMemcpyHostToDevice));
        traditional_process_kernel<<<grid, block>>>(d_input, d_output, N);
        CUDA_CHECK(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_traditional = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_traditional, start, stop));
    float avg_traditional = ms_traditional / ITERS;
    
    std::printf("  Time: %.3f ms\n", avg_traditional);
    std::printf("  Throughput: %.2f GB/s\n\n", (3.0f * BYTES / 1e9) / (avg_traditional / 1000.0f));
    
    // ============================================================
    // Test 2: Zero-copy (direct CPU memory access)
    // ============================================================
    std::printf("Test 2: Zero-Copy (direct coherent system-memory access)\n");
    
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        NVTX_RANGE("warmup");
        zero_copy_process_kernel<<<grid, block>>>(h_input, d_output, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        NVTX_RANGE("compute_kernel:zero_copy_process_kernel");
        zero_copy_process_kernel<<<grid, block>>>(h_input, d_output, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_zerocopy = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_zerocopy, start, stop));
    float avg_zerocopy = ms_zerocopy / ITERS;
    
    std::printf("  Time: %.3f ms\n", avg_zerocopy);
    std::printf("  Throughput: %.2f GB/s (effective system-memory reads)\n",
                (BYTES / 1e9) / (avg_zerocopy / 1000.0f));
    std::printf("  Speedup: %.2fx vs traditional\n\n", avg_traditional / avg_zerocopy);
    
    // ============================================================
    // Test 3: Bidirectional zero-copy (CPU read + CPU write)
    // ============================================================
    std::printf("Test 3: Bidirectional Zero-Copy (CPU→GPU→CPU, no explicit transfers)\n");
    
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        NVTX_RANGE("warmup");
        bidirectional_zero_copy_kernel<<<grid, block>>>(h_input, h_output, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        NVTX_RANGE("compute_kernel:bidirectional_zero_copy_kernel");
        bidirectional_zero_copy_kernel<<<grid, block>>>(h_input, h_output, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_bidir = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_bidir, start, stop));
    float avg_bidir = ms_bidir / ITERS;
    
    std::printf("  Time: %.3f ms\n", avg_bidir);
    std::printf("  Throughput: %.2f GB/s (effective system-memory read + write traffic)\n",
                (2.0f * BYTES / 1e9) / (avg_bidir / 1000.0f));
    std::printf("  Speedup: %.2fx vs traditional\n\n", avg_traditional / avg_bidir);
    
    // Verify
    CUDA_CHECK(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (size_t i = 0; i < std::min(N, size_t(1000)); ++i) {
        NVTX_RANGE("verify");
        float val = h_input[i];
        float expected = 0.0f;
        for (int j = 0; j < 8; ++j) {
            NVTX_RANGE("verify");
            expected += std::sqrt(val * val + float(j)) * 0.125f;
        }
        if (std::abs(h_output[i] - expected) > 1e-4) {
            correct = false;
            break;
        }
    }
    
    // Results summary
    std::printf("=== Summary ===\n");
    std::printf("Traditional:          %.3f ms (1.00x)\n", avg_traditional);
    std::printf("Zero-copy:            %.3f ms (%.2fx faster)\n", 
                avg_zerocopy, avg_traditional / avg_zerocopy);
    std::printf("Bidirectional:        %.3f ms (%.2fx faster)\n",
                avg_bidir, avg_traditional / avg_bidir);
    std::printf("\nCorrectness: %s\n", correct ? "✅ PASSED" : "❌ FAILED");
    
    std::printf("\n✅ GB10 Characteristics:\n");
    std::printf("  • No explicit H2D/D2H transfers needed\n");
    std::printf("  • 273 GB/s peak unified system-memory bandwidth\n");
    std::printf("  • Reported throughput is measured effective traffic, not a link-rate claim\n");
    std::printf("  • Shared system memory can reduce duplicate CPU/GPU allocations\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
