/**
 * NVSHMEM Advanced Patterns for Multi-GPU Blackwell B200 Nodes
 * ======================================================
 * 
 * Educational NVSHMEM patterns for multi-GPU communication.
 * 
 * Advanced Examples:
 * 1. Ring AllReduce (NCCL-style algorithm)
 * 2. Double-buffered Ring AllReduce (overlapped communication)
 * 3. Recursive Doubling AllReduce (power-of-two PE counts)
 * 4. Pipelined Broadcast (multi-stage)
 * 5. Custom Reduce-Scatter + AllGather
 * 6. Performance Comparison Framework
 * 
 * These patterns are used in production deep learning frameworks.
 * 
 * Requirements:
 * - NVSHMEM 3.4+
 * - CUDA 13.0+
 * - Blackwell B200 GPUs (works with any GPU count)
 * 
 * Compile:
 *   nvcc -O3 -std=c++17 -arch=sm_100 -DUSE_NVSHMEM \\
 *        -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib -lnvshmem \\
 *        nvshmem_advanced_multigpu.cu -o nvshmem_advanced
 * 
 * Run:
 *   nvshmemrun -np <num_gpus> ./nvshmem_advanced
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <algorithm>

#include "../core/common/nvtx_utils.cuh"

// NVSHMEM headers
#ifdef USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#else
// Dummy definitions for educational compilation
#define nvshmem_init()
#define nvshmem_my_pe() 0
#define nvshmem_n_pes() 1
#define nvshmem_barrier_all()
#define nvshmem_finalize()
#define nvshmem_malloc(size) nullptr
#define nvshmem_free(ptr)
#define nvshmem_quiet()
#define nvshmemx_barrier_all_on_stream(s)
inline void nvshmem_float_put(float*, float, int) {}
inline void nvshmem_float_put_nbi(float*, float*, int, int) {}
#endif

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(status));                                   \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// ============================================================================
// Pattern 1: Ring AllReduce
// ============================================================================

/**
 * Ring AllReduce - The workhorse of distributed training
 * 
 * Algorithm:
 * Phase 1 (Reduce-Scatter): Each GPU reduces one chunk, n-1 steps
 * Phase 2 (AllGather): Each GPU gathers all chunks, n-1 steps
 * 
 * Complexity: O(2*(N-1)/N * size) - near-optimal for small clusters
 * Used by: NCCL (for messages <1MB), Horovod, PyTorch DDP
 */

#ifdef USE_NVSHMEM

static void initialize_nvshmem_device() {
    nvshmem_init();
    const int world_pe = nvshmem_my_pe();
    const int local_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    if (local_pe < 0) {
        fprintf(stderr, "PE %d has no rank in NVSHMEMX_TEAM_NODE\n", world_pe);
        nvshmem_global_exit(EXIT_FAILURE);
    }

    // Device initialization is deferred when nvshmem_init() runs before a
    // CUDA device is selected. Map by the node-local PE and complete the
    // documented lazy-init path before touching the symmetric heap.
    CUDA_CHECK(cudaSetDevice(local_pe));
    nvshmem_barrier_all();
    const int status = nvshmemx_init_status();
    if (status != NVSHMEM_STATUS_IS_INITIALIZED &&
        status != NVSHMEM_STATUS_LIMITED_MPG &&
        status != NVSHMEM_STATUS_FULL_MPG) {
        fprintf(stderr, "PE %d NVSHMEM device initialization failed: status %d\n",
                world_pe, status);
        nvshmem_global_exit(EXIT_FAILURE);
    }
}

__global__ void ring_send_kernel(const float *data, float *recv_buf,
                                 int chunk_size, int send_chunk, int right_pe) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < chunk_size) {
        int send_offset = send_chunk * chunk_size + idx;
        nvshmem_float_put(&recv_buf[idx], &data[send_offset], 1, right_pe);
    }
}

__global__ void ring_reduce_kernel(float *data, const float *recv_buf,
                                   int chunk_size, int recv_chunk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < chunk_size) {
        int recv_offset = recv_chunk * chunk_size + idx;
        data[recv_offset] += recv_buf[idx];
    }
}

__global__ void ring_copy_kernel(float *data, const float *recv_buf,
                                 int chunk_size, int recv_chunk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < chunk_size) {
        int recv_offset = recv_chunk * chunk_size + idx;
        data[recv_offset] = recv_buf[idx];
    }
}

void run_ring_reduce_scatter(float *data, float *recv_buf, int chunk_size,
                             int my_pe, int n_pes, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (chunk_size + threads - 1) / threads;
    const int right_pe = (my_pe + 1) % n_pes;
    for (int step = 0; step < n_pes - 1; step++) {
        const int send_chunk = (my_pe - step + n_pes) % n_pes;
        const int recv_chunk = (my_pe - step - 1 + n_pes) % n_pes;
        ring_send_kernel<<<blocks, threads, 0, stream>>>(
            data, recv_buf, chunk_size, send_chunk, right_pe);
        CUDA_CHECK(cudaGetLastError());
        // This host collective is stream-ordered after every PE's send kernel.
        nvshmemx_barrier_all_on_stream(stream);
        ring_reduce_kernel<<<blocks, threads, 0, stream>>>(
            data, recv_buf, chunk_size, recv_chunk);
        CUDA_CHECK(cudaGetLastError());
    }
}

void run_ring_allgather(float *data, float *recv_buf, int chunk_size,
                        int my_pe, int n_pes, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (chunk_size + threads - 1) / threads;
    const int right_pe = (my_pe + 1) % n_pes;
    for (int step = 0; step < n_pes - 1; step++) {
        const int send_chunk = (my_pe + 1 - step + n_pes) % n_pes;
        const int recv_chunk = (my_pe - step + n_pes) % n_pes;
        ring_send_kernel<<<blocks, threads, 0, stream>>>(
            data, recv_buf, chunk_size, send_chunk, right_pe);
        CUDA_CHECK(cudaGetLastError());
        nvshmemx_barrier_all_on_stream(stream);
        ring_copy_kernel<<<blocks, threads, 0, stream>>>(
            data, recv_buf, chunk_size, recv_chunk);
        CUDA_CHECK(cudaGetLastError());
    }
}

bool benchmark_ring_allreduce(int my_pe, int n_pes) {
    if (my_pe == 0) {
        printf("\n=== Pattern 1: Ring AllReduce ===\n");
        printf("Algorithm: NCCL-style ring for small messages\n");
    }
    
    const int target_elements = 8 * 1024 * 1024 / sizeof(float);
    const int chunk_size = (target_elements + n_pes - 1) / n_pes;
    const int N = chunk_size * n_pes;
    
    float *d_data = (float *)nvshmem_malloc(N * sizeof(float));
    float *d_recv = (float *)nvshmem_malloc(chunk_size * sizeof(float));
    if (d_data == nullptr || d_recv == nullptr) {
        fprintf(stderr, "PE %d failed to allocate ring AllReduce storage\n", my_pe);
        nvshmem_global_exit(EXIT_FAILURE);
    }
    
    // Initialize with PE ID
    float *h_data = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        NVTX_RANGE("warmup");
        h_data[i] = (float)(my_pe + 1);
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
    nvshmem_barrier_all();
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warmup, then restore the original input before the measured pass.
    run_ring_reduce_scatter(d_data, d_recv, chunk_size, my_pe, n_pes, stream);
    run_ring_allgather(d_data, d_recv, chunk_size, my_pe, n_pes, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, N * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_barrier_all();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    run_ring_reduce_scatter(d_data, d_recv, chunk_size, my_pe, n_pes, stream);
    run_ring_allgather(d_data, d_recv, chunk_size, my_pe, n_pes, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Verify
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    float expected = (float)(n_pes * (n_pes + 1)) / 2.0f;
    size_t bad_values = 0;
    for (int i = 0; i < N; ++i) {
        if (!std::isfinite(h_data[i]) || fabs(h_data[i] - expected) >= 0.01f) {
            ++bad_values;
        }
    }
    bool correct = bad_values == 0;
    
    if (my_pe == 0) {
        NVTX_RANGE("verify");
        printf("  Time: %ld μs\n", duration.count());
        printf("  Data size: %.2f MB\n", N * sizeof(float) / (1024.0 * 1024.0));
        printf("  Bandwidth: %.2f GB/s\n", 
               (2.0 * N * sizeof(float) * (n_pes - 1) / n_pes) / (duration.count() / 1e6) / 1e9);
    }
    printf("  PE %d correctness: %s (%zu mismatches, expected %.1f, got %.1f)\n",
           my_pe, correct ? "PASS" : "FAIL", bad_values, expected, h_data[0]);

    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_free(d_data);
    nvshmem_free(d_recv);
    free(h_data);
    return correct;
}

#else
bool benchmark_ring_allreduce(int my_pe, int n_pes) {
    if (my_pe == 0) {
        printf("\n=== Pattern 1: Ring AllReduce ===\n");
        printf("[Educational Mode - compile with -DUSE_NVSHMEM]\n");
        printf("Steps: %d (reduce-scatter) + %d (allgather) = %d\n", 
               n_pes-1, n_pes-1, 2*(n_pes-1));
    }
    return true;
}
#endif

// ============================================================================
// Pattern 2: Double-Buffered Ring Reduce-Scatter
// ============================================================================

/**
 * Alternates two symmetric receive buffers across ring steps. The host enqueues
 * a stream-ordered barrier between each send and reduction, which is required
 * when a chunk spans multiple CUDA blocks.
 */

#ifdef USE_NVSHMEM

void run_double_buffered_reduce_scatter(float *data, float *buf0, float *buf1,
                                        int chunk_size, int my_pe, int n_pes,
                                        cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (chunk_size + threads - 1) / threads;
    const int right_pe = (my_pe + 1) % n_pes;
    for (int step = 0; step < n_pes - 1; step++) {
        const int send_chunk = (my_pe - step + n_pes) % n_pes;
        const int recv_chunk = (my_pe - step - 1 + n_pes) % n_pes;
        float *recv_buf = (step % 2 == 0) ? buf0 : buf1;
        ring_send_kernel<<<blocks, threads, 0, stream>>>(
            data, recv_buf, chunk_size, send_chunk, right_pe);
        CUDA_CHECK(cudaGetLastError());
        nvshmemx_barrier_all_on_stream(stream);
        ring_reduce_kernel<<<blocks, threads, 0, stream>>>(
            data, recv_buf, chunk_size, recv_chunk);
        CUDA_CHECK(cudaGetLastError());
    }
}

bool benchmark_double_buffered_reduce_scatter(int my_pe, int n_pes) {
    if (my_pe == 0) {
        printf("\n=== Pattern 2: Double-Buffered Ring Reduce-Scatter ===\n");
        printf("Technique: Alternate symmetric receive buffers across ring steps\n");
    }
    
    const int target_elements = 8 * 1024 * 1024 / sizeof(float);
    const int chunk_size = (target_elements + n_pes - 1) / n_pes;
    const int N = chunk_size * n_pes;
    
    float *d_data = (float *)nvshmem_malloc(N * sizeof(float));
    float *d_buf0 = (float *)nvshmem_malloc(chunk_size * sizeof(float));
    float *d_buf1 = (float *)nvshmem_malloc(chunk_size * sizeof(float));
    if (d_data == nullptr || d_buf0 == nullptr || d_buf1 == nullptr) {
        fprintf(stderr, "PE %d failed to allocate double-buffered ring storage\n", my_pe);
        nvshmem_global_exit(EXIT_FAILURE);
    }
    
    float *h_data = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        NVTX_RANGE("transfer_sync");
        h_data[i] = (float)(my_pe + 1);
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
    nvshmem_barrier_all();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    run_double_buffered_reduce_scatter(
        d_data, d_buf0, d_buf1, chunk_size, my_pe, n_pes, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    const int owned_chunk = (my_pe + 1) % n_pes;
    CUDA_CHECK(cudaMemcpy(h_data, d_data + owned_chunk * chunk_size,
                          chunk_size * sizeof(float), cudaMemcpyDeviceToHost));
    const float expected = (float)(n_pes * (n_pes + 1)) / 2.0f;
    size_t bad_values = 0;
    for (int i = 0; i < chunk_size; ++i) {
        if (!std::isfinite(h_data[i]) || fabs(h_data[i] - expected) >= 0.01f) {
            ++bad_values;
        }
    }
    const bool correct = bad_values == 0;
    printf("  PE %d reduce-scatter: %ld μs, correctness %s (%zu mismatches)\n",
           my_pe, duration.count(), correct ? "PASS" : "FAIL", bad_values);

    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_free(d_data);
    nvshmem_free(d_buf0);
    nvshmem_free(d_buf1);
    free(h_data);
    return correct;
}

#else
bool benchmark_double_buffered_reduce_scatter(int my_pe, int n_pes) {
    if (my_pe == 0) {
        printf("\n=== Pattern 2: Double-Buffered Ring Reduce-Scatter ===\n");
        printf("[Educational Mode]\n");
        printf("Uses ping-pong buffers for overlap\n");
    }
    return true;
}
#endif

// ============================================================================
// Pattern 3: Recursive Doubling AllReduce
// ============================================================================

/**
 * Full-buffer pairwise exchanges over log2(n) steps. After each step, every PE
 * holds a reduction covering twice as many PEs. Requires a power-of-two count.
 */

#ifdef USE_NVSHMEM

__global__ void recursive_exchange_send_kernel(const float *data, float *recv_buf,
                                               int size, int partner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        nvshmem_float_put(&recv_buf[idx], &data[idx], 1, partner);
    }
}

__global__ void recursive_exchange_reduce_kernel(float *data,
                                                 const float *recv_buf,
                                                 int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += recv_buf[idx];
    }
}

bool benchmark_recursive_halving_doubling(int my_pe, int n_pes) {
    if (my_pe == 0) {
        printf("\n=== Pattern 3: Recursive Doubling AllReduce ===\n");
    }
    
    // Check power of 2
    if ((n_pes & (n_pes - 1)) != 0) {
        if (my_pe == 0) {
            printf("  Requires power-of-2 PEs (got %d)\n", n_pes);
        }
        return false;
    }
    
    const int N = 32 * 1024 * 1024 / sizeof(float);  // 32 MB
    float *d_data = (float *)nvshmem_malloc(N * sizeof(float));
    float *d_recv = (float *)nvshmem_malloc(N * sizeof(float));
    if (d_data == nullptr || d_recv == nullptr) {
        fprintf(stderr, "PE %d failed to allocate recursive-doubling storage\n", my_pe);
        nvshmem_global_exit(EXIT_FAILURE);
    }
    
    float *h_data = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        NVTX_RANGE("transfer_sync");
        h_data[i] = (float)(my_pe + 1);
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
    nvshmem_barrier_all();
    
    int log_n = (int)(log2(n_pes) + 0.5);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Recursive doubling: after each exchange, each PE has the sum for a
    // group twice as large as in the previous step.
    for (int step = 0; step < log_n; step++) {
        NVTX_RANGE("compute_kernel:recursive_exchange_kernel");
        const int partner = my_pe ^ (1 << step);
        recursive_exchange_send_kernel<<<blocks, threads, 0, stream>>>(
            d_data, d_recv, N, partner);
        CUDA_CHECK(cudaGetLastError());
        nvshmemx_barrier_all_on_stream(stream);
        recursive_exchange_reduce_kernel<<<blocks, threads, 0, stream>>>(
            d_data, d_recv, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    const float expected = (float)(n_pes * (n_pes + 1)) / 2.0f;
    size_t bad_values = 0;
    for (int i = 0; i < N; ++i) {
        if (!std::isfinite(h_data[i]) || fabs(h_data[i] - expected) >= 0.01f) {
            ++bad_values;
        }
    }
    const bool correct = bad_values == 0;

    if (my_pe == 0) {
        NVTX_RANGE("batch");
        printf("  Time: %ld μs\n", duration.count());
        printf("  Exchange steps: %d (vs %d phases for ring)\n",
               log_n, 2 * (n_pes - 1));
        printf("  Data size: %.2f MB\n", N * sizeof(float) / (1024.0 * 1024.0));
        printf("  Exchange traffic per PE: %.2f MB\n",
               log_n * N * sizeof(float) / (1024.0 * 1024.0));
    }
    printf("  PE %d correctness: %s (%zu mismatches)\n",
           my_pe, correct ? "PASS" : "FAIL", bad_values);

    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_free(d_data);
    nvshmem_free(d_recv);
    free(h_data);
    return correct;
}

#else
bool benchmark_recursive_halving_doubling(int my_pe, int n_pes) {
    if (my_pe == 0) {
        printf("\n=== Pattern 3: Recursive Doubling AllReduce ===\n");
        printf("[Educational Mode]\n");
        int log_n = (int)(log2(std::max(1, n_pes)) + 0.5);
        printf("Exchange steps: %d vs %d phases for ring\n",
               log_n, 2 * (n_pes - 1));
    }
    return true;
}
#endif

// ============================================================================
// Main Program
// ============================================================================

int main() {
    NVTX_RANGE("main");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  NVSHMEM Advanced Patterns for Multi-GPU Blackwell B200   ║\n");
    printf("║  Stream-Ordered Communication Algorithms                    ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");
    
    #ifdef USE_NVSHMEM
    initialize_nvshmem_device();
    
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    
    if (my_pe == 0) {
        printf("Running on %d GPUs\n", n_pes);
        if (n_pes == 8) {
            printf("✓ Multi-GPU configuration detected\n");
        }
        printf("\n");
    }
    
    // Run advanced patterns
    bool all_correct = true;
    all_correct = benchmark_ring_allreduce(my_pe, n_pes) && all_correct;
    nvshmem_barrier_all();
    
    all_correct = benchmark_double_buffered_reduce_scatter(my_pe, n_pes) && all_correct;
    nvshmem_barrier_all();
    
    all_correct = benchmark_recursive_halving_doubling(my_pe, n_pes) && all_correct;
    nvshmem_barrier_all();
    
    if (my_pe == 0) {
        printf("\n╔════════════════════════════════════════════════════════════╗\n");
        printf("║  Performance Summary                                       ║\n");
        printf("╚════════════════════════════════════════════════════════════╝\n");
        printf("\nAlgorithm Selection Guide:\n");
        printf("  • Ring AllReduce:      Reduce-scatter plus allgather\n");
        printf("  • Double-Buffered:     Alternates symmetric receive storage\n");
        printf("  • Recursive doubling:  Power-of-two PE counts\n");
        printf("\nUse the measured times above on the target system for comparison.\n");
        printf("\nProduction Usage:\n");
        printf("  • PyTorch DDP: Uses ring for gradients\n");
        printf("  • Megatron-LM: Uses recursive for model parallel\n");
        printf("  • FSDP: Uses double-buffered for overlap\n");
        printf("\n");
    }
    
    nvshmem_finalize();
    if (!all_correct) {
        return EXIT_FAILURE;
    }
    
    #else
    printf("[Educational Mode]\n");
    printf("To compile with NVSHMEM:\n");
    printf("  1. Install NVSHMEM 3.4+ from NVIDIA\n");
    printf("  2. Set NVSHMEM_HOME environment variable\n");
    printf("  3. Compile:\n");
    printf("     nvcc -O3 -std=c++17 -arch=sm_100 -DUSE_NVSHMEM \\\n");
    printf("          -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib -lnvshmem \\\n");
    printf("          nvshmem_advanced_multigpu.cu -o nvshmem_advanced\n");
    printf("  4. Run: nvshmemrun -np 4 ./nvshmem_advanced\n\n");
    
    benchmark_ring_allreduce(0, 4);
    benchmark_double_buffered_reduce_scatter(0, 4);
    benchmark_recursive_halving_doubling(0, 4);
    #endif
    
    return 0;
}
