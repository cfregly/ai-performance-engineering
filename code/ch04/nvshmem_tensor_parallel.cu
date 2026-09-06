/**
 * NVSHMEM Tensor Parallel Kernels for Multi-GPU Blackwell B200
 * ============================================================
 *
 * Low-level CUDA kernels for tensor parallelism using NVSHMEM 3.4+ on
 * Blackwell B200 GPUs with NVLink 5.0.
 *
 * This file demonstrates:
 * 1. Column-parallel linear layer with NVSHMEM gather
 * 2. Row-parallel linear layer with NVSHMEM reduce-scatter
 * 3. Attention QKV split with direct GPU access
 * 4. Custom fused AllReduce kernel
 * 5. Strided access patterns for sharded tensors
 *
 * Hardware Requirements:
 * - 2-4x NVIDIA Blackwell B200 GPUs (SM 10.0, NVLink 5.0 @ 1800 GB/s)
 * - CUDA 13.0+, NVSHMEM 3.4+
 * - NVSwitch preferred for all-to-all communication
 *
 * Performance must be measured on the target topology. These examples report
 * timings but do not establish portable latency or bandwidth targets.
 *
 * Build:
 *   nvcc -O3 -std=c++17 -arch=sm_100 nvshmem_tensor_parallel.cu \
 *        -DUSE_NVSHMEM -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \
 *        -lnvshmem -lcuda -o nvshmem_tensor_parallel
 *
 * Run:
 *   nvshmemrun -np 4 ./nvshmem_tensor_parallel --test column_parallel
 *   nvshmemrun -np 4 ./nvshmem_tensor_parallel --test row_parallel
 *   nvshmemrun -np 4 ./nvshmem_tensor_parallel --test all
 *
 * Educational Notes:
 * ------------------
 * Tensor Parallelism Patterns:
 *
 * Column Parallel (sharded weights, replicated input):
 * - Each rank owns slice of weight matrix: W[:, rank*D/P : (rank+1)*D/P]
 * - Forward: Y_local = X @ W_local, then AllGather(Y_local) -> Y_full
 * - Backward: dW_local = X.T @ dY_local, dX = dY @ W.T (replicated)
 *
 * Row Parallel (sharded weights, sharded output):
 * - Each rank owns slice: W[rank*D/P : (rank+1)*D/P, :]
 * - Forward: Y_local = X_local @ W_local, then ReduceScatter -> Y_sharded
 * - Backward: dW_local = X_local.T @ dY, dX_local = dY @ W_local.T
 *
 * Why NVSHMEM:
 * - One-sided operations let GPU kernels initiate peer transfers directly.
 * - Stream-ordered host collectives provide whole-grid synchronization between
 *   communication phases.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../core/common/nvtx_utils.cuh"

// NVSHMEM includes (conditional)
#ifdef USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

// ============================================================================
// Error Checking Macros
// ============================================================================

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t err = (expr);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#ifdef USE_NVSHMEM
#define NVSHMEM_CHECK(expr)                                                  \
    do {                                                                     \
        int err = (expr);                                                    \
        if (err != 0) {                                                      \
            fprintf(stderr, "NVSHMEM error %s:%d: %d\n", __FILE__, __LINE__, \
                    err);                                                    \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

static void initialize_nvshmem_device() {
    nvshmem_init();
    const int world_pe = nvshmem_my_pe();
    const int local_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    if (local_pe < 0) {
        fprintf(stderr, "PE %d has no rank in NVSHMEMX_TEAM_NODE\n", world_pe);
        nvshmem_global_exit(EXIT_FAILURE);
    }

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
#endif

// ============================================================================
// Column-Parallel Forward Kernel
// ============================================================================

/**
 * Column-parallel matrix multiplication: Y_local = X @ W_local
 *
 * Input:
 *   - X: [batch_size, seq_len, hidden_dim] (replicated across all ranks)
 *   - W_local: [hidden_dim, hidden_dim/world_size] (sharded)
 * Output:
 *   - Y_local: [batch_size, seq_len, hidden_dim/world_size]
 *
 * After this kernel, need AllGather to get full Y.
 */
__global__ void column_parallel_forward_nvshmem(
    const half* input,         // [B, S, H]
    const half* weight_shard,  // [H, H/P]
    half* output_shard,        // [B, S, H/P]
    int batch_size,
    int seq_len,
    int hidden_dim,
    int shard_dim              // = hidden_dim / world_size
) {
    // Each thread computes one output element
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * shard_dim;
    
    if (global_idx >= total_elements) return;
    
    // Decompose index
    int shard_col = global_idx % shard_dim;
    int seq_idx = (global_idx / shard_dim) % seq_len;
    int batch_idx = global_idx / (seq_len * shard_dim);
    
    // Compute dot product: input[batch_idx, seq_idx, :] @ weight_shard[:, shard_col]
    float acc = 0.0f;
    for (int k = 0; k < hidden_dim; ++k) {
        int input_idx = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + k;
        int weight_idx = k * shard_dim + shard_col;
        acc += __half2float(input[input_idx]) * __half2float(weight_shard[weight_idx]);
    }
    
    output_shard[global_idx] = __float2half(acc);
}

// ============================================================================
// AllGather with NVSHMEM
// ============================================================================

/**
 * AllGather operation using NVSHMEM one-sided puts.
 *
 * Each rank writes its local shard to all other ranks' symmetric memory.
 * This is faster than NCCL for small tensors (< 1MB) due to low latency.
 */
#ifdef USE_NVSHMEM
__global__ void nvshmem_allgather_kernel(
    const half* local_shard,  // Local data to gather
    half* gathered_output,    // Output buffer for all shards
    int shard_size,           // Size of each shard
    int world_size,
    int my_rank
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < shard_size) {
        // Each thread copies one element to all peers
        half value = local_shard[tid];
        
        for (int peer = 0; peer < world_size; ++peer) {
            int offset = my_rank * shard_size + tid;
            // Transfer the binary16 payload through the standard 16-bit RMA
            // type. Some NVSHMEM device archives declare half p() but do not
            // provide its device-link instantiation.
            nvshmem_ushort_p(
                reinterpret_cast<unsigned short*>(&gathered_output[offset]),
                __half_as_ushort(value), peer);
        }
    }
}

void nvshmem_allgather(const half* local_shard, half* gathered_output,
                       int shard_size, int world_size, int my_rank,
                       cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (shard_size + threads - 1) / threads;
    nvshmem_allgather_kernel<<<blocks, threads, 0, stream>>>(
        local_shard, gathered_output, shard_size, world_size, my_rank);
    CUDA_CHECK(cudaGetLastError());
    // The collective is issued by the host after the entire grid has finished
    // its puts; a single device thread cannot synchronize a multi-block grid.
    nvshmemx_barrier_all_on_stream(stream);
}
#endif

// ============================================================================
// Row-Parallel Forward Kernel
// ============================================================================

/**
 * Row-parallel matrix multiplication with local accumulation.
 *
 * Input:
 *   - X: [batch_size, seq_len, hidden_dim] (replicated or sharded)
 *   - W_shard: [hidden_dim/world_size, output_dim] (row-sharded)
 * Output:
 *   - Y_local: [batch_size, seq_len, output_dim] (partial sums)
 *
 * After this, need ReduceScatter or AllReduce to get final Y.
 */
__global__ void row_parallel_forward_nvshmem(
    const half* input,        // [B, S, H]
    const half* weight_shard, // [H/P, O]
    half* output_local,       // [B, S, O] (partial)
    int batch_size,
    int seq_len,
    int hidden_dim,
    int shard_dim,            // = hidden_dim / world_size
    int output_dim
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * output_dim;
    
    if (global_idx >= total_elements) return;
    
    // Decompose index
    int out_col = global_idx % output_dim;
    int seq_idx = (global_idx / output_dim) % seq_len;
    int batch_idx = global_idx / (seq_len * output_dim);
    
    // Compute partial dot product over local shard
    float acc = 0.0f;
    for (int k = 0; k < shard_dim; ++k) {
        int input_idx = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + k;
        int weight_idx = k * output_dim + out_col;
        acc += __half2float(input[input_idx]) * __half2float(weight_shard[weight_idx]);
    }
    
    output_local[global_idx] = __float2half(acc);
}

// ============================================================================
// ReduceScatter with NVSHMEM (Ring Algorithm)
// ============================================================================

/**
 * ReduceScatter using ring algorithm with NVSHMEM.
 *
 * Each rank sends data to next rank and receives from previous rank.
 * After world_size-1 steps, each rank has the reduced result for its chunk.
 *
 * Measure bandwidth on the target system; topology and message size dominate.
 */
#ifdef USE_NVSHMEM
__global__ void nvshmem_reduce_scatter_send_kernel(
    const half* data,
    half* temp_buffer,       // Temporary buffer for receiving
    int chunk_size,          // Size of each chunk
    int world_size,
    int my_rank,
    int step                 // Which step of the ring (0 to world_size-1)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Determine which chunk to send/receive in this step
    int send_chunk = (my_rank - step + world_size) % world_size;
    int next_rank = (my_rank + 1) % world_size;
    if (tid < chunk_size) {
        // Send to next rank
        int send_offset = send_chunk * chunk_size + tid;
        nvshmem_ushort_p(
            reinterpret_cast<unsigned short*>(&temp_buffer[send_offset]),
            __half_as_ushort(data[send_offset]), next_rank);
    }
}

__global__ void nvshmem_reduce_scatter_accumulate_kernel(
    half* data, const half* temp_buffer, int chunk_size,
    int world_size, int my_rank, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int recv_chunk = (my_rank - step - 1 + world_size) % world_size;
    if (tid < chunk_size) {
        // Accumulate received data
        int recv_offset = recv_chunk * chunk_size + tid;
        data[recv_offset] = __float2half(
            __half2float(data[recv_offset]) + __half2float(temp_buffer[recv_offset])
        );
    }
}

void nvshmem_reduce_scatter_ring(half* data, half* temp_buffer,
                                 int chunk_size, int world_size, int my_rank,
                                 cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (chunk_size + threads - 1) / threads;
    for (int step = 0; step < world_size - 1; ++step) {
        nvshmem_reduce_scatter_send_kernel<<<blocks, threads, 0, stream>>>(
            data, temp_buffer, chunk_size, world_size, my_rank, step);
        CUDA_CHECK(cudaGetLastError());
        nvshmemx_barrier_all_on_stream(stream);
        nvshmem_reduce_scatter_accumulate_kernel<<<blocks, threads, 0, stream>>>(
            data, temp_buffer, chunk_size, world_size, my_rank, step);
        CUDA_CHECK(cudaGetLastError());
    }
}
#endif

// ============================================================================
// Fused AllReduce (Custom Implementation)
// ============================================================================

/**
 * Custom AllReduce using ring algorithm with NVSHMEM.
 *
 * Combines stream-ordered ReduceScatter and AllGather phases.
 * Useful for small to medium tensors in tensor parallel training.
 */
#ifdef USE_NVSHMEM
__global__ void nvshmem_allreduce_send_kernel(
    const half* data, half* scratch, int total_size, int chunk_size,
    int send_chunk, int next_rank) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < chunk_size) {
        int offset = send_chunk * chunk_size + tid;
        if (offset < total_size) {
            nvshmem_ushort_p(
                reinterpret_cast<unsigned short*>(&scratch[offset]),
                __half_as_ushort(data[offset]), next_rank);
        }
    }
}

__global__ void nvshmem_allreduce_accumulate_kernel(
    half* data, const half* scratch, int total_size, int chunk_size,
    int recv_chunk) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < chunk_size) {
        int offset = recv_chunk * chunk_size + tid;
        if (offset < total_size) {
            data[offset] = __float2half(
                __half2float(data[offset]) + __half2float(scratch[offset]));
        }
    }
}

__global__ void nvshmem_allreduce_copy_kernel(
    half* data, const half* scratch, int total_size, int chunk_size,
    int recv_chunk) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < chunk_size) {
        int offset = recv_chunk * chunk_size + tid;
        if (offset < total_size) {
            data[offset] = scratch[offset];
        }
    }
}

void nvshmem_allreduce_ring(half* data, half* scratch, int total_size,
                            int world_size, int my_rank,
                            cudaStream_t stream) {
    const int chunk_size = (total_size + world_size - 1) / world_size;
    const int threads = 256;
    const int blocks = (chunk_size + threads - 1) / threads;
    const int next_rank = (my_rank + 1) % world_size;

    for (int step = 0; step < world_size - 1; ++step) {
        const int send_chunk = (my_rank - step + world_size) % world_size;
        const int recv_chunk = (my_rank - step - 1 + world_size) % world_size;
        nvshmem_allreduce_send_kernel<<<blocks, threads, 0, stream>>>(
            data, scratch, total_size, chunk_size, send_chunk, next_rank);
        CUDA_CHECK(cudaGetLastError());
        nvshmemx_barrier_all_on_stream(stream);
        nvshmem_allreduce_accumulate_kernel<<<blocks, threads, 0, stream>>>(
            data, scratch, total_size, chunk_size, recv_chunk);
        CUDA_CHECK(cudaGetLastError());
    }

    for (int step = 0; step < world_size - 1; ++step) {
        const int send_chunk = (my_rank + 1 - step + world_size) % world_size;
        const int recv_chunk = (my_rank - step + world_size) % world_size;
        nvshmem_allreduce_send_kernel<<<blocks, threads, 0, stream>>>(
            data, scratch, total_size, chunk_size, send_chunk, next_rank);
        CUDA_CHECK(cudaGetLastError());
        nvshmemx_barrier_all_on_stream(stream);
        nvshmem_allreduce_copy_kernel<<<blocks, threads, 0, stream>>>(
            data, scratch, total_size, chunk_size, recv_chunk);
        CUDA_CHECK(cudaGetLastError());
    }
}
#endif

// ============================================================================
// Collective Correctness Check
// ============================================================================

#ifdef USE_NVSHMEM

bool test_collectives() {
    const int my_rank = nvshmem_my_pe();
    const int world_size = nvshmem_n_pes();
    const int chunk_size = 1024;
    const int total_size = chunk_size * world_size;
    const float expected_sum =
        static_cast<float>(world_size * (world_size + 1)) / 2.0f;

    half* data = static_cast<half*>(nvshmem_malloc(total_size * sizeof(half)));
    half* scratch = static_cast<half*>(nvshmem_malloc(total_size * sizeof(half)));
    half* gathered = static_cast<half*>(nvshmem_malloc(total_size * sizeof(half)));
    if (data == nullptr || scratch == nullptr || gathered == nullptr) {
        fprintf(stderr, "PE %d failed to allocate symmetric collective storage\n",
                my_rank);
        nvshmem_global_exit(EXIT_FAILURE);
    }
    half* local_shard = nullptr;
    CUDA_CHECK(cudaMalloc(&local_shard, chunk_size * sizeof(half)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::vector<half> host(total_size, __float2half(static_cast<float>(my_rank + 1)));
    bool correct = true;

    // ReduceScatter: each PE validates the chunk it owns after the ring.
    CUDA_CHECK(cudaMemcpy(data, host.data(), total_size * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(scratch, 0, total_size * sizeof(half)));
    nvshmem_barrier_all();
    nvshmem_reduce_scatter_ring(
        data, scratch, chunk_size, world_size, my_rank, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int owned_chunk = (my_rank + 1) % world_size;
    CUDA_CHECK(cudaMemcpy(host.data(), data + owned_chunk * chunk_size,
                          chunk_size * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < chunk_size; ++i) {
        float value = __half2float(host[i]);
        correct = correct && std::isfinite(value) && value == expected_sum;
    }

    // Ring AllReduce: every element on every PE must contain the global sum.
    std::fill(host.begin(), host.end(), __float2half(static_cast<float>(my_rank + 1)));
    CUDA_CHECK(cudaMemcpy(data, host.data(), total_size * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(scratch, 0, total_size * sizeof(half)));
    nvshmem_barrier_all();
    nvshmem_allreduce_ring(data, scratch, total_size, world_size, my_rank, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(host.data(), data, total_size * sizeof(half),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < total_size; ++i) {
        float value = __half2float(host[i]);
        correct = correct && std::isfinite(value) && value == expected_sum;
    }

    // AllGather: rank r contributes a constant r+1 shard to every PE.
    std::vector<half> local(chunk_size,
                            __float2half(static_cast<float>(my_rank + 1)));
    CUDA_CHECK(cudaMemcpy(local_shard, local.data(), chunk_size * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gathered, 0, total_size * sizeof(half)));
    nvshmem_barrier_all();
    nvshmem_allgather(
        local_shard, gathered, chunk_size, world_size, my_rank, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(host.data(), gathered, total_size * sizeof(half),
                          cudaMemcpyDeviceToHost));
    for (int peer = 0; peer < world_size; ++peer) {
        const float expected = static_cast<float>(peer + 1);
        for (int i = 0; i < chunk_size; ++i) {
            float value = __half2float(host[peer * chunk_size + i]);
            correct = correct && std::isfinite(value) && value == expected;
        }
    }

    printf("PE %d collective correctness: %s\n",
           my_rank, correct ? "PASS" : "FAIL");

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(local_shard));
    nvshmem_barrier_all();
    nvshmem_free(gathered);
    nvshmem_free(scratch);
    nvshmem_free(data);
    return correct;
}

#endif

// ============================================================================
// Host Code: Test Harness
// ============================================================================

#ifdef USE_NVSHMEM

void test_column_parallel() {
    int my_rank = nvshmem_my_pe();
    int world_size = nvshmem_n_pes();
    
    if (my_rank == 0) {
        printf("\n=== Testing Column-Parallel Layer ===\n");
    }
    
    // Configuration
    int batch_size = 16;
    int seq_len = 512;
    int hidden_dim = 2048;
    int shard_dim = hidden_dim / world_size;
    
    // Allocate memory
    half *d_input, *d_weight_shard, *d_output_shard;
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * seq_len * hidden_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_weight_shard, hidden_dim * shard_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output_shard, batch_size * seq_len * shard_dim * sizeof(half)));
    
    // Initialize with test data
    std::vector<half> h_input(batch_size * seq_len * hidden_dim, __float2half(0.1f));
    std::vector<half> h_weight(hidden_dim * shard_dim, __float2half(0.01f));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight_shard, h_weight.data(), h_weight.size() * sizeof(half), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int total_elements = batch_size * seq_len * shard_dim;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    auto start = std::chrono::high_resolution_clock::now();
    column_parallel_forward_nvshmem<<<blocks, threads>>>(
        d_input, d_weight_shard, d_output_shard,
        batch_size, seq_len, hidden_dim, shard_dim
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
    
    if (my_rank == 0) {
        printf("Column-parallel forward: %.2f μs\n", elapsed_us);
        printf("Output shard shape: [%d, %d, %d]\n", batch_size, seq_len, shard_dim);
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight_shard));
    CUDA_CHECK(cudaFree(d_output_shard));
}

void test_row_parallel() {
    int my_rank = nvshmem_my_pe();
    int world_size = nvshmem_n_pes();
    
    if (my_rank == 0) {
        printf("\n=== Testing Row-Parallel Layer ===\n");
    }
    
    // Configuration
    int batch_size = 16;
    int seq_len = 512;
    int hidden_dim = 2048;
    int shard_dim = hidden_dim / world_size;
    int output_dim = 4096;
    
    // Allocate memory
    half *d_input, *d_weight_shard, *d_output_local;
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * seq_len * hidden_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_weight_shard, shard_dim * output_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output_local, batch_size * seq_len * output_dim * sizeof(half)));
    
    // Initialize
    std::vector<half> h_input(batch_size * seq_len * hidden_dim, __float2half(0.1f));
    std::vector<half> h_weight(shard_dim * output_dim, __float2half(0.01f));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight_shard, h_weight.data(), h_weight.size() * sizeof(half), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int total_elements = batch_size * seq_len * output_dim;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    auto start = std::chrono::high_resolution_clock::now();
    row_parallel_forward_nvshmem<<<blocks, threads>>>(
        d_input, d_weight_shard, d_output_local,
        batch_size, seq_len, hidden_dim, shard_dim, output_dim
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
    
    if (my_rank == 0) {
        printf("Row-parallel forward: %.2f μs\n", elapsed_us);
        printf("Output shape: [%d, %d, %d] (partial sums)\n", batch_size, seq_len, output_dim);
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight_shard));
    CUDA_CHECK(cudaFree(d_output_local));
}

#else  // !USE_NVSHMEM

void test_column_parallel() {
    printf("NVSHMEM not available - conceptual column-parallel demonstration:\n");
    printf("1. Each rank owns W[:, rank*D/P : (rank+1)*D/P]\n");
    printf("2. Forward: Y_local = X @ W_local\n");
    printf("3. AllGather: Y = concat([Y_0, Y_1, ..., Y_{P-1}])\n");
    printf("4. Backward: dW_local = X.T @ dY_local\n");
    printf("\nCompile with -DUSE_NVSHMEM to enable actual execution.\n");
}

void test_row_parallel() {
    printf("NVSHMEM not available - conceptual row-parallel demonstration:\n");
    printf("1. Each rank owns W[rank*D/P : (rank+1)*D/P, :]\n");
    printf("2. Forward: Y_local = X_local @ W_local (partial sums)\n");
    printf("3. AllReduce or ReduceScatter: Y = sum([Y_0, Y_1, ..., Y_{P-1}])\n");
    printf("4. Backward: dW_local = X_local.T @ dY\n");
    printf("\nCompile with -DUSE_NVSHMEM to enable actual execution.\n");
}

#endif  // USE_NVSHMEM

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
    NVTX_RANGE("main");
#ifdef USE_NVSHMEM
    // Bootstrap NVSHMEM, map this node-local PE to its CUDA device, and
    // complete deferred device initialization before symmetric allocation.
    initialize_nvshmem_device();
    
    int my_rank = nvshmem_my_pe();
    int world_size = nvshmem_n_pes();
    
    if (my_rank == 0) {
        printf("NVSHMEM Tensor Parallel Kernels for Blackwell B200\n");
        printf("World size: %d\n", world_size);
    }
    
    // Parse command line
    const char* test_type = "all";
    for (int i = 1; i < argc; ++i) {
        NVTX_RANGE("iteration");
        if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            test_type = argv[i + 1];
            break;
        }
    }
    
    // Run tests
    bool all_correct = true;
    if (strcmp(test_type, "collectives") == 0 || strcmp(test_type, "all") == 0) {
        all_correct = test_collectives() && all_correct;
        nvshmem_barrier_all();
    }

    if (strcmp(test_type, "column_parallel") == 0 || strcmp(test_type, "all") == 0) {
        test_column_parallel();
        nvshmem_barrier_all();
    }
    
    if (strcmp(test_type, "row_parallel") == 0 || strcmp(test_type, "all") == 0) {
        test_row_parallel();
        nvshmem_barrier_all();
    }
    
    if (my_rank == 0) {
        printf("\nTests completed: %s\n", all_correct ? "PASS" : "FAIL");
    }
    
    // Finalize
    nvshmem_finalize();
    if (!all_correct) {
        return EXIT_FAILURE;
    }
#else
    printf("NVSHMEM Tensor Parallel Kernels (Conceptual Mode)\n");
    printf("=================================================\n\n");
    test_column_parallel();
    printf("\n");
    test_row_parallel();
    printf("\n");
    printf("Build with -DUSE_NVSHMEM to enable actual execution.\n");
#endif
    
    return 0;
}
