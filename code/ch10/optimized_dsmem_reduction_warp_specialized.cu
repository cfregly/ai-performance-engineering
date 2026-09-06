// optimized_dsmem_reduction_warp_specialized.cu - Warp-Specialized DSMEM Reduction (Ch10)
//
// CHAPTER 10 CONTEXT: "Tensor Core Pipelines & Cluster Features"
// 
// KEY PATTERN: Warp specialization with DSMEM for maximum throughput
//
// OPTIMIZATIONS:
//   1. Warp specialization - only warp 0 handles cluster communication
//   2. Vectorized float4 loads for 4x bandwidth efficiency
//   3. Four CTAs per cluster
//   4. One DSMEM atomic per CTA; the leader writes the final cluster output
//
// WHY WARP SPECIALIZATION HELPS:
//   - Dedicated warps for communication vs compute
//   - Better overlap of work phases
//   - Reduced contention on DSMEM access

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

constexpr int BLOCK_SIZE = 256;
constexpr int CLUSTER_SIZE = 4;
// Each block streams 65536 floats (256 KB) before the block-reduce + cluster.sync
// + DSMEM atomic. At the original 4096 (16 KB/block) the per-block sync/atomic
// overhead was not amortized: ncu showed 66% DRAM at 91% occupancy (read-only,
// should approach a copy's ~90%). Streaming more per block amortizes the fixed
// sync cost and the long grid-stride loop gives each thread many in-flight loads
// (MLP from ILP), so BW rises even as the block count falls. Measured GB300 knee:
// 4096->16384->32768->65536 = 5402->6389->6613->6726 GB/s (67.5%->84.0% HBM SoL);
// 131072 underfills back to 6529. 65536 is the peak.
constexpr int ELEMENTS_PER_BLOCK = 65536;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;

namespace {
int env_or_default(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return fallback;
    }
    char* end = nullptr;
    long parsed = std::strtol(raw, &end, 10);
    if (end == raw || (end && *end != '\0') || parsed < 0 || parsed > 1000000) {
        return fallback;
    }
    return static_cast<int>(parsed);
}
}  // namespace

//============================================================================
// Warp reduction with shuffle
//============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

//============================================================================
// Block reduction
//============================================================================

__device__ float block_reduce_sum_v2(float val, float* smem) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    val = warp_reduce_sum(val);
    
    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane_id < WARPS_PER_BLOCK) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

//============================================================================
// Warp-Specialized DSMEM Cluster Reduction - NO FALLBACK
//============================================================================

__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1)
__launch_bounds__(BLOCK_SIZE, 1)
void dsmem_warp_specialized_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int elements_per_cluster
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cg::thread_block block = cg::this_thread_block();
    cg::cluster_group cluster = cg::this_cluster();
    
    const int cluster_id = blockIdx.x / CLUSTER_SIZE;
    const int cluster_rank = cluster.block_rank();
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory
    __shared__ float smem_reduce[WARPS_PER_BLOCK];
    __shared__ float smem_cluster_sum;  // Single accumulator for atomic aggregation
    
    // Global offset
    const int cluster_offset = cluster_id * elements_per_cluster;
    const int block_offset = cluster_offset + cluster_rank * ELEMENTS_PER_BLOCK;
    
    // Initialize cluster accumulator (only leader does this)
    if (cluster_rank == 0 && tid == 0) {
        smem_cluster_sum = 0.0f;
    }
    cluster.sync();
    
    //========================================================================
    // STEP 1: Vectorized load and local reduction
    //========================================================================
    float local_sum = 0.0f;
    
    // Use float4 for vectorized loads
    const float4* input4 = reinterpret_cast<const float4*>(input + block_offset);
    const int vec_elements = ELEMENTS_PER_BLOCK / 4;
    
    #pragma unroll 4
    for (int i = tid; i < vec_elements; i += BLOCK_SIZE) {
        int global_vec_idx = (block_offset / 4) + i;
        const int base = global_vec_idx * 4;
        if (base + 3 < N) {
            const float4 v = input4[i];
            local_sum += v.x + v.y + v.z + v.w;
        } else {
            for (int lane = 0; lane < 4 && base + lane < N; ++lane) {
                local_sum += input[base + lane];
            }
        }
    }
    
    // Block-level reduction
    float block_sum = block_reduce_sum_v2(local_sum, smem_reduce);
    
    //========================================================================
    // STEP 2: WARP SPECIALIZED cluster aggregation via DSMEM atomic
    // Only warp 0 participates in cluster communication
    //========================================================================
    if (warp_id == 0 && lane_id == 0) {
        // Get pointer to leader's smem via DSMEM
        float* leader_sum = cluster.map_shared_rank(&smem_cluster_sum, 0);
        atomicAdd(leader_sum, block_sum);
    }
    
    // Cluster sync
    cluster.sync();
    
    // Rank 0 writes final result to global memory
    if (cluster_rank == 0 && tid == 0) {
        output[cluster_id] = smem_cluster_sum;
    }
    
#else
    // Fallback for older architectures - simple block reduction with atomics
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;
    
    __shared__ float smem_reduce[32];
    
    // Vectorized loads
    float local_sum = 0.0f;
    for (int i = tid; i < ELEMENTS_PER_BLOCK; i += BLOCK_SIZE) {
        int global_idx = block_offset + i;
        if (global_idx < N) {
            local_sum += input[global_idx];
        }
    }
    
    float block_sum = block_reduce_sum_v2(local_sum, smem_reduce);
    
    if (tid == 0) {
        atomicAdd(&output[blockIdx.x / CLUSTER_SIZE], block_sum);
    }
#endif
}

//============================================================================
// Main
//============================================================================

int main() {
    NVTX_RANGE("main");
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Warp-Specialized DSMEM Reduction (Chapter 10 - NO FALLBACK)\n");
    printf("============================================================\n");
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    
    // FAIL FAST
    if (prop.major < 9) {
        fprintf(stderr, "\nERROR: DSMEM requires SM 9.0+ (Hopper/Blackwell)\n");
        fprintf(stderr, "Use optimized_atomic_reduction.cu instead\n");
        return 1;
    }
    
    int cluster_supported = 0;
    cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch, 0);
    if (!cluster_supported) {
        fprintf(stderr, "\nERROR: Device does not support cluster launch\n");
        return 1;
    }
    printf("Cluster launch: SUPPORTED\n");
    
    // Problem size
    const int N = 64 * 1024 * 1024;  // Match baseline workload (64M elements)
    const int elements_per_cluster = ELEMENTS_PER_BLOCK * CLUSTER_SIZE;
    const int num_clusters = (N + elements_per_cluster - 1) / elements_per_cluster;
    const int num_blocks = num_clusters * CLUSTER_SIZE;
    
    printf("\nProblem: %d elements (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("Clusters: %d (%d CTAs each, vectorized float4)\n\n", num_clusters, CLUSTER_SIZE);
    
    // Allocate
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_clusters * sizeof(float)));
    
    // Initialize
    std::vector<float> h_input(N, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Configure cluster launch
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(num_blocks, 1, 1);
    config.blockDim = dim3(BLOCK_SIZE, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = 0;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_SIZE;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.numAttrs = 1;
    config.attrs = attrs;
    
    const int N_param = N;
    const int elements_param = elements_per_cluster;
    
    const int warmup = env_or_default("AISP_CUDA_BINARY_PROFILE_WARMUP", 10);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        NVTX_RANGE("warmup");
        CUDA_CHECK(cudaLaunchKernelEx(&config,
                                       dsmem_warp_specialized_reduction_kernel,
                                       d_input, d_output, N_param, elements_param));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    const int iterations = env_or_default("AISP_CUDA_BINARY_PROFILE_ITERATIONS", 100);
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        NVTX_RANGE("iteration");
        CUDA_CHECK(cudaLaunchKernelEx(&config,
                                       dsmem_warp_specialized_reduction_kernel,
                                       d_input, d_output, N_param, elements_param));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    float bandwidth_gbps = (N * sizeof(float) / 1e9) / (avg_ms / 1000.0);
    
    // Verify
    std::vector<float> h_output(num_clusters);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_clusters * sizeof(float), cudaMemcpyDeviceToHost));
    float total = std::accumulate(h_output.begin(), h_output.end(), 0.0f);
    
    printf("Warp-Specialized DSMEM Reduction:\n");
    printf("  Time: %.3f ms\n", avg_ms);
    printf("  Bandwidth: %.1f GB/s\n", bandwidth_gbps);
    printf("  Sum: %.0f (expected: %d) - %s\n", total, N,
           (abs(total - N) < 1000) ? "PASS" : "FAIL");
    printf("\nOptimizations:\n");
    printf("  - Warp specialization (only warp 0 does cluster work)\n");
    printf("  - Vectorized float4 loads\n");
    printf("  - 4-CTA cluster\n");
    printf("  - One DSMEM atomic per CTA, one final output writer per cluster\n");

    VERIFY_PRINT_CHECKSUM(total);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}
