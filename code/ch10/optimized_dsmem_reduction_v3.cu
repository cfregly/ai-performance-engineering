// optimized_dsmem_reduction_v3.cu - WORKING DSMEM reduction for B200
// 
// KEY FIXES for B200/CUDA 13.0:
// 1. NO __cluster_dims__ attribute (conflicts with runtime cluster dims)
// 2. STATIC shared memory (dynamic extern fails on B200)
// 3. cudaLaunchKernelExC with void* args[] (not typed parameters)
// 4. Final cluster.sync() before exit
//
// Based on working optimized_cluster_group.cu pattern

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

constexpr int BLOCK_SIZE = 256;
constexpr int CLUSTER_SIZE = 2;  // Keep small for reliability
// 256 KB/block (was 4096 = 16 KB): amortizes the per-block cluster.sync + DSMEM
// peer-read over far more streamed bytes, and the long grid-stride load loop gives
// each thread many in-flight loads (MLP from ILP) so HBM BW rises even as the block
// count falls. Same GB300 knee as the warp_specialized sibling (4096->65536 lifts
// it 54.5% -> ~80% HBM SoL).
constexpr int ELEMENTS_PER_BLOCK = 65536;

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

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// NO __cluster_dims__ attribute - use runtime specification only
__global__ __launch_bounds__(BLOCK_SIZE, 1)
void dsmem_reduction_kernel_v3(
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
    
    // STATIC shared memory (required for B200 DSMEM)
    __shared__ float smem_reduce[32];
    __shared__ float smem_result[1];  // Single result per block
    
    const int cluster_offset = cluster_id * elements_per_cluster;
    const int block_offset = cluster_offset + cluster_rank * ELEMENTS_PER_BLOCK;
    
    // Step 1: Each thread accumulates its portion
    float local_sum = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < ELEMENTS_PER_BLOCK; i += BLOCK_SIZE) {
        int global_idx = block_offset + i;
        if (global_idx < N) {
            local_sum += input[global_idx];
        }
    }
    
    // Step 2: Warp-level reduction
    local_sum = warp_reduce_sum(local_sum);
    
    // Step 3: Write warp results to shared memory
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    if (lane_id == 0) {
        smem_reduce[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Step 4: First warp reduces all warp results
    if (warp_id == 0) {
        float val = (lane_id < BLOCK_SIZE / 32) ? smem_reduce[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            smem_result[0] = val;
        }
    }
    __syncthreads();
    
    // Step 5: Cluster sync BEFORE DSMEM access
    cluster.sync();
    
    // Step 6: Block 0 reads all block results via DSMEM
    if (cluster_rank == 0) {
        float cluster_sum = smem_result[0];  // Own result
        
        // Read from peer CTAs via DSMEM
        for (int peer = 1; peer < CLUSTER_SIZE; ++peer) {
            float* peer_result = cluster.map_shared_rank(smem_result, peer);
            cluster_sum += peer_result[0];
        }
        
        // Write final result
        if (tid == 0) {
            output[cluster_id] = cluster_sum;
        }
    }
    
    // Step 7: Final cluster sync for clean exit (REQUIRED!)
    cluster.sync();
    
#else
    // Fallback for older architectures
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;
    
    __shared__ float smem_reduce[32];
    
    float local_sum = 0.0f;
    for (int i = tid; i < ELEMENTS_PER_BLOCK; i += BLOCK_SIZE) {
        int global_idx = block_offset + i;
        if (global_idx < N) {
            local_sum += input[global_idx];
        }
    }
    
    local_sum = warp_reduce_sum(local_sum);
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    if (lane_id == 0) smem_reduce[warp_id] = local_sum;
    __syncthreads();
    
    if (warp_id == 0) {
        float val = (lane_id < BLOCK_SIZE / 32) ? smem_reduce[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            atomicAdd(&output[blockIdx.x / CLUSTER_SIZE], val);
        }
    }
#endif
}

int main() {
    NVTX_RANGE("main");
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("DSMEM Cluster Reduction v3 (B200 Compatible)\n");
    printf("=============================================\n");
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    
    const bool has_clusters = prop.major >= 9;
    if (!has_clusters) {
        printf("SKIP: DSMEM requires SM 9.0+ (current: SM %d.%d)\n", prop.major, prop.minor);
        return 0;  // Skip gracefully, no fallback
    }
    
    // Problem size
    const int N = 64 * 1024 * 1024;
    const int elements_per_cluster = ELEMENTS_PER_BLOCK * CLUSTER_SIZE;
    const int num_clusters = (N + elements_per_cluster - 1) / elements_per_cluster;
    const int num_blocks = num_clusters * CLUSTER_SIZE;
    
    printf("\nProblem Size:\n");
    printf("  Elements: %d (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("  Clusters: %d (size=%d)\n", num_clusters, CLUSTER_SIZE);
    printf("  Blocks: %d\n\n", num_blocks);
    
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
    
    // Setup cluster launch config
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(num_blocks, 1, 1);
    config.blockDim = dim3(BLOCK_SIZE, 1, 1);
    config.dynamicSmemBytes = 0;  // Static shared memory only!
    config.stream = 0;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_SIZE;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.numAttrs = 1;
    config.attrs = attrs;
    
    // Prepare arguments (void* array for cudaLaunchKernelExC)
    int N_param = N;
    int elements_param = elements_per_cluster;
    void* args[] = {&d_input, &d_output, &N_param, &elements_param};
    
    printf("Launching with cudaLaunchKernelExC (void* args[])...\n");
    
    // Warmup
    const int warmup = env_or_default("AISP_CUDA_BINARY_PROFILE_WARMUP", 5);
    for (int i = 0; i < warmup; ++i) {
        NVTX_RANGE("warmup");
        CUDA_CHECK(cudaLaunchKernelExC(&config, (void*)dsmem_reduction_kernel_v3, args));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    const int iterations = env_or_default("AISP_CUDA_BINARY_PROFILE_ITERATIONS", 50);
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        NVTX_RANGE("iteration");
        CUDA_CHECK(cudaLaunchKernelExC(&config, (void*)dsmem_reduction_kernel_v3, args));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    
    // Verify
    std::vector<float> h_output(num_clusters);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_clusters * sizeof(float), cudaMemcpyDeviceToHost));
    float total = std::accumulate(h_output.begin(), h_output.end(), 0.0f);
    
    printf("\nResults:\n");
    printf("  Time: %.3f ms\n", avg_ms);
    printf("  Sum: %.0f (expected: %d) %s\n", total, N, 
           (fabs(total - N) < 0.01f * N) ? "✓" : "✗");
    printf("  Bandwidth: %.2f GB/s\n", (N * sizeof(float)) / (avg_ms * 1e6));

    VERIFY_PRINT_CHECKSUM(total);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}
