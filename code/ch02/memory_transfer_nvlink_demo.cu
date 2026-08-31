/**
 * NVLink-C2C CPU-GPU P2P Transfer Optimization for Blackwell
 * ===========================================================
 * 
 * Blackwell B200/B300 introduces NVLink-C2C (Chip-to-Chip) for
 * high-speed CPU-GPU interconnect:
 * - Up to 900 GB/s bandwidth (CPU ↔ GPU)
 * - Direct CPU memory access to GPU memory
 * - Better than PCIe 5.0 (128 GB/s)
 * - Grace-Blackwell Superchip optimized
 * 
 * This example demonstrates:
 * 1. CPU-GPU P2P transfers with NVLink-C2C
 * 2. Page migration hints for optimal placement
 * 3. Performance comparison vs PCIe
 * 4. Best practices for Grace-Blackwell
 * 
 * Requirements:
 * - Blackwell B200/B300 GPU
 * - Grace CPU (for NVLink-C2C) or PCIe 5.0
 * - CUDA 13.0+
 * 
 * Compile:
 *   nvcc -O3 -std=c++17 -arch=sm_100 memory_transfer_nvlink_demo.cu -o memory_transfer_nvlink_demo
 * 
 * Expected Performance:
 * - NVLink-C2C: ~900 GB/s (Grace-Blackwell)
 * - PCIe 5.0: ~128 GB/s (fallback)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include "../core/common/nvtx_utils.cuh"

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
// Detect NVLink-C2C Support
// ============================================================================

struct SystemInfo {
    bool has_nvlink_c2c;
    bool is_grace_blackwell;
    bool has_pcie_5;
    int num_gpus;
    size_t gpu_memory;
    int cpu_numa_nodes;
    bool is_b200_multigpu;
    bool has_nvswitch;
};

// Detect if running on Grace CPU (ARM architecture)
bool detect_grace_cpu() {
    #ifdef __linux__
    // Check for ARM architecture
    FILE* f = popen("uname -m", "r");
    if (f) {
        char arch[64];
        bool is_arm = false;
        if (fgets(arch, sizeof(arch), f)) {
            if (strstr(arch, "aarch64") || strstr(arch, "arm64")) {
                is_arm = true;
            }
        }
        pclose(f);
        if (is_arm) {
            // Further verify it's Grace Neoverse
            FILE* cpuinfo = fopen("/proc/cpuinfo", "r");
            if (cpuinfo) {
                char line[256];
                while (fgets(line, sizeof(line), cpuinfo)) {
                    NVTX_RANGE("verify");
                    if (strstr(line, "Neoverse") || strstr(line, "0xd40")) {
                        fclose(cpuinfo);
                        return true;
                    }
                }
                fclose(cpuinfo);
            }
            return true;  // At least ARM
        }
    }
    #endif
    return false;
}

SystemInfo detect_system_capabilities() {
    SystemInfo info = {false, false, false, 0, 0, 0, false, false};
    
    CUDA_CHECK(cudaGetDeviceCount(&info.num_gpus));
    
    if (info.num_gpus == 0) {
        printf("No CUDA devices found\n");
        return info;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    info.gpu_memory = prop.totalGlobalMem;
    // Compute capability identifies the GPU ISA family, not the host/GPU
    // product topology.  SM 10.x covers the datacenter Blackwell GPUs used by
    // GB200/GB300, while SM 12.x covers GB10/workstation-class Blackwell.  An
    // SM 12.x device must not be relabeled as a GB200/GB300 Superchip.
    bool is_datacenter_blackwell = (prop.major == 10);
    bool is_sm12_blackwell = (prop.major == 12);
    
    printf("=== System Capabilities ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("GPU Memory: %.2f GiB per GPU\n", info.gpu_memory / (1024.0 * 1024.0 * 1024.0));
    printf("Number of GPUs: %d\n", info.num_gpus);
    
    const bool is_b200 = prop.major == 10 && prop.minor == 0;
    const bool is_b300 = prop.major == 10 && prop.minor == 3;

    // Check for multi-GPU datacenter Blackwell configurations by compute
    // capability. CUDA reports capacity in bytes; using marketed decimal-GB
    // windows against a GiB conversion misclassifies both B200 and B300.
    if (info.num_gpus >= 2 && is_datacenter_blackwell) {
        double mem_gib = info.gpu_memory / (1024.0 * 1024.0 * 1024.0);
        if (is_b200) {
            info.is_b200_multigpu = true;
            printf("✓ B200 multi-GPU configuration detected\n");
            printf("  Total memory: %.2f TiB (%.1f GiB per GPU)\n",
                   info.num_gpus * mem_gib / 1024.0, mem_gib);
            printf("  Total SMs: %d (148 per GPU)\n", info.num_gpus * 148);
            printf("  NVLink 5.0: 1800 GB/s per GPU pair (bidirectional)\n");
        } else if (is_b300) {
            info.is_b200_multigpu = true;
            printf("✓ B300 multi-GPU configuration detected\n");
            printf("  Total memory: %.2f TiB (%.1f GiB per GPU)\n",
                   info.num_gpus * mem_gib / 1024.0, mem_gib);
            printf("  Total SMs: %d (148 per GPU)\n", info.num_gpus * 148);
            printf("  NVLink 5.0: 1800 GB/s per GPU pair (bidirectional)\n");
        }
    }
    
    // Check for Blackwell
    if (is_datacenter_blackwell) {
        printf("✓ Datacenter Blackwell GPU detected (GB200/GB300 GPU class)\n");
        
        // Detect Grace CPU
        bool has_grace_cpu = detect_grace_cpu();
        
        if (has_grace_cpu) {
            info.has_nvlink_c2c = true;
            info.is_grace_blackwell = true;
            printf("✓ Grace CPU detected (ARM Neoverse)\n");
            printf("✓ GB200/GB300 Grace-Blackwell Superchip\n");
            printf("  NVLink-C2C: 900 GB/s coherent CPU ↔ GPU bandwidth\n");
            printf("  Unified memory address space\n");
            printf("  Grace: 72 ARM cores, 480GB-1TB LPDDR5X\n");
        } else {
            // Check for NVLink-C2C via NVIDIA driver
            #ifdef __linux__
            FILE* f = fopen("/proc/driver/nvidia/gpus/0000:01:00.0/information", "r");
            if (f) {
                char line[256];
                while (fgets(line, sizeof(line), f)) {
                    NVTX_RANGE("iteration");
                    if (strstr(line, "NVLink-C2C")) {
                        info.has_nvlink_c2c = true;
                        break;
                    }
                }
                fclose(f);
            }
            #endif
            
            if (info.has_nvlink_c2c) {
                printf("✓ NVLink-C2C detected\n");
                printf("  Expected bandwidth: ~900 GB/s (CPU ↔ GPU)\n");
            } else {
                printf("⚠ PCIe connection (no NVLink-C2C)\n");
                printf("  Expected bandwidth: ~128 GB/s (PCIe 5.0)\n");
                info.has_pcie_5 = true;
            }
        }
        
        // Check for NVSwitch on multi-GPU systems.
        if (info.num_gpus >= 2) {
            #ifdef __linux__
            FILE* topo = popen("nvidia-smi topo -m 2>/dev/null | grep -i nvswitch", "r");
            if (topo) {
                char line[256];
                if (fgets(line, sizeof(line), topo)) {
                    info.has_nvswitch = true;
                    printf("✓ NVSwitch detected (optimal for dense multi-GPU topology)\n");
                }
                pclose(topo);
            }
            #endif
        }
    } else if (is_sm12_blackwell) {
        printf("⚠ SM 12.x Blackwell-family GPU detected\n");
        printf("  SM 12.x identifies GB10/workstation-class GPUs, not a GB200/GB300 Superchip\n");
        printf("  NVLink-C2C capability must be established from the platform topology\n");
    } else {
        printf("⚠ Not a Blackwell GPU - NVLink-C2C not available\n");
        info.has_pcie_5 = true;
    }
    
    printf("\n");
    return info;
}

// ============================================================================
// CPU-GPU Transfer Benchmarks
// ============================================================================

double benchmark_host_to_device(void* h_data, void* d_data, size_t size, 
                                 const char* method) {
    const int iterations = 100;
    
    // Warmup
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        NVTX_RANGE("transfer_sync:h2d");
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_ms = duration.count() / (double)iterations / 1000.0;
    double bandwidth_gbs = (size / (avg_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    
    printf("  %-25s: %7.2f ms, %7.2f GB/s\n", method, avg_ms, bandwidth_gbs);
    return bandwidth_gbs;
}

double benchmark_device_to_host(void* d_data, void* h_data, size_t size,
                                 const char* method) {
    const int iterations = 100;
    
    // Warmup
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        NVTX_RANGE("transfer_sync:d2h");
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_ms = duration.count() / (double)iterations / 1000.0;
    double bandwidth_gbs = (size / (avg_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    
    printf("  %-25s: %7.2f ms, %7.2f GB/s\n", method, avg_ms, bandwidth_gbs);
    return bandwidth_gbs;
}

// ============================================================================
// Page Migration Hints (CUDA 13 / Blackwell)
// ============================================================================

__global__ void touch_managed_pages(float* data, size_t elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) {
        data[idx] += 1.0f;
    }
}

void demonstrate_page_migration() {
    printf("\n=== Page Migration Hints (CUDA 13) ===\n");
    
    size_t size = 256 * 1024 * 1024;  // 256 MiB
    size_t elements = size / sizeof(float);
    dim3 block(256);
    dim3 grid((elements + block.x - 1) / block.x);
    
    // Allocate managed memory
    float* managed_data;
    CUDA_CHECK(cudaMallocManaged(&managed_data, size));
    
    // Initialize on CPU
    for (size_t i = 0; i < size / sizeof(float); i++) {
        NVTX_RANGE("setup");
        managed_data[i] = (float)i;
    }
    
    printf("Allocated %.2f MB managed memory\n", size / (1024.0 * 1024.0));
    
    // CUDA 13.0 API: Set up memory locations
    struct cudaMemLocation gpuLoc;
    gpuLoc.type = cudaMemLocationTypeDevice;
    gpuLoc.id = 0;
    
    struct cudaMemLocation cpuLoc;
    cpuLoc.type = cudaMemLocationTypeHost;
    cpuLoc.id = 0;
    
    // Strategy 1: No hints. The first GPU touch services page faults on demand.
    {
        auto start = std::chrono::high_resolution_clock::now();

        touch_managed_pages<<<grid, block>>>(managed_data, elements);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        printf("  On-demand first GPU touch:       %5ld ms\n", duration.count());
    }
    
    // Reset to CPU
    CUDA_CHECK(cudaMemPrefetchAsync(managed_data, size, cpuLoc, 0, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Strategy 2: Explicit prefetch followed by the same first-touch kernel.
    {
        auto start = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaMemPrefetchAsync(managed_data, size, gpuLoc, 0, 0));
        touch_managed_pages<<<grid, block>>>(managed_data, elements);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        printf("  Prefetch + first GPU touch:      %5ld ms\n",
               duration.count());
    }
    
    // Strategy 3: Advised location (CUDA 13 enhancement)
    {
        // Advise that this memory will be accessed mostly from GPU
        CUDA_CHECK(cudaMemAdvise(managed_data, size, 
                                 cudaMemAdviseSetPreferredLocation, gpuLoc));
        CUDA_CHECK(cudaMemAdvise(managed_data, size,
                                 cudaMemAdviseSetAccessedBy, gpuLoc));
        CUDA_CHECK(cudaMemAdvise(managed_data, size,
                                 cudaMemAdviseSetReadMostly, gpuLoc));
        
        // Reset to CPU
        CUDA_CHECK(cudaMemPrefetchAsync(managed_data, size, cpuLoc, 0, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto start = std::chrono::high_resolution_clock::now();
        
        CUDA_CHECK(cudaMemPrefetchAsync(managed_data, size, gpuLoc, 0, 0));
        touch_managed_pages<<<grid, block>>>(managed_data, elements);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        printf("  Advise + prefetch + GPU touch:   %5ld ms\n",
               duration.count());
    }
    
    CUDA_CHECK(cudaFree(managed_data));
    
    printf("\nBest Practice (Grace-Blackwell):\n");
    printf("1. Use cudaMemAdvise for frequently accessed data\n");
    printf("2. Prefetch before kernel launch\n");
    printf("3. NVLink-C2C enables seamless CPU-GPU memory\n");
}

// ============================================================================
// Multi-GPU P2P Bandwidth Benchmarks
// ============================================================================

void benchmark_multigpu_p2p_bandwidth(const SystemInfo& info) {
    if (!info.is_b200_multigpu) {
        printf("\n⚠ Multi-GPU P2P benchmark requires B200/B300-class GPUs (found %d GPU(s))\n",
               info.num_gpus);
        return;
    }
    
    const int max_gpus = info.num_gpus;
    printf("\n=== Multi-GPU P2P Bandwidth Benchmark ===\n");
    printf("Testing NVLink 5.0 bandwidth between all GPU pairs\n\n");
    
    const size_t transfer_size = 256 * 1024 * 1024;  // 256 MB
    const int iterations = 100;
    
    // Enable P2P access between all GPU pairs
    for (int i = 0; i < max_gpus; i++) {
        NVTX_RANGE("iteration");
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < max_gpus; j++) {
            NVTX_RANGE("iteration");
            if (i != j) {
                int can_access;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) {
                    cudaError_t status = cudaDeviceEnablePeerAccess(j, 0);
                    if (status == cudaErrorPeerAccessAlreadyEnabled) {
                        cudaGetLastError();  // clear sticky error from previous enable
                    } else if (status != cudaSuccess) {
                        fprintf(stderr,
                                "CUDA error enabling peer access %d->%d: %s\n",
                                i, j, cudaGetErrorString(status));
                        exit(EXIT_FAILURE);
                    }
                    if (i < j) {
                        int native_atomics = 0;
                        int perf_rank = 0;
                        CUDA_CHECK(cudaDeviceGetP2PAttribute(
                            &native_atomics, cudaDevP2PAttrNativeAtomicSupported, i, j));
                        CUDA_CHECK(cudaDeviceGetP2PAttribute(
                            &perf_rank, cudaDevP2PAttrPerformanceRank, i, j));
                        printf("P2P %d ↔ %d: nativeAtomics=%d, perfRank=%d (lower is better)\n",
                               i, j, native_atomics, perf_rank);
                    }
                }
            }
        }
    }
    
    // Allocate memory on each GPU
    std::vector<float*> d_data(max_gpus, nullptr);
    for (int i = 0; i < max_gpus; i++) {
        NVTX_RANGE("setup");
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&d_data[i], transfer_size));
    }
    
    // Benchmark all pairs
    printf("GPU Pair | Bandwidth (GB/s) | Latency (μs)\n");
    printf("---------|------------------|-------------\n");
    
    double total_bandwidth = 0.0;
    int pair_count = 0;
    
    for (int src = 0; src < max_gpus; src++) {
        NVTX_RANGE("warmup");
        for (int dst = src + 1; dst < max_gpus; dst++) {
            NVTX_RANGE("warmup");
            CUDA_CHECK(cudaSetDevice(src));
            
            // Warmup
            CUDA_CHECK(cudaMemcpyPeer(d_data[dst], dst, d_data[src], src, transfer_size));
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Benchmark
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; i++) {
                NVTX_RANGE("transfer_sync");
                CUDA_CHECK(cudaMemcpyPeer(d_data[dst], dst, d_data[src], src, transfer_size));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double avg_us = duration.count() / (double)iterations;
            double bandwidth_gbs = (transfer_size / (avg_us / 1e6)) / (1024.0 * 1024.0 * 1024.0);
            
            printf("  %d ↔ %d  |     %7.2f      |   %7.2f\n", 
                   src, dst, bandwidth_gbs, avg_us);
            
            total_bandwidth += bandwidth_gbs;
            pair_count++;
        }
    }
    
    // Summary
    printf("\nMulti-GPU P2P Summary:\n");
    printf("  Average bandwidth: %.2f GB/s per pair\n", total_bandwidth / pair_count);
    printf("  Total pairs: %d\n", pair_count);
    
    if (info.has_nvswitch) {
        printf("  ✓ NVSwitch topology: Full mesh connectivity\n");
        printf("  ✓ Expected: ~1800 GB/s bidirectional per pair (NVLink 5.0)\n");
    } else {
        printf("  ℹ Direct NVLink topology\n");
        printf("  ℹ Some pairs may have lower bandwidth (multi-hop)\n");
    }
    
    // Cleanup
    for (int i = 0; i < max_gpus; i++) {
        NVTX_RANGE("cleanup");
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(d_data[i]));
    }
    
    printf("\nRecommendations for multi-GPU B200:\n");
    printf("  - Use NCCL 2.28 with NVLS for optimal collectives\n");
    printf("  - Tensor Parallel: Split across 2, 4, or all GPUs\n");
    printf("  - Data Parallel: Remaining GPUs for batch parallelism\n");
    printf("  - Monitor with: nvidia-smi dmon -s u -i <gpu_ids>\n");
}

// ============================================================================
// GB200/GB300 Coherent Memory Examples
// ============================================================================

__global__ void coherent_memory_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Read-modify-write on coherent memory
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

void demonstrate_gb200_coherent_memory(const SystemInfo& info) {
    if (!info.is_grace_blackwell) {
        printf("\n⚠ GB200/GB300 coherent memory requires Grace-Blackwell Superchip\n");
        printf("  Current system: Standard CPU-GPU connection\n");
        return;
    }
    
    printf("\n=== GB200/GB300 Coherent Memory Demo ===\n");
    printf("NVLink-C2C enables coherent CPU-GPU memory access\n\n");
    
    const size_t size = 128 * 1024 * 1024;  // 128 MB
    const int num_elements = size / sizeof(float);
    
    // Allocate managed memory (coherent on GB200/GB300)
    float* coherent_data;
    CUDA_CHECK(cudaMallocManaged(&coherent_data, size));
    
    // Initialize on CPU
    printf("1. CPU: Initializing %.2f MB...\n", size / (1024.0 * 1024.0));
    for (int i = 0; i < num_elements; i++) {
        NVTX_RANGE("setup");
        coherent_data[i] = (float)i;
    }
    
    // CPU reads back first element
    float cpu_value_before = coherent_data[0];
    printf("   CPU value[0] before: %.2f\n", cpu_value_before);
    
    // GPU modifies the data (coherent access via NVLink-C2C)
    printf("\n2. GPU: Modifying data via NVLink-C2C...\n");
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // No explicit cudaMemPrefetchAsync needed - coherent!
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    coherent_memory_kernel<<<blocks, threads>>>(coherent_data, num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("   GPU kernel completed in %ld ms\n", duration.count());
    
    // CPU reads back modified value (coherent read via NVLink-C2C)
    printf("\n3. CPU: Reading modified data (coherent)...\n");
    float cpu_value_after = coherent_data[0];
    printf("   CPU value[0] after:  %.2f\n", cpu_value_after);
    printf("   Expected: %.2f\n", cpu_value_before * 2.0f + 1.0f);
    
    if (fabs(cpu_value_after - (cpu_value_before * 2.0f + 1.0f)) < 0.01f) {
        printf("   ✓ Coherent memory working correctly!\n");
    }
    
    // Benchmark coherent access bandwidth
    printf("\n4. Benchmark coherent memory bandwidth:\n");
    
    // CPU writes, GPU reads
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_elements; i++) {
        NVTX_RANGE("compute_kernel");
        coherent_data[i] = (float)i * 0.5f;
    }
    coherent_memory_kernel<<<blocks, threads>>>(coherent_data, num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double bandwidth_gbs = (2 * size / (duration_us.count() / 1e6)) / (1024.0 * 1024.0 * 1024.0);
    printf("   CPU write + GPU read: %.2f GB/s\n", bandwidth_gbs);
    
    CUDA_CHECK(cudaFree(coherent_data));
    
    printf("\nGB200/GB300 Key Benefits:\n");
    printf("  ✓ No explicit cudaMemcpy required\n");
    printf("  ✓ Coherent view of memory from CPU and GPU\n");
    printf("  ✓ ~900 GB/s bandwidth via NVLink-C2C\n");
    printf("  ✓ Unified memory programming model\n");
    printf("  ✓ Ideal for: CPU preprocessing + GPU compute\n");
    
    printf("\nUse Cases:\n");
    printf("  1. CPU data loading → GPU training (seamless)\n");
    printf("  2. Large KV cache in CPU memory (480GB-1TB)\n");
    printf("  3. CPU-GPU pipeline without explicit transfers\n");
    printf("  4. Optimizer state offloading to CPU\n");
}

// ============================================================================
// Main Benchmark
// ============================================================================

void run_transfer_benchmarks(const SystemInfo& info) {
    printf("\n=== CPU-GPU Transfer Benchmark ===\n");
    
    std::vector<size_t> sizes = {
        16 * 1024 * 1024,    // 16 MB
        64 * 1024 * 1024,    // 64 MB
        256 * 1024 * 1024,   // 256 MB
        1024 * 1024 * 1024   // 1 GB
    };
    
    for (size_t size : sizes) {
        NVTX_RANGE("setup");
        printf("\nTransfer size: %.2f MB\n", size / (1024.0 * 1024.0));
        
        // Allocate memory
        float* h_pageable = (float*)malloc(size);
        float* h_pinned;
        float* d_data;
        
        CUDA_CHECK(cudaMallocHost(&h_pinned, size));
        CUDA_CHECK(cudaMalloc(&d_data, size));
        
        // Initialize
        for (size_t i = 0; i < size / sizeof(float); i++) {
            NVTX_RANGE("setup");
            h_pageable[i] = (float)i;
            h_pinned[i] = (float)i;
        }
        
        // Benchmark H2D
        printf("\nHost → Device:\n");
        NVTX_RANGE("transfer_sync:h2d");
        double h2d_pageable = benchmark_host_to_device(h_pageable, d_data, size, 
                                                       "Pageable memory");
        NVTX_RANGE("transfer_sync:h2d");
        double h2d_pinned = benchmark_host_to_device(h_pinned, d_data, size,
                                                     "Pinned memory");
        
        // Benchmark D2H
        printf("\nDevice → Host:\n");
        NVTX_RANGE("transfer_sync:d2h");
        double d2h_pageable = benchmark_device_to_host(d_data, h_pageable, size,
                                                       "Pageable memory");
        NVTX_RANGE("transfer_sync:d2h");
        double d2h_pinned = benchmark_device_to_host(d_data, h_pinned, size,
                                                     "Pinned memory");
        
        // Analysis
        printf("\nSpeedup with pinned memory:\n");
        printf("  H2D: %.2fx faster\n", h2d_pinned / h2d_pageable);
        printf("  D2H: %.2fx faster\n", d2h_pinned / d2h_pageable);
        
        if (info.has_nvlink_c2c && h2d_pinned > 500.0) {
            printf("  ✓ NVLink-C2C performance detected!\n");
        } else if (info.has_pcie_5 && h2d_pinned > 100.0) {
            printf("  ✓ PCIe 5.0 performance achieved\n");
        }
        
        // Cleanup
        free(h_pageable);
        CUDA_CHECK(cudaFreeHost(h_pinned));
        CUDA_CHECK(cudaFree(d_data));
    }
}

int main() {
    NVTX_RANGE("main");
    printf("=== NVLink-C2C CPU-GPU P2P Transfer (Blackwell) ===\n\n");
    
    // Detect system
    SystemInfo info = detect_system_capabilities();
    
    // Run standard benchmarks
    run_transfer_benchmarks(info);
    
    // Demonstrate page migration
    demonstrate_page_migration();
    
    // NEW: multi-GPU P2P benchmarks
    if (info.is_b200_multigpu) {
        benchmark_multigpu_p2p_bandwidth(info);
    }
    
    // NEW: GB200/GB300 coherent memory
    if (info.is_grace_blackwell) {
        demonstrate_gb200_coherent_memory(info);
    }
    
    // Summary
    printf("\n=== Summary ===\n");
    printf("Key Findings:\n");
    printf("1. Pinned memory is 2-3x faster than pageable\n");
    printf("2. NVLink-C2C (Grace-Blackwell): up to 900 GB/s\n");
    printf("3. PCIe 5.0 fallback: ~128 GB/s\n");
    printf("4. cudaMemAdvise optimizes migration for CUDA 13\n");
    printf("5. Prefer pinned memory for frequent transfers\n");
    
    if (info.is_b200_multigpu) {
        double mem_gib = info.gpu_memory / (1024.0 * 1024.0 * 1024.0);
        printf("\n✓ B200/B300 multi-GPU Configuration Detected\n");
        printf("  - Total memory: %.2f TiB (%.1f GiB per GPU)\n",
               info.num_gpus * mem_gib / 1024.0, mem_gib);
        printf("  - NVLink 5.0: 1800 GB/s per GPU pair\n");
        printf("  - Recommendations:\n");
        printf("    * Use NCCL 2.28 for collectives\n");
        printf("    * Hybrid parallelism: choose TP/DP to match model size\n");
        printf("    * Monitor bandwidth: nvidia-smi dmon -s u\n");
    }
    
    if (info.is_grace_blackwell) {
        printf("\n✓ GB200/GB300 Grace-Blackwell Superchip Detected\n");
        printf("  - Grace CPU: 72 ARM cores, 480GB-1TB memory\n");
        printf("  - NVLink-C2C: 900 GB/s coherent bandwidth\n");
        printf("  - Recommendations:\n");
        printf("    * Use managed memory with cudaMemAdvise\n");
        printf("    * CPU preprocessing + GPU training pipeline\n");
        printf("    * Offload optimizer states to CPU memory\n");
        printf("    * Store large KV caches in CPU memory\n");
    }
    
    if (!info.is_grace_blackwell && !info.is_b200_multigpu) {
        printf("\n⚠ Standard GPU Configuration\n");
        if (info.has_pcie_5) {
            printf("  Connection: PCIe 5.0 (~128 GB/s)\n");
        }
        printf("  Recommendations:\n");
        printf("    * Minimize CPU-GPU transfers\n");
        printf("    * Use pinned memory when necessary\n");
        printf("    * Keep data on GPU when possible\n");
    }
    
    printf("\n=== Hardware-Specific Optimizations ===\n");
    printf("Multi-GPU:     ch04/training_multigpu_pipeline.py\n");
    printf("GB200/GB300:   ch04/gb200_grace_numa_optimization.py\n");
    printf("NCCL Config:   ch04/nccl_blackwell_config.py\n");
    printf("Inference:     ch16/inference_optimizations_blackwell.py\n");
    
    return 0;
}
