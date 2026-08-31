// fp8_32byte_loads_demo.cu - Where 32-byte loads ACTUALLY help
//
// KEY INSIGHT FROM CH7:
// The book talks about 32-byte SECTORS (cache granularity), not that 
// 32-byte loads are faster than 16-byte for simple vector add.
//
// WHERE 32-byte loads DO help:
// - FP8 inference: 32 FP8 values per 32-byte load (vs 8 FP8 per 8-byte)
// - Same memory transaction, 4x more data processed
// - FP8 is 1 byte per value, so 32-byte = 32 values
//
// This benchmark shows the REAL benefit of 32-byte loads for FP8

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

constexpr int BLOCK_SIZE = 256;

// 8-byte aligned FP8 vector (8 values)
struct alignas(8) fp8x8 {
    __nv_fp8_e4m3 data[8];
};

// 32-byte aligned FP8 vector (32 values)
struct alignas(32) fp8x32 {
    __nv_fp8_e4m3 data[32];
};

// Narrow loads: 8-byte (8 FP8 values per load)
__global__ void fp8_scale_8byte(
    const fp8x8* __restrict__ input,
    fp8x8* __restrict__ output,
    float scale,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp8x8 in = input[idx];  // 8-byte load (8 FP8 values)
        fp8x8 out;
        
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float val = float(in.data[i]) * scale;
            out.data[i] = __nv_fp8_e4m3(val);
        }
        
        output[idx] = out;  // 8-byte store
    }
}

// Wide loads: 32-byte (32 FP8 values per load)
__global__ void fp8_scale_32byte(
    const fp8x32* __restrict__ input,
    fp8x32* __restrict__ output,
    float scale,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp8x32 in = input[idx];  // 32-byte load (32 FP8 values)
        fp8x32 out;
        
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            float val = float(in.data[i]) * scale;
            out.data[i] = __nv_fp8_e4m3(val);
        }
        
        output[idx] = out;  // 32-byte store
    }
}

int main() {
    NVTX_RANGE("main");
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("FP8 32-byte Load Benchmark (Where Wide Loads Help)\n");
    printf("===================================================\n");
    printf("Device: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);
    
    // Large FP8 array
    const size_t N_FP8 = 512 * 1024 * 1024;  // 512M FP8 values = 512 MB
    const size_t bytes = N_FP8 * sizeof(__nv_fp8_e4m3);
    
    printf("Array size: %zu MB (%zu FP8 values)\n", bytes / (1024*1024), N_FP8);
    printf("Data movement: %zu MB (1 read + 1 write)\n\n", 2 * bytes / (1024*1024));
    
    // Allocate
    void *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    // Use the CUDA conversion instead of assuming an FP8 byte encoding.
    const __nv_fp8_e4m3 one(1.0f);
    CUDA_CHECK(cudaMemset(d_input, one.__x, bytes));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int warmup = 5;
    const int iterations = 20;
    float scale = 1.5f;
    
    printf("%-25s %10s %12s %12s\n", "Kernel", "Time (ms)", "BW (GB/s)", "Values/s");
    printf("%-25s %10s %12s %12s\n", "------", "---------", "---------", "--------");
    
    // Benchmark 8-byte loads
    {
        int n = N_FP8 / 8;
        dim3 block(BLOCK_SIZE);
        dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        for (int i = 0; i < warmup; ++i) {
            NVTX_RANGE("warmup");
            fp8_scale_8byte<<<grid, block>>>(
                (fp8x8*)d_input, (fp8x8*)d_output, scale, n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            NVTX_RANGE("compute_kernel:fp8_scale_8byte");
            fp8_scale_8byte<<<grid, block>>>(
                (fp8x8*)d_input, (fp8x8*)d_output, scale, n);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= iterations;
        
        float bw = (2.0f * bytes) / (ms / 1000.0f) / 1e9f;
        float values_per_sec = N_FP8 / (ms / 1000.0f) / 1e9f;
        
        printf("%-25s %10.3f %12.1f %12.1f B\n", 
               "8-byte (8 FP8/load)", ms, bw, values_per_sec);
    }
    
    // Benchmark 32-byte loads
    {
        int n = N_FP8 / 32;
        dim3 block(BLOCK_SIZE);
        dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        for (int i = 0; i < warmup; ++i) {
            NVTX_RANGE("warmup");
            fp8_scale_32byte<<<grid, block>>>(
                (fp8x32*)d_input, (fp8x32*)d_output, scale, n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            NVTX_RANGE("compute_kernel:fp8_scale_32byte");
            fp8_scale_32byte<<<grid, block>>>(
                (fp8x32*)d_input, (fp8x32*)d_output, scale, n);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= iterations;
        
        float bw = (2.0f * bytes) / (ms / 1000.0f) / 1e9f;
        float values_per_sec = N_FP8 / (ms / 1000.0f) / 1e9f;
        
        printf("%-25s %10.3f %12.1f %12.1f B\n", 
               "32-byte (32 FP8/load)", ms, bw, values_per_sec);
    }
    
    printf("\n");
    printf("KEY INSIGHT:\n");
    printf("  For FP8 data (1 byte per value), 32-byte loads process\n");
    printf("  4x more VALUES per instruction than 8-byte loads.\n");
    printf("  This reduces instruction count and register pressure.\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}
