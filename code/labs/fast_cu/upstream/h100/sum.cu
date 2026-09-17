/**
make sum && ./out/sum

The default workload remains N = 1<<30. Inputs are 0 or 1 so the exact sum is
provably within the int32 range used by the kernels.
*/

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

void checkCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        std::cerr << operation << " failed: " << cudaGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

const int WARP_SIZE = 32;

int ceilDiv(int N, int D) {
    return (N + D - 1) / D;
}

__device__ void warpReduce1(volatile int *sdata, int tid) {
    #pragma unroll
    for (int i = WARP_SIZE; i >= 1; i >>= 1) {
        sdata[tid] += sdata[tid + i];
    }
}

// Inspired from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <int BlockSize, int Batch>
__global__ void sumKernel1(int *d_in, int *d_out, int n) {
    __shared__ int sdata[BlockSize];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*BlockSize*Batch + tid;

    // Read Batch elements(strided)
    float value = i < n ? d_in[i] : 0;
    #pragma unroll
    for (int j = 1; j < Batch; ++j) {
        if (i + j*BlockSize < n) value += d_in[i + j*BlockSize];
    }
    // Save in shmem and compute sum(before warp level)
    sdata[tid] = value;
    __syncthreads();

    #pragma unroll
    for (int s = 512; s > WARP_SIZE; s >>= 1) {
        if (BlockSize >= s * 2) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();    
        }
    }

    // Compute warp sum
    if (tid < WARP_SIZE) warpReduce1(sdata, tid);
    // Final addition to global memory
    if (tid == 0) atomicAdd(d_out, sdata[0]);
}

inline int __device__ warpReduce2(int sum) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    return sum;
}

template <int BlockSize, int Batch>
__global__ void sumKernel2(int4 *d_in, int *d_out, int n) {
    // This kernel assumes 32 warps in block
    static_assert(BlockSize == 1024);
    __shared__ __align__(16) int sdata[WARP_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*BlockSize*Batch + tid;
    int4 value = i < n ? d_in[i] : make_int4(0, 0, 0, 0);
    int sum = value.x + value.y + value.z + value.w;
    #pragma unroll
    for (int j = 1; j < Batch; ++j) {
        if (i + j*BlockSize < n) {
            value = d_in[i + j*BlockSize];
            sum += value.x + value.y + value.z + value.w;
        }
    }
    // Sum within warps
    sum = warpReduce2(sum);

    // Store warp sums in sdata
    if (tid % WARP_SIZE == 0) {
        sdata[tid / WARP_SIZE] = sum;
    }
    __syncthreads();
    if (tid < WARP_SIZE) {
        sum = sdata[tid];
        sum = warpReduce2(sum);
    }
    if (tid == 0) atomicAdd(d_out, sum);
}

template <int BlockSize, int Batch>
__global__ void sumKernel3(int4 *d_in, int *d_out, int n) {
    __shared__ __align__(16) int sdata[1];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*BlockSize*Batch + tid;

    if (tid == 0) sdata[0] = 0;

    int4 value = i < n ? d_in[i] : make_int4(0, 0, 0, 0);
    int sum = value.x + value.y + value.z + value.w;
    #pragma unroll
    for (int j = 1; j < Batch; ++j) {
        if (i + j*BlockSize < n) {
            value = d_in[i + j*BlockSize];
            sum += value.x + value.y + value.z + value.w;
        }
    }
    // Sum within warps
    sum = warpReduce2(sum);
    __syncthreads();
    if (tid % WARP_SIZE == 0) {
        atomicAdd(&sdata[0], sum);
    }
    __syncthreads();
    if (tid == 0) {
        atomicAdd(d_out, sdata[0]);
    }
}

inline int __device__ warpReduce3(int sum) {
    int sum_warp = 0;
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-redux-sync
    // Use `redux.sync` to reduce within warp
    asm volatile("redux.sync.add.s32 %0, %1, 0xff;" : "=r"(sum_warp) : "r"(sum));
    return sum_warp;
}

template <int BlockSize, int Batch>
__global__ void sumKernel4(int4 *d_in, int *d_out, int n) {
    __shared__ __align__(16) int sdata[1];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*BlockSize*Batch + tid;

    if (tid == 0) sdata[0] = 0;

    int4 value = i < n ? __ldcg(&d_in[i]) : make_int4(0, 0, 0, 0);
    int sum = value.x + value.y + value.z + value.w;
    #pragma unroll
    for (int j = 1; j < Batch; ++j) {
        if (i + j*BlockSize < n) {
            value = d_in[i + j*BlockSize];
            sum += value.x + value.y + value.z + value.w;
        }
    }
    // Sum within warps
    sum = warpReduce3(sum);
    __syncthreads();
    if (tid % WARP_SIZE == 0) {
        atomicAdd(&sdata[0], sum);
    }
    __syncthreads();
    if (tid == 0) {
        atomicAdd(d_out, sdata[0]);
    }
}

const int NUM_KERNELS = 4;
const int NUM_KERNEL_RUNS = 100;
const int NUM_STD_RUNS = NUM_KERNEL_RUNS;

void kernelDispatch(int kernelNum, int *d_in, int *d_out, int *h_out, int N) {
    // N counts scalar ints. The int4 kernels require at least four values and
    // a multiple-of-four length.
    if (N < 4 || N % 4 != 0) {
        std::cerr << "N must be at least 4 and divisible by 4; got " << N << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // The reset must be ordered before every cross-CTA atomic reduction.
    // Keeping it in dispatch also keeps the reset inside the timed region.
    checkCuda(cudaMemsetAsync(d_out, 0, sizeof(int), 0), "reset reduction output");
    switch (kernelNum) {
        case 1:
            sumKernel1<512, 20><<<ceilDiv(N, 512*20), 512>>>(d_in, d_out, N);
            break;
        case 2:
            sumKernel2<1024, 16><<<ceilDiv(N, 1024*4*16), 1024>>>(reinterpret_cast<int4*>(d_in), d_out, N/4);
            break;
        case 3:
            sumKernel3<1024, 16><<<ceilDiv(N, 1024*4*16), 1024>>>(reinterpret_cast<int4*>(d_in), d_out, N/4);
            break;
        case 4:
            sumKernel4<1024, 16><<<ceilDiv(N, 1024*4*16), 1024>>>(reinterpret_cast<int4*>(d_in), d_out, N/4);
            break;
        default:
            std::cerr << "kernel number must be in [1, 4]; got " << kernelNum << std::endl;
            std::exit(EXIT_FAILURE);
    }
    checkCuda(cudaGetLastError(), "launch reduction kernel");
}

void printDetails(std::string info, float timeMs, int N, int sum) {
    std::cout << "<---------------| " << info << " |--------------->" << std::endl;
    float bw = N*sizeof(int)/timeMs/1e6;
    std::cout << "Bandwidth: " << bw << " GB/s" << std::endl;
    std::cout << "Time taken: " << timeMs << " ms" << std::endl;
    std::cout << "Sum: " << sum << std::endl << std::endl;
}

bool sumKernelCall(
    int kernelNum, int *d_in, int *d_out, int *h_out, int N, int times, int64_t expected) {
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");
    checkCuda(cudaEventRecord(start), "record start event");

    for (int i = 0; i < times; ++i) kernelDispatch(kernelNum, d_in, d_out, h_out, N);

    checkCuda(cudaEventRecord(stop), "record stop event");
    checkCuda(cudaEventSynchronize(stop), "synchronize stop event");

    float elapsedTime;
    checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop), "measure elapsed time");

    checkCuda(cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost), "copy kernel output");

    checkCuda(cudaEventDestroy(start), "destroy start event");
    checkCuda(cudaEventDestroy(stop), "destroy stop event");

    if (static_cast<int64_t>(*h_out) != expected) {
        std::cerr << "kernel" << kernelNum << " mismatch: expected " << expected
                  << ", got " << *h_out << std::endl;
        return false;
    }
    printDetails(std::string("kernel") + std::to_string(kernelNum), elapsedTime / times, N, *h_out);
    return true;
}

bool sumCubCall(int *d_in, int *d_out, int *h_out, int N, int times, int64_t expected) {
    void* d_temp = nullptr;
    size_t temp_storage = 0;

    // First call to determine temporary storage size
    checkCuda(
        cub::DeviceReduce::Sum(d_temp, temp_storage, d_in, d_out, N),
        "query CUB temporary storage");
    
    // Allocate temporary storage
    if (temp_storage == 0) {
        std::cerr << "CUB requested zero bytes of temporary storage" << std::endl;
        return false;
    }
    checkCuda(cudaMalloc(&d_temp, temp_storage), "allocate CUB temporary storage");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "create CUB start event");
    checkCuda(cudaEventCreate(&stop), "create CUB stop event");

    checkCuda(cudaEventRecord(start), "record CUB start event");

    for (int i = 0; i < times; ++i) {
        checkCuda(
            cub::DeviceReduce::Sum(d_temp, temp_storage, d_in, d_out, N),
            "launch CUB reduction");
    }

    checkCuda(cudaEventRecord(stop), "record CUB stop event");
    checkCuda(cudaEventSynchronize(stop), "synchronize CUB stop event");

    checkCuda(cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost), "copy CUB output");

    float elapsedTime;
    checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop), "measure CUB elapsed time");

    checkCuda(cudaFree(d_temp), "free CUB temporary storage");
    checkCuda(cudaEventDestroy(start), "destroy CUB start event");
    checkCuda(cudaEventDestroy(stop), "destroy CUB stop event");

    if (static_cast<int64_t>(*h_out) != expected) {
        std::cerr << "CUB mismatch: expected " << expected << ", got " << *h_out << std::endl;
        return false;
    }
    printDetails("cub", elapsedTime / times, N, *h_out);
    return true;
}

int64_t sumCpu(const int *h_in, int N) {
    int64_t value = 0;
    for (int i = 0; i < N; ++i) {
        value += h_in[i];
    }
    return value;
}

__global__ void warmupKernel() {
    extern __shared__ int sdata[];
}

int main() {
    warmupKernel<<<1024, 1024, 1024*sizeof(int)>>>();
    checkCuda(cudaGetLastError(), "launch warmup kernel");
    checkCuda(cudaDeviceSynchronize(), "synchronize warmup kernel");

    const int N = 1 << 30;
    size_t size = N * sizeof(int);

    // Allocate host memory
    int* h_in = new int[N];
    int h_out = 0;

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_in[i] = rand() % 2;
    }
    const int64_t expected = sumCpu(h_in, N);
    if (expected < std::numeric_limits<int>::min()
        || expected > std::numeric_limits<int>::max()) {
        std::cerr << "input sum exceeds the int32 output range: " << expected << std::endl;
        delete[] h_in;
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int* d_in;
    int* d_out;
    checkCuda(cudaMalloc(&d_in, size), "allocate input");
    checkCuda(cudaMalloc(&d_out, sizeof(int)), "allocate output");

    // Copy input data from host to device
    checkCuda(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice), "copy input");

    bool passed = true;
    for (int i = 1; i <= NUM_KERNELS; ++i) {
        if (!sumKernelCall(i, d_in, d_out, &h_out, N, NUM_KERNEL_RUNS, expected)) {
            passed = false;
        }
    }
    if (!sumCubCall(d_in, d_out, &h_out, N, NUM_STD_RUNS, expected)) {
        passed = false;
    }

    // Free memory
    checkCuda(cudaFree(d_in), "free input");
    checkCuda(cudaFree(d_out), "free output");
    delete[] h_in;
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
