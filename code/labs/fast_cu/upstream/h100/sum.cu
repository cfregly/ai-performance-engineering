/**
make sum && ./sum

N = 1<<30, GPU = H100:
<---------------| kernel1 |--------------->
Bandwidth: 3232.76 GB/s
Time taken: 1.32858 ms
Sum: 1609549285

<---------------| kernel2 |--------------->
Bandwidth: 3239.69 GB/s
Time taken: 1.32573 ms
Sum: 1609549285

<---------------| kernel3 |--------------->
Bandwidth: 3240.11 GB/s
Time taken: 1.32556 ms
Sum: 1609549285

<---------------| kernel4 |--------------->
Bandwidth: 3241.47 GB/s
Time taken: 1.32501 ms
Sum: 1609549285

<---------------| cub |--------------->
Bandwidth: 3193.36 GB/s
Time taken: 1.34497 ms
Sum: 1609549285
*/

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>

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

    // Set output to 0
    if (i == 0) *d_out = 0;

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
    if (i == 0) {
        *d_out = 0;
    }

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

    if (i == 0) *d_out = 0;
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

    if (i == 0) *d_out = 0;
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
    assert(N % 4 == 0);
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
    }
}

void printDetails(std::string info, float timeMs, int N, int sum) {
    std::cout << "<---------------| " << info << " |--------------->" << std::endl;
    float bw = N*sizeof(int)/timeMs/1e6;
    std::cout << "Bandwidth: " << bw << " GB/s" << std::endl;
    std::cout << "Time taken: " << timeMs << " ms" << std::endl;
    std::cout << "Sum: " << sum << std::endl << std::endl;
}

void sumKernelCall(int kernelNum, int *d_in, int *d_out, int *h_out, int N, int times) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < times; ++i) kernelDispatch(kernelNum, d_in, d_out, h_out, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printDetails(std::string("kernel") + std::to_string(kernelNum), elapsedTime / times, N, *h_out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void sumCubCall(int *d_in, int *d_out, int *h_out, int N, int times) {
    void* d_temp = nullptr;
    size_t temp_storage = 0;

    // First call to determine temporary storage size
    cub::DeviceReduce::Sum(d_temp, temp_storage, d_in, d_out, N);
    
    // Allocate temporary storage
    assert(temp_storage > 0);
    cudaMalloc(&d_temp, temp_storage);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < times; ++i) cub::DeviceReduce::Sum(d_temp, temp_storage, d_in, d_out, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printDetails("cub", elapsedTime / times, N, *h_out);

    cudaFree(d_temp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void sumCpuCall(int *h_in, int *h_out, int N) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    int value = 0;
    for (int i = 0; i < N; ++i) {
        value += h_in[i];
    }
    *h_out = value;
    float timeTaken = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count()/1e6f;
    printDetails("cpu", timeTaken, N, value);
}

__global__ void warmupKernel() {
    extern __shared__ int sdata[];
}

int main() {
    warmupKernel<<<1024, 1024, 1024*sizeof(int)>>>();
    cudaDeviceSynchronize();

    const int N = 1 << 30;
    size_t size = N * sizeof(int);

    // Allocate host memory
    int* h_in = new int[N];
    int h_out = 0.0f;

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_in[i] = rand() % 100;
    }

    // Allocate device memory
    int* d_in;
    int* d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, &h_out, sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 1; i <= NUM_KERNELS; ++i) sumKernelCall(i, d_in, d_out, &h_out, N, NUM_KERNEL_RUNS);
    sumCubCall(d_in, d_out, &h_out, N, NUM_STD_RUNS);

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    return 0;
}