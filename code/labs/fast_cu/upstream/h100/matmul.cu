#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <ctime>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_bf16.h>
#include <cassert>
#include <unistd.h>

typedef __nv_bfloat16 bf16;
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(1);
  }
}
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

#include "matmul/matmul_1.cuh"
#include "matmul/matmul_2.cuh"
#include "matmul/matmul_3.cuh"
#include "matmul/matmul_4.cuh"
#include "matmul/matmul_5.cuh"
#include "matmul/matmul_6.cuh"
#include "matmul/matmul_7.cuh"
#include "matmul/matmul_8.cuh"
#include "matmul/matmul_9.cuh"
#include "matmul/matmul_10.cuh"
#include "matmul/matmul_11.cuh"
#include "matmul/matmul_12.cuh"

std::default_random_engine generator(69);
cublasHandle_t cublas_handle;
void runCublasGemmBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  float alpha = 1, beta = 0;
  // C(column major) = A(row major) * B(column major)
  cublasStatus_t status = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_16BF,
    N, B, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS error: " << status << std::endl;
    exit(1);
  }
}

void run_kernel(int kernel_num, int M, int N, int K, bf16 *A, bf16 *B, bf16 *C, int *DB = nullptr) {
  switch (kernel_num) {
    case 0:
      runCublasGemmBF16(M, N, K, A, B, C);
      break;
    case 1:
      runKernel1(M, N, K, A, B, C);
      break;
    case 2:
      runKernel2(M, N, K, A, B, C);
      break;
    case 3:
      runKernel3(M, N, K, A, B, C, DB);
      break;
    case 4:
      runKernel4(M, N, K, A, B, C, DB);
      break;
    case 5:
      runKernel5(M, N, K, A, B, C, DB);
      break;
    case 6:
      runKernel6(M, N, K, A, B, C, DB);
      break;
    case 7:
      runKernel7(M, N, K, A, B, C, DB);
      break;
    case 8:
      runKernel8(M, N, K, A, B, C, DB);
      break;
    case 9:
      runKernel9(M, N, K, A, B, C, DB);
      break;
    case 10:
      runKernel10(M, N, K, A, B, C, DB);
      break;
    case 11:
      runKernel11(M, N, K, A, B, C, DB);
      break;
    case 12:
      runKernel12(M, N, K, A, B, C, DB);
      break;
  }
}
int yo = 0;
void randomize_matrix(bf16 *mat, int N) {
  std::normal_distribution<float> distribution(0, 1);
  for (int i = 0; i < N; i++) {
    mat[i] = distribution(generator);
  }
  ++yo;
}

bool verify_matrix(bf16 *matRef, bf16 *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    int r = i / 8192, c = i % 8192;
    int it = c*8192+r;
    diff = std::fabs(__bfloat162float(matRef[i] - matOut[i]));
    if (diff > 0.1) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
      __bfloat162float(matRef[i]), __bfloat162float(matOut[i]), diff, i);
      return false;
    }
  }
  return true;
}

__global__ void warmupKernel() {
  __shared__ int s[100];
  s[0] += s[1];
}

int main() {
  warmupKernel<<<1024, 1024>>>();

  cublasCreate(&cublas_handle);
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  long max_size = 8192;
  long m = max_size, n = max_size, k = max_size;

  bf16 *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr;  // host matrices
  bf16 *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices
  
  int *DB = nullptr; int *dDB = nullptr;  

  A = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  B = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  C = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  C_ref = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  DB = (int *)malloc(sizeof(int) * max_size * 128);
  cudaCheck(cudaMalloc((void **)&dDB, sizeof(int) * max_size * 128));

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);
  
  cudaCheck(cudaMalloc((void **)&dA, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(bf16) * max_size * max_size));
  
  cudaCheck(cudaMemcpy(dA, A, sizeof(bf16) * max_size * max_size,
  cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(bf16) * max_size * max_size,
      cudaMemcpyHostToDevice));

  int repeat_times = 8;
  bool run_verif = true;
  for (int kernel_num : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) {
    // for (int kernel_num : {0, 11}) {
    // Give the GPU some rest to avoid thermal throttling
    sleep(5);
    std::cout << "KERNEL " << kernel_num << std::endl;
    // Verify against cuBLAS. Also serves as a warmup step.
    if (run_verif) {
      memset(C, 0, sizeof(bf16) * max_size * max_size);
      cudaCheck(cudaMemcpy(dC, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(dC_ref, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
      memset(DB, ~0, sizeof(int) * max_size * 128);
      cudaCheck(cudaMemcpy(dDB, DB, sizeof(int) * max_size * 128,
        cudaMemcpyHostToDevice));
      run_kernel(0, m, n, k, dA, dB, dC_ref); // cuBLAS
      run_kernel(kernel_num, m, n, k, dA, dB, dC, dDB); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(C, dC, sizeof(bf16) * max_size * max_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(bf16) * max_size * max_size, cudaMemcpyDeviceToHost);

      if (kernel_num > 1 && !verify_matrix(C_ref, C, m * n)) {
        std::cout << "~~~~~~~~~~~~~~~~ Failed to pass the correctness verification against cuBLAS. ~~~~~~~~~~~~~~~~" << std::endl;
        printf("%f\n", __bfloat162float(C_ref[m]));
      }

      cudaMemcpy(DB, dDB, sizeof(int) * max_size * 8, cudaMemcpyDeviceToHost);

      int i = 0;
      long sumLoad = 0, cntLoad = 0;
      long sumCompute = 0, cntCompute = 0;
      long sumStore = 0, cntStore = 0;
      int times = 0;
      while (DB[i] != ~0) {
        sumLoad += DB[i], cntLoad += DB[i + 1];
        sumCompute += DB[i + 2], cntCompute += DB[i + 3];
        sumStore += DB[i + 4], cntStore += DB[i + 5];
        i += 6;
        times++;
      }
      if (times > 0) {
        printf("Load: %f, Compute: %f,  Store: %f, Datapoints: %d\n", (sumLoad + .0) / cntLoad, (sumCompute + .0) / cntCompute, (sumStore + .0) / cntStore, times);
      }

    }

    // Benchmark
    cudaEventRecord(start);
    for (int j = 0; j < repeat_times; j++) {
      run_kernel(kernel_num, m, n, k, dA, dB, dC);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    long flops = (2LL * m) * (n * k);
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) TFLOPS. size: (%ld).\n\n",
        elapsed_time / 1000.0 / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m);
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  return 0;
};
