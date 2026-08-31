// optimized_cublaslt_gemm.cu -- Optimized GEMM using cuBLASLt tensor cores.

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " "           \
                << cudaGetErrorString(status) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CUBLAS_CHECK(call)                                                       \
  do {                                                                           \
    cublasStatus_t status = (call);                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLAS error " << __FILE__ << ":" << __LINE__ << " code "    \
                << status << std::endl;                                          \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CUBLASLT_CHECK(call)                                                     \
  do {                                                                           \
    cublasStatus_t status = (call);                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLASLt error " << __FILE__ << ":" << __LINE__ << " code "  \
                << status << std::endl;                                          \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

int main() {
    NVTX_RANGE("main");
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    constexpr int batch_count = 32;
    constexpr int iterations = 5;
    constexpr size_t workspace_bytes = 64ull * 1024ull * 1024ull;

    const size_t elems_A = static_cast<size_t>(M) * K;
    const size_t elems_B = static_cast<size_t>(K) * N;
    const size_t elems_C = static_cast<size_t>(M) * N;
    const size_t size_A = elems_A * sizeof(float) * batch_count;
    const size_t size_B = elems_B * sizeof(float) * batch_count;
    const size_t size_C = elems_C * sizeof(float) * batch_count;

    float* h_A = nullptr;
    float* h_B = nullptr;
    float* h_C = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_A, size_A));
    CUDA_CHECK(cudaMallocHost(&h_B, size_B));
    CUDA_CHECK(cudaMallocHost(&h_C, size_C));

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < elems_A * batch_count; ++i) {
        NVTX_RANGE("batch");
        h_A[i] = dis(gen);
    }
    for (size_t i = 0; i < elems_B * batch_count; ++i) {
        NVTX_RANGE("batch");
        h_B[i] = dis(gen);
    }
    std::fill(h_C, h_C + (elems_C * batch_count), 0.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    cublasLtHandle_t ltHandle;
    CUBLASLT_CHECK(cublasLtCreate(&ltHandle));
    cublasHandle_t fallbackHandle;
    CUBLAS_CHECK(cublasCreate(&fallbackHandle));
    CUBLAS_CHECK(cublasSetMathMode(fallbackHandle, CUBLAS_DEFAULT_MATH));

    cublasLtMatmulDesc_t operationDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t trans = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, N, N));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, N));
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    const long long strideA = static_cast<long long>(elems_A);
    const long long strideB = static_cast<long long>(elems_B);
    const long long strideC = static_cast<long long>(elems_C);
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

    cublasLtMatmulPreference_t preference;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes,
        sizeof(workspace_bytes)));

    constexpr int kMaxAlgos = 8;
    cublasLtMatmulHeuristicResult_t heuristics[kMaxAlgos]{};
    int returnedResults = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ltHandle,
        operationDesc,
        Adesc,
        Bdesc,
        Cdesc,
        Cdesc,
        preference,
        kMaxAlgos,
        heuristics,
        &returnedResults));
    
    void* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Auto-tune over the returned candidates (rank-0 is not always fastest); this also supplies the
    // warmup this lab lacked (baseline_cublaslt_gemm warms up, so the optimized was timed cold).
    const cublasLtMatmulAlgo_t* bestAlgo = &heuristics[0].algo;
    if (returnedResults > 1) {
        auto run_lt = [&](const cublasLtMatmulAlgo_t* algo) {
            CUBLASLT_CHECK(cublasLtMatmul(ltHandle, operationDesc, &alpha, d_A, Adesc, d_B, Bdesc,
                                          &beta, d_C, Cdesc, d_C, Cdesc, algo, workspace, workspace_bytes, 0));
        };
        cudaEvent_t ts, te;
        CUDA_CHECK(cudaEventCreate(&ts));
        CUDA_CHECK(cudaEventCreate(&te));
        int bestIdx = 0;
        float bestMs = 1e30f;
        for (int a = 0; a < returnedResults; ++a) {
            run_lt(&heuristics[a].algo);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(ts));
            for (int w = 0; w < 3; ++w) run_lt(&heuristics[a].algo);
            CUDA_CHECK(cudaEventRecord(te));
            CUDA_CHECK(cudaEventSynchronize(te));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ts, te));
            if (ms < bestMs) { bestMs = ms; bestIdx = a; }
        }
        CUDA_CHECK(cudaEventDestroy(ts));
        CUDA_CHECK(cudaEventDestroy(te));
        bestAlgo = &heuristics[bestIdx].algo;
        std::cout << "cuBLASLt GEMM: auto-tuned over " << returnedResults
                  << " candidates, selected #" << bestIdx << std::endl;
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        NVTX_RANGE("compute_math:ltmatmul");
        if (returnedResults > 0) {
            CUBLASLT_CHECK(cublasLtMatmul(
                ltHandle,
                operationDesc,
                &alpha,
                d_A,
                Adesc,
                d_B,
                Bdesc,
                &beta,
                d_C,
                Cdesc,
                d_C,
                Cdesc,
                bestAlgo,
                workspace,
                workspace_bytes,
                0));
        } else {
            CUBLAS_CHECK(cublasSgemmStridedBatched(
                fallbackHandle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                M,
                K,
                &alpha,
                d_B,
                N,
                strideB,
                d_A,
                K,
                strideA,
                &beta,
                d_C,
                N,
                strideC,
                batch_count));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / (iterations * batch_count);
    std::cout << "cuBLASLt batched GEMM (optimized): " << avg_ms << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    std::cout << "Checksum sample: " << h_C[0] << std::endl;

#ifdef VERIFY
    double checksum = 0.0;
    for (size_t i = 0; i < elems_C; ++i) {
        NVTX_RANGE("verify");
        checksum += std::abs(static_cast<double>(h_C[i]));
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(workspace));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(operationDesc));
    CUBLASLT_CHECK(cublasLtDestroy(ltHandle));
    CUBLAS_CHECK(cublasDestroy(fallbackHandle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    return 0;
}
