// optimized_cublaslt_gemm_fp4.cu -- Native NVFP4 GEMM using cuBLASLt tensor cores
//
// This optimized version uses cuBLASLt with native NVFP4 (E2M1) data type and
// 16-element block scaling (VEC16_UE4M3) for maximum throughput on Blackwell.
//
// BOOK REFERENCE (Ch9/Ch19): NVFP4 tensor cores on Blackwell provide 3-5x
// throughput over FP16 using 4-bit precision with per-block scaling.
//
// REQUIRES: CUDA 12.9+, Blackwell GPU (SM 10.0+)
//
// GB300 recipe (validated, cuBLASLt 13.4 on sm_103). cuBLASLt NVFP4 GEMM needs:
//   1. TN format: transa=CUBLAS_OP_T, transb=CUBLAS_OP_N, K-major operands, FP16 out.
//      (an N/N layout returns CUBLAS_STATUS_NOT_SUPPORTED, which is the "unavailable"
//       skip earlier versions of this lab hit -- it was a layout bug, not a driver gap.)
//   2. VEC16_UE4M3 block scales in the SF swizzle layout (sfoff() below):
//      a 512-byte tile of 128 rows x 4 SF-K (CUTLASS/Colfax block16 SF layout).
// A is M x K row-major == K x M col-major (K-major already); B (K x N) is transposed
// to N x K so each N-column is K-contiguous (K-major). One strided-batched TN
// matmul processes all kBatchCount matrices. Standalone proof
// + the full derivation: code/docs/gb300-cublaslt-nvfp4-tn-reference.cu.

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>
#include <vector>
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

#define CUBLASLT_CHECK(call)                                                     \
  do {                                                                           \
    cublasStatus_t status = (call);                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLASLt error " << __FILE__ << ":" << __LINE__ << " "        \
                << status << std::endl;                                          \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

// Block size for NVFP4 scaling (16 elements per scale factor)
constexpr int FP4_BLOCK_SIZE = 16;

// Quantize a K x C matrix stored as C columns each with K contiguous elements
// (src[c*K + k]); pack along K (2 FP4 per byte), one UE4M3 scale per 16-element block.
// Outputs: packed = C*(K/2) bytes; scales = C*(K/16) UE4M3 bytes in plain [c][sk] order.
static void quantize_kmajor(const float* src, int K, int C,
                            uint8_t* packed, uint8_t* scales) {
    const int SFK = K / FP4_BLOCK_SIZE;
    for (int c = 0; c < C; ++c) {
        NVTX_RANGE("iteration");
        for (int b = 0; b < SFK; ++b) {
            NVTX_RANGE("iteration");
            float max_abs = 0.0f;
            for (int i = 0; i < FP4_BLOCK_SIZE; ++i) {
                max_abs = std::max(max_abs, std::fabs(src[(size_t)c * K + b * FP4_BLOCK_SIZE + i]));
            }
            float scale = (max_abs > 0.0f) ? max_abs / 6.0f : 1.0f;
            scales[(size_t)c * SFK + b] = __nv_cvt_float_to_fp8(scale, __NV_SATFINITE, __NV_E4M3);
            for (int i = 0; i < FP4_BLOCK_SIZE; i += 2) {
                float v0 = src[(size_t)c * K + b * FP4_BLOCK_SIZE + i];
                float v1 = src[(size_t)c * K + b * FP4_BLOCK_SIZE + i + 1];
                __nv_fp4_storage_t q0 = __nv_cvt_float_to_fp4(v0 / scale, __NV_E2M1, cudaRoundNearest);
                __nv_fp4_storage_t q1 = __nv_cvt_float_to_fp4(v1 / scale, __NV_E2M1, cudaRoundNearest);
                packed[((size_t)c * K + b * FP4_BLOCK_SIZE + i) / 2] =
                    (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
            }
        }
    }
}

// VEC16_UE4M3 scale-factor swizzle: row r (of `rows`), SF-K index sk, K columns.
// 512-byte tile of 128 rows x 4 SF-K (CUTLASS/Colfax block16 SF layout). Requires
// rows % 128 == 0 and (K/16) % 4 == 0.
static size_t sfoff(int r, int sk, int K) {
    int RK = (K / FP4_BLOCK_SIZE) / 4;
    return (size_t)(r / 128) * 512 * RK + (size_t)(sk / 4) * 512 +
           (size_t)(r % 32) * 16 + (size_t)((r % 128) / 32) * 4 + (size_t)(sk % 4);
}

int main() {
    NVTX_RANGE("main");
    // Check GPU architecture for NVFP4 support
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Running on " << prop.name << " (SM" << prop.major << "." << prop.minor << ")" << std::endl;

    if (prop.major < 10) {
        std::cerr << "SKIPPED: cuBLASLt NVFP4 requires Blackwell (SM 10.0+)." << std::endl;
        std::cerr << "Detected SM" << prop.major << "." << prop.minor << "." << std::endl;
        return 3;
    }

    // Matrix dimensions - must be multiples of 128 (rows, for the SF swizzle tile) and 16.
    constexpr int M = 4096;  // Rows of A and C
    constexpr int N = 4096;  // Cols of B and C
    constexpr int K = 4096;  // Cols of A, Rows of B
    constexpr int kIterations = 10;
    constexpr int kBatchCount = 8;

    static_assert(M % 128 == 0, "M must be multiple of 128 for SF swizzle");
    static_assert(N % 128 == 0, "N must be multiple of 128 for SF swizzle");
    static_assert((K / FP4_BLOCK_SIZE) % 4 == 0, "K/16 must be multiple of 4 for SF swizzle");

    // Per-batch sizes. A and B are both K-major; packed 2 FP4 per byte.
    const size_t packedA = (size_t)M * (K / 2);   // A: M cols x K, K-major
    const size_t packedB = (size_t)N * (K / 2);   // B^T: N cols x K, K-major
    const size_t scaleA  = (size_t)M * (K / FP4_BLOCK_SIZE);
    const size_t scaleB  = (size_t)N * (K / FP4_BLOCK_SIZE);
    const size_t elements_C = (size_t)M * N;
    const int SFK = K / FP4_BLOCK_SIZE;

    std::cout << "Matrix dimensions: M=" << M << " N=" << N << " K=" << K
              << " batch=" << kBatchCount << std::endl;

    // Host allocation
    std::vector<float> h_A_fp32((size_t)M * K * kBatchCount);
    std::vector<float> h_B_fp32((size_t)K * N * kBatchCount);
    std::vector<uint8_t> h_A_packed(packedA * kBatchCount);
    std::vector<uint8_t> h_B_packed(packedB * kBatchCount);
    std::vector<uint8_t> h_A_scales(scaleA * kBatchCount, 0);  // swizzled
    std::vector<uint8_t> h_B_scales(scaleB * kBatchCount, 0);  // swizzled
    std::vector<__half> h_C(elements_C * kBatchCount);

    // Initialize with random values (identical stream + order to the baseline:
    // all of A first, then all of B, mt19937(42), uniform[-1,1]).
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& v : h_A_fp32) { NVTX_RANGE("setup"); v = dis(gen); }
    for (auto& v : h_B_fp32) { NVTX_RANGE("setup"); v = dis(gen); }

    // Quantize to NVFP4 (K-major) + swizzle the block scales, per batch.
    std::cout << "Quantizing matrices to NVFP4 (TN, K-major) + swizzling scales..." << std::endl;
    std::vector<float> b_transpose((size_t)N * K);  // reused per batch: B^T (N cols x K)
    std::vector<uint8_t> sA_plain(scaleA), sB_plain(scaleB);
    for (int batch = 0; batch < kBatchCount; ++batch) {
        NVTX_RANGE("batch");
        const float* A_b = h_A_fp32.data() + (size_t)batch * M * K;  // M x K row-major == K-major cols
        const float* B_b = h_B_fp32.data() + (size_t)batch * K * N;  // K x N row-major

        // A is already K-major (each M-row is K contiguous).
        quantize_kmajor(A_b, K, M, h_A_packed.data() + batch * packedA, sA_plain.data());

        // Transpose B (K x N) -> B^T (N x K) so each N-column is K-contiguous.
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                b_transpose[(size_t)n * K + k] = B_b[(size_t)k * N + n];
        quantize_kmajor(b_transpose.data(), K, N, h_B_packed.data() + batch * packedB, sB_plain.data());

        // Swizzle plain [row][sk] scales into the VEC16_UE4M3 SF layout.
        uint8_t* sAsw = h_A_scales.data() + batch * scaleA;
        uint8_t* sBsw = h_B_scales.data() + batch * scaleB;
        for (int m = 0; m < M; ++m)
            for (int sk = 0; sk < SFK; ++sk)
                sAsw[sfoff(m, sk, K)] = sA_plain[(size_t)m * SFK + sk];
        for (int n = 0; n < N; ++n)
            for (int sk = 0; sk < SFK; ++sk)
                sBsw[sfoff(n, sk, K)] = sB_plain[(size_t)n * SFK + sk];
    }

    std::fill(h_C.begin(), h_C.end(), __float2half(0.0f));

    // Device allocation
    uint8_t *d_A = nullptr, *d_B = nullptr, *d_A_scales = nullptr, *d_B_scales = nullptr;
    __half *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, packedA * kBatchCount));
    CUDA_CHECK(cudaMalloc(&d_B, packedB * kBatchCount));
    CUDA_CHECK(cudaMalloc(&d_A_scales, scaleA * kBatchCount));
    CUDA_CHECK(cudaMalloc(&d_B_scales, scaleB * kBatchCount));
    CUDA_CHECK(cudaMalloc(&d_C, elements_C * kBatchCount * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_packed.data(), packedA * kBatchCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_packed.data(), packedB * kBatchCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_scales, h_A_scales.data(), scaleA * kBatchCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_scales, h_B_scales.data(), scaleB * kBatchCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), elements_C * kBatchCount * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // cuBLASLt setup: TN, NVFP4 operands, VEC16_UE4M3 scales, FP32 compute, FP16 out.
    cublasLtHandle_t ltHandle;
    CUBLASLT_CHECK(cublasLtCreate(&ltHandle));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cublasLtMatmulDesc_t matmulDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t transa = CUBLAS_OP_T;  // op(A): K x M -> M x K
    cublasOperation_t transb = CUBLAS_OP_N;  // op(B): K x N
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    // Scale pointers (batch 0 start). cuBLASLt advances the VEC16 block scale per batch for the
    // batched matmul below, because the SF buffers are laid out per-batch contiguous and the
    // operand layouts declare the batch (confirmed by a standalone 2-batch probe). Must be
    // non-null at heuristic time for the block-scaled path.
    void* as0 = d_A_scales;
    void* bs0 = d_B_scales;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &as0, sizeof(as0)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bs0, sizeof(bs0)));

    // Layouts (col-major): A = K x M (lda=K), B = K x N (ldb=K), C = M x N (ldc=M), batched over
    // kBatchCount matrices in ONE matmul. A single 4096^3 GEMM (256x256 tile -> 256 output tiles)
    // underfills the 152-SM GPU (ncu: 10% occupancy, 53% SM); batching all kBatchCount matrices
    // (~2048 tiles) fills it and lifts tensor-core throughput.
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_4F_E2M1, K, M, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_4F_E2M1, K, N, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16F, M, N, M));
    const int batchCount = kBatchCount;
    const long long strideA = (long long)K * M;   // elements per A matrix (FP4)
    const long long strideB = (long long)K * N;   // elements per B matrix (FP4)
    const long long strideC = (long long)M * N;   // elements per C matrix (FP16)
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));

    float alpha = 1.0f, beta = 0.0f;
    size_t workspaceSize = 64ull << 20;
    void* d_workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));

    cublasLtMatmulPreference_t preference;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // Request the top-K candidate algorithms (not just the first). cuBLASLt's first-ranked
    // heuristic is not always the fastest for a given shape, so auto-tune: time each candidate
    // and pick the fastest for the official run.
    std::cout << "Querying cuBLASLt for NVFP4 algorithms..." << std::endl;
    constexpr int kMaxAlgos = 8;
    cublasLtMatmulHeuristicResult_t heuristicResults[kMaxAlgos] = {};
    int returnedResults = 0;
    cublasStatus_t heuristicStatus = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
        preference, kMaxAlgos, heuristicResults, &returnedResults);

    if (heuristicStatus != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
        std::cerr << "SKIPPED: cuBLASLt NVFP4 algorithm unavailable on this driver/toolchain." << std::endl;
        std::cerr << "Diagnostic heuristic status=" << heuristicStatus
                  << ", returned_results=" << returnedResults << "." << std::endl;
        return 3;
    }

    // One batched matmul over all kBatchCount matrices with a chosen algo. cuBLASLt strides the
    // operands, C, AND the per-batch block-scale pointers (set once above); d_A/d_B/d_C and the
    // scale buffers are each laid out per-batch contiguous to match the strides.
    auto run_with = [&](const cublasLtMatmulAlgo_t* algo) {
        CUBLASLT_CHECK(cublasLtMatmul(ltHandle, matmulDesc,
                                      &alpha, d_A, layoutA, d_B, layoutB,
                                      &beta, d_C, layoutC, d_C, layoutC,
                                      algo, d_workspace, workspaceSize, stream));
    };

    // Auto-tune over the returned candidates: warmup + a few timed iters each, pick the fastest.
    cudaEvent_t tstart, tstop;
    CUDA_CHECK(cudaEventCreate(&tstart));
    CUDA_CHECK(cudaEventCreate(&tstop));
    int bestIdx = 0;
    float bestMs = 1e30f;
    for (int a = 0; a < returnedResults; ++a) {
        run_with(&heuristicResults[a].algo);  // warmup
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(tstart, stream));
        for (int w = 0; w < 3; ++w) run_with(&heuristicResults[a].algo);
        CUDA_CHECK(cudaEventRecord(tstop, stream));
        CUDA_CHECK(cudaEventSynchronize(tstop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, tstart, tstop));
        if (ms < bestMs) { bestMs = ms; bestIdx = a; }
    }
    CUDA_CHECK(cudaEventDestroy(tstart));
    CUDA_CHECK(cudaEventDestroy(tstop));
    std::cout << "NVFP4 GEMM: auto-tuned over " << returnedResults
              << " candidate(s), selected #" << bestIdx << ", running benchmark..." << std::endl;
    const cublasLtMatmulAlgo_t* bestAlgo = &heuristicResults[bestIdx].algo;
    auto run_batched = [&]() { run_with(bestAlgo); };

    // Warmup
    {
        NVTX_RANGE("compute_math:ltmatmul");
        run_batched();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iter = 0; iter < kIterations; ++iter) {
        NVTX_RANGE("compute_math:ltmatmul");
        run_batched();
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(kIterations * kBatchCount);

    const double flops = 2.0 * M * N * K * kBatchCount * kIterations;
    const double tflops = flops / (total_ms * 1e9);

    std::cout << "cuBLASLt NVFP4 GEMM (tensor cores): " << avg_ms << " ms" << std::endl;
    std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;

#ifdef VERIFY
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, elements_C * kBatchCount * sizeof(__half), cudaMemcpyDeviceToHost));
    const size_t elements = elements_C * kBatchCount;
    double checksum = 0.0;
    for (size_t i = 0; i < elements; ++i) {
        checksum += std::abs(__half2float(h_C[i]));
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    // Cleanup
    CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(matmulDesc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutA));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutB));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutC));
    CUBLASLT_CHECK(cublasLtDestroy(ltHandle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A_scales));
    CUDA_CHECK(cudaFree(d_B_scales));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
