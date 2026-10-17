/*
 * Optimized FP4 Hardware Kernel using cuBLASLt tensor cores.
 *
 * Uses native NVFP4 (CUDA_R_4F_E2M1) with VEC16_UE4M3 block scaling
 * for maximum tensor core throughput on Blackwell.
 *
 * Baseline uses manual FP4 packing without tensor cores.
 * This version demonstrates the massive speedup from proper tensor core usage.
 */

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <random>
#include <cmath>
#include <string>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/headers/nvfp4_scale_layout.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUBLASLT_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLASLt error " << __FILE__ << ":" << __LINE__ << " - " << status << std::endl; \
            exit(1); \
        } \
    } while(0)

constexpr int FP4_BLOCK_SIZE = 16;

// Quantize a row-major [rows, reduction_dim] matrix to NVFP4 along the
// reduction dimension. Values are packed two per byte, and each 16-value block
// scale is written in cuBLASLt's VEC16_UE4M3 scale-factor layout.
void quantize_to_nvfp4_kmajor(const float* input,
                              uint8_t* output_packed,
                              __nv_fp8_e4m3* scales,
                              int rows,
                              int reduction_dim) {
    NVTX_RANGE("setup:quantize_to_nvfp4");
    const int packed_cols = reduction_dim / 2;
    const int num_scale_cols = reduction_dim / FP4_BLOCK_SIZE;

    for (int r = 0; r < rows; ++r) {
        const float* row_input = input + static_cast<size_t>(r) * reduction_dim;
        for (int block = 0; block < num_scale_cols; ++block) {
            const int block_start = block * FP4_BLOCK_SIZE;

            float max_abs = 0.0f;
            for (int i = 0; i < FP4_BLOCK_SIZE; ++i) {
                max_abs = std::max(max_abs, std::abs(row_input[block_start + i]));
            }

            const __nv_fp8_e4m3 stored_scale((max_abs > 0.0f) ? max_abs / 6.0f : 1.0f);
            scales[aisp::nvfp4_vec16_scale_offset(r, block, reduction_dim)] = stored_scale;
            // Divide by the rounded UE4M3 scale because that is what cuBLASLt multiplies back.
            const float scale = static_cast<float>(stored_scale);

            for (int i = 0; i < FP4_BLOCK_SIZE; i += 2) {
                float v0 = row_input[block_start + i];
                float v1 = row_input[block_start + i + 1];

                __nv_fp4_storage_t fp4_0 =
                    __nv_cvt_float_to_fp4(v0 / scale, __NV_E2M1, cudaRoundNearest);
                __nv_fp4_storage_t fp4_1 =
                    __nv_cvt_float_to_fp4(v1 / scale, __NV_E2M1, cudaRoundNearest);

                const size_t packed_idx =
                    static_cast<size_t>(r) * packed_cols + (block_start + i) / 2;
                output_packed[packed_idx] = ((fp4_1 & 0x0F) << 4) | (fp4_0 & 0x0F);
            }
        }
    }
}

int main(int argc, char** argv) {
    NVTX_RANGE("main");
    std::cout << "=== Optimized FP4 Hardware Kernel (cuBLASLt Tensor Cores) ===" << std::endl;
    
    // Check GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (SM" << prop.major << "." << prop.minor << ")" << std::endl;
    
    // Matrix dimensions (aligned for tensor cores). Must match the baseline.
    constexpr int M = 1024, N = 1024, K = 1024;
    static_assert(M % 128 == 0, "M must be multiple of 128 for the scale layout");
    static_assert(N % 128 == 0, "N must be multiple of 128 for the scale layout");
    static_assert((K / FP4_BLOCK_SIZE) % 4 == 0,
                  "K/16 must be multiple of 4 for the scale layout");

    // FP4 packed sizes. Both operands are K-major: A is [M, K] and B^T is [N, K].
    const size_t packed_K = K / 2;
    const size_t elements_A_packed = static_cast<size_t>(M) * packed_K;
    const size_t elements_B_packed = static_cast<size_t>(N) * packed_K;
    const size_t elements_C = static_cast<size_t>(M) * N;
    const size_t num_scales_A = static_cast<size_t>(M) * (K / FP4_BLOCK_SIZE);
    const size_t num_scales_B = static_cast<size_t>(N) * (K / FP4_BLOCK_SIZE);

    // Host allocation. h_B_fp32 stores B^T as row-major [N, K]. The baseline
    // kernel reads the same [N, K] view, so both variants compute the same C.
    std::vector<float> h_A_fp32(static_cast<size_t>(M) * K);
    std::vector<float> h_B_fp32(static_cast<size_t>(N) * K);
    std::vector<uint8_t> h_A_packed(elements_A_packed);
    std::vector<uint8_t> h_B_packed(elements_B_packed);
    std::vector<__nv_fp8_e4m3> h_A_scales(num_scales_A);
    std::vector<__nv_fp8_e4m3> h_B_scales(num_scales_B);
    std::vector<__half> h_C(elements_C);

    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& v : h_A_fp32) {
        NVTX_RANGE("setup");
        v = dis(gen);
    }
    for (auto& v : h_B_fp32) {
        NVTX_RANGE("setup");
        v = dis(gen);
    }

    // Quantize to NVFP4 along K, the reduction dimension of both operands.
    quantize_to_nvfp4_kmajor(h_A_fp32.data(), h_A_packed.data(), h_A_scales.data(), M, K);
    quantize_to_nvfp4_kmajor(h_B_fp32.data(), h_B_packed.data(), h_B_scales.data(), N, K);

    // Device allocation
    uint8_t *d_A = nullptr, *d_B = nullptr;
    __nv_fp8_e4m3 *d_A_scales = nullptr, *d_B_scales = nullptr;
    __half *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, elements_A_packed));
    CUDA_CHECK(cudaMalloc(&d_B, elements_B_packed));
    CUDA_CHECK(cudaMalloc(&d_A_scales, num_scales_A * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_B_scales, num_scales_B * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_C, elements_C * sizeof(__half)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A_packed.data(), elements_A_packed, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_packed.data(), elements_B_packed, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_scales, h_A_scales.data(), num_scales_A * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_scales, h_B_scales.data(), num_scales_B * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize cuBLASLt
    cublasLtHandle_t ltHandle;
    CUBLASLT_CHECK(cublasLtCreate(&ltHandle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Create matmul descriptor
    cublasLtMatmulDesc_t matmulDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Set scale mode
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    void* d_A_scales_ptr = d_A_scales;
    void* d_B_scales_ptr = d_B_scales;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_A_scales_ptr, sizeof(d_A_scales_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_B_scales_ptr, sizeof(d_B_scales_ptr)));

    // Matrix layouts.
    //
    // NOTE: NVFP4 elements are stored packed (2 values per byte). The layout
    // leading dimensions are specified in *elements* (not bytes).
    // With transa=T, cuBLASLt reads A as the column-major K x M matrix, which is
    // row-major [M, K] storage. With transb=N, it reads B as the column-major
    // K x N matrix, which is row-major B^T [N, K] storage.
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_4F_E2M1, K, M, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_4F_E2M1, K, N, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16F, M, N, M));

    cublasLtOrder_t order = CUBLASLT_ORDER_COL;
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    float alpha = 1.0f, beta = 0.0f;

    // Workspace
    // cuBLASLt NVFP4 algorithms typically require a larger workspace.
    size_t workspaceSize = 1024 * 1024 * 64;
    void* d_workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));

    // Algorithm selection
    cublasLtMatmulPreference_t preference;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                         &workspaceSize, sizeof(workspaceSize)));

    constexpr int kMaxAlgos = 8;
    cublasLtMatmulHeuristicResult_t heuristicResults[kMaxAlgos] = {};
        NVTX_RANGE("compute_math");
    int returnedResults = 0;
    cublasStatus_t heuristicStatus = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
        preference, kMaxAlgos, heuristicResults, &returnedResults);
    
    if (heuristicStatus != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
        std::cerr << "No cuBLASLt NVFP4 algorithm found (status=" << heuristicStatus
                  << ", returnedResults=" << returnedResults << ")." << std::endl;
        return 1;
    }

    // Auto-tune over the returned candidates: rank-0 is not always fastest, especially on the
    // newer NVFP4 path (the ch09 FP4 GEMM saw candidate #3/#4 beat #0).
    const cublasLtMatmulAlgo_t* bestAlgo = &heuristicResults[0].algo;
    {
        auto run_lt = [&](const cublasLtMatmulAlgo_t* algo) {
            CUBLASLT_CHECK(cublasLtMatmul(ltHandle, matmulDesc, &alpha,
                                          d_A, layoutA, d_B, layoutB, &beta,
                                          d_C, layoutC, d_C, layoutC,
                                          algo, d_workspace, workspaceSize, stream));
        };
        cudaEvent_t ts, te;
        CUDA_CHECK(cudaEventCreate(&ts));
        CUDA_CHECK(cudaEventCreate(&te));
        int bestIdx = 0;
        float bestMs = 1e30f;
        for (int a = 0; a < returnedResults; ++a) {
            run_lt(&heuristicResults[a].algo);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaEventRecord(ts, stream));
            for (int w = 0; w < 3; ++w) run_lt(&heuristicResults[a].algo);
            CUDA_CHECK(cudaEventRecord(te, stream));
            CUDA_CHECK(cudaEventSynchronize(te));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ts, te));
            if (ms < bestMs) { bestMs = ms; bestIdx = a; }
        }
        CUDA_CHECK(cudaEventDestroy(ts));
        CUDA_CHECK(cudaEventDestroy(te));
        bestAlgo = &heuristicResults[bestIdx].algo;
        std::cout << "FP4 hardware GEMM: auto-tuned over " << returnedResults
                  << " candidates, selected #" << bestIdx << std::endl;
    }

    // Warmup
    for (int i = 0; i < 5; ++i) {
        NVTX_RANGE("warmup");
        CUBLASLT_CHECK(cublasLtMatmul(ltHandle, matmulDesc, &alpha,
                                       d_A, layoutA, d_B, layoutB, &beta,
                                       d_C, layoutC, d_C, layoutC,
                                       bestAlgo, d_workspace, workspaceSize, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 100;
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        NVTX_RANGE("compute_math:ltmatmul");
        CUBLASLT_CHECK(cublasLtMatmul(ltHandle, matmulDesc, &alpha,
                                       d_A, layoutA, d_B, layoutB, &beta,
                                       d_C, layoutC, d_C, layoutC,
                                       bestAlgo, d_workspace, workspaceSize, stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iterations;

    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e9);

    std::cout << "TIME_MS: " << avg_ms << std::endl;

    // Optional dump for harness verification: --dump-output <path>
    std::string dump_path;
    for (int i = 1; i < argc; ++i) {
        NVTX_RANGE("verify");
        std::string arg(argv[i]);
        if (arg == "--dump-output") {
            if (i + 1 >= argc) {
                std::cerr << "--dump-output requires a path argument" << std::endl;
                return 2;
            }
            dump_path = std::string(argv[i + 1]);
        }
    }
    if (!dump_path.empty()) {
        std::vector<__half> h_C(elements_C);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, elements_C * sizeof(__half), cudaMemcpyDeviceToHost));
        std::ofstream out(dump_path, std::ios::binary);
        if (!out) {
            std::cerr << "Failed to open dump path: " << dump_path << std::endl;
            return 2;
        }
        out.write(reinterpret_cast<const char*>(h_C.data()), h_C.size() * sizeof(__half));
        out.close();
    }

#ifdef VERIFY
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, elements_C * sizeof(__half), cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (const auto& v : h_C) {
        NVTX_RANGE("verify");
        checksum += static_cast<double>(__half2float(v));
    }
    checksum /= static_cast<double>(elements_C);
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
