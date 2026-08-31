// optimized_cublas_gemm_fp4_perchannel.cu -- cuBLASLt NVFP4 GEMM with per-channel output scaling
//
// Optimized uses a larger workspace budget to unlock faster algorithms.

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/headers/nvfp4_scale_layout.cuh"
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

constexpr int FP4_BLOCK_SIZE = 16;
constexpr int kM = 4096;
constexpr int kN = 4096;
constexpr int kK = 4096;
constexpr int kIterations = 10;
constexpr int kBatchCount = 1;
constexpr size_t kWorkspaceBytes = 64ull * 1024ull * 1024ull;

__global__ void apply_per_channel_scale(__half* __restrict__ output,
                                        const float* __restrict__ scales,
                                        int rows, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) {
        return;
    }
    // cuBLASLt's C layout is column-major with leading dimension `rows`.
    const size_t idx = static_cast<size_t>(col) * rows + row;
    float val = __half2float(output[idx]) * scales[col];
    output[idx] = __float2half(val);
}

void quantize_to_nvfp4_kmajor(const float* input,
                              uint8_t* output_packed,
                              __nv_fp8_e4m3* scales,
                              int rows,
                              int reduction_dim,
                              int input_row_stride,
                              int input_reduction_stride) {
    NVTX_RANGE("setup:quantize_to_nvfp4");
    const int packed_cols = reduction_dim / 2;
    const int num_scale_cols = reduction_dim / FP4_BLOCK_SIZE;

    for (int r = 0; r < rows; ++r) {
        for (int block = 0; block < num_scale_cols; ++block) {
            const int block_start = block * FP4_BLOCK_SIZE;

            float max_abs = 0.0f;
            for (int i = 0; i < FP4_BLOCK_SIZE; ++i) {
                const size_t input_idx = static_cast<size_t>(r) * input_row_stride
                    + static_cast<size_t>(block_start + i) * input_reduction_stride;
                max_abs = std::max(max_abs, std::abs(input[input_idx]));
            }

            float scale = (max_abs > 0.0f) ? max_abs / 6.0f : 1.0f;
            scales[aisp::nvfp4_vec16_scale_offset(r, block, reduction_dim)] =
                __nv_fp8_e4m3(scale);

            for (int i = 0; i < FP4_BLOCK_SIZE; i += 2) {
                const size_t input_idx = static_cast<size_t>(r) * input_row_stride
                    + static_cast<size_t>(block_start + i) * input_reduction_stride;
                float v0 = input[input_idx];
                float v1 = input[input_idx + input_reduction_stride];

                __nv_fp4_storage_t fp4_0 =
                    __nv_cvt_float_to_fp4(v0 / scale, __NV_E2M1, cudaRoundNearest);
                __nv_fp4_storage_t fp4_1 =
                    __nv_cvt_float_to_fp4(v1 / scale, __NV_E2M1, cudaRoundNearest);

                int packed_idx = r * packed_cols + (block_start + i) / 2;
                output_packed[packed_idx] = ((fp4_1 & 0x0F) << 4) | (fp4_0 & 0x0F);
            }
        }
    }
}

int main() {
    NVTX_RANGE("main");
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 10) {
        std::cerr << "SKIPPED: NVFP4 requires SM100+." << std::endl;
        return 3;
    }

    static_assert(kM % 128 == 0, "M must be multiple of 128 for the scale layout");
    static_assert(kN % 128 == 0, "N must be multiple of 128 for the scale layout");
    static_assert((kK / FP4_BLOCK_SIZE) % 4 == 0,
                  "K/16 must be multiple of 4 for the scale layout");

    const size_t packed_K = kK / 2;
    const size_t elements_A_packed = static_cast<size_t>(kM) * packed_K;
    const size_t elements_B_packed = static_cast<size_t>(kN) * packed_K;
    const size_t elements_C = static_cast<size_t>(kM) * kN;

    const size_t num_scales_A = static_cast<size_t>(kM) * (kK / FP4_BLOCK_SIZE);
    const size_t num_scales_B = static_cast<size_t>(kN) * (kK / FP4_BLOCK_SIZE);

    std::vector<float> h_A_fp32(kM * kK * kBatchCount);
    std::vector<float> h_B_fp32(kK * kN * kBatchCount);
    std::vector<uint8_t> h_A_packed(elements_A_packed * kBatchCount);
    std::vector<uint8_t> h_B_packed(elements_B_packed * kBatchCount);
    std::vector<__nv_fp8_e4m3> h_A_scales(num_scales_A * kBatchCount);
    std::vector<__nv_fp8_e4m3> h_B_scales(num_scales_B * kBatchCount);
    std::vector<__half> h_C(elements_C * kBatchCount);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    {
        NVTX_RANGE("setup:fill_host_a");
        for (auto& v : h_A_fp32) {
            v = dis(gen);
        }
    }
    {
        NVTX_RANGE("setup:fill_host_b");
        for (auto& v : h_B_fp32) {
        v = dis(gen);
        }
    }

    for (int batch = 0; batch < kBatchCount; ++batch) {
        NVTX_RANGE("setup:quantize_batch");
        quantize_to_nvfp4_kmajor(h_A_fp32.data() + batch * kM * kK,
                                 h_A_packed.data() + batch * elements_A_packed,
                                 h_A_scales.data() + batch * num_scales_A,
                                 kM, kK, kK, 1);
        // Read row-major B[K,N] as B^T[N,K], making each reduction row
        // K-contiguous in the packed operand and its scale-factor layout.
        quantize_to_nvfp4_kmajor(h_B_fp32.data() + batch * kK * kN,
                                 h_B_packed.data() + batch * elements_B_packed,
                                 h_B_scales.data() + batch * num_scales_B,
                                 kN, kK, 1, kN);
    }

    std::fill(h_C.begin(), h_C.end(), __float2half(0.0f));

    uint8_t* d_A = nullptr;
    uint8_t* d_B = nullptr;
    __nv_fp8_e4m3* d_A_scales = nullptr;
    __nv_fp8_e4m3* d_B_scales = nullptr;
    __half* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, elements_A_packed * kBatchCount));
    CUDA_CHECK(cudaMalloc(&d_B, elements_B_packed * kBatchCount));
    CUDA_CHECK(cudaMalloc(&d_A_scales, num_scales_A * kBatchCount * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_B_scales, num_scales_B * kBatchCount * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_C, elements_C * kBatchCount * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_packed.data(), elements_A_packed * kBatchCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_packed.data(), elements_B_packed * kBatchCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_scales, h_A_scales.data(), num_scales_A * kBatchCount * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_scales, h_B_scales.data(), num_scales_B * kBatchCount * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), elements_C * kBatchCount * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    cublasLtHandle_t ltHandle;
    CUBLASLT_CHECK(cublasLtCreate(&ltHandle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cublasLtMatmulDesc_t matmulDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    void* d_A_scales_ptr = d_A_scales;
    void* d_B_scales_ptr = d_B_scales;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_A_scales_ptr, sizeof(d_A_scales_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_B_scales_ptr, sizeof(d_B_scales_ptr)));

    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_4F_E2M1, kM, kK, kM));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_4F_E2M1, kK, kN, kK));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16F, kM, kN, kM));

    const long long strideA = static_cast<long long>(elements_A_packed);
    const long long strideB = static_cast<long long>(elements_B_packed);
    const long long strideC = static_cast<long long>(elements_C);
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &kBatchCount, sizeof(kBatchCount)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &kBatchCount, sizeof(kBatchCount)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &kBatchCount, sizeof(kBatchCount)));

    cublasLtMatmulPreference_t preference;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
                                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                         &kWorkspaceBytes,
                                                         sizeof(kWorkspaceBytes)));

    constexpr int kHeuristicCandidates = 8;
    cublasLtMatmulHeuristicResult_t heuristicResults[kHeuristicCandidates]{};
    int returnedResults = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
        preference, kHeuristicCandidates, heuristicResults, &returnedResults));
    if (returnedResults == 0) {
        std::cerr << "No suitable cuBLASLt algorithm found for optimized NVFP4 GEMM." << std::endl;
        return 1;
    }

    void* d_workspace = nullptr;
    if (kWorkspaceBytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, kWorkspaceBytes));
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    cudaEvent_t tuneStart, tuneStop;
    CUDA_CHECK(cudaEventCreate(&tuneStart));
    CUDA_CHECK(cudaEventCreate(&tuneStop));
    int bestIdx = -1;
    float bestMs = std::numeric_limits<float>::max();

    for (int candidate = 0; candidate < returnedResults; ++candidate) {
        if (heuristicResults[candidate].workspaceSize > kWorkspaceBytes) {
            continue;
        }

        for (int batch = 0; batch < kBatchCount; ++batch) {
            const size_t offset_A = batch * elements_A_packed;
            const size_t offset_B = batch * elements_B_packed;
            const size_t offset_C = batch * elements_C;
            CUBLASLT_CHECK(cublasLtMatmul(
                ltHandle, matmulDesc,
                &alpha,
                d_A + offset_A, layoutA,
                d_B + offset_B, layoutB,
                &beta,
                d_C + offset_C, layoutC,
                d_C + offset_C, layoutC,
                &heuristicResults[candidate].algo,
                d_workspace, kWorkspaceBytes,
                stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaEventRecord(tuneStart, stream));
        for (int trial = 0; trial < 3; ++trial) {
            for (int batch = 0; batch < kBatchCount; ++batch) {
                const size_t offset_A = batch * elements_A_packed;
                const size_t offset_B = batch * elements_B_packed;
                const size_t offset_C = batch * elements_C;
                CUBLASLT_CHECK(cublasLtMatmul(
                    ltHandle, matmulDesc,
                    &alpha,
                    d_A + offset_A, layoutA,
                    d_B + offset_B, layoutB,
                    &beta,
                    d_C + offset_C, layoutC,
                    d_C + offset_C, layoutC,
                    &heuristicResults[candidate].algo,
                    d_workspace, kWorkspaceBytes,
                    stream));
            }
        }
        CUDA_CHECK(cudaEventRecord(tuneStop, stream));
        CUDA_CHECK(cudaEventSynchronize(tuneStop));

        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, tuneStart, tuneStop));
        const float avgCandidateMs = elapsedMs / (3.0f * static_cast<float>(kBatchCount));
        if (avgCandidateMs < bestMs) {
            bestMs = avgCandidateMs;
            bestIdx = candidate;
        }
    }

    CUDA_CHECK(cudaEventDestroy(tuneStart));
    CUDA_CHECK(cudaEventDestroy(tuneStop));

    if (bestIdx < 0) {
        std::cerr << "No viable cuBLASLt heuristic candidate within workspace budget." << std::endl;
        return 1;
    }
    const cublasLtMatmulAlgo_t* selectedAlgo = &heuristicResults[bestIdx].algo;

    for (int batch = 0; batch < kBatchCount; ++batch) {
        NVTX_RANGE("compute_math:ltmatmul");
        const size_t offset_A = batch * elements_A_packed;
        const size_t offset_B = batch * elements_B_packed;
        const size_t offset_C = batch * elements_C;
        CUBLASLT_CHECK(cublasLtMatmul(
            ltHandle, matmulDesc,
            &alpha,
            d_A + offset_A, layoutA,
            d_B + offset_B, layoutB,
            &beta,
            d_C + offset_C, layoutC,
            d_C + offset_C, layoutC,
            selectedAlgo,
            d_workspace, kWorkspaceBytes,
            stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    NVTX_RANGE("benchmark:ltmatmul_loop");
    for (int iter = 0; iter < kIterations; ++iter) {
        for (int batch = 0; batch < kBatchCount; ++batch) {
            const size_t offset_A = batch * elements_A_packed;
            const size_t offset_B = batch * elements_B_packed;
            const size_t offset_C = batch * elements_C;
            CUBLASLT_CHECK(cublasLtMatmul(
                ltHandle, matmulDesc,
                &alpha,
                d_A + offset_A, layoutA,
                d_B + offset_B, layoutB,
                &beta,
                d_C + offset_C, layoutC,
                d_C + offset_C, layoutC,
                selectedAlgo,
                d_workspace, kWorkspaceBytes,
                stream));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(kIterations * kBatchCount);
    std::cout << "cuBLASLt NVFP4 GEMM (optimized, per-channel): " << avg_ms << " ms" << std::endl;

    std::vector<float> h_scales(kN);
    std::mt19937 scale_gen(42);
    std::uniform_real_distribution<float> scale_dis(0.75f, 1.25f);
    {
        NVTX_RANGE("setup:fill_output_scales");
        for (auto& v : h_scales) {
        v = scale_dis(scale_gen);
        }
    }
    float* d_scales = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scales, kN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), kN * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((kN + block.x - 1) / block.x, (kM + block.y - 1) / block.y);
    for (int batch = 0; batch < kBatchCount; ++batch) {
        NVTX_RANGE("compute_kernel:apply_per_channel_scale");
        const size_t offset_C = batch * elements_C;
        apply_per_channel_scale<<<grid, block, 0, stream>>>(d_C + offset_C, d_scales, kM, kN);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_scales));

#ifdef VERIFY
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, elements_C * kBatchCount * sizeof(__half), cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    NVTX_RANGE("verify:checksum");
    for (size_t i = 0; i < elements_C * kBatchCount; ++i) {
        checksum += std::abs(static_cast<double>(__half2float(h_C[i])));
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(matmulDesc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutA));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutB));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutC));
    CUBLASLT_CHECK(cublasLtDestroy(ltHandle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (d_workspace) {
        CUDA_CHECK(cudaFree(d_workspace));
    }
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A_scales));
    CUDA_CHECK(cudaFree(d_B_scales));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
