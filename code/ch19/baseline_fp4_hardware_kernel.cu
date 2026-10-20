/*
 * Baseline FP4 hardware kernel benchmark (manual GEMM).
 *
 * This is the intentionally-slow baseline for Chapter 19's FP4 kernel example.
 * It performs an NVFP4 (E2M1) GEMM by unpacking FP4 values and accumulating in
 * FP32 inside a naive CUDA kernel.
 *
 * The optimized variant (optimized_fp4_hardware_kernel.cu) performs the *same*
 * GEMM using cuBLASLt NVFP4 tensor cores.
 *
 * Harness integration:
 *   --dump-output <path> writes C (FP16, column-major) to disk for verification.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

constexpr int FP4_BLOCK_SIZE = 16;

static void quantize_to_nvfp4(
    const float* input,
    uint8_t* output_packed,
    __nv_fp8_e4m3* scales,
    int rows,
    int cols) {
    // Pack as 2 FP4 values per byte, along the contiguous (row-major) dimension.
    // This matches the packing used by NVIDIA cuBLASLt NVFP4 examples.
    const int packed_cols = cols / 2;
    const int num_scale_cols = cols / FP4_BLOCK_SIZE;
    // Compute per-row scales (block-scaled along the column dimension).
    for (int r = 0; r < rows; ++r) {
        NVTX_RANGE("iteration");
        for (int block = 0; block < num_scale_cols; ++block) {
            NVTX_RANGE("iteration");
            const int block_start = block * FP4_BLOCK_SIZE;

            float max_abs = 0.0f;
            for (int i = 0; i < FP4_BLOCK_SIZE; ++i) {
                NVTX_RANGE("iteration");
                max_abs = std::max(max_abs, std::abs(input[r * cols + block_start + i]));
            }

            float scale = (max_abs > 0.0f) ? (max_abs / 6.0f) : 1.0f;
            scales[r * num_scale_cols + block] = __nv_fp8_e4m3(scale);
        }
    }

    // Pack in row-major order: consecutive elements are adjacent columns.
    for (int r = 0; r < rows; ++r) {
        NVTX_RANGE("iteration");
        for (int block = 0; block < num_scale_cols; ++block) {
            NVTX_RANGE("iteration");
            const int block_start = block * FP4_BLOCK_SIZE;
            float scale = static_cast<float>(scales[r * num_scale_cols + block]);
            for (int i = 0; i < FP4_BLOCK_SIZE; i += 2) {
                NVTX_RANGE("iteration");
                float v0 = input[r * cols + block_start + i];
                float v1 = input[r * cols + block_start + i + 1];

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

__device__ __forceinline__ float fp4_to_float(__nv_fp4_storage_t value) {
    __half_raw raw = __nv_cvt_fp4_to_halfraw(value, __NV_E2M1);
    return __half2float(__half(raw));
}

__device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 value) {
    return static_cast<float>(value);
}

__global__ void nvfp4_gemm_manual(
    const uint8_t* A_packed,
    const uint8_t* B_packed,
    const __nv_fp8_e4m3* A_scales,
    const __nv_fp8_e4m3* B_scales,
    __half* C,
    int M_dim,
    int N_dim,
    int K_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M_dim || col >= N_dim) {
        return;
    }

    // A is row-major [M, K] and B is stored as row-major B^T [N, K]. Both
    // operands are packed and block-scaled along K, the same K-major storage
    // the optimized cuBLASLt variant reads.
    const int packed_K = K_dim / 2;
    const int scale_cols_K = K_dim / FP4_BLOCK_SIZE;

    float sum = 0.0f;
    for (int k = 0; k < K_dim; ++k) {
        uint8_t a_byte = A_packed[row * packed_K + (k / 2)];
        // Row col of B^T holds column col of B.
        uint8_t b_byte = B_packed[col * packed_K + (k / 2)];

        __nv_fp4_storage_t a_raw =
            (k % 2 == 0) ? static_cast<__nv_fp4_storage_t>(a_byte & 0x0F)
                         : static_cast<__nv_fp4_storage_t>((a_byte >> 4) & 0x0F);
        __nv_fp4_storage_t b_raw =
            (k % 2 == 0) ? static_cast<__nv_fp4_storage_t>(b_byte & 0x0F)
                         : static_cast<__nv_fp4_storage_t>((b_byte >> 4) & 0x0F);

        float a_scale = fp8_to_float(A_scales[row * scale_cols_K + (k / FP4_BLOCK_SIZE)]);
        float b_scale = fp8_to_float(B_scales[col * scale_cols_K + (k / FP4_BLOCK_SIZE)]);

        float a_val = fp4_to_float(a_raw) * a_scale;
        float b_val = fp4_to_float(b_raw) * b_scale;
        sum += a_val * b_val;
    }

    // Column-major output (matches cuBLASLt default layout).
    C[row + col * M_dim] = __float2half(sum);
}

static std::string parse_dump_path(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        NVTX_RANGE("iteration");
        std::string arg(argv[i]);
        if (arg == "--dump-output") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--dump-output requires a path argument");
            }
            return std::string(argv[i + 1]);
        }
    }
    return "";
}

int main(int argc, char** argv) {
    NVTX_RANGE("main");
    try {
        std::string dump_path = parse_dump_path(argc, argv);

        std::cout << "=== Baseline FP4 Hardware Kernel (manual GEMM) ===" << std::endl;

        const int M = 1024;
        const int N = 1024;
        const int K = 1024;

        // Both operands are K-major: A is [M, K] and B^T is [N, K], matching
        // the optimized cuBLASLt variant.
        const int packed_K = K / 2;
        const size_t elements_A_packed = static_cast<size_t>(M) * packed_K;
        const size_t elements_B_packed = static_cast<size_t>(N) * packed_K;
        const size_t elements_C = static_cast<size_t>(M) * N;
        const size_t num_scales_A = static_cast<size_t>(M) * (K / FP4_BLOCK_SIZE);
        const size_t num_scales_B = static_cast<size_t>(N) * (K / FP4_BLOCK_SIZE);

        // h_B_fp32 holds B^T as row-major [N, K]. The seed and fill order (A,
        // then B^T) match the optimized variant, so both quantize the same values.
        std::vector<float> h_A_fp32(static_cast<size_t>(M) * K);
        std::vector<float> h_B_fp32(static_cast<size_t>(N) * K);
        std::vector<uint8_t> h_A_packed(elements_A_packed);
        std::vector<uint8_t> h_B_packed(elements_B_packed);
        std::vector<__nv_fp8_e4m3> h_A_scales(num_scales_A);
        std::vector<__nv_fp8_e4m3> h_B_scales(num_scales_B);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (auto& v : h_A_fp32) {
            NVTX_RANGE("iteration");
            v = dis(gen);
        }
        for (auto& v : h_B_fp32) {
            NVTX_RANGE("iteration");
            v = dis(gen);
        }

        // Quantize both operands along K, the reduction dimension.
        quantize_to_nvfp4(h_A_fp32.data(), h_A_packed.data(), h_A_scales.data(), M, K);
        quantize_to_nvfp4(h_B_fp32.data(), h_B_packed.data(), h_B_scales.data(), N, K);

        uint8_t* d_A = nullptr;
        uint8_t* d_B = nullptr;
        __nv_fp8_e4m3* d_A_scales = nullptr;
        __nv_fp8_e4m3* d_B_scales = nullptr;
        __half* d_C = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, elements_A_packed));
        CUDA_CHECK(cudaMalloc(&d_B, elements_B_packed));
        CUDA_CHECK(cudaMalloc(&d_A_scales, num_scales_A * sizeof(__nv_fp8_e4m3)));
        CUDA_CHECK(cudaMalloc(&d_B_scales, num_scales_B * sizeof(__nv_fp8_e4m3)));
        CUDA_CHECK(cudaMalloc(&d_C, elements_C * sizeof(__half)));

        CUDA_CHECK(cudaMemcpy(d_A, h_A_packed.data(), elements_A_packed, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B_packed.data(), elements_B_packed, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_A_scales, h_A_scales.data(), num_scales_A * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_B_scales, h_B_scales.data(), num_scales_B * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        nvfp4_gemm_manual<<<grid, block>>>(d_A, d_B, d_A_scales, d_B_scales, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        const int iterations = 10;
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            NVTX_RANGE("compute_kernel:nvfp4_gemm_manual");
            nvfp4_gemm_manual<<<grid, block>>>(d_A, d_B, d_A_scales, d_B_scales, d_C, M, N, K);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        float avg_ms = total_ms / iterations;

        std::cout << "TIME_MS: " << avg_ms << std::endl;

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
        std::vector<__half> h_C_verify(elements_C);
        CUDA_CHECK(cudaMemcpy(h_C_verify.data(), d_C, elements_C * sizeof(__half), cudaMemcpyDeviceToHost));
        double checksum = 0.0;
        for (const auto& v : h_C_verify) {
            NVTX_RANGE("verify");
            checksum += static_cast<double>(__half2float(v));
        }
        checksum /= static_cast<double>(elements_C);
        VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_A_scales));
        CUDA_CHECK(cudaFree(d_B_scales));
        CUDA_CHECK(cudaFree(d_C));
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << std::endl;
        return 1;
    }
}
