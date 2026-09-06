// Baseline HBM binary.

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "hbm_common.cuh"
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

using namespace ch08;

int main() {
    NVTX_RANGE("main");
    const int rows = 4096;
    const int cols = 2048;
    const size_t col_bytes = static_cast<size_t>(cols) * rows * sizeof(float);

    std::vector<float> host_matrix(rows * cols);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& value : host_matrix) {
        NVTX_RANGE("setup");
        value = dist(gen);
    }

    std::vector<float> host_col(cols * rows);
    for (int r = 0; r < rows; ++r) {
        NVTX_RANGE("setup");
        for (int c = 0; c < cols; ++c) {
            NVTX_RANGE("setup");
            host_col[c * rows + r] = host_matrix[r * cols + c];
        }
    }

    float* d_col = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_col, col_bytes);
    cudaMalloc(&d_output, rows * sizeof(float));
    cudaMemcpy(d_col, host_col.data(), col_bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 5; ++i) {
        NVTX_RANGE("iteration");
        launch_hbm_naive(d_col, d_output, rows, cols, 0);
    }
    cudaDeviceSynchronize();

    const int iterations = 30;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        NVTX_RANGE("iteration");
        launch_hbm_naive(d_col, d_output, rows, cols, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    const float avg_ms = total_ms / iterations;
    std::cout << "Baseline HBM: " << avg_ms << " ms\n";
    std::printf("TIME_MS: %.6f\n", avg_ms);

#ifdef VERIFY
    std::vector<float> host_output(rows, 0.0f);
    cudaMemcpy(host_output.data(), d_output, rows * sizeof(float), cudaMemcpyDeviceToHost);
    double checksum = 0.0;
    for (float v : host_output) {
        checksum += std::abs(v);
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_col);
    cudaFree(d_output);
    return 0;
}
