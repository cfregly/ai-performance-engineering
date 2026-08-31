// Baseline threshold binary gated for Blackwell TMA comparisons.

#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#include "blackwell_guard.cuh"
#include "threshold_common.cuh"
#include "../core/common/nvtx_utils.cuh"

using namespace ch08;

namespace {

bool ensure_blackwell(int& exit_code) {
    cudaDeviceProp props{};
    cudaError_t err = cudaSuccess;
    if (is_blackwell_device(&props, &err, true)) {
        return true;
    }

    if (err == cudaSuccess && props.major > 0) {
        std::cerr << "SKIPPED: threshold_tma requires SM 10.x+ (Blackwell/GB), found SM "
                  << props.major << "." << props.minor << "\n";
    } else {
        std::cerr << "SKIPPED: threshold_tma requires Blackwell/GB GPUs ("
                  << cudaGetErrorString(err) << ")\n";
    }
    exit_code = 3;
    return false;
}

}  // namespace

int main() {
    NVTX_RANGE("main");
    int skip_code = 0;
    if (!ensure_blackwell(skip_code)) {
        return skip_code;
    }

    const int count = 1 << 25;  // 33M elements
    const float threshold = 0.25f;
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);

    std::vector<float> h_input(count);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < count; ++i) {
        NVTX_RANGE("setup");
        h_input[i] = dist(gen);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaMemcpyAsync(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    for (int i = 0; i < 5; ++i) {
        launch_threshold_naive(d_input, d_output, threshold, count, stream);
    }
    cudaDeviceSynchronize();

    const int iterations = 50;
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; ++i) {
        NVTX_RANGE("iteration");
        launch_threshold_naive(d_input, d_output, threshold, count, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    const float avg_ms = total_ms / iterations;

    std::cout << "=== Baseline Threshold (Blackwell gate) ===\n";
    std::cout << "Elements: " << count << " (" << bytes / 1e6 << " MB)\n";
    std::cout << "Average kernel time: " << avg_ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
