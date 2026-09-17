// Build instructions are in checks/README.md.
#define main fast_cu_sum_benchmark_main
#include "sum.cu"
#undef main

#include <climits>
#include <vector>

int64_t hostOracle(const std::vector<int>& input) {
    int64_t total = 0;
    for (int value : input) total += value;
    return total;
}

bool runCase(const char *name, const std::vector<int>& input) {
    const int N = static_cast<int>(input.size());
    const int64_t expected = hostOracle(input);
    int *d_in = nullptr;
    int *d_out = nullptr;
    int result = 0;
    checkCuda(cudaMalloc(&d_in, input.size() * sizeof(int)), "test allocate input");
    checkCuda(cudaMalloc(&d_out, sizeof(int)), "test allocate output");
    checkCuda(
        cudaMemcpy(d_in, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice),
        "test copy input");

    bool passed = true;
    for (int kernel = 1; kernel <= NUM_KERNELS; ++kernel) {
        for (int repetition = 0; repetition < 3; ++repetition) {
            const int poison = repetition % 2 == 0 ? INT_MAX : INT_MIN;
            checkCuda(
                cudaMemcpy(d_out, &poison, sizeof(int), cudaMemcpyHostToDevice),
                "poison reduction output");
            kernelDispatch(kernel, d_in, d_out, &result, N);
            checkCuda(
                cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost),
                "copy reduction output");
            if (static_cast<int64_t>(result) != expected) {
                std::cerr << name << " kernel" << kernel << " repetition" << repetition
                          << " mismatch: expected " << expected << ", got " << result
                          << std::endl;
                passed = false;
            }
        }
    }

    checkCuda(cudaFree(d_in), "test free input");
    checkCuda(cudaFree(d_out), "test free output");
    return passed;
}

int main() {
    constexpr int N = 1 << 20;
    std::vector<int> all_ones(N, 1);
    std::vector<int> balanced(N);
    for (int i = 0; i < N; ++i) balanced[i] = i % 3 - 1;

    const bool passed = runCase("all_ones", all_ones)
        && runCase("balanced_minus_one_zero_one", balanced);
    if (passed) std::cout << "sum reset regression: PASS" << std::endl;
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
