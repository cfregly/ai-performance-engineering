// Negative tests for the copied source. See checks/README.md.
#define main fast_cu_sum_benchmark_main
#include "sum.cu"
#undef main

#include <string>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const std::string mode = argv[1];
    int input[4] = {1, 1, 1, 1};
    int output = 0;
    int *d_in = nullptr;
    int *d_out = nullptr;
    checkCuda(cudaMalloc(&d_in, sizeof(input)), "allocate test input");
    checkCuda(cudaMalloc(&d_out, sizeof(output)), "allocate test output");
    checkCuda(cudaMemcpy(d_in, input, sizeof(input), cudaMemcpyHostToDevice), "copy test input");
    bool passed;
    if (mode == "kernel-mismatch") {
        passed = sumKernelCall(2, d_in, d_out, &output, 4, 1, 5);
    } else if (mode == "cub-mismatch") {
        passed = sumCubCall(d_in, d_out, &output, 4, 1, 5);
    } else if (mode == "invalid-length") {
        kernelDispatch(2, d_in, d_out, &output, 3);
        passed = true;
    } else if (mode == "invalid-kernel") {
        kernelDispatch(5, d_in, d_out, &output, 4);
        passed = true;
    } else {
        return 2;
    }
    checkCuda(cudaFree(d_in), "free test input");
    checkCuda(cudaFree(d_out), "free test output");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
