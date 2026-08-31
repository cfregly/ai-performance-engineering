// Executes production scalar/cluster controls; never substitutes a host model.
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>
#define main feature_demo_main
#if FEATURE_VALIDATION_CASE == 0
#include "../../core/benchmark/blackwell_optimizations/blackwell_optimizations/test_tma.cu"
using Input = float;
#else
#include "../../core/benchmark/blackwell_optimizations/blackwell_optimizations/test_all_features.cu"
using Input = __nv_fp8_e4m3;
#endif
#undef main

constexpr int kGuard = 8;
constexpr float kCanary = -12345.0f;
constexpr std::array<std::array<int, 3>, 7> kShapes{{
    {1, 1, 1}, {33, 67, 31}, {65, 35, 33}, {97, 129, 96},
    {95, 67, 129}, {129, 193, 257}, {64, 32, 512}}};

template<int Stages>
bool check(int m, int n, int k) {
    std::vector<Input> a(m * k), b(k * n);
    for (int i = 0; i < m * k; ++i) a[i] = Input(float((i * 7) % 17 - 8) / 8);
    for (int i = 0; i < k * n; ++i) b[i] = Input(float((i * 11) % 19 - 9) / 8);
    std::vector<float> actual(m * n + 2 * kGuard, kCanary);
    std::vector<float> expected(m * n, 0);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double sum = 0;
            for (int inner = 0; inner < k; ++inner)
                sum += double(float(a[row * k + inner])) * double(float(b[inner * n + col]));
            expected[row * n + col] = float(sum);
        }
    }
    Input *da, *db;
    float* dc;
    CUDA_CHECK(cudaMalloc(&da, a.size() * sizeof(Input)));
    CUDA_CHECK(cudaMalloc(&db, b.size() * sizeof(Input)));
    CUDA_CHECK(cudaMalloc(&dc, actual.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(Input), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(Input), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, actual.data(), actual.size() * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(32, 32);
    dim3 grid((n + 31) / 32, (m + 31) / 32);
#if FEATURE_VALIDATION_CASE == 1
    grid.x = ((grid.x + 1) / 2) * 2;
    grid.y = ((grid.y + 1) / 2) * 2;
#endif
    for (int repeat = 0; repeat < 5; ++repeat) {
#if FEATURE_VALIDATION_CASE == 0
        if constexpr (Stages == 0) {
            tma_gemm_kernel<32, 32, 32><<<grid, block>>>(da, db, dc + kGuard, m, n, k);
        } else {
            tma_pipelined_gemm_kernel<32, 32, 32, Stages><<<grid, block>>>(da, db, dc + kGuard, m, n, k);
        }
#else
        blackwell_ultra_gemm_kernel<32, 32, 32, Stages><<<grid, block>>>(da, db, dc + kGuard, m, n, k);
#endif
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), dc, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    bool passed = true;
    for (int i = 0; i < kGuard; ++i) {
        if (actual[i] != kCanary || actual[kGuard + m * n + i] != kCanary) passed = false;
    }
    // This input family is exactly representable in FP8/FP32. Products are
    // multiples of 1/64 and K<=512; every FP32 partial sum is exact.
    for (int i = 0; i < m * n; ++i) {
        if (!std::isfinite(actual[kGuard + i]) || actual[kGuard + i] != expected[i]) {
            std::fprintf(stderr, "mismatch stages=%d shape=%d,%d,%d index=%d actual=%.9g expected=%.9g\n",
                         Stages, m, n, k, i, actual[kGuard + i], expected[i]);
            passed = false;
            break;
        }
    }
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    std::printf("case=%d buffers=%d shape=%d,%d,%d full_output_and_canaries=%s\n",
                FEATURE_VALIDATION_CASE, Stages, m, n, k, passed ? "PASS" : "FAIL");
    return passed;
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::puts("UNSUPPORTED: CUDA device unavailable");
        return 3;
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major * 10 + prop.minor != FEATURE_VALIDATION_TARGET) {
        std::puts("UNSUPPORTED: requested and actual CUDA targets differ");
        return 3;
    }
#if FEATURE_VALIDATION_CASE == 1
    int clusters = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clusters, cudaDevAttrClusterLaunch, 0));
    if (!clusters) {
        std::puts("UNSUPPORTED: cluster launch unavailable");
        return 3;
    }
#endif
    int checks = 0;
    for (const auto& shape : kShapes) {
        const int m = shape[0], n = shape[1], k = shape[2];
#if FEATURE_VALIDATION_CASE == 0
        if (!check<0>(m, n, k)) return 1;
        ++checks;
#endif
        if (!check<1>(m, n, k) || !check<2>(m, n, k) || !check<4>(m, n, k)) return 1;
        checks += 3;
    }
    std::printf("FEATURE_CUDA_PASS case=%d checks=%d full_output_and_canaries=checked\n",
                FEATURE_VALIDATION_CASE, checks);
    return 0;
}
