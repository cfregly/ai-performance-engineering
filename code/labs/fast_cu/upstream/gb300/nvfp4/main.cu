#ifndef NVFP4_GEMM_HEADER
#define NVFP4_GEMM_HEADER "gemm9.cuh"
#endif

#include NVFP4_GEMM_HEADER

#include <cublasLt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define CUBLAS_CHECK(expr) do {                                      \
    cublasStatus_t _s = (expr);                                    \
    if (_s != CUBLAS_STATUS_SUCCESS) {                             \
        std::fprintf(stderr, "cuBLASLt error %d at %s:%d\n",     \
                     int(_s), __FILE__, __LINE__);                 \
        std::abort();                                              \
    }                                                              \
} while (0)

// Shared correctness, cuBLASLt, and benchmark harness. The selected
// gemmN.cuh above contains only the implementation for that rung.
namespace host {

// ---- NVFP4 quantization -----------------------------------------------------
// The format: 4-bit E2M1 values, one UE4M3 scale per 16 consecutive K
// elements. Scale = E4M3-quantized amax/6 (NOT power-of-two rounded).

constexpr float E2M1_MAX = 6.0f;

inline float fp4_e2m1_decode(uint8_t nibble) {
    static constexpr float lut[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return lut[nibble & 0xF];
}

// Round-to-nearest-EVEN, matching the device-side cvt.rn.satfinite.e2m1x2:
// at each midpoint pick the code whose LSB is 0.
inline uint8_t fp4_e2m1_quantize(float x) {
    const float boundaries[7] = { 0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f };
    float ax = std::fabs(x);
    if (ax > E2M1_MAX) ax = E2M1_MAX;
    int idx = 0;
    for (int b = 0; b < 7; ++b) {
        if (ax > boundaries[b]) {
            ++idx;
        } else if (ax == boundaries[b]) {
            if ((b & 1) == 1) ++idx;   // tie: round to the even code
            break;
        } else {
            break;
        }
    }
    uint8_t code = uint8_t(idx);
    if (x < 0 && idx != 0) code |= 0x8;
    return code;
}

inline uint8_t fp8_e4m3_quantize(float x) {
    if (!std::isfinite(x)) return 0x7F;
    if (x > 448.0f)  x = 448.0f;
    if (x < -448.0f) x = -448.0f;
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    const uint32_t sign = (bits >> 31) & 0x1;
    const int32_t  exp  = int32_t((bits >> 23) & 0xFF) - 127;
    const uint32_t man  = bits & 0x7FFFFF;
    if (x == 0.0f) return uint8_t(sign << 7);
    if (exp < -9) return uint8_t(sign << 7);
    int32_t e = exp + 7;
    uint32_t m;
    if (e <= 0) {
        const uint32_t shift = uint32_t(1 - e);
        if (shift > 23) return uint8_t(sign << 7);
        m = ((man | (1u << 23)) + (1u << (shift + 19))) >> (shift + 20);
        e = 0;
    } else {
        m = (man + (1u << 19)) >> 20;
        if (m >= 8) { m = 0; ++e; }
    }
    if (e >= 16) { e = 15; m = 6; }   // saturate at 448
    return uint8_t((sign << 7) | (uint32_t(e) << 3) | (m & 0x7));
}

inline float fp8_e4m3_decode(uint8_t code) {
    if ((code & 0x7F) == 0x7F) return std::nanf("");
    const uint32_t sign = (code >> 7) & 0x1;
    const uint32_t exp  = (code >> 3) & 0xF;
    const uint32_t man  = code & 0x7;
    float v;
    if (exp == 0) v = std::ldexp(float(man), -9);
    else          v = std::ldexp(1.0f + float(man) / 8.0f, int(exp) - 7);
    return sign ? -v : v;
}

inline float ue4m3_byte_to_sf(uint8_t code) {
    return fp8_e4m3_decode(uint8_t(code & 0x7Fu));
}

// (M, K) float row-major -> packed E2M1 [M][K/2] + UE4M3 scales [M][K/16].
inline void cast_to_fp4_with_ue4m3(const float* x, int M, int K,
                                   uint8_t* fp4_out, uint8_t* sf_out) {
    const int n_chunks = K / 16;
    for (int m = 0; m < M; ++m) {
        for (int g = 0; g < n_chunks; ++g) {
            float amax = 1e-4f;
            for (int j = 0; j < 16; ++j)
                amax = std::max(amax, std::fabs(x[size_t(m) * K + g * 16 + j]));
            const uint8_t sf_byte =
                    uint8_t(fp8_e4m3_quantize(amax / E2M1_MAX) & 0x7Fu);
            sf_out[size_t(m) * n_chunks + g] = sf_byte;
            const float sf_rec = ue4m3_byte_to_sf(sf_byte);
            const float inv_sf = sf_rec > 0.0f ? 1.0f / sf_rec : 0.0f;
            for (int j = 0; j < 16; j += 2) {
                const uint8_t lo =
                        fp4_e2m1_quantize(x[size_t(m) * K + g * 16 + j] * inv_sf);
                const uint8_t hi =
                        fp4_e2m1_quantize(x[size_t(m) * K + g * 16 + j + 1] * inv_sf);
                fp4_out[size_t(m) * (K / 2) + (g * 16 + j) / 2] =
                        uint8_t((lo & 0xF) | ((hi & 0xF) << 4));
            }
        }
    }
}

// Dequantize one element back to float (the host reference reads this).
inline float dequant_nvfp4(const uint8_t* fp4, const uint8_t* sf,
                           int m, int k, int K) {
    const float sf_v = ue4m3_byte_to_sf(sf[size_t(m) * (K / 16) + k / 16]);
    const uint8_t byte = fp4[size_t(m) * (K / 2) + k / 2];
    const uint8_t nib  = (k & 1) ? uint8_t(byte >> 4) : uint8_t(byte & 0xF);
    return fp4_e2m1_decode(nib) * sf_v;
}

// ---- shared input buffers --------------------------------------------------
// A/B use the row-major packed-E2M1 bytes accepted by cuBLASLt, and both
// implementations receive the same standard VEC16_UE4M3 scale buffers. There
// is no kernel-specific repack.

constexpr size_t ALLOCATION_GUARD = 4096;

inline size_t vec16_sf_offset(int outer, int sf_inner,
                              int inner, int outer_dim) {
    const int sf_inner_dim = int((size_t(inner) + 63) / 64 * 4);
    const int outer_blocks = int((size_t(outer_dim) + 127) / 128);
    const int outer_block = outer / 128;
    if (outer_block >= outer_blocks || sf_inner >= sf_inner_dim) std::abort();
    const int inner_block = (sf_inner / 4) * 4;
    const size_t block =
            size_t(inner_block + outer_block * sf_inner_dim) * 128;
    const int local_outer = outer % 128;
    return block + size_t(local_outer % 32) * 16
                 + size_t(local_outer / 32) * 4 + size_t(sf_inner % 4);
}

inline std::vector<uint8_t> pack_vec16_scales(
        const std::vector<uint8_t>& logical, int outer, int inner) {
    const int logical_inner = inner / 16;
    if (inner % 16 != 0 ||
        logical.size() != size_t(outer) * logical_inner) std::abort();

    const int packed_inner = int((size_t(inner) + 63) / 64 * 4);
    const int packed_outer = int((size_t(outer) + 127) / 128 * 128);
    std::vector<uint8_t> packed(size_t(packed_inner) * packed_outer, 0);
    std::vector<uint8_t> seen(packed.size(), 0);
    for (int o = 0; o < outer; ++o) {
        for (int j = 0; j < logical_inner; ++j) {
            const size_t offset = vec16_sf_offset(o, j, inner, outer);
            if (offset >= packed.size() || seen[offset]) std::abort();
            seen[offset] = 1;
            packed[offset] = logical[size_t(o) * logical_inner + j];
        }
    }
    for (int o = 0; o < outer; ++o)
        for (int j = 0; j < logical_inner; ++j)
            if (packed[vec16_sf_offset(o, j, inner, outer)] !=
                logical[size_t(o) * logical_inner + j]) std::abort();
    return packed;
}

struct InputBuffers {
    std::vector<uint8_t> A, B, SFA, SFB;
    int K = 0;
    int ab_row_stride = 0;
};

inline InputBuffers make_input_buffers(
        const std::vector<uint8_t>& A_logical,
        const std::vector<uint8_t>& B_logical,
        const std::vector<uint8_t>& SFA_logical,
        const std::vector<uint8_t>& SFB_logical,
        int M, int N, int K) {
    if (K <= 0 || K % 16 != 0) std::abort();

    InputBuffers inputs;
    inputs.K = K;
    const int row_bytes = K / 2;
    inputs.ab_row_stride = (row_bytes + 15) & ~15;

    auto materialize_rows = [&](const std::vector<uint8_t>& logical, int rows) {
        if (logical.size() != size_t(rows) * row_bytes) std::abort();
        std::vector<uint8_t> physical(
                size_t(rows) * inputs.ab_row_stride, 0);
        for (int row = 0; row < rows; ++row)
            std::memcpy(&physical[size_t(row) * inputs.ab_row_stride],
                        &logical[size_t(row) * row_bytes], row_bytes);
        return physical;
    };

    // Public benchmark shapes require K%32, so the aligned TMA row stride adds
    // no bytes and A/B remain exactly the cuBLASLt buffers. The private
    // K%16-only correctness gate may add zero stride slack.
    inputs.A = materialize_rows(A_logical, M);
    inputs.B = materialize_rows(B_logical, N);
    inputs.SFA = pack_vec16_scales(SFA_logical, M, K);
    inputs.SFB = pack_vec16_scales(SFB_logical, N, K);
    return inputs;
}
// ---- device-resident operand set (one rotation slot) ----------------------

struct DeviceSlot {
    int M = 0, N = 0, K = 0;
    uint8_t *dA = nullptr, *dB = nullptr, *dSFA = nullptr, *dSFB = nullptr;
    __half* dC = nullptr;
    size_t a_sz = 0, b_sz = 0, sfa_sz = 0, sfb_sz = 0;
    CUtensorMap A_t, B_t, SFA_t, SFB_t;
};

inline DeviceSlot* make_device_slot(const InputBuffers& inputs, int M, int N) {
    auto* slot = new DeviceSlot{};
    slot->M = M;
    slot->N = N;
    slot->K = inputs.K;
    slot->a_sz = inputs.A.size();
    slot->b_sz = inputs.B.size();
    slot->sfa_sz = inputs.SFA.size();
    slot->sfb_sz = inputs.SFB.size();

    CUDA_CHECK(cudaMalloc(&slot->dA, slot->a_sz + ALLOCATION_GUARD));
    CUDA_CHECK(cudaMalloc(&slot->dB, slot->b_sz + ALLOCATION_GUARD));
    CUDA_CHECK(cudaMalloc(&slot->dSFA, slot->sfa_sz + ALLOCATION_GUARD));
    CUDA_CHECK(cudaMalloc(&slot->dSFB, slot->sfb_sz + ALLOCATION_GUARD));
    CUDA_CHECK(cudaMalloc(&slot->dC, size_t(M) * N * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(slot->dA, inputs.A.data(), slot->a_sz,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(slot->dB, inputs.B.data(), slot->b_sz,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(slot->dSFA, inputs.SFA.data(), slot->sfa_sz,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(slot->dSFB, inputs.SFB.data(), slot->sfb_sz,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(slot->dC, 0, size_t(M) * N * sizeof(__half)));

    // Poisoning the guard catches any TMA read beyond the declared buffers.
    CUDA_CHECK(cudaMemset(slot->dA + slot->a_sz, 0xA5, ALLOCATION_GUARD));
    CUDA_CHECK(cudaMemset(slot->dB + slot->b_sz, 0xA5, ALLOCATION_GUARD));
    CUDA_CHECK(cudaMemset(slot->dSFA + slot->sfa_sz, 0xA5, ALLOCATION_GUARD));
    CUDA_CHECK(cudaMemset(slot->dSFB + slot->sfb_sz, 0xA5, ALLOCATION_GUARD));

    slot->A_t = make_ab_tmap(
            slot->dA, M, inputs.K, inputs.ab_row_stride);
    slot->B_t = make_ab_tmap(
            slot->dB, N, inputs.K, inputs.ab_row_stride);
    slot->SFA_t = make_sf_tmap(slot->dSFA, M, inputs.K);
    slot->SFB_t = make_sf_tmap(slot->dSFB, N, inputs.K);
    return slot;
}

inline void free_device_slot(DeviceSlot* slot) {
    cudaFree(slot->dC);
    cudaFree(slot->dSFB);
    cudaFree(slot->dSFA);
    cudaFree(slot->dB);
    cudaFree(slot->dA);
    delete slot;
}
// ---- the input fixture -----------------------------------------------------

struct Fixture {
    int M = 0, N = 0, K = 0;
    std::vector<uint8_t> A_logical, B_logical;
    std::vector<uint8_t> SFA_logical, SFB_logical;
    std::vector<float> A_deq, B_deq;
    InputBuffers inputs;
};

inline Fixture make_fixture(int M, int N, int K, uint64_t seed, bool keep_deq) {
    Fixture fixture;
    fixture.M = M;
    fixture.N = N;
    fixture.K = K;

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> A_src(size_t(M) * K), B_src(size_t(N) * K);
    for (auto& value : A_src) value = dist(rng);
    for (auto& value : B_src) value = dist(rng);

    fixture.A_logical.resize(size_t(M) * (K / 2));
    fixture.B_logical.resize(size_t(N) * (K / 2));
    fixture.SFA_logical.resize(size_t(M) * (K / 16));
    fixture.SFB_logical.resize(size_t(N) * (K / 16));
    cast_to_fp4_with_ue4m3(A_src.data(), M, K,
                           fixture.A_logical.data(),
                           fixture.SFA_logical.data());
    cast_to_fp4_with_ue4m3(B_src.data(), N, K,
                           fixture.B_logical.data(),
                           fixture.SFB_logical.data());

    if (keep_deq) {
        fixture.A_deq.resize(size_t(M) * K);
        fixture.B_deq.resize(size_t(N) * K);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                fixture.A_deq[size_t(m) * K + k] = dequant_nvfp4(
                        fixture.A_logical.data(),
                        fixture.SFA_logical.data(), m, k, K);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                fixture.B_deq[size_t(n) * K + k] = dequant_nvfp4(
                        fixture.B_logical.data(),
                        fixture.SFB_logical.data(), n, k, K);
    }

    fixture.inputs = make_input_buffers(
            fixture.A_logical, fixture.B_logical,
            fixture.SFA_logical, fixture.SFB_logical, M, N, K);
    return fixture;
}
}  // namespace host

// ============================================================================
// The cuBLASLt baseline. Request one recommendation and use it for correctness
// and timing at every shape.
// ============================================================================
namespace bench {

constexpr size_t MIB = size_t(1) << 20;
constexpr size_t CUBLAS_WORKSPACE_LIMIT = 64 * MIB;
constexpr int SACRIFICIAL_PASSES = 2;
constexpr int TIMED_PASSES = 5;
// FP16 round-to-nearest is <= 2^-11 relative; allow four ulps plus 0.5 for
// reduction-order differences in the FP32 accumulation.
constexpr double FP16_ABS_TOL = 0.5;
constexpr double FP16_REL_TOL = 2e-3;

struct LtSlot {
    void *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
    void *sfa = nullptr, *sfb = nullptr, *workspace = nullptr;
    cublasLtMatmulDesc_t op = nullptr;
};

struct LtPlan {
    int M, N, K;
    cublasLtHandle_t handle = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr;
    cublasLtMatrixLayout_t out_layout = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;

    LtPlan(int m, int n, int k) : M(m), N(n), K(k) {
        CUBLAS_CHECK(cublasLtCreate(&handle));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_4F_E2M1, K, M, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_4F_E2M1, K, N, K));
        const int initial_ld = std::max(M, N);
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&out_layout, CUDA_R_16F, M, N,
                                                initial_ld));
        cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
        int64_t ld = N;
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
            out_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
            out_layout, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &CUBLAS_WORKSPACE_LIMIT, sizeof(CUBLAS_WORKSPACE_LIMIT)));
    }

    ~LtPlan() {
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(out_layout);
        cublasLtMatrixLayoutDestroy(b_layout);
        cublasLtMatrixLayoutDestroy(a_layout);
        cublasLtDestroy(handle);
    }

    LtSlot make_slot(const host::Fixture& f) {
        const host::InputBuffers& inputs = f.inputs;
        LtSlot slot;
        CUDA_CHECK(cudaMalloc(&slot.a, inputs.A.size()));
        CUDA_CHECK(cudaMalloc(&slot.b, inputs.B.size()));
        CUDA_CHECK(cudaMalloc(&slot.sfa, inputs.SFA.size()));
        CUDA_CHECK(cudaMalloc(&slot.sfb, inputs.SFB.size()));
        CUDA_CHECK(cudaMalloc(&slot.d, size_t(M) * N * sizeof(__half)));
        slot.c = slot.d;   // beta = 0 permits C == D
        CUDA_CHECK(cudaMemcpy(slot.a, inputs.A.data(), inputs.A.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(slot.b, inputs.B.data(), inputs.B.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(slot.sfa, inputs.SFA.data(), inputs.SFA.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(slot.sfb, inputs.SFB.data(), inputs.SFB.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(slot.d, 0, size_t(M) * N * sizeof(__half)));

        CUBLAS_CHECK(cublasLtMatmulDescCreate(&slot.op, CUBLAS_COMPUTE_32F,
                                              CUDA_R_32F));
        cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
        int32_t sf_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        int8_t fast_accum = 0;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            slot.op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            slot.op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            slot.op, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum,
            sizeof(fast_accum)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            slot.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sf_mode, sizeof(sf_mode)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            slot.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sf_mode, sizeof(sf_mode)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            slot.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &slot.sfa,
            sizeof(slot.sfa)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            slot.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &slot.sfb,
            sizeof(slot.sfb)));
        return slot;
    }

    void initialize_algorithm(const LtSlot& slot) {
        cublasLtMatmulHeuristicResult_t first{};
        int returned = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
            handle, slot.op, a_layout, b_layout, out_layout, out_layout,
            preference, 1, &first, &returned));
        if (returned != 1 || first.state != CUBLAS_STATUS_SUCCESS ||
            first.workspaceSize > CUBLAS_WORKSPACE_LIMIT) {
            std::fprintf(stderr,
                         "cuBLASLt returned no usable first algorithm\n");
            std::abort();
        }
        algo = first.algo;
        workspace_bytes = first.workspaceSize;
        std::printf("CUBLAS_ALGO index=0 workspace=%zu\n", workspace_bytes);
    }

    void add_workspace(LtSlot& slot) const {
        if (workspace_bytes)
            CUDA_CHECK(cudaMalloc(&slot.workspace, workspace_bytes));
        if ((reinterpret_cast<uintptr_t>(slot.sfa) & 15u) != 0 ||
            (reinterpret_cast<uintptr_t>(slot.sfb) & 15u) != 0 ||
            (slot.workspace &&
             (reinterpret_cast<uintptr_t>(slot.workspace) & 255u) != 0)) {
            std::fprintf(stderr, "cuBLASLt slot alignment contract failed\n");
            std::abort();
        }
    }

    cublasStatus_t launch_status(const LtSlot& slot) const {
        float alpha = 1.0f, beta = 0.0f;
        return cublasLtMatmul(
            handle, slot.op, &alpha, slot.a, a_layout, slot.b, b_layout,
            &beta, slot.c, out_layout, slot.d, out_layout, &algo,
            workspace_bytes ? slot.workspace : nullptr,
            workspace_bytes, nullptr);
    }

    void launch(const LtSlot& slot) const { CUBLAS_CHECK(launch_status(slot)); }

    void free_slot(LtSlot& slot) const {
        if (slot.op) cublasLtMatmulDescDestroy(slot.op);
        if (slot.workspace) cudaFree(slot.workspace);
        if (slot.sfb) cudaFree(slot.sfb);
        if (slot.sfa) cudaFree(slot.sfa);
        if (slot.d) cudaFree(slot.d);
        if (slot.b) cudaFree(slot.b);
        if (slot.a) cudaFree(slot.a);
        slot = {};
    }
};

}  // namespace bench

// ============================================================================
// Harness: correctness gates before ANY timing, then a rotating cache-cold
// benchmark against cuBLASLt's first correctness-gated algorithm (blog:
// "How to benchmark a cache trick without lying to yourself").
// ============================================================================
namespace bench {

enum class BenchMode {
    CURRENT,
    CLOCK_PINNED,
    SUSTAINED,
    TRITON,
};

inline const char* bench_mode_name(BenchMode mode) {
    switch (mode) {
        case BenchMode::CURRENT:      return "current";
        case BenchMode::CLOCK_PINNED: return "clock-pinned";
        case BenchMode::SUSTAINED:    return "sustained";
        case BenchMode::TRITON:       return "triton";
    }
    return "unknown";
}

inline bool parse_bench_mode(const std::string& text, BenchMode* mode) {
    if (text == "current")      *mode = BenchMode::CURRENT;
    else if (text == "clock-pinned") *mode = BenchMode::CLOCK_PINNED;
    else if (text == "sustained")    *mode = BenchMode::SUSTAINED;
    else if (text == "triton")       *mode = BenchMode::TRITON;
    else return false;
    return true;
}

struct Ctx {
    int sms = 0;
    int clusters = 0;
    int l2_bytes = 0;
    dim3 grid{};
    int cooldown_sec = 0;
    int reference_clock_mhz = -1;
    BenchMode mode = BenchMode::CURRENT;
};
static Ctx g;

constexpr int ACTIVE_IDLE_CLOCK_MAX_MHZ = 2100;
constexpr int CLOCK_MATCH_TOL_MHZ = 60;
constexpr int CLOCK_COOLDOWN_MAX_SEC = 20;
constexpr int PINNED_CLOCK_MHZ = 1305;
constexpr int PINNED_CLOCK_TOL_MHZ = 15;
constexpr size_t TRITON_FLUSH_BYTES = size_t(256) << 20;
constexpr double SUSTAINED_BLOCK_SECONDS = 60.0;
constexpr double SUSTAINED_FIRST_SECONDS = 10.0;
constexpr double SUSTAINED_TAIL_FROM_SECONDS = 20.0;

inline const char* gemm_source_file() {
#ifdef NVFP4_GEMM_HEADER
    return NVFP4_GEMM_HEADER;
#else
    return "nvfp4_gemm.cu";
#endif
}

inline void print_result_box(const std::string& heading,
                             const std::string& result) {
    const size_t width = std::max(heading.size(), result.size());
    const std::string border(width + 2, '-');
    std::printf("\n+%s+\n", border.c_str());
    std::printf("| %-*s |\n", int(width), heading.c_str());
    std::printf("| %-*s |\n", int(width), result.c_str());
    std::printf("+%s+\n", border.c_str());
}

#if NVFP4_HAS_SCHEDULE
// Table slot 0 carries the gate shape's identity map; slot 1 carries the user
// shape's ownership schedule (or identity when probing fails closed).
constexpr int GATE_TABLE = 0;
constexpr int MAIN_TABLE = 1;
#endif

inline void launch_ours(const host::DeviceSlot& s
#if NVFP4_HAS_SCHEDULE
                        , int table_mode
#endif
                        ) {
    nvfp4::nvfp4_gemm_kernel
            <<<g.grid, nvfp4::TB_SIZE, sizeof(nvfp4::SmemCD)>>>(
        s.A_t, s.B_t, s.SFA_t, s.SFB_t,
        s.dC, s.M, s.N, s.K
#if NVFP4_HAS_SCHEDULE
        , table_mode * nvfp4::L2A_ROUTE_WORK_CAP
#endif
    );
    CUDA_CHECK(cudaGetLastError());
}

#if NVFP4_HAS_SCHEDULE
// A nonzero placement counter voids whatever was just measured.
inline void require_clean_placement(const char* what) {
    unsigned v = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&v, nvfp4::l2a_placement_errors, sizeof(v)));
    if (v != 0) {
        std::fprintf(stderr,
                     "placement audit failed (%s): %u mismatches, result void\n",
                     what, v);
        std::abort();
    }
}
#endif

inline void setup_device() {
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    g.sms = props.multiProcessorCount;
    g.clusters = g.sms / 2;
    g.l2_bytes = props.l2CacheSize;
    g.grid = nvfp4::launch_grid(g.clusters);
    auto* fn = nvfp4::nvfp4_gemm_kernel;
    CUDA_CHECK(cudaFuncSetAttribute(
        fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(sizeof(nvfp4::SmemCD))));
#if NVFP4_HAS_SCHEDULE
    CUDA_CHECK(cudaFuncSetAttribute(
        nvfp4::l2a_cluster_probe, cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(sizeof(nvfp4::SmemCD))));
    // The whole schedule rests on all clusters being co-resident, one per
    // SM pair — verify instead of assuming.
    cudaLaunchConfig_t occ{};
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = 2;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    occ.blockDim = dim3(nvfp4::TB_SIZE, 1, 1);
    occ.dynamicSmemBytes = sizeof(nvfp4::SmemCD);
    occ.attrs = &attr;
    occ.numAttrs = 1;
    int active = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveClusters(&active, fn, &occ));
    if (active != g.clusters) {
        std::fprintf(stderr, "active-cluster gate failed: %d of %d resident\n",
                     active, g.clusters);
        std::abort();
    }
#endif
    std::printf("DEVICE %s sm_%d%d sms=%d clusters=%d l2_bytes=%d smem/CTA=%zu\n",
                props.name, props.major, props.minor, g.sms, g.clusters,
                props.l2CacheSize, sizeof(nvfp4::SmemCD));
}

#if NVFP4_HAS_SCHEDULE
inline void upload_identity_table(int total_work, int mode) {
    std::vector<int> t(total_work);
    std::iota(t.begin(), t.end(), 0);
    if (!sched::verify_identity(t)) std::abort();
    sched::upload_table(t, mode);
}

static std::vector<int> g_schedule_duel_table;
static sched::ScheduleMode g_schedule_duel_mode = sched::ScheduleMode::AUTO;

inline bool activate_schedule_duel() {
    if (g_schedule_duel_table.empty()) {
        std::fprintf(stderr, "schedule duel table was not built\n");
        return false;
    }
    sched::upload_table(g_schedule_duel_table, GATE_TABLE);
    std::printf("SCHEDULE_DUEL activated=%s table_slot=%d\n",
                sched::schedule_name(g_schedule_duel_mode), GATE_TABLE);
    return true;
}

// Census the SM sides, verify the address-hash model, classify the
// persistent grid's clusters, build + prove + upload the user shape's
// schedule. Fails CLOSED: any probe/model/shape gate miss drops the whole
// schedule to natural raster (identity table) and says so.
inline void setup_schedule(int M, int N
                           , size_t operand_read_bytes,
                           sched::ScheduleMode requested_mode,
                           bool prepare_duel,
                           sched::ScheduleMode requested_duel_mode
                           ) {
    const int mblocks = (M + nvfp4::CLUSTER_M - 1) / nvfp4::CLUSTER_M;
    const int nblocks = (N + nvfp4::BLOCK_N - 1) / nvfp4::BLOCK_N;
    const int total_work = mblocks * nblocks;
    const sched::ScheduleMode mode =
            requested_mode == sched::ScheduleMode::AUTO
                    ? sched::pick_schedule(operand_read_bytes, mblocks, nblocks)
                    : requested_mode;
    const sched::ScheduleMode duel_mode =
            requested_duel_mode == sched::ScheduleMode::AUTO
                    ? sched::pick_schedule(operand_read_bytes, mblocks, nblocks)
                    : requested_duel_mode;
    g_schedule_duel_table.clear();
    g_schedule_duel_mode = duel_mode;
    if (prepare_duel && sched::requires_side_census(mode) !=
                                sched::requires_side_census(duel_mode)) {
        std::fprintf(stderr,
                     "schedule duel requires both modes to use the census "
                     "or both to skip it\n");
        std::abort();
    }

    // The pure controls use the same route-table lookup as the final kernel,
    // but leave the topology census and ownership partition completely out.
    if (!sched::requires_side_census(mode)) {
        if (total_work > nvfp4::L2A_ROUTE_WORK_CAP) {
            std::fprintf(stderr,
                         "schedule table too small: %d tiles exceed %d entries\n",
                         total_work, nvfp4::L2A_ROUTE_WORK_CAP);
            std::abort();
        }
        sched::upload_fallback_sides();
        const std::vector<int> table =
                sched::build_unowned_schedule(mode, mblocks, nblocks);
        sched::upload_table(table, MAIN_TABLE);
        if (prepare_duel)
            g_schedule_duel_table =
                    sched::build_unowned_schedule(duel_mode, mblocks, nblocks);
        std::printf("SCHEDULE order=%s requested=%s ownership=off "
                    "grid=%dx%d tiles=%d operand_read=%.1fMiB\n",
                    sched::schedule_name(mode),
                    sched::schedule_name(requested_mode), mblocks, nblocks,
                    total_work, double(operand_read_bytes) / double(MIB));
        return;
    }
    try {
        constexpr uint64_t PROBE_BYTES = uint64_t(2) << 20;
        void* raw = nullptr;
        CUDA_CHECK(cudaMalloc(&raw, 2 * PROBE_BYTES));
        char* arena = reinterpret_cast<char*>(
            (reinterpret_cast<uintptr_t>(raw) + PROBE_BYTES - 1) &
            ~(PROBE_BYTES - 1));
        const l2side::RuntimeMap map =
                l2side::probe_stable(arena, PROBE_BYTES, /*repeats=*/3);
        CUDA_CHECK(cudaFree(raw));
        if (map.nsm != g.sms || map.hash != l2side::kExpectedHash ||
            map.model_mismatches != 0)
            throw std::runtime_error("runtime map inconsistent with device");
        std::printf("CENSUS sm_split=%d/%d near=%.0fns far=%.0fns hash=0x%X "
                    "gap=%.1fns min_margin=%.1fns jitter=%.1fns conf=%.1fns "
                    "repeats=%d\n",
                    map.side_counts[0], map.side_counts[1], map.near_ns,
                    map.far_ns, map.hash, map.classification_gap_ns,
                    map.minimum_sample_margin_ns,
                    map.maximum_effective_jitter_ns,
                    map.minimum_confidence_ns, map.stability_repeats);

        // Cluster -> SM census on the exact persistent grid we will time.
        unsigned* d_smids = nullptr;
        const size_t n = size_t(g.clusters) * 2;
        CUDA_CHECK(cudaMalloc(&d_smids, n * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(d_smids, 0xff, n * sizeof(unsigned)));
        nvfp4::l2a_cluster_probe<<<g.grid, nvfp4::TB_SIZE,
                                   sizeof(nvfp4::SmemCD)>>>(d_smids);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<unsigned> smids(n);
        CUDA_CHECK(cudaMemcpy(smids.data(), d_smids, n * sizeof(unsigned),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_smids));
        const sched::Census census = sched::classify_census(smids, map, g.sms);
        std::printf("CENSUS clusters=%d cluster_split=%d/%d\n", census.clusters,
                    census.side_clusters[0], census.side_clusters[1]);
        sched::upload_side_constants(map, census);

        if (total_work > nvfp4::L2A_ROUTE_WORK_CAP) {
            std::printf("SCHEDULE fallback: %d tiles exceed the %d-entry table "
                        "— natural raster (audit stays armed)\n",
                        total_work, nvfp4::L2A_ROUTE_WORK_CAP);
            upload_identity_table(total_work, MAIN_TABLE);
            return;
        }
        std::vector<int> table =
                sched::build_schedule(census, mblocks, nblocks, mode);
        if (table.empty()) {
            std::printf("SCHEDULE fallback: %dx%d tile grid too small for a "
                        "side partition — natural raster (audit stays armed)\n",
                        mblocks, nblocks);
            upload_identity_table(total_work, MAIN_TABLE);
            return;
        }
        if (prepare_duel) {
            g_schedule_duel_table =
                    sched::build_schedule(census, mblocks, nblocks, duel_mode);
            if (g_schedule_duel_table.empty()) {
                std::fprintf(stderr, "second duel schedule is not buildable\n");
                std::abort();
            }
        }
        sched::upload_table(table, MAIN_TABLE);
        std::printf("SCHEDULE order=%s requested=%s requester_assignment=%s "
                    "grid=%dx%d tiles=%d operand_read=%.1fMiB\n",
                    sched::schedule_name(mode),
                    sched::schedule_name(requested_mode),
                    sched::requester_assignment(mode), mblocks, nblocks,
                    total_work,
                    double(operand_read_bytes) / double(MIB));
        if (prepare_duel)
            std::printf("SCHEDULE_DUEL prepared=%s requester_assignment=%s\n",
                        sched::schedule_name(duel_mode),
                        sched::requester_assignment(duel_mode));
    } catch (const std::exception& e) {
        std::printf("SCHEDULE fallback: %s — the side-hash model or census "
                    "gates failed closed; running natural raster with an "
                    "identity table and an inert placement audit\n", e.what());
        sched::upload_fallback_sides();
        upload_identity_table(total_work, MAIN_TABLE);
    }
}
#endif  // NVFP4_HAS_SCHEDULE

// ---- validation ----------------------------------------------------------------

inline float half_bits_to_float(uint16_t bits) {
    __half_raw r;
    r.x = bits;
    return __half2float(__half(r));
}

inline std::vector<uint16_t> copy_output(const void* device, size_t elements) {
    std::vector<uint16_t> result(elements);
    CUDA_CHECK(cudaMemcpy(result.data(), device, elements * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));
    return result;
}

inline bool validate_against(const char* label, const std::vector<uint16_t>& got,
                             const std::vector<float>& want) {
    if (got.size() != want.size()) return false;
    int bad = 0, nonfinite = 0;
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double actual = half_bits_to_float(got[i]);
        const double expected = want[i];
        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            ++bad;
            ++nonfinite;
            continue;
        }
        const double abs_error = std::fabs(actual - expected);
        max_abs = std::max(max_abs, abs_error);
        max_rel = std::max(max_rel, abs_error / std::max(1.0, std::fabs(expected)));
        if (abs_error > FP16_ABS_TOL + FP16_REL_TOL * std::fabs(expected)) ++bad;
    }
    std::printf("CORRECTNESS %s bad=%d/%zu nonfinite=%d max_abs=%.5f "
                "max_rel=%.6f %s\n",
                label, bad, got.size(), nonfinite, max_abs, max_rel,
                bad ? "FAIL" : "PASS");
    return bad == 0;
}

inline bool validate_pair(const char* label, const std::vector<uint16_t>& lhs,
                          const std::vector<uint16_t>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    int bad = 0, nonfinite = 0;
    double max_abs = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const double a = half_bits_to_float(lhs[i]);
        const double b = half_bits_to_float(rhs[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ++bad;
            ++nonfinite;
            continue;
        }
        const double abs_error = std::fabs(a - b);
        max_abs = std::max(max_abs, abs_error);
        if (abs_error > FP16_ABS_TOL + FP16_REL_TOL * std::fabs(b)) ++bad;
    }
    std::printf("PAIR_GATE %s bad=%d/%zu nonfinite=%d max_abs=%.5f %s\n",
                label, bad, lhs.size(), nonfinite, max_abs,
                bad ? "FAIL" : "PASS");
    return bad == 0;
}

inline double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

inline int read_sm_clock_mhz() {
    int device = 0;
    char pci_bus_id[32]{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device));

    const std::string bus_id(pci_bus_id);
    const bool valid_bus_id = !bus_id.empty() &&
            std::all_of(bus_id.begin(), bus_id.end(), [](char c) {
                return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                       (c >= 'A' && c <= 'F') || c == ':' || c == '.';
            });
    if (!valid_bus_id) {
        std::fprintf(stderr, "invalid CUDA PCI bus id for clock gate: %s\n",
                     bus_id.c_str());
        std::abort();
    }

    const std::string command = "nvidia-smi --id=" + bus_id +
            " --query-gpu=clocks.sm --format=csv,noheader,nounits";
    FILE* pipe = popen(command.c_str(), "r");
    int clock_mhz = -1;
    const int scanned = pipe ? std::fscanf(pipe, "%d", &clock_mhz) : 0;
    const int status = pipe ? pclose(pipe) : -1;
    if (scanned != 1 || status != 0) {
        std::fprintf(stderr,
                     "SM clock query failed for %s: scanned=%d status=%d\n",
                     bus_id.c_str(), scanned, status);
        std::abort();
    }
    return clock_mhz;
}

inline void require_pinned_clock(int clock_mhz) {
    if (g.mode != BenchMode::CLOCK_PINNED) return;
    if (std::abs(clock_mhz - PINNED_CLOCK_MHZ) > PINNED_CLOCK_TOL_MHZ) {
        std::fprintf(stderr,
                     "clock-pinned mode requires %d MHz (+/-%d), got %d MHz; "
                     "use gb300/nvfp4/run_bench_modes.sh or lock the clock first\n",
                     PINNED_CLOCK_MHZ, PINNED_CLOCK_TOL_MHZ, clock_mhz);
        std::abort();
    }
}

inline void establish_clock_reference() {
    if (g.cooldown_sec <= 0 || g.reference_clock_mhz >= 0) return;
    CUDA_CHECK(cudaDeviceSynchronize());

    int elapsed_sec = 0;
    int previous_mhz = read_sm_clock_mhz();
    int clock_mhz = previous_mhz;
    while (elapsed_sec < CLOCK_COOLDOWN_MAX_SEC) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        ++elapsed_sec;
        clock_mhz = read_sm_clock_mhz();
        if (previous_mhz <= ACTIVE_IDLE_CLOCK_MAX_MHZ &&
            clock_mhz <= ACTIVE_IDLE_CLOCK_MAX_MHZ &&
            std::abs(clock_mhz - previous_mhz) <= CLOCK_MATCH_TOL_MHZ) {
            g.reference_clock_mhz = clock_mhz;
            require_pinned_clock(clock_mhz);
            std::printf("CLOCK_REFERENCE waited=%ds clock_mhz=%d tolerance_mhz=%d PASS\n",
                        elapsed_sec, clock_mhz, CLOCK_MATCH_TOL_MHZ);
            if (g.mode == BenchMode::CLOCK_PINNED) {
                std::printf("CLOCK_PIN target_mhz=%d clock_mhz=%d tolerance_mhz=%d PASS\n",
                            PINNED_CLOCK_MHZ, clock_mhz,
                            PINNED_CLOCK_TOL_MHZ);
            }
            return;
        }
        previous_mhz = clock_mhz;
    }

    std::fprintf(stderr,
                 "clock reference failed: last_clock_mhz=%d after %ds\n",
                 clock_mhz, elapsed_sec);
    std::abort();
}

inline void cooldown() {
    if (g.cooldown_sec <= 0) return;
    if (g.reference_clock_mhz < 0) {
        std::fprintf(stderr, "clock reference was not established before timing\n");
        std::abort();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    int elapsed_sec = g.cooldown_sec;
    std::this_thread::sleep_for(std::chrono::seconds(elapsed_sec));
    int clock_mhz = read_sm_clock_mhz();
    while (clock_mhz > ACTIVE_IDLE_CLOCK_MAX_MHZ ||
           std::abs(clock_mhz - g.reference_clock_mhz) > CLOCK_MATCH_TOL_MHZ) {
        if (elapsed_sec >= CLOCK_COOLDOWN_MAX_SEC) {
            std::fprintf(stderr,
                         "cooldown clock gate failed: clock_mhz=%d "
                         "reference_mhz=%d tolerance_mhz=%d after %ds\n",
                         clock_mhz, g.reference_clock_mhz,
                         CLOCK_MATCH_TOL_MHZ, elapsed_sec);
            std::abort();
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        ++elapsed_sec;
        clock_mhz = read_sm_clock_mhz();
    }
    require_pinned_clock(clock_mhz);
    std::printf("COOLDOWN seconds=%d clock_mhz=%d reference_mhz=%d PASS\n",
                elapsed_sec, clock_mhz, g.reference_clock_mhz);
}

// ---- the small correctness gates ---------------------------------------------------
// Two small shapes, each checked against an O(MNK) host reference over the
// dequantized operands, both with partial M and N tiles, random N(0,1) data
// on principle (several classes of scale-layout bugs pass with constant
// inputs). cuBLASLt's FP4 path only has kernels for K % 32 == 0, so K=1136
// is ours-only while K=1152 can also cross-check the vendor result.
// The first shape exercises a 16-mod-32 ragged tail; the second exercises a
// 32-aligned partial 768-element group. Rounded-mainloop rungs rely on TMA
// zero fill, while exact-tail rungs stop issuing dead batches.

// Shared gate: fixture, poisoned output, host reference, and determinism. It
// hands the fixture, reference, and output back so the aligned case can also
// cross-validate cuBLASLt.
inline bool small_gate_ours(int M, int N, int K, uint64_t seed,
                            const char* tag, host::Fixture* f_out,
                            std::vector<float>* ref_out,
                            std::vector<uint16_t>* out) {
    *f_out = host::make_fixture(M, N, K, seed, /*deq=*/true);
    const host::Fixture& f = *f_out;
#if NVFP4_HAS_SCHEDULE
    {
        const int mblocks = (M + nvfp4::CLUSTER_M - 1) / nvfp4::CLUSTER_M;
        const int nblocks = (N + nvfp4::BLOCK_N - 1) / nvfp4::BLOCK_N;
        upload_identity_table(mblocks * nblocks, GATE_TABLE);
    }
#endif
    host::DeviceSlot* ours = host::make_device_slot(f.inputs, M, N);
    const size_t elems = size_t(M) * N;
    // Output poisoned before every gated run: a kernel that silently skipped
    // work would be caught by the leftover NaNs.
    CUDA_CHECK(cudaMemset(ours->dC, 0xFF, elems * sizeof(uint16_t)));
    launch_ours(*ours
#if NVFP4_HAS_SCHEDULE
                , GATE_TABLE
#endif
                );
    CUDA_CHECK(cudaDeviceSynchronize());
#if NVFP4_HAS_SCHEDULE
    require_clean_placement(tag);
#endif
    *out = copy_output(ours->dC, elems);

    ref_out->assign(elems, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            const float a = f.A_deq[size_t(m) * K + k];
            for (int n = 0; n < N; ++n)
                (*ref_out)[size_t(m) * N + n] += a * f.B_deq[size_t(n) * K + k];
        }
    const std::string label = std::string("OURS-") + tag + "-vs-host-ref";
    bool ok = validate_against(label.c_str(), *out, *ref_out);

    CUDA_CHECK(cudaMemset(ours->dC, 0xFF, elems * sizeof(uint16_t)));
    launch_ours(*ours
#if NVFP4_HAS_SCHEDULE
                , GATE_TABLE
#endif
                );
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto rerun = copy_output(ours->dC, elems);
    const bool det = (*out == rerun);
    std::printf("DETERMINISM OURS-%s %s\n", tag, det ? "PASS" : "FAIL");
    host::free_device_slot(ours);
    return ok && det;
}

inline bool small_correctness_gates() {
    const int M = 768, N = 512;
    host::Fixture f;
    std::vector<float> ref;
    std::vector<uint16_t> ours_out;

    // The ragged logical K, checked against the host reference.
    bool ok = small_gate_ours(M, N, /*K=*/1136, 0xC011AB1EULL,
                              "small-ragged", &f, &ref, &ours_out);

#if NVFP4_HAS_K64_TAIL
    // (a2) the K64-dispatch classes, against the host reference: K=1280
    // (r=512: 4xK96 + 2xK64 — both plain SF words, two live A/B windows) and
    // K=1024 (r=256: 2xK96 + 1xK64 — word-0 SF, ONE live window, the
    // t_win==1 cursor path; the K=16384 tail class). K=1120 (r=352, odd a)
    // must CLASSIFY as inactive and run the original tail — a regression
    // gate on the dispatch itself. The K=1136/1152 gates above/below never
    // dispatch (r=368 is 16 mod 32; r=384 is 96-aligned).
    ok = small_gate_ours(M, N, /*K=*/1280, 0xC011AB20ULL,
                         "small-k64-pair", &f, &ref, &ours_out) && ok;
    ok = small_gate_ours(M, N, /*K=*/1024, 0xC011AB22ULL,
                         "small-k64-single", &f, &ref, &ours_out) && ok;
    ok = small_gate_ours(M, N, /*K=*/1120, 0xC011AB21ULL,
                         "small-k64-fallback", &f, &ref, &ours_out) && ok;
#endif

    // (b) the 32-aligned twin, cross-validated against cuBLASLt.
    const int K = 1152;
    ok = small_gate_ours(M, N, K, 0xC011AB1FULL, "small-aligned",
                         &f, &ref, &ours_out) && ok;
    LtPlan plan(M, N, K);
    LtSlot vendor = plan.make_slot(f);
    plan.initialize_algorithm(vendor);
    plan.add_workspace(vendor);
    const size_t elems = size_t(M) * N;
    CUDA_CHECK(cudaMemset(vendor.d, 0xFF, elems * sizeof(uint16_t)));
    plan.launch(vendor);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto vout = copy_output(vendor.d, elems);
    ok = validate_against("CUBLAS-h0-small-vs-host-ref", vout, ref) && ok;
    ok = validate_pair("OURS-vs-CUBLAS-h0-small", ours_out, vout) && ok;

    plan.free_slot(vendor);
    return ok;
}

// ---- user-shape gates -------------------------------------------------------------
// Our kernel's deterministic user-shape output is the anchor. The first
// cuBLASLt recommendation must launch, agree with the anchor within FP16
// tolerance, and be deterministic across two launches.

inline bool user_shape_gates(host::DeviceSlot& ours, LtPlan& plan,
                             LtSlot& vendor) {
    const size_t elems = size_t(ours.M) * ours.N;
    CUDA_CHECK(cudaMemset(ours.dC, 0xFF, elems * sizeof(uint16_t)));
    launch_ours(ours
#if NVFP4_HAS_SCHEDULE
                , MAIN_TABLE
#endif
                );
    CUDA_CHECK(cudaDeviceSynchronize());
#if NVFP4_HAS_SCHEDULE
    require_clean_placement("user-shape gate");
#endif
    auto anchor = copy_output(ours.dC, elems);
    CUDA_CHECK(cudaMemset(ours.dC, 0xFF, elems * sizeof(uint16_t)));
    launch_ours(ours
#if NVFP4_HAS_SCHEDULE
                , MAIN_TABLE
#endif
                );
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto rerun = copy_output(ours.dC, elems);
    bool all_ok = (anchor == rerun);
    std::printf("DETERMINISM OURS-user %s\n", all_ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaMemset(vendor.d, 0xFF, elems * sizeof(uint16_t)));
    const cublasStatus_t status = plan.launch_status(vendor);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::printf("CUBLAS_GATE index=0 launch_status=%d FAIL\n", int(status));
        return false;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto first = copy_output(vendor.d, elems);
    bool ok = validate_pair("OURS-vs-CUBLAS-h0", anchor, first);
    CUDA_CHECK(cudaMemset(vendor.d, 0xFF, elems * sizeof(uint16_t)));
    plan.launch(vendor);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto second = copy_output(vendor.d, elems);
    const bool deterministic = (first == second);
    ok = ok && deterministic;
    std::printf("CUBLAS_GATE index=0 workspace=%zu determinism=%s %s\n",
                plan.workspace_bytes, deterministic ? "PASS" : "FAIL",
                ok ? "PASS" : "FAIL");
    all_ok = all_ok && ok;
    return all_ok;
}
#if NVFP4_HAS_SCHEDULE

inline bool schedule_duel_gate(host::DeviceSlot& ours) {
    const size_t elems = size_t(ours.M) * ours.N;
    CUDA_CHECK(cudaMemset(ours.dC, 0xFF, elems * sizeof(uint16_t)));
    launch_ours(ours, MAIN_TABLE);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto primary = copy_output(ours.dC, elems);

    CUDA_CHECK(cudaMemset(ours.dC, 0xFF, elems * sizeof(uint16_t)));
    launch_ours(ours, GATE_TABLE);
    CUDA_CHECK(cudaDeviceSynchronize());
    require_clean_placement("schedule-duel gate");
    const auto duel = copy_output(ours.dC, elems);
    bool ok = validate_pair("SCHEDULE-DUEL-vs-primary", duel, primary);

    CUDA_CHECK(cudaMemset(ours.dC, 0xFF, elems * sizeof(uint16_t)));
    launch_ours(ours, GATE_TABLE);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto rerun = copy_output(ours.dC, elems);
    const bool deterministic = (duel == rerun);
    std::printf("DETERMINISM SCHEDULE-DUEL %s\n",
                deterministic ? "PASS" : "FAIL");
    return ok && deterministic;
}
#endif

// ---- the rotating cache-cold benchmark ----------------------------------------------
// Every measured launch reads a DIFFERENT copy of A/B/scales, and the copies
// cycle so that by the time one is reused, far more data than the L2 holds
// has passed through the cache. Sacrificial slots (never timed) flush the
// previous arm's residue before the events start.

template <class Launch>
double measure_rotating(int sacrificial, int timed, int iters, Launch launch) {
    for (int i = 0; i < sacrificial * SACRIFICIAL_PASSES; ++i)
        launch(i % sacrificial);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        launch(sacrificial + i % timed);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
    cudaEventDestroy(end);
    cudaEventDestroy(start);
    return double(elapsed_ms) * 1000.0 / iters;   // microseconds per launch
}

struct TritonTiming {
    double median_us = 0.0;
    double q1_us = 0.0;
    double q3_us = 0.0;
    int warmup_iters = 0;
    int timed_iters = 0;
};

template <class Launch, class ClearL2>
TritonTiming measure_triton_style(Launch launch, ClearL2 clear_l2,
                                  double warmup_ms = 25.0,
                                  double measure_ms = 100.0) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t estimate_start, estimate_end;
    CUDA_CHECK(cudaEventCreate(&estimate_start));
    CUDA_CHECK(cudaEventCreate(&estimate_end));
    CUDA_CHECK(cudaEventRecord(estimate_start));
    for (int i = 0; i < 5; ++i) launch();
    CUDA_CHECK(cudaEventRecord(estimate_end));
    CUDA_CHECK(cudaEventSynchronize(estimate_end));
    float estimate_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&estimate_ms, estimate_start, estimate_end));
    cudaEventDestroy(estimate_end);
    cudaEventDestroy(estimate_start);

    const double per_launch_ms = double(estimate_ms) / 5.0;
    TritonTiming timing;
    timing.warmup_iters = std::max(1, int(warmup_ms / per_launch_ms));
    timing.timed_iters = std::max(1, int(measure_ms / per_launch_ms));
    for (int i = 0; i < timing.warmup_iters; ++i) launch();

    std::vector<cudaEvent_t> starts(size_t(timing.timed_iters));
    std::vector<cudaEvent_t> ends(size_t(timing.timed_iters));
    for (int i = 0; i < timing.timed_iters; ++i) {
        CUDA_CHECK(cudaEventCreate(&starts[size_t(i)]));
        CUDA_CHECK(cudaEventCreate(&ends[size_t(i)]));
        clear_l2();
        CUDA_CHECK(cudaEventRecord(starts[size_t(i)]));
        launch();
        CUDA_CHECK(cudaEventRecord(ends[size_t(i)]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> samples;
    samples.reserve(size_t(timing.timed_iters));
    for (int i = 0; i < timing.timed_iters; ++i) {
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
                &elapsed_ms, starts[size_t(i)], ends[size_t(i)]));
        samples.push_back(double(elapsed_ms) * 1000.0);
        cudaEventDestroy(ends[size_t(i)]);
        cudaEventDestroy(starts[size_t(i)]);
    }
    std::sort(samples.begin(), samples.end());
    auto quantile = [&](double q) {
        const double position = q * double(samples.size() - 1);
        const size_t low = size_t(position);
        const size_t high = std::min(low + 1, samples.size() - 1);
        return samples[low] + (position - double(low)) *
                                   (samples[high] - samples[low]);
    };
    timing.q1_us = quantile(0.25);
    timing.median_us = quantile(0.50);
    timing.q3_us = quantile(0.75);
    return timing;
}

struct SustainedTiming {
    double elapsed_seconds = 0.0;
    double first_us = 0.0;
    double tail_us = 0.0;
    int chunks = 0;
    int first_chunks = 0;
    int tail_chunks = 0;
};

template <class Launch>
SustainedTiming measure_sustained_block(int timed_slot_offset, int timed_slots,
                                        int chunk_iters, Launch launch) {
    long long launch_index = 0;
    auto launch_chunk = [&] {
        for (int i = 0; i < chunk_iters; ++i) {
            launch(timed_slot_offset +
                   int(launch_index++ % timed_slots));
        }
    };

    cudaEvent_t boundaries[3];
    for (auto& event : boundaries) CUDA_CHECK(cudaEventCreate(&event));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(boundaries[0]));
    launch_chunk();
    CUDA_CHECK(cudaEventRecord(boundaries[1]));
    launch_chunk();
    CUDA_CHECK(cudaEventRecord(boundaries[2]));

    SustainedTiming timing;
    double first_sum_us = 0.0;
    double tail_sum_us = 0.0;
    int in_flight = 2;
    bool feeding = true;
    for (int chunk = 0; in_flight > 0; ++chunk, --in_flight) {
        CUDA_CHECK(cudaEventSynchronize(boundaries[(chunk + 1) % 3]));
        CUDA_CHECK(cudaGetLastError());
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
                &elapsed_ms, boundaries[chunk % 3],
                boundaries[(chunk + 1) % 3]));
        const double chunk_start_seconds = timing.elapsed_seconds;
        const double launch_us = double(elapsed_ms) * 1000.0 / chunk_iters;
        timing.elapsed_seconds += double(elapsed_ms) / 1000.0;
        ++timing.chunks;
        if (chunk_start_seconds < SUSTAINED_FIRST_SECONDS) {
            first_sum_us += launch_us;
            ++timing.first_chunks;
        }
        if (chunk_start_seconds >= SUSTAINED_TAIL_FROM_SECONDS) {
            tail_sum_us += launch_us;
            ++timing.tail_chunks;
        }
        if (feeding && timing.elapsed_seconds >= SUSTAINED_BLOCK_SECONDS)
            feeding = false;
        if (feeding) {
            launch_chunk();
            CUDA_CHECK(cudaEventRecord(boundaries[chunk % 3]));
            ++in_flight;
        }
    }
    for (auto& event : boundaries) cudaEventDestroy(event);
    if (timing.first_chunks == 0 || timing.tail_chunks == 0) {
        std::fprintf(stderr, "sustained block was too short for tail sampling\n");
        std::abort();
    }
    timing.first_us = first_sum_us / timing.first_chunks;
    timing.tail_us = tail_sum_us / timing.tail_chunks;
    return timing;
}

inline double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
           double(values.size());
}

}  // namespace bench

int main(int argc, char** argv) {
    if (argc < 4) {
#if NVFP4_HAS_SCHEDULE
        std::fprintf(stderr,
            "usage: %s M N K [--rounds R] [--cooldown-sec S] "
            "[--bench-mode MODE] [--check-only] [--ours-only] [--schedule MODE] "
            "[--schedule-duel MODE_A MODE_B]\n",
            argv[0]);
        std::fprintf(stderr,
            "benchmark modes: current, clock-pinned, sustained, triton\n");
        std::fprintf(stderr,
            "schedule modes: auto, raster, pocket-8x4, pocket-8x8, hilbert, "
            "blind-plain, blind-pocket-8x4, blind-pocket-8x8, "
            "hilbert-in-blind, "
            "owned-plain, owned-pocket-8x4, owned-pocket-8x8, "
            "hilbert-in-owned\n");
#else
        std::fprintf(stderr,
            "usage: %s M N K [--rounds R] [--cooldown-sec S] "
            "[--bench-mode MODE] [--check-only] [--ours-only]\n",
            argv[0]);
        std::fprintf(stderr,
            "benchmark modes: current, clock-pinned, sustained, triton\n");
#endif
        return 2;
    }
    const int M = std::atoi(argv[1]);
    const int N = std::atoi(argv[2]);
    const int K = std::atoi(argv[3]);
    int rounds = -1;
    bool check_only = false;
    // --ours-only: the quick-signal lane — skip timed cuBLASLt vendor legs;
    // absolute paired old-vs-new reads come from two
    // alternated invocations of this mode (gates still run first).
    bool ours_only = false;
#if NVFP4_HAS_SCHEDULE
    sched::ScheduleMode schedule_mode = sched::ScheduleMode::AUTO;
    sched::ScheduleMode duel_mode = sched::ScheduleMode::AUTO;
    bool schedule_explicit = false;
    bool schedule_duel = false;
#endif
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (a == "--cooldown-sec" && i + 1 < argc)
            bench::g.cooldown_sec = std::atoi(argv[++i]);
        else if (a == "--bench-mode" && i + 1 < argc) {
            if (!bench::parse_bench_mode(argv[++i], &bench::g.mode)) {
                std::fprintf(stderr,
                             "--bench-mode needs current, clock-pinned, "
                             "sustained, or triton\n");
                return 2;
            }
        }
        else if (a == "--check-only") check_only = true;
        else if (a == "--ours-only") ours_only = true;
#if NVFP4_HAS_SCHEDULE
        else if (a == "--schedule") {
            if (schedule_explicit || schedule_duel || i + 1 >= argc ||
                !sched::parse_schedule_mode(argv[++i], &schedule_mode)) {
                std::fprintf(stderr,
                    "--schedule may appear once and needs one of: auto, "
                    "raster, pocket-8x4, "
                    "pocket-8x8, hilbert, blind-plain, blind-pocket-8x4, "
                    "blind-pocket-8x8, hilbert-in-blind, owned-plain, "
                    "owned-pocket-8x4, "
                    "owned-pocket-8x8, hilbert-in-owned\n");
                return 2;
            }
            schedule_explicit = true;
        }
        else if (a == "--schedule-duel") {
            if (schedule_explicit || schedule_duel || i + 2 >= argc ||
                !sched::parse_schedule_mode(argv[i + 1], &schedule_mode) ||
                !sched::parse_schedule_mode(argv[i + 2], &duel_mode) ||
                schedule_mode == sched::ScheduleMode::AUTO ||
                duel_mode == sched::ScheduleMode::AUTO) {
                std::fprintf(stderr,
                    "--schedule-duel needs two explicit schedule modes from "
                    "the same ownership class\n");
                return 2;
            }
            i += 2;
            schedule_duel = true;
        }
#endif
        else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (rounds < 0) {
        rounds = bench::g.mode == bench::BenchMode::SUSTAINED ? 3
               : bench::g.mode == bench::BenchMode::TRITON    ? 1
                                                               : 5;
    }
    if (M < 1 || N < 1 || K < 16 || K % 16 != 0 || rounds < 1) {
        std::fprintf(stderr,
                     "shape gate: need M,N >= 1 and K >= 16 with K %% 16 == 0 "
                     "(one UE4M3 scale per 16 K elements)\n");
        return 2;
    }
    if (bench::g.mode != bench::BenchMode::CURRENT && ours_only) {
        std::fprintf(stderr,
                     "--ours-only is supported only by --bench-mode current\n");
        return 2;
    }
#if NVFP4_HAS_SCHEDULE
    if (bench::g.mode != bench::BenchMode::CURRENT && schedule_duel) {
        std::fprintf(stderr,
                     "--schedule-duel is supported only by --bench-mode current\n");
        return 2;
    }
#endif
    if (bench::g.mode == bench::BenchMode::CLOCK_PINNED &&
        bench::g.cooldown_sec <= 0) {
        std::fprintf(stderr,
                     "clock-pinned mode requires a positive --cooldown-sec\n");
        return 2;
    }
    if (bench::g.mode == bench::BenchMode::TRITON && rounds != 1) {
        std::fprintf(stderr, "triton mode uses one do_bench-style sample; "
                             "set --rounds 1 or omit it\n");
        return 2;
    }
    if (bench::g.cooldown_sec < 0 ||
        bench::g.cooldown_sec > bench::CLOCK_COOLDOWN_MAX_SEC) {
        std::fprintf(stderr, "--cooldown-sec must be between 0 and %d\n",
                     bench::CLOCK_COOLDOWN_MAX_SEC);
        return 2;
    }
    // The baseline's constraint, not ours: cuBLASLt's FP4 path has kernels
    // only for K % 32 == 0 (its algorithm query returns NOT_SUPPORTED
    // otherwise). This kernel takes any K % 16 == 0 — the built-in small
    // gate covers the ragged 16-mod-32 case against a host reference — but
    // this harness exists to race cuBLASLt, so refuse rather than skip.
    // (M and N stay unconstrained: the FP4 packing constraint rides the
    // K-major leading dimension, and a shape cuBLASLt still refuses fails
    // loudly at the gated algorithm query, never silently.)
    if (K % 32 != 0) {
        std::fprintf(stderr,
                     "cuBLASLt comparison refused: its FP4 path requires "
                     "K %% 32 == 0 and K=%d gives no supporting kernel. "
                     "The kernel itself accepts this K; pick a 32-aligned K "
                     "to race the baseline.\n", K);
        return 2;
    }
    std::printf("NVFP4 GEMM build=%s shape=%dx%dx%d bench_mode=%s "
                "rounds=%d cooldown=%ds\n",
                NVFP4_BUILD_NAME, M, N, K,
                bench::bench_mode_name(bench::g.mode), rounds,
                bench::g.cooldown_sec);

    bench::setup_device();

    // The user-shape fixture. Deterministic seed, mixed with the shape so
    // different shapes get different data.
    const uint64_t seed = 0xCAFEF00DULL ^ uint64_t(M) ^ (uint64_t(K) << 17);
    host::Fixture f = host::make_fixture(M, N, K, seed, /*deq=*/false);
    // Both launch paths copy these same host buffers. At public K%32 shapes,
    // the TMA stride adds no A/B padding, so they also equal the logical
    // packed-E2M1 inputs byte for byte.
    const bool identical_inputs =
            f.inputs.A == f.A_logical && f.inputs.B == f.B_logical;
    std::printf("INPUT_GATE ours-vs-cublas source buffers=%s\n",
                identical_inputs ? "IDENTICAL" : "DIFFERENT");
    if (!identical_inputs) return 1;
    const size_t ours_read = f.inputs.A.size() + f.inputs.B.size() +
                             f.inputs.SFA.size() + f.inputs.SFB.size();
    const size_t vendor_read = ours_read;
#if NVFP4_HAS_SCHEDULE
    bench::setup_schedule(M, N
                          , ours_read, schedule_mode, schedule_duel, duel_mode
                          );
#endif

    // Gate 1: the small clipped problems against an O(MNK) host reference.
    if (!bench::small_correctness_gates()) {
        std::fprintf(stderr, "small correctness gates FAILED\n");
        return 1;
    }
#if NVFP4_HAS_SCHEDULE
    if (schedule_duel && !bench::activate_schedule_duel()) return 1;
#endif

    // Gate 2: use the first cuBLASLt recommendation for every shape.
    bench::LtPlan plan(M, N, K);
    host::DeviceSlot* ours0 = host::make_device_slot(f.inputs, M, N);
    bench::LtSlot vendor0 = plan.make_slot(f);
    plan.initialize_algorithm(vendor0);
    plan.add_workspace(vendor0);
    if (!bench::user_shape_gates(*ours0, plan, vendor0)) {
        std::fprintf(stderr, "user-shape correctness gates FAILED\n");
        return 1;
    }
#if NVFP4_HAS_SCHEDULE
    if (schedule_duel && !bench::schedule_duel_gate(*ours0)) {
        std::fprintf(stderr, "schedule-duel correctness gate FAILED\n");
        return 1;
    }
#endif
    if (check_only) {
        std::printf("CHECK-ONLY all gates PASS shape=%dx%dx%d\n", M, N, K);
        return 0;
    }

    // ---- benchmark sizing: enough independent copies that nothing a kernel
    // is timed on can still be resident from its previous use.
    const size_t required_reuse = std::max(
            size_t(256) * bench::MIB, size_t(2) * size_t(bench::g.l2_bytes));
    const size_t min_read = std::min(ours_read, vendor_read);
    const int timed_slots = std::max(
            3, int((required_reuse + min_read - 1) / min_read) + 1);
    const int sacrificial_slots = timed_slots - 1;
    const int total_slots = timed_slots + sacrificial_slots;
    const int iters = timed_slots * bench::TIMED_PASSES;
    if (total_slots > 40) {
        std::fprintf(stderr,
                     "shape too small for the rotating cache-cold protocol "
                     "(%d slots needed; operand read is only %zu bytes)\n",
                     total_slots, min_read);
        return 1;
    }
    std::printf("CACHE_COLD required_reuse=%zu ours_read=%zu cublas_read=%zu "
                "sacrificial=%d timed=%d iters=%d\n",
                required_reuse, ours_read, vendor_read, sacrificial_slots,
                timed_slots, iters);

    std::vector<host::DeviceSlot*> ours_slots{ours0};
    std::vector<bench::LtSlot> vendor_slots{vendor0};
    for (int i = 1; i < total_slots; ++i) {
        ours_slots.push_back(host::make_device_slot(f.inputs, M, N));
        vendor_slots.push_back(plan.make_slot(f));
        plan.add_workspace(vendor_slots.back());
    }

    auto launch_ours_slot = [&](int s) {
        bench::launch_ours(*ours_slots[size_t(s)]
#if NVFP4_HAS_SCHEDULE
                           , bench::MAIN_TABLE
#endif
                           );
    };
#if NVFP4_HAS_SCHEDULE
    auto launch_duel_slot = [&](int s) {
        bench::launch_ours(*ours_slots[size_t(s)], bench::GATE_TABLE);
    };
#endif
    auto launch_vendor_slot = [&](int s) {
        plan.launch(vendor_slots[size_t(s)]);
    };
    auto free_benchmark_slots = [&] {
        for (auto& slot : vendor_slots) plan.free_slot(slot);
        for (auto* slot : ours_slots) host::free_device_slot(slot);
    };
    bench::establish_clock_reference();
#if NVFP4_HAS_SCHEDULE

    if (schedule_duel) {
        const double flop = 2.0 * double(M) * double(N) * double(K);
        auto pflops = [&](double us) { return flop / (us * 1e9); };
        std::vector<double> primary_us_all, duel_us_all, duel_over_primary;
        int duel_wins = 0;
        for (int r = 0; r < rounds; ++r) {
            double primary_us = 0.0, duel_us = 0.0;
            const bool primary_first = (r % 2 == 0);
            for (int arm = 0; arm < 2; ++arm) {
                const bool run_primary = (arm == 0) == primary_first;
                bench::cooldown();
                const double us = run_primary
                        ? bench::measure_rotating(sacrificial_slots, timed_slots,
                                                  iters, launch_ours_slot)
                        : bench::measure_rotating(sacrificial_slots, timed_slots,
                                                  iters, launch_duel_slot);
                bench::require_clean_placement("schedule-duel timed round");
                if (run_primary) primary_us = us;
                else duel_us = us;
            }
            const double ratio = primary_us / duel_us;
            primary_us_all.push_back(primary_us);
            duel_us_all.push_back(duel_us);
            duel_over_primary.push_back(ratio);
            duel_wins += duel_us < primary_us;
            std::printf("SCHEDULE_DUEL_ROUND %d/%d order=%s "
                        "a=%s %.3fus (%.3f PF) b=%s %.3fus (%.3f PF) "
                        "b_over_a=%.6f %s\n",
                        r + 1, rounds, primary_first ? "a-first" : "b-first",
                        sched::schedule_name(schedule_mode), primary_us,
                        pflops(primary_us), sched::schedule_name(duel_mode),
                        duel_us, pflops(duel_us), ratio,
                        duel_us < primary_us ? "B_WIN" : "A_WIN");
        }
        const double primary_med = bench::median(primary_us_all);
        const double duel_med = bench::median(duel_us_all);
        const double pair_med = bench::median(duel_over_primary);
        double log_ratio_sum = 0.0;
        for (const double ratio : duel_over_primary)
            log_ratio_sum += std::log(ratio);
        const double pair_geomean =
                std::exp(log_ratio_sum / duel_over_primary.size());
        std::printf("SCHEDULE_DUEL_RESULT a=%s %.3fus %.3fPF b=%s %.3fus "
                    "%.3fPF paired_median=%.6f paired_geomean=%.6f "
                    "b_wins=%d/%d\n",
                    sched::schedule_name(schedule_mode), primary_med,
                    pflops(primary_med), sched::schedule_name(duel_mode),
                    duel_med, pflops(duel_med), pair_med, pair_geomean,
                    duel_wins, rounds);
        free_benchmark_slots();
        return 0;
    }
#endif

    const double flop = 2.0 * double(M) * double(N) * double(K);
    auto pflops = [&](double us) { return flop / (us * 1e9); };

    if (bench::g.mode == bench::BenchMode::TRITON) {
        void* l2_clear = nullptr;
        CUDA_CHECK(cudaMalloc(&l2_clear, bench::TRITON_FLUSH_BYTES));
        auto clear_l2 = [&] {
            CUDA_CHECK(cudaMemset(l2_clear, 0, bench::TRITON_FLUSH_BYTES));
        };
        const auto ours = bench::measure_triton_style(
                [&] { launch_ours_slot(0); }, clear_l2);
#if NVFP4_HAS_SCHEDULE
        bench::require_clean_placement("triton-style timing");
#endif
        const auto vendor = bench::measure_triton_style(
                [&] { launch_vendor_slot(0); }, clear_l2);
        CUDA_CHECK(cudaFree(l2_clear));

        std::printf("TRITON_RESULT flush_bytes=%zu warmup_ms=25 measure_ms=100 "
                    "ours_us=%.3f ours_iqr=%.3f..%.3f ours_n=%d "
                    "cublas_us=%.3f cublas_iqr=%.3f..%.3f cublas_n=%d "
                    "ratio=%.5f\n",
                    bench::TRITON_FLUSH_BYTES, ours.median_us,
                    ours.q1_us, ours.q3_us, ours.timed_iters,
                    vendor.median_us, vendor.q1_us, vendor.q3_us,
                    vendor.timed_iters, vendor.median_us / ours.median_us);
        char heading[256];
        char result[256];
        std::snprintf(heading, sizeof(heading),
                      "%s (%s) | shape %dx%dx%d | mode triton",
                      bench::gemm_source_file(), NVFP4_BUILD_NAME, M, N, K);
        std::snprintf(result, sizeof(result),
                      "RESULT | ours %.3f PFLOP/s | cuBLASLt %.3f PFLOP/s | "
                      "%.2f%% of cuBLASLt",
                      pflops(ours.median_us), pflops(vendor.median_us),
                      100.0 * vendor.median_us / ours.median_us);
        bench::print_result_box(heading, result);
        free_benchmark_slots();
        return 0;
    }

    if (bench::g.mode == bench::BenchMode::SUSTAINED) {
        const double ours_calibration = bench::measure_rotating(
                sacrificial_slots, timed_slots, iters, launch_ours_slot);
#if NVFP4_HAS_SCHEDULE
        bench::require_clean_placement("sustained calibration");
#endif
        const double vendor_calibration = bench::measure_rotating(
                sacrificial_slots, timed_slots, iters, launch_vendor_slot);
        std::printf("SUSTAINED_CALIBRATION ours_us=%.3f cublas_us=%.3f "
                    "block_seconds=%.0f tail_from_seconds=%.0f\n",
                    ours_calibration, vendor_calibration,
                    bench::SUSTAINED_BLOCK_SECONDS,
                    bench::SUSTAINED_TAIL_FROM_SECONDS);

        std::vector<double> ours_tail_us;
        std::vector<double> vendor_tail_us;
        int wins = 0;
        for (int round = 0; round < rounds; ++round) {
            double ours_tail = 0.0;
            double vendor_tail = 0.0;
            const bool ours_first = (round % 2) == 0;
            for (int arm = 0; arm < 2; ++arm) {
                const bool run_ours = (arm == 0) == ours_first;
                const double estimate = run_ours ? ours_calibration
                                                 : vendor_calibration;
                const int chunk_iters =
                        std::max(1, int(500000.0 / estimate));
                const auto timing = run_ours
                        ? bench::measure_sustained_block(
                                  sacrificial_slots, timed_slots, chunk_iters,
                                  launch_ours_slot)
                        : bench::measure_sustained_block(
                                  sacrificial_slots, timed_slots, chunk_iters,
                                  launch_vendor_slot);
#if NVFP4_HAS_SCHEDULE
                if (run_ours)
                    bench::require_clean_placement("sustained block");
#endif
                std::printf("SUSTAINED_BLOCK round=%d/%d arm=%s "
                            "elapsed_s=%.2f chunks=%d chunk_iters=%d "
                            "first_us=%.3f tail_us=%.3f tail_pf=%.3f\n",
                            round + 1, rounds,
                            run_ours ? "ours" : "cublasLt",
                            timing.elapsed_seconds, timing.chunks, chunk_iters,
                            timing.first_us, timing.tail_us,
                            pflops(timing.tail_us));
                if (run_ours) ours_tail = timing.tail_us;
                else vendor_tail = timing.tail_us;
            }
            ours_tail_us.push_back(ours_tail);
            vendor_tail_us.push_back(vendor_tail);
            const bool win = ours_tail < vendor_tail;
            wins += win;
            std::printf("SUSTAINED_ROUND %d/%d order=%s ours_tail_us=%.3f "
                        "cublas_tail_us=%.3f ratio=%.5f %s\n",
                        round + 1, rounds,
                        ours_first ? "ours-first" : "cublas-first",
                        ours_tail, vendor_tail, vendor_tail / ours_tail,
                        win ? "WIN" : "LOSS");
        }
        const double ours_us = bench::mean(ours_tail_us);
        const double vendor_us = bench::mean(vendor_tail_us);
        std::printf("SUSTAINED_RESULT blocks_per_arm=%d block_seconds=%.0f "
                    "tail_from_seconds=%.0f ours_tail_us=%.3f "
                    "cublas_tail_us=%.3f ratio=%.5f wins=%d/%d\n",
                    rounds, bench::SUSTAINED_BLOCK_SECONDS,
                    bench::SUSTAINED_TAIL_FROM_SECONDS, ours_us, vendor_us,
                    vendor_us / ours_us, wins, rounds);
        char heading[256];
        char result[256];
        std::snprintf(heading, sizeof(heading),
                      "%s (%s) | shape %dx%dx%d | mode sustained",
                      bench::gemm_source_file(), NVFP4_BUILD_NAME, M, N, K);
        std::snprintf(result, sizeof(result),
                      "RESULT | ours %.3f PFLOP/s | cuBLASLt %.3f PFLOP/s | "
                      "%.2f%% of cuBLASLt | wins %d/%d",
                      pflops(ours_us), pflops(vendor_us),
                      100.0 * vendor_us / ours_us, wins, rounds);
        bench::print_result_box(heading, result);
        free_benchmark_slots();
        return 0;
    }

    // The current and clock-pinned modes use R rounds with alternating arm
    // order. FLOPs use logical K, so padded work receives no extra credit.
    std::vector<double> ours_us_all, vendor_us_all;
    int wins = 0;
    if (ours_only) {
        for (int r = 0; r < rounds; ++r) {
            bench::cooldown();
            const double us = bench::measure_rotating(
                    sacrificial_slots, timed_slots, iters, launch_ours_slot);
#if NVFP4_HAS_SCHEDULE
            bench::require_clean_placement("timed round");
#endif
            ours_us_all.push_back(us);
            std::printf("OURS_ONLY_ROUND %d/%d ours=%.3fus (%.3f PF)\n",
                        r + 1, rounds, us, pflops(us));
        }
        const double med = bench::median(ours_us_all);
        char heading[256];
        char result[256];
        std::snprintf(heading, sizeof(heading),
                      "%s (%s) | shape %dx%dx%d | rounds %d",
                      bench::gemm_source_file(), NVFP4_BUILD_NAME,
                      M, N, K, rounds);
        std::snprintf(result, sizeof(result),
                      "RESULT | ours %.3f PFLOP/s | median %.3f us | ours-only",
                      pflops(med), med);
        bench::print_result_box(heading, result);
        return 0;
    }
    for (int r = 0; r < rounds; ++r) {
        double ours_us = 0.0, vendor_us = 0.0;
        const bool ours_first = (r % 2 == 0);
        for (int arm = 0; arm < 2; ++arm) {
            const bool run_ours = (arm == 0) == ours_first;
            bench::cooldown();
            if (run_ours) {
                ours_us = bench::measure_rotating(
                        sacrificial_slots, timed_slots, iters, launch_ours_slot);
#if NVFP4_HAS_SCHEDULE
                bench::require_clean_placement("timed round");
#endif
            } else {
                vendor_us = bench::measure_rotating(
                        sacrificial_slots, timed_slots, iters,
                        launch_vendor_slot);
            }
        }
        ours_us_all.push_back(ours_us);
        vendor_us_all.push_back(vendor_us);
        const bool win = ours_us < vendor_us;
        wins += win;
        std::printf("ROUND %d/%d order=%s ours=%.3fus (%.3f PF) "
                    "cublasLt=%.3fus (%.3f PF) ratio=%.4f %s\n",
                    r + 1, rounds, ours_first ? "ours-first" : "cublas-first",
                    ours_us, pflops(ours_us), vendor_us, pflops(vendor_us),
                    vendor_us / ours_us, win ? "WIN" : "LOSS");
    }

    // Comparator-symmetry receipt: both arms' round dispersion, side by side.
    // The comparator is a measured object — its power-governor state can mix
    // regimes across rounds and windows, and a mixed window can fake or erase
    // a small win. An arm's CoV beyond 3x the other's (above the ~0.3% CoV a
    // quiet window reads) or beyond 3x that quiet class outright marks the
    // window regime-suspect: explain it before trusting the ratio.
    auto cov_pct = [](const std::vector<double>& v) {
        if (v.size() < 2) return 0.0;
        double mean = 0.0;
        for (double x : v) mean += x;
        mean /= double(v.size());
        if (mean == 0.0) return 0.0;
        double ss = 0.0;
        for (double x : v) ss += (x - mean) * (x - mean);
        return std::sqrt(ss / double(v.size() - 1)) / mean * 100.0;
    };
    const double ours_cov = cov_pct(ours_us_all);
    const double vendor_cov = cov_pct(vendor_us_all);
    const double asym = std::max(ours_cov, vendor_cov) /
                        std::max(std::min(ours_cov, vendor_cov), 1e-9);
    constexpr double kAsymFlag = 3.0;      // flag threshold, both clauses
    constexpr double kQuietCovPct = 0.3;   // a quiet window's round-CoV class
    auto unstable = [&](double cov, double sibling) {
        return cov > kAsymFlag * kQuietCovPct ||
               (cov > kAsymFlag * sibling && cov > kQuietCovPct);
    };
    std::printf("ARMS %dx%dx%d ours_med=%.3fus cov=%.3f%% | comp_med=%.3fus "
                "cov=%.3f%% | asym=%.1fx %s\n",
                M, N, K, bench::median(ours_us_all), ours_cov,
                bench::median(vendor_us_all), vendor_cov, asym,
                unstable(vendor_cov, ours_cov)   ? "REGIME=comparator-unstable"
                : unstable(ours_cov, vendor_cov) ? "REGIME=ours-unstable"
                                                 : "REGIME=quiet");

    const double ours_med = bench::median(ours_us_all);
    const double vendor_med = bench::median(vendor_us_all);
    char heading[256];
    char result[256];
    std::snprintf(heading, sizeof(heading),
                  "%s (%s) | shape %dx%dx%d | rounds %d",
                  bench::gemm_source_file(), NVFP4_BUILD_NAME,
                  M, N, K, rounds);
    std::snprintf(result, sizeof(result),
                  "RESULT | ours %.3f PFLOP/s | cuBLASLt %.3f PFLOP/s | "
                  "%.2f%% of cuBLASLt | wins %d/%d",
                  pflops(ours_med), pflops(vendor_med),
                  100.0 * vendor_med / ours_med, wins, rounds);
    bench::print_result_box(heading, result);

    free_benchmark_slots();
    return 0;
}
