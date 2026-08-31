// optimized_cutlass_gemm_fp16.cu -- CUTLASS FP16 Tensor Core GEMM optimized.
//
// On GB300 (Blackwell Ultra, sm_103) this uses the CUTLASS 3.x Sm100 collective
// builder (GemmUniversalAdapter, TMA warp-specialized 2SM), the same engine the
// FP8 sibling already uses. The previous version declared cutlass::arch::Sm80,
// which compiled the Ampere HMMA (mma.sync.m16n8k16) path on Blackwell: ncu
// showed it pinned to the Ampere FP tensor sub-pipe and underfilled (0.84
// waves/SM) at 440 TFLOPS. The Sm100 path runs on the Blackwell tensor cores.
//
// The host data (mt19937 fill, RowMajor A/B, M=N=K=2048) and the |C| checksum
// are kept identical to the baseline so the A/B and verification stay valid. For
// C = A @ B with A RowMajor (MxK) and B RowMajor (KxN), the declarations use
// LayoutA=RowMajor and LayoutB=RowMajor. The packed B stride is built from the
// logical (N,K,1) operand shape while preserving the baseline's KxN row-major
// storage.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

using namespace cute;

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " "           \
                << cudaGetErrorString(status) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CUTLASS_CHECK(status)                                                    \
  do {                                                                           \
    cutlass::Status error = (status);                                            \
    if (error != cutlass::Status::kSuccess) {                                    \
      std::cerr << "CUTLASS error " << __FILE__ << ":" << __LINE__ << " "        \
                << cutlassGetStatusString(error) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 8

using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::RowMajor;  // B stored KxN row-major (ldb=N), matching the baseline
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 8

using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

// 2SM Blackwell tile + cluster (mirrors the FP8 sibling).
using TileShape = Shape<_256, _128, _64>;
using ClusterShape = Shape<_2, _1, _1>;

using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    LayoutC,
    AlignmentC,
    ElementD,
    LayoutD,
    AlignmentD,
    EpilogueSchedule>::CollectiveOp;

using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    LayoutA,
    AlignmentA,
    ElementB,
    LayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

static int run_fp16() {
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    constexpr int kIterations = 10;
    constexpr int kRepeats = 64;

    using Element = cutlass::half_t;

    const size_t elements_A = static_cast<size_t>(M) * K;
    const size_t elements_B = static_cast<size_t>(K) * N;
    const size_t elements_C = static_cast<size_t>(M) * N;
    const size_t size_A = elements_A * sizeof(Element);
    const size_t size_B = elements_B * sizeof(Element);
    const size_t size_C = elements_C * sizeof(Element);

    Element* h_A = nullptr;
    Element* h_B = nullptr;
    Element* h_C = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_A, size_A));
    CUDA_CHECK(cudaMallocHost(&h_B, size_B));
    CUDA_CHECK(cudaMallocHost(&h_C, size_C));

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    for (size_t i = 0; i < elements_A; ++i) {
        NVTX_RANGE("setup");
        h_A[i] = Element(dis(gen));
    }
    for (size_t i = 0; i < elements_B; ++i) {
        NVTX_RANGE("setup");
        h_B[i] = Element(dis(gen));
    }
    std::fill(h_C, h_C + elements_C, Element(0));

    Element* d_A = nullptr;
    Element* d_B = nullptr;
    Element* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {d_A, stride_A, d_B, stride_B},
        {{1.0f, 0.0f}, d_C, stride_C, d_C, stride_D}};
    arguments.scheduler.max_swizzle_size = 1;

    Gemm gemm;
    const size_t workspace_size = Gemm::get_workspace_size(arguments);
    uint8_t* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }
    CUTLASS_CHECK(gemm.can_implement(arguments));
    CUTLASS_CHECK(gemm.initialize(arguments, workspace));

    // Warmup (also matches the baseline's warmup so the A/B is warm-vs-warm).
    CUTLASS_CHECK(gemm.run());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < kIterations; ++iter) {
        NVTX_RANGE("compute_math:cutlass_fp16_tensorop");
        for (int rep = 0; rep < kRepeats; ++rep) {
            CUTLASS_CHECK(gemm.run());
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(kIterations * kRepeats);

    const double flops = 2.0 * M * N * K * static_cast<double>(kRepeats) * kIterations;
    const double tflops = flops / (total_ms * 1e9);

    std::cout << "CUTLASS FP16 Tensor Core GEMM (optimized): " << avg_ms << " ms" << std::endl;
    std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    std::cout << "Checksum sample: " << static_cast<float>(h_C[0]) << std::endl;

#ifdef VERIFY
    double checksum = 0.0;
    for (size_t i = 0; i < elements_C; ++i) {
        NVTX_RANGE("verify");
        checksum += std::abs(static_cast<double>(static_cast<float>(h_C[i])));
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (workspace) {
        CUDA_CHECK(cudaFree(workspace));
    }
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    return 0;
}

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

int main() {
    NVTX_RANGE("main");
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return run_fp16();
#else
    std::cerr << "SKIPPED: CUTLASS FP16 Sm100 kernel requires CUDA 13.0+ and SM100+." << std::endl;
    return 1;
#endif
}
