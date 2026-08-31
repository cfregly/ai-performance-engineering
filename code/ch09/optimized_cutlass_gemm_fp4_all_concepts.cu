// optimized_cutlass_gemm_fp4_all_concepts.cu -- CUTLASS NVFP4 GEMM (all-concepts optimized).
//
// Combines CUTLASS' blockscaled NVFP4 mainloop with:
// - Explicit TMA warp-specialized schedule
// - CuTe CTA clustering along N to enable TMA multicast of A/SF across CTAs
// - CUDA Graph replay to reduce CPU launch overhead for microsecond-scale kernels
//
// Inspired by:
// https://obolensky.xyz/blog/nvfp4_gemm_kernel_explanation/

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "cute/tensor.hpp"

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

using namespace cute;

// GPU Mode NVFP4 GEMM leaderboard uses the geometric mean over these 3 shapes.
// Source of truth: gpu-mode/reference-kernels `problems/nvidia/nvfp4_gemm/task.yml`.
struct Nvfp4GemmShape {
    int m;
    int n;
    int k;
};
constexpr Nvfp4GemmShape kBenchShapes[] = {
    {128, 7168, 16384},
    {128, 4096, 7168},
    {128, 7168, 2048},
};
constexpr int kIterations = 50;
// Capture multiple GEMM launches per graph to reduce per-launch overhead.
// We still report per-GEMM time by dividing the total elapsed time by the
// total GEMM count (options.iterations).
constexpr int kGraphUnrollMax = 50;
constexpr int kSwizzle = 1;

struct Options {
    int m;
    int n;
    int k;
    int iterations;
    float alpha;
    float beta;
    int swizzle;
};

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
      std::cerr << "CUTLASS error " << __FILE__ << ":" << __LINE__ << " "         \
                << cutlassGetStatusString(error) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

struct GpuTimer {
    cudaStream_t stream = 0;
    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
    }

    void start(cudaStream_t stream_id = 0) {
        stream = stream_id;
        CUDA_CHECK(cudaEventRecord(start_event, stream));
    }

    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
    }

    float elapsed_millis() {
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_event, stop_event));
        return elapsed;
    }
};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// For SM100 blockscaled NVF4 (MXF4/NVF4) 1SM mainloop kernels, TileShape_M must be 128 and
// TileShape_N must be one of {64, 128, 192, 256} (see sm100_make_blockscaled_1sm_trivial_tiled_mma()).
// We keep the smallest supported N tile (64), and cluster along N to enable TMA multicast.
using MmaTileShape = Shape<_128, _64, _256>;
// N-dimension CTA clustering enables TMA multicast for A/SF across multiple CTAs.
// We use 4-way clustering here to increase multicast fanout for the leaderboard shapes
// where N is divisible by 256 (N-tile=64, cluster_n=4 => 256 columns per cluster).
using ClusterShape = Shape<_1, _4, _1>;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

// Shape-specific dispatch (request from tuning pass):
// - shape (128,7168,16384): keep StageCountAutoCarveout
// - other leaderboard shapes: use fixed StageCount<7>
using CollectiveMainloopAuto = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    MainloopSchedule
  >::CollectiveOp;

using CollectiveMainloopStage7 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCount<7>,
    MainloopSchedule
  >::CollectiveOp;

using GemmKernelAuto = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopAuto,
    CollectiveEpilogue,
    void>;

using GemmKernelStage7 = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopStage7,
    CollectiveEpilogue,
    void>;

using GemmAuto = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelAuto>;
using GemmStage7 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelStage7>;
constexpr uint64_t kSeed = 42;

enum class KernelVariant {
    AutoStage,
    Stage7
};

KernelVariant pick_kernel_variant(int m, int n, int k) {
    if (const char* force_auto = std::getenv("AISP_NVFP4_FORCE_AUTO_STAGE")) {
        if (std::atoi(force_auto) != 0) {
            return KernelVariant::AutoStage;
        }
    }
    if (const char* force_stage7 = std::getenv("AISP_NVFP4_FORCE_STAGE7")) {
        if (std::atoi(force_stage7) != 0) {
            return KernelVariant::Stage7;
        }
    }
    if (m == 128 && n == 7168 && k == 16384) {
        return KernelVariant::AutoStage;
    }
    return KernelVariant::Stage7;
}

const char* kernel_variant_name(KernelVariant variant) {
    switch (variant) {
        case KernelVariant::AutoStage: return "auto_stage";
        case KernelVariant::Stage7: return "stage7";
    }
    return "unknown";
}

int swizzle_for_variant(KernelVariant variant) {
    switch (variant) {
        case KernelVariant::AutoStage: return 1;
        case KernelVariant::Stage7: return 1;
    }
    return 1;
}

template <typename Element, typename Layout>
bool initialize_block(cutlass::TensorView<Element, Layout> view, uint64_t seed_value) {
    double scope_max = 2.0;
    double scope_min = -2.0;
    constexpr int bits_input = cutlass::sizeof_bits<Element>::value;
    if constexpr (bits_input <= 6) {
        scope_max = 2;
        scope_min = -2;
    } else if constexpr (bits_input <= 8) {
        scope_max = 1;
        scope_min = -1;
    } else {
        scope_max = 4;
        scope_min = -4;
    }
    cutlass::reference::host::TensorFillRandomUniform(view, seed_value, scope_max, scope_min, 0);
    return true;
}

template <typename Element, typename Layout>
void initialize_scale(cutlass::TensorView<Element, Layout> view, float value) {
    cutlass::reference::host::TensorFill(view, Element(value));
}

template <typename GemmT>
float run_cutlass_with_gemm(const Options& options, double* checksum_accum = nullptr) {
    using StrideA = typename GemmT::GemmKernel::StrideA;
    using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
    using LayoutSFA = typename GemmT::GemmKernel::CollectiveMainloop::LayoutSFA;
    using StrideB = typename GemmT::GemmKernel::StrideB;
    using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
    using LayoutSFB = typename GemmT::GemmKernel::CollectiveMainloop::LayoutSFB;
    using StrideC = typename GemmT::GemmKernel::StrideC;
    using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
    using StrideD = typename GemmT::GemmKernel::StrideD;
    using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));
    using Sm1xxBlkScaledConfig = typename GemmT::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

    LayoutA layout_A = make_layout(make_shape(options.m, options.k, 1), stride_A);
    LayoutB layout_B = make_layout(make_shape(options.n, options.k, 1), stride_B);
    LayoutC layout_C = make_layout(make_shape(options.m, options.n, 1), stride_C);
    LayoutD layout_D = make_layout(make_shape(options.m, options.n, 1), stride_D);
    LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(options.m, options.n, options.k, 1));
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(options.m, options.n, options.k, 1));

    cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
    cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
    cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
    cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
    cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
    cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;

    block_A.reset(cutlass::make_Coord(size(layout_A)));
    block_B.reset(cutlass::make_Coord(size(layout_B)));
    block_C.reset(cutlass::make_Coord(size(layout_C)));
    block_D.reset(cutlass::make_Coord(size(layout_D)));
    block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
    block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

    initialize_block(block_A.host_view(), kSeed + 2021);
    initialize_block(block_B.host_view(), kSeed + 2022);
    cutlass::reference::host::TensorFill(block_C.host_view(), ElementC(0));
    initialize_scale(block_SFA.host_view(), 1.0f);
    initialize_scale(block_SFB.host_view(), 1.0f);

    block_A.sync_device();
    block_B.sync_device();
    block_C.sync_device();
    block_SFA.sync_device();
    block_SFB.sync_device();

    typename GemmT::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {options.m, options.n, options.k, 1},
        {
            block_A.device_data(), stride_A,
            block_B.device_data(), stride_B,
            block_SFA.device_data(), layout_SFA,
            block_SFB.device_data(), layout_SFB
        },
        {
            {options.alpha, options.beta},
            block_C.device_data(), stride_C,
            block_D.device_data(), stride_D
        }
    };
    arguments.scheduler.max_swizzle_size = options.swizzle;

    GemmT gemm;
    size_t workspace_size = GemmT::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm.can_implement(arguments));
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm.run());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaGraph_t graph{};
    cudaGraphExec_t graph_exec{};

    // Pick a capture-unroll that divides the iteration count so we execute
    // exactly options.iterations GEMMs and can divide by that count.
    int graph_unroll = (options.iterations < kGraphUnrollMax) ? options.iterations : kGraphUnrollMax;
    while (graph_unroll > 1 && (options.iterations % graph_unroll) != 0) {
        --graph_unroll;
    }
    const int graph_launches = options.iterations / graph_unroll;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < graph_unroll; ++i) {
        CUTLASS_CHECK(gemm.run(stream));
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphUpload(graph_exec, stream));

    GpuTimer timer;
    timer.start(stream);
    {
        // Avoid per-iteration NVTX pushes/pops in microsecond-scale loops.
        NVTX_RANGE("compute_graph:cutlass_fp4_graph");
        for (int iter = 0; iter < graph_launches; ++iter) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
        }
    }
    timer.stop();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));

    const float elapsed_ms = timer.elapsed_millis();
    const float avg_ms = elapsed_ms / static_cast<float>(options.iterations);

#ifdef VERIFY
    if (checksum_accum) {
        NVTX_RANGE("verify");
        block_D.sync_host();
        const size_t elements = static_cast<size_t>(options.m) * options.n;
        double checksum = 0.0;
        const ElementD* h_out = block_D.host_data();
        for (size_t i = 0; i < elements; ++i) {
            checksum += std::abs(static_cast<float>(h_out[i]));
        }
        *checksum_accum += checksum;
    }
#endif

    return avg_ms;
}

float run_cutlass(const Options& options, KernelVariant variant, double* checksum_accum = nullptr) {
    switch (variant) {
        case KernelVariant::AutoStage:
            return run_cutlass_with_gemm<GemmAuto>(options, checksum_accum);
        case KernelVariant::Stage7:
            return run_cutlass_with_gemm<GemmStage7>(options, checksum_accum);
    }
    return run_cutlass_with_gemm<GemmAuto>(options, checksum_accum);
}

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

int main() {
    NVTX_RANGE("main");
    if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
        std::cerr << "SKIPPED: CUTLASS NVFP4 requires CUDA 12.8+." << std::endl;
        return 3;
    }

    cudaDeviceProp props{};
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    if (props.major < 10) {
        std::cerr << "SKIPPED: CUTLASS NVFP4 requires SM100+." << std::endl;
        return 3;
    }

    Options options{};
    options.iterations = kIterations;
    if (const char* profile_iters = std::getenv("AISP_NCU_PROFILE_ITERS")) {
        const int iters = std::atoi(profile_iters);
        if (iters > 0) {
            // Keep NCU kernel replay bounded. This affects only profiling runs
            // where the wrapper sets AISP_NCU_PROFILE_ITERS; timing runs keep
            // the full kIterations loop for stability.
            options.iterations = (iters < options.iterations) ? iters : options.iterations;
        }
    }
    options.alpha = 1.0f;
    options.beta = 0.0f;
    options.swizzle = kSwizzle;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    std::vector<double> times_ms;
    times_ms.reserve(sizeof(kBenchShapes) / sizeof(kBenchShapes[0]));

#ifdef VERIFY
    double checksum_accum = 0.0;
    double* checksum_ptr = &checksum_accum;
#else
    double* checksum_ptr = nullptr;
#endif

    for (const auto& s : kBenchShapes) {
        options.m = s.m;
        options.n = s.n;
        options.k = s.k;
        const KernelVariant variant = pick_kernel_variant(options.m, options.n, options.k);
        options.swizzle = swizzle_for_variant(variant);
        const float t_ms = run_cutlass(options, variant, checksum_ptr);
        times_ms.push_back(static_cast<double>(t_ms));
        std::cout << "CUTLASS NVFP4 GEMM (all-concepts optimized) shape=("
                  << s.m << "," << s.n << "," << s.k << ") variant="
                  << kernel_variant_name(variant) << " swizzle=" << options.swizzle
                  << " TIME_MS: " << t_ms << std::endl;
    }

    // Geometric mean over leaderboard shapes.
    double log_sum = 0.0;
    for (double t : times_ms) {
        if (!(t > 0.0)) {
            std::cerr << "Invalid timing (non-positive) encountered for geomean." << std::endl;
            return 2;
        }
        log_sum += std::log(t);
    }
    const double geom_ms = std::exp(log_sum / static_cast<double>(times_ms.size()));
    std::cout << "CUTLASS NVFP4 GEMM (all-concepts optimized) GEOMEAN_MS: " << geom_ms << " ms" << std::endl;
    // The harness parses the *last* TIME_MS token from stdout.
    std::cout << "TIME_MS: " << geom_ms << std::endl;

#ifdef VERIFY
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum_accum));
#endif

    return 0;
#else
    std::cerr << "SKIPPED: CUTLASS SM100 blockscaled support not compiled." << std::endl;
    return 3;
#endif
}
