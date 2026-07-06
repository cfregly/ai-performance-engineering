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

#include "../../core/common/headers/cuda_verify.cuh"
#include "../../core/common/nvtx_utils.cuh"

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
// We keep 2 tile families and dispatch per leaderboard shape:
// - N64 lane: Shape<_128,_64,_256> (current fast default lane)
// - N128 lane: Shape<_128,_128,_256> (deeper tile-family exploration lane)
using MmaTileShapeN64 = Shape<_128, _64, _256>;
using MmaTileShapeN128 = Shape<_128, _128, _256>;
using ClusterShapeC1 = Shape<_1, _1, _1>;
using ClusterShapeC4 = Shape<_1, _4, _1>;
using MainloopScheduleNvf4 = cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;
using MainloopScheduleBlockScaled = cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledSm100;

using CollectiveEpilogueN64C1 = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShapeN64, ClusterShapeC1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveEpilogueN64C4 = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShapeN64, ClusterShapeC4,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

// Shape-specific dispatch (request from tuning pass):
// - shape (128,7168,16384): keep StageCountAutoCarveout
// - other leaderboard shapes: use fixed StageCount<7>
using CollectiveMainloopAutoN64C1 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShapeN64, ClusterShapeC1,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogueN64C1::SharedStorage))>,
    MainloopScheduleNvf4
  >::CollectiveOp;

using CollectiveMainloopStage7N64C1 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShapeN64, ClusterShapeC1,
    cutlass::gemm::collective::StageCount<7>,
    MainloopScheduleNvf4
  >::CollectiveOp;

using CollectiveMainloopStage5N64C1 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShapeN64, ClusterShapeC1,
    cutlass::gemm::collective::StageCount<5>,
    MainloopScheduleNvf4
  >::CollectiveOp;

using CollectiveMainloopStage6N64C1 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShapeN64, ClusterShapeC1,
    cutlass::gemm::collective::StageCount<6>,
    MainloopScheduleNvf4
  >::CollectiveOp;

using CollectiveMainloopAutoN64BsC1 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShapeN64, ClusterShapeC1,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogueN64C1::SharedStorage))>,
    MainloopScheduleBlockScaled
  >::CollectiveOp;

using CollectiveMainloopStage5N64BsC1 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShapeN64, ClusterShapeC1,
    cutlass::gemm::collective::StageCount<5>,
    MainloopScheduleBlockScaled
  >::CollectiveOp;

using CollectiveMainloopAutoN64C4 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShapeN64, ClusterShapeC4,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogueN64C4::SharedStorage))>,
    MainloopScheduleNvf4
  >::CollectiveOp;

using CollectiveMainloopStage7N64C4 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShapeN64, ClusterShapeC4,
    cutlass::gemm::collective::StageCount<7>,
    MainloopScheduleNvf4
  >::CollectiveOp;

using CollectiveEpilogueN128C1 = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShapeN128, ClusterShapeC1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloopAutoN128C1 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShapeN128, ClusterShapeC1,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogueN128C1::SharedStorage))>,
    MainloopScheduleNvf4
  >::CollectiveOp;

using CollectiveMainloopStage5N128C1 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShapeN128, ClusterShapeC1,
    cutlass::gemm::collective::StageCount<5>,
    MainloopScheduleNvf4
  >::CollectiveOp;

using GemmKernelAutoN64C1 = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopAutoN64C1,
    CollectiveEpilogueN64C1,
    void>;

using GemmKernelStage7N64C1 = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopStage7N64C1,
    CollectiveEpilogueN64C1,
    void>;

using GemmKernelStage5N64C1 = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopStage5N64C1,
    CollectiveEpilogueN64C1,
    void>;

using GemmKernelStage6N64C1 = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopStage6N64C1,
    CollectiveEpilogueN64C1,
    void>;

using GemmKernelAutoN64BsC1 = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopAutoN64BsC1,
    CollectiveEpilogueN64C1,
    void>;

using GemmKernelStage5N64BsC1 = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopStage5N64BsC1,
    CollectiveEpilogueN64C1,
    void>;

using GemmKernelAutoN64C4 = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopAutoN64C4,
    CollectiveEpilogueN64C4,
    void>;

using GemmKernelStage7N64C4 = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopStage7N64C4,
    CollectiveEpilogueN64C4,
    void>;

using GemmKernelAutoN128C1 = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopAutoN128C1,
    CollectiveEpilogueN128C1,
    void>;

using GemmKernelStage5N128C1 = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopStage5N128C1,
    CollectiveEpilogueN128C1,
    void>;

using GemmAutoN64C1 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelAutoN64C1>;
// Experimental StreamK lane for the decode shape (M=128): fills the 148-SM GPU by
// splitting the K=16384 reduction across CTAs (the 112-CTA data-parallel grid leaves
// 36 SMs idle). Gated by AISP_NVFP4_FORCE_STREAMK at runtime.
using GemmKernelStreamKN64C1 = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopAutoN64C1,
    CollectiveEpilogueN64C1,
    cutlass::gemm::StreamKScheduler>;
using GemmStreamKN64C1 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelStreamKN64C1>;
using GemmStage5N64C1 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelStage5N64C1>;
using GemmStage6N64C1 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelStage6N64C1>;
using GemmStage7N64C1 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelStage7N64C1>;
using GemmAutoN64BsC1 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelAutoN64BsC1>;
using GemmStage5N64BsC1 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelStage5N64BsC1>;
using GemmAutoN64C4 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelAutoN64C4>;
using GemmStage7N64C4 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelStage7N64C4>;
using GemmAutoN128C1 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelAutoN128C1>;
using GemmStage5N128C1 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelStage5N128C1>;
constexpr uint64_t kSeed = 42;

enum class TileFamily {
    N64,
    N128
};

enum class ScheduleFamily {
    Nvf4,
    BlockScaled
};

enum class KernelVariant {
    AutoStageN64C1,
    Stage5N64C1,
    Stage6N64C1,
    Stage7N64C1,
    AutoStageN64BsC1,
    Stage5N64BsC1,
    AutoStageN64C4,
    Stage7N64C4,
    AutoStageN128C1,
    Stage5N128C1
};

TileFamily pick_tile_family(int m, int n, int k) {
    auto read_tile_override = [](const char* name) -> int {
        if (const char* value = std::getenv(name)) {
            const int tile_n = std::atoi(value);
            if (tile_n == 64 || tile_n == 128) {
                return tile_n;
            }
        }
        return -1;
    };

    const int forced_tile = read_tile_override("AISP_NVFP4_FORCE_TILE_N");
    if (forced_tile > 0) {
        return forced_tile == 128 ? TileFamily::N128 : TileFamily::N64;
    }

    int tile_n = -1;
    if (m == 128 && n == 7168 && k == 16384) {
        tile_n = read_tile_override("AISP_NVFP4_TILE_CASE0");
    } else if (m == 128 && n == 4096 && k == 7168) {
        tile_n = read_tile_override("AISP_NVFP4_TILE_CASE1");
    } else if (m == 128 && n == 7168 && k == 2048) {
        tile_n = read_tile_override("AISP_NVFP4_TILE_CASE2");
    }
    if (tile_n == 128) {
        return TileFamily::N128;
    }
    return TileFamily::N64;
}

ScheduleFamily pick_schedule_family(int m, int n, int k) {
    auto read_sched_override = [](const char* name) -> int {
        if (const char* value = std::getenv(name)) {
            const int family = std::atoi(value);
            if (family == 0 || family == 1) {
                return family;
            }
        }
        return -1;
    };

    const int forced_family = read_sched_override("AISP_NVFP4_FORCE_SCHED_FAMILY");
    if (forced_family >= 0) {
        return forced_family == 1 ? ScheduleFamily::BlockScaled : ScheduleFamily::Nvf4;
    }

    int family = -1;
    if (m == 128 && n == 7168 && k == 16384) {
        family = read_sched_override("AISP_NVFP4_SCHED_CASE0");
    } else if (m == 128 && n == 4096 && k == 7168) {
        family = read_sched_override("AISP_NVFP4_SCHED_CASE1");
    } else if (m == 128 && n == 7168 && k == 2048) {
        family = read_sched_override("AISP_NVFP4_SCHED_CASE2");
    }
    if (family == 1) {
        return ScheduleFamily::BlockScaled;
    }
    return ScheduleFamily::Nvf4;
}

KernelVariant pick_kernel_variant(int m, int n, int k, TileFamily tile_family, ScheduleFamily schedule_family) {
    bool force_auto = false;
    bool force_stage7 = false;
    bool force_cluster4 = false;
    bool force_cluster1 = false;

    if (const char* env_force_auto = std::getenv("AISP_NVFP4_FORCE_AUTO_STAGE")) {
        if (std::atoi(env_force_auto) != 0) {
            force_auto = true;
        }
    }
    if (const char* env_force_stage7 = std::getenv("AISP_NVFP4_FORCE_STAGE7")) {
        if (std::atoi(env_force_stage7) != 0) {
            force_stage7 = true;
        }
    }
    if (const char* cluster4 = std::getenv("AISP_NVFP4_FORCE_CLUSTER4")) {
        if (std::atoi(cluster4) != 0) {
            force_cluster4 = true;
        }
    }
    if (const char* cluster1 = std::getenv("AISP_NVFP4_FORCE_CLUSTER1")) {
        if (std::atoi(cluster1) != 0) {
            force_cluster1 = true;
        }
    }

    auto read_stage_override = [](const char* name) -> int {
        if (const char* value = std::getenv(name)) {
            const int stage = std::atoi(value);
            if (stage == 0 || stage == 5 || stage == 6 || stage == 7) {
                return stage;
            }
        }
        return -1;
    };

    int stage = -1;  // 0=auto
    if (force_auto) {
        stage = 0;
    } else if (force_stage7) {
        stage = 7;
    } else {
        const int forced_stage = read_stage_override("AISP_NVFP4_FORCE_STAGE");
        if (forced_stage >= 0) {
            stage = forced_stage;
        }
    }

    if (stage < 0) {
        // Per-shape stage overrides for quick tuning without rebuild.
        if (m == 128 && n == 7168 && k == 16384) {
            const int ov = read_stage_override("AISP_NVFP4_STAGE_CASE0");
            stage = (ov >= 0) ? ov : 0;  // default auto
        } else if (m == 128 && n == 4096 && k == 7168) {
            const int ov = read_stage_override("AISP_NVFP4_STAGE_CASE1");
            stage = (ov >= 0) ? ov : 5;
        } else if (m == 128 && n == 7168 && k == 2048) {
            const int ov = read_stage_override("AISP_NVFP4_STAGE_CASE2");
            stage = (ov >= 0) ? ov : 5;
        } else {
            stage = 7;
        }
    }

    bool use_cluster4 = false;
    if (force_cluster4) {
        use_cluster4 = true;
    } else if (force_cluster1) {
        use_cluster4 = false;
    } else {
        // Default to non-clustered lane. Clustered lane remains available via
        // AISP_NVFP4_FORCE_CLUSTER4 for targeted experiments.
        use_cluster4 = false;
    }

    if (tile_family != TileFamily::N64) {
        use_cluster4 = false;
    }

    if (tile_family == TileFamily::N128) {
        switch (stage) {
            case 0: return KernelVariant::AutoStageN128C1;
            case 5: return KernelVariant::Stage5N128C1;
            case 6: return KernelVariant::Stage5N128C1;
            case 7: return KernelVariant::Stage5N128C1;
        }
        return KernelVariant::Stage5N128C1;
    }

    if (schedule_family == ScheduleFamily::BlockScaled) {
        use_cluster4 = false;
        if (stage == 0) {
            return KernelVariant::AutoStageN64BsC1;
        }
        return KernelVariant::Stage5N64BsC1;
    }

    if (use_cluster4) {
        // Clustered lane currently provides Auto + Stage7 implementations.
        if (stage == 0) {
            return KernelVariant::AutoStageN64C4;
        }
        return KernelVariant::Stage7N64C4;
    }
    switch (stage) {
        case 0: return KernelVariant::AutoStageN64C1;
        case 5: return KernelVariant::Stage5N64C1;
        case 6: return KernelVariant::Stage6N64C1;
        case 7: return KernelVariant::Stage7N64C1;
    }
    return KernelVariant::Stage7N64C1;
}

const char* kernel_variant_name(KernelVariant variant) {
    switch (variant) {
        case KernelVariant::AutoStageN64C1: return "auto_stage_n64_c1";
        case KernelVariant::Stage5N64C1: return "stage5_n64_c1";
        case KernelVariant::Stage6N64C1: return "stage6_n64_c1";
        case KernelVariant::Stage7N64C1: return "stage7_n64_c1";
        case KernelVariant::AutoStageN64BsC1: return "auto_stage_n64_bs_c1";
        case KernelVariant::Stage5N64BsC1: return "stage5_n64_bs_c1";
        case KernelVariant::AutoStageN64C4: return "auto_stage_n64_c4";
        case KernelVariant::Stage7N64C4: return "stage7_n64_c4";
        case KernelVariant::AutoStageN128C1: return "auto_stage_n128_c1";
        case KernelVariant::Stage5N128C1: return "stage5_n128_c1";
    }
    return "unknown";
}

int swizzle_for_variant(KernelVariant variant) {
    // Global override for fast runtime tuning sweeps.
    if (const char* force_swizzle = std::getenv("AISP_NVFP4_FORCE_SWIZZLE")) {
        const int v = std::atoi(force_swizzle);
        if (v >= 0 && v <= 8) {
            return v;
        }
    }
    switch (variant) {
        case KernelVariant::AutoStageN64C1: return 1;
        case KernelVariant::Stage5N64C1: return 1;
        case KernelVariant::Stage6N64C1: return 1;
        case KernelVariant::Stage7N64C1: return 1;
        case KernelVariant::AutoStageN64BsC1: return 1;
        case KernelVariant::Stage5N64BsC1: return 1;
        case KernelVariant::AutoStageN64C4: return 1;
        case KernelVariant::Stage7N64C4: return 1;
        case KernelVariant::AutoStageN128C1: return 1;
        case KernelVariant::Stage5N128C1: return 1;
    }
    return 1;
}

int swizzle_for_shape(KernelVariant variant, int m, int n, int k) {
    // Optional per-shape overrides:
    // - case0: (128,7168,16384) via AISP_NVFP4_SWIZZLE_CASE0
    // - case1: (128,4096,7168)  via AISP_NVFP4_SWIZZLE_CASE1
    // - case2: (128,7168,2048)  via AISP_NVFP4_SWIZZLE_CASE2
    auto read_case_override = [](const char* name) -> int {
        if (const char* value = std::getenv(name)) {
            const int v = std::atoi(value);
            if (v >= 0 && v <= 8) {
                return v;
            }
        }
        return -1;
    };

    if (m == 128 && n == 7168 && k == 16384) {
        const int v = read_case_override("AISP_NVFP4_SWIZZLE_CASE0");
        if (v >= 0) {
            return v;
        }
    } else if (m == 128 && n == 4096 && k == 7168) {
        const int v = read_case_override("AISP_NVFP4_SWIZZLE_CASE1");
        if (v >= 0) {
            return v;
        }
    } else if (m == 128 && n == 7168 && k == 2048) {
        const int v = read_case_override("AISP_NVFP4_SWIZZLE_CASE2");
        if (v >= 0) {
            return v;
        }
    }
    return swizzle_for_variant(variant);
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

    cutlass::Status can_status = gemm.can_implement(arguments);
    if (can_status != cutlass::Status::kSuccess) {
        return -1.0f;
    }
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
    float t_ms = -1.0f;
    switch (variant) {
        case KernelVariant::AutoStageN64C1:
            t_ms = run_cutlass_with_gemm<GemmAutoN64C1>(options, checksum_accum);
            break;
        case KernelVariant::Stage5N64C1:
            t_ms = run_cutlass_with_gemm<GemmStage5N64C1>(options, checksum_accum);
            break;
        case KernelVariant::Stage6N64C1:
            t_ms = run_cutlass_with_gemm<GemmStage6N64C1>(options, checksum_accum);
            break;
        case KernelVariant::Stage7N64C1:
            t_ms = run_cutlass_with_gemm<GemmStage7N64C1>(options, checksum_accum);
            break;
        case KernelVariant::AutoStageN64BsC1:
            t_ms = run_cutlass_with_gemm<GemmAutoN64BsC1>(options, checksum_accum);
            break;
        case KernelVariant::Stage5N64BsC1:
            t_ms = run_cutlass_with_gemm<GemmStage5N64BsC1>(options, checksum_accum);
            break;
        case KernelVariant::AutoStageN64C4:
            t_ms = run_cutlass_with_gemm<GemmAutoN64C4>(options, checksum_accum);
            break;
        case KernelVariant::Stage7N64C4:
            t_ms = run_cutlass_with_gemm<GemmStage7N64C4>(options, checksum_accum);
            break;
        case KernelVariant::AutoStageN128C1:
            t_ms = run_cutlass_with_gemm<GemmAutoN128C1>(options, checksum_accum);
            break;
        case KernelVariant::Stage5N128C1:
            t_ms = run_cutlass_with_gemm<GemmStage5N128C1>(options, checksum_accum);
            break;
    }

    if (t_ms > 0.0f) {
        return t_ms;
    }

    // Fallback: if a clustered variant fails can_implement, retry the same stage
    // policy with the non-clustered kernel so the benchmark can continue.
    switch (variant) {
        case KernelVariant::AutoStageN64C4:
            return run_cutlass_with_gemm<GemmAutoN64C1>(options, checksum_accum);
        case KernelVariant::Stage7N64C4:
            return run_cutlass_with_gemm<GemmStage7N64C1>(options, checksum_accum);
        case KernelVariant::AutoStageN64C1:
            return run_cutlass_with_gemm<GemmAutoN64C1>(options, checksum_accum);
        case KernelVariant::Stage5N64C1:
            return run_cutlass_with_gemm<GemmStage5N64C1>(options, checksum_accum);
        case KernelVariant::Stage6N64C1:
            return run_cutlass_with_gemm<GemmStage6N64C1>(options, checksum_accum);
        case KernelVariant::Stage7N64C1:
            return run_cutlass_with_gemm<GemmStage7N64C1>(options, checksum_accum);
        case KernelVariant::AutoStageN64BsC1:
            return run_cutlass_with_gemm<GemmAutoN64BsC1>(options, checksum_accum);
        case KernelVariant::Stage5N64BsC1:
            return run_cutlass_with_gemm<GemmStage5N64BsC1>(options, checksum_accum);
        case KernelVariant::AutoStageN128C1:
            return run_cutlass_with_gemm<GemmAutoN128C1>(options, checksum_accum);
        case KernelVariant::Stage5N128C1:
            return run_cutlass_with_gemm<GemmStage5N128C1>(options, checksum_accum);
    }
    return -1.0f;
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
        const TileFamily tile_family = pick_tile_family(options.m, options.n, options.k);
        const ScheduleFamily schedule_family = pick_schedule_family(options.m, options.n, options.k);
        const KernelVariant variant = pick_kernel_variant(options.m, options.n, options.k, tile_family, schedule_family);
        options.swizzle = swizzle_for_shape(variant, options.m, options.n, options.k);
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
