// baseline_cutlass_gemm_fp8.cu -- CUTLASS FP8 GEMM baseline.
//
// NOTE: CUTLASS 2.x DefaultGemmConfiguration does not cover FP8 types for Sm100.
// Use CUTLASS 3.x collective builders (GemmUniversalAdapter) so the kernel is fully specified.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "cute/tensor.hpp"

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

using namespace cute;

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " "         \
                << cudaGetErrorString(status) << std::endl;                     \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CUTLASS_CHECK(status)                                                    \
  do {                                                                           \
    cutlass::Status error = (status);                                            \
    if (error != cutlass::Status::kSuccess) {                                    \
      std::cerr << "CUTLASS error " << __FILE__ << ":" << __LINE__ << " "      \
                << cutlassGetStatusString(error) << std::endl;                  \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

struct Options {
  int m;
  int n;
  int k;
  int iterations;
  int repeats;
  float alpha;
  float beta;
};

struct GpuTimer {
  cudaEvent_t start_event{};
  cudaEvent_t stop_event{};
  cudaStream_t stream = 0;

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

  void stop() { CUDA_CHECK(cudaEventRecord(stop_event, stream)); }

  float elapsed_millis() {
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_event, stop_event));
    return elapsed;
  }
};

template <typename Element, typename Layout>
static void initialize_tensor(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
  // Match CUTLASS examples: keep small magnitude for <= 8-bit inputs.
  double scope_max = 2.0;
  double scope_min = -2.0;
  const int bits = cutlass::sizeof_bits<Element>::value;
  if (bits > 8) {
    scope_max = 5.0;
    scope_min = -5.0;
  }
  cutlass::reference::host::TensorFillRandomUniform(view, seed, scope_max, scope_min, 0);
}

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// A matrix configuration
using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16

// B matrix configuration
using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16

// C/D matrix configuration
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

// CUTLASS 4.5.2 supports a K=128 MMA tile for dense FP8 x FP8 on SM100.
// Keep the baseline on the smaller 1SM 128x128 tile.
using TileShape = Shape<_128, _128, _128>;  // (M, N, K)
using ClusterShape = Shape<_1, _1, _1>;

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

using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
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

static StrideA stride_A;
static StrideB stride_B;
static StrideC stride_C;
static StrideD stride_D;

static cutlass::HostTensor<ElementA, LayoutA> tensor_A;
static cutlass::HostTensor<ElementB, LayoutB> tensor_B;
static cutlass::HostTensor<ElementC, LayoutC> tensor_C;
static cutlass::HostTensor<ElementD, LayoutD> tensor_D;

static void initialize(const Options& options) {
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, 1));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, 1));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, 1));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, 1));

  auto a_coord = cutlass::make_Coord(options.m, options.k);
  auto b_coord = cutlass::make_Coord(options.k, options.n);
  auto c_coord = cutlass::make_Coord(options.m, options.n);

  tensor_A.resize(a_coord);
  tensor_B.resize(b_coord);
  tensor_C.resize(c_coord);
  tensor_D.resize(c_coord);

  initialize_tensor(tensor_A.host_view(), 42);
  initialize_tensor(tensor_B.host_view(), 43);
  cutlass::reference::host::TensorFill(tensor_C.host_view(), ElementC(0));

  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();
}

static typename Gemm::Arguments args_from_options(const Options& options) {
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, 1},
      {tensor_A.device_data(), stride_A, tensor_B.device_data(), stride_B},
      {{options.alpha, options.beta}, tensor_C.device_data(), stride_C, tensor_D.device_data(), stride_D}};
  // Keep the scheduler simple and deterministic.
  arguments.scheduler.max_swizzle_size = 1;
  return arguments;
}

static int run_cutlass(const Options& options) {
  initialize(options);

  Gemm gemm;
  auto arguments = args_from_options(options);

  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Warmup
  CUTLASS_CHECK(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer timer;
  timer.start();
  for (int iter = 0; iter < options.iterations; ++iter) {
    NVTX_RANGE("compute_math:cutlass_fp8");
    for (int rep = 0; rep < options.repeats; ++rep) {
      CUTLASS_CHECK(gemm.run());
    }
  }
  timer.stop();

  const float total_ms = timer.elapsed_millis();
  const float avg_ms = total_ms / static_cast<float>(options.iterations * options.repeats);

  const double flops = 2.0 * static_cast<double>(options.m) * options.n * options.k * options.repeats * options.iterations;
  const double tflops = flops / (total_ms * 1e9);

  std::cout << "CUTLASS FP8 GEMM (baseline): " << avg_ms << " ms" << std::endl;
  std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;

#ifdef VERIFY
  tensor_D.sync_host();
  const size_t elements = static_cast<size_t>(options.m) * options.n;
  double checksum = 0.0;
  const ElementD* h_out = tensor_D.host_data();
  for (size_t i = 0; i < elements; ++i) {
    checksum += std::abs(static_cast<double>(static_cast<float>(h_out[i])));
  }
  VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

  return 0;
}

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

int main() {
  NVTX_RANGE("main");

  Options options{};
  options.m = 4096;
  options.n = 4096;
  options.k = 4096;
  options.iterations = 10;
  options.repeats = 16;
  options.alpha = 1.0f;
  options.beta = 0.0f;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return run_cutlass(options);
#else
  std::cerr << "SKIPPED: CUTLASS FP8 kernel requires CUDA 13.0+ and SM100+." << std::endl;
  return 1;
#endif
}
