/**
 * CUTLASS Blackwell GEMM with CollectiveBuilder
 * ==============================================
 * 
 * Uses CUTLASS's high-level CollectiveBuilder API for optimal Blackwell GEMM.
 * This automatically configures:
 * - SM100 TMA operations (SM100_TMA_2SM_LOAD/MULTICAST)
 * - True warp specialization
 * - PipelineTmaUmmaAsync
 * - Cluster launch with TMA multicast
 * 
 * Reference: examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Element types
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAccumulator = float;

// For A @ B^T:
// - A is (M, K) in RowMajor: element (i,j) at offset i*K + j
// - B is (N, K) in RowMajor: we want B^T which is (K, N)
//   Treating B as ColumnMajor gives us (K, N) access pattern
// - C is (M, N) in RowMajor
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;  // Transpose B: (N,K) stored -> (K,N) accessed
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

// Architecture
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

// MMA and Cluster shapes - optimized for Blackwell
// 256x128 tile for tcgen05 MMA
using MmaTileShape_MNK = Shape<_256, _128, _64>;
// 2x2 cluster for TMA multicast
using ClusterShape_MNK = Shape<_2, _2, _1>;

// Build epilogue using CollectiveBuilder
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Build mainloop using CollectiveBuilder
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// Complete GEMM kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,  // ProblemShape: M, N, K, L (batch)
    CollectiveMainloop,
    CollectiveEpilogue,
    void  // Use default ClusterLaunchControl scheduler
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

torch::Tensor cutlass_gemm(torch::Tensor a, torch::Tensor b, float alpha = 1.0f, float beta = 0.0f) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(a.size(1) == b.size(1), "Inner dimensions must match for A @ B^T");
    TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16, "Must be FP16");
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Must be CUDA tensors");

    auto a_contig = a.contiguous();
    auto b_contig = b.contiguous();

    int64_t M = a_contig.size(0);
    int64_t K = a_contig.size(1);
    int64_t N = b_contig.size(0);

    // Output tensor
    auto c = torch::zeros({M, N}, a.options().dtype(torch::kFloat32));
    auto d = torch::empty({M, N}, a.options().dtype(torch::kFloat32));

    // CUTLASS strides (for RowMajor: stride = inner dimension)
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    int m = static_cast<int>(M);
    int n = static_cast<int>(N);
    int k = static_cast<int>(K);
    
    // For RowMajor A (M,K): stride is K
    // B is stored as an (N,K) tensor. StrideB already encodes ColumnMajor access.
    // For RowMajor C (M,N): stride is N
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

    // GEMM arguments
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},  // problem shape
        {reinterpret_cast<ElementA const*>(a_contig.data_ptr<at::Half>()),
         stride_A,
         reinterpret_cast<ElementB const*>(b_contig.data_ptr<at::Half>()),
         stride_B},  // mainloop
        {{alpha, beta},
         reinterpret_cast<ElementC const*>(c.data_ptr<float>()),
         stride_C,
         reinterpret_cast<ElementC*>(d.data_ptr<float>()),
         stride_D}  // epilogue
    };

    Gemm gemm_op;
    auto status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS cannot implement this GEMM: ", cutlass::cutlassGetStatusString(status));

    // Workspace
    size_t workspace_size = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                   torch::TensorOptions().dtype(torch::kByte).device(a.device()));

    status = gemm_op.initialize(args, workspace.data_ptr<uint8_t>());
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS initialization failed");

    // Run GEMM
    status = gemm_op(at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS execution failed");

    return d.to(torch::kFloat16);
}

#else

torch::Tensor cutlass_gemm(torch::Tensor a, torch::Tensor b, float alpha = 1.0f, float beta = 0.0f) {
    TORCH_CHECK(false, "CUTLASS SM100 not supported - requires Blackwell GPU");
    return torch::Tensor();
}

#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_gemm", &cutlass_gemm, "CUTLASS Blackwell FP16 GEMM (C = A @ B^T)",
          py::arg("a"), py::arg("b"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);
}
