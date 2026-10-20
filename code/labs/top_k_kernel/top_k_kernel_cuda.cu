/**
 * CUTLASS GEMM helper for the grouped Top-K selection lab.
 *
 * Exposes one GEMM primitive:
 *   A[M, K] x B_rows[N, K]^T -> C[M, N]
 *
 * The Python benchmark uses this to score reduced grouped query rows against
 * block-compressed K tiles. The grouped backward GEMMs run through torch.matmul.
 */

#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/arch/arch.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t status = (call);                                          \
        if (status != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(status));             \
        }                                                                     \
    } while (0)

namespace {

using ElementInput = cutlass::half_t;
using LayoutInput = cutlass::layout::RowMajor;
using ElementOutput = float;
using LayoutOutput = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ElementCompute = float;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute
>;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInput,
    LayoutInput,
    ElementInput,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;

void validate_inputs(const torch::Tensor& a, const torch::Tensor& b_rows) {
    TORCH_CHECK(a.is_cuda(), "Input A must be on CUDA device");
    TORCH_CHECK(b_rows.is_cuda(), "Input B must be on CUDA device");
    TORCH_CHECK(a.dtype() == torch::kFloat16, "Input A must be float16");
    TORCH_CHECK(b_rows.dtype() == torch::kFloat16, "Input B must be float16");
    TORCH_CHECK(a.dim() == 2, "Input A must be 2D");
    TORCH_CHECK(b_rows.dim() == 2, "Input B must be 2D");
    TORCH_CHECK(
        a.size(1) == b_rows.size(1),
        "Inner dimensions must match for GEMM (A: ",
        a.sizes(),
        ", B: ",
        b_rows.sizes(),
        ")"
    );
}

}  // namespace

torch::Tensor matmul_cutlass_topk(const torch::Tensor& a, const torch::Tensor& b_rows) {
    validate_inputs(a, b_rows);

    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = b_rows.size(0);

    auto out = torch::empty({m, n}, a.options().dtype(torch::kFloat32));

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    ElementInput const* ptr_a = reinterpret_cast<ElementInput const*>(a.data_ptr<at::Half>());
    ElementInput const* ptr_b = reinterpret_cast<ElementInput const*>(b_rows.data_ptr<at::Half>());
    ElementOutput* ptr_out = out.data_ptr<float>();

    int lda = static_cast<int>(a.stride(0));
    int ldb = static_cast<int>(b_rows.stride(0));
    int ldc = static_cast<int>(out.stride(0));

    ElementCompute alpha = 1.0f;
    ElementCompute beta = 0.0f;

    typename Gemm::Arguments args(
        problem_size,
        {ptr_a, lda},
        {ptr_b, ldb},
        {ptr_out, ldc},
        {ptr_out, ldc},
        {alpha, beta}
    );

    Gemm gemm_op;
    auto support = gemm_op.can_implement(args);
    TORCH_CHECK(
        support == cutlass::Status::kSuccess,
        "CUTLASS arguments unsupported: ",
        cutlassGetStatusString(support)
    );
    auto status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        cudaError_t cuda_status = cudaGetLastError();
        std::string error_msg = cutlassGetStatusString(status);
        if (cuda_status != cudaSuccess) {
            error_msg += std::string(" | CUDA: ") + cudaGetErrorString(cuda_status);
        }
        TORCH_CHECK(false, "CUTLASS GEMM failed: ", error_msg);
    }
    CUDA_CHECK(cudaGetLastError());

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cutlass_topk", &matmul_cutlass_topk);
}
