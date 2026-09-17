// Repository adapter for pranjalssh/fast.cu's latest Hopper BF16 GEMM.
//
// The vendored matmul_12 kernel is included unchanged. This translation unit
// supplies strict tensor validation, setup-only TMA/schedule construction,
// cuBLAS parity, and current-PyTorch-stream launches.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using std::max;
using bf16 = __nv_bfloat16;

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

namespace {

void checked_cuda(cudaError_t status, const char* file, int line) {
  TORCH_CHECK(
      status == cudaSuccess,
      "CUDA failure in vendored fast.cu host helper at ",
      file,
      ":",
      line,
      ": ",
      cudaGetErrorString(status));
}

}  // namespace

#define cudaCheck(error) checked_cuda((error), __FILE__, __LINE__)

// Upstream revision 2dfe5e26aecfd9e5f27bf9d5837deea01acda24b.
// NDEBUG is intentionally supplied by h100_common.py because the pinned header
// contains an upstream-only debug assert with an undeclared identifier. Every
// launch precondition hidden by that release-mode build is checked below.
#include "upstream/h100/matmul/matmul_12.cuh"

namespace {

constexpr int kBlockM = 128;
constexpr int kBlockN = 256;
constexpr int kBlockK = 64;
constexpr int kThreads = 128 * 3;
constexpr int kStages = 3;
constexpr int kClusterM = 2;
constexpr int kClusterN = 1;
constexpr int kGridCtas = 128;
constexpr int kScheduleClusters = kGridCtas / (kClusterM * kClusterN);
constexpr int kScheduleEntries = M12::SPACE_LEN;

void check_cublas(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      operation,
      " failed with cuBLAS status ",
      static_cast<int>(status));
}

void check_exact_h100(int device) {
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  TORCH_CHECK(
      properties.major == 9 && properties.minor == 0,
      "fast.cu H100 BF16 GEMM requires exact SM 9.0 (sm_90a); got SM ",
      properties.major,
      ".",
      properties.minor);
}

void check_gemm_tensors(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& physical_output) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && physical_output.is_cuda(), "all tensors must be CUDA tensors");
  TORCH_CHECK(a.device() == b.device() && a.device() == physical_output.device(), "all tensors must use one CUDA device");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16, "A must have dtype torch.bfloat16");
  TORCH_CHECK(b.scalar_type() == at::kBFloat16, "B must have dtype torch.bfloat16");
  TORCH_CHECK(physical_output.scalar_type() == at::kBFloat16, "output must have dtype torch.bfloat16");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && physical_output.dim() == 2, "A, B, and output must be rank-2 tensors");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && physical_output.is_contiguous(), "A, B, and physical output must be contiguous");
  TORCH_CHECK(a.size(1) == b.size(1), "A[M,K] and B[N,K] must share K");

  const int64_t m = a.size(0);
  const int64_t n = b.size(0);
  const int64_t k = a.size(1);
  TORCH_CHECK(m > 0 && n > 0 && k > 0, "GEMM dimensions must be positive");
  TORCH_CHECK(
      m <= std::numeric_limits<int>::max() &&
          n <= std::numeric_limits<int>::max() &&
          k <= std::numeric_limits<int>::max(),
      "GEMM dimensions exceed the upstream int32 ABI");
  TORCH_CHECK(
      physical_output.size(0) == n && physical_output.size(1) == m,
      "physical output must have transposed [N,M] layout; expected [",
      n,
      ",",
      m,
      "]");
  TORCH_CHECK(m % (kBlockM * kClusterM) == 0, "M must be divisible by 256");
  TORCH_CHECK(n % kBlockN == 0, "N must be divisible by 256");
  TORCH_CHECK(k % kBlockK == 0, "K must be divisible by 64");
  TORCH_CHECK(
      std::max(m / (kBlockM * kClusterM), n / kBlockN) > 1,
      "upstream Hilbert schedule requires at least two tiles along M or N");
  const int64_t cluster_tiles =
      (m / (kBlockM * kClusterM)) * (n / (kBlockN * kClusterN));
  TORCH_CHECK(
      cluster_tiles <= static_cast<int64_t>(kScheduleClusters) * kScheduleEntries,
      "upstream schedule capacity exceeded: ",
      cluster_tiles,
      " cluster tiles for ",
      kScheduleClusters * kScheduleEntries,
      " slots");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0, "A must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0, "B must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(physical_output.data_ptr()) % 16 == 0, "output must be 16-byte aligned");
  check_exact_h100(a.get_device());
}

template <int BlockMajorSize, int BlockMinorSize, bool Swizzle = true, bool Padding = false>
CUtensorMap checked_tensor_map(bf16* pointer, int height, int width) {
  static_assert(BlockMinorSize >= 64);
  TORCH_CHECK(width % 64 == 0, "TMA width must be divisible by 64");
  CUtensorMap tensor_map{};
  uint64_t global_shape[5] = {
      64, static_cast<uint64_t>(height), static_cast<uint64_t>(width / 64), 1, 1};
  uint64_t global_stride[5] = {
      sizeof(bf16) * static_cast<uint64_t>(width), 64 * sizeof(bf16), 0, 0, 0};
  uint32_t box_shape[5] = {
      Padding ? 72u : 64u,
      static_cast<uint32_t>(BlockMajorSize),
      static_cast<uint32_t>(BlockMinorSize / 64),
      1,
      1};
  uint32_t element_stride[5] = {1, 1, 1, 1, 1};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      3,
      pointer,
      global_shape,
      global_stride,
      box_shape,
      element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      Swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  const char* error = nullptr;
  if (result != CUDA_SUCCESS) {
    cuGetErrorString(result, &error);
  }
  TORCH_CHECK(
      result == CUDA_SUCCESS,
      "cuTensorMapEncodeTiled failed: ",
      error == nullptr ? "unknown CUDA driver error" : error);
  return tensor_map;
}

struct CublasState {
  cublasHandle_t handle = nullptr;
  int device = -1;
  bool ready = false;
};

struct UpstreamGemmPlan {
  bool ready = false;
  int device = -1;
  int m = 0;
  int n = 0;
  int k = 0;
  const void* a_pointer = nullptr;
  const void* b_pointer = nullptr;
  void* output_pointer = nullptr;
  CUtensorMap a_map{};
  CUtensorMap b_map{};
  CUtensorMap output_map{};
  torch::Tensor schedule;
};

CublasState g_cublas;
UpstreamGemmPlan g_upstream;

void prepare_cublas_gemm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& physical_output) {
  check_gemm_tensors(a, b, physical_output);
  const c10::cuda::CUDAGuard guard(a.device());
  const int device = a.get_device();
  if (g_cublas.handle != nullptr && g_cublas.device != device) {
    check_cublas(cublasDestroy(g_cublas.handle), "cublasDestroy");
    g_cublas = CublasState{};
  }
  if (g_cublas.handle == nullptr) {
    check_cublas(cublasCreate(&g_cublas.handle), "cublasCreate");
    check_cublas(
        cublasSetMathMode(g_cublas.handle, CUBLAS_TENSOR_OP_MATH),
        "cublasSetMathMode");
  }
  g_cublas.device = device;
  g_cublas.ready = true;
}

void run_cublas_gemm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& physical_output) {
  check_gemm_tensors(a, b, physical_output);
  const c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(
      g_cublas.ready && g_cublas.handle != nullptr && g_cublas.device == a.get_device(),
      "prepare_cublas_gemm() must run on this device before timing");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device());
  check_cublas(cublasSetStream(g_cublas.handle, stream), "cublasSetStream");

  const int m = static_cast<int>(a.size(0));
  const int n = static_cast<int>(b.size(0));
  const int k = static_cast<int>(a.size(1));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(
      cublasGemmEx(
          g_cublas.handle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          m,
          n,
          k,
          &alpha,
          a.data_ptr(),
          CUDA_R_16BF,
          k,
          b.data_ptr(),
          CUDA_R_16BF,
          k,
          &beta,
          physical_output.data_ptr(),
          CUDA_R_16BF,
          m,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "cublasGemmEx");
}

void prepare_upstream_gemm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& physical_output) {
  check_gemm_tensors(a, b, physical_output);
  const c10::cuda::CUDAGuard guard(a.device());
  const int m = static_cast<int>(a.size(0));
  const int n = static_cast<int>(b.size(0));
  const int k = static_cast<int>(a.size(1));

  UpstreamGemmPlan next{};
  next.device = a.get_device();
  next.m = m;
  next.n = n;
  next.k = k;
  next.a_pointer = a.data_ptr();
  next.b_pointer = b.data_ptr();
  next.output_pointer = physical_output.data_ptr();
  next.a_map = checked_tensor_map<kBlockM, kBlockK>(
      reinterpret_cast<bf16*>(a.data_ptr()), m, k);
  next.b_map = checked_tensor_map<kBlockN, kBlockK>(
      reinterpret_cast<bf16*>(b.data_ptr()), n, k);
  next.output_map = checked_tensor_map<
      kBlockN, kBlockM / ((kThreads / 128) - 1), false, true>(
      reinterpret_cast<bf16*>(physical_output.data_ptr()), n, m);

  std::vector<int> host_schedule(kGridCtas * kScheduleEntries, -1);
  M12::createHilbert(
      m / (kBlockM * kClusterM),
      n / (kBlockN * kClusterN),
      kScheduleClusters,
      host_schedule.data());
  next.schedule = torch::empty(
      {static_cast<int64_t>(host_schedule.size())},
      a.options().dtype(torch::kInt32));
  C10_CUDA_CHECK(cudaMemcpy(
      next.schedule.data_ptr(),
      host_schedule.data(),
      host_schedule.size() * sizeof(int),
      cudaMemcpyHostToDevice));

  auto* kernel = M12::matmulKernel12<
      kBlockM,
      kBlockN,
      kBlockK,
      kThreads,
      kStages,
      kGridCtas,
      kClusterM,
      kClusterN>;
  constexpr size_t shared_bytes =
      sizeof(M12::SMem<kBlockM, kBlockN, kBlockK, kStages>);
  static_assert(shared_bytes < 256 * 1024);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  next.ready = true;
  g_upstream = std::move(next);
}

void run_upstream_gemm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& physical_output) {
  check_gemm_tensors(a, b, physical_output);
  const c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(g_upstream.ready, "prepare_upstream_gemm() must run before timing");
  TORCH_CHECK(g_upstream.device == a.get_device(), "prepared GEMM device changed");
  TORCH_CHECK(
      g_upstream.m == a.size(0) &&
          g_upstream.n == b.size(0) &&
          g_upstream.k == a.size(1),
      "prepared GEMM dimensions changed");
  TORCH_CHECK(
      g_upstream.a_pointer == a.data_ptr() &&
          g_upstream.b_pointer == b.data_ptr() &&
          g_upstream.output_pointer == physical_output.data_ptr(),
      "prepared GEMM storage changed; call prepare_upstream_gemm() again");

  auto* kernel = M12::matmulKernel12<
      kBlockM,
      kBlockN,
      kBlockK,
      kThreads,
      kStages,
      kGridCtas,
      kClusterM,
      kClusterN>;
  constexpr size_t shared_bytes =
      sizeof(M12::SMem<kBlockM, kBlockN, kBlockK, kStages>);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device());
  kernel<<<kGridCtas, kThreads, shared_bytes, stream>>>(
      g_upstream.m,
      g_upstream.n,
      g_upstream.k,
      g_upstream.output_map,
      g_upstream.a_map,
      g_upstream.b_map,
      g_upstream.schedule.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("prepare_cublas_gemm", &prepare_cublas_gemm);
  module.def("run_cublas_gemm", &run_cublas_gemm);
  module.def("prepare_upstream_gemm", &prepare_upstream_gemm);
  module.def("run_upstream_gemm", &run_upstream_gemm);
}
