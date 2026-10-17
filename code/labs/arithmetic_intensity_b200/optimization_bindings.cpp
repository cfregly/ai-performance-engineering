#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                            \
  do {                                                                              \
    const cudaError_t status = (call);                                               \
    TORCH_CHECK(status == cudaSuccess, "CUDA error: ", cudaGetErrorString(status)); \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    const cublasStatus_t status = (call);                                       \
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,                                \
                "cuBLASLt error ", static_cast<int>(status), " in ", #call); \
  } while (0)

void copy_uint4_cg_unroll4_cuda(
    torch::Tensor source,
    torch::Tensor destination,
    int threads,
    int blocks);

namespace {

void check_b200() {
  int device = -1;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  const std::string device_name(properties.name);
  TORCH_CHECK(
      properties.major == 10 && properties.minor == 0 &&
          device_name.find("B200") != std::string::npos,
      "The B200 optimization kernels require an NVIDIA B200 at compute capability 10.0, got ",
      device_name, " at ", properties.major, ".", properties.minor);
}

void check_bf16_matrix(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous row-major");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must use BF16");
  TORCH_CHECK(tensor.dim() == 2, name, " must be rank two");
}

bool tensors_overlap(const torch::Tensor& left, const torch::Tensor& right) {
  const auto left_start =
      reinterpret_cast<uintptr_t>(left.data_ptr<at::BFloat16>());
  const auto right_start =
      reinterpret_cast<uintptr_t>(right.data_ptr<at::BFloat16>());
  const auto left_end = left_start + left.numel() * left.element_size();
  const auto right_end = right_start + right.numel() * right.element_size();
  return left_start < right_end && right_start < left_end;
}

template <typename T>
int64_t algo_attribute(
    const cublasLtMatmulAlgo_t& algo,
    cublasLtMatmulAlgoConfigAttributes_t attribute) {
  T value{};
  size_t written = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
      &algo, attribute, &value, sizeof(value), &written));
  TORCH_CHECK(written == sizeof(value), "cuBLASLt returned a truncated algorithm attribute");
  return static_cast<int64_t>(value);
}

std::string algorithm_signature(const cublasLtMatmulAlgo_t& algo) {
  std::ostringstream stream;
  stream << "id=" << algo_attribute<int>(algo, CUBLASLT_ALGO_CONFIG_ID)
         << ",tile=" << algo_attribute<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_TILE_ID)
         << ",stages=" << algo_attribute<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID)
         << ",split_k=" << algo_attribute<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM)
         << ",reduction="
         << algo_attribute<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)
         << ",swizzle="
         << algo_attribute<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)
         << ",custom="
         << algo_attribute<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)
         << ",cluster="
         << algo_attribute<uint16_t>(algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID);
  return stream.str();
}

pybind11::dict blas_runtime_versions() {
  cublasHandle_t cublas_handle = nullptr;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  int cublas_version = 0;
  const cublasStatus_t version_status =
      cublasGetVersion(cublas_handle, &cublas_version);
  const cublasStatus_t destroy_status = cublasDestroy(cublas_handle);
  CUBLAS_CHECK(version_status);
  CUBLAS_CHECK(destroy_status);
  pybind11::dict versions;
  versions["cublas"] = static_cast<int64_t>(cublas_version);
  versions["cublaslt"] = static_cast<int64_t>(cublasLtGetVersion());
  return versions;
}

}  // namespace

class Bf16LtMatmulPlan {
 public:
  Bf16LtMatmulPlan(
      torch::Tensor a,
      torch::Tensor b,
      torch::Tensor output,
      int64_t workspace_bytes,
      int max_algorithms)
      : device_index_(a.get_device()),
        m_(a.size(0)),
        k_(a.size(1)),
        n_(b.size(1)),
        workspace_bytes_(workspace_bytes) {
    c10::cuda::CUDAGuard guard(a.device());
    check_b200();
    check_problem(a, b, output);
    TORCH_CHECK(workspace_bytes >= 0, "workspace_bytes must be nonnegative");
    TORCH_CHECK(max_algorithms > 0 && max_algorithms <= 128,
                "max_algorithms must be in [1, 128]");
    CUBLAS_CHECK(cublasLtMatmulDescCreate(
        &operation_desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t operation = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operation_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &operation, sizeof(operation)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operation_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &operation, sizeof(operation)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_layout_, CUDA_R_16BF, m_, k_, k_));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_layout_, CUDA_R_16BF, k_, n_, n_));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_layout_, CUDA_R_16BF, m_, n_, n_));
    const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        a_layout_, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        b_layout_, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        c_layout_, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference_));
    const size_t maximum_workspace = static_cast<size_t>(workspace_bytes_);
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference_,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &maximum_workspace,
        sizeof(maximum_workspace)));
    workspace_ = torch::empty(
        {workspace_bytes_},
        torch::TensorOptions().dtype(torch::kUInt8).device(a.device()));
    std::vector<cublasLtMatmulHeuristicResult_t> candidates(max_algorithms);
    int returned = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        at::cuda::getCurrentCUDABlasLtHandle(),
        operation_desc_,
        a_layout_,
        b_layout_,
        c_layout_,
        c_layout_,
        preference_,
        max_algorithms,
        candidates.data(),
        &returned));
    for (int index = 0; index < returned; ++index) {
      if (candidates[index].state == CUBLAS_STATUS_SUCCESS &&
          candidates[index].workspaceSize <= maximum_workspace) {
        algorithms_.push_back(candidates[index]);
      }
    }
    TORCH_CHECK(!algorithms_.empty(),
                "cuBLASLt returned no supported BF16 row-major algorithms for the problem");
  }

  ~Bf16LtMatmulPlan() {
    if (preference_ != nullptr) {
      cublasLtMatmulPreferenceDestroy(preference_);
    }
    if (c_layout_ != nullptr) {
      cublasLtMatrixLayoutDestroy(c_layout_);
    }
    if (b_layout_ != nullptr) {
      cublasLtMatrixLayoutDestroy(b_layout_);
    }
    if (a_layout_ != nullptr) {
      cublasLtMatrixLayoutDestroy(a_layout_);
    }
    if (operation_desc_ != nullptr) {
      cublasLtMatmulDescDestroy(operation_desc_);
    }
  }

  Bf16LtMatmulPlan(const Bf16LtMatmulPlan&) = delete;
  Bf16LtMatmulPlan& operator=(const Bf16LtMatmulPlan&) = delete;

  int64_t algorithm_count() const {
    return static_cast<int64_t>(algorithms_.size());
  }

  pybind11::list algorithms() const {
    pybind11::list result;
    for (size_t index = 0; index < algorithms_.size(); ++index) {
      const auto& candidate = algorithms_[index];
      pybind11::dict metadata;
      metadata["heuristic_index"] = static_cast<int64_t>(index);
      metadata["algorithm_id"] =
          algo_attribute<int>(candidate.algo, CUBLASLT_ALGO_CONFIG_ID);
      metadata["tile_id"] =
          algo_attribute<uint32_t>(candidate.algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
      metadata["stages_id"] =
          algo_attribute<uint32_t>(candidate.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
      metadata["split_k"] =
          algo_attribute<uint32_t>(candidate.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
      metadata["reduction_scheme"] = algo_attribute<uint32_t>(
          candidate.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
      metadata["cta_swizzle"] = algo_attribute<uint32_t>(
          candidate.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
      metadata["custom_option"] = algo_attribute<uint32_t>(
          candidate.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
      metadata["cluster_shape_id"] = algo_attribute<uint16_t>(
          candidate.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID);
      metadata["workspace_bytes"] = static_cast<int64_t>(candidate.workspaceSize);
      metadata["waves_count"] = static_cast<double>(candidate.wavesCount);
      metadata["signature"] = algorithm_signature(candidate.algo);
      result.append(metadata);
    }
    return result;
  }

  void run(torch::Tensor a, torch::Tensor b, torch::Tensor output, int algorithm_index) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    c10::cuda::CUDAGuard guard(a.device());
    check_problem(a, b, output);
    TORCH_CHECK(algorithm_index >= 0 &&
                    algorithm_index < static_cast<int>(algorithms_.size()),
                "algorithm_index is outside the supported heuristic set");
    const float alpha = 1.0F;
    const float beta = 0.0F;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto& candidate = algorithms_[algorithm_index];
    CUBLAS_CHECK(cublasLtMatmul(
        at::cuda::getCurrentCUDABlasLtHandle(),
        operation_desc_,
        &alpha,
        a.data_ptr<at::BFloat16>(),
        a_layout_,
        b.data_ptr<at::BFloat16>(),
        b_layout_,
        &beta,
        output.data_ptr<at::BFloat16>(),
        c_layout_,
        output.data_ptr<at::BFloat16>(),
        c_layout_,
        &candidate.algo,
        workspace_bytes_ == 0 ? nullptr : workspace_.data_ptr<uint8_t>(),
        static_cast<size_t>(workspace_bytes_),
        stream));
  }

 private:
  void check_problem(
      const torch::Tensor& a,
      const torch::Tensor& b,
      const torch::Tensor& output) const {
    check_bf16_matrix(a, "a");
    check_bf16_matrix(b, "b");
    check_bf16_matrix(output, "output");
    TORCH_CHECK(a.get_device() == device_index_ && b.get_device() == device_index_ &&
                    output.get_device() == device_index_,
                "all tensors must remain on the plan device");
    TORCH_CHECK(a.size(0) == m_ && a.size(1) == k_, "a shape differs from the plan");
    TORCH_CHECK(b.size(0) == k_ && b.size(1) == n_, "b shape differs from the plan");
    TORCH_CHECK(output.size(0) == m_ && output.size(1) == n_,
                "output shape differs from the plan");
    TORCH_CHECK(!tensors_overlap(a, output),
                "output must not overlap input a");
    TORCH_CHECK(!tensors_overlap(b, output),
                "output must not overlap input b");
  }

  int device_index_;
  int64_t m_;
  int64_t k_;
  int64_t n_;
  int64_t workspace_bytes_;
  cublasLtMatmulDesc_t operation_desc_{nullptr};
  cublasLtMatrixLayout_t a_layout_{nullptr};
  cublasLtMatrixLayout_t b_layout_{nullptr};
  cublasLtMatrixLayout_t c_layout_{nullptr};
  cublasLtMatmulPreference_t preference_{nullptr};
  torch::Tensor workspace_;
  std::vector<cublasLtMatmulHeuristicResult_t> algorithms_;
};

void copy_uint4_cg_unroll4(
    torch::Tensor source,
    torch::Tensor destination,
    int threads,
    int blocks) {
  TORCH_CHECK(source.is_cuda(), "source must be a CUDA tensor");
  c10::cuda::CUDAGuard guard(source.device());
  check_b200();
  copy_uint4_cg_unroll4_cuda(source, destination, threads, blocks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "blas_runtime_versions",
      &blas_runtime_versions,
      "Return the loaded cuBLAS and cuBLASLt runtime versions");
  pybind11::class_<Bf16LtMatmulPlan, std::shared_ptr<Bf16LtMatmulPlan>>(
      module, "Bf16LtMatmulPlan")
      .def(pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int>())
      .def("algorithm_count", &Bf16LtMatmulPlan::algorithm_count)
      .def("algorithms", &Bf16LtMatmulPlan::algorithms)
      .def("run", &Bf16LtMatmulPlan::run);
  module.def(
      "copy_uint4_cg_unroll4",
      &copy_uint4_cg_unroll4,
      "Cache-global unrolled uint4 copy");
}
