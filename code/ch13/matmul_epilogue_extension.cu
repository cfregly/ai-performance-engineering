#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__global__ void epilogue_half2_kernel(
    __half* out,
    const __half* bias,
    const __half* residual,
    int64_t half2_count,
    __half2 scale) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= half2_count) {
    return;
  }

  auto* out2 = reinterpret_cast<__half2*>(out);
  const auto* bias2 = reinterpret_cast<const __half2*>(bias);
  const auto* residual2 = reinterpret_cast<const __half2*>(residual);

  __half2 value = __hadd2(out2[idx], bias2[idx]);
  value = __hmax2(value, __float2half2_rn(0.0f));
  value = __hadd2(value, residual2[idx]);
  out2[idx] = __hmul2(value, scale);
}

__global__ void epilogue_scalar_kernel(
    __half* out,
    const __half* bias,
    const __half* residual,
    int64_t count,
    float scale) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  float value = __half2float(out[idx]) + __half2float(bias[idx]);
  value = fmaxf(value, 0.0f);
  value += __half2float(residual[idx]);
  out[idx] = __float2half_rn(value * scale);
}

bool is_half2_aligned(const torch::Tensor& tensor) {
  return (reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignof(__half2)) == 0;
}

}  // namespace

torch::Tensor matmul_epilogue_(
    torch::Tensor out,
    torch::Tensor bias,
    torch::Tensor residual,
    double scale) {
  TORCH_CHECK(out.is_cuda() && bias.is_cuda() && residual.is_cuda(),
              "matmul epilogue tensors must be CUDA tensors");
  TORCH_CHECK(out.dtype() == torch::kFloat16 && bias.dtype() == torch::kFloat16 &&
                  residual.dtype() == torch::kFloat16,
              "matmul epilogue tensors must be float16");
  TORCH_CHECK(out.is_contiguous() && bias.is_contiguous() && residual.is_contiguous(),
              "matmul epilogue tensors must be contiguous");
  TORCH_CHECK(out.sizes() == bias.sizes() && out.sizes() == residual.sizes(),
              "matmul epilogue tensors must have matching shapes");

  const int64_t count = out.numel();
  if (count == 0) {
    return out;
  }

  constexpr int threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool use_half2 =
      (count % 2 == 0) && is_half2_aligned(out) && is_half2_aligned(bias) &&
      is_half2_aligned(residual);

  if (use_half2) {
    const int64_t half2_count = count / 2;
    const int blocks = static_cast<int>((half2_count + threads - 1) / threads);
    epilogue_half2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(residual.data_ptr<at::Half>()),
        half2_count,
        __float2half2_rn(static_cast<float>(scale)));
  } else {
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    epilogue_scalar_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(residual.data_ptr<at::Half>()),
        count,
        static_cast<float>(scale));
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_epilogue_", &matmul_epilogue_,
        "Fused in-place Ch13 matmul epilogue");
}
