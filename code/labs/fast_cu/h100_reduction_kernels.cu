// Safe, stream-correct adapter for pranjalssh/fast.cu h100/sum.cu.
//
// The upstream vectorized reduction resets the global output from one CTA while
// other CTAs can already atomicAdd into it. This adaptation keeps its int4 load,
// per-thread batching, warp shuffle, shared warp sums, and global atomic, while
// moving the reset to cudaMemsetAsync before launch on the same PyTorch stream.
// It intentionally lives outside the WGMMA translation unit so the same source
// can compile natively for both H100 SM90a and B200 SM100a.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockSize = 1024;
constexpr int kBatchVectors = 16;

void check_supported_device(int device) {
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  const bool supported =
      (properties.major == 9 && properties.minor == 0) ||
      (properties.major == 10 && properties.minor == 0);
  TORCH_CHECK(
      supported,
      "fast.cu int32 reduction supports exact SM 9.0 (H100) and SM 10.0 (B200); got SM ",
      properties.major,
      ".",
      properties.minor);
}

void check_reduction_tensors(
    const torch::Tensor& input, const torch::Tensor& output) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input and output must be CUDA tensors");
  TORCH_CHECK(input.device() == output.device(), "input and output must use one CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kInt, "input must have dtype torch.int32");
  TORCH_CHECK(output.scalar_type() == at::kInt, "output must have dtype torch.int32");
  TORCH_CHECK(input.dim() == 1 && output.dim() == 1, "input and output must be rank-1 tensors");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "input and output must be contiguous");
  TORCH_CHECK(output.numel() == 1, "output must contain exactly one int32 value");
  TORCH_CHECK(input.numel() > 0, "input must be non-empty");
  TORCH_CHECK(input.numel() % 4 == 0, "input length must be divisible by four for int4 loads");
  TORCH_CHECK(
      input.numel() <= std::numeric_limits<int>::max(),
      "input length exceeds the upstream int32 indexing ABI");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.data_ptr()) % alignof(int4) == 0,
      "input storage must be 16-byte aligned for int4 loads");
  check_supported_device(input.get_device());
}

__device__ __forceinline__ int warp_reduce_sum(int value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// Derived from upstream sumKernel2<1024, 16>. The only semantic change inside
// the kernel is removal of `if (i == 0) *d_out = 0`; reset ordering belongs to
// the host launch sequence below, where it cannot race another CTA.
__global__ __launch_bounds__(kBlockSize) void safe_vectorized_sum_kernel(
    const int4* input, int* output, int vector_count) {
  __shared__ __align__(16) int warp_sums[kWarpSize];

  const unsigned int tid = threadIdx.x;
  const unsigned int index =
      blockIdx.x * kBlockSize * kBatchVectors + tid;
  int4 values = index < vector_count
      ? input[index]
      : make_int4(0, 0, 0, 0);
  int sum = values.x + values.y + values.z + values.w;
#pragma unroll
  for (int batch = 1; batch < kBatchVectors; ++batch) {
    const unsigned int next = index + batch * kBlockSize;
    if (next < vector_count) {
      values = input[next];
      sum += values.x + values.y + values.z + values.w;
    }
  }

  sum = warp_reduce_sum(sum);
  if ((tid & (kWarpSize - 1)) == 0) {
    warp_sums[tid / kWarpSize] = sum;
  }
  __syncthreads();

  if (tid < kWarpSize) {
    sum = warp_reduce_sum(warp_sums[tid]);
  }
  if (tid == 0) {
    atomicAdd(output, sum);
  }
}

int64_t cub_temp_storage_bytes(
    const torch::Tensor& input, const torch::Tensor& output) {
  check_reduction_tensors(input, output);
  const c10::cuda::CUDAGuard guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  size_t storage_bytes = 0;
  C10_CUDA_CHECK(cub::DeviceReduce::Sum(
      nullptr,
      storage_bytes,
      input.data_ptr<int>(),
      output.data_ptr<int>(),
      static_cast<int>(input.numel()),
      stream));
  TORCH_CHECK(storage_bytes > 0, "CUB returned zero temporary storage bytes");
  TORCH_CHECK(
      storage_bytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
      "CUB temporary storage size exceeds Python int64");
  return static_cast<int64_t>(storage_bytes);
}

void run_cub_reduction(
    const torch::Tensor& input,
    const torch::Tensor& output,
    const torch::Tensor& temp_storage) {
  check_reduction_tensors(input, output);
  TORCH_CHECK(temp_storage.is_cuda(), "CUB temporary storage must be a CUDA tensor");
  TORCH_CHECK(temp_storage.device() == input.device(), "CUB temporary storage must use the input device");
  TORCH_CHECK(temp_storage.scalar_type() == at::kByte, "CUB temporary storage must have dtype torch.uint8");
  TORCH_CHECK(temp_storage.dim() == 1 && temp_storage.is_contiguous(), "CUB temporary storage must be contiguous and rank 1");
  TORCH_CHECK(temp_storage.numel() > 0, "CUB temporary storage must be non-empty");

  const c10::cuda::CUDAGuard guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  size_t storage_bytes = static_cast<size_t>(temp_storage.numel());
  C10_CUDA_CHECK(cub::DeviceReduce::Sum(
      temp_storage.data_ptr(),
      storage_bytes,
      input.data_ptr<int>(),
      output.data_ptr<int>(),
      static_cast<int>(input.numel()),
      stream));
}

void run_safe_vectorized_reduction(
    const torch::Tensor& input, const torch::Tensor& output) {
  check_reduction_tensors(input, output);
  const c10::cuda::CUDAGuard guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

  // This stream-ordered reset is part of every measured reduction. It must not
  // be hoisted into setup because each launch accumulates into the same buffer.
  C10_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, sizeof(int), stream));
  const int vector_count = static_cast<int>(input.numel() / 4);
  const int vectors_per_block = kBlockSize * kBatchVectors;
  const int blocks = (vector_count + vectors_per_block - 1) / vectors_per_block;
  safe_vectorized_sum_kernel<<<blocks, kBlockSize, 0, stream>>>(
      reinterpret_cast<const int4*>(input.data_ptr<int>()),
      output.data_ptr<int>(),
      vector_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("cub_temp_storage_bytes", &cub_temp_storage_bytes);
  module.def("run_cub_reduction", &run_cub_reduction);
  module.def("run_safe_vectorized_reduction", &run_safe_vectorized_reduction);
}
