#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>

#define CUDA_CHECK(call)                                                            \
  do {                                                                              \
    const cudaError_t status = (call);                                               \
    TORCH_CHECK(status == cudaSuccess, "CUDA error: ", cudaGetErrorString(status)); \
  } while (0)

namespace {

void check_byte_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8, name, " must use uint8");
}

__global__ void copy_uint4_cg_unroll4_kernel(
    const uint4* __restrict__ source,
    uint4* __restrict__ destination,
    uint64_t count) {
  const uint64_t thread_id =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  const uint64_t unrolled_stride = stride * 4;
  for (uint64_t index = thread_id; index < count; index += unrolled_stride) {
    destination[index] = __ldcg(source + index);
    if (index + stride < count) {
      destination[index + stride] = __ldcg(source + index + stride);
    }
    if (index + 2 * stride < count) {
      destination[index + 2 * stride] = __ldcg(source + index + 2 * stride);
    }
    if (index + 3 * stride < count) {
      destination[index + 3 * stride] = __ldcg(source + index + 3 * stride);
    }
  }
}

}  // namespace

void copy_uint4_cg_unroll4_cuda(
    torch::Tensor source,
    torch::Tensor destination,
    int threads,
    int blocks) {
  check_byte_tensor(source, "source");
  check_byte_tensor(destination, "destination");
  TORCH_CHECK(source.numel() == destination.numel(), "copy tensors must have equal sizes");
  TORCH_CHECK(source.device() == destination.device(), "copy tensors must share a device");
  TORCH_CHECK(source.numel() % 16 == 0, "uint4 copy requires a byte count divisible by 16");
  const auto source_address = reinterpret_cast<uintptr_t>(source.data_ptr<uint8_t>());
  const auto destination_address = reinterpret_cast<uintptr_t>(destination.data_ptr<uint8_t>());
  TORCH_CHECK(source_address % alignof(uint4) == 0,
              "source storage offset must preserve 16-byte uint4 alignment");
  TORCH_CHECK(destination_address % alignof(uint4) == 0,
              "destination storage offset must preserve 16-byte uint4 alignment");
  const auto byte_count = static_cast<uintptr_t>(source.numel());
  TORCH_CHECK(source_address + byte_count <= destination_address ||
                  destination_address + byte_count <= source_address,
              "source and destination must not overlap");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512,
              "threads must be 128, 256, or 512");
  TORCH_CHECK(blocks > 0, "blocks must be positive");
  c10::cuda::CUDAGuard guard(source.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  copy_uint4_cg_unroll4_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint4*>(source.data_ptr<uint8_t>()),
      reinterpret_cast<uint4*>(destination.data_ptr<uint8_t>()),
      static_cast<uint64_t>(source.numel()) / 16);
  CUDA_CHECK(cudaGetLastError());
}
