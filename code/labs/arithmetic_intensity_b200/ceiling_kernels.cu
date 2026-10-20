#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#define CUDA_CHECK(call)                                                            \
  do {                                                                              \
    const cudaError_t status = (call);                                               \
    TORCH_CHECK(status == cudaSuccess, "CUDA error: ", cudaGetErrorString(status)); \
  } while (0)

namespace {

void check_byte_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8, name, " must have dtype uint8");
}

void check_launch(int threads, int blocks) {
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512,
              "threads must be 128, 256, or 512");
  TORCH_CHECK(blocks > 0, "blocks must be positive");
}

__device__ __forceinline__ uint32_t mix32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

__device__ __forceinline__ uint4 generated_value(uint64_t index, uint32_t seed) {
  const uint32_t folded = static_cast<uint32_t>(index) ^
                          static_cast<uint32_t>(index >> 32) ^ seed;
  const uint32_t base = mix32(folded);
  return make_uint4(
      base,
      mix32(base ^ 0x9e3779b9U),
      mix32(base ^ 0x243f6a88U),
      mix32(base ^ 0xb7e15162U));
}

__global__ void copy_scalar_kernel(
    const uint8_t* __restrict__ source,
    uint8_t* __restrict__ destination,
    uint64_t count) {
  const uint64_t thread_id =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = thread_id; index < count; index += stride) {
    destination[index] = source[index];
  }
}

__global__ void copy_uint4_kernel(
    const uint4* __restrict__ source,
    uint4* __restrict__ destination,
    uint64_t count) {
  const uint64_t thread_id =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = thread_id; index < count; index += stride) {
    destination[index] = source[index];
  }
}

__global__ void write_uint4_kernel(
    uint4* __restrict__ destination,
    uint64_t count,
    uint32_t seed) {
  const uint64_t thread_id =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = thread_id; index < count; index += stride) {
    destination[index] = generated_value(index, seed);
  }
}

__global__ void validate_write_uint4_kernel(
    const uint4* __restrict__ actual,
    uint64_t count,
    uint32_t seed,
    unsigned long long* __restrict__ mismatches) {
  const uint64_t thread_id =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  unsigned long long local = 0;
  for (uint64_t index = thread_id; index < count; index += stride) {
    const uint4 observed = actual[index];
    const uint4 expected = generated_value(index, seed);
    local += observed.x != expected.x;
    local += observed.y != expected.y;
    local += observed.z != expected.z;
    local += observed.w != expected.w;
  }
  if (local != 0) {
    atomicAdd(mismatches, local);
  }
}

__device__ __forceinline__ unsigned long long warp_sum(unsigned long long value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

__global__ void read_chunk_sum_kernel(
    const uint32_t* __restrict__ source,
    int64_t* __restrict__ chunk_sums,
    uint64_t word_count,
    uint64_t words_per_chunk) {
  __shared__ unsigned long long warp_sums[32];
  const uint64_t chunk_begin = static_cast<uint64_t>(blockIdx.x) * words_per_chunk;
  const uint64_t chunk_end = min(chunk_begin + words_per_chunk, word_count);
  unsigned long long local = 0;
  for (uint64_t index = chunk_begin + threadIdx.x;
       index < chunk_end;
       index += blockDim.x) {
    local += static_cast<unsigned long long>(source[index]);
  }

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  local = warp_sum(local);
  if (lane == 0) {
    warp_sums[warp] = local;
  }
  __syncthreads();

  if (warp == 0) {
    const int warp_count = (blockDim.x + 31) / 32;
    unsigned long long block_sum = lane < warp_count ? warp_sums[lane] : 0;
    block_sum = warp_sum(block_sum);
    if (lane == 0) {
      chunk_sums[blockIdx.x] = static_cast<int64_t>(block_sum);
    }
  }
}

}  // namespace

void copy_scalar(torch::Tensor source, torch::Tensor destination, int threads, int blocks) {
  check_byte_tensor(source, "source");
  check_byte_tensor(destination, "destination");
  TORCH_CHECK(source.numel() == destination.numel(), "copy tensors must have equal sizes");
  TORCH_CHECK(source.device() == destination.device(), "copy tensors must share a device");
  check_launch(threads, blocks);
  c10::cuda::CUDAGuard guard(source.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  copy_scalar_kernel<<<blocks, threads, 0, stream>>>(
      source.data_ptr<uint8_t>(),
      destination.data_ptr<uint8_t>(),
      static_cast<uint64_t>(source.numel()));
  CUDA_CHECK(cudaGetLastError());
}

void copy_uint4(torch::Tensor source, torch::Tensor destination, int threads, int blocks) {
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
  check_launch(threads, blocks);
  c10::cuda::CUDAGuard guard(source.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  copy_uint4_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint4*>(source.data_ptr<uint8_t>()),
      reinterpret_cast<uint4*>(destination.data_ptr<uint8_t>()),
      static_cast<uint64_t>(source.numel()) / 16);
  CUDA_CHECK(cudaGetLastError());
}

void write_uint4(
    torch::Tensor destination,
    int threads,
    int blocks,
    uint32_t seed) {
  check_byte_tensor(destination, "destination");
  TORCH_CHECK(destination.numel() % 16 == 0, "uint4 write requires a byte count divisible by 16");
  check_launch(threads, blocks);
  c10::cuda::CUDAGuard guard(destination.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  write_uint4_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<uint4*>(destination.data_ptr<uint8_t>()),
      static_cast<uint64_t>(destination.numel()) / 16,
      seed);
  CUDA_CHECK(cudaGetLastError());
}

torch::Tensor validate_write_uint4(
    torch::Tensor destination,
    int threads,
    int blocks,
    uint32_t seed) {
  check_byte_tensor(destination, "destination");
  TORCH_CHECK(destination.numel() % 16 == 0, "uint4 validation requires a byte count divisible by 16");
  check_launch(threads, blocks);
  c10::cuda::CUDAGuard guard(destination.device());
  auto mismatches = torch::zeros(
      {1},
      torch::TensorOptions().dtype(torch::kInt64).device(destination.device()));
  const auto stream = at::cuda::getCurrentCUDAStream();
  validate_write_uint4_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint4*>(destination.data_ptr<uint8_t>()),
      static_cast<uint64_t>(destination.numel()) / 16,
      seed,
      reinterpret_cast<unsigned long long*>(mismatches.data_ptr<int64_t>()));
  CUDA_CHECK(cudaGetLastError());
  return mismatches;
}

void read_chunk_sums(
    torch::Tensor source,
    torch::Tensor sums,
    int threads,
    int64_t chunk_bytes) {
  check_byte_tensor(source, "source");
  TORCH_CHECK(sums.is_cuda(), "sums must be a CUDA tensor");
  TORCH_CHECK(sums.is_contiguous(), "sums must be contiguous");
  TORCH_CHECK(sums.scalar_type() == torch::kInt64, "sums must have dtype int64");
  TORCH_CHECK(sums.device() == source.device(), "source and sums must share a device");
  TORCH_CHECK(source.numel() % 4 == 0, "read checksum requires a byte count divisible by 4");
  TORCH_CHECK(chunk_bytes > 0 && chunk_bytes % 4 == 0,
              "chunk_bytes must be positive and divisible by 4");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512,
              "threads must be 128, 256, or 512");
  const uint64_t word_count = static_cast<uint64_t>(source.numel()) / 4;
  const uint64_t words_per_chunk = static_cast<uint64_t>(chunk_bytes) / 4;
  const uint64_t chunk_count = (word_count + words_per_chunk - 1) / words_per_chunk;
  TORCH_CHECK(chunk_count <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
              "chunk count exceeds CUDA grid limits");
  TORCH_CHECK(sums.numel() == static_cast<int64_t>(chunk_count),
              "sums has the wrong number of chunks");
  c10::cuda::CUDAGuard guard(source.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  read_chunk_sum_kernel<<<static_cast<int>(chunk_count), threads, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(source.data_ptr<uint8_t>()),
      sums.data_ptr<int64_t>(),
      word_count,
      words_per_chunk);
  CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("copy_scalar", &copy_scalar, "Scalar byte copy");
  module.def("copy_uint4", &copy_uint4, "Vectorized uint4 copy");
  module.def("write_uint4", &write_uint4, "Deterministic high-entropy uint4 write");
  module.def("validate_write_uint4", &validate_write_uint4, "Full deterministic write validation");
  module.def("read_chunk_sums", &read_chunk_sums, "Exact uint32 sums for fixed input chunks");
}
