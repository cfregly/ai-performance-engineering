#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#if __CUDACC_VER_MAJOR__ < 12 || \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 9)
#error "fast.cu B200 epilogue requires CUDA 12.9 or newer for st.global.v8.b32"
#endif

namespace {

constexpr int kThreads = 256;
constexpr int64_t kElementsPerThread = 16;
constexpr uintptr_t kWideStoreAlignment = 32;

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t packed;
  // PTX places its first f32 source in the high 16 bits. Reverse the operands
  // so the lower-addressed FP16 value occupies the low half of the b32 word.
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;"
               : "=r"(packed)
               : "f"(hi), "f"(lo));
  return packed;
}

__device__ __forceinline__ uint64_t make_evict_first_policy() {
  uint64_t policy;
  asm volatile(
      "createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
      : "=l"(policy));
  return policy;
}

__device__ __forceinline__ void store_b128(
    void* pointer,
    uint4 value,
    uint64_t policy) {
  asm volatile(
      "st.global.L1::no_allocate.L2::cache_hint.v4.b32 "
      "[%0], {%1, %2, %3, %4}, %5;"
      :
      : "l"(pointer), "r"(value.x), "r"(value.y), "r"(value.z),
        "r"(value.w), "l"(policy)
      : "memory");
}

__device__ __forceinline__ void store_b256(
    void* pointer,
    uint4 lo,
    uint4 hi,
    uint64_t policy) {
  asm volatile(
      "st.global.L1::no_allocate.L2::cache_hint.v8.b32 "
      "[%0], {%1, %2, %3, %4, %5, %6, %7, %8}, %9;"
      :
      : "l"(pointer), "r"(lo.x), "r"(lo.y), "r"(lo.z), "r"(lo.w),
        "r"(hi.x), "r"(hi.y), "r"(hi.z), "r"(hi.w), "l"(policy)
      : "memory");
}

template <bool WideStore>
__global__ void epilogue_kernel(
    const float* accumulators,
    __half* output,
    int64_t work_items) {
  const int64_t work_item =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (work_item >= work_items) {
    return;
  }

  const int64_t base = work_item * kElementsPerThread;
  uint4 lo;
  uint4 hi;
  lo.x = pack_f16x2(accumulators[base + 0], accumulators[base + 1]);
  lo.y = pack_f16x2(accumulators[base + 2], accumulators[base + 3]);
  lo.z = pack_f16x2(accumulators[base + 4], accumulators[base + 5]);
  lo.w = pack_f16x2(accumulators[base + 6], accumulators[base + 7]);
  hi.x = pack_f16x2(accumulators[base + 8], accumulators[base + 9]);
  hi.y = pack_f16x2(accumulators[base + 10], accumulators[base + 11]);
  hi.z = pack_f16x2(accumulators[base + 12], accumulators[base + 13]);
  hi.w = pack_f16x2(accumulators[base + 14], accumulators[base + 15]);

  const uint64_t policy = make_evict_first_policy();
  void* const destination = static_cast<void*>(output + base);
  if constexpr (WideStore) {
    store_b256(destination, lo, hi, policy);
  } else {
    store_b128(destination, lo, policy);
    store_b128(static_cast<void*>(output + base + 8), hi, policy);
  }
}

void check_tensor_contract(
    const torch::Tensor& accumulators,
    const torch::Tensor& output) {
  TORCH_CHECK(
      accumulators.is_cuda() && output.is_cuda(),
      "fast.cu B200 epilogue tensors must be CUDA tensors");
  TORCH_CHECK(
      accumulators.device() == output.device(),
      "fast.cu B200 epilogue tensors must be on the same CUDA device");
  TORCH_CHECK(
      accumulators.scalar_type() == at::kFloat,
      "fast.cu B200 epilogue accumulators must be float32");
  TORCH_CHECK(
      output.scalar_type() == at::kHalf,
      "fast.cu B200 epilogue output must be float16");
  TORCH_CHECK(
      accumulators.is_contiguous() && output.is_contiguous(),
      "fast.cu B200 epilogue tensors must be contiguous");
  TORCH_CHECK(
      accumulators.sizes() == output.sizes(),
      "fast.cu B200 epilogue input and output shapes must match");

  const int64_t count = accumulators.numel();
  TORCH_CHECK(count > 0, "fast.cu B200 epilogue count must be positive");
  TORCH_CHECK(
      count % kElementsPerThread == 0,
      "fast.cu B200 epilogue count must be divisible by 16. Found ",
      count);
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(output.data_ptr()) % kWideStoreAlignment == 0,
      "fast.cu B200 epilogue output must be 32-byte aligned");

  const uintptr_t input_begin =
      reinterpret_cast<uintptr_t>(accumulators.data_ptr());
  const uintptr_t input_end =
      input_begin + static_cast<uintptr_t>(count) * sizeof(float);
  const uintptr_t output_begin = reinterpret_cast<uintptr_t>(output.data_ptr());
  const uintptr_t output_end =
      output_begin + static_cast<uintptr_t>(count) * sizeof(__half);
  TORCH_CHECK(
      input_end <= output_begin || output_end <= input_begin,
      "fast.cu B200 epilogue input and output storage must not overlap");

  const int64_t work_items = count / kElementsPerThread;
  constexpr int64_t kMaxWorkItems =
      static_cast<int64_t>(std::numeric_limits<int>::max()) * kThreads;
  TORCH_CHECK(
      work_items <= kMaxWorkItems,
      "fast.cu B200 epilogue tensor is too large for the launch grid");
}

template <bool WideStore>
torch::Tensor launch_epilogue(
    const torch::Tensor& accumulators,
    torch::Tensor output) {
  check_tensor_contract(accumulators, output);
  const c10::cuda::CUDAGuard guard(accumulators.device());
  const int64_t work_items = accumulators.numel() / kElementsPerThread;
  const int blocks = static_cast<int>((work_items + kThreads - 1) / kThreads);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(accumulators.get_device());
  epilogue_kernel<WideStore><<<blocks, kThreads, 0, stream>>>(
      accumulators.data_ptr<float>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      work_items);
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

void require_exact_sm100() {
  int device = -1;
  AT_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  AT_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  TORCH_CHECK(
      properties.major == 10 && properties.minor == 0,
      "fast.cu B200 epilogue requires exact SM100 (compute capability 10.0). Found ",
      properties.major,
      ".",
      properties.minor);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "baseline",
      &launch_epilogue<false>,
      "FP32-to-FP16 epilogue using two 128-bit stores");
  module.def(
      "optimized",
      &launch_epilogue<true>,
      "FP32-to-FP16 epilogue using one 256-bit store");
  module.def(
      "require_exact_sm100",
      &require_exact_sm100,
      "Reject every CUDA architecture except exact SM100");
}
