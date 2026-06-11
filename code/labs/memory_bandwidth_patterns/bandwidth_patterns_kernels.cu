// CUDA kernels for the memory-bandwidth-patterns lab.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace cg = cooperative_groups;

#define CHECK_CUDA(x)                                                                    \
  do {                                                                                   \
    cudaError_t status__ = (x);                                                          \
    TORCH_CHECK(status__ == cudaSuccess, "CUDA error: ", cudaGetErrorString(status__));  \
  } while (0)

namespace {

constexpr int kCopyThreads = 256;
constexpr int kTransposeTileDim = 32;
constexpr int kTransposeBlockRows = 8;
constexpr int kAsyncTileElems = 4096;

inline int ceil_div_int(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

__global__ void copy_scalar_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int idx = tid; idx < n; idx += stride) {
    dst[idx] = src[idx];
  }
}

__global__ void copy_vectorized_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    int vec_count) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int idx = tid; idx < vec_count; idx += stride) {
    dst[idx] = src[idx];
  }
}

__global__ void copy_tail_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int start,
    int n) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int idx = start + tid; idx < n; idx += stride) {
    dst[idx] = src[idx];
  }
}

__global__ void copy_async_double_buffered_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n) {
  cg::thread_block block = cg::this_thread_block();
  __shared__ alignas(16) float stages[2][kAsyncTileElems];

  const int tile_stride = gridDim.x * kAsyncTileElems;
  int base = blockIdx.x * kAsyncTileElems;
  if (base >= n) {
    return;
  }

  int stage = 0;
  int count = min(kAsyncTileElems, n - base);
  cg::memcpy_async(block, stages[stage], src + base, count * static_cast<int>(sizeof(float)));

  while (true) {
    const int next_base = base + tile_stride;
    const int next_stage = stage ^ 1;
    int next_count = 0;
    const bool has_next = next_base < n;
    if (has_next) {
      next_count = min(kAsyncTileElems, n - next_base);
      cg::memcpy_async(
          block,
          stages[next_stage],
          src + next_base,
          next_count * static_cast<int>(sizeof(float)));
      cg::wait_prior<1>(block);
    } else {
      cg::wait(block);
    }

    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
      dst[base + idx] = stages[stage][idx];
    }
    block.sync();

    if (!has_next) {
      break;
    }
    base = next_base;
    count = next_count;
    stage = next_stage;
  }
}

__global__ void transpose_naive_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int rows,
    int cols) {
  const int x = blockIdx.x * kTransposeTileDim + threadIdx.x;
  const int y = blockIdx.y * kTransposeTileDim + threadIdx.y;

  #pragma unroll
  for (int j = 0; j < kTransposeTileDim; j += kTransposeBlockRows) {
    const int row = y + j;
    if (x < cols && row < rows) {
      dst[x * rows + row] = src[row * cols + x];
    }
  }
}

__global__ void transpose_tiled_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int rows,
    int cols) {
  __shared__ float tile[kTransposeTileDim][kTransposeTileDim + 1];

  int x = blockIdx.x * kTransposeTileDim + threadIdx.x;
  int y = blockIdx.y * kTransposeTileDim + threadIdx.y;

  #pragma unroll
  for (int j = 0; j < kTransposeTileDim; j += kTransposeBlockRows) {
    const int row = y + j;
    if (x < cols && row < rows) {
      tile[threadIdx.y + j][threadIdx.x] = src[row * cols + x];
    }
  }

  __syncthreads();

  x = blockIdx.y * kTransposeTileDim + threadIdx.x;
  y = blockIdx.x * kTransposeTileDim + threadIdx.y;

  #pragma unroll
  for (int j = 0; j < kTransposeTileDim; j += kTransposeBlockRows) {
    const int row = y + j;
    if (x < rows && row < cols) {
      dst[row * rows + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

void check_copy_tensors(torch::Tensor src, torch::Tensor dst) {
  TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src and dst must be CUDA tensors");
  TORCH_CHECK(src.dtype() == torch::kFloat32, "src must be float32");
  TORCH_CHECK(dst.dtype() == torch::kFloat32, "dst must be float32");
  TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "src and dst must be contiguous");
  TORCH_CHECK(src.dim() == 1 && dst.dim() == 1, "copy kernels expect 1D tensors");
  TORCH_CHECK(src.numel() == dst.numel(), "src and dst size mismatch");
}

void check_transpose_tensors(torch::Tensor src, torch::Tensor dst) {
  TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src and dst must be CUDA tensors");
  TORCH_CHECK(src.dtype() == torch::kFloat32, "src must be float32");
  TORCH_CHECK(dst.dtype() == torch::kFloat32, "dst must be float32");
  TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "src and dst must be contiguous");
  TORCH_CHECK(src.dim() == 2 && dst.dim() == 2, "transpose kernels expect 2D tensors");
  TORCH_CHECK(dst.size(0) == src.size(1) && dst.size(1) == src.size(0), "dst must be transposed shape");
}

}  // namespace

void copy_scalar(torch::Tensor src, torch::Tensor dst) {
  check_copy_tensors(src, dst);
  c10::cuda::CUDAGuard guard(src.device());
  // Use the CURRENT stream, not getDefaultCUDAStream() (here and in the four
  // launchers below): an explicit-but-default stream is invisible to
  // side-stream CUDA graph capture exactly like a raw <<<>>> legacy launch
  // (B45/B62/B65 silent-drop class). Eager behavior is identical (current ==
  // default outside capture).
  auto stream = at::cuda::getCurrentCUDAStream();
  const int n = static_cast<int>(src.numel());
  const int blocks = std::max(1, std::min(4096, ceil_div_int(n, kCopyThreads)));
  copy_scalar_kernel<<<blocks, kCopyThreads, 0, stream>>>(
      src.data_ptr<float>(),
      dst.data_ptr<float>(),
      n);
  CHECK_CUDA(cudaGetLastError());
}

void copy_vectorized(torch::Tensor src, torch::Tensor dst) {
  check_copy_tensors(src, dst);
  const auto src_ptr = reinterpret_cast<std::uintptr_t>(src.data_ptr<float>());
  const auto dst_ptr = reinterpret_cast<std::uintptr_t>(dst.data_ptr<float>());
  TORCH_CHECK(src_ptr % alignof(float4) == 0, "src pointer must be 16-byte aligned");
  TORCH_CHECK(dst_ptr % alignof(float4) == 0, "dst pointer must be 16-byte aligned");

  c10::cuda::CUDAGuard guard(src.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int n = static_cast<int>(src.numel());
  const int vec_count = n / 4;
  if (vec_count > 0) {
    const int blocks = std::max(1, std::min(4096, ceil_div_int(vec_count, kCopyThreads)));
    copy_vectorized_kernel<<<blocks, kCopyThreads, 0, stream>>>(
        reinterpret_cast<const float4*>(src.data_ptr<float>()),
        reinterpret_cast<float4*>(dst.data_ptr<float>()),
        vec_count);
    CHECK_CUDA(cudaGetLastError());
  }
  const int tail_start = vec_count * 4;
  if (tail_start < n) {
    const int tail_elems = n - tail_start;
    const int blocks = std::max(1, std::min(4096, ceil_div_int(tail_elems, kCopyThreads)));
    copy_tail_kernel<<<blocks, kCopyThreads, 0, stream>>>(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        tail_start,
        n);
    CHECK_CUDA(cudaGetLastError());
  }
}

void copy_async_double_buffered(torch::Tensor src, torch::Tensor dst) {
  check_copy_tensors(src, dst);
  c10::cuda::CUDAGuard guard(src.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int n = static_cast<int>(src.numel());
  const int blocks = std::max(1, std::min(4096, ceil_div_int(n, kAsyncTileElems)));
  copy_async_double_buffered_kernel<<<blocks, kCopyThreads, 0, stream>>>(
      src.data_ptr<float>(),
      dst.data_ptr<float>(),
      n);
  CHECK_CUDA(cudaGetLastError());
}

void transpose_naive(torch::Tensor src, torch::Tensor dst) {
  check_transpose_tensors(src, dst);
  c10::cuda::CUDAGuard guard(src.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int rows = static_cast<int>(src.size(0));
  const int cols = static_cast<int>(src.size(1));
  const dim3 block(kTransposeTileDim, kTransposeBlockRows);
  const dim3 grid(ceil_div_int(cols, kTransposeTileDim), ceil_div_int(rows, kTransposeTileDim));
  transpose_naive_kernel<<<grid, block, 0, stream>>>(
      src.data_ptr<float>(),
      dst.data_ptr<float>(),
      rows,
      cols);
  CHECK_CUDA(cudaGetLastError());
}

void transpose_tiled(torch::Tensor src, torch::Tensor dst) {
  check_transpose_tensors(src, dst);
  c10::cuda::CUDAGuard guard(src.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int rows = static_cast<int>(src.size(0));
  const int cols = static_cast<int>(src.size(1));
  const dim3 block(kTransposeTileDim, kTransposeBlockRows);
  const dim3 grid(ceil_div_int(cols, kTransposeTileDim), ceil_div_int(rows, kTransposeTileDim));
  transpose_tiled_kernel<<<grid, block, 0, stream>>>(
      src.data_ptr<float>(),
      dst.data_ptr<float>(),
      rows,
      cols);
  CHECK_CUDA(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("copy_scalar", &copy_scalar, "Scalar contiguous copy");
  m.def("copy_vectorized", &copy_vectorized, "Vectorized contiguous copy");
  m.def("copy_async_double_buffered", &copy_async_double_buffered, "cp.async-style double-buffered copy");
  m.def("transpose_naive", &transpose_naive, "Naive transpose with strided writes");
  m.def("transpose_tiled", &transpose_tiled, "Shared-memory tiled transpose");
}
