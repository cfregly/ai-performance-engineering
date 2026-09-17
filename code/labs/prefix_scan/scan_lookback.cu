// Single-pass inclusive scan with decoupled look-back tile state.
// Dynamic tile allocation prevents a resident CTA from waiting for a tile that
// has not yet been assigned to a running worker. Acquire/release flags publish
// aggregate/prefix data independently from the local block scan.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_scan.cuh>
#include <cuda/atomic>
#include <algorithm>
#include <climits>

constexpr int kBlock = 256;

__global__ void lookback_scan(const unsigned int* input, unsigned int* output,
                             int64_t n, unsigned int* next_tile,
                             unsigned int* aggregates, unsigned int* prefixes,
                             int* states) {
  using Scan = cub::BlockScan<unsigned int, kBlock>;
  __shared__ Scan::TempStorage temp;
  __shared__ unsigned int tile;
  __shared__ unsigned int carry;
  const unsigned int tiles = (n + kBlock - 1) / kBlock;
  while (true) {
    if (threadIdx.x == 0) tile = atomicAdd(next_tile, 1u);
    __syncthreads();
    if (tile >= tiles) return;
    const int64_t index = static_cast<int64_t>(tile) * kBlock + threadIdx.x;
    const unsigned int value = index < n ? input[index] : 0u;
    unsigned int local_prefix, aggregate;
    Scan(temp).InclusiveSum(value, local_prefix, aggregate);
    if (threadIdx.x == 0) {
      aggregates[tile] = aggregate;
      cuda::atomic_ref<int, cuda::thread_scope_device> status(states[tile]);
      status.store(1, cuda::memory_order_release);
      unsigned int before = 0;
      for (int previous = static_cast<int>(tile) - 1; previous >= 0; --previous) {
        cuda::atomic_ref<int, cuda::thread_scope_device> predecessor(states[previous]);
        int state;
        while ((state = predecessor.load(cuda::memory_order_acquire)) == 0) {
          // No block-grid barrier: only already-assigned predecessor tiles.
        }
        if (state == 2) {
          before += prefixes[previous];
          break;
        }
        before += aggregates[previous];
      }
      carry = before;
      prefixes[tile] = before + aggregate;
      status.store(2, cuda::memory_order_release);
    }
    __syncthreads();
    if (index < n) output[index] = local_prefix + carry;
    __syncthreads();
  }
}

void scan(torch::Tensor input, torch::Tensor output, torch::Tensor next_tile,
          torch::Tensor aggregates, torch::Tensor prefixes, torch::Tensor states) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 1 && input.numel() > 0 && input.numel() <= INT_MAX,
              "scan requires a nonempty CUDA vector with at most INT_MAX elements");
  const int64_t tiles = (input.numel() + kBlock - 1) / kBlock;
  for (const auto& tensor : {input, output, next_tile, aggregates, prefixes, states}) {
    TORCH_CHECK(tensor.device() == input.device() && tensor.scalar_type() == torch::kInt32 && tensor.is_contiguous(),
                "all scan buffers must be contiguous same-device int32 tensors");
  }
  TORCH_CHECK(output.numel() == input.numel() && next_tile.numel() == 1 &&
              aggregates.numel() == tiles && prefixes.numel() == tiles && states.numel() == tiles,
              "invalid scan scratch sizes");
  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const int workers = std::min<int64_t>(tiles, at::cuda::getDeviceProperties(input.get_device())->multiProcessorCount);
  lookback_scan<<<workers, kBlock, 0, stream.stream()>>>(
      reinterpret_cast<unsigned int*>(input.data_ptr<int>()), reinterpret_cast<unsigned int*>(output.data_ptr<int>()),
      input.numel(), reinterpret_cast<unsigned int*>(next_tile.data_ptr<int>()),
      reinterpret_cast<unsigned int*>(aggregates.data_ptr<int>()),
      reinterpret_cast<unsigned int*>(prefixes.data_ptr<int>()), states.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scan", &scan); }
