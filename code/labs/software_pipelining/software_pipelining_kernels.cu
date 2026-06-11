#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <new>
#include <stdexcept>
#include <string>

namespace cg = cooperative_groups;

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize;
constexpr int kElemsPerThread = 4;
constexpr int kTileElems = kBlockThreads * kElemsPerThread;
constexpr int kPipelineStages = 2;
constexpr int kCopyWarpsPerTensor = kWarpsPerBlock / 2;
constexpr int kChunkElems = kTileElems / kCopyWarpsPerTensor;

__device__ inline float tile_math(float x, float y, int repeat_fmas) {
  for (int iter = 0; iter < repeat_fmas; ++iter) {
    x = fmaf(x, 1.0001f, y * 0.0003f);
    y = fmaf(y, 0.9997f, x * 0.0002f);
  }
  return x + 0.5f * y;
}

__global__ void baseline_tile_pipeline_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    int numel,
    int repeat_fmas) {
  cg::thread_block cta = cg::this_thread_block();
  auto warp = cg::tiled_partition<kWarpSize>(cta);
  __shared__ float lhs_tile[kTileElems];
  __shared__ float rhs_tile[kTileElems];

  const int num_tiles = (numel + kTileElems - 1) / kTileElems;
  const int stride = gridDim.x;

  for (int tile = blockIdx.x; tile < num_tiles; tile += stride) {
    const int tile_base = tile * kTileElems;

    if (warp.meta_group_rank() == 0) {
      for (int local = warp.thread_rank(); local < kTileElems; local += warp.size()) {
        const int idx = tile_base + local;
        lhs_tile[local] = idx < numel ? lhs[idx] : 0.0f;
      }
      for (int local = warp.thread_rank(); local < kTileElems; local += warp.size()) {
        const int idx = tile_base + local;
        rhs_tile[local] = idx < numel ? rhs[idx] : 0.0f;
      }
    }
    cta.sync();

    for (int local = threadIdx.x; local < kTileElems; local += blockDim.x) {
      const int idx = tile_base + local;
      if (idx < numel) {
        out[idx] = tile_math(lhs_tile[local], rhs_tile[local], repeat_fmas);
      }
    }
    cta.sync();
  }
}

__global__ void optimized_tile_pipeline_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    int numel,
    int repeat_fmas) {
  cg::thread_block cta = cg::this_thread_block();

  __shared__ alignas(16) float lhs_tile[kPipelineStages][kTileElems];
  __shared__ alignas(16) float rhs_tile[kPipelineStages][kTileElems];
  using pipe_state_t = cuda::pipeline_shared_state<cuda::thread_scope_block, kPipelineStages>;
  __shared__ alignas(pipe_state_t) unsigned char pipe_state_bytes[sizeof(pipe_state_t)];
  auto* pipe_state = reinterpret_cast<pipe_state_t*>(pipe_state_bytes);
  if (threadIdx.x == 0) {
    new (pipe_state) pipe_state_t();
  }
  cta.sync();
  auto pipe = cuda::make_pipeline(cta, pipe_state);

  const int num_tiles = (numel + kTileElems - 1) / kTileElems;
  const int stride = gridDim.x;
  auto stage_tile = [&](int stage, int tile) {
    const int tile_base = tile * kTileElems;
    const bool full_tile = tile_base + kTileElems <= numel;

    pipe.producer_acquire();
    if (full_tile) {
      auto warp = cg::tiled_partition<kWarpSize>(cta);
      const int warp_id = warp.meta_group_rank();
      // 16B-aligned static size lets memcpy_async lower to 128-bit cp.async
      // without a runtime alignment probe.
      constexpr auto kChunkBytes =
          cuda::aligned_size_t<16>(static_cast<size_t>(kChunkElems) * sizeof(float));
      if (warp_id < kCopyWarpsPerTensor) {
        const int chunk_base = warp_id * kChunkElems;
        cuda::memcpy_async(
            warp,
            lhs_tile[stage] + chunk_base,
            lhs + tile_base + chunk_base,
            kChunkBytes,
            pipe);
      } else {
        const int chunk_base = (warp_id - kCopyWarpsPerTensor) * kChunkElems;
        cuda::memcpy_async(
            warp,
            rhs_tile[stage] + chunk_base,
            rhs + tile_base + chunk_base,
            kChunkBytes,
            pipe);
      }
    } else {
      for (int local = threadIdx.x; local < kTileElems; local += blockDim.x) {
        const int idx = tile_base + local;
        lhs_tile[stage][local] = idx < numel ? lhs[idx] : 0.0f;
        rhs_tile[stage][local] = idx < numel ? rhs[idx] : 0.0f;
      }
      cta.sync();
    }
    pipe.producer_commit();
  };

  // Prime all pipeline stages before starting steady-state consumption.
  for (int stage = 0; stage < kPipelineStages; ++stage) {
    const int tile = blockIdx.x + stage * stride;
    if (tile >= num_tiles) {
      break;
    }
    stage_tile(stage, tile);
  }

  cta.sync();

  int iteration = 0;
  for (int tile = blockIdx.x; tile < num_tiles; tile += stride, ++iteration) {
    const int stage = iteration % kPipelineStages;
    const int tile_base = tile * kTileElems;
    const bool full_tile = tile_base + kTileElems <= numel;

    // consumer_wait returns once the stage's async copies (committed by every
    // thread in the block) are complete, so no extra block barrier is needed
    // before reading the staged tile.
    pipe.consumer_wait();

    if (full_tile) {
      // 128-bit vector path: 4 elements per thread per pass cuts the issued
      // instruction count ~4x vs the scalar loop (the kernel is issue-bound,
      // not DRAM-bound, at 256-thread blocks).
      const float4* lhs_vec = reinterpret_cast<const float4*>(lhs_tile[stage]);
      const float4* rhs_vec = reinterpret_cast<const float4*>(rhs_tile[stage]);
      float4* out_vec = reinterpret_cast<float4*>(out + tile_base);
      constexpr int kVecPerTile = kTileElems / 4;
      for (int v = threadIdx.x; v < kVecPerTile; v += blockDim.x) {
        const float4 a = lhs_vec[v];
        const float4 b = rhs_vec[v];
        float4 r;
        r.x = tile_math(a.x, b.x, repeat_fmas);
        r.y = tile_math(a.y, b.y, repeat_fmas);
        r.z = tile_math(a.z, b.z, repeat_fmas);
        r.w = tile_math(a.w, b.w, repeat_fmas);
        out_vec[v] = r;
      }
    } else {
      for (int local = threadIdx.x; local < kTileElems; local += blockDim.x) {
        const int idx = tile_base + local;
        if (idx < numel) {
          out[idx] = tile_math(lhs_tile[stage][local], rhs_tile[stage][local], repeat_fmas);
        }
      }
    }

    // Per-thread release: producer_acquire on this stage blocks until every
    // thread has released, so the explicit cta.sync()s around release and the
    // refill are redundant and only serialize the pipeline.
    pipe.consumer_release();

    const int next_tile = tile + kPipelineStages * stride;
    if (next_tile < num_tiles) {
      stage_tile(stage, next_tile);
    }
  }
}

torch::Tensor check_inputs(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    int64_t repeat_fmas,
    const char* label) {
  TORCH_CHECK(lhs.is_cuda(), label, " requires CUDA lhs tensor");
  TORCH_CHECK(rhs.is_cuda(), label, " requires CUDA rhs tensor");
  TORCH_CHECK(lhs.scalar_type() == torch::kFloat32, label, " requires float32 lhs tensor");
  TORCH_CHECK(rhs.scalar_type() == torch::kFloat32, label, " requires float32 rhs tensor");
  TORCH_CHECK(lhs.is_contiguous(), label, " requires contiguous lhs tensor");
  TORCH_CHECK(rhs.is_contiguous(), label, " requires contiguous rhs tensor");
  TORCH_CHECK(lhs.sizes() == rhs.sizes(), label, " requires same-shaped inputs");
  TORCH_CHECK(lhs.dim() == 1, label, " expects 1D inputs");
  TORCH_CHECK(repeat_fmas >= 1, label, " requires repeat_fmas >= 1");
  return torch::empty_like(lhs);
}

torch::Tensor launch_common(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    int64_t repeat_fmas,
    const char* label,
    bool use_pipeline) {
  auto out = check_inputs(lhs, rhs, repeat_fmas, label);

  const int numel = static_cast<int>(lhs.numel());
  if (numel == 0) {
    return out;
  }

  auto* props = at::cuda::getCurrentDeviceProperties();
  const int num_tiles = (numel + kTileElems - 1) / kTileElems;
  const int grid = std::max(1, std::min(num_tiles, props->multiProcessorCount * 8));
  const dim3 block(kBlockThreads);
  const cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  if (use_pipeline) {
    optimized_tile_pipeline_kernel<<<grid, block, 0, stream>>>(
        lhs.data_ptr<float>(),
        rhs.data_ptr<float>(),
        out.data_ptr<float>(),
        numel,
        static_cast<int>(repeat_fmas));
  } else {
    baseline_tile_pipeline_kernel<<<grid, block, 0, stream>>>(
        lhs.data_ptr<float>(),
        rhs.data_ptr<float>(),
        out.data_ptr<float>(),
        numel,
        static_cast<int>(repeat_fmas));
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

torch::Tensor baseline_tile_pipeline(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    int64_t repeat_fmas) {
  return launch_common(lhs, rhs, repeat_fmas, "baseline_tile_pipeline", /*use_pipeline=*/false);
}

torch::Tensor optimized_tile_pipeline(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    int64_t repeat_fmas) {
  return launch_common(lhs, rhs, repeat_fmas, "optimized_tile_pipeline", /*use_pipeline=*/true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("baseline_tile_pipeline", &baseline_tile_pipeline, "Serialized tile loop baseline");
  m.def("optimized_tile_pipeline", &optimized_tile_pipeline, "Two-stage software-pipelined tile loop");
}
