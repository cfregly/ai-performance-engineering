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

// ---------------------------------------------------------------------------
// TMA (cp.async.bulk) warp-specialized pipeline for sm_90+.
//
// Warp 0 lane 0 is the sole producer: it issues one bulk tensor copy per
// operand per tile and signals a tx-count mbarrier. Warps 1..N-1 are
// consumers: they spin on the stage's "full" barrier parity, compute with
// 128-bit vectors, then arrive on the stage's "empty" barrier so the producer
// can reuse the stage. This removes the per-thread cp.async + pipeline
// barrier instruction overhead that capped the cuda::pipeline version at
// ~76% DRAM SoL (issue-slots-bound, not DRAM-bound).
// ---------------------------------------------------------------------------
constexpr int kTmaStages = 2;
constexpr int kTmaTileElems = 2048;  // floats per operand per stage
constexpr int kTmaProducerThreads = kWarpSize;  // warp 0
constexpr int kTmaVecPerConsumer = 1;  // float4 pairs per consumer thread per tile
// 1 producer warp + consumers that exactly cover the tile (no ragged loop).
constexpr int kTmaConsumerThreads = kTmaTileElems / 4 / kTmaVecPerConsumer;
constexpr int kTmaBlockThreads = kTmaProducerThreads + kTmaConsumerThreads;
static_assert(kTmaConsumerThreads * 4 * kTmaVecPerConsumer == kTmaTileElems);
static_assert(kTmaBlockThreads <= 1024 && kTmaConsumerThreads % kWarpSize == 0);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t tx_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(tx_bytes)
               : "memory");
}

__device__ __forceinline__ void mbar_wait_parity(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE;\n"
      "bra LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void tma_bulk_load(
    void* dst_smem,
    const void* src_gmem,
    uint32_t bytes,
    uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
          "r"(smem_u32(dst_smem)),
      "l"(src_gmem),
      "r"(bytes),
      "r"(smem_u32(bar))
      : "memory");
}

#endif  // __CUDA_ARCH__ >= 900

__global__ void optimized_tile_pipeline_tma_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    int numel,
    int repeat_fmas) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ __align__(128) unsigned char smem_raw[];
  float* lhs_tiles = reinterpret_cast<float*>(smem_raw);
  float* rhs_tiles = lhs_tiles + kTmaStages * kTmaTileElems;
  __shared__ uint64_t full_bar[kTmaStages];
  __shared__ uint64_t empty_bar[kTmaStages];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kTmaStages; ++s) {
      mbar_init(&full_bar[s], 1);  // producer's arrive.expect_tx only
      mbar_init(&empty_bar[s], kTmaConsumerThreads);
    }
  }
  __syncthreads();

  const int num_tiles = (numel + kTmaTileElems - 1) / kTmaTileElems;
  const int stride = gridDim.x;
  constexpr uint32_t kTileBytes = kTmaTileElems * sizeof(float);

  if (threadIdx.x < kTmaProducerThreads) {
    if (threadIdx.x == 0) {
      // Producer: keep up to kTmaStages tiles in flight.
      int j = 0;
      for (int tile = blockIdx.x; tile < num_tiles; tile += stride, ++j) {
        const int s = j % kTmaStages;
        const int k = j / kTmaStages;
        if (k > 0) {
          mbar_wait_parity(&empty_bar[s], (k - 1) & 1);
        }
        const int tile_base = tile * kTmaTileElems;
        if (tile_base + kTmaTileElems <= numel) {
          tma_bulk_load(lhs_tiles + s * kTmaTileElems, lhs + tile_base, kTileBytes, &full_bar[s]);
          tma_bulk_load(rhs_tiles + s * kTmaTileElems, rhs + tile_base, kTileBytes, &full_bar[s]);
          mbar_arrive_expect_tx(&full_bar[s], 2 * kTileBytes);
        } else {
          // Partial tail tile: consumers read global directly.
          mbar_arrive_expect_tx(&full_bar[s], 0);
        }
      }
    }
    return;  // whole producer warp exits the consumer path
  }

  // Consumers: warps 1..N-1.
  const int ctid = threadIdx.x - kTmaProducerThreads;
  int j = 0;
  for (int tile = blockIdx.x; tile < num_tiles; tile += stride, ++j) {
    const int s = j % kTmaStages;
    const int k = j / kTmaStages;
    const int tile_base = tile * kTmaTileElems;

    mbar_wait_parity(&full_bar[s], k & 1);

    if (tile_base + kTmaTileElems <= numel) {
      const float4* lhs_vec = reinterpret_cast<const float4*>(lhs_tiles + s * kTmaTileElems);
      const float4* rhs_vec = reinterpret_cast<const float4*>(rhs_tiles + s * kTmaTileElems);
      float4* out_vec = reinterpret_cast<float4*>(out + tile_base);
      constexpr int kVecPerTile = kTmaTileElems / 4;
      constexpr int kMaxVecPerThread =
          (kVecPerTile + kTmaConsumerThreads - 1) / kTmaConsumerThreads;
      // Stage-early-release: drain the staged tile into registers, free the
      // stage for the producer immediately, then compute/store while the next
      // bulk copy is already in flight. This decouples the refill from the
      // global-store drain.
      float4 a[kMaxVecPerThread];
      float4 b[kMaxVecPerThread];
      int cnt = 0;
#pragma unroll
      for (int v = ctid; v < kVecPerTile; v += kTmaConsumerThreads, ++cnt) {
        a[cnt] = lhs_vec[v];
        b[cnt] = rhs_vec[v];
      }
      mbar_arrive(&empty_bar[s]);
      cnt = 0;
#pragma unroll
      for (int v = ctid; v < kVecPerTile; v += kTmaConsumerThreads, ++cnt) {
        float4 r;
        r.x = tile_math(a[cnt].x, b[cnt].x, repeat_fmas);
        r.y = tile_math(a[cnt].y, b[cnt].y, repeat_fmas);
        r.z = tile_math(a[cnt].z, b[cnt].z, repeat_fmas);
        r.w = tile_math(a[cnt].w, b[cnt].w, repeat_fmas);
        out_vec[v] = r;
      }
    } else {
      mbar_arrive(&empty_bar[s]);  // tail tile bypasses the staged buffers
      for (int local = ctid; local < kTmaTileElems; local += kTmaConsumerThreads) {
        const int idx = tile_base + local;
        if (idx < numel) {
          out[idx] = tile_math(lhs[idx], rhs[idx], repeat_fmas);
        }
      }
    }
  }
#endif  // __CUDA_ARCH__ >= 900
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
  // Use the CURRENT stream, not getDefaultCUDAStream(): an explicit-but-default
  // stream is invisible to side-stream CUDA graph capture exactly like a raw
  // <<<>>> legacy launch (B45/B62 silent-drop class; caught by the graphed-mode
  // capture-fidelity audit). Eager behavior is identical (current == default
  // outside capture).
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (use_pipeline && props->major >= 9) {
    // TMA bulk-copy pipeline (sm_90+): dynamic smem above the 48KB static cap.
    constexpr int kTmaSmemBytes = kTmaStages * kTmaTileElems * 2 * sizeof(float);
    static int blocks_per_sm = [] {
      cudaFuncSetAttribute(
          optimized_tile_pipeline_tma_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          kTmaSmemBytes);
      int blocks = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, optimized_tile_pipeline_tma_kernel, kTmaBlockThreads, kTmaSmemBytes);
      return std::max(1, blocks);
    }();
    const int tma_tiles = (numel + kTmaTileElems - 1) / kTmaTileElems;
    const int tma_grid =
        std::max(1, std::min(tma_tiles, props->multiProcessorCount * blocks_per_sm));
    const dim3 tma_block(kTmaBlockThreads);
    optimized_tile_pipeline_tma_kernel<<<tma_grid, tma_block, kTmaSmemBytes, stream>>>(
        lhs.data_ptr<float>(),
        rhs.data_ptr<float>(),
        out.data_ptr<float>(),
        numel,
        static_cast<int>(repeat_fmas));
  } else if (use_pipeline) {
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
