#pragma once

#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cstddef>

namespace cg = cooperative_groups;

namespace ch11 {

constexpr int kBaselineWarpSize = 32;
constexpr int kBaselineWarpsPerBlock = 3;
constexpr int kBaselineThreadsPerBlock = kBaselineWarpsPerBlock * kBaselineWarpSize;
constexpr int kBaselinePipelineStages = 2;
constexpr int kBaselineTileSize = 1024;
constexpr int kBaselineTileElems = kBaselineTileSize;
constexpr size_t kBaselineSharedBytes =
    3 * kBaselinePipelineStages * static_cast<size_t>(kBaselineTileSize) * sizeof(float);

__global__ void baseline_warp_specialized_two_pipelines_kernel(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float* __restrict__ C_global,
    int num_tiles) {
  cg::thread_block cta = cg::this_thread_block();

  extern __shared__ float shared_mem[];
  float* A_stage = shared_mem;
  float* B_stage = A_stage + kBaselinePipelineStages * kBaselineTileElems;
  float* C_stage = B_stage + kBaselinePipelineStages * kBaselineTileElems;

  using pipe_state = cuda::pipeline_shared_state<cuda::thread_scope_block, kBaselinePipelineStages>;
#if defined(__CUDACC__)
// CUDA 13 diagnoses these standard CCCL shared-state declarations as dynamic
// initialization. make_pipeline initializes every barrier before either state
// is used, so suppress only that front-end diagnostic around the two objects.
#pragma nv_diag_suppress 20054
#endif
  __shared__ pipe_state state_lc;
  __shared__ pipe_state state_cs;
#if defined(__CUDACC__)
#pragma nv_diag_default 20054
#endif
  auto pipe_lc = cuda::make_pipeline(cta, &state_lc);
  auto pipe_cs = cuda::make_pipeline(cta, &state_cs);

  const int warp_id = threadIdx.x / kBaselineWarpSize;
  const int lane_id = threadIdx.x % kBaselineWarpSize;
  const int stride = gridDim.x;

  auto stage_ptr = [&](float* base, int stage) -> float* {
    return base + stage * kBaselineTileElems;
  };

  const int primed_tiles = min(kBaselinePipelineStages, num_tiles);
  for (int stage = 0; stage < primed_tiles; ++stage) {
    int tile = blockIdx.x + stage * stride;
    if (tile >= num_tiles) break;
    float* A_buf = stage_ptr(A_stage, stage);
    float* B_buf = stage_ptr(B_stage, stage);
    size_t base = static_cast<size_t>(tile) * kBaselineTileElems;

    pipe_lc.producer_acquire();
    if (warp_id == 0) {
#pragma unroll
      for (int idx = lane_id; idx < kBaselineTileElems; idx += kBaselineWarpSize) {
        A_buf[idx] = A_global[base + idx];
        B_buf[idx] = B_global[base + idx];
      }
    }
    pipe_lc.producer_commit();
  }

  int iteration = 0;
  for (int tile = blockIdx.x; tile < num_tiles; tile += stride, ++iteration) {
    int stage = iteration % kBaselinePipelineStages;
    float* A_buf = stage_ptr(A_stage, stage);
    float* B_buf = stage_ptr(B_stage, stage);
    float* C_buf = stage_ptr(C_stage, stage);
    const size_t base = static_cast<size_t>(tile) * kBaselineTileElems;

    pipe_lc.consumer_wait();
    if (warp_id == 1) {
#pragma unroll
      for (int chunk = 0; chunk < kBaselineTileSize; chunk += kBaselineWarpSize) {
        C_buf[chunk + lane_id] = A_buf[chunk + lane_id] + B_buf[chunk + lane_id];
      }
    }
    pipe_lc.consumer_release();

    // Only reuse a stage buffer after the consumer releases it; otherwise we can
    // overwrite live data and deadlock the pipeline.
    int next_tile = tile + kBaselinePipelineStages * stride;
    if (next_tile < num_tiles) {
      float* A_next = stage_ptr(A_stage, stage);
      float* B_next = stage_ptr(B_stage, stage);
      size_t next_base = static_cast<size_t>(next_tile) * kBaselineTileElems;
      pipe_lc.producer_acquire();
      if (warp_id == 0) {
#pragma unroll
        for (int idx = lane_id; idx < kBaselineTileElems; idx += kBaselineWarpSize) {
          A_next[idx] = A_global[next_base + idx];
          B_next[idx] = B_global[next_base + idx];
        }
      }
      pipe_lc.producer_commit();
    }

    pipe_cs.producer_acquire();
    // compute warp already filled C_buf; no additional work needed.
    pipe_cs.producer_commit();

    pipe_cs.consumer_wait();
    if (warp_id == 2) {
#pragma unroll
      for (int chunk = 0; chunk < kBaselineTileSize; chunk += kBaselineWarpSize) {
        C_global[base + chunk + lane_id] = C_buf[chunk + lane_id];
      }
    }
    pipe_cs.consumer_release();

    cta.sync();
  }
}

inline void launch_baseline_warp_specialized_two_pipelines(
    const float* A_global,
    const float* B_global,
    float* C_global,
    int num_tiles,
    cudaStream_t stream) {
  if (num_tiles <= 0) {
    return;
  }
  const int grid_dim = std::max(1, std::min(num_tiles, 64));
  const dim3 block(kBaselineThreadsPerBlock);
  const dim3 grid(grid_dim);
  baseline_warp_specialized_two_pipelines_kernel<<<grid,
                                                   block,
                                                   kBaselineSharedBytes,
                                                   stream>>>(
      A_global, B_global, C_global, num_tiles);
}

}  // namespace ch11
