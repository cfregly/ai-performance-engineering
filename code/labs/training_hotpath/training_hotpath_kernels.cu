// CUDA kernels for the training-hotpath lab.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#define CHECK_CUDA(x)                                                        \
  do {                                                                       \
    cudaError_t status__ = (x);                                              \
    TORCH_CHECK(status__ == cudaSuccess, "CUDA error: ", cudaGetErrorString(status__)); \
  } while (0)

namespace {

// Replicated-accumulator count for the single-launch metric kernels: blocks
// fold into replica (blockIdx % K), cutting the same-address atomic chain
// depth K-fold. The chains are what the pre-ticket __threadfence has to wait
// out, so K directly shortens every block's completion tail (measured ~5us at
// K=1 with a 592-block grid). The last block sums the K replicas into out.
constexpr int kMetricAccumReplicas = 8;

// grid.x indexes segments, grid.y indexes chunks within a segment.
// Each block reduces its chunk and atomically accumulates partial/count into
// out[segment]; out must be zero-initialized by the caller. Chunking restores
// occupancy when num_segments is far below the SM count of the device
// (a 128-segment launch fills ~11% of one wave on GB300 otherwise).
__global__ void segment_abs_mean_kernel(
    const float* flat,
    const int64_t* offsets,
    float* out,
    int64_t num_segments) {
  int segment = blockIdx.x;
  if (segment >= num_segments) {
    return;
  }

  int64_t start = offsets[segment];
  int64_t stop = offsets[segment + 1];
  int64_t length = stop - start;
  if (length <= 0) {
    return;  // out[segment] stays at its zero initialization (matches 0/1).
  }

  int64_t chunk_span = (length + gridDim.y - 1) / gridDim.y;
  int64_t chunk_start = start + static_cast<int64_t>(blockIdx.y) * chunk_span;
  int64_t chunk_stop = chunk_start + chunk_span;
  if (chunk_stop > stop) {
    chunk_stop = stop;
  }
  if (chunk_start >= chunk_stop) {
    return;
  }

  float local_sum = 0.0f;
  for (int64_t idx = chunk_start + threadIdx.x; idx < chunk_stop; idx += blockDim.x) {
    local_sum += fabsf(flat[idx]);
  }

  __shared__ float shared[256];
  shared[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(&out[segment], shared[0] / static_cast<float>(length));
  }
}

// Fused metric reduction: a single pass over preds/targets computes all three
// per-responder sums (pred*pred, target*target, pred*target) that the torch
// path needs three separate mul+sum passes (plus temporaries) for. Layout is
// row-major [num_rows, responders], so consecutive threads cover consecutive
// responder columns and global loads coalesce. Each block strides over rows
// with fp32 register accumulators, then atomically folds its partials into the
// zero-initialized out[3 * responders] (gridDim.x partials per output element).
__global__ void metric_reduction_fused_kernel(
    const float* __restrict__ preds,
    const float* __restrict__ targets,
    float* __restrict__ out,
    int64_t num_rows,
    int64_t responders) {
  for (int64_t col = threadIdx.x; col < responders; col += blockDim.x) {
    float pred_sq = 0.0f;
    float target_sq = 0.0f;
    float covar = 0.0f;
    for (int64_t row = blockIdx.x; row < num_rows; row += gridDim.x) {
      int64_t idx = row * responders + col;
      float pred = preds[idx];
      float target = targets[idx];
      pred_sq = fmaf(pred, pred, pred_sq);
      target_sq = fmaf(target, target, target_sq);
      covar = fmaf(pred, target, covar);
    }
    atomicAdd(&out[col], pred_sq);
    atomicAdd(&out[responders + col], target_sq);
    atomicAdd(&out[2 * responders + col], covar);
  }
}

// Single-launch fused metric reduction (B65-named lever, drops the zeros-fill
// launch of the two-launch B50 shape). Both variants below accumulate the
// three sums in fp32 registers (fmaf), atomically fold into a PERSISTENT
// all-zero replicated accumulator, and let the last block to arrive (atomic
// ticket, threadFenceReduction pattern) fold the replicas into `out` and
// restore the accumulator+ticket to zero -- so `out` can be torch::empty and
// no separate fill kernel is ever launched. Numerics class is unchanged from
// the incumbent: fp32 fmaf accumulation + nondeterministic atomic fold order.
// NOTE: the persistent scratch is stream-serialized state; concurrent calls
// from different streams on one device are not supported (the lab and capture
// paths are single-stream).
// Scalar single-launch variant (the DEFAULT; measured GB300 winner): keeps the
// incumbent kernel's proven load/fold structure EXACTLY (column-per-thread
// coalesced loads, block row-striding, fp32 fmaf accumulators, 3 atomicAdds
// per thread, no barriers in the load path) and only redirects the atomic fold
// to the persistent replicated accumulator, appending the last-block
// ticket/fold/re-zero tail that lets `out` be torch::empty.
__global__ void metric_reduction_fused_single_launch_scalar_kernel(
    const float* __restrict__ preds,
    const float* __restrict__ targets,
    float* __restrict__ accum,          // persistent, all-zero on entry
    unsigned int* __restrict__ ticket,  // persistent, zero on entry
    float* __restrict__ out,
    int64_t num_rows,
    int64_t responders) {
  for (int64_t col = threadIdx.x; col < responders; col += blockDim.x) {
    float pred_sq = 0.0f;
    float target_sq = 0.0f;
    float covar = 0.0f;
    for (int64_t row = blockIdx.x; row < num_rows; row += gridDim.x) {
      int64_t idx = row * responders + col;
      float pred = preds[idx];
      float target = targets[idx];
      pred_sq = fmaf(pred, pred, pred_sq);
      target_sq = fmaf(target, target, target_sq);
      covar = fmaf(pred, target, covar);
    }
    const int64_t out_len = 3 * responders;
    float* replica = accum + (blockIdx.x % kMetricAccumReplicas) * out_len;
    atomicAdd(&replica[col], pred_sq);
    atomicAdd(&replica[responders + col], target_sq);
    atomicAdd(&replica[2 * responders + col], covar);
  }

  __threadfence();
  __syncthreads();
  __shared__ bool is_last;
  if (threadIdx.x == 0) {
    const unsigned int done = atomicAdd(ticket, 1u);
    is_last = (done == gridDim.x - 1);
  }
  __syncthreads();
  if (is_last) {
    __threadfence();
    const int64_t out_len = 3 * responders;
    for (int64_t o = threadIdx.x; o < out_len; o += blockDim.x) {
      float total = 0.0f;
#pragma unroll
      for (int k = 0; k < kMetricAccumReplicas; ++k) {
        total += __ldcg(&accum[k * out_len + o]);
        __stcg(&accum[k * out_len + o], 0.0f);
      }
      out[o] = total;
    }
    if (threadIdx.x == 0) {
      *ticket = 0u;
    }
  }
}

// float4 single-launch variant (AISP_METRIC_REDUCTION_VARIANT=vec4; REFUTED as
// slower than the scalar variant on GB300 at the lab shape -- see the host
// dispatch comment): float4 grid-stride loads over the row-major
// [num_rows, responders] pair; each thread owns 4 consecutive responder
// columns (host enforces blockDim % (responders/4) == 0 so the column slot is
// loop-invariant), with a shared-memory block fold before the global fold.
__global__ void metric_reduction_fused_single_launch_kernel(
    const float4* __restrict__ preds,
    const float4* __restrict__ targets,
    float* __restrict__ accum,          // persistent, all-zero on entry
    unsigned int* __restrict__ ticket,  // persistent, zero on entry
    float* __restrict__ out,
    int64_t total_vec,  // num_rows * (responders / 4)
    int64_t responders) {
  const int rvec = static_cast<int>(responders >> 2);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  float pred_sq[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float target_sq[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float covar[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  auto accumulate = [&](const float4 p, const float4 t) {
    pred_sq[0] = fmaf(p.x, p.x, pred_sq[0]);
    pred_sq[1] = fmaf(p.y, p.y, pred_sq[1]);
    pred_sq[2] = fmaf(p.z, p.z, pred_sq[2]);
    pred_sq[3] = fmaf(p.w, p.w, pred_sq[3]);
    target_sq[0] = fmaf(t.x, t.x, target_sq[0]);
    target_sq[1] = fmaf(t.y, t.y, target_sq[1]);
    target_sq[2] = fmaf(t.z, t.z, target_sq[2]);
    target_sq[3] = fmaf(t.w, t.w, target_sq[3]);
    covar[0] = fmaf(p.x, t.x, covar[0]);
    covar[1] = fmaf(p.y, t.y, covar[1]);
    covar[2] = fmaf(p.z, t.z, covar[2]);
    covar[3] = fmaf(p.w, t.w, covar[3]);
  };

  // 4-deep software pipeline: with a device-filling grid each thread only has
  // ~5 grid-stride trips, so a plain loop leaves almost no loads in flight and
  // the kernel goes latency-bound (ncu: 18% DRAM, vs the incumbent scalar
  // kernel's 21-trip loop at 26%). Batch 4 independent float4 pairs per round
  // (8 x 512B loads in flight per warp) before any FMA consumes them.
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; idx + 3 * stride < total_vec; idx += 4 * stride) {
    const float4 p0 = preds[idx];
    const float4 p1 = preds[idx + stride];
    const float4 p2 = preds[idx + 2 * stride];
    const float4 p3 = preds[idx + 3 * stride];
    const float4 t0 = targets[idx];
    const float4 t1 = targets[idx + stride];
    const float4 t2 = targets[idx + 2 * stride];
    const float4 t3 = targets[idx + 3 * stride];
    accumulate(p0, t0);
    accumulate(p1, t1);
    accumulate(p2, t2);
    accumulate(p3, t3);
  }
  for (; idx < total_vec; idx += stride) {
    accumulate(preds[idx], targets[idx]);
  }

  // Block-level fold: slot stride 13 (coprime with 32 banks) kills write
  // conflicts. Thread t's slot holds [pred_sq[0..3], target_sq[0..3],
  // covar[0..3]] for columns 4*(t % rvec) .. 4*(t % rvec) + 3.
  extern __shared__ float smem[];  // blockDim.x * 13 floats
  float* slot = &smem[threadIdx.x * 13];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    slot[q] = pred_sq[q];
    slot[4 + q] = target_sq[q];
    slot[8 + q] = covar[q];
  }
  __syncthreads();

  const int group = blockDim.x / rvec;  // threads sharing one column slot
  const int64_t out_len = 3 * responders;
  float* replica = accum + (blockIdx.x % kMetricAccumReplicas) * out_len;
  for (int64_t o = threadIdx.x; o < out_len; o += blockDim.x) {
    const int m = static_cast<int>(o / responders);  // 0=pred_sq 1=target_sq 2=covar
    const int c = static_cast<int>(o % responders);
    const int c4 = c >> 2;
    const int q = c & 3;
    float sum = 0.0f;
    for (int s = 0; s < group; ++s) {
      sum += smem[(c4 + rvec * s) * 13 + m * 4 + q];
    }
    atomicAdd(&replica[o], sum);
  }

  // Publish this block's fold, then take a ticket; the last block to arrive
  // sees every other block's accumulator contribution (threadfence-before-
  // ticket on the producer side, canonical threadFenceReduction ordering).
  __threadfence();
  __syncthreads();
  __shared__ bool is_last;
  if (threadIdx.x == 0) {
    const unsigned int done = atomicAdd(ticket, 1u);
    is_last = (done == gridDim.x - 1);
  }
  __syncthreads();
  if (is_last) {
    __threadfence();
    for (int64_t o = threadIdx.x; o < out_len; o += blockDim.x) {
      float total = 0.0f;
#pragma unroll
      for (int k = 0; k < kMetricAccumReplicas; ++k) {
        total += __ldcg(&accum[k * out_len + o]);  // L2 load: atomics landed in L2
        __stcg(&accum[k * out_len + o], 0.0f);     // restore the all-zero invariant
      }
      out[o] = total;
    }
    if (threadIdx.x == 0) {
      *ticket = 0u;  // next launch on this stream sees a clean ticket
    }
  }
}

__global__ void pack_rows_kernel(
    const float* input,
    const int64_t* row_indices,
    float* output,
    int64_t num_rows,
    int64_t num_cols) {
  int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_rows * num_cols;
  if (linear_idx >= total) {
    return;
  }
  int64_t out_row = linear_idx / num_cols;
  int64_t col = linear_idx % num_cols;
  int64_t in_row = row_indices[out_row];
  output[linear_idx] = input[in_row * num_cols + col];
}

__global__ void scatter_rows_kernel(
    const float* packed,
    const int64_t* row_indices,
    float* output,
    int64_t num_rows,
    int64_t num_cols) {
  int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_rows * num_cols;
  if (linear_idx >= total) {
    return;
  }
  int64_t packed_row = linear_idx / num_cols;
  int64_t col = linear_idx % num_cols;
  int64_t out_row = row_indices[packed_row];
  output[out_row * num_cols + col] = packed[linear_idx];
}

}  // namespace

torch::Tensor segment_abs_mean(torch::Tensor flat, torch::Tensor offsets) {
  TORCH_CHECK(flat.is_cuda(), "flat must be a CUDA tensor");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be a CUDA tensor");
  TORCH_CHECK(flat.dtype() == torch::kFloat32, "flat must be float32");
  TORCH_CHECK(offsets.dtype() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(flat.dim() == 1, "flat must be 1D");
  TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D");
  TORCH_CHECK(offsets.size(0) >= 2, "offsets must contain at least two elements");

  auto flat_contig = flat.contiguous();
  auto offsets_contig = offsets.contiguous();
  auto num_segments = offsets_contig.size(0) - 1;
  auto out = torch::zeros({num_segments}, flat.options());

  // Query the SM count once per process; the attribute query costs ~10us per
  // call on some driver paths, which would dominate this microsecond kernel.
  static const int sm_count = [device_index = flat_contig.get_device()] {
    int count = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_index));
    return count;
  }();

  // Aim for ~8 blocks per SM so small segment counts still fill the device.
  int64_t target_blocks = static_cast<int64_t>(sm_count) * 8;
  int64_t chunks = (target_blocks + num_segments - 1) / num_segments;
  chunks = std::max<int64_t>(1, std::min<int64_t>(chunks, 64));

  constexpr int threads = 256;
  dim3 grid(static_cast<unsigned int>(num_segments), static_cast<unsigned int>(chunks));
  segment_abs_mean_kernel<<<grid, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      flat_contig.data_ptr<float>(),
      offsets_contig.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      num_segments);
  CHECK_CUDA(cudaGetLastError());
  return out;
}

namespace {

// Persistent per-device scratch for the single-launch kernel: 3*responders
// fp32 accumulator slots + one trailing 32-bit ticket, allocated and zeroed
// ONCE (outside any capture; the warm setup() call triggers it). The kernel
// restores the all-zero invariant on every launch, so steady-state calls and
// CUDA-graph replays never need a fill kernel.
struct MetricReductionScratch {
  torch::Tensor buf;       // float32 [capacity + 1]; last element is the ticket
  int64_t capacity = 0;    // accumulator slots (3 * responders)
};

MetricReductionScratch& metric_reduction_scratch(const torch::Tensor& like, int64_t needed) {
  static std::unordered_map<int, MetricReductionScratch> cache;  // keyed by device index
  auto& scratch = cache[like.get_device()];
  if (scratch.capacity < needed) {
    scratch.buf = torch::zeros({needed + 1}, like.options().dtype(torch::kFloat32));
    scratch.capacity = needed;
  }
  return scratch;
}

}  // namespace

torch::Tensor metric_reduction_fused(torch::Tensor preds, torch::Tensor targets) {
  TORCH_CHECK(preds.is_cuda(), "preds must be a CUDA tensor");
  TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
  TORCH_CHECK(preds.dtype() == torch::kFloat32, "preds must be float32");
  TORCH_CHECK(targets.dtype() == torch::kFloat32, "targets must be float32");
  TORCH_CHECK(preds.sizes() == targets.sizes(), "preds and targets must have matching shapes");
  TORCH_CHECK(preds.dim() >= 1, "preds must have at least one dimension");

  auto preds_contig = preds.contiguous();
  auto targets_contig = targets.contiguous();
  int64_t responders = preds_contig.size(-1);
  TORCH_CHECK(responders > 0, "responders dimension must be positive");
  int64_t num_rows = preds_contig.numel() / responders;

  static const int sm_count = [device_index = preds_contig.get_device()] {
    int count = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_index));
    return count;
  }();

  constexpr int threads = 256;

  // Kill switch: AISP_METRIC_REDUCTION_SINGLE_LAUNCH=0 restores the B50/B65
  // two-launch incumbent (zeros fill + scalar-load fused kernel) exactly.
  static const bool single_launch_enabled = [] {
    const char* env = std::getenv("AISP_METRIC_REDUCTION_SINGLE_LAUNCH");
    return env == nullptr || env[0] != '0';
  }();

  const int64_t rvec = responders / 4;
  const bool vec_ok =
      single_launch_enabled && num_rows > 0 && responders % 4 == 0 && rvec > 0 &&
      threads % rvec == 0 &&
      reinterpret_cast<uintptr_t>(preds_contig.data_ptr<float>()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(targets_contig.data_ptr<float>()) % 16 == 0;

  if (vec_ok) {
    // Single-launch path: out needs no fill (the kernel's last block writes
    // every element), so this is exactly one kernel per call.
    auto out = torch::empty({3 * responders}, preds.options());
    auto& scratch = metric_reduction_scratch(preds_contig, kMetricAccumReplicas * 3 * responders);
    float* accum = scratch.buf.data_ptr<float>();
    auto* ticket = reinterpret_cast<unsigned int*>(accum + scratch.capacity);

    // Grid-size sweep knob (B48-style, env-gated). Default 2 blocks/SM: the
    // measured GB300 optimum for the scalar single-launch kernel (8.26us vs
    // 8.33us at 4/SM and 12.2us at 1/SM, K=8) -- fewer blocks means shorter
    // same-address atomic chains for the pre-ticket fence to wait out.
    static const int blocks_per_sm = [] {
      const char* env = std::getenv("AISP_METRIC_REDUCTION_BLOCKS_PER_SM");
      const int parsed = env ? std::atoi(env) : 0;
      return parsed > 0 ? parsed : 2;
    }();
    // Variant knob: default "scalar" (incumbent load structure + ticket tail,
    // the measured GB300 winner); AISP_METRIC_REDUCTION_VARIANT=vec4 selects
    // the float4 + smem-fold variant (REFUTED as slower here: the kernel is
    // latency-bound, and 4x fewer grid-stride trips per thread starves the
    // load pipeline; kept for re-evaluation on other parts/shapes).
    static const bool use_scalar_variant = [] {
      const char* env = std::getenv("AISP_METRIC_REDUCTION_VARIANT");
      return env == nullptr || env[0] != 'v';
    }();

    if (use_scalar_variant) {
      int64_t blocks = std::min<int64_t>(num_rows, static_cast<int64_t>(sm_count) * blocks_per_sm);
      blocks = std::max<int64_t>(1, blocks);
      metric_reduction_fused_single_launch_scalar_kernel<<<
          static_cast<unsigned int>(blocks), threads, 0,
          c10::cuda::getCurrentCUDAStream()>>>(
          preds_contig.data_ptr<float>(),
          targets_contig.data_ptr<float>(),
          accum,
          ticket,
          out.data_ptr<float>(),
          num_rows,
          responders);
      CHECK_CUDA(cudaGetLastError());
      return out;
    }

    const int64_t total_vec = num_rows * rvec;
    int64_t blocks = std::min<int64_t>((total_vec + threads - 1) / threads,
                                       static_cast<int64_t>(sm_count) * blocks_per_sm);
    blocks = std::max<int64_t>(1, blocks);
    const size_t smem_bytes = static_cast<size_t>(threads) * 13 * sizeof(float);

    metric_reduction_fused_single_launch_kernel<<<
        static_cast<unsigned int>(blocks), threads, smem_bytes,
        c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float4*>(preds_contig.data_ptr<float>()),
        reinterpret_cast<const float4*>(targets_contig.data_ptr<float>()),
        accum,
        ticket,
        out.data_ptr<float>(),
        total_vec,
        responders);
    CHECK_CUDA(cudaGetLastError());
    return out;
  }

  // Legacy/fallback path (odd responders, misaligned inputs, or kill switch):
  // output layout matches torch.cat((pred_sq, target_sq, covar)): fp32
  // accumulation into a zero-initialized buffer (atomic partial folds).
  auto out = torch::zeros({3 * responders}, preds.options());
  if (num_rows == 0) {
    return out;
  }

  // Enough row-blocks to fill the device while keeping per-output atomic
  // traffic bounded at gridDim partials per element.
  int64_t blocks = std::min<int64_t>(num_rows, static_cast<int64_t>(sm_count) * 4);
  blocks = std::max<int64_t>(1, blocks);

  metric_reduction_fused_kernel<<<static_cast<unsigned int>(blocks), threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      preds_contig.data_ptr<float>(),
      targets_contig.data_ptr<float>(),
      out.data_ptr<float>(),
      num_rows,
      responders);
  CHECK_CUDA(cudaGetLastError());
  return out;
}

torch::Tensor pack_rows(torch::Tensor input, torch::Tensor row_indices) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(row_indices.is_cuda(), "row_indices must be a CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(row_indices.dtype() == torch::kInt64, "row_indices must be int64");
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(row_indices.dim() == 1, "row_indices must be 1D");

  auto input_contig = input.contiguous();
  auto rows_contig = row_indices.contiguous();
  auto num_rows = rows_contig.size(0);
  auto num_cols = input_contig.size(1);
  auto output = torch::empty({num_rows, num_cols}, input.options());

  constexpr int threads = 256;
  int64_t total = num_rows * num_cols;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  pack_rows_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      input_contig.data_ptr<float>(),
      rows_contig.data_ptr<int64_t>(),
      output.data_ptr<float>(),
      num_rows,
      num_cols);
  CHECK_CUDA(cudaGetLastError());
  return output;
}

torch::Tensor scatter_rows(torch::Tensor packed, torch::Tensor row_indices, int64_t total_rows) {
  TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA tensor");
  TORCH_CHECK(row_indices.is_cuda(), "row_indices must be a CUDA tensor");
  TORCH_CHECK(packed.dtype() == torch::kFloat32, "packed must be float32");
  TORCH_CHECK(row_indices.dtype() == torch::kInt64, "row_indices must be int64");
  TORCH_CHECK(packed.dim() == 2, "packed must be 2D");
  TORCH_CHECK(row_indices.dim() == 1, "row_indices must be 1D");

  auto packed_contig = packed.contiguous();
  auto rows_contig = row_indices.contiguous();
  auto num_rows = rows_contig.size(0);
  auto num_cols = packed_contig.size(1);
  auto output = torch::zeros({total_rows, num_cols}, packed.options());

  constexpr int threads = 256;
  int64_t total = num_rows * num_cols;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  scatter_rows_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      packed_contig.data_ptr<float>(),
      rows_contig.data_ptr<int64_t>(),
      output.data_ptr<float>(),
      num_rows,
      num_cols);
  CHECK_CUDA(cudaGetLastError());
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("segment_abs_mean", &segment_abs_mean, "Segmented abs-mean reduction");
  m.def(
      "metric_reduction_fused",
      &metric_reduction_fused,
      "Fused single-pass pred_sq/target_sq/covar metric reduction");
  m.def("pack_rows", &pack_rows, "Pack active rows into a dense tensor");
  m.def("scatter_rows", &scatter_rows, "Scatter packed rows back into padded layout");
}
