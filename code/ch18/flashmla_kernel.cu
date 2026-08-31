// Chapter 18: one-block-per-head scaled dot-product decode reference.
// This illustrates correct attention math, not a production FlashMLA implementation.
// Launch a power-of-two block with head_dim <= blockDim.x <= 1024.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cassert>
#include "../core/common/nvtx_utils.cuh"

__global__ void flashmla_decode(const half* __restrict__ q,
                                const half* __restrict__ k_cache,
                                const half* __restrict__ v_cache,
                                half* __restrict__ out,
                                const int* __restrict__ lengths,
                                int num_heads,
                                int head_dim,
                                int stride) {
  assert(num_heads > 0 && head_dim > 0 && head_dim <= blockDim.x);
  assert(blockDim.x <= 1024 && (blockDim.x & (blockDim.x - 1)) == 0);
  int batch = blockIdx.x / num_heads;
  int head = blockIdx.x % num_heads;
  assert(lengths[batch] >= 0 && lengths[batch] <= stride / (num_heads * head_dim));
  const int d = threadIdx.x;
  const bool active = d < head_dim;
  __shared__ float partial[1024];
  const half* q_head = q + (batch * num_heads + head) * head_dim;
  const half* k_head = k_cache + batch * stride + head * head_dim;
  const half* v_head = v_cache + batch * stride + head * head_dim;
  float acc = 0.0f;
  float denom = 0.0f;
  float running_max = -INFINITY;
  for (int pos = 0; pos < lengths[batch]; ++pos) {
    partial[d] = active ? __half2float(q_head[d]) *
        __half2float(k_head[pos * num_heads * head_dim + d]) : 0.0f;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (d < offset) partial[d] += partial[d + offset];
      __syncthreads();
    }
    const float score = partial[0] * rsqrtf(static_cast<float>(head_dim));
    __syncthreads(); // All threads read the sum before the next token reuses it.
    const float next_max = fmaxf(running_max, score);
    const float alpha = expf(running_max - next_max);
    const float weight = expf(score - next_max);
    denom = denom * alpha + weight;
    if (active) acc = acc * alpha + weight * __half2float(v_head[pos * num_heads * head_dim + d]);
    running_max = next_max;
  }
  if (active) {
    out[(batch * num_heads + head) * head_dim + d] = __float2half(denom > 0 ? acc / denom : 0.0f);
  }
}

#ifndef FLASHMLA_NO_MAIN
int main() {
    NVTX_RANGE("main");
  printf("Scaled dot-product decode reference (not a timed FlashMLA implementation)\n");
  return 0;
}
#endif
