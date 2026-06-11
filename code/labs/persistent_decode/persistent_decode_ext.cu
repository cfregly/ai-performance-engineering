/**
 * Persistent decode kernel for SM100+ (Blackwell).
 *
 * This kernel computes attention scores and weighted value aggregation.
 * Uses shared memory for efficient data movement.
 *
 * TMEM NOTE: TMEM is for Tensor Core MMA accumulators, not general data copies.
 * See labs/custom_vs_cublas/tcgen05_*.cu for proper TMEM+MMA usage.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

namespace {

constexpr int MAX_HEAD_DIM = 128;
constexpr int MAX_SEQ_LEN = 64;

__device__ inline float dot_product(const float* q, const float* k, int head_dim, float* smem) {
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        acc += q[d] * k[d];
    }
    smem[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

__global__ void persistent_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int batch,
    int seq_len,
    int head_dim
) {
    extern __shared__ float smem[];
    float* reduce_buf = smem;

    // Grid-stride over sequences so `blocks` can be < batch.
    for (int seq_id = blockIdx.x; seq_id < batch; seq_id += gridDim.x) {
        // Per-token decode: compute dot(q_t, k_t) and write v_t * dot to out_t.
        for (int t = 0; t < seq_len; ++t) {
            const int base = (seq_id * seq_len + t) * head_dim;
            const float* q_ptr = q + base;
            const float* k_ptr = k + base;
            const float* v_ptr = v + base;

            float score = dot_product(q_ptr, k_ptr, head_dim, reduce_buf);

            float* out_ptr = out + base;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                out_ptr[d] = v_ptr[d] * score;
            }
        }
    }
}

} // namespace

void persistent_decode_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out, int blocks) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA tensor");
    TORCH_CHECK(q.scalar_type() == torch::kFloat, "q must be float32");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v shapes must match");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA tensor");
    TORCH_CHECK(out.scalar_type() == torch::kFloat, "out must be float32");
    
    const int batch = static_cast<int>(q.size(0));
    const int seq_len = static_cast<int>(q.size(1));
    const int head_dim = static_cast<int>(q.size(2));
    
    TORCH_CHECK(head_dim <= MAX_HEAD_DIM, "head_dim exceeds MAX_HEAD_DIM");
    TORCH_CHECK(seq_len <= MAX_SEQ_LEN, "seq_len exceeds MAX_SEQ_LEN");
    TORCH_CHECK(out.sizes() == q.sizes(), "out shape mismatch (expected [batch, seq_len, head_dim])");

    c10::cuda::CUDAGuard guard(q.get_device());
    
    const int threads = 64;
    const size_t smem_bytes = threads * sizeof(float);
    
    // Use the CURRENT stream, not getDefaultCUDAStream(): an explicit-but-default
    // stream is invisible to side-stream CUDA graph capture exactly like a raw
    // <<<>>> legacy launch (B45/B62/B65 silent-drop class). Eager behavior is
    // identical (current == default outside capture).
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int grid = std::min(blocks, batch);
    
    persistent_decode_kernel<<<grid, threads, smem_bytes, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        seq_len,
        head_dim);
    AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("persistent_decode", &persistent_decode_cuda, "Persistent decode (CUDA)");
}
