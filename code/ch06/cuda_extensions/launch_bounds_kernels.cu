// launch_bounds_kernels.cu - CUDA kernels for launch bounds benchmarks
// Can be loaded as PyTorch CUDA extension

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

namespace {

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t _status = (call);                                          \
        TORCH_CHECK(_status == cudaSuccess,                                    \
                    "CUDA error at ", __FILE__, ":", __LINE__, " - ",          \
                    cudaGetErrorString(_status));                              \
    } while (0)

constexpr int kLaunchBoundsWorkIters = 64;
constexpr float kLaunchBoundsEps = 1e-6f;
constexpr int kBaselineThreadsPerBlock = 1024;
// B200 has 64K registers and 64 resident warps per SM.  The previous
// 1024-thread, two-block bound limited this register-heavy workload to 32
// registers per thread.  Three 512-thread blocks retain 48 resident warps
// while raising ptxas's register budget to roughly 42 registers per thread.
constexpr int kOptimizedThreadsPerBlock = 512;
constexpr int kOptimizedMinBlocksPerSm = 3;

#define KEEP_LIVE_16(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,  \
                     a13, a14, a15)                                            \
    asm volatile(""                                                            \
                 : "+f"(a0), "+f"(a1), "+f"(a2), "+f"(a3), "+f"(a4),          \
                   "+f"(a5), "+f"(a6), "+f"(a7), "+f"(a8), "+f"(a9),          \
                   "+f"(a10), "+f"(a11), "+f"(a12), "+f"(a13), "+f"(a14),     \
                   "+f"(a15))

__device__ __forceinline__ float launch_bounds_workload(float value) {
    float acc0 = value + 0.03125f;
    float acc1 = value - 0.0625f;
    float acc2 = value + 0.09375f;
    float acc3 = value - 0.125f;
    float acc4 = value + 0.15625f;
    float acc5 = value - 0.1875f;
    float acc6 = value + 0.21875f;
    float acc7 = value - 0.25f;
    float acc8 = value + 0.28125f;
    float acc9 = value - 0.3125f;
    float acc10 = value + 0.34375f;
    float acc11 = value - 0.375f;
    float acc12 = value + 0.40625f;
    float acc13 = value - 0.4375f;
    float acc14 = value + 0.46875f;
    float acc15 = value - 0.5f;

    #pragma unroll 4
    for (int iter = 0; iter < kLaunchBoundsWorkIters; ++iter) {
        const float mix0 = acc0 * 0.75f + acc6 * 0.25f;
        const float mix1 = acc1 * 0.70f - acc7 * 0.30f;
        const float mix2 = acc2 * 0.65f + acc8 * 0.35f;
        const float mix3 = acc3 * 0.60f - acc9 * 0.40f;
        const float mix4 = acc4 * 0.55f + acc10 * 0.45f;
        const float mix5 = acc5 * 0.50f - acc11 * 0.50f;
        const float mix6 = acc12 * 0.65f + acc14 * 0.35f - acc2 * 0.15f;
        const float mix7 = acc13 * 0.60f - acc15 * 0.40f + acc3 * 0.10f;

        const float inv0 = rsqrtf(fabsf(mix0) + fabsf(acc1) + fabsf(acc2) + kLaunchBoundsEps);
        const float inv1 = rsqrtf(fabsf(mix1) + fabsf(acc3) + fabsf(acc4) + kLaunchBoundsEps);
        const float inv2 = rsqrtf(fabsf(mix2) + fabsf(acc5) + fabsf(acc6) + kLaunchBoundsEps);
        const float inv3 = rsqrtf(fabsf(mix3) + fabsf(acc7) + fabsf(acc8) + kLaunchBoundsEps);
        const float inv4 = rsqrtf(fabsf(mix4) + fabsf(acc9) + fabsf(acc10) + kLaunchBoundsEps);
        const float inv5 = rsqrtf(fabsf(mix5) + fabsf(acc11) + fabsf(acc0) + kLaunchBoundsEps);
        const float inv6 = rsqrtf(fabsf(mix6) + fabsf(acc12) + fabsf(acc13) + kLaunchBoundsEps);
        const float inv7 = rsqrtf(fabsf(mix7) + fabsf(acc14) + fabsf(acc15) + kLaunchBoundsEps);

        acc0 = fmaf(acc0, 1.00003f, inv0 * 0.0002f + mix3 * 0.0001f);
        acc1 = fmaf(acc1, 0.99997f, inv1 * 0.0003f - mix4 * 0.0001f);
        acc2 = fmaf(acc2, 1.00005f, inv2 * 0.0004f + mix5 * 0.0001f);
        acc3 = fmaf(acc3, 0.99995f, inv3 * 0.0002f - mix0 * 0.0001f);
        acc4 = fmaf(acc4, 1.00007f, inv4 * 0.0003f + mix1 * 0.0001f);
        acc5 = fmaf(acc5, 0.99993f, inv5 * 0.0004f - mix2 * 0.0001f);
        acc6 = fmaf(acc6, 1.00002f, inv3 * 0.0002f + mix4 * 0.0001f);
        acc7 = fmaf(acc7, 0.99998f, inv4 * 0.0003f - mix5 * 0.0001f);
        acc8 = fmaf(acc8, 1.00004f, inv5 * 0.0004f + mix0 * 0.0001f);
        acc9 = fmaf(acc9, 0.99996f, inv0 * 0.0002f - mix1 * 0.0001f);
        acc10 = fmaf(acc10, 1.00006f, inv1 * 0.0003f + mix2 * 0.0001f);
        acc11 = fmaf(acc11, 0.99994f, inv2 * 0.0004f - mix3 * 0.0001f);
        acc12 = fmaf(acc12, 1.00008f, inv6 * 0.0003f + mix7 * 0.0001f);
        acc13 = fmaf(acc13, 0.99992f, inv7 * 0.0004f - mix6 * 0.0001f);
        acc14 = fmaf(acc14, 1.00009f, inv4 * 0.0002f + mix6 * 0.0001f);
        acc15 = fmaf(acc15, 0.99991f, inv5 * 0.0003f - mix7 * 0.0001f);

        KEEP_LIVE_16(
            acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
            acc8, acc9, acc10, acc11, acc12, acc13, acc14, acc15);
    }

    return acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 +
           acc8 + acc9 + acc10 + acc11 + acc12 + acc13 + acc14 + acc15;
}

} // anonymous namespace

// Kernel without launch bounds (baseline)
__global__ void kernel_no_launch_bounds(const float* input, float* output, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        output[idx] = launch_bounds_workload(input[idx]);
    }
}

// Kernel with launch bounds annotation (optimized)
__global__ __launch_bounds__(kOptimizedThreadsPerBlock, kOptimizedMinBlocksPerSm)
void kernel_with_launch_bounds(const float* input, float* output, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        output[idx] = launch_bounds_workload(input[idx]);
    }
}

void launch_bounds_baseline(torch::Tensor input, torch::Tensor output, int iterations) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.size(0) == output.size(0), "input and output must have same size");
    
    int n = input.size(0);
    int threads_per_block = kBaselineThreadsPerBlock;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Use PyTorch's current CUDA stream for consistency
    c10::cuda::CUDAGuard guard(input.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input.device().index());
    cudaStream_t stream_handle = stream.stream();
    
    for (int i = 0; i < iterations; ++i) {
        kernel_no_launch_bounds<<<num_blocks, threads_per_block, 0, stream_handle>>>(
            const_cast<float*>(input.data_ptr<float>()),
            output.data_ptr<float>(),
            n
        );
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream_handle));
}

void launch_bounds_optimized(torch::Tensor input, torch::Tensor output, int iterations) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.size(0) == output.size(0), "input and output must have same size");
    
    int n = input.size(0);
    int threads_per_block = kOptimizedThreadsPerBlock;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Use PyTorch's current CUDA stream for consistency
    c10::cuda::CUDAGuard guard(input.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input.device().index());
    cudaStream_t stream_handle = stream.stream();
    
    for (int i = 0; i < iterations; ++i) {
        kernel_with_launch_bounds<<<num_blocks, threads_per_block, 0, stream_handle>>>(
            const_cast<float*>(input.data_ptr<float>()),
            output.data_ptr<float>(),
            n
        );
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream_handle));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_bounds_baseline", &launch_bounds_baseline, "Kernel without launch bounds (baseline)");
    m.def("launch_bounds_optimized", &launch_bounds_optimized, "Kernel with launch bounds (optimized)");
}
