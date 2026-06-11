// CUDA kernels demonstrating low vs high occupancy execution.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t status = (call);                                          \
        if (status != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(status));             \
        }                                                                     \
    } while (0)

__global__ void high_occupancy_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int n,
                                      int work_iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        float val = input[i];
        #pragma unroll 1
        for (int it = 0; it < work_iters; ++it) {
            val = val * 1.0001f + 0.1f;
        }
        output[i] = val;
    }
}

__launch_bounds__(64, 1)
__global__ void low_occupancy_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int n,
                                     int work_iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        float val = input[i];
        #pragma unroll 1
        for (int it = 0; it < work_iters; ++it) {
            val = val * 1.0001f + 0.1f;
        }
        output[i] = val;
    }
}

void validate_tensors(const torch::Tensor& input, const torch::Tensor& output) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 tensors supported");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Only float32 tensors supported");
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(input.numel() == output.numel(), "Tensor sizes must match");
}

}  // namespace

void run_low_occupancy(const torch::Tensor& input, torch::Tensor& output, int work_iters) {
    validate_tensors(input, output);
    int n = static_cast<int>(input.numel());
    int block = 64;
    int grid = std::min(std::max(1, (n + block - 1) / block / 8), 64);
    low_occupancy_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        work_iters
    );
    CUDA_CHECK(cudaGetLastError());
}

void run_high_occupancy(const torch::Tensor& input, torch::Tensor& output, int work_iters) {
    validate_tensors(input, output);
    int n = static_cast<int>(input.numel());
    int block = 256;
    int grid = std::min(std::max(1, (n + block - 1) / block), 512);
    high_occupancy_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        work_iters
    );
    CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_low_occupancy", &run_low_occupancy, "Launch low-occupancy kernel");
    m.def("run_high_occupancy", &run_high_occupancy, "Launch high-occupancy kernel");
}
