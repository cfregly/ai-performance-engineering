// Memory access pattern CUDA kernels for coalescing and bank-conflict benchmarks.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

template <typename T>
__global__ void coalesced_copy_kernel(const T* __restrict__ src, T* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

template <typename T>
__global__ void uncoalesced_stride_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int stride,
    int n
) {
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread / warpSize;
    int lane = global_thread % warpSize;
    int dst_idx = warp_id * warpSize + lane;
    if (dst_idx >= n) {
        return;
    }
    int64_t base = static_cast<int64_t>(warp_id) * stride * warpSize;
    int src_idx = static_cast<int>((base + lane * stride) % n);
    dst[dst_idx] = src[src_idx];
}

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_bank_conflict_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    bool padded
) {
    extern __shared__ float tile_smem[];
    float* tile = tile_smem;
    
    auto smem_index = [=] __device__(int row, int col) -> int {
        int pitch = padded ? (TILE_DIM + 1) : TILE_DIM;
        return row * pitch + col;
    };
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = threadIdx.y + j;
        if (x < width && (y + j) < height) {
            tile[smem_index(row, threadIdx.x)] = input[(y + j) * width + x];
        }
    }
    
    __syncthreads();
    
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int col = threadIdx.y + j;
        if (x < height && (y + j) < width) {
            output[(y + j) * height + x] = tile[smem_index(threadIdx.x, col)];
        }
    }
}

void launch_coalesced_copy(torch::Tensor src, torch::Tensor dst) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(src.dtype() == torch::kFloat32, "Only float32 tensors supported");
    TORCH_CHECK(dst.dtype() == torch::kFloat32, "Only float32 tensors supported");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(src.numel() == dst.numel(), "Tensor sizes must match");
    int n = static_cast<int>(src.numel());
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    coalesced_copy_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        n
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "coalesced_copy kernel launch failed");
}

void launch_uncoalesced_copy(torch::Tensor src, torch::Tensor dst, int stride) {
    TORCH_CHECK(stride >= 1 && stride <= 1024, "Stride must be between 1 and 1024");
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(src.dtype() == torch::kFloat32, "Only float32 tensors supported");
    TORCH_CHECK(dst.dtype() == torch::kFloat32, "Only float32 tensors supported");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(src.numel() == dst.numel(), "Tensor sizes must match");
    int n = static_cast<int>(src.numel());
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    uncoalesced_stride_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        stride,
        n
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "uncoalesced_copy kernel launch failed");
}

void launch_bank_conflict_transpose(torch::Tensor src, torch::Tensor dst, bool padded) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(src.dtype() == torch::kFloat32, "Only float32 tensors supported");
    TORCH_CHECK(dst.dtype() == torch::kFloat32, "Only float32 tensors supported");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(src.numel() == dst.numel(), "Tensor sizes must match");
    TORCH_CHECK(src.dim() == 2, "Input must be 2D");
    TORCH_CHECK(dst.dim() == 2, "Output must be 2D");
    TORCH_CHECK(
        src.size(0) == dst.size(1) && src.size(1) == dst.size(0),
        "Output must be the transpose of input"
    );
    int width = static_cast<int>(src.size(1));
    int height = static_cast<int>(src.size(0));
    
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(
        (width + TILE_DIM - 1) / TILE_DIM,
        (height + TILE_DIM - 1) / TILE_DIM
    );
    
    size_t shared_bytes = sizeof(float) * TILE_DIM * (padded ? (TILE_DIM + 1) : TILE_DIM);
    
    transpose_bank_conflict_kernel<<<grid, block, shared_bytes, c10::cuda::getCurrentCUDAStream()>>>(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        width,
        height,
        padded
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "bank_conflict kernel launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("coalesced_copy", &launch_coalesced_copy, "Coalesced copy kernel");
    m.def("uncoalesced_copy", &launch_uncoalesced_copy, "Uncoalesced stride copy kernel");
    m.def("bank_conflict_transpose", &launch_bank_conflict_transpose, "Shared memory transpose with optional padding");
}
