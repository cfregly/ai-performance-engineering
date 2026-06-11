// Minimal CUDA extension built with -lineinfo to demonstrate source/line capture.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__global__ void lineinfo_demo_kernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ out,
                                     int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Intentional simple computation for profiling
    out[idx] = a[idx] * 2.0f + b[idx];
  }
}

torch::Tensor lineinfo_demo_forward(torch::Tensor a, torch::Tensor b) {
  const int n = a.numel();
  auto out = torch::zeros_like(a);
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  lineinfo_demo_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      out.data_ptr<float>(),
      n);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lineinfo_demo_forward, "Lineinfo demo forward");
}
