// optimized_cuda_graphs.cu -- Capture & replay micro-kernels with CUDA Graphs (optimized).

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cuda_graphs_workload.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(status));                              \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

__global__ void stage_kernel(float* data,
                             int n,
                             float scale,
                             float bias,
                             float frequency) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  float x = data[idx];
#pragma unroll
  for (int pass = 0; pass < kInnerPasses; ++pass) {
    x = tanhf(x * scale + bias);
    float s = __sinf(x * frequency + 0.05f * pass);
    float c = __cosf(x * 0.35f + 0.02f * pass);
    x = 0.65f * s + 0.35f * c;
  }
  data[idx] = x;
}

int main(int argc, char** argv) {
    NVTX_RANGE("main");
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf(
      "Optimized CUDA Graphs (graph replay) on %s (CC %d.%d)\n",
      prop.name,
      prop.major,
      prop.minor);

  if (prop.major < 7 || (prop.major == 7 && prop.minor < 5)) {
    std::printf("CUDA Graphs require compute capability 7.5 or newer.\n");
    return 3;
  }

  int N = 1 << 14, CAPTURE_ITERS = 1000, REPLAYS = 20;
  bool verify = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--elements") == 0 && i + 1 < argc) N = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--capture-iters") == 0 && i + 1 < argc) CAPTURE_ITERS = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--replays") == 0 && i + 1 < argc) REPLAYS = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--verify") == 0) verify = true;
    else return 1;
  }
  if (N <= 0 || CAPTURE_ITERS <= 0 || REPLAYS <= 0) return 1;
  const int ITER = CAPTURE_ITERS * REPLAYS;
  const size_t bytes = N * sizeof(float);

  std::vector<float> host(N);
  for (int i = 0; i < N; ++i) {
      NVTX_RANGE("setup");
    host[i] = sinf(0.001f * static_cast<float>(i));
  }

  float* device_ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&device_ptr, bytes));
  CUDA_CHECK(cudaMemcpy(device_ptr, host.data(), bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  dim3 block(128);
  dim3 grid((N + block.x - 1) / block.x);

  // Capture a large iteration block once.
  cudaGraph_t graph;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int iter = 0; iter < CAPTURE_ITERS; ++iter) {
      NVTX_RANGE("compute_kernel");
    for (const StageSpec& spec : kStageSpecs) {
        NVTX_RANGE("compute_kernel:stage_kernel");
      stage_kernel<<<grid, block, 0, stream>>>(device_ptr, N, spec.scale, spec.bias, spec.frequency);
    }
  }
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec;
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphUpload(exec, stream));

  // Warmup replay to match baseline's cold-start behavior.
  CUDA_CHECK(cudaGraphLaunch(exec, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Use the original input for measured work after warmup.
  CUDA_CHECK(cudaMemcpyAsync(device_ptr, host.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int replay = 0; replay < REPLAYS; ++replay) {
      NVTX_RANGE("compute_graph:launch");
    CUDA_CHECK(cudaGraphLaunch(exec, stream));
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  std::printf(
      "Optimized (graph replay): %.3f ms total, %.6f ms/iter\n",
      total_ms,
      total_ms / static_cast<float>(ITER));

  CUDA_CHECK(cudaMemcpy(host.data(), device_ptr, bytes, cudaMemcpyDeviceToHost));
  const bool correct = !verify || verify_graph_output(host.data(), N, ITER);
  if (verify) std::printf("Graph full-output verification: %s\n", correct ? "PASS" : "FAIL");
  double checksum = 0.0;
  for (float v : host) {
      NVTX_RANGE("verify");
    checksum += static_cast<double>(v);
  }
  std::printf("Optimized checksum: %.6e\n", checksum);

  CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(device_ptr));
  return correct ? 0 : 1;
}
