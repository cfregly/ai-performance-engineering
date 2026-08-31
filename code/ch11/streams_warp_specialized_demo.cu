#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status__ = (call);                                           \
    if (status__ != cudaSuccess) {                                           \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(status__));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

namespace {

constexpr int kDefaultStreams = 3;
constexpr int kDefaultBatches = 24;
constexpr int kDefaultBatchElems = 1 << 17;          // 131,072 floats per batch
constexpr int kThreadsPerBlock = 256;
constexpr double kDefaultReleaseThresholdGiB = 2.0;

__global__ void fused_bias_kernel(const float* __restrict__ in_a,
                                  const float* __restrict__ in_b,
                                  float* __restrict__ out,
                                  int elems) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < elems) {
    float a = in_a[idx];
    float b = in_b[idx];
    // Simulate a modest amount of work to occupy the SM.
    out[idx] = (a * 2.0f + b) * 0.5f;
  }
}

struct DeviceBuffers {
  float* a = nullptr;
  float* b = nullptr;
  float* out = nullptr;
};

struct Options {
  int streams = kDefaultStreams;
  int batches = kDefaultBatches;
  int batch_elems = kDefaultBatchElems;
  double release_threshold_gib = kDefaultReleaseThresholdGiB;
  bool verify = true;
};

Options ParseOptions(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
      NVTX_RANGE("verify");
    const char* arg = argv[i];
    if (std::strcmp(arg, "--streams") == 0 && i + 1 < argc) {
      opts.streams = std::max(1, std::atoi(argv[++i]));
    } else if (std::strcmp(arg, "--batches") == 0 && i + 1 < argc) {
      opts.batches = std::max(1, std::atoi(argv[++i]));
    } else if (std::strcmp(arg, "--batch-elems") == 0 && i + 1 < argc) {
      opts.batch_elems = std::max(1, std::atoi(argv[++i]));
    } else if (std::strcmp(arg, "--release-threshold-gib") == 0 && i + 1 < argc) {
      opts.release_threshold_gib = std::max(0.0, std::atof(argv[++i]));
    } else if (std::strcmp(arg, "--skip-verify") == 0) {
      opts.verify = false;
    } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      std::printf(
          "Usage: %s [options]\n"
          "  --streams <int>                Number of CUDA streams (default %d)\n"
          "  --batches <int>                Mini-batches to process (default %d)\n"
          "  --batch-elems <int>            Elements per batch (default %d)\n"
          "  --release-threshold-gib <f>    Stream allocator release threshold (GiB, default %.1f)\n"
          "  --skip-verify                  Skip host-side result verification\n"
          "  --help                         Show this message\n",
          argv[0],
          kDefaultStreams,
          kDefaultBatches,
          kDefaultBatchElems,
          kDefaultReleaseThresholdGiB);
      std::exit(EXIT_SUCCESS);
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", arg);
      std::exit(EXIT_FAILURE);
    }
  }
  return opts;
}

}  // namespace

int main(int argc, char** argv) {
    NVTX_RANGE("main");
  Options opts = ParseOptions(argc, argv);

  const size_t batch_bytes = static_cast<size_t>(opts.batch_elems) * sizeof(float);
  const size_t total_elems = static_cast<size_t>(opts.batch_elems) * opts.batches;
  const size_t total_bytes = batch_bytes * opts.batches;

  float *h_a = nullptr, *h_b = nullptr, *h_out = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_a, total_bytes));
  CUDA_CHECK(cudaMallocHost(&h_b, total_bytes));
  CUDA_CHECK(cudaMallocHost(&h_out, total_bytes));

  for (size_t i = 0; i < total_elems; ++i) {
      NVTX_RANGE("setup");
    h_a[i] = static_cast<float>(i % 1024) * 0.25f;
    h_b[i] = static_cast<float>(i % 2048) * 0.5f;
    h_out[i] = -1.0f;
  }

  std::vector<cudaStream_t> streams(opts.streams);
  for (auto& stream : streams) {
      NVTX_RANGE("iteration");
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }

  // Tune the default memory pool for stream-ordered allocations.
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  cudaMemPool_t pool;
  CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, device));
  uint64_t threshold = prop.totalGlobalMem / 2;
  if (opts.release_threshold_gib > 0.0) {
    const long double bytes = opts.release_threshold_gib * (1024.0L * 1024.0L * 1024.0L);
    const uint64_t requested = static_cast<uint64_t>(std::llround(std::max<long double>(0.0L, bytes)));
    threshold = std::min<uint64_t>(requested, prop.totalGlobalMem);
  }
  CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));

  std::vector<DeviceBuffers> device_buffers(opts.streams);
  for (int s = 0; s < opts.streams; ++s) {
      NVTX_RANGE("iteration");
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device_buffers[s].a),
                               batch_bytes, streams[s]));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device_buffers[s].b),
                               batch_bytes, streams[s]));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device_buffers[s].out),
                               batch_bytes, streams[s]));
  }

  dim3 block(kThreadsPerBlock);
  dim3 grid((opts.batch_elems + block.x - 1) / block.x);

  for (int batch = 0; batch < opts.batches; ++batch) {
      NVTX_RANGE("batch");
    const int stream_idx = batch % opts.streams;
    cudaStream_t stream = streams[stream_idx];
    DeviceBuffers& buffers = device_buffers[stream_idx];

    const size_t offset_elems = static_cast<size_t>(batch) * opts.batch_elems;
    float* batch_a = h_a + offset_elems;
    float* batch_b = h_b + offset_elems;
    float* batch_out = h_out + offset_elems;

    CUDA_CHECK(cudaMemcpyAsync(buffers.a, batch_a, batch_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers.b, batch_b, batch_bytes,
                               cudaMemcpyHostToDevice, stream));

    fused_bias_kernel<<<grid, block, 0, stream>>>(buffers.a, buffers.b, buffers.out, opts.batch_elems);
    fused_bias_kernel<<<grid, block, 0, stream>>>(buffers.out, buffers.a, buffers.b, opts.batch_elems);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(batch_out, buffers.b, batch_bytes,
                               cudaMemcpyDeviceToHost, stream));
  }

  for (auto& stream : streams) {
      NVTX_RANGE("iteration");
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Add timed run for benchmark harness
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  
  cudaStream_t timing_stream;
  std::vector<cudaEvent_t> done(opts.streams);
  CUDA_CHECK(cudaStreamCreateWithFlags(&timing_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventRecord(start, timing_stream));
  for (int s = 0; s < opts.streams; ++s) {
    CUDA_CHECK(cudaEventCreateWithFlags(&done[s], cudaEventDisableTiming));
    CUDA_CHECK(cudaStreamWaitEvent(streams[s], start, 0));
  }
  for (int batch = 0; batch < opts.batches; ++batch) {
      NVTX_RANGE("compute_kernel:fused_bias_kernel");
    const int stream_idx = batch % opts.streams;
    cudaStream_t stream = streams[stream_idx];
    DeviceBuffers& buffers = device_buffers[stream_idx];
    fused_bias_kernel<<<grid, block, 0, stream>>>(buffers.a, buffers.b, buffers.out, opts.batch_elems);
    fused_bias_kernel<<<grid, block, 0, stream>>>(buffers.out, buffers.a, buffers.b, opts.batch_elems);
  }
  CUDA_CHECK(cudaGetLastError());
  for (int s = 0; s < opts.streams; ++s) {
    CUDA_CHECK(cudaEventRecord(done[s], streams[s]));
    CUDA_CHECK(cudaStreamWaitEvent(timing_stream, done[s], 0));
  }
  CUDA_CHECK(cudaEventRecord(stop, timing_stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  std::printf("Kernel time: %.4f ms\n", elapsed_ms);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  for (auto event : done) CUDA_CHECK(cudaEventDestroy(event));
  CUDA_CHECK(cudaStreamDestroy(timing_stream));

  double max_error = 0.0;
  if (opts.verify) {
    for (size_t i = 0; i < total_elems; ++i) {
        NVTX_RANGE("verify");
      double expected = 1.5 * h_a[i] + 0.5 * h_b[i];
      if (!std::isfinite(h_out[i])) max_error = INFINITY;
      else max_error = std::max(max_error, std::abs(expected - h_out[i]));
    }
  }

  if (opts.verify) {
    // Validate the timed feedback sequence too, including every output in the
    // final buffer of every active stream (not just the earlier warmup output).
    std::vector<float> timed_output(opts.batch_elems);
    for (int s = 0; s < std::min(opts.streams, opts.batches); ++s) {
      const int launches = (opts.batches - 1 - s) / opts.streams + 1;
      const int last_batch = s + (launches - 1) * opts.streams;
      const size_t offset = static_cast<size_t>(last_batch) * opts.batch_elems;
      CUDA_CHECK(cudaMemcpyAsync(timed_output.data(), device_buffers[s].b, batch_bytes,
                                 cudaMemcpyDeviceToHost, streams[s]));
      CUDA_CHECK(cudaStreamSynchronize(streams[s]));
      for (int i = 0; i < opts.batch_elems; ++i) {
        const double a = h_a[offset + i];
        double expected = 1.5 * a + 0.5 * h_b[offset + i];
        for (int launch = 0; launch < launches; ++launch) expected = 1.5 * a + 0.5 * expected;
        if (!std::isfinite(timed_output[i])) max_error = INFINITY;
        else max_error = std::max(max_error, std::abs(expected - timed_output[i]));
      }
    }
  }

  std::printf("warp_specialized_pipeline_multistream: batches=%d streams=%d elems/batch=%d release=%.1f GiB verify=%s max_error=%.3e\n",
              opts.batches,
              opts.streams,
              opts.batch_elems,
              opts.release_threshold_gib,
              opts.verify ? "yes" : "no",
              max_error);

  for (int s = 0; s < opts.streams; ++s) {
      NVTX_RANGE("cleanup");
    CUDA_CHECK(cudaFreeAsync(device_buffers[s].a, streams[s]));
    CUDA_CHECK(cudaFreeAsync(device_buffers[s].b, streams[s]));
    CUDA_CHECK(cudaFreeAsync(device_buffers[s].out, streams[s]));
  }
  for (auto& stream : streams) {
      NVTX_RANGE("cleanup");
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  CUDA_CHECK(cudaFreeHost(h_a));
  CUDA_CHECK(cudaFreeHost(h_b));
  CUDA_CHECK(cudaFreeHost(h_out));
  return (!opts.verify || max_error <= 1e-3) ? 0 : 1;
}
