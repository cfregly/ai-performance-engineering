// Compile each case separately; include and exercise production kernels/demos.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define main chapter_demo_main
#if CHAPTER_VALIDATION_CASE == 0
#include "../../ch06/occupancy_api.cu"
#elif CHAPTER_VALIDATION_CASE == 1
#include "../../ch06/optimized_ilp_low_occupancy_vec4_impl.cuh"
#elif CHAPTER_VALIDATION_CASE == 2
#include "../../ch08/threshold_async_kernel.cuh"
#include "../../ch08/threshold_tma_kernel.cuh"
#elif CHAPTER_VALIDATION_CASE == 3
#include "../../ch11/streams_overlap_demo.cu"
#elif CHAPTER_VALIDATION_CASE == 4
#include "../../ch11/streams_ordered_demo.cu"
#elif CHAPTER_VALIDATION_CASE == 5
#include "../../ch11/streams_warp_specialized_demo.cu"
#elif CHAPTER_VALIDATION_CASE == 6
#include "../../ch12/baseline_cuda_graphs.cu"
#elif CHAPTER_VALIDATION_CASE == 7
#include "../../ch12/optimized_cuda_graphs.cu"
#elif CHAPTER_VALIDATION_CASE == 8
#include "../../ch07/fp8_32byte_loads_demo.cu"
#elif CHAPTER_VALIDATION_CASE == 9
#include "../../ch10/optimized_dsmem_reduction_warp_specialized.cu"
#elif CHAPTER_VALIDATION_CASE == 10
#include "../../ch02/memory_transfer_pcie_demo.cu"
#else
#error Unknown chapter validation case
#endif
#undef main

#define CHECK(call) do { const auto err = (call); if (err != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); std::exit(1); } } while (0)
constexpr float guard_value = -12345.5f;
int checks_completed = 0;

template<class Launch, class Reference>
void check_unary(int n, Launch launch, Reference reference, int input_prefix = 8) {
  std::vector<float> input(n + 16, guard_value), output(n + 16, guard_value);
  for (int i = 0; i < n; ++i) {
    input[i + input_prefix] = static_cast<float>((i * 17 % 257) - 128) / 128.0f;
    output[i + 8] = NAN;
  }
  float *d_in, *d_out;
  CHECK(cudaMalloc(&d_in, input.size() * sizeof(float)));
  CHECK(cudaMalloc(&d_out, output.size() * sizeof(float)));
  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CHECK(cudaMemcpyAsync(d_in, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
  CHECK(cudaMemcpyAsync(d_out, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
  launch(d_in + input_prefix, d_out + 8, n, stream);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpyAsync(output.data(), d_out, output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CHECK(cudaStreamSynchronize(stream));
  for (int i = 0; i < n + 16; ++i) {
    const double expected = i < 8 || i >= n + 8 ? guard_value : reference(input[i - 8 + input_prefix], i - 8);
    const double tolerance = i < 8 || i >= n + 8 ? 0 : 3e-5 * std::max(1.0, std::abs(expected));
    if (!std::isfinite(output[i]) || std::abs(output[i] - expected) > tolerance) {
      std::fprintf(stderr, "case %d n=%d index=%d actual=%.9g expected=%.9g\n", CHAPTER_VALIDATION_CASE, n, i - 8, output[i], expected);
      std::exit(1);
    }
  }
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); CHECK(cudaStreamDestroy(stream));
  ++checks_completed;
}

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "UNSUPPORTED: CUDA GPU unavailable\n"); return 3;
  }
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));
  if (prop.major * 10 + prop.minor != CHAPTER_VALIDATION_TARGET) {
    std::fprintf(stderr, "UNSUPPORTED: binary target %d, actual CC %d.%d\n", CHAPTER_VALIDATION_TARGET, prop.major, prop.minor); return 3;
  }
  std::printf("Actual GPU: %s CC %d.%d\n", prop.name, prop.major, prop.minor);
#if CHAPTER_VALIDATION_CASE == 0
  int min_grid = 0, threads = 0;
  CHECK(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&min_grid, &threads, sampleKernel, sample_shared_bytes, 0));
  for (int n : {1, 3, 31, 257, 1023, 4099}) {
    check_unary(n, [=](float* in, float* out, int count, cudaStream_t s) {
      CHECK(cudaMemcpyAsync(out, in, count * sizeof(float), cudaMemcpyDeviceToDevice, s));
      sampleKernel<<<(count + threads - 1) / threads, threads, sample_shared_bytes(threads), s>>>(out, count);
    }, [](float x, int) { return std::sqrt(double(x) * x + 1.0); });
  }
#elif CHAPTER_VALIDATION_CASE == 1
  const double coefficients[] = {2, 3, 4, 5, 2.5, 3.5, 4.5, 5.5};
  const double biases[] = {1, -5, 2, -3, .5, -2, 1.5, -4};
  for (int n : {1, 3, 7, 8, 9, 257, 2051, 10003}) {
    check_unary(n, [](float* in, float* out, int count, cudaStream_t s) {
      ilp_low_occ_vec4::unrolled_ilp_kernel<<<2, 128, 0, s>>>(out, in, count);
    }, [&](float x, int i) { return x * coefficients[i % 8] + biases[i % 8]; });
    check_unary(n, [](float* in, float* out, int count, cudaStream_t s) {
      ilp_low_occ_vec4::independent_ops_kernel<<<2, 128, 0, s>>>(out, in, count);
    }, [](float x, int) { return 7.0 * x - 4.0; });
  }
#elif CHAPTER_VALIDATION_CASE == 2
  auto reference = [](float x, int) {
    const double magnitude = std::abs(double(x));
    if (magnitude <= .25) return 0.0;
    const double scale = (magnitude > .375 ? 1.25 : .85) * (x >= 0 ? 1 : -1);
    return (magnitude + std::sin(double(x)) * std::cos(double(x)) * .0001) * scale;
  };
  for (int n : {1, 3, 4, 7, 2047, 2048, 2049, 4095, 4096, 4097, 6144, 8191, 8192, 8193, 12293}) {
    check_unary(n, [](float* in, float* out, int count, cudaStream_t s) {
      ch08::threshold_naive_kernel<<<(count + 31) / 32, 32, 0, s>>>(in, out, .25f, count);
    }, reference);
    check_unary(n, [](float* in, float* out, int count, cudaStream_t s) {
      if (ch08::launch_threshold_predicated_async(in, out, .25f, count, s) != ch08::ThresholdAsyncLaunchResult::kSuccess) std::exit(1);
    }, reference);
    check_unary(n, [](float* in, float* out, int count, cudaStream_t s) {
      ch08::threshold_tma_pipeline_kernel<4><<<2, 512, 2 * 512 * 4 * sizeof(float), s>>>(in, out, .25f, count);
    }, reference);
    check_unary(n, [](float* in, float* out, int count, cudaStream_t s) {
      ch08::threshold_tma_pipeline_kernel<6><<<2, 512, 2 * 512 * 6 * sizeof(float), s>>>(in, out, .25f, count);
    }, reference);
    check_unary(n, [](float* in, float* out, int count, cudaStream_t s) {
      ch08::threshold_tma_pipeline_kernel<8><<<2, 512, 2 * 512 * 8 * sizeof(float), s>>>(in, out, .25f, count);
    }, reference);
  }
  // Offset views remain legal float pointers but cannot assert 16-byte alignment.
  for (int n : {4, 2048, 4097}) {
    check_unary(n, [](float* in, float* out, int count, cudaStream_t s) {
      ch08::threshold_tma_pipeline_kernel<4><<<2, 512, 2 * 512 * 4 * sizeof(float), s>>>(in, out, .25f, count);
    }, reference, 1);
  }
#elif CHAPTER_VALIDATION_CASE == 3
  for (int n : {1, 3, 4, 7, 8, 9, 255, 257, 1025, 2051, 16387}) {
    for (int kind = 0; kind < 4; ++kind) {
      check_unary(n, [=](float* in, float* out, int count, cudaStream_t s) {
        CHECK(cudaMemcpyAsync(out, in, count * sizeof(float), cudaMemcpyDeviceToDevice, s));
        if (kind == 0) scale_kernel<<<(count + 255) / 256, 256, 0, s>>>(out, count, 1.1f);
        else if (kind == 1) scale_kernel_vectorized<<<(count + 1023) / 1024, 256, 0, s>>>(out, count, 1.1f);
        else if (kind == 2) scale_kernel_vectorized_float8<<<(count + 2047) / 2048, 256, 0, s>>>(out, count, 1.1f);
        else scale_kernel_async<<<(count + 1023) / 1024, 256, 0, s>>>(out, count, 1.1f);
      }, [](float x, int) {
        double value = x;
        for (int i = 0; i < 4; ++i) value = (value * double(1.1f) + .001) * .9 - .0002;
        return value;
      });
    }
  }
  for (int n : {1, 257, 4099}) {
    if (run_stream_overlap_demo(n, 7, 5) != 0) return 1;
    ++checks_completed;
  }
#elif CHAPTER_VALIDATION_CASE == 4
  for (int n : {1, 2, 7, 17, 1025, 4099}) for (int pipelines : {1, 3, 8, 11}) {
    if (run_stream_ordered_demo(n, pipelines) != 0) return 1;
    ++checks_completed;
  }
#elif CHAPTER_VALIDATION_CASE == 5
  for (const char* streams : {"1", "3", "11"}) {
    const char* argv[] = {"warp-demo", "--streams", streams, "--batches", "8", "--batch-elems", "257"};
    if (chapter_demo_main(7, const_cast<char**>(argv)) != 0) return 1;
    ++checks_completed;
  }
#elif CHAPTER_VALIDATION_CASE == 6
  for (const char* n : {"1", "257", "1025"}) {
    const char* argv[] = {"graph-baseline", "--elements", n, "--iterations", "6", "--verify"};
    if (chapter_demo_main(6, const_cast<char**>(argv)) != 0) return 1;
    ++checks_completed;
  }
#elif CHAPTER_VALIDATION_CASE == 7
  for (const char* n : {"1", "257", "1025"}) {
    const char* argv[] = {"graph-optimized", "--elements", n, "--capture-iters", "2", "--replays", "3", "--verify"};
    if (chapter_demo_main(8, const_cast<char**>(argv)) != 0) return 1;
    ++checks_completed;
  }
#elif CHAPTER_VALIDATION_CASE == 8
  for (int n : {32, 64, 32 * 257}) for (int width : {8, 32}) {
    std::vector<__nv_fp8_e4m3> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = __nv_fp8_e4m3(float((i % 31) - 15) / 8.0f);
    __nv_fp8_e4m3 *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, n)); CHECK(cudaMalloc(&d_out, n));
    CHECK(cudaMemcpy(d_in, input.data(), n, cudaMemcpyHostToDevice));
    if (width == 8) fp8_scale_8byte<<<(n / 8 + 255) / 256, 256>>>(reinterpret_cast<fp8x8*>(d_in), reinterpret_cast<fp8x8*>(d_out), 1.5f, n / 8);
    else fp8_scale_32byte<<<(n / 32 + 255) / 256, 256>>>(reinterpret_cast<fp8x32*>(d_in), reinterpret_cast<fp8x32*>(d_out), 1.5f, n / 32);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(output.data(), d_out, n, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) if (output[i].__x != __nv_fp8_e4m3(float(input[i]) * 1.5f).__x) return 1;
    // Check the exact production initialization path, including its raw byte.
    const __nv_fp8_e4m3 one(1.0f);
    CHECK(cudaMemset(d_in, one.__x, n));
    CHECK(cudaMemcpy(output.data(), d_in, n, cudaMemcpyDeviceToHost));
    for (const auto value : output) if (float(value) != 1.0f) return 1;
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); ++checks_completed;
  }
#elif CHAPTER_VALIDATION_CASE == 9
  int cluster_support = 0;
  CHECK(cudaDeviceGetAttribute(&cluster_support, cudaDevAttrClusterLaunch, 0));
  if (!cluster_support) { std::fprintf(stderr, "UNSUPPORTED: cluster launch unavailable\n"); return 3; }
  constexpr int per_cluster = ELEMENTS_PER_BLOCK * CLUSTER_SIZE;
  for (int n : {1, 3, 17, per_cluster - 1, per_cluster, per_cluster + 3}) {
    const int clusters = (n + per_cluster - 1) / per_cluster;
    std::vector<float> input(n), output(clusters + 16, guard_value);
    for (int i = 0; i < n; ++i) input[i] = float((i % 17) - 8) / 16.0f;
    for (int i = 0; i < clusters; ++i) output[i + 8] = NAN;
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, n * sizeof(float))); CHECK(cudaMalloc(&d_out, output.size() * sizeof(float)));
    CHECK(cudaMemcpy(d_in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_out, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice));
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(clusters * CLUSTER_SIZE); config.blockDim = dim3(BLOCK_SIZE);
    cudaLaunchAttribute attr{}; attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {CLUSTER_SIZE, 1, 1}; config.numAttrs = 1; config.attrs = &attr;
    CHECK(cudaLaunchKernelEx(&config, dsmem_warp_specialized_reduction_kernel, d_in, d_out + 8, n, per_cluster));
    CHECK(cudaMemcpy(output.data(), d_out, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < clusters + 16; ++i) {
      double expected = guard_value;
      if (i >= 8 && i < clusters + 8) {
        expected = 0;
        for (int j = (i - 8) * per_cluster; j < std::min(n, (i - 7) * per_cluster); ++j) expected += input[j];
      }
      if (!std::isfinite(output[i]) || output[i] != expected) return 1;
    }
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); ++checks_completed;
  }
#elif CHAPTER_VALIDATION_CASE == 10
  for (int n : {1, 257, 4099}) check_unary(n, [](float* in, float* out, int count, cudaStream_t s) {
    traditional_process_kernel<<<(count + 255) / 256, 256, 0, s>>>(in, out, count);
  }, [](float x, int) {
    double result = 0;
    for (int i = 0; i < 8; ++i) result += std::sqrt(double(x) * x + i) * .125;
    return result;
  });
#endif
  std::printf("CHAPTER_CUDA_PASS case=%d checks=%d full_output=checked\n", CHAPTER_VALIDATION_CASE, checks_completed);
  return 0;
}
