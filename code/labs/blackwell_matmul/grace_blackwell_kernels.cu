#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../../core/common/headers/tma_helpers.cuh"

namespace cg = cooperative_groups;
using std::max;
using std::min;

namespace {
constexpr int BASELINE_BLOCK = 16;
// Tuned for B200 shared-memory limits and TMA alignment (16-byte stride).
constexpr int PIPE_TILE_M = 80;
constexpr int PIPE_TILE_N = 64;
constexpr int PIPE_TILE_K = 32;
constexpr int PIPE_THREADS = 8 * 32;   // more warps to cover 96x64 tile compute
constexpr int CLUSTER_THREADS = 3 * 32;
constexpr int TMA_THREADS = 8 * 32;    // keep consistent with PIPE_THREADS

__global__ void baseline_kernel(const half* __restrict__ A,
                                const half* __restrict__ B,
                                half* __restrict__ C,
                                int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    float lhs = __half2float(A[row * K + k]);
    float rhs = __half2float(B[col * K + k]);
    acc += lhs * rhs;
  }
  C[row * N + col] = __float2half(acc);
}

__device__ inline void zero_tile(float* tile, int elements) {
  for (int idx = threadIdx.x; idx < elements; idx += blockDim.x) {
    tile[idx] = 0.0f;
  }
}

__device__ inline void zero_tile_range(float* tile,
                                       int cols,
                                       int row_begin,
                                       int row_end) {
  const int rows = max(0, row_end - row_begin);
  const int elements = rows * cols;
  for (int idx = threadIdx.x; idx < elements; idx += blockDim.x) {
    const int local_row = idx / cols;
    const int col = idx - local_row * cols;
    const int row = row_begin + local_row;
    tile[row * PIPE_TILE_N + col] = 0.0f;
  }
}

__device__ inline void store_tile(const float* __restrict__ tile,
                                  half* __restrict__ C,
                                  int ld_c,
                                  int block_row,
                                  int block_col,
                                  int rows,
                                  int cols) {
  for (int idx = threadIdx.x; idx < rows * cols; idx += blockDim.x) {
    const int row = idx / cols;
    const int col = idx - row * cols;
    const int global_row = block_row + row;
    const int global_col = block_col + col;
    C[global_row * ld_c + global_col] = __float2half(tile[row * PIPE_TILE_N + col]);
  }
}

__device__ inline void compute_rows(const half* __restrict__ A_tile,
                                    const half* __restrict__ B_tile,
                                    float* __restrict__ accum_tile,
                                    int rows,
                                    int cols,
                                    int k_extent) {
  for (int idx = threadIdx.x; idx < rows * cols; idx += blockDim.x) {
    const int row = idx / cols;
    const int col = idx - row * cols;
    float acc = accum_tile[row * PIPE_TILE_N + col];
    for (int k_it = 0; k_it < k_extent; ++k_it) {
      float lhs = __half2float(A_tile[row * PIPE_TILE_K + k_it]);
      float rhs = __half2float(B_tile[col * PIPE_TILE_K + k_it]);
      acc += lhs * rhs;
    }
    accum_tile[row * PIPE_TILE_N + col] = acc;
  }
}

__device__ inline void compute_rows_range(const half* __restrict__ A_tile,
                                          const half* __restrict__ B_tile,
                                          float* __restrict__ accum_tile,
                                          int cols,
                                          int k_extent,
                                          int row_begin,
                                          int row_end) {
  const int rows = max(0, row_end - row_begin);
  const int elements = rows * cols;
  for (int idx = threadIdx.x; idx < elements; idx += blockDim.x) {
    const int local_row = idx / cols;
    const int col = idx - local_row * cols;
    const int row = row_begin + local_row;
    float acc = accum_tile[row * PIPE_TILE_N + col];
    for (int k_it = 0; k_it < k_extent; ++k_it) {
      float lhs = __half2float(A_tile[row * PIPE_TILE_K + k_it]);
      float rhs = __half2float(B_tile[col * PIPE_TILE_K + k_it]);
      acc += lhs * rhs;
    }
    accum_tile[row * PIPE_TILE_N + col] = acc;
  }
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void pipeline_prefetch_kernel(const half* __restrict__ A,
                                         const half* __restrict__ B,
                                         half* __restrict__ C,
                                         int M,
                                         int N,
                                         int K) {
  cg::thread_block cta = cg::this_thread_block();
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipe_state;
  auto pipe = cuda::make_pipeline(cta, &pipe_state);

  extern __shared__ unsigned char shared_mem[];
  half* A_tile = reinterpret_cast<half*>(shared_mem);
  half* B_tile = A_tile + TILE_M * TILE_K;
  float* C_tile = reinterpret_cast<float*>(B_tile + TILE_K * TILE_N);

  const int block_row = blockIdx.y * TILE_M;
  const int block_col = blockIdx.x * TILE_N;
  const int rows = max(0, min(TILE_M, M - block_row));
  const int cols = max(0, min(TILE_N, N - block_col));
  if (rows <= 0 || cols <= 0) {
    return;
  }

  const int warp_id = threadIdx.x / warpSize;
  const int lane_id = threadIdx.x % warpSize;

  zero_tile(C_tile, TILE_M * TILE_N);
  cta.sync();

  const int total_k_tiles = (K + TILE_K - 1) / TILE_K;
  for (int tile_idx = 0; tile_idx < total_k_tiles; ++tile_idx) {
    const int global_k = tile_idx * TILE_K;
    const int k_extent = min(TILE_K, K - global_k);

    pipe.producer_acquire();
    if (warp_id == 0) {
      for (int idx = lane_id; idx < TILE_M * TILE_K; idx += warpSize) {
        const int i = idx / TILE_K;
        const int j = idx % TILE_K;
        const int global_i = block_row + i;
        const int global_j = global_k + j;
        half val = __float2half(0.0f);
        if (i < rows && j < k_extent && global_i < M && global_j < K) {
          val = A[global_i * K + global_j];
        }
        A_tile[idx] = val;
      }
      for (int idx = lane_id; idx < TILE_K * TILE_N; idx += warpSize) {
        const int j = idx / TILE_K;  // N within tile
        const int i = idx - j * TILE_K;  // K within tile
        const int global_j = block_col + j;
        const int global_i = global_k + i;
        half val = __float2half(0.0f);
        if (i < k_extent && j < cols && global_i < K && global_j < N) {
          val = B[global_j * K + global_i];
        }
        B_tile[j * TILE_K + i] = val;
      }
  }
    pipe.producer_commit();
    pipe.consumer_wait();
    pipe.consumer_release();

    cta.sync();

    compute_rows(A_tile, B_tile, C_tile, rows, cols, k_extent);

    cta.sync();
  }

  store_tile(C_tile, C, N, block_row, block_col, rows, cols);
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void cluster_kernel(const half* __restrict__ A,
                               const half* __restrict__ B,
                               half* __restrict__ C,
                               int M,
                               int N,
                               int K) {
  cg::thread_block cta = cg::this_thread_block();
  cg::cluster_group cluster = cg::this_cluster();
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipe_state;
  auto pipe = cuda::make_pipeline(cta, &pipe_state);

  extern __shared__ unsigned char shared_mem[];
  half* A_tile = reinterpret_cast<half*>(shared_mem);
  half* B_tile = A_tile + TILE_M * TILE_K;
  float* C_tile = reinterpret_cast<float*>(B_tile + TILE_K * TILE_N);

  const int warp_id = threadIdx.x / warpSize;
  const int lane_id = threadIdx.x % warpSize;
  const int cluster_rank = cluster.block_rank();
  const int blocks_in_cluster =
      cluster.dim_blocks().x * cluster.dim_blocks().y * cluster.dim_blocks().z;

  const int cluster_dim_x = cluster.dim_blocks().x;
  const int tile_col = blockIdx.x / cluster_dim_x;
  const int tile_row = blockIdx.y;

  const int block_row = tile_row * TILE_M;
  const int block_col = tile_col * TILE_N;
  const int rows = max(0, min(TILE_M, M - block_row));
  const int cols = max(0, min(TILE_N, N - block_col));
  if (rows <= 0 || cols <= 0) {
    return;
  }

  const int rows_per_block = max(1, (rows + blocks_in_cluster - 1) / blocks_in_cluster);
  const int row_begin = min(cluster_rank * rows_per_block, rows);
  const int row_end = min(row_begin + rows_per_block, rows);

  zero_tile_range(C_tile, cols, row_begin, row_end);
  cta.sync();

  const int total_k_tiles = (K + TILE_K - 1) / TILE_K;
  for (int tile_idx = 0; tile_idx < total_k_tiles; ++tile_idx) {
    const int global_k = tile_idx * TILE_K;
    const int k_extent = min(TILE_K, K - global_k);

    if (cluster_rank == 0) {
      pipe.producer_acquire();
      if (warp_id == 0) {
        for (int idx = lane_id; idx < TILE_M * TILE_K; idx += warpSize) {
          const int i = idx / TILE_K;
          const int j = idx % TILE_K;
          const int global_i = block_row + i;
          const int global_j = global_k + j;
          half val = __float2half(0.0f);
          if (i < rows && j < k_extent && global_i < M && global_j < K) {
            val = A[global_i * K + global_j];
          }
          A_tile[idx] = val;
        }
        for (int idx = lane_id; idx < TILE_K * TILE_N; idx += warpSize) {
          const int j = idx / TILE_K;  // N within tile
          const int i = idx - j * TILE_K;  // K within tile
          const int global_j = block_col + j;
          const int global_i = global_k + i;
          half val = __float2half(0.0f);
          if (i < k_extent && j < cols && global_i < K && global_j < N) {
            val = B[global_j * K + global_i];
          }
          B_tile[j * TILE_K + i] = val;
        }
      }
      pipe.producer_commit();
      pipe.consumer_wait();
      pipe.consumer_release();
    }

    cluster.sync();

    const half* A_src = cluster.map_shared_rank(A_tile, 0);
    const half* B_src = cluster.map_shared_rank(B_tile, 0);

    const int row_count = max(0, row_end - row_begin);
    for (int idx = threadIdx.x; idx < row_count * k_extent; idx += blockDim.x) {
      const int local_row = idx / k_extent;
      const int col = idx - local_row * k_extent;
      const int row = row_begin + local_row;
      A_tile[row * PIPE_TILE_K + col] = A_src[row * PIPE_TILE_K + col];
    }
    for (int idx = threadIdx.x; idx < cols * k_extent; idx += blockDim.x) {
      const int col = idx / k_extent;
      const int k_it = idx - col * k_extent;
      B_tile[col * PIPE_TILE_K + k_it] = B_src[col * PIPE_TILE_K + k_it];
    }
    cta.sync();

    if (row_begin < row_end) {
      compute_rows_range(A_tile, B_tile, C_tile, cols, k_extent, row_begin, row_end);
    }

    cta.sync();
    cluster.sync();
  }

  if (warp_id == 2 && row_begin < row_end) {
    for (int row = row_begin + lane_id; row < row_end; row += warpSize) {
      for (int col = 0; col < cols; ++col) {
        const int global_row = block_row + row;
        const int global_col = block_col + col;
        C[global_row * N + global_col] = __float2half(C_tile[row * TILE_N + col]);
      }
    }
  }
}

// Encode a 2D tensor map for half data with the provided box dimensions.
inline bool encode_tensor_map_half(CUtensorMap& desc,
                                   PFN_cuTensorMapEncodeTiled_v12000 encode,
                                   void* base,
                                   int width,
                                   int height,
                                   int ld,
                                   int box_width,
                                   int box_height,
                                   CUtensorMapSwizzle swizzle_mode) {
  constexpr uint32_t rank = 2;
  std::uint64_t dims[rank] = {static_cast<std::uint64_t>(width),
                              static_cast<std::uint64_t>(height)};
  std::uint64_t stride[rank - 1] = {static_cast<std::uint64_t>(ld * sizeof(half))};
  std::uint32_t box[rank] = {static_cast<uint32_t>(box_width),
                             static_cast<uint32_t>(box_height)};
  std::uint32_t elem_stride[rank] = {1, 1};

  constexpr auto interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  constexpr auto promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
  constexpr auto oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  auto fn = encode ? encode : cuTensorMapEncodeTiled;
  CUresult res = fn(&desc,
                    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                    rank,
                    base,
                    dims,
                    stride,
                    box,
                    elem_stride,
                    interleave,
                    swizzle_mode,
                    promotion,
                    oob_fill);
  if (res != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    const char* err_name = nullptr;
    cuGetErrorString(res, &err_str);
    cuGetErrorName(res, &err_name);
    std::fprintf(stderr,
                 "[TMA] cuTensorMapEncodeTiled failed: %s (%s) "
                 "dtype=F16 dims={%llu,%llu} stride_bytes=%llu box={%u,%u} "
                 "swizzle=%d ld=%d\n",
                 err_str ? err_str : "unknown",
                 err_name ? err_name : "unknown",
                 static_cast<unsigned long long>(dims[0]),
                 static_cast<unsigned long long>(dims[1]),
                 static_cast<unsigned long long>(stride[0]),
                 box[0],
                 box[1],
                 static_cast<int>(swizzle_mode),
                 ld);
    return false;
  }
  return true;
}

// Real TMA path (requires hardware support). Uses 2-stage double-buffered TMA
// for A and B tiles and accumulates into a single shared C tile.
__global__ void tma_prefetch_kernel(const __grid_constant__ CUtensorMap A_desc,
                                    const __grid_constant__ CUtensorMap B_desc,
                                    const half* __restrict__ A,
                                    const half* __restrict__ B,
                                    half* __restrict__ C,
                                    int M,
                                    int N,
                                    int K) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  using block_barrier = cuda::barrier<cuda::thread_scope_block>;
  cg::thread_block cta = cg::this_thread_block();

  __shared__ alignas(128) half A_stage[2][PIPE_TILE_M][PIPE_TILE_K];
  __shared__ alignas(128) half B_stage[2][PIPE_TILE_N][PIPE_TILE_K];
  __shared__ float C_tile[PIPE_TILE_M][PIPE_TILE_N];
  __shared__ alignas(block_barrier) unsigned char barrier_storage[2][sizeof(block_barrier)];

  if (threadIdx.x == 0) {
    for (int i = 0; i < 2; ++i) {
      auto* bar = reinterpret_cast<block_barrier*>(barrier_storage[i]);
      init(bar, blockDim.x);
    }
    cuda::device::experimental::fence_proxy_async_shared_cta();
  }
  cta.sync();

  const int block_row = blockIdx.y * PIPE_TILE_M;
  const int block_col = blockIdx.x * PIPE_TILE_N;
  const int rows = max(0, min(PIPE_TILE_M, M - block_row));
  const int cols = max(0, min(PIPE_TILE_N, N - block_col));
  if (rows <= 0 || cols <= 0) {
    return;
  }

  zero_tile(reinterpret_cast<float*>(C_tile), PIPE_TILE_M * PIPE_TILE_N);
  cta.sync();

  const int total_k_tiles = (K + PIPE_TILE_K - 1) / PIPE_TILE_K;
  block_barrier::arrival_token tokens[2];

  auto issue_tile = [&](int tile_idx) -> block_barrier::arrival_token {
    const int stage = tile_idx % 2;
    auto* bar = reinterpret_cast<block_barrier*>(barrier_storage[stage]);
    const int global_k = tile_idx * PIPE_TILE_K;
    if (global_k >= K) {
      return bar->arrive();
    }
    const int k_extent = min(PIPE_TILE_K, K - global_k);
    const bool full_tile = (rows == PIPE_TILE_M) && (cols == PIPE_TILE_N) &&
                           (k_extent == PIPE_TILE_K);

    if (full_tile) {
      const std::size_t bytes = (PIPE_TILE_M * PIPE_TILE_K + PIPE_TILE_K * PIPE_TILE_N) *
                                sizeof(half);
      if (threadIdx.x == 0) {
        cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
            &A_stage[stage], &A_desc, global_k, block_row, *bar);
        cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
            &B_stage[stage], &B_desc, global_k, block_col, *bar);
        return cuda::device::barrier_arrive_tx(*bar, 1, bytes);
      }
      return bar->arrive();
    }

    // Edge tiles: manual copy with bounds checking.
    for (int idx = threadIdx.x; idx < rows * k_extent; idx += blockDim.x) {
      const int r = idx / k_extent;
      const int k = idx - r * k_extent;
      const int g_row = block_row + r;
      const int g_k = global_k + k;
      half val = __float2half(0.0f);
      if (g_row < M && g_k < K) {
        val = A[g_row * K + g_k];
      }
      A_stage[stage][r][k] = val;
    }
    for (int idx = threadIdx.x; idx < cols * k_extent; idx += blockDim.x) {
      const int c = idx / k_extent;
      const int k = idx - c * k_extent;
      const int g_c = block_col + c;
      const int g_k = global_k + k;
      half val = __float2half(0.0f);
      if (g_k < K && g_c < N) {
        val = B[g_c * K + g_k];
      }
      B_stage[stage][c][k] = val;
    }
    return bar->arrive();
  };

  const int preload = min(total_k_tiles, 2);
  for (int t = 0; t < preload; ++t) {
    tokens[t] = issue_tile(t);
  }

  for (int tile_idx = 0; tile_idx < total_k_tiles; ++tile_idx) {
    const int stage = tile_idx % 2;
    auto* bar = reinterpret_cast<block_barrier*>(barrier_storage[stage]);
    bar->wait(std::move(tokens[stage]));
    cta.sync();

    const int global_k = tile_idx * PIPE_TILE_K;
    const int k_extent = min(PIPE_TILE_K, K - global_k);
    compute_rows(&A_stage[stage][0][0], &B_stage[stage][0][0], &C_tile[0][0],
                 rows, cols, k_extent);
    cta.sync();

    const int next = tile_idx + 2;
    if (next < total_k_tiles) {
      tokens[stage] = issue_tile(next);
    }
  }

  store_tile(&C_tile[0][0], C, N, block_row, block_col, rows, cols);
#else
  (void)A_desc;
  (void)B_desc;
  (void)C;
  (void)M;
  (void)N;
  (void)K;
#endif
}

inline bool cluster_launch_supported_impl() {
#if CUDART_VERSION >= 12000
  const int device = at::cuda::current_device();
  int value = 0;
  return cudaDeviceGetAttribute(&value, cudaDevAttrClusterLaunch, device) == cudaSuccess && value != 0;
#else
  return false;
#endif
}

inline bool tma_supported_impl() {
#if CUDART_VERSION >= 12000
  const int device = at::cuda::current_device();
  int major = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) return false;
  // Hopper introduced the SM90 tensor-map copy instructions used here.
  return major >= 9;
#else
  return false;
#endif
}

void launch_baseline(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  dim3 block(BASELINE_BLOCK, BASELINE_BLOCK);
  dim3 grid((b.size(0) + block.x - 1) / block.x,
            (a.size(0) + block.y - 1) / block.y);
  auto stream = at::cuda::getCurrentCUDAStream();
  baseline_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
      reinterpret_cast<half*>(c.data_ptr<at::Half>()), a.size(0),
      b.size(0), a.size(1));
  AT_CUDA_CHECK(cudaGetLastError());
}

void launch_pipeline(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  dim3 block(PIPE_THREADS);
  dim3 grid((b.size(0) + PIPE_TILE_N - 1) / PIPE_TILE_N,
            (a.size(0) + PIPE_TILE_M - 1) / PIPE_TILE_M);
  const size_t shared_bytes =
      (PIPE_TILE_M * PIPE_TILE_K + PIPE_TILE_K * PIPE_TILE_N) * sizeof(half) +
      PIPE_TILE_M * PIPE_TILE_N * sizeof(float);
  AT_CUDA_CHECK(cudaFuncSetAttribute(
      pipeline_prefetch_kernel<PIPE_TILE_M, PIPE_TILE_N, PIPE_TILE_K>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  auto stream = at::cuda::getCurrentCUDAStream();
  pipeline_prefetch_kernel<PIPE_TILE_M, PIPE_TILE_N, PIPE_TILE_K>
      <<<grid, block, shared_bytes, stream>>>(
          reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<half*>(c.data_ptr<at::Half>()), a.size(0),
          b.size(0), a.size(1));
  AT_CUDA_CHECK(cudaGetLastError());
}

void launch_cluster(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  TORCH_CHECK(
      cluster_launch_supported_impl(),
      "Grace-Blackwell cluster launch requires cudaDevAttrClusterLaunch=1.");

  const int device = at::cuda::current_device();
  cudaDeviceProp prop{};
  AT_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int cluster_dim = prop.major >= 10 ? 8 : 4;

  const int tiles_x = (b.size(0) + PIPE_TILE_N - 1) / PIPE_TILE_N;
  const int tiles_y = (a.size(0) + PIPE_TILE_M - 1) / PIPE_TILE_M;

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(tiles_x * cluster_dim, tiles_y, 1);
  cfg.blockDim = dim3(CLUSTER_THREADS, 1, 1);
  cfg.dynamicSmemBytes =
      (PIPE_TILE_M * PIPE_TILE_K + PIPE_TILE_K * PIPE_TILE_N) * sizeof(half) +
      PIPE_TILE_M * PIPE_TILE_N * sizeof(float);

  cudaLaunchAttribute cluster_attr{};
  cluster_attr.id = cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim.x = cluster_dim;
  cluster_attr.val.clusterDim.y = 1;
  cluster_attr.val.clusterDim.z = 1;
  cfg.attrs = &cluster_attr;
  cfg.numAttrs = 1;

  AT_CUDA_CHECK(cudaFuncSetAttribute(cluster_kernel<PIPE_TILE_M, PIPE_TILE_N, PIPE_TILE_K>,
                       cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  AT_CUDA_CHECK(cudaFuncSetAttribute(cluster_kernel<PIPE_TILE_M, PIPE_TILE_N, PIPE_TILE_K>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       cfg.dynamicSmemBytes));

  auto stream = at::cuda::getCurrentCUDAStream();
  cfg.stream = stream;

  const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
  const half* B_ptr = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
  half* C_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());
  int M = a.size(0);
  int N = b.size(0);
  int K = a.size(1);

  AT_CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, cluster_kernel<PIPE_TILE_M, PIPE_TILE_N, PIPE_TILE_K>, A_ptr, B_ptr,
      C_ptr, M, N, K));
  AT_CUDA_CHECK(cudaGetLastError());
}

void launch_tma(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  if (!tma_supported_impl()) {
    TORCH_CHECK(false,
                "TMA requires Hopper or newer and a compatible CUDA runtime; "
                "use optimized_blackwell_matmul_pseudo instead.");
  }

  c10::cuda::CUDAGuard device_guard(a.device());
  AT_CUDA_CHECK(cudaFree(nullptr));  // ensure primary context is initialized
  CUresult cu_init = cuInit(0);
  TORCH_CHECK(cu_init == CUDA_SUCCESS, "cuInit failed for TMA path");

  PFN_cuTensorMapEncodeTiled_v12000 encode = cuda_tma::load_cuTensorMapEncodeTiled();
  TORCH_CHECK(encode != nullptr, "cuTensorMapEncodeTiled unavailable on this runtime");

  CUtensorMap A_desc{};
  CUtensorMap B_desc{};
  TORCH_CHECK(
      encode_tensor_map_half(A_desc,
                             encode,
                             a.data_ptr(),
                             a.size(1),   // width = K
                             a.size(0),   // height = M
                             a.size(1),   // ld = K
                             PIPE_TILE_K,
                             PIPE_TILE_M,
                             CU_TENSOR_MAP_SWIZZLE_NONE),
      "failed to encode A tensor map");
  TORCH_CHECK(
      encode_tensor_map_half(B_desc,
                             encode,
                             b.data_ptr(),
                             b.size(1),   // width = K
                             b.size(0),   // height = N
                             b.size(1),   // ld = K
                             PIPE_TILE_K,
                             PIPE_TILE_N,
                             CU_TENSOR_MAP_SWIZZLE_NONE),
      "failed to encode B tensor map");

  dim3 block(TMA_THREADS);
  dim3 grid((b.size(0) + PIPE_TILE_N - 1) / PIPE_TILE_N,
            (a.size(0) + PIPE_TILE_M - 1) / PIPE_TILE_M);
  auto stream = at::cuda::getCurrentCUDAStream();
  tma_prefetch_kernel<<<grid, block, 0, stream>>>(
      A_desc, B_desc,
      reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
      reinterpret_cast<half*>(c.data_ptr<at::Half>()), a.size(0), b.size(0),
      a.size(1));
  AT_CUDA_CHECK(cudaGetLastError());
}

torch::Tensor run_kernel(torch::Tensor a,
                         torch::Tensor b,
                         void (*launcher)(torch::Tensor, torch::Tensor, torch::Tensor)) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "expected 2D tensors");
  TORCH_CHECK(a.size(1) == b.size(1), "incompatible shapes");
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "tensors must live on CUDA");
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16,
              "use float16 tensors");
  TORCH_CHECK(a.device() == b.device(), "inputs must use the same CUDA device");
  TORCH_CHECK(a.size(0) > 0 && b.size(0) > 0 && a.size(1) > 0, "matrix dimensions must be positive");
  c10::cuda::CUDAGuard device_guard(a.device());
  auto c = torch::empty({a.size(0), b.size(0)}, a.options());
  launcher(a.contiguous(), b.contiguous(), c);
  return c;
}

}  // namespace

TORCH_LIBRARY(grace_blackwell_capstone, m) {
  m.def("baseline_blackwell_matmul(Tensor a, Tensor b) -> Tensor");
  m.def("optimized_blackwell_matmul_tma(Tensor a, Tensor b) -> Tensor");
  m.def("optimized_blackwell_matmul_pipeline(Tensor a, Tensor b) -> Tensor");
  m.def("optimized_blackwell_matmul_cluster(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(grace_blackwell_capstone, CUDA, m) {
  m.impl("baseline_blackwell_matmul",
         [](torch::Tensor a, torch::Tensor b) {
           return run_kernel(a, b, launch_baseline);
         });
  m.impl("optimized_blackwell_matmul_tma",
         [](torch::Tensor a, torch::Tensor b) {
           return run_kernel(a, b, launch_tma);
         });
  m.impl("optimized_blackwell_matmul_pipeline",
         [](torch::Tensor a, torch::Tensor b) {
           return run_kernel(a, b, launch_pipeline);
         });
  m.impl("optimized_blackwell_matmul_cluster",
         [](torch::Tensor a, torch::Tensor b) {
           return run_kernel(a, b, launch_cluster);
         });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("baseline_blackwell_matmul", [](torch::Tensor a, torch::Tensor b) {
    return run_kernel(a, b, launch_baseline);
  });
  m.def("optimized_blackwell_matmul_pseudo", [](torch::Tensor a, torch::Tensor b) {
    return run_kernel(a, b, launch_pipeline);
  });
  m.def("optimized_blackwell_matmul_pipeline", [](torch::Tensor a, torch::Tensor b) {
    return run_kernel(a, b, launch_pipeline);
  });
  m.def("optimized_blackwell_matmul_tma",
        [](torch::Tensor a, torch::Tensor b) {
          return run_kernel(a, b, launch_tma);
        });
  m.def("optimized_blackwell_matmul_cluster",
        [](torch::Tensor a, torch::Tensor b) {
          return run_kernel(a, b, launch_cluster);
        });
  m.def("cluster_launch_supported", []() { return cluster_launch_supported_impl(); });
  m.def("tma_supported", []() { return tma_supported_impl(); });
}
