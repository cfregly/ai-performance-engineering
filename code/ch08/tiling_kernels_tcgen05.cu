#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <cuda_runtime.h>
// CUTLASS includes
#include <cutlass/arch/barrier.h>
#include <cutlass/half.h>

// CuTe includes - main header first (includes all base algorithms)
#include <cute/tensor.hpp>

// SM100 (Blackwell) specific headers - must come AFTER cute/tensor.hpp
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include "../core/common/nvtx_utils.cuh"

using namespace cute;

namespace tiling_tcgen05_impl {

using TypeA = cutlass::half_t;
using TypeB = cutlass::half_t;
using TypeC = float;
using TypeD = float;
using Accumulator = float;

template <class TypeA_, class TypeB_, class ASmemLayout, class BSmemLayout>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA_, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB_, cute::cosize_v<BSmemLayout>> B;

  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint64_t tma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_sA() {
    return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{});
  }

  CUTE_DEVICE constexpr auto tensor_sB() {
    return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{});
  }
};

using TiledMmaTag =
    SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256,
                         UMMA::Major::K, UMMA::Major::K>;

struct VariantCTA1 {
  using Mma = TiledMmaTag;
  using TmemAllocator = cute::TMEM::Allocator1Sm;

  static constexpr auto cluster_shape() {
    return make_shape(Int<1>{}, Int<1>{}, Int<1>{});
  }
};

template <class SharedStorageT,
          class ATensor, class BTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA, class ClusterShape_MNK,
          class TmaAtomA, class TmaAtomB>
__global__ void gemm_device(ATensor mA,
                            BTensor mB,
                            DTensor mD,
                            MmaTiler_MNK mma_tiler,
                            TiledMMA tiled_mma,
                            ClusterShape_MNK cluster_shape,
                            CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
                            CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B) {
  auto mma_coord = make_coord(blockIdx.x, blockIdx.y, _);

  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  SharedStorageT &shared_storage =
      *reinterpret_cast<SharedStorageT*>(shared_memory);

  Tensor tCsA = shared_storage.tensor_sA();
  Tensor tCsB = shared_storage.tensor_sB();

  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  Tensor tCgA = cta_mma.partition_A(gA);
  Tensor tCgB = cta_mma.partition_B(gB);
  Tensor tCgD = cta_mma.partition_C(gD);

  Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgD);

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  typename VariantCTA1::TmemAllocator tmem_allocator{};

  if (elect_one_warp) {
    tmem_allocator.allocate(
        decltype(tmem_allocator)::Sm100TmemCapacityColumns,
        &shared_storage.tmem_base_ptr);
  }
  __syncthreads();
  uint32_t tmem_base = shared_storage.tmem_base_ptr;
  tCtAcc.data() = tmem_base;

  Tensor tma_coord_A = tma_atom_A.get_tma_tensor(shape(mA));
  Tensor tma_coord_B = tma_atom_B.get_tma_tensor(shape(mB));
  Tensor gCoordA = local_tile(tma_coord_A, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gCoordB = local_tile(tma_coord_B, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor tCgCoordA = cta_mma.partition_A(gCoordA);
  Tensor tCgCoordB = cta_mma.partition_B(gCoordB);

  auto [tAgA, tAsA] = tma_partition(
      tma_atom_A, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsA), group_modes<0,3>(tCgCoordA));
  auto [tBgB, tBsB] = tma_partition(
      tma_atom_B, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsB), group_modes<0,3>(tCgCoordB));

  int tma_transaction_bytes =
      sizeof(make_tensor_like(tAsA)) + sizeof(make_tensor_like(tBsB));

  if (elect_one_warp && elect_one_thr) {
    cute::initialize_barrier(shared_storage.mma_barrier, 1);
    cute::initialize_barrier(shared_storage.tma_barrier, 1);
  }
  int mma_barrier_phase_bit = 0;
  int tma_barrier_phase_bit = 0;
  __syncthreads();

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
    if (elect_one_warp && elect_one_thr) {
      cute::set_barrier_transaction_bytes(
          shared_storage.tma_barrier, tma_transaction_bytes);
      copy(tma_atom_A.with(shared_storage.tma_barrier), tAgA(_, k_tile), tAsA);
      copy(tma_atom_B.with(shared_storage.tma_barrier), tBgB(_, k_tile), tBsB);
    }

    cute::wait_barrier(shared_storage.tma_barrier, tma_barrier_phase_bit);
    tma_barrier_phase_bit ^= 1;

    if (elect_one_warp) {
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&shared_storage.mma_barrier);
    }

    cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
    mma_barrier_phase_bit ^= 1;
  }

  auto tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  auto thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
  Tensor tDgD = thr_t2r_copy.partition_D(tCgD);
  Tensor tDrAcc = make_tensor<Accumulator>(shape(tDgD));
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);
  copy(tDrAcc, tDgD);

  __syncthreads();
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(
        tmem_base,
        decltype(tmem_allocator)::Sm100TmemCapacityColumns);
  }
}

torch::Tensor run_tcgen05_matmul_pretransposed(torch::Tensor a,
                                               torch::Tensor b_transposed);

torch::Tensor run_tcgen05_matmul(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "expected 2D inputs");
  TORCH_CHECK(a.size(1) == b.size(0), "incompatible matmul shapes: A[M,K] @ B[K,N]");
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16,
              "tcgen05 kernel expects float16 tensors");
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "tensors must be CUDA tensors");

  // B is KxN in standard notation, but UMMA expects NxK layout (K-major)
  // So we transpose B to get NxK layout
  auto b_transposed = b.t().contiguous();
  return run_tcgen05_matmul_pretransposed(a, b_transposed);
}

void launch_tcgen05_matmul_pretransposed(torch::Tensor a,
                                         torch::Tensor b_transposed,
                                         torch::Tensor d_buffer) {
  TORCH_CHECK(a.dim() == 2 && b_transposed.dim() == 2, "expected 2D inputs");
  TORCH_CHECK(a.size(1) == b_transposed.size(1),
              "incompatible matmul shapes: A[M,K] @ B^T[N,K]");
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b_transposed.dtype() == torch::kFloat16,
              "tcgen05 kernel expects float16 tensors");
  TORCH_CHECK(d_buffer.dtype() == torch::kFloat32,
              "tcgen05 output buffer must be float32");
  TORCH_CHECK(a.is_cuda() && b_transposed.is_cuda() && d_buffer.is_cuda(),
              "tensors must be CUDA tensors");
  TORCH_CHECK(b_transposed.is_contiguous(),
              "pre-transposed B must be contiguous (N x K)");
  TORCH_CHECK(d_buffer.is_contiguous(),
              "tcgen05 output buffer must be contiguous");

  auto a_contig = a.contiguous();
  auto m = a_contig.size(0);
  auto k = a_contig.size(1);
  auto n = b_transposed.size(0);

  TORCH_CHECK(d_buffer.size(0) == m && d_buffer.size(1) == n,
              "tcgen05 output buffer has incompatible shape");
  TORCH_CHECK(d_buffer.device() == a.device(),
              "tcgen05 output buffer must be on the same device as A");

  auto tiled_mma = make_tiled_mma(typename VariantCTA1::Mma{});

  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);

  TORCH_CHECK(evenly_divides(shape(mma_tiler), tile_shape(tiled_mma)),
              "tcgen05 MMA tile must divide instruction tile");
  TORCH_CHECK(evenly_divides(make_shape(m, n, k), mma_tiler),
              "Problem size must be divisible by the tcgen05 tile");

  auto mma_shape_A =
      partition_shape_A(tiled_mma,
                        make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B =
      partition_shape_B(tiled_mma,
                        make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  auto sA_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{},
                        mma_shape_A);
  auto sB_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{},
                        mma_shape_B);

  using SharedStorageT =
      SharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout)>;

  Tensor mA = make_tensor(
      make_gmem_ptr(reinterpret_cast<TypeA const*>(
          a_contig.data_ptr<at::Half>())),
      make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
  Tensor mB = make_tensor(
      make_gmem_ptr(reinterpret_cast<TypeB const*>(
          b_transposed.data_ptr<at::Half>())),
      make_layout(make_shape(n, k), make_stride(k, Int<1>{})));
  Tensor mD = make_tensor(
      make_gmem_ptr(d_buffer.data_ptr<TypeD>()),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));

  auto tma_atom_A =
      make_tma_atom(SM90_TMA_LOAD{}, mA, sA_layout, select<0, 2>(mma_tiler));
  auto tma_atom_B =
      make_tma_atom(SM90_TMA_LOAD{}, mB, sB_layout, select<1, 2>(mma_tiler));

  dim3 dimBlock(128);
  dim3 dimGrid((m + size(bM) - 1) / size(bM),
               (n + size(bN) - 1) / size(bN));

  int smem_bytes = sizeof(SharedStorageT);

  using ClusterShape = decltype(VariantCTA1::cluster_shape());

  auto *kernel_ptr = &gemm_device<
      SharedStorageT,
      decltype(mA), decltype(mB),
      decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma),
      ClusterShape,
      decltype(tma_atom_A), decltype(tma_atom_B)>;

  AT_CUDA_CHECK(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  ClusterShape cluster_shape = VariantCTA1::cluster_shape();

  gemm_device<
      SharedStorageT,
      decltype(mA), decltype(mB),
      decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma),
      ClusterShape,
      decltype(tma_atom_A), decltype(tma_atom_B)>
      <<<dimGrid, dimBlock, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
          mA, mB, mD,
          mma_tiler, tiled_mma, cluster_shape,
          tma_atom_A, tma_atom_B);
  AT_CUDA_CHECK(cudaGetLastError());
}

torch::Tensor run_tcgen05_matmul_pretransposed(torch::Tensor a, torch::Tensor b_transposed) {
  TORCH_CHECK(a.dim() == 2 && b_transposed.dim() == 2, "expected 2D inputs");
  TORCH_CHECK(a.size(1) == b_transposed.size(1),
              "incompatible matmul shapes: A[M,K] @ B^T[N,K]");
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b_transposed.dtype() == torch::kFloat16,
              "tcgen05 kernel expects float16 tensors");
  TORCH_CHECK(a.is_cuda() && b_transposed.is_cuda(), "tensors must be CUDA tensors");
  TORCH_CHECK(b_transposed.is_contiguous(),
              "pre-transposed B must be contiguous (N x K)");

  auto m = a.size(0);
  auto n = b_transposed.size(0);

  auto d_options = a.options().dtype(torch::kFloat32);
  auto d_buffer = torch::empty({m, n}, d_options);
  launch_tcgen05_matmul_pretransposed(a, b_transposed, d_buffer);
  return d_buffer.to(torch::kFloat16);
}

torch::Tensor run_tcgen05_matmul_pretransposed_out(torch::Tensor a,
                                                   torch::Tensor b_transposed,
                                                   torch::Tensor d_buffer) {
  launch_tcgen05_matmul_pretransposed(a, b_transposed, d_buffer);
  return d_buffer;
}

}  // namespace tiling_tcgen05_impl

torch::Tensor matmul_tiling_tcgen05(torch::Tensor a, torch::Tensor b) {
  return tiling_tcgen05_impl::run_tcgen05_matmul(a, b);
}

torch::Tensor matmul_tiling_tcgen05_pretransposed(torch::Tensor a, torch::Tensor b_transposed) {
  return tiling_tcgen05_impl::run_tcgen05_matmul_pretransposed(a, b_transposed);
}

torch::Tensor matmul_tiling_tcgen05_pretransposed_out(torch::Tensor a,
                                                      torch::Tensor b_transposed,
                                                      torch::Tensor output) {
  return tiling_tcgen05_impl::run_tcgen05_matmul_pretransposed_out(
      a, b_transposed, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_tiling_tcgen05", &matmul_tiling_tcgen05);
  m.def("matmul_tiling_tcgen05_pretransposed", &matmul_tiling_tcgen05_pretransposed);
  m.def("matmul_tiling_tcgen05_pretransposed_out", &matmul_tiling_tcgen05_pretransposed_out);
}
