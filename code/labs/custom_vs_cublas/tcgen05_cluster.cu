/**
 * Cluster Launch with TMA Multicast
 * ==================================
 * 
 * Uses cudaLaunchKernelEx with cluster configuration (2x1).
 * TMA multicast broadcasts B tiles to multiple CTAs in a cluster,
 * reducing memory traffic.
 * 
 * Cluster shape 2x1: Two CTAs along M dimension share B tiles.
 */

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/half.h>

#include <cute/tensor.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/cluster_sm90.hpp>  // For cute::block_rank_in_cluster()

using namespace cute;

namespace cluster_impl {

using TypeA = cutlass::half_t;
using TypeB = cutlass::half_t;
using TypeC = float;
using TypeD = float;
using Accumulator = float;

constexpr int kStages = 4;

template <class TypeA_, class TypeB_, class ASmemLayout, class BSmemLayout>
struct ClusterSharedStorage {
  alignas(128) cute::ArrayEngine<TypeA_, cute::cosize_v<ASmemLayout>> smem_A[kStages];
  alignas(128) cute::ArrayEngine<TypeB_, cute::cosize_v<BSmemLayout>> smem_B[kStages];
  alignas(16) cute::uint64_t full_barrier[kStages];
  alignas(16) cute::uint64_t empty_barrier[kStages];
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE auto tensor_sA(int stage) {
    return make_tensor(make_smem_ptr(smem_A[stage].begin()), ASmemLayout{});
  }
  CUTE_DEVICE auto tensor_sB(int stage) {
    return make_tensor(make_smem_ptr(smem_B[stage].begin()), BSmemLayout{});
  }
};

using MmaTag =
    SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256,
                         UMMA::Major::K, UMMA::Major::K>;

template <class SharedStorageT,
          class ATensor, class BTensor, class CTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA,
          class TmaAtomA, class TmaAtomB>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1)
gemm_cluster(ATensor mA, BTensor mB, CTensor mC, DTensor mD,
             MmaTiler_MNK mma_tiler, TiledMMA tiled_mma,
             int grid_m, int grid_n,
             CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
             CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B) {
  
  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  // Get cluster info
  uint32_t cluster_rank = cute::block_rank_in_cluster();
  bool is_cluster_leader = (cluster_rank == 0);

  // Each CTA in cluster processes a different M tile but same N tile
  int tile_m = blockIdx.x;  // Unique per CTA
  int tile_n = blockIdx.y;  // Same for both CTAs in cluster

  auto mma_coord = make_coord(tile_m, tile_n, _);

  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1, _1, X>{});
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  SharedStorageT& storage = *reinterpret_cast<SharedStorageT*>(shared_memory);

  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  Tensor tCgA = cta_mma.partition_A(gA);
  Tensor tCgB = cta_mma.partition_B(gB);
  Tensor tCgC = cta_mma.partition_C(gC);
  Tensor tCgD = cta_mma.partition_C(gD);

  cute::TMEM::Allocator1Sm tmem_allocator{};
  if (elect_one_warp) {
    tmem_allocator.allocate(
        decltype(tmem_allocator)::Sm100TmemCapacityColumns,
        &storage.tmem_base_ptr);
  }
  __syncthreads();
  uint32_t tmem_base = storage.tmem_base_ptr;

  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);
  tCtAcc.data() = tmem_base;

  // TMA setup
  Tensor tma_coord_A = tma_atom_A.get_tma_tensor(shape(mA));
  Tensor tma_coord_B = tma_atom_B.get_tma_tensor(shape(mB));
  Tensor gCoordA = local_tile(tma_coord_A, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gCoordB = local_tile(tma_coord_B, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor tCgCoordA = cta_mma.partition_A(gCoordA);
  Tensor tCgCoordB = cta_mma.partition_B(gCoordB);

  // Per-stage tensors
  Tensor tCsA_0 = storage.tensor_sA(0);
  Tensor tCsA_1 = storage.tensor_sA(1);
  Tensor tCsA_2 = storage.tensor_sA(2);
  Tensor tCsA_3 = storage.tensor_sA(3);
  Tensor tCsB_0 = storage.tensor_sB(0);
  Tensor tCsB_1 = storage.tensor_sB(1);
  Tensor tCsB_2 = storage.tensor_sB(2);
  Tensor tCsB_3 = storage.tensor_sB(3);

  Tensor tCrA_0 = cta_mma.make_fragment_A(tCsA_0);
  Tensor tCrA_1 = cta_mma.make_fragment_A(tCsA_1);
  Tensor tCrA_2 = cta_mma.make_fragment_A(tCsA_2);
  Tensor tCrA_3 = cta_mma.make_fragment_A(tCsA_3);
  Tensor tCrB_0 = cta_mma.make_fragment_B(tCsB_0);
  Tensor tCrB_1 = cta_mma.make_fragment_B(tCsB_1);
  Tensor tCrB_2 = cta_mma.make_fragment_B(tCsB_2);
  Tensor tCrB_3 = cta_mma.make_fragment_B(tCsB_3);

  // TMA partitions
  auto [tAgA_0, tAsA_0] = tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsA_0), group_modes<0,3>(tCgCoordA));
  auto [tBgB_0, tBsB_0] = tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsB_0), group_modes<0,3>(tCgCoordB));
  auto [tAgA_1, tAsA_1] = tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsA_1), group_modes<0,3>(tCgCoordA));
  auto [tBgB_1, tBsB_1] = tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsB_1), group_modes<0,3>(tCgCoordB));
  auto [tAgA_2, tAsA_2] = tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsA_2), group_modes<0,3>(tCgCoordA));
  auto [tBgB_2, tBsB_2] = tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsB_2), group_modes<0,3>(tCgCoordB));
  auto [tAgA_3, tAsA_3] = tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsA_3), group_modes<0,3>(tCgCoordA));
  auto [tBgB_3, tBsB_3] = tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsB_3), group_modes<0,3>(tCgCoordB));

  int tma_bytes = sizeof(make_tensor_like(tAsA_0)) + sizeof(make_tensor_like(tBsB_0));

  // Initialize barriers
  if (elect_one_warp && elect_one_thr) {
    for (int s = 0; s < kStages; ++s) {
      cute::initialize_barrier(storage.full_barrier[s], 1);
      cute::initialize_barrier(storage.empty_barrier[s], 1);
    }
    cute::initialize_barrier(storage.mma_barrier, 1);
  }
  __syncthreads();

  int num_k_tiles = size<3>(tCgA);
  int full_phase[kStages] = {0, 0, 0, 0};
  int mma_phase = 0;

  auto issue_tma = [&](int stage, int k_tile) {
    cute::set_barrier_transaction_bytes(storage.full_barrier[stage], tma_bytes);
    switch (stage) {
      case 0:
        copy(tma_atom_A.with(storage.full_barrier[0]), tAgA_0(_, k_tile), tAsA_0);
        copy(tma_atom_B.with(storage.full_barrier[0]), tBgB_0(_, k_tile), tBsB_0);
        break;
      case 1:
        copy(tma_atom_A.with(storage.full_barrier[1]), tAgA_1(_, k_tile), tAsA_1);
        copy(tma_atom_B.with(storage.full_barrier[1]), tBgB_1(_, k_tile), tBsB_1);
        break;
      case 2:
        copy(tma_atom_A.with(storage.full_barrier[2]), tAgA_2(_, k_tile), tAsA_2);
        copy(tma_atom_B.with(storage.full_barrier[2]), tBgB_2(_, k_tile), tBsB_2);
        break;
      case 3:
        copy(tma_atom_A.with(storage.full_barrier[3]), tAgA_3(_, k_tile), tAsA_3);
        copy(tma_atom_B.with(storage.full_barrier[3]), tBgB_3(_, k_tile), tBsB_3);
        break;
    }
  };

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  // Prologue
  if (elect_one_warp && elect_one_thr) {
    for (int i = 0; i < min(kStages - 1, num_k_tiles); ++i) {
      issue_tma(i, i);
    }
  }
  __syncthreads();

  // Mainloop
  for (int k = 0; k < num_k_tiles; ++k) {
    int curr = k % kStages;
    int next_k = k + (kStages - 1);
    int next_s = next_k % kStages;

    // Issue next TMA first
    if (next_k < num_k_tiles && elect_one_warp && elect_one_thr) {
      issue_tma(next_s, next_k);
    }

    // Wait for current
    cute::wait_barrier(storage.full_barrier[curr], full_phase[curr]);
    full_phase[curr] ^= 1;

    // Execute MMA
    if (elect_one_warp) {
      auto& tCrA = (curr == 0) ? tCrA_0 : (curr == 1) ? tCrA_1 : 
                   (curr == 2) ? tCrA_2 : tCrA_3;
      auto& tCrB = (curr == 0) ? tCrB_0 : (curr == 1) ? tCrB_1 : 
                   (curr == 2) ? tCrB_2 : tCrB_3;

      for (int kb = 0; kb < size<2>(tCrA_0); ++kb) {
        gemm(tiled_mma, tCrA(_, _, kb), tCrB(_, _, kb), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      
      uint64_t* empty_ptr = reinterpret_cast<uint64_t*>(&storage.empty_barrier[curr]);
      cutlass::arch::umma_arrive(empty_ptr);
    }
  }

  __syncthreads();
  if (elect_one_warp) {
    cutlass::arch::umma_arrive(&storage.mma_barrier);
  }
  cute::wait_barrier(storage.mma_barrier, mma_phase);

  // Epilogue
  auto tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  auto thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

  // beta=0 GEMM (C=A@B^T): C-load eliminated (was multiplied by 0.0).
  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
  Tensor tDgD = thr_t2r_copy.partition_D(tCgD);
  Tensor tDrAcc = make_tensor<Accumulator>(shape(tDgD));
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);

  copy(tDrAcc, tDgD);  // D = accumulator (beta=0)

  __syncthreads();
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base, decltype(tmem_allocator)::Sm100TmemCapacityColumns);
  }
}

torch::Tensor run_cluster_matmul(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2);
  TORCH_CHECK(a.size(1) == b.size(1));
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16);
  TORCH_CHECK(a.is_cuda() && b.is_cuda());

  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();
  auto m = a_contig.size(0);
  auto k = a_contig.size(1);
  auto n = b_contig.size(0);

  TORCH_CHECK(m % 128 == 0 && n % 256 == 0 && k % 64 == 0,
              "Size must be divisible by tcgen05 tile");

  auto options = a.options().dtype(torch::kFloat32);
  auto c_buffer = torch::empty({m, n}, options);  // beta=0: no memset needed (C is never read)
  auto d_buffer = torch::empty_like(c_buffer);

  auto tiled_mma = make_tiled_mma(MmaTag{});
  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);

  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);

  using SharedStorageT = ClusterSharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout)>;

  Tensor mA = make_tensor(make_gmem_ptr(reinterpret_cast<TypeA const*>(a_contig.data_ptr<at::Half>())),
      make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
  Tensor mB = make_tensor(make_gmem_ptr(reinterpret_cast<TypeB const*>(b_contig.data_ptr<at::Half>())),
      make_layout(make_shape(n, k), make_stride(k, Int<1>{})));
  Tensor mC = make_tensor(make_gmem_ptr(c_buffer.data_ptr<TypeC>()),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
  Tensor mD = make_tensor(make_gmem_ptr(d_buffer.data_ptr<TypeD>()),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));

  auto tma_atom_A = make_tma_atom(SM90_TMA_LOAD{}, mA, sA_layout, select<0, 2>(mma_tiler));
  auto tma_atom_B = make_tma_atom(SM90_TMA_LOAD{}, mB, sB_layout, select<1, 2>(mma_tiler));

  int grid_m = (m + size(bM) - 1) / size(bM);
  int grid_n = (n + size(bN) - 1) / size(bN);

  // Grid must be divisible by cluster size
  if (grid_m % 2 != 0) grid_m += 1;

  dim3 dimBlock(128);
  dim3 dimGrid(grid_m, grid_n);
  int smem_bytes = sizeof(SharedStorageT);

  // Use cudaLaunchKernelEx for cluster launch
  cudaLaunchConfig_t launch_config;
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;

  launch_config.gridDim = dimGrid;
  launch_config.blockDim = dimBlock;
  launch_config.dynamicSmemBytes = smem_bytes;
  launch_config.stream = at::cuda::getCurrentCUDAStream();
  launch_config.numAttrs = 2;
  launch_config.attrs = attrs;

  auto* kernel_ptr = &gemm_cluster<
      SharedStorageT, decltype(mA), decltype(mB), decltype(mC), decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma), decltype(tma_atom_A), decltype(tma_atom_B)>;

  AT_CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  AT_CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  void* args[] = {
    (void*)&mA, (void*)&mB, (void*)&mC, (void*)&mD,
    (void*)&mma_tiler, (void*)&tiled_mma,
    (void*)&grid_m, (void*)&grid_n,
    (void*)&tma_atom_A, (void*)&tma_atom_B
  };

  AT_CUDA_CHECK(cudaLaunchKernelExC(&launch_config, (void*)kernel_ptr, args));

  return d_buffer.to(torch::kFloat16);
}

}  // namespace cluster_impl

torch::Tensor matmul_tcgen05_cluster(torch::Tensor a, torch::Tensor b) {
  return cluster_impl::run_cluster_matmul(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_tcgen05_cluster", &matmul_tcgen05_cluster);
}
