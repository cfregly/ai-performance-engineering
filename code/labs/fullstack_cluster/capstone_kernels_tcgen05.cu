#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <type_traits>
#include <utility>

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>

namespace cg = cooperative_groups;
using namespace cute;

using TypeA = cutlass::half_t;
using TypeB = cutlass::half_t;
using TypeC = float;
// fp16-direct epilogue: D is written as fp16 straight from the epilogue
// registers (fp32 accumulator -> one round-to-nearest-even conversion), so the
// host-side d_buffer.to(kFloat16) pass disappears. TypeC stays float: it is
// the UMMA accumulator type and (post C-load removal) only feeds layout math.
using TypeD = cutlass::half_t;
using Accumulator = float;
using Alpha = float;
using Beta = float;

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

template <typename Allocator>
struct ClusterTmemHelper; // unused: CTA-group kernels allocate locally

template <class MmaTag, int ClusterM>
struct TcgenVariantConfig {
  using Mma = MmaTag;
  static constexpr int kClusterM = ClusterM;
  static constexpr int kClusterN = 1;
  static constexpr int kClusterK = 1;
  using TmemAllocator = std::conditional_t<
      ClusterM == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

  static constexpr auto cluster_shape() {
    return make_shape(Int<ClusterM>{}, Int<1>{}, Int<1>{});
  }
};

using VariantCTA1 = TcgenVariantConfig<
    SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256,
                         UMMA::Major::K, UMMA::Major::K>,
    1>;

using VariantCTA2 = TcgenVariantConfig<
    SM100_MMA_F16BF16_2x1SM_SS<TypeA, TypeB, TypeC, 256, 256,
                               UMMA::Major::K, UMMA::Major::K>,
    2>;

template <class Variant,
          class SharedStorageT,
          class ATensor, class BTensor, class CTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA, class ClusterShape_MNK,
          class TmaAtomA, class TmaAtomB,
          class AlphaT, class BetaT>
__global__ void gemm_device_variant(ATensor mA,
                                    BTensor mB,
                                    CTensor mC,
                                    DTensor mD,
                                    MmaTiler_MNK mma_tiler,
                                    TiledMMA tiled_mma,
                                    ClusterShape_MNK cluster_shape,
                                    CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
                                    CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B,
                                    AlphaT alpha,
                                    BetaT beta) {
  Layout cluster_layout_vmnk =
      tiled_divide(make_layout(cluster_shape),
                   make_tile(typename TiledMMA::AtomThrID{}));

  auto mma_coord_vmnk =
      make_coord(blockIdx.x % size<0>(cluster_layout_vmnk),
                 blockIdx.x / size<0>(cluster_layout_vmnk),
                 blockIdx.y,
                 _);

  auto mma_coord = select<1, 2, 3>(mma_coord_vmnk);

  Tensor tma_mA = tma_atom_A.get_tma_tensor(shape(mA));
  Tensor tma_mB = tma_atom_B.get_tma_tensor(shape(mB));

  Tensor gA = local_tile(tma_mA, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(tma_mB, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1, _1, X>{});
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  SharedStorageT &shared_storage =
      *reinterpret_cast<SharedStorageT*>(shared_memory);

  uint64_t* mma_barrier_ptr = &shared_storage.mma_barrier;
  uint64_t* tma_barrier_ptr = &shared_storage.tma_barrier;

  Tensor tCsA = shared_storage.tensor_sA();
  Tensor tCsB = shared_storage.tensor_sB();

  auto mma_v = get<0>(mma_coord_vmnk);
  auto cta_mma = tiled_mma.get_slice(mma_v);
  Tensor tCgA = cta_mma.partition_A(gA);
  Tensor tCgB = cta_mma.partition_B(gB);
  Tensor tCgC = cta_mma.partition_C(gC);
  Tensor tCgD = cta_mma.partition_C(gD);

  Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = typename Variant::TmemAllocator;
  TmemAllocator tmem_allocator{};

  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &shared_storage.tmem_base_ptr);
  }
  __syncthreads();
  uint32_t tmem_base = shared_storage.tmem_base_ptr;
  tCtAcc.data() = tmem_base;

  auto cta_in_cluster_coord_vmnk =
      cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
  auto elect_one_cta = get<0>(cta_in_cluster_coord_vmnk) == Int<0>{};

  int mma_barrier_phase_bit = 0;
  int tma_barrier_phase_bit = 0;

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  if constexpr (Variant::kClusterM == 2) {
    // Cutlass tutorial 04: 2SM MMA + 2SM TMA multicast.
    auto [tAgA, tAsA] = tma_partition(
        tma_atom_A,
        get<2>(cta_in_cluster_coord_vmnk),
        make_layout(size<2>(cluster_layout_vmnk)),
        group_modes<0, 3>(tCsA),
        group_modes<0, 3>(tCgA));
    auto [tBgB, tBsB] = tma_partition(
        tma_atom_B,
        get<1>(cta_in_cluster_coord_vmnk),
        make_layout(size<1>(cluster_layout_vmnk)),
        group_modes<0, 3>(tCsB),
        group_modes<0, 3>(tCgB));

    uint16_t tma_mcast_mask_a =
        create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
    uint16_t tma_mcast_mask_b =
        create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
    uint16_t mma_mcast_mask_c =
        create_tma_multicast_mask<0, 1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk) |
        create_tma_multicast_mask<0, 2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);

    int tma_transaction_bytes =
        size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tAsA)) +
        size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tBsB));

    if (elect_one_warp && elect_one_thr) {
      int num_mcast_participants =
          size<1>(cluster_layout_vmnk) + size<2>(cluster_layout_vmnk) - 1;
      cute::initialize_barrier(*mma_barrier_ptr, num_mcast_participants);
      cute::initialize_barrier(*tma_barrier_ptr, 1);
    }
    cute::cluster_sync();

    for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
      if (elect_one_warp && elect_one_thr) {
        if (elect_one_cta) {
          cute::set_barrier_transaction_bytes(*tma_barrier_ptr, tma_transaction_bytes);
        }
        copy(tma_atom_A.with(*tma_barrier_ptr, tma_mcast_mask_a), tAgA(_, k_tile), tAsA);
        copy(tma_atom_B.with(*tma_barrier_ptr, tma_mcast_mask_b), tBgB(_, k_tile), tBsB);
      }

      if (elect_one_cta) {
        cute::wait_barrier(*tma_barrier_ptr, tma_barrier_phase_bit);
        tma_barrier_phase_bit ^= 1;

        if (elect_one_warp) {
          for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }
          cutlass::arch::umma_arrive_multicast_2x1SM(mma_barrier_ptr, mma_mcast_mask_c);
        }
      }

      cute::wait_barrier(*mma_barrier_ptr, mma_barrier_phase_bit);
      mma_barrier_phase_bit ^= 1;
    }
  } else {
    // 1SM MMA + SM90 TMA (no multicast).
    auto [tAgA, tAsA] = tma_partition(
        tma_atom_A, Int<0>{}, Layout<_1>{},
        group_modes<0, 3>(tCsA), group_modes<0, 3>(tCgA));
    auto [tBgB, tBsB] = tma_partition(
        tma_atom_B, Int<0>{}, Layout<_1>{},
        group_modes<0, 3>(tCsB), group_modes<0, 3>(tCgB));

    int tma_transaction_bytes =
        sizeof(make_tensor_like(tAsA)) + sizeof(make_tensor_like(tBsB));

    if (elect_one_warp && elect_one_thr) {
      cute::initialize_barrier(*mma_barrier_ptr, 1);
      cute::initialize_barrier(*tma_barrier_ptr, 1);
    }
    __syncthreads();

    for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
      if (elect_one_warp && elect_one_thr) {
        cute::set_barrier_transaction_bytes(*tma_barrier_ptr,
                                            tma_transaction_bytes);
        copy(tma_atom_A.with(*tma_barrier_ptr), tAgA(_, k_tile), tAsA);
        copy(tma_atom_B.with(*tma_barrier_ptr), tBgB(_, k_tile), tBsB);
      }

      cute::wait_barrier(*tma_barrier_ptr, tma_barrier_phase_bit);
      tma_barrier_phase_bit ^= 1;

      if (elect_one_warp) {
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
          gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(mma_barrier_ptr);
      }

      cute::wait_barrier(*mma_barrier_ptr, mma_barrier_phase_bit);
      mma_barrier_phase_bit ^= 1;
    }
  }

  auto tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  auto thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

  // Host-overhead fix: every call site hardcodes Alpha(1.0f), Beta(0.0f), so
  // the C operand is never observable in the output. Drop the global->register
  // load of C and the beta multiply (bit-identical: C was zero-filled before).
  (void)beta;

  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
  Tensor tDgD = thr_t2r_copy.partition_D(tCgD);
  Tensor tDrAcc = make_tensor<Accumulator>(shape(tDgD));
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);

  // fp16-direct epilogue: convert the fp32 accumulator to fp16 in registers
  // and store straight to the fp16 D buffer. This is the same single
  // round-to-nearest-even conversion of the same fp32 value that the old
  // fp32-store + host .to(kFloat16) path performed, so the output is
  // bit-identical while the separate cast kernel (and its host dispatch)
  // disappears and the D store traffic halves.
  using DType = typename DTensor::value_type;
  Tensor tDrD = make_tensor<DType>(shape(tDgD));
  CUTE_UNROLL
  for (int i = 0; i < size(tDrD); ++i) {
    tDrD(i) = static_cast<DType>(alpha * tDrAcc(i));
  }
  copy(tDrD, tDgD);

  __syncthreads();
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base,
                        TmemAllocator::Sm100TmemCapacityColumns);
  }
}

template <class Variant>
torch::Tensor run_tcgen05_variant(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "expected 2D inputs");
  TORCH_CHECK(a.size(1) == b.size(1), "incompatible matmul shapes");
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16,
              "tcgen05 kernels expect float16 inputs");
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "tensors must reside on CUDA");

  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();

  auto m = a_contig.size(0);
  auto k = a_contig.size(1);
  auto n = b_contig.size(0);

  // Host-overhead fix: the device epilogue no longer reads C (alpha=1, beta=0
  // at every call site), so no zero-filled C operand is needed. fp16-direct
  // epilogue: the kernel writes fp16 D, so allocate the output at its final
  // dtype and return it without a .to(kFloat16) pass.
  auto d_buffer = torch::empty({m, n}, a.options().dtype(torch::kFloat16));

  auto cluster_shape = Variant::cluster_shape();
  auto tiled_mma = make_tiled_mma(typename Variant::Mma{});

  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);

  TORCH_CHECK(evenly_divides(shape(mma_tiler), tile_shape(tiled_mma)),
              "tcgen05 MMA tile must divide instruction tile");
  TORCH_CHECK(
      evenly_divides(make_shape(m, n, k), mma_tiler),
      "Problem size must be divisible by the tcgen05 tile (128x256x64 or "
      "256x256x64 depending on the variant)");

  auto mma_shape_A =
      partition_shape_A(tiled_mma,
                        make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B =
      partition_shape_B(tiled_mma,
                        make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  auto sA_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
  auto sB_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);

  using SharedStorageT =
      SharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout)>;

  Tensor mA = make_tensor(
      make_gmem_ptr(reinterpret_cast<TypeA const*>(
          a_contig.data_ptr<at::Half>())),
      make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
  Tensor mB = make_tensor(
      make_gmem_ptr(reinterpret_cast<TypeB const*>(
          b_contig.data_ptr<at::Half>())),
      make_layout(make_shape(n, k), make_stride(k, Int<1>{})));
  // mC is layout-plumbing only: the epilogue dropped the C load/store, so this
  // pointer is never dereferenced (it exists to type partition_C /
  // make_fragment_C with the fp32 accumulator type). Point it at the D
  // allocation rather than allocating dead fp32 memory.
  Tensor mC = make_tensor(
      make_gmem_ptr(reinterpret_cast<TypeC*>(d_buffer.data_ptr<at::Half>())),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
  Tensor mD = make_tensor(
      make_gmem_ptr(reinterpret_cast<TypeD*>(d_buffer.data_ptr<at::Half>())),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));

  auto cluster_layout_vmnk =
      tiled_divide(make_layout(cluster_shape),
                   make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  // Build TMA atoms. 2SM variants must use SM100 TMA atoms + multicast semantics.
  auto build_tma_atom_A = [&] {
    if constexpr (Variant::kClusterM == 2) {
      return make_tma_atom_A_sm100(
          SM100_TMA_2SM_LOAD_MULTICAST{},
          mA,
          sA_layout,
          mma_tiler,
          tiled_mma,
          cluster_layout_vmnk);
    } else {
      return make_tma_atom(
          SM90_TMA_LOAD{},
          mA,
          sA_layout,
          make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
    }
  };

  auto build_tma_atom_B = [&] {
    if constexpr (Variant::kClusterM == 2) {
      return make_tma_atom_B_sm100(
          SM100_TMA_2SM_LOAD_MULTICAST{},
          mB,
          sB_layout,
          mma_tiler,
          tiled_mma,
          cluster_layout_vmnk);
    } else {
      return make_tma_atom(
          SM90_TMA_LOAD{},
          mB,
          sB_layout,
          make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));
    }
  };

  // Host-overhead fix: cache the TMA atoms across calls. A CUtensorMap encodes
  // only the base address, shape, and layout of the operand -- never tensor
  // contents -- so a rebuild is needed only when the data pointers, problem
  // shape, or device change. Cache is per template instantiation per thread;
  // intentionally leaked (the atoms hold no CUDA resources requiring teardown).
  struct TmaAtomCacheKey {
    const void* a_ptr;
    const void* b_ptr;
    int64_t m, n, k;
    int device;
    bool matches(const void* ap, const void* bp, int64_t mm, int64_t nn,
                 int64_t kk, int dev) const {
      return a_ptr == ap && b_ptr == bp && m == mm && n == nn && k == kk &&
             device == dev;
    }
  };
  using TmaAtomPair = std::pair<decltype(build_tma_atom_A()),
                                decltype(build_tma_atom_B())>;
  struct TmaAtomCache {
    TmaAtomCacheKey key;
    TmaAtomPair atoms;
  };
  static thread_local TmaAtomCache* tma_cache = nullptr;
  const void* a_ptr = a_contig.data_ptr();
  const void* b_ptr = b_contig.data_ptr();
  const int device = static_cast<int>(a_contig.get_device());
  if (tma_cache == nullptr ||
      !tma_cache->key.matches(a_ptr, b_ptr, m, n, k, device)) {
    auto* fresh = new TmaAtomCache{
        TmaAtomCacheKey{a_ptr, b_ptr, m, n, k, device},
        TmaAtomPair{build_tma_atom_A(), build_tma_atom_B()}};
    delete tma_cache;
    tma_cache = fresh;
  }
  // Copy (not reference): keeps decltype(tma_atom_A) a value type for the
  // kernel template instantiation below. The atom is a small trivially
  // copyable descriptor; the expensive part is the build, not the copy.
  auto tma_atom_A = tma_cache->atoms.first;
  auto tma_atom_B = tma_cache->atoms.second;

  auto cluster_m_tiles = size<1>(cluster_layout_vmnk);
  auto cluster_n_tiles = size<2>(cluster_layout_vmnk);

  int tile_m = size(bM) * cluster_m_tiles;
  int tile_n = size(bN) * cluster_n_tiles;

  dim3 dimBlock(128);
  dim3 dimCluster(Variant::kClusterM,
                  Variant::kClusterN,
                  Variant::kClusterK);
  dim3 dimGrid(
      (m + tile_m - 1) / tile_m * dimCluster.x,
      (n + tile_n - 1) / tile_n * dimCluster.y,
      1);

  int smem_bytes = sizeof(SharedStorageT);

  auto *kernel_ptr = &gemm_device_variant<
      Variant,
      SharedStorageT,
      decltype(mA), decltype(mB),
      decltype(mC), decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma),
      decltype(cluster_shape),
      decltype(tma_atom_A), decltype(tma_atom_B),
      Alpha, Beta>;

  // Host-overhead fix: the max-dynamic-smem attribute is sticky per
  // (function, device). Set it once per device instead of on every call.
  static thread_local int attr_set_for_device = -1;
  if (attr_set_for_device != device) {
    AT_CUDA_CHECK(cudaFuncSetAttribute(
        kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    attr_set_for_device = device;
  }

  cutlass::ClusterLaunchParams params{
      dimGrid, dimBlock, dimCluster, smem_bytes};
  params.cuda_stream = at::cuda::getCurrentCUDAStream();

  auto status = cutlass::launch_kernel_on_cluster(
      params, (void const*)kernel_ptr,
      mA, mB, mC, mD,
      mma_tiler, tiled_mma, cluster_shape,
      tma_atom_A, tma_atom_B,
      Alpha(1.0f), Beta(0.0f));
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "tcgen05 inline kernel launch failed");

  // fp16-direct epilogue: D is already fp16; no conversion pass.
  return d_buffer;
}

torch::Tensor optimized_matmul_tcgen05(torch::Tensor a, torch::Tensor b) {
  return run_tcgen05_variant<VariantCTA1>(a, b);
}

torch::Tensor optimized_matmul_tcgen05_cta2(torch::Tensor a, torch::Tensor b) {
  return run_tcgen05_variant<VariantCTA2>(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("optimized_matmul_tcgen05", &optimized_matmul_tcgen05);
  m.def("optimized_matmul_tcgen05_cta2", &optimized_matmul_tcgen05_cta2);
}

TORCH_LIBRARY(capstone_tcgen05, m) {
  m.def("optimized_matmul_tcgen05(Tensor a, Tensor b) -> Tensor");
  m.def("optimized_matmul_tcgen05_cta2(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(capstone_tcgen05, CUDA, m) {
  m.impl("optimized_matmul_tcgen05", optimized_matmul_tcgen05);
  m.impl("optimized_matmul_tcgen05_cta2", optimized_matmul_tcgen05_cta2);
}
