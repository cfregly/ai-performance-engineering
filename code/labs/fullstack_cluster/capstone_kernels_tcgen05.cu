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

// k-stage pipelining (D4): the smem layouts carry a trailing PIPE mode
// (CUTLASS sm100 collective pattern: UMMA::tile_to_mma_shape over
// append(mma_shape, Int<Stages>)), so A/B below are Stages-deep ring buffers
// and each stage gets its own TMA mbarrier (expect-tx protocol).
template <class TypeA_, class TypeB_, class ASmemLayout, class BSmemLayout,
          int Stages>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA_, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB_, cute::cosize_v<BSmemLayout>> B;

  alignas(16) cute::uint64_t mma_barrier[Stages];
  alignas(16) cute::uint64_t tma_barrier[Stages];
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

// Pipeline depth. Per-stage smem cost at the 128x256x64 (CTA1, per-CTA) tile
// is 48KB (A 16KB + B 32KB), so 4 stages = 192KB, inside the 227KB/block
// opt-in limit. Occupancy is unaffected: the TMEM allocator already grabs the
// full 512-column capacity, pinning every variant to 1 CTA/SM regardless of
// smem footprint.
constexpr int kPipelineStages = 2;

template <class MmaTag, int ClusterM, int Stages>
struct TcgenVariantConfig {
  using Mma = MmaTag;
  static constexpr int kClusterM = ClusterM;
  static constexpr int kClusterN = 1;
  static constexpr int kClusterK = 1;
  static constexpr int kStages = Stages;
  static_assert(Stages >= 2, "pipeline needs at least double buffering");
  using TmemAllocator = std::conditional_t<
      ClusterM == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

  static constexpr auto cluster_shape() {
    return make_shape(Int<ClusterM>{}, Int<1>{}, Int<1>{});
  }
};

using VariantCTA1 = TcgenVariantConfig<
    SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256,
                         UMMA::Major::K, UMMA::Major::K>,
    1, kPipelineStages>;

using VariantCTA2 = TcgenVariantConfig<
    SM100_MMA_F16BF16_2x1SM_SS<TypeA, TypeB, TypeC, 256, 256,
                               UMMA::Major::K, UMMA::Major::K>,
    2, kPipelineStages>;

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

  constexpr int kStages = Variant::kStages;
  uint64_t* mma_barrier = shared_storage.mma_barrier;
  uint64_t* tma_barrier = shared_storage.tma_barrier;

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
  // Warp specialization (D4): warp 1's elected lane is the TMA producer;
  // warp 0 issues the UMMAs. Splitting the roles takes the producer's
  // mbarrier waits off the MMA warp's issue chain.
  uint32_t producer_warp = (threadIdx.x / 32 == 1);

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

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  // k-stage pipelined mainloop (both branches), CUTLASS sm100
  // producer/consumer protocol. Stage s of the smem ring holds k-tile
  // (k % kStages); per-stage tma barriers signal "stage filled" (expect-tx)
  // and per-stage mma barriers signal "stage drained" (tcgen05 commit /
  // umma_arrive after the k-tile's UMMAs). The prologue fills stages
  // 0..kStages-2; iteration k refills the stage iteration k-1 read with
  // k-tile k+kStages-1, gated only on that stage's mma barrier -- so UMMAs of
  // consecutive k-tiles issue back-to-back into the in-order tensor pipe
  // (no full-drain wait per iteration). The k-tile MMA issue ORDER is
  // unchanged vs the serial loop, so outputs stay bit-identical; only the
  // residence of the loads changes. The epilogue waits the final k-tile's
  // mma barrier (commit covers all previously issued UMMAs) before t2r.
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
        size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tAsA(_, Int<0>{}))) +
        size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tBsB(_, Int<0>{})));

    if (elect_one_warp && elect_one_thr) {
      int num_mcast_participants =
          size<1>(cluster_layout_vmnk) + size<2>(cluster_layout_vmnk) - 1;
      for (int s = 0; s < kStages; ++s) {
        cute::initialize_barrier(mma_barrier[s], num_mcast_participants);
        cute::initialize_barrier(tma_barrier[s], 1);
      }
    }
    cute::cluster_sync();

    int num_k_tiles = size<3>(tCgA);

    if (producer_warp && elect_one_thr) {
      // Producer (warp 1, one lane, BOTH CTAs of the pair): fill the smem
      // ring tile after tile, gated only on the local mma commit for the
      // stage being reused (the multicast commit signals both CTAs'
      // barriers). Only the leader CTA arms expect-tx (2SM TMA semantics,
      // unchanged); both CTAs issue their copies.
      int stage = 0;
      int gate_phase = 0;
      int gate_count = 0;
      for (int p = 0; p < num_k_tiles; ++p) {
        if (p >= kStages) {
          cute::wait_barrier(mma_barrier[stage], gate_phase);
          if (++gate_count == kStages) {
            gate_count = 0;
            gate_phase ^= 1;
          }
        }
        if (elect_one_cta) {
          cute::set_barrier_transaction_bytes(tma_barrier[stage],
                                              tma_transaction_bytes);
        }
        copy(tma_atom_A.with(tma_barrier[stage], tma_mcast_mask_a),
             tAgA(_, p), tAsA(_, stage));
        copy(tma_atom_B.with(tma_barrier[stage], tma_mcast_mask_b),
             tBgB(_, p), tBsB(_, stage));
        ++stage;
        if (stage == kStages) {
          stage = 0;
        }
      }
    } else {
      // Consumer + phase-locking companions. Leader-CTA threads wait the
      // tma barriers (their warp 0 issues the 2SM UMMAs and the multicast
      // commit); peer-CTA threads have no armed tma barriers, so they
      // phase-lock on the local mma commits instead -- otherwise their
      // final drain parity could alias an earlier same-parity round.
      int read_stage = 0;
      int tma_read_phase = 0;
      for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        if (elect_one_cta) {
          cute::wait_barrier(tma_barrier[read_stage], tma_read_phase);
          if (elect_one_warp) {
            for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
              gemm(tiled_mma, tCrA(_, _, k_block, read_stage),
                   tCrB(_, _, k_block, read_stage), tCtAcc);
              tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive_multicast_2x1SM(&mma_barrier[read_stage],
                                                       mma_mcast_mask_c);
          }
        } else {
          cute::wait_barrier(mma_barrier[read_stage], tma_read_phase);
        }

        ++read_stage;
        if (read_stage == kStages) {
          read_stage = 0;
          tma_read_phase ^= 1;
        }
      }
    }

    // Drain: wait for the last k-tile's commit (covers all prior UMMAs)
    // before the TMEM->register epilogue. Both CTAs of the pair wait (the
    // multicast commit signals both barriers); parity cannot alias because
    // every thread consumed (or gated on) round floor((num-1)/S)-1 or later
    // in the loop above.
    {
      int last_stage = (num_k_tiles - 1) % kStages;
      int last_phase = ((num_k_tiles - 1) / kStages) & 1;
      cute::wait_barrier(mma_barrier[last_stage], last_phase);
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
        sizeof(make_tensor_like(tAsA(_, Int<0>{}))) +
        sizeof(make_tensor_like(tBsB(_, Int<0>{})));

    if (elect_one_warp && elect_one_thr) {
      for (int s = 0; s < kStages; ++s) {
        cute::initialize_barrier(mma_barrier[s], 1);
        cute::initialize_barrier(tma_barrier[s], 1);
      }
    }
    __syncthreads();

    int num_k_tiles = size<3>(tCgA);

    if (producer_warp && elect_one_thr) {
      // Producer (warp 1, one lane): fill the smem ring tile after tile,
      // gated only on the consumer's commit for the stage being reused
      // (tiles 0..kStages-1 hit fresh stages, no gate). Running this loop on
      // its own warp takes both mbarrier wake-ups off the MMA warp's issue
      // chain: by the time the consumer waits a tma barrier the phase has
      // already flipped, so its try_wait falls through.
      int stage = 0;
      int gate_phase = 0;
      int gate_count = 0;
      for (int p = 0; p < num_k_tiles; ++p) {
        if (p >= kStages) {
          cute::wait_barrier(mma_barrier[stage], gate_phase);
          if (++gate_count == kStages) {
            gate_count = 0;
            gate_phase ^= 1;
          }
        }
        cute::set_barrier_transaction_bytes(tma_barrier[stage],
                                            tma_transaction_bytes);
        copy(tma_atom_A.with(tma_barrier[stage]), tAgA(_, p), tAsA(_, stage));
        copy(tma_atom_B.with(tma_barrier[stage]), tBgB(_, p), tBsB(_, stage));
        ++stage;
        if (stage == kStages) {
          stage = 0;
        }
      }
    } else {
      // Consumer (warp 0) + phase-locking companions (warps 2-3 and the
      // non-elect lanes of warp 1): wait each tile's TMA in round order,
      // issue its UMMAs, commit to the stage's mma barrier. No drain wait in
      // the loop -- the tensor pipe is fed back-to-back across k-tiles. The
      // k-tile MMA issue ORDER is unchanged vs the serial loop.
      int read_stage = 0;
      int tma_read_phase = 0;
      for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        cute::wait_barrier(tma_barrier[read_stage], tma_read_phase);

        if (elect_one_warp) {
          for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            gemm(tiled_mma, tCrA(_, _, k_block, read_stage),
                 tCrB(_, _, k_block, read_stage), tCtAcc);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }
          cutlass::arch::umma_arrive(&mma_barrier[read_stage]);
        }

        ++read_stage;
        if (read_stage == kStages) {
          read_stage = 0;
          tma_read_phase ^= 1;
        }
      }
    }

    // Drain: wait for the last k-tile's commit (covers all prior UMMAs)
    // before the TMEM->register epilogue. Parity is safe for every thread:
    // consumers/companions consumed tma round r* in-loop (which required the
    // producer gate on mma round r*-1), and the producer's last gate was
    // itself round r*-1, so nobody can alias an earlier same-parity round.
    {
      int last_stage = (num_k_tiles - 1) % kStages;
      int last_phase = ((num_k_tiles - 1) / kStages) & 1;
      cute::wait_barrier(mma_barrier[last_stage], last_phase);
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

  // k-stage pipelining: append the PIPE mode to the smem layouts (CUTLASS
  // sm100 collective pattern). The TMA atoms are built against the stage-0
  // slice, which is identical to the pre-pipeline single-stage layout.
  constexpr int kStages = Variant::kStages;
  auto sA_layout = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<TypeA>{},
      append(mma_shape_A, Int<kStages>{}));
  auto sB_layout = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<TypeB>{},
      append(mma_shape_B, Int<kStages>{}));
  auto sA_layout_stage0 = sA_layout(_, _, _, Int<0>{});
  auto sB_layout_stage0 = sB_layout(_, _, _, Int<0>{});

  using SharedStorageT =
      SharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout),
                    kStages>;

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
          sA_layout_stage0,
          mma_tiler,
          tiled_mma,
          cluster_layout_vmnk);
    } else {
      return make_tma_atom(
          SM90_TMA_LOAD{},
          mA,
          sA_layout_stage0,
          make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
    }
  };

  auto build_tma_atom_B = [&] {
    if constexpr (Variant::kClusterM == 2) {
      return make_tma_atom_B_sm100(
          SM100_TMA_2SM_LOAD_MULTICAST{},
          mB,
          sB_layout_stage0,
          mma_tiler,
          tiled_mma,
          cluster_layout_vmnk);
    } else {
      return make_tma_atom(
          SM90_TMA_LOAD{},
          mB,
          sB_layout_stage0,
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
  // sm_100/sm_103 opt-in dynamic smem ceiling per block is 227KB (232448B).
  static_assert(sizeof(SharedStorageT) <= 232448,
                "pipeline stages exceed the 227KB smem/block limit");

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
