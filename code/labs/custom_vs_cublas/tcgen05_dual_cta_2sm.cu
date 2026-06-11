/**
 * 2-SM UMMA Pair Variant (tcgen05 cta_group::2) on the dual-CTA footprint
 * =======================================================================
 *
 * Why (B43/B44 + E5 ncu): the measured-best plain dual-CTA (256,2) kernel is
 * TMA-LATENCY/ISSUE-bound, not bandwidth-bound (DRAM 12.5% of peak,
 * long_scoreboard ~73% of stalls at 8 active warps/SM). Both B-multicast
 * (E4/E5 honest negative) and smem-feasible stage-deepening (stage 3 needs
 * 144KB > 110KB cap) are exhausted. The remaining structural lever is to
 * spend the cluster on what GB300 accelerates: fuse the CTA pair into ONE
 * 256-wide MMA via SM100_MMA_F16BF16_2x1SM_SS (tcgen05.mma.cta_group::2).
 *
 * What changes vs tcgen05_dual_cta.cu (which stays intact and selectable):
 *   - Cluster (2,1,1): two vertically-adjacent M-tiles form an SM pair.
 *     Pair tile = 256 x kTileN; per CTA output is still 128 x kTileN.
 *   - ONE tcgen05.mma.cta_group::2 instruction per (pair, k-block) computes
 *     what previously took two cta_group::1 instructions: instruction and
 *     barrier traffic per fed byte HALVES.
 *   - Only the EVEN cluster rank (leader) issues MMAs and waits full
 *     barriers. The odd CTA is a pure TMA producer: its loop free-runs
 *     ahead (bounded by the empty barriers), so one CTA's TMA latency is
 *     covered by the pair's in-flight stages -- whole-CTA role assignment
 *     (and warp-uniform waits, per the B44 ncu-replay trap).
 *   - Operand residency per the 2x1SM atom traits (MEASURED on GB300 via
 *     ncu smem footprints + mbarrier tx-byte balance; note the cute
 *     tutorial 04's printed comments claiming full-N B per CTA are stale):
 *       A: each CTA's smem holds ITS 128-row M-half        (16KB/stage)
 *       B: each CTA's smem holds ITS kTileN/2 column half  (split, not dup)
 *       C: each CTA's TMEM holds its 128 rows x kTileN     (kTileN cols)
 *     So per-CTA B bytes/stage HALVE vs the plain dual config: the lever
 *     attacks issue rate, pipeline structure, AND per-CTA feed traffic.
 *   - TMA: SM100_TMA_2SM_LOAD (cp.async.bulk.tensor.cta_group::2). BOTH
 *     CTAs issue their own loads into their own smem with their OWN
 *     full_barrier[stage] handle; the hardware redirects every mbarrier
 *     arrival to the EVEN CTA's barrier (Sm100MmaPeerBitMask), so the
 *     leader's barrier counts the PAIR's bytes 2 x (A-half + B-half) =
 *     one full A tile + one full B tile per stage.
 *     Only the leader runs set_barrier_transaction_bytes; a peer TMA
 *     completing before the leader's expect_tx is legal (mbarrier tx-count
 *     may go transiently negative).
 *   - Stage drain: the leader's tcgen05.commit is MULTICAST to both pair
 *     CTAs' empty_barrier[stage] (umma_arrive_multicast_2x1SM, mask 0b11,
 *     init count 1); each CTA waits its own empty barrier before reissuing
 *     TMA into a stage. Final commit is multicast to mma_barrier so the
 *     peer's epilogue sees its TMEM half complete.
 *   - TMEM: cute::TMEM::Allocator2Sm (tcgen05.alloc.cta_group::2) --
 *     pair-collective, allocates the SAME kAccTmemCols base in both SMs'
 *     TMEM; the allocation permit is released IMMEDIATELY so a co-resident
 *     CTA (from another pair) can allocate -> 2 CTAs/SM stays possible
 *     (2 x 256 = 512 cols at kTileN=256).
 *
 * Tunables (compile-time, see tcgen05_loader.py):
 *   -DDUAL2SM_TILE_N=128|256  (per-CTA TMEM cols = TILE_N; pair M fixed 256)
 *   -DDUAL2SM_STAGES=2|3|4    (per-CTA smem 32KB/stage at N=256,
 *                              24KB/stage at N=128; 110KB static cap.
 *                              MEASURED BEST: (128,3) = 72KB -> with 152
 *                              regs/thread the epilogue at N=128 unlocks
 *                              Block Limits 3/3 -> THREE CTAs/SM, TMEM
 *                              3x128 of 512 cols)
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
#include <cute/atom/copy_traits_sm100_tma.hpp>  // make_tma_atom_A/B_sm100, SM100_TMA_2SM_LOAD
#include <cute/arch/cluster_sm90.hpp>           // elect_one_sync, block_rank_in_cluster, cluster_sync

#ifndef DUAL2SM_TILE_N
#define DUAL2SM_TILE_N 256
#endif
#ifndef DUAL2SM_STAGES
#define DUAL2SM_STAGES 2
#endif

using namespace cute;

namespace dual_cta_2sm_impl {

using TypeA = cutlass::half_t;
using TypeB = cutlass::half_t;
using TypeC = float;
using TypeD = float;
using Accumulator = float;

constexpr int kTileM = 256;  // pair-wide M: 128 rows per CTA of the pair
constexpr int kTileN = DUAL2SM_TILE_N;
constexpr int kStages = DUAL2SM_STAGES;

static_assert(kStages >= 2 && kStages <= 4, "DUAL2SM_STAGES must be 2..4");
static_assert(kTileN >= 32 && (kTileN & (kTileN - 1)) == 0 && kTileN <= 256,
              "DUAL2SM_TILE_N must be a power of two in [32, 256]");

// fp32 accumulator: each CTA holds its 128-row M-half x kTileN in TMEM.
constexpr int kAccTmemCols = kTileN;
static_assert(2 * kAccTmemCols <= cute::TMEM::Allocator2Sm::Sm100TmemCapacityColumns,
              "Accumulator must leave TMEM room for a co-resident CTA");

template <class TypeA_, class TypeB_, class ASmemLayout, class BSmemLayout>
struct Dual2SmSharedStorage {
  alignas(128) cute::ArrayEngine<TypeA_, cute::cosize_v<ASmemLayout>> smem_A[kStages];
  alignas(128) cute::ArrayEngine<TypeB_, cute::cosize_v<BSmemLayout>> smem_B[kStages];
  alignas(16) cute::uint64_t full_barrier[kStages];   // leader-only: pair TMA bytes
  alignas(16) cute::uint64_t empty_barrier[kStages];  // per-CTA: multicast commit
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
    SM100_MMA_F16BF16_2x1SM_SS<TypeA, TypeB, TypeC, kTileM, kTileN,
                               UMMA::Major::K, UMMA::Major::K>;

template <class SharedStorageT,
          class ATensor, class BTensor, class CTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA,
          class TmaAtomA, class TmaAtomB>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 2)
gemm_dual_cta_2sm(ATensor mA, BTensor mB, CTensor mC, DTensor mD,
                  MmaTiler_MNK mma_tiler, TiledMMA tiled_mma,
                  CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
                  CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B) {

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  int v = int(cute::block_rank_in_cluster()) & 1;  // peer rank within the SM pair
  bool leader_cta = (v == 0);                      // even rank issues the 2SM MMA

  int tile_m = blockIdx.x / 2;  // pair-wide 256-row M tile
  int tile_n = blockIdx.y;

  auto mma_coord = make_coord(tile_m, tile_n, _);

  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1, _1, X>{});
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  SharedStorageT& storage = *reinterpret_cast<SharedStorageT*>(shared_memory);

  // Per-CTA partitioning via the 2x1SM atom's ThrID: v slices the pair.
  auto cta_mma = tiled_mma.get_slice(v);
  Tensor tCgA = cta_mma.partition_A(gA);  // this CTA's 128-row A half
  Tensor tCgB = cta_mma.partition_B(gB);  // this CTA's kTileN/2-column B half
  Tensor tCgC = cta_mma.partition_C(gC);  // this CTA's 128 x kTileN output
  Tensor tCgD = cta_mma.partition_C(gD);

  // ---------------------------------------------------------------------
  // TMEM: pair-collective allocation (tcgen05.alloc.cta_group::2). Both
  // CTAs' warp 0 issue it with the same smem dst offset; the permit is
  // released IMMEDIATELY so a co-resident CTA from another pair can
  // allocate (2 CTAs/SM is the point of the footprint).
  // ---------------------------------------------------------------------
  cute::TMEM::Allocator2Sm tmem_allocator{};
  if (elect_one_warp) {
    tmem_allocator.allocate(kAccTmemCols, &storage.tmem_base_ptr);
    tmem_allocator.release_allocation_lock();
  }

  if (elect_one_warp && elect_one_thr) {
    for (int s = 0; s < kStages; ++s) {
      // full: the leader's expect_tx arrive is the single arrival; both
      // CTAs' TMA bytes redirect-complete into the LEADER's barrier.
      cute::initialize_barrier(storage.full_barrier[s], 1);
      // empty: one multicast tcgen05.commit arrival from the leader.
      cute::initialize_barrier(storage.empty_barrier[s], 1);
    }
    cute::initialize_barrier(storage.mma_barrier, 1);
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  // Peer TMA arrivals target the leader's barriers and the leader's commit
  // multicast targets the peer's: every cluster CTA must observe init first.
  cute::cluster_sync();

  uint32_t tmem_base = storage.tmem_base_ptr;

  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);  // tmem_frg_2sm
  tCtAcc.data() = tmem_base;

  // TMA coordinate tensors
  Tensor tma_coord_A = tma_atom_A.get_tma_tensor(shape(mA));
  Tensor tma_coord_B = tma_atom_B.get_tma_tensor(shape(mB));
  Tensor gCoordA = local_tile(tma_coord_A, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gCoordB = local_tile(tma_coord_B, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor tCgCoordA = cta_mma.partition_A(gCoordA);
  Tensor tCgCoordB = cta_mma.partition_B(gCoordB);

  // Non-multicast partition (coord 0 / layout 1): the V-mode pair split is
  // already baked into tCgCoordA/B by partition_A/B above, so per-stage
  // smem tensors may be rebuilt by base-pointer swap (no iterator offset).
  Tensor tCsA_0 = storage.tensor_sA(0);
  auto [tAgA, tAsA_0] = tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsA_0), group_modes<0,3>(tCgCoordA));
  Tensor tCsB_0 = storage.tensor_sB(0);
  auto [tBgB, tBsB_0] = tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsB_0), group_modes<0,3>(tCgCoordB));

  // The leader's full barrier sees the PAIR's bytes: 2 x (A-half + B-half)
  // = one full A tile + one full B tile per stage.
  int tma_bytes = 2 * (sizeof(make_tensor_like(tAsA_0)) + sizeof(make_tensor_like(tBsB_0)));

  auto issue_tma = [&](int stage, int k_tile) {
    if (leader_cta) {
      cute::set_barrier_transaction_bytes(storage.full_barrier[stage], tma_bytes);
    }
    auto tAsA = make_tensor(make_smem_ptr(storage.smem_A[stage].begin()), tAsA_0.layout());
    copy(tma_atom_A.with(storage.full_barrier[stage]), tAgA(_, k_tile), tAsA);
    auto tBsB = make_tensor(make_smem_ptr(storage.smem_B[stage].begin()), tBsB_0.layout());
    copy(tma_atom_B.with(storage.full_barrier[stage]), tBgB(_, k_tile), tBsB);
  };

  // Per-stage UMMA smem-descriptor fragments (same type per stage -> array).
  auto fragA = [&](int s) { return cta_mma.make_fragment_A(storage.tensor_sA(s)); };
  auto fragB = [&](int s) { return cta_mma.make_fragment_B(storage.tensor_sB(s)); };
#if DUAL2SM_STAGES == 2
  decltype(fragA(0)) tCrA_st[kStages] = { fragA(0), fragA(1) };
  decltype(fragB(0)) tCrB_st[kStages] = { fragB(0), fragB(1) };
#elif DUAL2SM_STAGES == 3
  decltype(fragA(0)) tCrA_st[kStages] = { fragA(0), fragA(1), fragA(2) };
  decltype(fragB(0)) tCrB_st[kStages] = { fragB(0), fragB(1), fragB(2) };
#else
  decltype(fragA(0)) tCrA_st[kStages] = { fragA(0), fragA(1), fragA(2), fragA(3) };
  decltype(fragB(0)) tCrB_st[kStages] = { fragB(0), fragB(1), fragB(2), fragB(3) };
#endif

  int num_k_tiles = size<3>(tCgA);

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  constexpr uint16_t kPairMask = 0x3;  // both CTAs of the (2,1,1) cluster

  // =====================================================================
  // Mainloop: warp 0 of BOTH CTAs. The leader consumes (full-wait + MMA +
  // commit); the peer only produces (TMA), throttled by its empty
  // barriers. Warps 1-3 park on mma_barrier until the epilogue.
  // =====================================================================
  if (elect_one_warp) {
    int full_phase[kStages] = {};
    int empty_phase[kStages] = {};

    // Prologue: fill kStages-1 stages (both CTAs issue their own loads).
    if (elect_one_thr) {
      for (int i = 0; i < min(kStages - 1, num_k_tiles); ++i) {
        issue_tma(i, i);
      }
    }

    for (int k = 0; k < num_k_tiles; ++k) {
      int curr = k % kStages;
      int next_k = k + (kStages - 1);

      // Refill ahead of the consumer. Reusing a stage requires the pair
      // MMA to have drained it (empty barrier <- multicast tcgen05.commit).
      if (next_k < num_k_tiles) {
        int next_s = next_k % kStages;
        if (next_k >= kStages) {
          cute::wait_barrier(storage.empty_barrier[next_s], empty_phase[next_s]);
          empty_phase[next_s] ^= 1;
        }
        if (elect_one_thr) {
          issue_tma(next_s, next_k);
        }
      }

      // Consume current stage: leader only (the single 2SM MMA issuer).
      if (leader_cta) {
        cute::wait_barrier(storage.full_barrier[curr], full_phase[curr]);
        full_phase[curr] ^= 1;

        for (int kb = 0; kb < size<2>(tCrA_st[0]); ++kb) {
          gemm(tiled_mma, tCrA_st[curr](_, _, kb), tCrB_st[curr](_, _, kb), tCtAcc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }

        // tcgen05.commit.cta_group::2 multicast: BOTH pair CTAs' stage-curr
        // smem was read by the pair MMA; release both empty barriers.
        cutlass::arch::umma_arrive_multicast_2x1SM(
            reinterpret_cast<uint64_t*>(&storage.empty_barrier[curr]), kPairMask);
      }
    }

    // Drain the final multicast commits on BOTH CTAs: every commit the
    // leader multicast at the peer must be DELIVERED before the peer's
    // smem barriers die with it (and vice versa for the leader's own copy).
    for (int k = num_k_tiles - min(kStages, num_k_tiles); k < num_k_tiles; ++k) {
      int s = k % kStages;
      cute::wait_barrier(storage.empty_barrier[s], empty_phase[s]);
      empty_phase[s] ^= 1;
    }

    // Final commit: multicast so the PEER also learns its TMEM half is
    // complete (the pair MMA writes both CTAs' TMEM).
    if (leader_cta) {
      cutlass::arch::umma_arrive_multicast_2x1SM(
          reinterpret_cast<uint64_t*>(&storage.mma_barrier), kPairMask);
    }
  }

  // All 128 threads of both CTAs rendezvous here for the epilogue.
  cute::wait_barrier(storage.mma_barrier, 0);

  // Epilogue: TMEM -> registers -> gmem (beta=0: C never read).
  auto tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  auto thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
  Tensor tDgD = thr_t2r_copy.partition_D(tCgD);
  Tensor tDrAcc = make_tensor<Accumulator>(shape(tDgD));
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);
  cutlass::arch::fence_view_async_tmem_load();

  copy(tDrAcc, tDgD);  // D = accumulator (beta=0)

  __syncthreads();
  if (elect_one_warp) {
    // Pair-collective dealloc (tcgen05.dealloc.cta_group::2): implicit
    // pair rendezvous; both CTAs' warp 0 issue it.
    tmem_allocator.free(tmem_base, kAccTmemCols);
  }
}

torch::Tensor run_dual_cta_2sm_matmul(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2);
  TORCH_CHECK(a.size(1) == b.size(1));
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16);
  TORCH_CHECK(a.is_cuda() && b.is_cuda());

  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();
  auto m = a_contig.size(0);
  auto k = a_contig.size(1);
  auto n = b_contig.size(0);

  TORCH_CHECK(m % kTileM == 0 && n % kTileN == 0 && k % 64 == 0,
              "Size must be divisible by the 2SM pair tile (", kTileM, "x", kTileN, "x64)");

  auto options = a.options().dtype(torch::kFloat32);
  auto c_buffer = torch::empty({m, n}, options);  // beta=0: never read
  auto d_buffer = torch::empty_like(c_buffer);

  auto tiled_mma = make_tiled_mma(MmaTag{});
  auto bM = tile_size<0>(tiled_mma);              // 256 (pair-wide)
  auto bN = tile_size<1>(tiled_mma);              // kTileN
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};   // 64
  auto mma_tiler = make_shape(bM, bN, bK);

  // Post-partitioned per-CTA shapes: A = 128-row half, B = FULL kTileN.
  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);

  using SharedStorageT = Dual2SmSharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout)>;

  // 2 CTAs/SM requires per-CTA smem to fit twice in the 227KB/SM budget.
  static_assert(sizeof(SharedStorageT) <= 110 * 1024,
                "Shared storage too large for 2 CTAs/SM; reduce DUAL2SM_STAGES or DUAL2SM_TILE_N");

  Tensor mA = make_tensor(make_gmem_ptr(reinterpret_cast<TypeA const*>(a_contig.data_ptr<at::Half>())),
      make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
  Tensor mB = make_tensor(make_gmem_ptr(reinterpret_cast<TypeB const*>(b_contig.data_ptr<at::Half>())),
      make_layout(make_shape(n, k), make_stride(k, Int<1>{})));
  Tensor mC = make_tensor(make_gmem_ptr(c_buffer.data_ptr<TypeC>()),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
  Tensor mD = make_tensor(make_gmem_ptr(d_buffer.data_ptr<TypeD>()),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));

  // Cluster (2,1,1): the V mode is the SM pair (AtomThrID = 2).
  auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  auto tma_atom_A = make_tma_atom_A_sm100(SM100_TMA_2SM_LOAD{}, mA, sA_layout,
                                          mma_tiler, tiled_mma, cluster_layout_vmnk);
  auto tma_atom_B = make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD{}, mB, sB_layout,
                                          mma_tiler, tiled_mma, cluster_layout_vmnk);

  int grid_pairs_m = int(m) / kTileM;
  int grid_n = int(n) / kTileN;

  dim3 dimBlock(128);
  dim3 dimGrid(2 * grid_pairs_m, grid_n);  // blockIdx.x: (pair, v) interleaved
  int smem_bytes = sizeof(SharedStorageT);

  auto* kernel_ptr = &gemm_dual_cta_2sm<
      SharedStorageT, decltype(mA), decltype(mB), decltype(mC), decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma), decltype(tma_atom_A), decltype(tma_atom_B)>;

  AT_CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  AT_CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributePreferredSharedMemoryCarveout,
                                     cudaSharedmemCarveoutMaxShared));

  cudaLaunchConfig_t launch_config;
  launch_config.gridDim = dimGrid;
  launch_config.blockDim = dimBlock;
  launch_config.dynamicSmemBytes = smem_bytes;
  launch_config.stream = at::cuda::getCurrentCUDAStream();

  cudaLaunchAttribute cluster_attr;
  cluster_attr.id = cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim.x = 2;
  cluster_attr.val.clusterDim.y = 1;
  cluster_attr.val.clusterDim.z = 1;
  launch_config.numAttrs = 1;
  launch_config.attrs = &cluster_attr;

  void* args[] = {
    (void*)&mA, (void*)&mB, (void*)&mC, (void*)&mD,
    (void*)&mma_tiler, (void*)&tiled_mma,
    (void*)&tma_atom_A, (void*)&tma_atom_B
  };

  AT_CUDA_CHECK(cudaLaunchKernelExC(&launch_config, (void*)kernel_ptr, args));

  return d_buffer.to(torch::kFloat16);
}

}  // namespace dual_cta_2sm_impl

torch::Tensor matmul_tcgen05_dual_cta_2sm(torch::Tensor a, torch::Tensor b) {
  return dual_cta_2sm_impl::run_dual_cta_2sm_matmul(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_tcgen05_dual_cta_2sm", &matmul_tcgen05_dual_cta_2sm);
}
