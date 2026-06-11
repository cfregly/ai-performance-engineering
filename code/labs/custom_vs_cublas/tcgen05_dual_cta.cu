/**
 * Dual-CTA Occupancy Variant (2 CTAs/SM)
 * =======================================
 *
 * Why: the incumbent gemm_cluster (128x256 tile, 4 smem stages, FULL 512-col
 * TMEM allocation) is hard-capped at 1 CTA/SM (achieved occupancy 6.2%):
 *   (a) smem  ~192KB of 227KB  -> Block Limit Shared Mem = 1
 *   (b) TMEM  512/512 columns  -> a 2nd CTA's tcgen05.alloc cannot be served
 * Top stall is scoreboard-on-smem (46.5%): one CTA cannot keep the tensor
 * core fed across TMA round-trip latency.
 *
 * This variant restores SM-level concurrency instead of deepening one CTA:
 *   - 128x128 MMA tile        -> fp32 accumulator needs 128 TMEM cols/CTA
 *                                (2 CTAs = 256 of 512 cols: both fit)
 *   - 3 smem stages x 32KB    -> ~96KB/CTA (2 CTAs = ~192KB <= 227KB)
 *   - tcgen05 alloc permit released IMMEDIATELY after allocation, so the
 *     co-resident CTA can allocate (the incumbent held it to kernel end)
 *   - __launch_bounds__(128, 2) to bound regs for 2 blocks/SM
 *   - proper producer/consumer mbarrier pipeline: TMA reissue into a stage
 *     WAITS on that stage's empty barrier (umma_arrive = tcgen05.commit
 *     signals MMA-read completion). With 2 CTAs sharing the MMA pipe the
 *     incumbent's "no-wait" overwrite race is no longer benign.
 *   - mainloop runs on warp 0 only; warps 1-3 park on the mma barrier until
 *     the epilogue (they did nothing useful in the incumbent mainloop).
 *
 * While one CTA waits on TMA, the co-resident CTA's MMAs keep the tensor
 * core busy: 2 CTAs x 3 stages = 6 stages of latency hiding per SM vs 4.
 *
 * Tunables (compile-time, see tcgen05_loader.py):
 *   -DDUAL_TILE_N=128|256   (256 requires DUAL_STAGES=2 to keep 2 CTAs/SM)
 *   -DDUAL_STAGES=2|3|4
 *   -DDUAL_CLUSTER_M=1|2|4  (E3 lever: >1 launches a (DUAL_CLUSTER_M,1,1)
 *     cluster of vertically-adjacent M-tiles sharing the same B k-tiles and
 *     TMA-MULTICASTS B: each CTA issues 1/DUAL_CLUSTER_M of the B tile and
 *     the hardware delivers every slice into ALL cluster CTAs' smem. B-side
 *     L2->SM traffic divides by DUAL_CLUSTER_M, attacking the dominant
 *     long_scoreboard stall (28.5/38.9 warp-cycles/instr at (256,2) plain).
 *     Each CTA's full_barrier still receives the full A+B byte count per
 *     stage (own A + own B-slice + peer B-slices), so expect_tx math is
 *     unchanged. Stage-drain agreement becomes cluster-wide: tcgen05.commit
 *     is multicast to every cluster CTA's empty barrier (init count
 *     DUAL_CLUSTER_M), because a producer's B-multicast overwrites the
 *     PEER's stage too, not just its own.)
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
#include <cute/arch/cluster_sm90.hpp>  // For cute::elect_one_sync()

#ifndef DUAL_TILE_N
#define DUAL_TILE_N 128
#endif
#ifndef DUAL_STAGES
#define DUAL_STAGES 3
#endif
#ifndef DUAL_CLUSTER_M
#define DUAL_CLUSTER_M 1
#endif

#if DUAL_CLUSTER_M > 1
#define DUAL_CLUSTER_ATTR __cluster_dims__(DUAL_CLUSTER_M, 1, 1)
#else
#define DUAL_CLUSTER_ATTR
#endif

using namespace cute;

namespace dual_cta_impl {

using TypeA = cutlass::half_t;
using TypeB = cutlass::half_t;
using TypeC = float;
using TypeD = float;
using Accumulator = float;

constexpr int kTileM = 128;
constexpr int kTileN = DUAL_TILE_N;
constexpr int kStages = DUAL_STAGES;

static_assert(kStages >= 2 && kStages <= 4, "DUAL_STAGES must be 2..4");
static_assert(DUAL_CLUSTER_M == 1 || DUAL_CLUSTER_M == 2 || DUAL_CLUSTER_M == 4,
              "DUAL_CLUSTER_M must be 1, 2, or 4 (power-of-2 TMA multicast width)");
static_assert(kTileN >= 32 && (kTileN & (kTileN - 1)) == 0 && kTileN <= 256,
              "DUAL_TILE_N must be a power of two in [32, 256] (TMEM alloc granularity)");

// fp32 accumulator: one TMEM column (128 lanes x 32b) per N column.
constexpr int kAccTmemCols = kTileN;
static_assert(2 * kAccTmemCols <= cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns,
              "Accumulator must leave TMEM room for a co-resident CTA");

template <class TypeA_, class TypeB_, class ASmemLayout, class BSmemLayout>
struct DualSharedStorage {
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
    SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, kTileM, kTileN,
                         UMMA::Major::K, UMMA::Major::K>;

template <class SharedStorageT,
          class ATensor, class BTensor, class CTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA,
          class TmaAtomA, class TmaAtomB>
__global__ void DUAL_CLUSTER_ATTR __launch_bounds__(128, 2)
gemm_dual_cta(ATensor mA, BTensor mB, CTensor mC, DTensor mD,
              MmaTiler_MNK mma_tiler, TiledMMA tiled_mma,
              int grid_m, int grid_n,
              CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
              CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B) {

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  int tile_m = blockIdx.x;
  int tile_n = blockIdx.y;

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

  // ---------------------------------------------------------------------
  // TMEM: allocate ONLY the accumulator footprint (kAccTmemCols), then
  // immediately relinquish the allocation permit so the second CTA on this
  // SM can allocate. (Incumbent allocated all 512 cols and held the permit
  // until kernel end -- the structural 1-CTA/SM cap.)
  // ---------------------------------------------------------------------
  cute::TMEM::Allocator1Sm tmem_allocator{};
  if (elect_one_warp) {
    tmem_allocator.allocate(kAccTmemCols, &storage.tmem_base_ptr);
    tmem_allocator.release_allocation_lock();
  }

  // Initialize barriers (single thread), then make both TMEM base and
  // barriers visible CTA-wide.
  if (elect_one_warp && elect_one_thr) {
    for (int s = 0; s < kStages; ++s) {
      cute::initialize_barrier(storage.full_barrier[s], 1);
      // Stage drain is cluster-wide under B-multicast: one tcgen05.commit
      // arrival per cluster CTA (DUAL_CLUSTER_M=1 -> plain single arrival).
      cute::initialize_barrier(storage.empty_barrier[s], DUAL_CLUSTER_M);
    }
    cute::initialize_barrier(storage.mma_barrier, 1);
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();
#if DUAL_CLUSTER_M > 1
  // Every cluster CTA must observe our barrier init before its TMA multicast
  // (data + transaction bytes) or commit multicast can target our smem.
  cute::cluster_sync();
#endif

  uint32_t tmem_base = storage.tmem_base_ptr;

  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);
  tCtAcc.data() = tmem_base;

  // TMA coordinate tensors
  Tensor tma_coord_A = tma_atom_A.get_tma_tensor(shape(mA));
  Tensor tma_coord_B = tma_atom_B.get_tma_tensor(shape(mB));
  Tensor gCoordA = local_tile(tma_coord_A, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gCoordB = local_tile(tma_coord_B, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor tCgCoordA = cta_mma.partition_A(gCoordA);
  Tensor tCgCoordB = cta_mma.partition_B(gCoordB);

  // gmem-side TMA partitions are stage-independent; smem-side tensors for
  // stage s differ from stage 0 only by base pointer (identical layout).
  Tensor tCsA_0 = storage.tensor_sA(0);
  auto [tAgA, tAsA_0] = tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsA_0), group_modes<0,3>(tCgCoordA));

#if DUAL_CLUSTER_M > 1
  // B is TMA-multicast across the cluster M-mode: tma_partition selects this
  // CTA's 1/DUAL_CLUSTER_M slice of the B k-tile (by cluster rank) on BOTH
  // the gmem and smem side; the copy below delivers each slice into every
  // cluster CTA's smem and signals every CTA's full_barrier with that
  // slice's bytes. The multicast slice offset is baked into the partitioned
  // tensors' ITERATORS (not the layout), so per-stage smem tensors cannot be
  // rebuilt by base-pointer swap -- partition each stage explicitly.
  uint16_t mcast_mask_b = uint16_t((1u << DUAL_CLUSTER_M) - 1);
  int cta_rank_m = int(cute::block_rank_in_cluster());
  auto cta_layout_m = make_layout(Int<DUAL_CLUSTER_M>{});
  auto partB = [&](int s) {
    return tma_partition(tma_atom_B, cta_rank_m, cta_layout_m,
        group_modes<0,3>(storage.tensor_sB(s)), group_modes<0,3>(tCgCoordB));
  };
  auto partB_s = [&](int s) { auto [g, smem] = partB(s); return smem; };
  auto [tBgB, tBsB_0] = partB(0);
#if DUAL_STAGES == 2
  decltype(tBsB_0) tBsB_st[kStages] = { tBsB_0, partB_s(1) };
#elif DUAL_STAGES == 3
  decltype(tBsB_0) tBsB_st[kStages] = { tBsB_0, partB_s(1), partB_s(2) };
#else
  decltype(tBsB_0) tBsB_st[kStages] = { tBsB_0, partB_s(1), partB_s(2), partB_s(3) };
#endif
#else
  Tensor tCsB_0 = storage.tensor_sB(0);
  auto [tBgB, tBsB_0] = tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsB_0), group_modes<0,3>(tCgCoordB));
#endif

  // NOTE: under multicast, tAsA_0/tBsB_0 mode-0 still spans the FULL tile
  // coordinate space, so this is the full A+B byte count per stage -- which
  // is exactly what each CTA's barrier receives (own A + all B slices).
  int tma_bytes = sizeof(make_tensor_like(tAsA_0)) + sizeof(make_tensor_like(tBsB_0));

  auto issue_tma = [&](int stage, int k_tile) {
    cute::set_barrier_transaction_bytes(storage.full_barrier[stage], tma_bytes);
    auto tAsA = make_tensor(make_smem_ptr(storage.smem_A[stage].begin()), tAsA_0.layout());
    copy(tma_atom_A.with(storage.full_barrier[stage]), tAgA(_, k_tile), tAsA);
#if DUAL_CLUSTER_M > 1
    copy(tma_atom_B.with(storage.full_barrier[stage], mcast_mask_b), tBgB(_, k_tile), tBsB_st[stage]);
#else
    auto tBsB = make_tensor(make_smem_ptr(storage.smem_B[stage].begin()), tBsB_0.layout());
    copy(tma_atom_B.with(storage.full_barrier[stage]), tBgB(_, k_tile), tBsB);
#endif
  };

  // Per-stage UMMA smem-descriptor fragments (same type per stage -> array).
  auto fragA = [&](int s) { return cta_mma.make_fragment_A(storage.tensor_sA(s)); };
  auto fragB = [&](int s) { return cta_mma.make_fragment_B(storage.tensor_sB(s)); };
#if DUAL_STAGES == 2
  decltype(fragA(0)) tCrA_st[kStages] = { fragA(0), fragA(1) };
  decltype(fragB(0)) tCrB_st[kStages] = { fragB(0), fragB(1) };
#elif DUAL_STAGES == 3
  decltype(fragA(0)) tCrA_st[kStages] = { fragA(0), fragA(1), fragA(2) };
  decltype(fragB(0)) tCrB_st[kStages] = { fragB(0), fragB(1), fragB(2) };
#else
  decltype(fragA(0)) tCrA_st[kStages] = { fragA(0), fragA(1), fragA(2), fragA(3) };
  decltype(fragB(0)) tCrB_st[kStages] = { fragB(0), fragB(1), fragB(2), fragB(3) };
#endif

  int num_k_tiles = size<3>(tCgA);

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  // =====================================================================
  // Mainloop: warp 0 only. Warps 1-3 park on mma_barrier until epilogue.
  // =====================================================================
  if (elect_one_warp) {
    int full_phase[kStages] = {};
    int empty_phase[kStages] = {};

    // Prologue: fill kStages-1 stages.
    if (elect_one_thr) {
      for (int i = 0; i < min(kStages - 1, num_k_tiles); ++i) {
        issue_tma(i, i);
      }
    }

    for (int k = 0; k < num_k_tiles; ++k) {
      int curr = k % kStages;
      int next_k = k + (kStages - 1);

      // Refill pipeline ahead of the consumer. Reusing a stage requires its
      // previous MMA reads to be complete (empty barrier <- tcgen05.commit).
      if (next_k < num_k_tiles) {
        int next_s = next_k % kStages;
        if (next_k >= kStages) {  // stage has been used before -> wait for drain
          cute::wait_barrier(storage.empty_barrier[next_s], empty_phase[next_s]);
          empty_phase[next_s] ^= 1;
        }
        if (elect_one_thr) {
          issue_tma(next_s, next_k);
        }
      }

      // Consume current stage.
      cute::wait_barrier(storage.full_barrier[curr], full_phase[curr]);
      full_phase[curr] ^= 1;

      for (int kb = 0; kb < size<2>(tCrA_st[0]); ++kb) {
        gemm(tiled_mma, tCrA_st[curr](_, _, kb), tCrB_st[curr](_, _, kb), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }

      // tcgen05.commit: empty_barrier[curr] arrives when MMA reads are done.
#if DUAL_CLUSTER_M > 1
      // Multicast the commit to EVERY cluster CTA's empty barrier: reusing a
      // stage overwrites the peers' copies of B too, so the producer must
      // know all cluster CTAs drained it (barrier init count DUAL_CLUSTER_M).
      cutlass::arch::umma_arrive_multicast(
          reinterpret_cast<uint64_t*>(&storage.empty_barrier[curr]), mcast_mask_b);
#else
      cutlass::arch::umma_arrive(
          reinterpret_cast<uint64_t*>(&storage.empty_barrier[curr]));
#endif
    }

#if DUAL_CLUSTER_M > 1
    // Drain the final multicast commits. Every commit we multicast at a peer
    // must be DELIVERED before that peer exits (its smem barriers die with
    // it). Waiting our own empty barriers proves the peers' final commits
    // reached us; the peers' identical drain proves ours reached them, so no
    // remote arrival is in flight when any CTA exits.
    for (int k = num_k_tiles - min(kStages, num_k_tiles); k < num_k_tiles; ++k) {
      int s = k % kStages;
      cute::wait_barrier(storage.empty_barrier[s], empty_phase[s]);
      empty_phase[s] ^= 1;
    }
#endif

    // Final commit: mma_barrier arrives when ALL issued MMAs completed.
    cutlass::arch::umma_arrive(&storage.mma_barrier);
  }

  // All 128 threads rendezvous here for the epilogue.
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
    tmem_allocator.free(tmem_base, kAccTmemCols);
  }
}

torch::Tensor run_dual_cta_matmul(torch::Tensor a, torch::Tensor b) {
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
              "Size must be divisible by tcgen05 tile (", kTileM, "x", kTileN, "x64)");

  auto options = a.options().dtype(torch::kFloat32);
  auto c_buffer = torch::empty({m, n}, options);  // beta=0: never read
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

  using SharedStorageT = DualSharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout)>;

  // 2 CTAs/SM is the whole point: per-CTA smem must fit twice in the
  // 227KB/SM budget (alloc granularity headroom included).
  static_assert(sizeof(SharedStorageT) <= 110 * 1024,
                "Shared storage too large for 2 CTAs/SM; reduce DUAL_STAGES or DUAL_TILE_N");

  Tensor mA = make_tensor(make_gmem_ptr(reinterpret_cast<TypeA const*>(a_contig.data_ptr<at::Half>())),
      make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
  Tensor mB = make_tensor(make_gmem_ptr(reinterpret_cast<TypeB const*>(b_contig.data_ptr<at::Half>())),
      make_layout(make_shape(n, k), make_stride(k, Int<1>{})));
  Tensor mC = make_tensor(make_gmem_ptr(c_buffer.data_ptr<TypeC>()),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
  Tensor mD = make_tensor(make_gmem_ptr(d_buffer.data_ptr<TypeD>()),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));

  auto tma_atom_A = make_tma_atom(SM90_TMA_LOAD{}, mA, sA_layout, select<0, 2>(mma_tiler));
#if DUAL_CLUSTER_M > 1
  // Multicast atom: the TMA box is truncated to 1/DUAL_CLUSTER_M of the B
  // tile; each cluster CTA issues its slice and the hardware fans it out to
  // all CTAs in the 16-bit cta mask.
  auto tma_atom_B = make_tma_atom(SM90_TMA_LOAD_MULTICAST{}, mB, sB_layout,
                                  select<1, 2>(mma_tiler), Int<DUAL_CLUSTER_M>{});
#else
  auto tma_atom_B = make_tma_atom(SM90_TMA_LOAD{}, mB, sB_layout, select<1, 2>(mma_tiler));
#endif

  int grid_m = (m + size(bM) - 1) / size(bM);
  int grid_n = (n + size(bN) - 1) / size(bN);

  TORCH_CHECK(grid_m % DUAL_CLUSTER_M == 0,
              "M / ", kTileM, " must be divisible by DUAL_CLUSTER_M=", DUAL_CLUSTER_M);

  dim3 dimBlock(128);
  dim3 dimGrid(grid_m, grid_n);
  int smem_bytes = sizeof(SharedStorageT);

  auto* kernel_ptr = &gemm_dual_cta<
      SharedStorageT, decltype(mA), decltype(mB), decltype(mC), decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma), decltype(tma_atom_A), decltype(tma_atom_B)>;

  AT_CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  // Ask for the maximum smem carveout so two ~96KB CTAs can co-reside.
  AT_CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributePreferredSharedMemoryCarveout,
                                     cudaSharedmemCarveoutMaxShared));

  cudaLaunchConfig_t launch_config;
  launch_config.gridDim = dimGrid;
  launch_config.blockDim = dimBlock;
  launch_config.dynamicSmemBytes = smem_bytes;
  launch_config.stream = at::cuda::getCurrentCUDAStream();
#if DUAL_CLUSTER_M > 1
  // Must match the kernel's __cluster_dims__: (DUAL_CLUSTER_M, 1, 1).
  cudaLaunchAttribute cluster_attr;
  cluster_attr.id = cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim.x = DUAL_CLUSTER_M;
  cluster_attr.val.clusterDim.y = 1;
  cluster_attr.val.clusterDim.z = 1;
  launch_config.numAttrs = 1;
  launch_config.attrs = &cluster_attr;
#else
  launch_config.numAttrs = 0;
  launch_config.attrs = nullptr;
#endif

  void* args[] = {
    (void*)&mA, (void*)&mB, (void*)&mC, (void*)&mD,
    (void*)&mma_tiler, (void*)&tiled_mma,
    (void*)&grid_m, (void*)&grid_n,
    (void*)&tma_atom_A, (void*)&tma_atom_B
  };

  AT_CUDA_CHECK(cudaLaunchKernelExC(&launch_config, (void*)kernel_ptr, args));

  return d_buffer.to(torch::kFloat16);
}

}  // namespace dual_cta_impl

torch::Tensor matmul_tcgen05_dual_cta(torch::Tensor a, torch::Tensor b) {
  return dual_cta_impl::run_dual_cta_matmul(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_tcgen05_dual_cta", &matmul_tcgen05_dual_cta);
}
