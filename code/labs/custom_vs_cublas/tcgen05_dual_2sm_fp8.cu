/**
 * FP8 (e4m3) port of the 2-SM UMMA Pair kernel (Front F8)
 * ========================================================
 *
 * NEW VARIANT FILE: a type-level port of tcgen05_dual_cta_2sm.cu (the FP16
 * champion: 2-SM UMMA pair, warp-split producer/consumer, persistent
 * clusters + GROUP_M raster, TMA-store epilogue; ~70% real FP16 SoL at
 * 8192^3) to dense FP8: e4m3 x e4m3 -> fp32 accumulate -> fp16 D (converted
 * host-side from the fp32 D buffer, same as the FP16 file). NO block scales
 * (plain dense GEMM; the NVFP4 scaled family lives elsewhere).
 *
 * What changes vs the FP16 file (everything else is line-identical):
 *   - MMA atom: SM100_MMA_F8F6F4_2x1SM_SS (tcgen05.mma.cta_group::2
 *     .kind::f8f6f4). In this CUTLASS drop the F8F6F4 op is a NON-template
 *     tag: types/shape go through MMA_Traits (the 77_blackwell_fmha
 *     pattern). Atom K-extent = 32 elements (256 bits / 8) vs FP16's 16.
 *   - Host-print-checked (B49): the FP8 atom partitions B as kTileN/2 per
 *     CTA and A as the 128-row M-half, exactly like the FP16 atom:
 *       partition_shape_A((256,128)) = ((_128,_32),_1,_4)
 *       partition_shape_B((128,128)) = ((_64,_32),_1,_4)
 *   - kTileK default 128 (was 64): e4m3 halves bytes/element, so the
 *     128-deep K stage is byte-identical to the FP16 champion's 64-deep
 *     stage (A-half 16KB + B-half 8KB = 24KB/stage at TILE_N=128) while
 *     feeding 2x the K-elements: barrier round-trips per fed byte HALVE
 *     (num_k_tiles at 8192 drops 128 -> 64; still 4 atom k-blocks/stage).
 *   - Swizzle atom is kTileK-conditional: e4m3 K-rows are kTileK BYTES, so
 *     kTileK=128 -> SW128 and kTileK=64 -> SW64 (12KB/stage: the
 *     deeper-ring sweep axis, stages up to 6 at the same 72KB footprint).
 *   - TMA_EPI (lever h) keeps the fp32 staging chunk (128x32 fp32 = 16KB):
 *     it only fits the drained smem_A stage when kTileK >= 128 (asserted).
 *   - TMEM accumulator stays fp32 kTileN cols/CTA: the occupancy math
 *     (3 CTAs/SM at TILE_N=128: smem-implied 3, TMEM-implied 4, min 3) is
 *     UNCHANGED from the FP16 champion (B63 sizing law).
 *
 * Tunables (compile-time, see tcgen05_loader.py): same DUAL2SM_* macros as
 * the FP16 file (separate translation unit; separate .so). Default config
 * is the FP16 champion's geometry adapted: (TILE_N=128, STAGES=3,
 * TILE_K=128, ws=1, mb=3, persist=1, raster_gm=8, tma_epi=1).
 */

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <cuda.h>          // CUtensorMap / cuTensorMapEncodeTiled (lever h)
#include <cuda_fp16.h>     // __floats2half2_rn (F8b D_HALF staging convert)
#include <cuda_runtime.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/half.h>
#include <cutlass/numeric_types.h>  // float_e4m3_t

#include <cute/tensor.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>  // make_tma_atom_A/B_sm100, SM100_TMA_2SM_LOAD
#include <cute/atom/copy_traits_sm90_tma.hpp>   // SM90_TMA_STORE, tma_store_arrive/wait (lever h)
#include <cute/arch/cluster_sm90.hpp>           // elect_one_sync, block_rank_in_cluster, cluster_sync
#include <cute/algorithm/prefetch.hpp>          // cute::prefetch (TMA PREFETCH op, lever g)

#ifndef DUAL2SM_TILE_N
#define DUAL2SM_TILE_N 128
#endif
#ifndef DUAL2SM_STAGES
#define DUAL2SM_STAGES 2
#endif
// Lever (a), V-front: split the leader's serialized empty-wait -> TMA ->
// full-wait -> MMA chain across two WHOLE warps (warp 0 = producer on both
// CTAs, warp 1 = consumer on the leader). 0 = incumbent single-warp loop.
#ifndef DUAL2SM_WARP_SPLIT
#define DUAL2SM_WARP_SPLIT 0
#endif
// Lever (b), V-front: (2,2,1) cluster + SM100_TMA_2SM_LOAD_MULTICAST on A
// across the cluster's N mode (two SM pairs sharing tile_m re-read the same
// A tile; ncu measured 376.8M L2 sectors at n=128 vs 257.5M plain).
// 0 = incumbent (2,1,1) non-multicast cluster.
#ifndef DUAL2SM_AMCAST
#define DUAL2SM_AMCAST 0
#endif
// Lever (c), V3-front: __launch_bounds__ min-blocks-per-SM hint. 2 = the
// B57 default register budget (caps at 255 regs; measured 152 at n=128);
// 4 caps regs at 128/thread so a 4th co-resident CTA is REACHABLE on the
// tile_n=64 footprint (smem 40KB/CTA at s=2, TMEM 4x64 of 512 cols).
// Hint only: occupancy is still bound by min(smem, TMEM, regs) at runtime.
#ifndef DUAL2SM_MIN_BLOCKS
#define DUAL2SM_MIN_BLOCKS 2
#endif
// Lever (d), V3-front fallback axis: K-extent of one pipeline stage.
// 128 doubles the per-stage TMA box (A 32KB + B-half 8KB at n=128) and
// HALVES barrier round-trips per fed byte; 64 = the incumbent k-block.
#ifndef DUAL2SM_TILE_K
#define DUAL2SM_TILE_K 128
#endif
// Lever (e), V4-front: cross-tile TMEM double-buffered epilogue overlap.
// 0 = incumbent (single TMEM buffer; epilogue serialized after the full
// k-loop). T>=2 = each cluster processes T consecutive n-tiles with TWO
// TMEM accumulator buffers (2*TILE_N cols/CTA) and a SECOND warpgroup
// (threads 128..255) as a dedicated epilogue: it drains buffer (t%2) and
// stores tile t WHILE the consumer's MMA stream fills buffer ((t+1)%2);
// the smem k-pipeline flows straight across tile boundaries (no per-tile
// prologue bubble). Structural notes that force this formulation:
//   - the 2x1SM atom computes the FULL 256xTILE_N pair tile per k-block,
//     so both N-halves advance through K in lockstep -- a within-tile
//     k-tail split can only overlap (kStages-1)/num_k_tiles (~1.6%) of
//     the mainloop. Cross-TILE double-buffering is the only real window.
//   - the t2r drain is warpgroup-bound (TMEM subpartition w is only
//     addressable by a warp of rank w within its warpgroup), so the
//     drain needs a full second warpgroup, not the parked warps 2-3.
// TMEM cost: 2x128=256 cols/CTA -> TWO CTAs/SM (the incumbent's THREE
// CTAs/SM needs 3x256=768 > 512); the B60-flagged occupancy-vs-overlap
// tradeoff, measured here. Under EPI_OVERLAP the mainloop is always the
// warp-split producer/consumer structure (DUAL2SM_WARP_SPLIT ignored).
#ifndef DUAL2SM_EPI_OVERLAP
#define DUAL2SM_EPI_OVERLAP 0
#endif
// t2r atom column-repeat for the overlap epilogue's CHUNKED drain (B55
// trap: re-sweep atom width when the epilogue structure changes). 32 ->
// four 32-col chunks at TILE_N=128 with 32-reg fragments, fitting the
// 128-reg/thread budget that 256 threads x 2 CTAs/SM imposes.
#ifndef DUAL2SM_EPI_ATOM
#define DUAL2SM_EPI_ATOM 32
#endif
// Lever (f), V5-front: persistent clusters + L2-aware GROUP_M raster.
// DUAL2SM_PERSIST=1 launches exactly the co-residable cluster count
// (occupancy API clamped by the TMEM-implied CTAs/SM; B63 sizing law) and
// each cluster walks raster indices r, r+C, r+2C... so the in-flight
// window is ~C consecutive raster indices at all times. Composes with
// EPI_OVERLAP=2 (V4's double-TMEM + epilogue-warpgroup cross-tile pipeline
// as the persistent inner loop; 2 CTAs/SM) or EPI_OVERLAP=0 (champion
// occupancy, 3 CTAs/SM, per-round serialized epilogue that co-residency
// hides per the V4 finding, chunked t2r to hold the 170-reg budget).
#ifndef DUAL2SM_PERSIST
#define DUAL2SM_PERSIST 0
#endif
// DUAL2SM_RASTER_GM=g (0=off): GROUP_M tile rasterization -- groups of g
// pair-rows sweep n together, so a window of W consecutive indices touches
// g A row-panels (g x 4MiB at 8192^3, L2-resident across the n-sweep) +
// W/g B col-panels (2MiB each, streamed once per group) instead of the
// incumbent x-fastest order's full-M column (128MiB of A per wave, which
// thrashes L2 and re-reads A every wave: the measured 1.36GB vs the
// 640MiB raster floor). Without PERSIST the launch grid is flattened to
// 1D and relies on ascending-blockIdx hardware rasterization (the
// standard Triton GROUP_M trick); the wave window is then ~228 indices
// (3 CTAs/SM) -> gm=8: 32MiB A + 57MiB B + ~29MiB D in flight < 129MiB L2.
#ifndef DUAL2SM_RASTER_GM
#define DUAL2SM_RASTER_GM 0
#endif
// Lever (g), V6-front: raster-aware L2 prefetch, scoped to the PERSIST
// eo=0 path (the B65 winner). That path's smem ring FULLY DRAINS at each
// round boundary and refills cold: the first kStages TMA issues of round
// t+1 all miss to DRAM. But round t+1's tile is deterministic
// (raster_map(rr + (t+1)*C)), so issue cp.async.bulk.prefetch.tensor
// (the TMA atoms' PREFETCH op -- same descriptors, same coord slices,
// L2-only, no smem/barrier) for its first DUAL2SM_PREFETCH k-stages'
// A/B panels during round t. Each CTA prefetches its OWN issue slices
// (A-half + B N/2-half), so the pair covers the full tile boxes.
// 0 = off; 1..kStages = #leading k-stages prefetched.
#ifndef DUAL2SM_PREFETCH
#define DUAL2SM_PREFETCH 0
#endif
// Prefetch issuer (sweep axis): 0 = producer warp 0 (both CTAs) right
// after round t's LAST TMA issue (lead = consumer's remaining ~kStages
// MMAs + epilogue, a few us); 1 = epilogue-parked warp 2 right after the
// round's mma_barrier release (latest issue, shortest lead); 2 = parked
// warp 3 at round-t START (max ~33us lead, but the prefetched lines must
// survive a full round of demand traffic on a ~118MiB in-flight window).
#ifndef DUAL2SM_PF_ISSUER
#define DUAL2SM_PF_ISSUER 0
#endif
// Lever (h), V7-front: TMA-store epilogue through the drained ring stage
// (the B66-named LAST scheduling lever). Scoped to the PERSIST eo=0
// incumbent: its per-round epilogue is kTileN/32 x [chunked t2r -> fence
// -> 8 per-thread STG.E.128], whose 16B-per-32B-sector pattern is
// STRUCTURAL to the 32dp32b t2r layout (thread == row, so the 32 lanes'
// 16B vectors hit 32 different rows = 32 different sectors; V6 ncu:
// 16.78M global-store sectors = exactly 2x the 8.39M floor, 20.05M L1->L2
// write sectors = 2.39x). Staging each 128x32 fp32 chunk through the
// DRAINED stage smem_A[c % kStages] (16KB, free at every round boundary)
// with the SW128 XOR pattern (16B-group ^ (row & 7); the B58 capstone
// pattern, conflict-free STS.128) turns the store path into ONE
// cp.async.bulk.tensor.2d per chunk at 100% sector efficiency. Zero
// TMEM/occupancy cost (unlike the retired eo=2 family). Completion: one
// commit group per chunk; staging-buffer reuse waits
// tma_store_wait<kStages-1> (the TMA engine is the ring's last consumer
// -- the B58 trap); the producer's elected thread waits <0> AFTER the
// tmem_empty arrive, so only the smem-reuse path (not the consumer's
// next-round MMA) is gated on the store drain.
#ifndef DUAL2SM_TMA_EPI
#define DUAL2SM_TMA_EPI 0
#endif
// F8b secondary lever: in-kernel fp16 D output, scoped to the TMA_EPI
// store path. The epilogue converts each PAIR of 32-col fp32 t2r chunks
// (__float2half_rn, the same round-to-nearest-even as torch's host-side
// .to(fp16)) into ONE 128x64 fp16 staged tile -- still 128B/row, still
// the SW128 16B-group XOR physical pattern, still one drained 16KB A
// stage -- so D-store bytes HALVE and the host-side fp32->fp16
// conversion kernel (which re-read + re-wrote the whole fp32 D inside
// the timed region) is RETIRED entirely. Exactness: on the exact
// dataset the fp32 accumulator value is exact, so in-kernel RN rounding
// equals host-side RN rounding bit-for-bit (rel_err 0.0 must still hold).
#ifndef DUAL2SM_D_HALF
#define DUAL2SM_D_HALF 0
#endif
// F8c lever: in-CTA epilogue/mainloop overlap WITHIN the 2-CTAs/SM
// footprint -- the eo=2 structure at eo=0's occupancy. TMEM arithmetic:
// double-buffering at n256 needs 2x256 = 512 cols/CTA (the WHOLE TMEM ->
// 1 CTA/SM; inadmissible), and a half-drain of one n256 accumulation is
// impossible (every 256-wide MMA k-block writes all 256 cols). The only
// formulation inside the 256-col/CTA budget is n128 tiles with 2x128-col
// TMEM buffers (kNumAccBufs=2) -- exactly the champion's TMEM/CTA spend,
// so 2 CTAs/SM is preserved. EPI_OVERLAP=2 + PERSIST give the structure;
// THIS lever adds the F8b te1+dh1 store path to that epilogue: under
// eo=2 the smem ring NEVER drains (the mainloop streams across tile
// boundaries), so the eo=0 trick of staging through a drained smem_A
// stage is unavailable -- a DEDICATED double-buffered staging pair
// (2 x 128x64 fp16 SW128 boxes, 32KB) is added to SharedStorage, and the
// epilogue warpgroup synchronizes with bar.sync 1,128 (warpgroup-local;
// __syncthreads would deadlock against the mainloop warps). The drain
// unit must be a full second WARPGROUP, not a warp pair: tcgen05.ld
// 32dp32b from TMEM subpartition w is only addressable by the warp of
// rank w within its warpgroup (ranks 0-3 needed for 128 lanes).

using namespace cute;

namespace dual_2sm_fp8_impl {

using TypeA = cutlass::float_e4m3_t;
using TypeB = cutlass::float_e4m3_t;
using TypeC = float;
// D gmem element: fp16 under the F8b D_HALF lever (in-kernel convert in
// the TMA_EPI staging path), fp32 otherwise (host-side .to(fp16)).
#if DUAL2SM_D_HALF
using TypeD = cutlass::half_t;
#else
using TypeD = float;
#endif
using Accumulator = float;

constexpr int kTileM = 256;  // pair-wide M: 128 rows per CTA of the pair
constexpr int kTileN = DUAL2SM_TILE_N;
constexpr int kStages = DUAL2SM_STAGES;
constexpr int kTileK = DUAL2SM_TILE_K;
// Cluster shape is (2, kClusterN, 1): V mode is the SM pair; under AMCAST
// the N mode groups two SM pairs that share tile_m (A multicast partners).
constexpr int kClusterN = DUAL2SM_AMCAST ? 2 : 1;

static_assert(kStages >= 2 && kStages <= 6, "DUAL2SM_STAGES must be 2..6");
static_assert(kTileK == 64 || kTileK == 128, "DUAL2SM_TILE_K must be 64 or 128");
static_assert(kTileN >= 32 && (kTileN & (kTileN - 1)) == 0 && kTileN <= 256,
              "DUAL2SM_TILE_N must be a power of two in [32, 256]");

// V4 epilogue-overlap shape (lever e): T output tiles per cluster walked
// consecutively along n; 2 TMEM accumulator buffers; 2 warpgroups.
constexpr int kTilesPerCta = (DUAL2SM_EPI_OVERLAP >= 2) ? DUAL2SM_EPI_OVERLAP : 1;
constexpr bool kEpiOverlap = (DUAL2SM_EPI_OVERLAP >= 2);
constexpr int kNumAccBufs = kEpiOverlap ? 2 : 1;
constexpr int kNumThreads = kEpiOverlap ? 256 : 128;
static_assert(DUAL2SM_EPI_OVERLAP == 0 || DUAL2SM_EPI_OVERLAP == 2 ||
              DUAL2SM_EPI_OVERLAP == 4 || DUAL2SM_EPI_OVERLAP == 8,
              "DUAL2SM_EPI_OVERLAP must be 0 (off) or 2|4|8 tiles per CTA");
static_assert(!(kEpiOverlap && DUAL2SM_AMCAST),
              "Epilogue overlap is not combined with A-multicast (scope)");
static_assert(!kEpiOverlap || kTileN % DUAL2SM_EPI_ATOM == 0,
              "DUAL2SM_EPI_ATOM must divide DUAL2SM_TILE_N");

// V5 persistent + raster shape (lever f).
constexpr bool kPersist = (DUAL2SM_PERSIST != 0);
static_assert(!kPersist || DUAL2SM_AMCAST == 0,
              "PERSIST is not combined with A-multicast (scope)");
static_assert(!kPersist || DUAL2SM_EPI_OVERLAP == 0 || DUAL2SM_EPI_OVERLAP == 2,
              "Under PERSIST the walk is dynamic: EPI_OVERLAP must be 0 "
              "(3 CTAs/SM, per-round epilogue) or 2 (double-buffered "
              "epilogue warpgroup)");
static_assert(DUAL2SM_RASTER_GM >= 0, "DUAL2SM_RASTER_GM must be >= 0");
static_assert(DUAL2SM_RASTER_GM == 0 || kPersist ||
              (DUAL2SM_EPI_OVERLAP == 0 && DUAL2SM_AMCAST == 0),
              "Non-persistent raster is scoped to the incumbent (eo=0, "
              "non-multicast) shape");
static_assert(DUAL2SM_PREFETCH == 0 ||
              (kPersist && DUAL2SM_EPI_OVERLAP == 0),
              "PREFETCH is scoped to the persistent eo=0 path (the B65 "
              "winner; its per-round cold ring refill is the target)");
static_assert(DUAL2SM_PREFETCH >= 0 && DUAL2SM_PREFETCH <= kStages,
              "DUAL2SM_PREFETCH is a leading-stage count (0..kStages)");
static_assert(DUAL2SM_PF_ISSUER >= 0 && DUAL2SM_PF_ISSUER <= 2,
              "DUAL2SM_PF_ISSUER must be 0 (producer), 1 (epilogue), or "
              "2 (parked warp, round start)");
static_assert(DUAL2SM_TMA_EPI == 0 || (kPersist && DUAL2SM_EPI_OVERLAP == 0) ||
              (kPersist && DUAL2SM_EPI_OVERLAP == 2),
              "TMA_EPI is scoped to the persistent paths: eo=0 (per-round "
              "serialized epilogue) or eo=2 (F8c overlap epilogue)");
static_assert(DUAL2SM_TMA_EPI == 0 || DUAL2SM_EPI_ATOM == 32,
              "TMA_EPI staging assumes 32-col t2r chunks (128x32 fp32 = 16KB "
              "= one drained A stage)");
static_assert(DUAL2SM_TMA_EPI == 0 || DUAL2SM_EPI_OVERLAP != 0 || kTileK >= 128,
              "FP8 TMA_EPI eo=0: the 16KB fp32 staging chunk only fits a "
              "drained A stage when kTileK >= 128 (128 rows x kTileK x 1B)");
static_assert(DUAL2SM_TMA_EPI == 0 || DUAL2SM_EPI_OVERLAP == 0 ||
              (DUAL2SM_D_HALF == 1 && kTileN == 128),
              "F8c overlap TMA_EPI: scoped to kTileN=128 (2x128-col TMEM "
              "double-buffer = the champion's 256-col/CTA budget) with "
              "in-kernel fp16 D (dedicated 128x64 fp16 staging boxes)");
static_assert(DUAL2SM_D_HALF == 0 || DUAL2SM_TMA_EPI == 1,
              "D_HALF is scoped to the TMA_EPI store path (the other store "
              "paths copy fp32 fragments straight to gmem)");
static_assert(DUAL2SM_D_HALF == 0 || kTileN % 64 == 0,
              "D_HALF stages chunk PAIRS (64 fp16 cols = 128B rows)");

// Raster-order tile mapping: index idx in [0, grid_m*grid_n) ->
// (tile_m, tile_n). GROUP_M (gm>0): groups of gm pair-rows sweep n
// together (m-fastest within a group column) so the group's A row-panels
// stay L2-resident while B col-panels stream once per group. gm=0:
// m-fastest linear (the incumbent 2D grid's x-fastest wave order).
CUTE_DEVICE void raster_map(int idx, int grid_m, int grid_n, int& tm, int& tn) {
#if DUAL2SM_RASTER_GM > 0
  constexpr int gm = DUAL2SM_RASTER_GM;
  int per_group = gm * grid_n;
  int group = idx / per_group;
  int first_m = group * gm;
  int gsz = grid_m - first_m;       // tail group when gm does not divide grid_m
  gsz = gsz < gm ? gsz : gm;
  int rem = idx - group * per_group;
  tm = first_m + rem % gsz;
  tn = rem / gsz;
#else
  tm = idx % grid_m;
  tn = idx / grid_m;
#endif
}

// fp32 accumulator: each CTA holds its 128-row M-half x kTileN per buffer.
constexpr int kAccTmemCols = kNumAccBufs * kTileN;
static_assert(2 * kAccTmemCols <= cute::TMEM::Allocator2Sm::Sm100TmemCapacityColumns,
              "Accumulator must leave TMEM room for a co-resident CTA");

template <class TypeA_, class TypeB_, class ASmemLayout, class BSmemLayout>
struct Dual2SmSharedStorage {
  alignas(128) cute::ArrayEngine<TypeA_, cute::cosize_v<ASmemLayout>> smem_A[kStages];
  alignas(128) cute::ArrayEngine<TypeB_, cute::cosize_v<BSmemLayout>> smem_B[kStages];
#if DUAL2SM_TMA_EPI && DUAL2SM_EPI_OVERLAP
  // F8c: dedicated epilogue staging (the ring never drains under eo=2, so
  // the eo=0 drained-A-stage trick is unavailable). Two 128-row x 64-col
  // fp16 SWIZZLE_128B boxes (16KB each): chunk-pair stores double-buffer
  // against the TMA store engine.
  alignas(1024) uint8_t epi_stage[2][128 * 64 * 2];
#endif
  alignas(16) cute::uint64_t full_barrier[kStages];   // leader-only: pair TMA bytes
  alignas(16) cute::uint64_t empty_barrier[kStages];  // per-CTA: multicast commit
  alignas(16) cute::uint64_t mma_barrier[kNumAccBufs];  // per acc buffer
  alignas(16) cute::uint64_t tmem_empty[kNumAccBufs];   // leader-only: both CTAs' drains
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE auto tensor_sA(int stage) {
    return make_tensor(make_smem_ptr(smem_A[stage].begin()), ASmemLayout{});
  }
  CUTE_DEVICE auto tensor_sB(int stage) {
    return make_tensor(make_smem_ptr(smem_B[stage].begin()), BSmemLayout{});
  }
};

// FP8 dense 2SM atom. The F8F6F4 op is a non-template tag in this CUTLASS
// drop; types/shape are MMA_Traits parameters (the 77_blackwell_fmha
// pattern). K-extent per MMA = 32 e4m3 elements (vs 16 for FP16).
using MmaTag =
    MMA_Traits<SM100_MMA_F8F6F4_2x1SM_SS, TypeA, TypeB, TypeC,
               cute::C<kTileM>, cute::C<kTileN>,
               cute::integral_constant<UMMA::Major, UMMA::Major::K>,
               cute::integral_constant<UMMA::Major, UMMA::Major::K>,
               cute::integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
               cute::integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;

// Overlap-epilogue t2r atom (column repeat per tcgen05.ld; B55 trap knob).
#if DUAL2SM_EPI_ATOM == 1
using EpiLoadOp = SM100_TMEM_LOAD_32dp32b1x;
#elif DUAL2SM_EPI_ATOM == 2
using EpiLoadOp = SM100_TMEM_LOAD_32dp32b2x;
#elif DUAL2SM_EPI_ATOM == 4
using EpiLoadOp = SM100_TMEM_LOAD_32dp32b4x;
#elif DUAL2SM_EPI_ATOM == 8
using EpiLoadOp = SM100_TMEM_LOAD_32dp32b8x;
#elif DUAL2SM_EPI_ATOM == 16
using EpiLoadOp = SM100_TMEM_LOAD_32dp32b16x;
#elif DUAL2SM_EPI_ATOM == 32
using EpiLoadOp = SM100_TMEM_LOAD_32dp32b32x;
#elif DUAL2SM_EPI_ATOM == 64
using EpiLoadOp = SM100_TMEM_LOAD_32dp32b64x;
#elif DUAL2SM_EPI_ATOM == 128
using EpiLoadOp = SM100_TMEM_LOAD_32dp32b128x;
#else
#error "DUAL2SM_EPI_ATOM must be 1|2|4|8|16|32|64|128"
#endif

#if DUAL2SM_TMA_EPI
// V7 lever (h): the D store descriptor is built RAW (cuTensorMapEncodeTiled,
// the nvfp4-lab pattern) instead of through cute's make_tma_copy: the cute
// path's element-space swizzle conventions (get_tma_swizzle_base demands a
// byte-space Swizzle<3,4,3>) don't compose cleanly with an fp32 staging
// tile, and nothing else on the store path needs the layout algebra. Box =
// (n=32, m=128) fp32 = 128B x 128 rows = 16KB, CU_TENSOR_MAP_SWIZZLE_128B:
// the engine reads smem expecting the PHYSICAL 128B-swizzle pattern
// (16B-group g at row r lives at r*128B + (g ^ (r & 7))*16B from the box
// base), which is exactly what the staging writes.
static CUtensorMap make_tma_desc_D(void const* d_ptr, int64_t m, int64_t n) {
  CUtensorMap desc{};
  uint64_t global_dim[2] = {uint64_t(n), uint64_t(m)};          // dim0 = n (stride 1)
  uint64_t global_strides[1] = {uint64_t(n) * sizeof(TypeD)};   // dim1 byte stride
#if DUAL2SM_D_HALF
  // fp16 D: 64 cols x 2B = the same 128B box rows; chunk PAIRS per store.
  uint32_t box_dim[2] = {64, 128};
  constexpr CUtensorMapDataType kDDtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
#else
  uint32_t box_dim[2] = {32, 128};                              // 128B x 128 rows
  constexpr CUtensorMapDataType kDDtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
#endif
  uint32_t elem_strides[2] = {1, 1};
  CUresult res = cuTensorMapEncodeTiled(
      &desc, kDDtype, 2, const_cast<void*>(d_ptr),
      global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled(D) failed: ", int(res));
  return desc;
}
#endif

// Compile-config tags: configs that differ ONLY in body-level macros (ws,
// mb, eo, ea, persist, raster_gm) would otherwise instantiate the SAME
// mangled kernel symbol across separately-built .so's -- the B61 in-process
// alias trap. The tags force a distinct symbol per config.
constexpr int kCfgMain = DUAL2SM_WARP_SPLIT | (DUAL2SM_AMCAST << 1) | (DUAL2SM_MIN_BLOCKS << 2);
constexpr int kCfgEpi = DUAL2SM_EPI_OVERLAP | (DUAL2SM_EPI_ATOM << 8);
constexpr int kCfgV5 = DUAL2SM_PERSIST | (DUAL2SM_RASTER_GM << 1) |
                       (DUAL2SM_PREFETCH << 8) | (DUAL2SM_PF_ISSUER << 12) |
                       (DUAL2SM_TMA_EPI << 14) | (DUAL2SM_D_HALF << 15);

template <class SharedStorageT, int CfgMain, int CfgEpi, int CfgV5,
          class ATensor, class BTensor, class CTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA,
          class TmaAtomA, class TmaAtomB>
#if DUAL2SM_AMCAST
__global__ void __cluster_dims__(2, 2, 1) __launch_bounds__(kNumThreads, DUAL2SM_MIN_BLOCKS)
#else
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(kNumThreads, DUAL2SM_MIN_BLOCKS)
#endif
gemm_dual_2sm_fp8(ATensor mA, BTensor mB, CTensor mC, DTensor mD,
                  MmaTiler_MNK mma_tiler, TiledMMA tiled_mma,
                  CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
                  CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B
#if DUAL2SM_TMA_EPI
                  , CUTE_GRID_CONSTANT CUtensorMap const tma_desc_D
#endif
                  ) {

  uint32_t elect_one_thr = cute::elect_one_sync();
  const int warp_idx = int(threadIdx.x / 32);
  uint32_t elect_one_warp = (warp_idx == 0);

  int v = int(cute::block_rank_in_cluster()) & 1;  // peer rank within the SM pair
  bool leader_cta = (v == 0);                      // even rank issues the 2SM MMA

#if DUAL2SM_PERSIST || DUAL2SM_RASTER_GM > 0
  const int grid_m_t = int(size<0>(shape(mA))) / kTileM;  // pair-rows
  const int grid_n_t = int(size<0>(shape(mB))) / kTileN;  // n-cols
#endif
#if DUAL2SM_PERSIST
  // Persistent: real tile coords come from the round loop below; (0,0)
  // builds the shape templates (all partition shapes are coord-invariant).
  int tile_m = 0;
  int tile_n = 0;
  const int num_clusters = int(gridDim.x) >> 1;
  const int rr = int(blockIdx.x) >> 1;                   // this cluster's id
  const int total_tiles = grid_m_t * grid_n_t;
  const int my_tiles = (total_tiles - rr + num_clusters - 1) / num_clusters;
#elif DUAL2SM_RASTER_GM > 0
  // 1D-flattened grid: blockIdx.x = (raster index, v) interleaved.
  int tile_m, tile_n;
  raster_map(int(blockIdx.x) >> 1, grid_m_t, grid_n_t, tile_m, tile_n);
#else
  int tile_m = blockIdx.x / 2;  // pair-wide 256-row M tile
  // Under EPI_OVERLAP each cluster walks kTilesPerCta CONSECUTIVE n-tiles
  // (tile_n .. tile_n+kTilesPerCta-1); A is shared across the walk (L2-hot).
  int tile_n = blockIdx.y * kTilesPerCta;
#endif

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
      // empty: one multicast tcgen05.commit arrival per pair-leader. Under
      // AMCAST a stage's A buffer is written by BOTH pairs' multicast TMAs,
      // so reuse must wait for BOTH leaders' commits (count kClusterN).
      cute::initialize_barrier(storage.empty_barrier[s], kClusterN);
    }
    for (int bb = 0; bb < kNumAccBufs; ++bb) {
      cute::initialize_barrier(storage.mma_barrier[bb], 1);
      if constexpr (kEpiOverlap || kPersist) {
        // Leader-side acc-buffer-free barrier: BOTH CTAs' epilogues arrive
        // (cluster-scope) once their TMEM half of the buffer is drained;
        // the consumer waits before reusing the buffer (under PERSIST
        // eo=0 this gates the next round's first MMA on both drains).
        cute::initialize_barrier(storage.tmem_empty[bb], 2);
      }
    }
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

#if DUAL2SM_AMCAST
  // (2,2,1) cluster: the two SM pairs of a cluster share tile_m, so A is
  // TMA-multicast across the cluster's N mode (vmnk mode 2). Each CTA
  // issues HALF of its A-half, masked to the same-v CTA of the other pair;
  // each CTA's smem still receives its full A-half (own slice + partner's
  // multicast slice), but A's L2 reads halve.
  auto cluster_layout_vmnk = tiled_divide(make_layout(Shape<_2, Int<kClusterN>, _1>{}),
                                          make_tile(typename TiledMMA::AtomThrID{}));
  auto cta_coord_vmnk = cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
  uint16_t mcast_mask_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_coord_vmnk);
  Tensor tCsA_0 = storage.tensor_sA(0);
  auto [tAgA, tAsA_0] = tma_partition(tma_atom_A,
      get<2>(cta_coord_vmnk), make_layout(size<2>(cluster_layout_vmnk)),
      group_modes<0,3>(tCsA_0), group_modes<0,3>(tCgCoordA));
#else
  // Non-multicast partition (coord 0 / layout 1): the V-mode pair split is
  // already baked into tCgCoordA/B by partition_A/B above, so per-stage
  // smem tensors may be rebuilt by base-pointer swap (no iterator offset).
  Tensor tCsA_0 = storage.tensor_sA(0);
  auto [tAgA, tAsA_0] = tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsA_0), group_modes<0,3>(tCgCoordA));
#endif
  Tensor tCsB_0 = storage.tensor_sB(0);
  auto [tBgB, tBsB_0] = tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsB_0), group_modes<0,3>(tCgCoordB));

  // The leader's full barrier counts 2 x (per-issue A slice + B-half) per
  // stage (tutorial 04_mma_tma_2sm formula: size<0>(cluster_layout_vmnk) x
  // sizeof each tma_partition slice). Under AMCAST the A slice is 1/kClusterN
  // of the A-half and there is NO participant multiplier: cta_group::2
  // multicast complete-tx counts only the issuing slice once per destination
  // pair (a kClusterN multiplier over-expects by A-half/stage and the barrier
  // never fires -> the V-front 25-min hang; the B53 expect-tx trap).
  int tma_bytes = 2 * (sizeof(make_tensor_like(tAsA_0)) + sizeof(make_tensor_like(tBsB_0)));

#if DUAL2SM_AMCAST
  // The multicast slice has a CTA-dependent offset inside smem_A[stage], so
  // per-stage smem tensors must come from tma_partition itself (the
  // base-pointer-swap trick would drop the slice offset).
  auto sliceA = [&](int s) {
    Tensor tCsA_s = storage.tensor_sA(s);
    return get<1>(tma_partition(tma_atom_A,
        get<2>(cta_coord_vmnk), make_layout(size<2>(cluster_layout_vmnk)),
        group_modes<0,3>(tCsA_s), group_modes<0,3>(tCgCoordA)));
  };
#if DUAL2SM_STAGES == 2
  decltype(sliceA(0)) tAsA_st[kStages] = { sliceA(0), sliceA(1) };
#elif DUAL2SM_STAGES == 3
  decltype(sliceA(0)) tAsA_st[kStages] = { sliceA(0), sliceA(1), sliceA(2) };
#elif DUAL2SM_STAGES == 4
  decltype(sliceA(0)) tAsA_st[kStages] = { sliceA(0), sliceA(1), sliceA(2), sliceA(3) };
#elif DUAL2SM_STAGES == 5
  decltype(sliceA(0)) tAsA_st[kStages] = { sliceA(0), sliceA(1), sliceA(2), sliceA(3), sliceA(4) };
#else
  decltype(sliceA(0)) tAsA_st[kStages] = { sliceA(0), sliceA(1), sliceA(2), sliceA(3), sliceA(4), sliceA(5) };
#endif
#endif

  auto issue_tma = [&](int stage, int k_tile) {
    if (leader_cta) {
      cute::set_barrier_transaction_bytes(storage.full_barrier[stage], tma_bytes);
    }
#if DUAL2SM_AMCAST
    copy(tma_atom_A.with(storage.full_barrier[stage], mcast_mask_a), tAgA(_, k_tile), tAsA_st[stage]);
#else
    auto tAsA = make_tensor(make_smem_ptr(storage.smem_A[stage].begin()), tAsA_0.layout());
    copy(tma_atom_A.with(storage.full_barrier[stage]), tAgA(_, k_tile), tAsA);
#endif
    auto tBsB = make_tensor(make_smem_ptr(storage.smem_B[stage].begin()), tBsB_0.layout());
    copy(tma_atom_B.with(storage.full_barrier[stage]), tBgB(_, k_tile), tBsB);
  };

#if DUAL2SM_PERSIST
  // Per-round gmem TMA coordinate slices: under the persistent raster walk
  // BOTH tile_m and tile_n vary per round; the layout algebra is static so
  // a slice is a few integer ops, recomputed once per ~33us tile.
  auto tAgA_for = [&](int tm, int tn) {
    auto coord_t = make_coord(tm, tn, _);
    Tensor gCoordA_t = local_tile(tma_coord_A, mma_tiler, coord_t, Step<_1, X, _1>{});
    Tensor tCgCoordA_t = cta_mma.partition_A(gCoordA_t);
    return cute::get<0>(tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{},
        group_modes<0,3>(tCsA_0), group_modes<0,3>(tCgCoordA_t)));
  };
  auto tBgB_dyn = [&](int tm, int tn) {
    auto coord_t = make_coord(tm, tn, _);
    Tensor gCoordB_t = local_tile(tma_coord_B, mma_tiler, coord_t, Step<X, _1, _1>{});
    Tensor tCgCoordB_t = cta_mma.partition_B(gCoordB_t);
    return cute::get<0>(tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{},
        group_modes<0,3>(tCsB_0), group_modes<0,3>(tCgCoordB_t)));
  };
#elif DUAL2SM_EPI_OVERLAP
  // Per-walk-tile B gmem TMA coordinate slices (A is tile_m-only: shared).
  auto tBgB_for = [&](int t) {
    auto coord_t = make_coord(tile_m, tile_n + t, _);
    Tensor gCoordB_t = local_tile(tma_coord_B, mma_tiler, coord_t, Step<X, _1, _1>{});
    Tensor tCgCoordB_t = cta_mma.partition_B(gCoordB_t);
    return cute::get<0>(tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{},
        group_modes<0,3>(tCsB_0), group_modes<0,3>(tCgCoordB_t)));
  };
#if DUAL2SM_EPI_OVERLAP == 2
  decltype(tBgB_for(0)) tBgB_t[kTilesPerCta] = { tBgB_for(0), tBgB_for(1) };
#elif DUAL2SM_EPI_OVERLAP == 4
  decltype(tBgB_for(0)) tBgB_t[kTilesPerCta] = { tBgB_for(0), tBgB_for(1), tBgB_for(2), tBgB_for(3) };
#else
  decltype(tBgB_for(0)) tBgB_t[kTilesPerCta] = { tBgB_for(0), tBgB_for(1), tBgB_for(2), tBgB_for(3),
                                                 tBgB_for(4), tBgB_for(5), tBgB_for(6), tBgB_for(7) };
#endif

  auto issue_tma_t = [&](int stage, int t, int k_tile) {
    if (leader_cta) {
      cute::set_barrier_transaction_bytes(storage.full_barrier[stage], tma_bytes);
    }
    auto tAsA = make_tensor(make_smem_ptr(storage.smem_A[stage].begin()), tAsA_0.layout());
    copy(tma_atom_A.with(storage.full_barrier[stage]), tAgA(_, k_tile), tAsA);
    auto tBsB = make_tensor(make_smem_ptr(storage.smem_B[stage].begin()), tBsB_0.layout());
    copy(tma_atom_B.with(storage.full_barrier[stage]), tBgB_t[t](_, k_tile), tBsB);
  };
#endif

  // Per-stage UMMA smem-descriptor fragments (same type per stage -> array).
  auto fragA = [&](int s) { return cta_mma.make_fragment_A(storage.tensor_sA(s)); };
  auto fragB = [&](int s) { return cta_mma.make_fragment_B(storage.tensor_sB(s)); };
#if DUAL2SM_STAGES == 2
  decltype(fragA(0)) tCrA_st[kStages] = { fragA(0), fragA(1) };
  decltype(fragB(0)) tCrB_st[kStages] = { fragB(0), fragB(1) };
#elif DUAL2SM_STAGES == 3
  decltype(fragA(0)) tCrA_st[kStages] = { fragA(0), fragA(1), fragA(2) };
  decltype(fragB(0)) tCrB_st[kStages] = { fragB(0), fragB(1), fragB(2) };
#elif DUAL2SM_STAGES == 4
  decltype(fragA(0)) tCrA_st[kStages] = { fragA(0), fragA(1), fragA(2), fragA(3) };
  decltype(fragB(0)) tCrB_st[kStages] = { fragB(0), fragB(1), fragB(2), fragB(3) };
#elif DUAL2SM_STAGES == 5
  decltype(fragA(0)) tCrA_st[kStages] = { fragA(0), fragA(1), fragA(2), fragA(3), fragA(4) };
  decltype(fragB(0)) tCrB_st[kStages] = { fragB(0), fragB(1), fragB(2), fragB(3), fragB(4) };
#else
  decltype(fragA(0)) tCrA_st[kStages] = { fragA(0), fragA(1), fragA(2), fragA(3), fragA(4), fragA(5) };
  decltype(fragB(0)) tCrB_st[kStages] = { fragB(0), fragB(1), fragB(2), fragB(3), fragB(4), fragB(5) };
#endif

  int num_k_tiles = size<3>(tCgA);

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  // empty release: every CTA whose stage smem fed a consuming MMA. Under
  // AMCAST the multicast A loads also wrote the OTHER pair's smem, so both
  // pair-leaders commit to all four CTAs (empty init count = kClusterN).
  constexpr uint16_t kEmptyMask = DUAL2SM_AMCAST ? 0xF : 0x3;
  // This CTA's own SM pair (for the final TMEM-complete commit).
  const uint16_t kPairMask = uint16_t(0x3) << (int(cute::block_rank_in_cluster()) & ~1);

#if DUAL2SM_PERSIST && DUAL2SM_EPI_OVERLAP
  // =====================================================================
  // V5 persistent mainloop, eo=2 inner loop (lever f over lever e): the
  // V4 epilogue-warpgroup + double-TMEM cross-tile pipeline IS the
  // persistent inner structure; only the tile sequence changed -- cluster
  // rr walks raster indices rr, rr+C, rr+2C... (GROUP_M-swizzled), so the
  // smem stage ring crosses tile boundaries with no prologue bubble and
  // the in-flight window stays ~C consecutive raster indices.
  // =====================================================================
  const uint32_t leader_rank = uint32_t(cute::block_rank_in_cluster()) & ~1u;
  if (warp_idx == 0) {
    // Producer: WHOLE warp 0 of BOTH CTAs (empty-wait + TMA issue).
    int empty_phase[kStages] = {};
    int g = 0;
    for (int t = 0; t < my_tiles; ++t) {
      int tm, tn;
      raster_map(rr + t * num_clusters, grid_m_t, grid_n_t, tm, tn);
      auto tAgA_t = tAgA_for(tm, tn);
      auto tBgB_t = tBgB_dyn(tm, tn);
      for (int k = 0; k < num_k_tiles; ++k, ++g) {
        int s = g % kStages;
        if (g >= kStages) {
          cute::wait_barrier(storage.empty_barrier[s], empty_phase[s]);
          empty_phase[s] ^= 1;
        }
        if (elect_one_thr) {
          if (leader_cta) {
            cute::set_barrier_transaction_bytes(storage.full_barrier[s], tma_bytes);
          }
          auto tAsA = make_tensor(make_smem_ptr(storage.smem_A[s].begin()), tAsA_0.layout());
          copy(tma_atom_A.with(storage.full_barrier[s]), tAgA_t(_, k), tAsA);
          auto tBsB = make_tensor(make_smem_ptr(storage.smem_B[s].begin()), tBsB_0.layout());
          copy(tma_atom_B.with(storage.full_barrier[s]), tBgB_t(_, k), tBsB);
        }
      }
    }
    // Drain the final multicast commits on BOTH CTAs (smem barrier
    // lifetime, as in the incumbent paths).
    const int total_iters = my_tiles * num_k_tiles;
    for (int g2 = total_iters - min(kStages, total_iters); g2 < total_iters; ++g2) {
      int s = g2 % kStages;
      cute::wait_barrier(storage.empty_barrier[s], empty_phase[s]);
      empty_phase[s] ^= 1;
    }
  } else if (warp_idx == 1 && leader_cta) {
    // Consumer: leader warp 1 (full-wait + MMA + commit). Buffer b = t%2
    // is reused by round t+2: its first MMA waits until BOTH CTAs'
    // epilogues drained it (tmem_empty, cluster-scope arrivals).
    int full_phase[kStages] = {};
    int tmem_phase[kNumAccBufs] = {};
    int g = 0;
    for (int t = 0; t < my_tiles; ++t) {
      int b = t % kNumAccBufs;
      if (t >= kNumAccBufs) {
        cute::wait_barrier(storage.tmem_empty[b], tmem_phase[b]);
        tmem_phase[b] ^= 1;
      }
      tCtAcc.data() = tmem_base + uint32_t(b) * uint32_t(kTileN);
      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
      for (int k = 0; k < num_k_tiles; ++k, ++g) {
        int s = g % kStages;
        cute::wait_barrier(storage.full_barrier[s], full_phase[s]);
        full_phase[s] ^= 1;

        for (int kb = 0; kb < size<2>(tCrA_st[0]); ++kb) {
          gemm(tiled_mma, tCrA_st[s](_, _, kb), tCrB_st[s](_, _, kb), tCtAcc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive_multicast_2x1SM(
            reinterpret_cast<uint64_t*>(&storage.empty_barrier[s]), kEmptyMask);
      }
      // Round t complete in BOTH CTAs' TMEM halves: release the pair's
      // mma_barrier[b] so both epilogue warpgroups can drain it.
      cutlass::arch::umma_arrive_multicast_2x1SM(
          reinterpret_cast<uint64_t*>(&storage.mma_barrier[b]), kPairMask);
    }
  } else if (warp_idx >= 4) {
    // Dedicated epilogue warpgroup (BOTH CTAs): drains buffer t%2 + stores
    // the round's raster tile while the consumer fills the other buffer.
    const int epi_tid = int(threadIdx.x) - 128;
    Tensor tEpiAcc = cta_mma.make_fragment_C(tCgC);
    auto tiled_t2r_copy = make_tmem_copy(EpiLoadOp{}, tEpiAcc);
    auto thr_t2r_copy = tiled_t2r_copy.get_slice(epi_tid);
    int mma_phase[kNumAccBufs] = {};
#if DUAL2SM_TMA_EPI
    int epi_store_q = 0;  // global chunk-pair store counter (slot = q & 1)
#endif
    for (int t = 0; t < my_tiles; ++t) {
      int b = t % kNumAccBufs;
      cute::wait_barrier(storage.mma_barrier[b], mma_phase[b]);
      mma_phase[b] ^= 1;
      tEpiAcc.data() = tmem_base + uint32_t(b) * uint32_t(kTileN);

      int tm, tn;
      raster_map(rr + t * num_clusters, grid_m_t, grid_n_t, tm, tn);
      auto coord_t = make_coord(tm, tn, _);
      Tensor gD_t = local_tile(mD, mma_tiler, coord_t, Step<_1, _1, X>{});
      Tensor tCgD_t = cta_mma.partition_C(gD_t);
      Tensor tDtAcc = thr_t2r_copy.partition_S(tEpiAcc);   // (CPY, rest...)
      Tensor tDgD = thr_t2r_copy.partition_D(tCgD_t);      // (CPY, rest...)
      constexpr int kEpiRank = decltype(rank(tDtAcc))::value;
      Tensor tDtAccG = group_modes<1, kEpiRank>(tDtAcc);   // (CPY, REST)
      Tensor tDgDG = group_modes<1, kEpiRank>(tDgD);       // (CPY, REST)
#if DUAL2SM_TMA_EPI
      // F8c staged fp16 TMA store (the F8b te1+dh1 path, re-hosted in the
      // overlap epilogue): each PAIR of 32-col fp32 t2r chunks packs
      // (__floats2half2_rn) into one 128x64 fp16 SW128 box in a DEDICATED
      // staging buffer (the ring smem is live with the overlapped
      // mainloop and never drains). Sync is warpgroup-local bar.sync 1,128
      // -- __syncthreads would deadlock against the mainloop warps. The
      // B58 crux: staging-slot reuse waits tma_store_wait<1> (engine has
      // READ the box issued two stores ago) on the ISSUING thread, then
      // the named barrier republishes that to all 128 epi threads.
      constexpr int kDStores = kTileN / 64;  // chunk-pair stores per tile
      const int row0 = (tm * 2 + v) * 128;   // 2x1SM CLayout: CTA v owns
                                             // contiguous D rows
      const int col0 = tn * kTileN;
      const int r = epi_tid;                 // 32dp32b t2r: thread == row
      auto pack2 = [](float x, float y) -> uint32_t {
        __half2 h = __floats2half2_rn(x, y);
        return *reinterpret_cast<uint32_t*>(&h);
      };
      CUTE_UNROLL
      for (int c2 = 0; c2 < kDStores; ++c2) {
        Tensor tDrAcc0 = make_tensor<Accumulator>(shape(tDgDG(_, 2 * c2)));
        Tensor tDrAcc1 = make_tensor<Accumulator>(shape(tDgDG(_, 2 * c2 + 1)));
        copy(tiled_t2r_copy, tDtAccG(_, 2 * c2), tDrAcc0);
        copy(tiled_t2r_copy, tDtAccG(_, 2 * c2 + 1), tDrAcc1);
        cutlass::arch::fence_view_async_tmem_load();
        const int slot = epi_store_q & 1;
        if (epi_store_q >= 2) {
          if (warp_idx == 4 && elect_one_thr) {
            cute::tma_store_wait<1>();
          }
          asm volatile("bar.sync 1, 128;" ::: "memory");
        }
        uint4* sbuf = reinterpret_cast<uint4*>(storage.epi_stage[slot]);
        CUTE_UNROLL
        for (int g = 0; g < 8; ++g) {
          // Groups 0-3 = chunk 2*c2 (fp16 cols 0-31), groups 4-7 = chunk
          // 2*c2+1 (cols 32-63); g is compile-time so the branch folds.
          int o = 8 * (g & 3);
          uint4 packed;
          if (g < 4) {
            packed = make_uint4(pack2(tDrAcc0(o + 0), tDrAcc0(o + 1)),
                                pack2(tDrAcc0(o + 2), tDrAcc0(o + 3)),
                                pack2(tDrAcc0(o + 4), tDrAcc0(o + 5)),
                                pack2(tDrAcc0(o + 6), tDrAcc0(o + 7)));
          } else {
            packed = make_uint4(pack2(tDrAcc1(o + 0), tDrAcc1(o + 1)),
                                pack2(tDrAcc1(o + 2), tDrAcc1(o + 3)),
                                pack2(tDrAcc1(o + 4), tDrAcc1(o + 5)),
                                pack2(tDrAcc1(o + 6), tDrAcc1(o + 7)));
          }
          sbuf[r * 8 + (g ^ (r & 7))] = packed;   // SW128 16B-group XOR
        }
        cute::tma_store_fence();   // STS visible to the async proxy
        asm volatile("bar.sync 1, 128;" ::: "memory");  // box fully staged
        if (warp_idx == 4 && elect_one_thr) {
          uint32_t s_ptr = cute::cast_smem_ptr_to_uint(storage.epi_stage[slot]);
          int32_t crd_n = col0 + c2 * 64;  // descriptor dim0 = n (fp16 elems)
          int32_t crd_m = row0;            // descriptor dim1 = m
          asm volatile(
              "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
              " [%0, {%2, %3}], [%1];"
              :
              : "l"(reinterpret_cast<uint64_t>(&tma_desc_D)), "r"(s_ptr),
                "r"(crd_n), "r"(crd_m)
              : "memory");
          cute::tma_store_arrive();
        }
        ++epi_store_q;
      }
#else
      CUTE_UNROLL
      for (int c = 0; c < size<1>(tDtAccG); ++c) {
        Tensor tDrAcc = make_tensor<Accumulator>(shape(tDgDG(_, c)));
        copy(tiled_t2r_copy, tDtAccG(_, c), tDrAcc);
        cutlass::arch::fence_view_async_tmem_load();
        copy(tDrAcc, tDgDG(_, c));  // D = accumulator (beta=0)
      }
#endif  // DUAL2SM_TMA_EPI
      if (t + kNumAccBufs < my_tiles) {
        // Buffer b will be reused: signal the LEADER's consumer that this
        // CTA's TMEM half is drained (skipped for the last kNumAccBufs
        // rounds so no remote arrive lands after the leader's smem
        // barriers die -- the B49/V2 lifetime trap).
        cutlass::arch::ClusterBarrier::arrive(
            &storage.tmem_empty[b], leader_rank,
            uint32_t((warp_idx == 4) && elect_one_thr));
      }
    }
#if DUAL2SM_TMA_EPI
    // All staging boxes must be engine-READ before the CTA's smem dies
    // (wait_group.read; gmem landing is covered by kernel completion --
    // the banked te1 semantics).
    if (warp_idx == 4 && elect_one_thr) {
      cute::tma_store_wait<0>();
    }
#endif
  }
#elif DUAL2SM_PERSIST
  // =====================================================================
  // V5 persistent mainloop, eo=0 inner loop: champion occupancy (3
  // CTAs/SM at (128,3)) is kept -- each round runs the warp-split
  // producer/consumer k-loop, then ALL 128 threads drain+store the single
  // TMEM buffer (serialized per round; co-residency hides it, the V4
  // finding). TMEM reuse across rounds is gated by tmem_empty: both CTAs
  // arrive after their drain, the leader's consumer waits before round
  // t+1's first MMA. The t2r drain is CHUNKED (EpiLoadOp) so the live
  // range fits the 170-reg budget that 3 CTAs/SM x 128 threads imposes.
  // =====================================================================
  const uint32_t leader_rank = uint32_t(cute::block_rank_in_cluster()) & ~1u;
  int empty_phase[kStages] = {};
  int full_phase[kStages] = {};
  int tmem_phase = 0;
#if DUAL2SM_PREFETCH > 0
  // Lever (g): L2-prefetch round t_next's leading k-stage panels via the
  // TMA atoms' PREFETCH op (cp.async.bulk.prefetch.tensor on the SAME
  // descriptors + coord slices as the round's eventual demand loads; each
  // CTA covers its own issue slices, the pair covers the full boxes).
  auto prefetch_round = [&](int t_next) {
    if (t_next < my_tiles) {
      int ptm, ptn;
      raster_map(rr + t_next * num_clusters, grid_m_t, grid_n_t, ptm, ptn);
      auto tAgA_p = tAgA_for(ptm, ptn);
      auto tBgB_p = tBgB_dyn(ptm, ptn);
      CUTE_UNROLL
      for (int pk = 0; pk < DUAL2SM_PREFETCH; ++pk) {
        cute::prefetch(tma_atom_A, tAgA_p(_, pk));
        cute::prefetch(tma_atom_B, tBgB_p(_, pk));
      }
    }
  };
#endif
  auto tiled_t2r_copy = make_tmem_copy(EpiLoadOp{}, tCtAcc);
  auto thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);
  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);        // (CPY, rest...)
  constexpr int kEpiRank = decltype(rank(tDtAcc))::value;
  Tensor tDtAccG = group_modes<1, kEpiRank>(tDtAcc);       // (CPY, REST)
  for (int t = 0; t < my_tiles; ++t) {
    int tm, tn;
    raster_map(rr + t * num_clusters, grid_m_t, grid_n_t, tm, tn);
#if DUAL2SM_PREFETCH > 0 && DUAL2SM_PF_ISSUER == 2
    // Issuer 2: parked warp 3, round-t start (max lead, eviction risk).
    if (warp_idx == 3 && elect_one_thr) {
      prefetch_round(t + 1);
    }
#endif
    if (warp_idx == 0) {
      // Producer: WHOLE warp 0 of BOTH CTAs. The ring fully drains at
      // each round boundary (the commit-drain loop below), so the first
      // kStages issues of a round need no empty-wait.
      auto tAgA_t = tAgA_for(tm, tn);
      auto tBgB_t = tBgB_dyn(tm, tn);
      for (int k = 0; k < num_k_tiles; ++k) {
        int s = k % kStages;
        if (k >= kStages) {
          cute::wait_barrier(storage.empty_barrier[s], empty_phase[s]);
          empty_phase[s] ^= 1;
        }
        if (elect_one_thr) {
          if (leader_cta) {
            cute::set_barrier_transaction_bytes(storage.full_barrier[s], tma_bytes);
          }
          auto tAsA = make_tensor(make_smem_ptr(storage.smem_A[s].begin()), tAsA_0.layout());
          copy(tma_atom_A.with(storage.full_barrier[s]), tAgA_t(_, k), tAsA);
          auto tBsB = make_tensor(make_smem_ptr(storage.smem_B[s].begin()), tBsB_0.layout());
          copy(tma_atom_B.with(storage.full_barrier[s]), tBgB_t(_, k), tBsB);
        }
      }
#if DUAL2SM_PREFETCH > 0 && DUAL2SM_PF_ISSUER == 0
      // Issuer 0: producer, right after round t's LAST TMA issue -- the
      // commit-drain below blocks until the consumer catches up, so the
      // prefetch gets that whole window of lead time.
      if (elect_one_thr) {
        prefetch_round(t + 1);
      }
#endif
      // Drain this round's final multicast commits on BOTH CTAs.
      for (int k = num_k_tiles - min(kStages, num_k_tiles); k < num_k_tiles; ++k) {
        int s = k % kStages;
        cute::wait_barrier(storage.empty_barrier[s], empty_phase[s]);
        empty_phase[s] ^= 1;
      }
    } else if (warp_idx == 1 && leader_cta) {
      // Consumer: TMEM buffer reuse waits for BOTH CTAs' round-(t-1)
      // drains (cluster-scope tmem_empty arrivals).
      if (t >= 1) {
        cute::wait_barrier(storage.tmem_empty[0], tmem_phase);
        tmem_phase ^= 1;
      }
      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
      for (int k = 0; k < num_k_tiles; ++k) {
        int s = k % kStages;
        cute::wait_barrier(storage.full_barrier[s], full_phase[s]);
        full_phase[s] ^= 1;

        for (int kb = 0; kb < size<2>(tCrA_st[0]); ++kb) {
          gemm(tiled_mma, tCrA_st[s](_, _, kb), tCrB_st[s](_, _, kb), tCtAcc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive_multicast_2x1SM(
            reinterpret_cast<uint64_t*>(&storage.empty_barrier[s]), kEmptyMask);
      }
      cutlass::arch::umma_arrive_multicast_2x1SM(
          reinterpret_cast<uint64_t*>(&storage.mma_barrier[0]), kPairMask);
    }

    // Per-round epilogue: ALL 128 threads of both CTAs (one multicast
    // commit per round -> mma_barrier phase alternates with t).
    cute::wait_barrier(storage.mma_barrier[0], t & 1);
#if DUAL2SM_PREFETCH > 0 && DUAL2SM_PF_ISSUER == 1
    // Issuer 1: epilogue-parked warp 2, right after the round's MMAs
    // complete (latest issue point, shortest lead).
    if (warp_idx == 2 && elect_one_thr) {
      prefetch_round(t + 1);
    }
#endif
#if DUAL2SM_TMA_EPI
    {
      // V7 lever (h): stage each (128 x 32) fp32 chunk of this CTA's D
      // half through the DRAINED ring stage smem_A[c % kStages], then ONE
      // cp.async.bulk.tensor.2d store per chunk. CLayout of the 2x1SM
      // atom (Stride<Int<M/2>, ...> on the thr mode): CTA v owns the
      // CONTIGUOUS D rows [tm*256 + v*128, +128).
      constexpr int kDChunks = kTileN / 32;
      const int row0 = (tm * 2 + v) * 128;  // D row base (the 2x1SM CLayout
                                            // gives CTA v the CONTIGUOUS rows
                                            // [tm*256 + v*128, +128))
      const int col0 = tn * kTileN;         // D col base
      const int r = int(threadIdx.x);       // 32dp32b t2r: thread == row
      // D-side partition used for the register-fragment SHAPE only (the
      // t2r CopyAtom requires a dst layout that vectorizes into regs);
      // no stores go through it under TMA_EPI.
      auto coord_t = make_coord(tm, tn, _);
      Tensor gD_t = local_tile(mD, mma_tiler, coord_t, Step<_1, _1, X>{});
      Tensor tCgD_t = cta_mma.partition_C(gD_t);
      Tensor tDgD = thr_t2r_copy.partition_D(tCgD_t);      // (CPY, rest...)
      Tensor tDgDG = group_modes<1, kEpiRank>(tDgD);       // (CPY, REST)
#if DUAL2SM_D_HALF
      // F8b D_HALF: each store consumes a PAIR of 32-col fp32 t2r chunks
      // converted (__float2half_rn) into one 128x64 fp16 staged tile --
      // the same 128B rows, the same SW128 16B-group XOR, half the D
      // bytes, and NO host-side conversion kernel afterwards.
      constexpr int kDStores = kDChunks / 2;
      auto pack2 = [](float x, float y) -> uint32_t {
        __half2 h = __floats2half2_rn(x, y);
        return *reinterpret_cast<uint32_t*>(&h);
      };
      CUTE_UNROLL
      for (int c2 = 0; c2 < kDStores; ++c2) {
        Tensor tDrAcc0 = make_tensor<Accumulator>(shape(tDgDG(_, 2 * c2)));
        Tensor tDrAcc1 = make_tensor<Accumulator>(shape(tDgDG(_, 2 * c2 + 1)));
        copy(tiled_t2r_copy, tDtAccG(_, 2 * c2), tDrAcc0);
        copy(tiled_t2r_copy, tDtAccG(_, 2 * c2 + 1), tDrAcc1);
        cutlass::arch::fence_view_async_tmem_load();
        if (c2 >= kStages) {
          // B58: the TMA store engine is the ring's LAST consumer.
          if (warp_idx == 0 && elect_one_thr) {
            cute::tma_store_wait<kStages - 1>();
          }
          __syncthreads();
        }
        uint4* sbuf = reinterpret_cast<uint4*>(storage.smem_A[c2 % kStages].begin());
        CUTE_UNROLL
        for (int g = 0; g < 8; ++g) {
          // Groups 0-3 = chunk 2*c2 (fp16 cols 0-31 of the 64), groups
          // 4-7 = chunk 2*c2+1 (cols 32-63); g is compile-time (full
          // unroll) so the branch folds.
          int o = 8 * (g & 3);
          uint4 packed;
          if (g < 4) {
            packed = make_uint4(pack2(tDrAcc0(o + 0), tDrAcc0(o + 1)),
                                pack2(tDrAcc0(o + 2), tDrAcc0(o + 3)),
                                pack2(tDrAcc0(o + 4), tDrAcc0(o + 5)),
                                pack2(tDrAcc0(o + 6), tDrAcc0(o + 7)));
          } else {
            packed = make_uint4(pack2(tDrAcc1(o + 0), tDrAcc1(o + 1)),
                                pack2(tDrAcc1(o + 2), tDrAcc1(o + 3)),
                                pack2(tDrAcc1(o + 4), tDrAcc1(o + 5)),
                                pack2(tDrAcc1(o + 6), tDrAcc1(o + 7)));
          }
          sbuf[r * 8 + (g ^ (r & 7))] = packed;
        }
        cute::tma_store_fence();   // STS visible to the async proxy
        __syncthreads();           // whole CTA's chunk pair staged
        if (warp_idx == 0 && elect_one_thr) {
          uint32_t s_ptr = cute::cast_smem_ptr_to_uint(storage.smem_A[c2 % kStages].begin());
          int32_t crd_n = col0 + c2 * 64;  // descriptor dim0 = n (fp16 elems)
          int32_t crd_m = row0;            // descriptor dim1 = m
          asm volatile(
              "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
              " [%0, {%2, %3}], [%1];"
              :
              : "l"(reinterpret_cast<uint64_t>(&tma_desc_D)), "r"(s_ptr),
                "r"(crd_n), "r"(crd_m)
              : "memory");
          cute::tma_store_arrive();
        }
      }
#else
      CUTE_UNROLL
      for (int c = 0; c < kDChunks; ++c) {
        Tensor tDrAcc = make_tensor<Accumulator>(shape(tDgDG(_, c)));
        copy(tiled_t2r_copy, tDtAccG(_, c), tDrAcc);
        cutlass::arch::fence_view_async_tmem_load();
        if (c >= kStages) {
          // Staging-buffer reuse: the TMA store engine is the ring's
          // LAST consumer (the B58 trap) -- chunk c-kStages must be read
          // out of this stage before the CTA overwrites it.
          if (warp_idx == 0 && elect_one_thr) {
            cute::tma_store_wait<kStages - 1>();
          }
          __syncthreads();
        }
        float4* sbuf = reinterpret_cast<float4*>(storage.smem_A[c % kStages].begin());
        CUTE_UNROLL
        for (int g = 0; g < 8; ++g) {
          // SW128 16B-group XOR (g ^ (row & 7)): conflict-free STS.128,
          // and exactly the descriptor's SWIZZLE_128B physical pattern.
          sbuf[r * 8 + (g ^ (r & 7))] =
              make_float4(tDrAcc(4 * g + 0), tDrAcc(4 * g + 1),
                          tDrAcc(4 * g + 2), tDrAcc(4 * g + 3));
        }
        cute::tma_store_fence();   // STS visible to the async proxy
        __syncthreads();           // whole CTA's chunk staged
        if (warp_idx == 0 && elect_one_thr) {
          uint32_t s_ptr = cute::cast_smem_ptr_to_uint(storage.smem_A[c % kStages].begin());
          int32_t crd_n = col0 + c * 32;   // descriptor dim0 = n
          int32_t crd_m = row0;            // descriptor dim1 = m
          asm volatile(
              "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
              " [%0, {%2, %3}], [%1];"
              :
              : "l"(reinterpret_cast<uint64_t>(&tma_desc_D)), "r"(s_ptr),
                "r"(crd_n), "r"(crd_m)
              : "memory");
          cute::tma_store_arrive();
        }
      }
#endif  // DUAL2SM_D_HALF
    }
#else
    {
      auto coord_t = make_coord(tm, tn, _);
      Tensor gD_t = local_tile(mD, mma_tiler, coord_t, Step<_1, _1, X>{});
      Tensor tCgD_t = cta_mma.partition_C(gD_t);
      Tensor tDgD = thr_t2r_copy.partition_D(tCgD_t);      // (CPY, rest...)
      Tensor tDgDG = group_modes<1, kEpiRank>(tDgD);       // (CPY, REST)
      CUTE_UNROLL
      for (int c = 0; c < size<1>(tDtAccG); ++c) {
        Tensor tDrAcc = make_tensor<Accumulator>(shape(tDgDG(_, c)));
        copy(tiled_t2r_copy, tDtAccG(_, c), tDrAcc);
        cutlass::arch::fence_view_async_tmem_load();
        copy(tDrAcc, tDgDG(_, c));  // D = accumulator (beta=0)
      }
    }
#endif  // DUAL2SM_TMA_EPI
    if (t + 1 < my_tiles) {
      // All this CTA's t2r loads must be done before signaling the drain
      // (skipped on the last round -- the B49/V2 smem-lifetime trap).
      __syncthreads();
      cutlass::arch::ClusterBarrier::arrive(
          &storage.tmem_empty[0], leader_rank,
          uint32_t((warp_idx == 0) && elect_one_thr));
    }
#if DUAL2SM_TMA_EPI
    // Smem-reuse gate ONLY: the elected warp-0 thread -- the SAME thread
    // that issues round t+1's TMA loads into smem_A -- drains its store
    // groups (wait_group.read: the engine has READ the staging buffers;
    // gmem landing is not needed for reuse and D is never re-read). The
    // consumer's round t+1 is NOT gated: the tmem_empty arrive above is
    // already sent, and its first MMA transitively waits on this thread's
    // TMA loads anyway.
    if (warp_idx == 0 && elect_one_thr) {
      cute::tma_store_wait<0>();
    }
#endif
  }
#elif DUAL2SM_EPI_OVERLAP
  // =====================================================================
  // V4 cross-tile epilogue-overlap mainloop (lever e). Warpgroup 0 runs
  // the warp-split producer/consumer over the FLATTENED (tile, k)
  // iteration space -- the smem stage ring crosses tile boundaries with
  // no prologue bubble. Warpgroup 1 (warps 4-7, BOTH CTAs) is a
  // dedicated epilogue: it drains TMEM buffer (t%2) + stores tile t
  // while the consumer fills buffer ((t+1)%2) of the next tile.
  // =====================================================================
  const int total_iters = kTilesPerCta * num_k_tiles;
  const uint32_t leader_rank = uint32_t(cute::block_rank_in_cluster()) & ~1u;
  if (warp_idx == 0) {
    // Producer: WHOLE warp 0 of BOTH CTAs (empty-wait + TMA issue).
    int empty_phase[kStages] = {};
    for (int g = 0; g < total_iters; ++g) {
      int s = g % kStages;
      if (g >= kStages) {
        cute::wait_barrier(storage.empty_barrier[s], empty_phase[s]);
        empty_phase[s] ^= 1;
      }
      if (elect_one_thr) {
        issue_tma_t(s, g / num_k_tiles, g % num_k_tiles);
      }
    }
    // Drain the final multicast commits on BOTH CTAs (smem barrier
    // lifetime, as in the incumbent paths).
    for (int g = total_iters - min(kStages, total_iters); g < total_iters; ++g) {
      int s = g % kStages;
      cute::wait_barrier(storage.empty_barrier[s], empty_phase[s]);
      empty_phase[s] ^= 1;
    }
  } else if (warp_idx == 1 && leader_cta) {
    // Consumer: leader warp 1 (full-wait + MMA + commit). Buffer b = t%2
    // is reused by tile t+2: its first MMA must wait until BOTH CTAs'
    // epilogues drained it (tmem_empty, cluster-scope arrivals).
    int full_phase[kStages] = {};
    int tmem_phase[kNumAccBufs] = {};
    for (int t = 0; t < kTilesPerCta; ++t) {
      int b = t % kNumAccBufs;
      if (t >= kNumAccBufs) {
        cute::wait_barrier(storage.tmem_empty[b], tmem_phase[b]);
        tmem_phase[b] ^= 1;
      }
      tCtAcc.data() = tmem_base + uint32_t(b) * uint32_t(kTileN);
      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
      for (int k = 0; k < num_k_tiles; ++k) {
        int curr = (t * num_k_tiles + k) % kStages;
        cute::wait_barrier(storage.full_barrier[curr], full_phase[curr]);
        full_phase[curr] ^= 1;

        for (int kb = 0; kb < size<2>(tCrA_st[0]); ++kb) {
          gemm(tiled_mma, tCrA_st[curr](_, _, kb), tCrB_st[curr](_, _, kb), tCtAcc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive_multicast_2x1SM(
            reinterpret_cast<uint64_t*>(&storage.empty_barrier[curr]), kEmptyMask);
      }
      // Tile t complete in BOTH CTAs' TMEM halves: release the pair's
      // mma_barrier[b] so both epilogue warpgroups can drain it.
      cutlass::arch::umma_arrive_multicast_2x1SM(
          reinterpret_cast<uint64_t*>(&storage.mma_barrier[b]), kPairMask);
    }
  } else if (warp_idx >= 4) {
    // Dedicated epilogue warpgroup (BOTH CTAs): warps 4-7 have ranks 0-3
    // within their warpgroup, so they cover TMEM subpartitions 0-3 and
    // can drain the full 128-lane accumulator concurrently with the
    // mainloop warps. Chunked t2r (32-col fragments) keeps registers
    // inside the 128/thread budget of 256 threads x 2 CTAs/SM.
    const int epi_tid = int(threadIdx.x) - 128;
    Tensor tEpiAcc = cta_mma.make_fragment_C(tCgC);
    auto tiled_t2r_copy = make_tmem_copy(EpiLoadOp{}, tEpiAcc);
    auto thr_t2r_copy = tiled_t2r_copy.get_slice(epi_tid);
    int mma_phase[kNumAccBufs] = {};
    for (int t = 0; t < kTilesPerCta; ++t) {
      int b = t % kNumAccBufs;
      cute::wait_barrier(storage.mma_barrier[b], mma_phase[b]);
      mma_phase[b] ^= 1;
      tEpiAcc.data() = tmem_base + uint32_t(b) * uint32_t(kTileN);

      auto coord_t = make_coord(tile_m, tile_n + t, _);
      Tensor gD_t = local_tile(mD, mma_tiler, coord_t, Step<_1, _1, X>{});
      Tensor tCgD_t = cta_mma.partition_C(gD_t);
      Tensor tDtAcc = thr_t2r_copy.partition_S(tEpiAcc);   // (CPY, rest...)
      Tensor tDgD = thr_t2r_copy.partition_D(tCgD_t);      // (CPY, rest...)
      // Chunk over ALL rest modes (rank varies with atom width/tile_n):
      // one EPI_ATOM-wide t2r fragment -> fence -> gmem store per chunk.
      constexpr int kEpiRank = decltype(rank(tDtAcc))::value;
      Tensor tDtAccG = group_modes<1, kEpiRank>(tDtAcc);   // (CPY, REST)
      Tensor tDgDG = group_modes<1, kEpiRank>(tDgD);       // (CPY, REST)
      CUTE_UNROLL
      for (int c = 0; c < size<1>(tDtAccG); ++c) {
        Tensor tDrAcc = make_tensor<Accumulator>(shape(tDgDG(_, c)));
        copy(tiled_t2r_copy, tDtAccG(_, c), tDrAcc);
        cutlass::arch::fence_view_async_tmem_load();
        copy(tDrAcc, tDgDG(_, c));  // D = accumulator (beta=0)
      }
      if (t + kNumAccBufs < kTilesPerCta) {
        // Buffer b will be reused: signal the LEADER's consumer that this
        // CTA's TMEM half is drained (one cluster-scope arrive per CTA;
        // not signalled for the last kNumAccBufs tiles so no arrive can
        // land after the leader's barriers die -- smem lifetime trap).
        cutlass::arch::ClusterBarrier::arrive(
            &storage.tmem_empty[b], leader_rank,
            uint32_t((warp_idx == 4) && elect_one_thr));
      }
    }
  }
#elif DUAL2SM_WARP_SPLIT
  // =====================================================================
  // Warp-split mainloop (lever a): producer = WHOLE warp 0 of BOTH CTAs
  // (empty-wait + TMA issue); consumer = WHOLE warp 1 of the leader CTA
  // (full-wait + MMA + commit). Decouples the leader's serialized
  // empty-wait -> TMA -> full-wait -> MMA chain so the producer keeps all
  // kStages loads in flight regardless of consumer stalls. Whole-warp
  // roles only (B44 ncu-replay trap). Warps 2-3 park on mma_barrier.
  // =====================================================================
  if (warp_idx == 0) {
    int empty_phase[kStages] = {};
    for (int k = 0; k < num_k_tiles; ++k) {
      int s = k % kStages;
      // Reusing a stage requires the consuming MMA(s) to have drained it
      // (empty barrier <- multicast tcgen05.commit). First kStages issues
      // are free.
      if (k >= kStages) {
        cute::wait_barrier(storage.empty_barrier[s], empty_phase[s]);
        empty_phase[s] ^= 1;
      }
      if (elect_one_thr) {
        issue_tma(s, k);
      }
    }
    // Drain the final multicast commits on BOTH CTAs: every commit
    // multicast at this CTA must be DELIVERED before its smem barriers
    // die with it.
    for (int k = num_k_tiles - min(kStages, num_k_tiles); k < num_k_tiles; ++k) {
      int s = k % kStages;
      cute::wait_barrier(storage.empty_barrier[s], empty_phase[s]);
      empty_phase[s] ^= 1;
    }
  } else if (warp_idx == 1 && leader_cta) {
    int full_phase[kStages] = {};
    for (int k = 0; k < num_k_tiles; ++k) {
      int curr = k % kStages;
      cute::wait_barrier(storage.full_barrier[curr], full_phase[curr]);
      full_phase[curr] ^= 1;

      for (int kb = 0; kb < size<2>(tCrA_st[0]); ++kb) {
        gemm(tiled_mma, tCrA_st[curr](_, _, kb), tCrB_st[curr](_, _, kb), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }

      // tcgen05.commit.cta_group::2 multicast: release the empty barriers
      // of every CTA whose stage-curr smem the pair MMA read (and, under
      // AMCAST, that the multicast loads wrote).
      cutlass::arch::umma_arrive_multicast_2x1SM(
          reinterpret_cast<uint64_t*>(&storage.empty_barrier[curr]), kEmptyMask);
    }

    // Final commit: multicast so the PEER also learns its TMEM half is
    // complete (the pair MMA writes both CTAs' TMEM).
    cutlass::arch::umma_arrive_multicast_2x1SM(
        reinterpret_cast<uint64_t*>(&storage.mma_barrier[0]), kPairMask);
  }
#else
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
            reinterpret_cast<uint64_t*>(&storage.empty_barrier[curr]), kEmptyMask);
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
          reinterpret_cast<uint64_t*>(&storage.mma_barrier[0]), kPairMask);
    }
  }
#endif

#if !DUAL2SM_EPI_OVERLAP && !DUAL2SM_PERSIST
  // All 128 threads of both CTAs rendezvous here for the epilogue.
  cute::wait_barrier(storage.mma_barrier[0], 0);

  // Epilogue: TMEM -> registers -> gmem (beta=0: C never read).
  auto tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  auto thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
  Tensor tDgD = thr_t2r_copy.partition_D(tCgD);
  Tensor tDrAcc = make_tensor<Accumulator>(shape(tDgD));
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);
  cutlass::arch::fence_view_async_tmem_load();

  copy(tDrAcc, tDgD);  // D = accumulator (beta=0)
#endif  // !DUAL2SM_EPI_OVERLAP && !DUAL2SM_PERSIST

  // Under EPI_OVERLAP the dedicated warpgroup already drained + stored all
  // tiles; all warps (8 under overlap, 4 otherwise) rendezvous here.
  __syncthreads();
  if (elect_one_warp) {
    // Pair-collective dealloc (tcgen05.dealloc.cta_group::2): implicit
    // pair rendezvous; both CTAs' warp 0 issue it.
    tmem_allocator.free(tmem_base, kAccTmemCols);
  }
}

torch::Tensor run_dual_2sm_fp8_matmul(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2);
  TORCH_CHECK(a.size(1) == b.size(1));
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn && b.dtype() == torch::kFloat8_e4m3fn,
              "FP8 kernel requires float8_e4m3fn inputs");
  TORCH_CHECK(a.is_cuda() && b.is_cuda());

  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();
  auto m = a_contig.size(0);
  auto k = a_contig.size(1);
  auto n = b_contig.size(0);

  TORCH_CHECK(m % kTileM == 0 && n % kTileN == 0 && k % kTileK == 0,
              "Size must be divisible by the 2SM pair tile (", kTileM, "x", kTileN, "x", kTileK, ")");

  auto options = a.options().dtype(torch::kFloat32);
  auto c_buffer = torch::empty({m, n}, options);  // beta=0: never read
#if DUAL2SM_D_HALF
  // F8b D_HALF: D lands in gmem as fp16 (in-kernel convert); no host-side
  // conversion kernel runs after the GEMM.
  auto d_buffer = torch::empty({m, n}, a.options().dtype(torch::kFloat16));
#else
  auto d_buffer = torch::empty_like(c_buffer);
#endif

  auto tiled_mma = make_tiled_mma(MMA_Atom<MmaTag>{});
  auto bM = tile_size<0>(tiled_mma);              // 256 (pair-wide)
  auto bN = tile_size<1>(tiled_mma);              // kTileN
  // kTileK/32 atom k-blocks per stage: 4 -> kTileK=128 (default; the FP16
  // champion's byte geometry at half the barrier round-trips), 2 -> 64.
  auto bK = tile_size<2>(tiled_mma) * Int<kTileK / 32>{};
  auto mma_tiler = make_shape(bM, bN, bK);

  // Post-partitioned per-CTA shapes: A = 128-row half, B = FULL kTileN.
  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  // e4m3 K-rows are kTileK BYTES: the SW128 atom needs 128B rows, so
  // kTileK=64 takes the SW64 atom (both are legal UMMA descriptor swizzles).
  using SwAtomA = cute::conditional_t<(kTileK % 128 == 0),
      UMMA::Layout_K_SW128_Atom<TypeA>, UMMA::Layout_K_SW64_Atom<TypeA>>;
  using SwAtomB = cute::conditional_t<(kTileK % 128 == 0),
      UMMA::Layout_K_SW128_Atom<TypeB>, UMMA::Layout_K_SW64_Atom<TypeB>>;
  auto sA_layout = UMMA::tile_to_mma_shape(SwAtomA{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(SwAtomB{}, mma_shape_B);

  using SharedStorageT = Dual2SmSharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout)>;

  // Per-CTA smem cap keyed on the requested co-residency (F8b deep-ring
  // lever): MIN_BLOCKS >= 2 keeps the incumbent 110KB cap (per-CTA smem
  // must fit TWICE in the 227KB/SM budget for 2 CTAs/SM). MIN_BLOCKS == 1
  // is the 1-CTA/SM deep-ring build: one CTA owns the whole SM budget, so
  // the cap is the sm_103 per-block dynamic-smem opt-in maximum (227KB);
  // e4m3 makes a 6-stage n256/k128 ring (6 x 32KB = 192KB) fit. The
  // PERSIST grid sizing below derives 1 CTA/SM from the same footprint
  // (B65 static sizing; TMEM 256 of 512 cols is NOT binding at 1 CTA/SM).
#if DUAL2SM_MIN_BLOCKS == 1
  static_assert(sizeof(SharedStorageT) <= 227 * 1024,
                "Shared storage exceeds the sm_103 per-block dynamic smem opt-in cap");
#else
  static_assert(sizeof(SharedStorageT) <= 110 * 1024,
                "Shared storage too large for 2 CTAs/SM; reduce DUAL2SM_STAGES or DUAL2SM_TILE_N");
#endif

  Tensor mA = make_tensor(make_gmem_ptr(reinterpret_cast<TypeA const*>(a_contig.data_ptr())),
      make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
  Tensor mB = make_tensor(make_gmem_ptr(reinterpret_cast<TypeB const*>(b_contig.data_ptr())),
      make_layout(make_shape(n, k), make_stride(k, Int<1>{})));
  Tensor mC = make_tensor(make_gmem_ptr(c_buffer.data_ptr<TypeC>()),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
  Tensor mD = make_tensor(make_gmem_ptr(reinterpret_cast<TypeD*>(d_buffer.data_ptr())),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));

  // Cluster (2, kClusterN, 1): the V mode is the SM pair (AtomThrID = 2);
  // under AMCAST the N mode groups two SM pairs for A-multicast.
  auto cluster_shape = make_shape(Int<2>{}, Int<kClusterN>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));

#if DUAL2SM_AMCAST
  auto tma_atom_A = make_tma_atom_A_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mA, sA_layout,
                                          mma_tiler, tiled_mma, cluster_layout_vmnk);
#else
  auto tma_atom_A = make_tma_atom_A_sm100(SM100_TMA_2SM_LOAD{}, mA, sA_layout,
                                          mma_tiler, tiled_mma, cluster_layout_vmnk);
#endif
  auto tma_atom_B = make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD{}, mB, sB_layout,
                                          mma_tiler, tiled_mma, cluster_layout_vmnk);

#if DUAL2SM_TMA_EPI
  // V7 lever (h): raw TMA store descriptor for D -- (n=32, m=128) fp32
  // box, SWIZZLE_128B; one descriptor for the whole kernel, per-chunk
  // element coords on the device side.
  CUtensorMap tma_desc_D = make_tma_desc_D(d_buffer.data_ptr(), m, n);
#endif

  int grid_pairs_m = int(m) / kTileM;
  int grid_n = int(n) / kTileN;
#if !DUAL2SM_PERSIST
  TORCH_CHECK(grid_n % (kClusterN * kTilesPerCta) == 0,
              "N tiles (", grid_n, ") must be divisible by cluster N x tiles/CTA (",
              kClusterN * kTilesPerCta, ")");
#endif

  dim3 dimBlock(kNumThreads);  // 256 under EPI_OVERLAP (2nd = epilogue warpgroup)
  int smem_bytes = sizeof(SharedStorageT);

  auto* kernel_ptr = &gemm_dual_2sm_fp8<
      SharedStorageT, kCfgMain, kCfgEpi, kCfgV5,
      decltype(mA), decltype(mB), decltype(mC), decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma), decltype(tma_atom_A), decltype(tma_atom_B)>;

  AT_CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  AT_CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributePreferredSharedMemoryCarveout,
                                     cudaSharedmemCarveoutMaxShared));

#if DUAL2SM_PERSIST
  // Persistent grid = exactly the co-residable cluster count, sized
  // STATICALLY as min(smem-implied, TMEM-implied) CTAs/SM. The V5 ncu
  // trap (bankable): cudaOccupancyMaxActiveBlocksPerMultiprocessor is
  // UNSTABLE for this cluster kernel -- identical back-to-back launches
  // got grids of 152 and 456 CTAs (waves 0.33 vs 1.0) from the same
  // query, and the undersized grid runs persistently at 1/3 occupancy
  // (1.32ms vs 742us). Registers are not binding here (84 regs at eo=0 /
  // 64 at eo=2, measured), so the static minimum is exact.
  auto* props = at::cuda::getCurrentDeviceProperties();
  // +1KB/CTA: the per-block reserved smem (cudaDevAttrReservedSharedMemoryPerBlock).
  int smem_blocks = int(props->sharedMemPerMultiprocessor / size_t(smem_bytes + 1024));
  constexpr int kTmemBlocks =
      cute::TMEM::Allocator2Sm::Sm100TmemCapacityColumns / kAccTmemCols;
  int blocks_per_sm = smem_blocks < kTmemBlocks ? smem_blocks : kTmemBlocks;
  TORCH_CHECK(blocks_per_sm >= 1, "persistent grid: zero co-residable CTAs/SM");
  int sm_count = props->multiProcessorCount;
  int total_pair_tiles = grid_pairs_m * grid_n;
  int persist_clusters = (sm_count * blocks_per_sm) / 2;
  if (persist_clusters > total_pair_tiles) persist_clusters = total_pair_tiles;
  dim3 dimGrid(2 * persist_clusters);  // cluster rr walks rr, rr+C, rr+2C...
#elif DUAL2SM_RASTER_GM > 0
  // 1D flatten: ascending-blockIdx hardware rasterization executes the
  // GROUP_M order produced by raster_map (the standard Triton trick).
  dim3 dimGrid(2 * grid_pairs_m * grid_n);
#else
  // blockIdx.x: (pair, v) interleaved; y: each cluster walks kTilesPerCta n-tiles
  dim3 dimGrid(2 * grid_pairs_m, grid_n / kTilesPerCta);
#endif

  cudaLaunchConfig_t launch_config;
  launch_config.gridDim = dimGrid;
  launch_config.blockDim = dimBlock;
  launch_config.dynamicSmemBytes = smem_bytes;
  launch_config.stream = at::cuda::getCurrentCUDAStream();

  cudaLaunchAttribute cluster_attr;
  cluster_attr.id = cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim.x = 2;
  cluster_attr.val.clusterDim.y = kClusterN;
  cluster_attr.val.clusterDim.z = 1;
  launch_config.numAttrs = 1;
  launch_config.attrs = &cluster_attr;

  void* args[] = {
    (void*)&mA, (void*)&mB, (void*)&mC, (void*)&mD,
    (void*)&mma_tiler, (void*)&tiled_mma,
    (void*)&tma_atom_A, (void*)&tma_atom_B
#if DUAL2SM_TMA_EPI
    , (void*)&tma_desc_D
#endif
  };

  AT_CUDA_CHECK(cudaLaunchKernelExC(&launch_config, (void*)kernel_ptr, args));

#if DUAL2SM_D_HALF
  return d_buffer;  // already fp16 (in-kernel convert)
#else
  return d_buffer.to(torch::kFloat16);
#endif
}

}  // namespace dual_2sm_fp8_impl

torch::Tensor matmul_tcgen05_dual_2sm_fp8(torch::Tensor a, torch::Tensor b) {
  return dual_2sm_fp8_impl::run_dual_2sm_fp8_matmul(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_tcgen05_dual_2sm_fp8", &matmul_tcgen05_dual_2sm_fp8);
}
