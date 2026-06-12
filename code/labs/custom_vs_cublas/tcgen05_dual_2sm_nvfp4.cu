/**
 * NVFP4 (block-scaled e2m1) port of the 2-SM UMMA Pair kernel (Front N4b)
 * ========================================================================
 *
 * NEW VARIANT FILE: a block-scaled port of tcgen05_dual_2sm_fp8.cu (the FP8
 * champion at B75: persistent clusters + GROUP_M raster + warp-split
 * producer/consumer + TMA-store fp16 epilogue, 313.5us / 0.907x cuBLASLt
 * FP8 at 8192^3) to NVFP4: e2m1 x e2m1 with ue4m3 block scales (SF_VEC=16)
 * -> fp32 accumulate -> fp16 D. SCOPE: the champion path ONLY (PERSIST=1,
 * eo=0); the AMCAST / EPI_OVERLAP / PREFETCH / non-persistent branches of
 * the FP8 file are intentionally not ported.
 *
 * What changes vs the FP8 file:
 *   - MMA atom: SM100_MMA_MXF4_2x1SM_SS<e2m1,e2m1,f32,ue4m3,256,kTileN,
 *     VS=16,K,K> (tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale
 *     .block16). Atom K-extent = 64 elements (256 bits / 4).
 *   - Block scales: SFA/SFB in the cuBLASLt 128x4-blocked gmem layout
 *     (Sm1xxBlockScaledConfig<16>; == torch._scaled_mm's expected layout),
 *     TMA'd per stage into smem alongside A/B (SFA per-CTA M-half slices
 *     via SM100_TMA_2SM_LOAD; SFB needs FULL kTileN in BOTH CTAs, so each
 *     CTA issues HALF via SM100_TMA_2SM_LOAD_MULTICAST masked to the pair
 *     -- the vendor collective's exact scheme), then UTCCP
 *     (SM100_UTCCP_4x32dp128bit_2cta, tcgen05.cp) from smem -> TMEM by the
 *     leader consumer thread right before the stage's MMAs. Single TMEM SF
 *     buffer: same-thread same-pipe in-order issue orders UTCCP(s+1) after
 *     MMA(s) (the vendor collective relies on this; no fence).
 *   - expect-tx (B53/B57): the leader's full barrier counts A + B + SFA +
 *     SFB slices: 2 x (A-half + B-half + SFA-half + SFB-half-multicast-
 *     slice). The multicast slice counts ONCE per destination pair (B53).
 *   - TMEM budget (B75, stated before building): acc = kTileN(=128) fp32
 *     cols/CTA + SFA + SFB tmem_sf_frg cols (ScaleFactorDuplicated4by1 at
 *     M=256). tcgen05.alloc takes a POWER OF TWO column count -> allocate
 *     256 cols/CTA -> TMEM-implied 2 CTAs/SM (the FP8 n256 champion's
 *     occupancy). kTileN=256 would need >512 alloc -> 1 CTA/SM ->
 *     INADMISSIBLE; therefore this port is kTileN=128 only. The named
 *     phase-2 lever is a dual alloc (128-col acc + 32-col SF) -> 3 CTAs/SM.
 *   - kTileK default 256 (SW128 atom: e2m1 K-rows are kTileK/2 bytes);
 *     kTileK=128 selects SW64. Stage bytes at k256/n128: A 16KB + B 8KB +
 *     SFA 2KB + SFB 2KB = 28KB. num_k_tiles at 8192 = 32.
 *   - Epilogue: UNCHANGED from the FP8 champion (fp32 TMEM -> chunked t2r
 *     -> staged SW128 smem -> cp.async.bulk.tensor.2d; D_HALF=1 packs fp16
 *     pairs in-kernel). dh1 carries: dtype-independent of operands.
 *
 * Entry: matmul_tcgen05_dual_2sm_nvfp4(a_u8[m,k/2], b_u8[n,k/2],
 *   sfa_e4m3[blocked flat], sfb_e4m3[blocked flat]) -> fp16 [m,n] (A@B^T).
 */

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>

#include <cutlass/arch/barrier.h>
#include <cutlass/half.h>
#include <cutlass/numeric_types.h>            // float_e2m1_t, float_ue4m3_t
#include <cutlass/detail/sm100_blockscaled_layout.hpp>  // Sm1xxBlockScaledConfig
#include <cutlass/detail/sm100_tmem_helper.hpp>         // find_tmem_tensor_col_offset

#include <cute/tensor.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>           // SM100_UTCCP_4x32dp128bit_2cta
#include <cute/atom/copy_traits_sm100.hpp>    // make_utccp_copy
#include <cute/atom/mma_traits_sm100.hpp>     // MXF4 traits + get_utccp_smem_desc_tensor
#include <cute/atom/copy_traits_sm100_tma.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/arch/cluster_sm90.hpp>

#ifndef DUAL2SM_TILE_N
#define DUAL2SM_TILE_N 128
#endif
#ifndef DUAL2SM_STAGES
#define DUAL2SM_STAGES 3
#endif
#ifndef DUAL2SM_MIN_BLOCKS
#define DUAL2SM_MIN_BLOCKS 2
#endif
// K-extent of one pipeline stage: 256 -> SW128 smem atom (128B e2m1 rows),
// 128 -> SW64 (64B rows; halves the stage so the ring can go deeper).
#ifndef DUAL2SM_TILE_K
#define DUAL2SM_TILE_K 256
#endif
// Persistent clusters + GROUP_M raster (the FP8 champion structure). This
// port REQUIRES persist=1 (the only mainloop shape carried over).
#ifndef DUAL2SM_PERSIST
#define DUAL2SM_PERSIST 1
#endif
#ifndef DUAL2SM_RASTER_GM
#define DUAL2SM_RASTER_GM 8
#endif
// t2r chunk width for the per-round epilogue (B55 knob).
#ifndef DUAL2SM_EPI_ATOM
#define DUAL2SM_EPI_ATOM 32
#endif
// TMA-store epilogue through the drained ring stage (lever h / te1).
#ifndef DUAL2SM_TMA_EPI
#define DUAL2SM_TMA_EPI 1
#endif
// In-kernel fp16 D conversion in the TMA_EPI staging path (F8b dh1).
#ifndef DUAL2SM_D_HALF
#define DUAL2SM_D_HALF 1
#endif

using namespace cute;

namespace dual_2sm_nvfp4_impl {

using TypeA = cutlass::float_e2m1_t;
using TypeB = cutlass::float_e2m1_t;
using TypeSF = cutlass::float_ue4m3_t;
using TypeC = float;
#if DUAL2SM_D_HALF
using TypeD = cutlass::half_t;
#else
using TypeD = float;
#endif
using Accumulator = float;

constexpr int kTileM = 256;  // pair-wide M: 128 rows per CTA
constexpr int kTileN = DUAL2SM_TILE_N;
constexpr int kStages = DUAL2SM_STAGES;
constexpr int kTileK = DUAL2SM_TILE_K;
constexpr int kSFVec = 16;   // NVFP4: one ue4m3 scale per 16 e2m1 elements
constexpr int kNumThreads = 128;

static_assert(kStages >= 2 && kStages <= 6, "DUAL2SM_STAGES must be 2..6");
static_assert(kTileK == 128 || kTileK == 256, "DUAL2SM_TILE_K must be 128 or 256");
static_assert(kTileN == 128,
              "NVFP4 port is kTileN=128 only: acc(kTileN) + SF cols must fit "
              "a 256-col pow2 TMEM alloc for 2 CTAs/SM (B75 admissibility); "
              "n256 implies a 512-col alloc = 1 CTA/SM (inadmissible)");
static_assert(DUAL2SM_PERSIST == 1, "NVFP4 port carries the persistent eo=0 champion path only");
static_assert(DUAL2SM_TMA_EPI == 0 || kTileK == 256,
              "TMA_EPI stages 16KB chunks through a drained A stage: needs "
              "kTileK=256 (128 rows x 128B)");
static_assert(DUAL2SM_D_HALF == 0 || DUAL2SM_TMA_EPI == 1,
              "D_HALF is scoped to the TMA_EPI store path");
static_assert(DUAL2SM_TMA_EPI == 0 || DUAL2SM_EPI_ATOM == 32,
              "TMA_EPI staging assumes 32-col t2r chunks");
static_assert(kTileN % DUAL2SM_EPI_ATOM == 0, "EPI_ATOM must divide TILE_N");

// Raster mapping (GROUP_M), identical to the FP8 file.
CUTE_DEVICE void raster_map(int idx, int grid_m, int grid_n, int& tm, int& tn) {
#if DUAL2SM_RASTER_GM > 0
  constexpr int gm = DUAL2SM_RASTER_GM;
  int per_group = gm * grid_n;
  int group = idx / per_group;
  int first_m = group * gm;
  int gsz = grid_m - first_m;
  gsz = gsz < gm ? gsz : gm;
  int rem = idx - group * per_group;
  tm = first_m + rem % gsz;
  tn = rem / gsz;
#else
  tm = idx % grid_m;
  tn = idx / grid_m;
#endif
}

// ---------------------------------------------------------------------------
// MMA atom: dense block-scaled 2SM NVFP4 (VS=16 -> kind::mxf4nvf4.block16).
// ---------------------------------------------------------------------------
using MmaOp = SM100_MMA_MXF4_2x1SM_SS<TypeA, TypeB, TypeC, TypeSF,
                                      kTileM, kTileN, kSFVec,
                                      UMMA::Major::K, UMMA::Major::K>;
using MmaTraits = MMA_Traits<MmaOp>;
static_assert(MmaTraits::K == 64, "MXF4NVF4 dense atom K-extent must be 64");

// SF partitioning helper TiledMMA (1-SM MMA_ScaleFactor atom): used to
// partition + TMA the FULL-kTileN SFB per CTA (the vendor collective's
// TiledMMA_SF), while the main tiled_mma halves B across the pair.
using TiledMmaSF = TiledMMA<MMA_Atom<typename MmaTraits::MMA_ScaleFactor>,
                            Layout<Shape<_1, _1, _1>>,
                            Tile<Underscore, Underscore, Underscore>>;

using BlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<kSFVec>;

// fp32 accumulator: 128-row M-half x kTileN cols/CTA; SF lives above it.
// tcgen05.alloc requires a POWER-OF-TWO column count: 256 covers
// acc(128) + SFA + SFB (checked against the real frg layouts below).
constexpr int kAccTmemCols = kTileN;
constexpr int kTmemColsAlloc = 256;
static_assert(2 * kTmemColsAlloc <= cute::TMEM::Allocator2Sm::Sm100TmemCapacityColumns,
              "TMEM alloc must leave room for a co-resident CTA (2 CTAs/SM)");

template <class TypeA_, class TypeB_, class ASmemLayout, class BSmemLayout,
          class SFASmemLayout, class SFBSmemLayout>
struct Dual2SmNvfp4SharedStorage {
  using SFASmemLayoutT = SFASmemLayout;
  using SFBSmemLayoutT = SFBSmemLayout;
  alignas(1024) cute::ArrayEngine<TypeA_, cute::cosize_v<ASmemLayout>> smem_A[kStages];
  alignas(1024) cute::ArrayEngine<TypeB_, cute::cosize_v<BSmemLayout>> smem_B[kStages];
  alignas(128) cute::ArrayEngine<TypeSF, cute::cosize_v<SFASmemLayout>> smem_SFA[kStages];
  alignas(128) cute::ArrayEngine<TypeSF, cute::cosize_v<SFBSmemLayout>> smem_SFB[kStages];
  alignas(16) cute::uint64_t full_barrier[kStages];   // leader-only: pair TMA bytes (A+B+SFA+SFB)
  alignas(16) cute::uint64_t empty_barrier[kStages];  // per-CTA: multicast commit
  alignas(16) cute::uint64_t mma_barrier[1];
  alignas(16) cute::uint64_t tmem_empty[1];           // leader-only: both CTAs' drains
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE auto tensor_sA(int stage) {
    return make_tensor(make_smem_ptr(smem_A[stage].begin()), ASmemLayout{});
  }
  CUTE_DEVICE auto tensor_sB(int stage) {
    return make_tensor(make_smem_ptr(smem_B[stage].begin()), BSmemLayout{});
  }
  CUTE_DEVICE auto tensor_sSFA(int stage) {
    return make_tensor(make_smem_ptr(smem_SFA[stage].begin()), SFASmemLayout{});
  }
  CUTE_DEVICE auto tensor_sSFB(int stage) {
    return make_tensor(make_smem_ptr(smem_SFB[stage].begin()), SFBSmemLayout{});
  }
};

// Overlap-epilogue t2r atom (B55 knob).
#if DUAL2SM_EPI_ATOM == 32
using EpiLoadOp = SM100_TMEM_LOAD_32dp32b32x;
#elif DUAL2SM_EPI_ATOM == 16
using EpiLoadOp = SM100_TMEM_LOAD_32dp32b16x;
#elif DUAL2SM_EPI_ATOM == 64
using EpiLoadOp = SM100_TMEM_LOAD_32dp32b64x;
#else
#error "DUAL2SM_EPI_ATOM must be 16|32|64 in the NVFP4 port"
#endif

#if DUAL2SM_TMA_EPI
// Raw TMA store descriptor for D (the banked te1 path, verbatim from FP8).
static CUtensorMap make_tma_desc_D(void const* d_ptr, int64_t m, int64_t n) {
  CUtensorMap desc{};
  uint64_t global_dim[2] = {uint64_t(n), uint64_t(m)};
  uint64_t global_strides[1] = {uint64_t(n) * sizeof(TypeD)};
#if DUAL2SM_D_HALF
  uint32_t box_dim[2] = {64, 128};
  constexpr CUtensorMapDataType kDDtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
#else
  uint32_t box_dim[2] = {32, 128};
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

// Distinct kernel symbol per config (B61 alias trap).
constexpr int kCfg = (DUAL2SM_MIN_BLOCKS) | (DUAL2SM_RASTER_GM << 4) |
                     (DUAL2SM_TMA_EPI << 10) | (DUAL2SM_D_HALF << 11) |
                     (DUAL2SM_EPI_ATOM << 12);

template <class SharedStorageT, int Cfg,
          class ATensor, class BTensor, class SFATensor, class SFBTensor,
          class CTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMAT, class TileShapeSF, class TiledMMASFT,
          class TmaAtomA, class TmaAtomB, class TmaAtomSFA, class TmaAtomSFB>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(kNumThreads, DUAL2SM_MIN_BLOCKS)
gemm_dual_2sm_nvfp4(ATensor mA, BTensor mB, SFATensor mSFA, SFBTensor mSFB,
                    CTensor mC, DTensor mD,
                    MmaTiler_MNK mma_tiler, TiledMMAT tiled_mma,
                    TileShapeSF tile_shape_sf, TiledMMASFT tiled_mma_sf,
                    CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
                    CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B,
                    CUTE_GRID_CONSTANT TmaAtomSFA const tma_atom_SFA,
                    CUTE_GRID_CONSTANT TmaAtomSFB const tma_atom_SFB
#if DUAL2SM_TMA_EPI
                    , CUTE_GRID_CONSTANT CUtensorMap const tma_desc_D
#endif
                    ) {

  uint32_t elect_one_thr = cute::elect_one_sync();
  const int warp_idx = int(threadIdx.x / 32);
  uint32_t elect_one_warp = (warp_idx == 0);

  int v = int(cute::block_rank_in_cluster()) & 1;
  bool leader_cta = (v == 0);

  const int grid_m_t = int(size<0>(shape(mA))) / kTileM;  // pair-rows
  const int grid_n_t = int(size<0>(shape(mB))) / kTileN;  // n-cols

  // Persistent walk bookkeeping (B65 static grid).
  int tile_m = 0;
  int tile_n = 0;
  const int num_clusters = int(gridDim.x) >> 1;
  const int rr = int(blockIdx.x) >> 1;
  const int total_tiles = grid_m_t * grid_n_t;
  const int my_tiles = (total_tiles - rr + num_clusters - 1) / num_clusters;

  auto mma_coord = make_coord(tile_m, tile_n, _);

  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  SharedStorageT& storage = *reinterpret_cast<SharedStorageT*>(shared_memory);

  auto cta_mma = tiled_mma.get_slice(v);
  Tensor tCgA = cta_mma.partition_A(gA);
  Tensor tCgB = cta_mma.partition_B(gB);
  Tensor tCgC = cta_mma.partition_C(gC);

  // SF partitioning: SFA via the main atom's A partition (per-CTA M-half);
  // SFB via the 1-SM SF atom (FULL kTileN per CTA).
  Tensor gSFA = local_tile(mSFA, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gSFB = local_tile(mSFB, tile_shape_sf, mma_coord, Step<X, _1, _1>{});
  auto cta_mma_sfb = tiled_mma_sf.get_slice(0);
  Tensor tCgSFA = cta_mma.partition_A(gSFA);
  Tensor tCgSFB = cta_mma_sfb.partition_B(gSFB);

  // -------------------------------------------------------------------
  // TMEM: pair-collective 256-col alloc (acc 128 + SF region above it).
  // -------------------------------------------------------------------
  cute::TMEM::Allocator2Sm tmem_allocator{};
  if (elect_one_warp) {
    tmem_allocator.allocate(kTmemColsAlloc, &storage.tmem_base_ptr);
    tmem_allocator.release_allocation_lock();
  }

  if (elect_one_warp && elect_one_thr) {
    for (int s = 0; s < kStages; ++s) {
      cute::initialize_barrier(storage.full_barrier[s], 1);
      cute::initialize_barrier(storage.empty_barrier[s], 1);
    }
    cute::initialize_barrier(storage.mma_barrier[0], 1);
    cute::initialize_barrier(storage.tmem_empty[0], 2);
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  cute::cluster_sync();

  uint32_t tmem_base = storage.tmem_base_ptr;

  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);  // tmem_frg_2sm
  tCtAcc.data() = tmem_base;

  // TMEM SF fragments (vendor scheme: shape of the per-stage smem atom).
  // Column spans are static-typed; the host runner TORCH_CHECKs that
  // acc + SFA + SFB fits the 256-col alloc (B75 admissibility).
  Tensor tCtSFA = make_tensor<typename MmaTraits::FrgTypeSFA>(
      shape(typename SharedStorageT::SFASmemLayoutT{}));
  Tensor tCtSFB = make_tensor<typename MmaTraits::FrgTypeSFB>(
      shape(typename SharedStorageT::SFBSmemLayoutT{}));
  const int sfa_cols = int(cutlass::detail::find_tmem_tensor_col_offset(tCtSFA));
  const int sfb_cols = int(cutlass::detail::find_tmem_tensor_col_offset(tCtSFB));
  if (kAccTmemCols + sfa_cols + sfb_cols > kTmemColsAlloc) {
    __trap();  // B75 TMEM admissibility (folds at compile time: static layouts)
  }
  tCtSFA.data() = tmem_base + uint32_t(kAccTmemCols);
  tCtSFB.data() = tmem_base + uint32_t(kAccTmemCols + sfa_cols);

  // UTCCP smem->TMEM tiled copies (leader-issued, cta_group::2 paired).
  using UtccpOp = SM100_UTCCP_4x32dp128bit_2cta;
  auto tCtSFA_compact = make_tensor(tCtSFA.data(), filter_zeros(tCtSFA.layout()));
  auto tCtSFB_compact = make_tensor(tCtSFB.data(), filter_zeros(tCtSFB.layout()));
  auto tiled_copy_s2t_SFA = make_utccp_copy(UtccpOp{}, tCtSFA_compact);
  auto tiled_copy_s2t_SFB = make_utccp_copy(UtccpOp{}, tCtSFB_compact);
  auto thr_copy_s2t_SFA = tiled_copy_s2t_SFA.get_slice(0);
  auto thr_copy_s2t_SFB = tiled_copy_s2t_SFB.get_slice(0);
  auto thr_tCtSFA_s2t = thr_copy_s2t_SFA.partition_D(tCtSFA_compact);
  auto thr_tCtSFB_s2t = thr_copy_s2t_SFB.partition_D(tCtSFB_compact);
  auto utccp_src_SFA = [&](int s) {
    auto t = storage.tensor_sSFA(s);
    auto tc = make_tensor(t.data(), filter_zeros(t.layout()));
    return get_utccp_smem_desc_tensor<UtccpOp>(thr_copy_s2t_SFA.partition_S(tc));
  };
  auto utccp_src_SFB = [&](int s) {
    auto t = storage.tensor_sSFB(s);
    auto tc = make_tensor(t.data(), filter_zeros(t.layout()));
    return get_utccp_smem_desc_tensor<UtccpOp>(thr_copy_s2t_SFB.partition_S(tc));
  };
#if DUAL2SM_STAGES == 2
  decltype(utccp_src_SFA(0)) tSFAs_s2t[kStages] = { utccp_src_SFA(0), utccp_src_SFA(1) };
  decltype(utccp_src_SFB(0)) tSFBs_s2t[kStages] = { utccp_src_SFB(0), utccp_src_SFB(1) };
#elif DUAL2SM_STAGES == 3
  decltype(utccp_src_SFA(0)) tSFAs_s2t[kStages] = { utccp_src_SFA(0), utccp_src_SFA(1), utccp_src_SFA(2) };
  decltype(utccp_src_SFB(0)) tSFBs_s2t[kStages] = { utccp_src_SFB(0), utccp_src_SFB(1), utccp_src_SFB(2) };
#elif DUAL2SM_STAGES == 4
  decltype(utccp_src_SFA(0)) tSFAs_s2t[kStages] = { utccp_src_SFA(0), utccp_src_SFA(1), utccp_src_SFA(2), utccp_src_SFA(3) };
  decltype(utccp_src_SFB(0)) tSFBs_s2t[kStages] = { utccp_src_SFB(0), utccp_src_SFB(1), utccp_src_SFB(2), utccp_src_SFB(3) };
#elif DUAL2SM_STAGES == 5
  decltype(utccp_src_SFA(0)) tSFAs_s2t[kStages] = { utccp_src_SFA(0), utccp_src_SFA(1), utccp_src_SFA(2), utccp_src_SFA(3), utccp_src_SFA(4) };
  decltype(utccp_src_SFB(0)) tSFBs_s2t[kStages] = { utccp_src_SFB(0), utccp_src_SFB(1), utccp_src_SFB(2), utccp_src_SFB(3), utccp_src_SFB(4) };
#else
  decltype(utccp_src_SFA(0)) tSFAs_s2t[kStages] = { utccp_src_SFA(0), utccp_src_SFA(1), utccp_src_SFA(2), utccp_src_SFA(3), utccp_src_SFA(4), utccp_src_SFA(5) };
  decltype(utccp_src_SFB(0)) tSFBs_s2t[kStages] = { utccp_src_SFB(0), utccp_src_SFB(1), utccp_src_SFB(2), utccp_src_SFB(3), utccp_src_SFB(4), utccp_src_SFB(5) };
#endif

  // TMA coordinate tensors.
  Tensor tma_coord_A = tma_atom_A.get_tma_tensor(shape(mA));
  Tensor tma_coord_B = tma_atom_B.get_tma_tensor(shape(mB));
  Tensor tma_coord_SFA = tma_atom_SFA.get_tma_tensor(shape(mSFA));
  Tensor tma_coord_SFB = tma_atom_SFB.get_tma_tensor(shape(mSFB));
  Tensor gCoordA = local_tile(tma_coord_A, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gCoordB = local_tile(tma_coord_B, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor gCoordSFA = local_tile(tma_coord_SFA, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gCoordSFB = local_tile(tma_coord_SFB, tile_shape_sf, mma_coord, Step<X, _1, _1>{});
  Tensor tCgCoordA = cta_mma.partition_A(gCoordA);
  Tensor tCgCoordB = cta_mma.partition_B(gCoordB);
  Tensor tCgCoordSFA = cta_mma.partition_A(gCoordSFA);
  Tensor tCgCoordSFB = cta_mma_sfb.partition_B(gCoordSFB);

  // A / B / SFA: non-multicast 2SM partition (coord 0 / layout 1).
  Tensor tCsA_0 = storage.tensor_sA(0);
  auto [tAgA, tAsA_0] = tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsA_0), group_modes<0,3>(tCgCoordA));
  Tensor tCsB_0 = storage.tensor_sB(0);
  auto [tBgB, tBsB_0] = tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsB_0), group_modes<0,3>(tCgCoordB));
  Tensor tCsSFA_0 = storage.tensor_sSFA(0);
  auto [tSFAg, tSFAs_0] = tma_partition(tma_atom_SFA, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsSFA_0), group_modes<0,3>(tCgCoordSFA));

  // SFB: pair-multicast (each CTA issues HALF, both CTAs' smem get FULL).
  auto cluster_layout_sfb_vmnk =
      tiled_divide(make_layout(Shape<_2, _1, _1>{}),
                   make_tile(typename TiledMMASFT::AtomThrID{}));
  auto cta_coord_sfb_vmnk =
      cluster_layout_sfb_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
  uint16_t mcast_mask_sfb =
      create_tma_multicast_mask<1>(cluster_layout_sfb_vmnk, cta_coord_sfb_vmnk);
  Tensor tCsSFB_0 = storage.tensor_sSFB(0);
  auto [tSFBg, tSFBs_0] = tma_partition(tma_atom_SFB,
      get<1>(cta_coord_sfb_vmnk), make_layout(size<1>(cluster_layout_sfb_vmnk)),
      group_modes<0,3>(tCsSFB_0), group_modes<0,3>(tCgCoordSFB));
  // Multicast slice has a CTA-dependent smem offset: per-stage slices must
  // come from tma_partition itself (the AMCAST lesson from the FP8 file).
  auto sliceSFB = [&](int s) {
    Tensor tCsSFB_s = storage.tensor_sSFB(s);
    return get<1>(tma_partition(tma_atom_SFB,
        get<1>(cta_coord_sfb_vmnk), make_layout(size<1>(cluster_layout_sfb_vmnk)),
        group_modes<0,3>(tCsSFB_s), group_modes<0,3>(tCgCoordSFB)));
  };
#if DUAL2SM_STAGES == 2
  decltype(sliceSFB(0)) tSFBs_st[kStages] = { sliceSFB(0), sliceSFB(1) };
#elif DUAL2SM_STAGES == 3
  decltype(sliceSFB(0)) tSFBs_st[kStages] = { sliceSFB(0), sliceSFB(1), sliceSFB(2) };
#elif DUAL2SM_STAGES == 4
  decltype(sliceSFB(0)) tSFBs_st[kStages] = { sliceSFB(0), sliceSFB(1), sliceSFB(2), sliceSFB(3) };
#elif DUAL2SM_STAGES == 5
  decltype(sliceSFB(0)) tSFBs_st[kStages] = { sliceSFB(0), sliceSFB(1), sliceSFB(2), sliceSFB(3), sliceSFB(4) };
#else
  decltype(sliceSFB(0)) tSFBs_st[kStages] = { sliceSFB(0), sliceSFB(1), sliceSFB(2), sliceSFB(3), sliceSFB(4), sliceSFB(5) };
#endif

  // expect-tx (B53/B57 INCLUDING SF slices): 2 x (A-half + B-half +
  // SFA-half + SFB-half-multicast-slice); all 2SM ops complete at the
  // LEADER's barrier; the multicast slice counts once per destination pair.
  int tma_bytes = 2 * (sizeof(make_tensor_like(tAsA_0)) + sizeof(make_tensor_like(tBsB_0)) +
                       sizeof(make_tensor_like(tSFAs_0)) + sizeof(make_tensor_like(tSFBs_0)));

  // Per-round gmem TMA coordinate slices (persistent walk).
  auto tAgA_for = [&](int tm, int tn) {
    auto coord_t = make_coord(tm, tn, _);
    Tensor g = local_tile(tma_coord_A, mma_tiler, coord_t, Step<_1, X, _1>{});
    Tensor t = cta_mma.partition_A(g);
    return cute::get<0>(tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{},
        group_modes<0,3>(tCsA_0), group_modes<0,3>(t)));
  };
  auto tBgB_dyn = [&](int tm, int tn) {
    auto coord_t = make_coord(tm, tn, _);
    Tensor g = local_tile(tma_coord_B, mma_tiler, coord_t, Step<X, _1, _1>{});
    Tensor t = cta_mma.partition_B(g);
    return cute::get<0>(tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{},
        group_modes<0,3>(tCsB_0), group_modes<0,3>(t)));
  };
  auto tSFAg_for = [&](int tm, int tn) {
    auto coord_t = make_coord(tm, tn, _);
    Tensor g = local_tile(tma_coord_SFA, mma_tiler, coord_t, Step<_1, X, _1>{});
    Tensor t = cta_mma.partition_A(g);
    return cute::get<0>(tma_partition(tma_atom_SFA, Int<0>{}, Layout<_1>{},
        group_modes<0,3>(tCsSFA_0), group_modes<0,3>(t)));
  };
  auto tSFBg_dyn = [&](int tm, int tn) {
    auto coord_t = make_coord(tm, tn, _);
    Tensor g = local_tile(tma_coord_SFB, tile_shape_sf, coord_t, Step<X, _1, _1>{});
    Tensor t = cta_mma_sfb.partition_B(g);
    return cute::get<0>(tma_partition(tma_atom_SFB,
        get<1>(cta_coord_sfb_vmnk), make_layout(size<1>(cluster_layout_sfb_vmnk)),
        group_modes<0,3>(tCsSFB_0), group_modes<0,3>(t)));
  };

  // Per-stage UMMA smem-descriptor fragments.
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

  constexpr uint16_t kEmptyMask = 0x3;
  const uint16_t kPairMask = uint16_t(0x3) << (int(cute::block_rank_in_cluster()) & ~1);

  // =====================================================================
  // V5 persistent mainloop, eo=0 (the FP8 champion structure) + SF feed.
  // =====================================================================
  const uint32_t leader_rank = uint32_t(cute::block_rank_in_cluster()) & ~1u;
  int empty_phase[kStages] = {};
  int full_phase[kStages] = {};
  int tmem_phase = 0;
  auto tiled_t2r_copy = make_tmem_copy(EpiLoadOp{}, tCtAcc);
  auto thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);
  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
  constexpr int kEpiRank = decltype(rank(tDtAcc))::value;
  Tensor tDtAccG = group_modes<1, kEpiRank>(tDtAcc);
  for (int t = 0; t < my_tiles; ++t) {
    int tm, tn;
    raster_map(rr + t * num_clusters, grid_m_t, grid_n_t, tm, tn);
    if (warp_idx == 0) {
      // Producer: WHOLE warp 0 of BOTH CTAs. Ring fully drains per round.
      auto tAgA_t = tAgA_for(tm, tn);
      auto tBgB_t = tBgB_dyn(tm, tn);
      auto tSFAg_t = tSFAg_for(tm, tn);
      auto tSFBg_t = tSFBg_dyn(tm, tn);
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
          auto tSFAs = make_tensor(make_smem_ptr(storage.smem_SFA[s].begin()), tSFAs_0.layout());
          copy(tma_atom_SFA.with(storage.full_barrier[s]), tSFAg_t(_, k), tSFAs);
          copy(tma_atom_SFB.with(storage.full_barrier[s], mcast_mask_sfb),
               tSFBg_t(_, k), tSFBs_st[s]);
        }
      }
      // Drain this round's final multicast commits on BOTH CTAs.
      for (int k = num_k_tiles - min(kStages, num_k_tiles); k < num_k_tiles; ++k) {
        int s = k % kStages;
        cute::wait_barrier(storage.empty_barrier[s], empty_phase[s]);
        empty_phase[s] ^= 1;
      }
    } else if (warp_idx == 1 && leader_cta) {
      // Consumer: leader warp 1. UTCCP the stage's SFs (cta_group::2 paired
      // copies into BOTH CTAs' TMEM), then the k-block MMAs. Same-thread
      // same-pipe in-order issue orders UTCCP(s+1) after MMA(s) (the
      // vendor collective's scheme; single TMEM SF buffer).
      if (t >= 1) {
        cute::wait_barrier(storage.tmem_empty[0], tmem_phase);
        tmem_phase ^= 1;
      }
      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
      for (int k = 0; k < num_k_tiles; ++k) {
        int s = k % kStages;
        cute::wait_barrier(storage.full_barrier[s], full_phase[s]);
        full_phase[s] ^= 1;

        if (elect_one_thr) {
          copy(tiled_copy_s2t_SFA, tSFAs_s2t[s], thr_tCtSFA_s2t);
          copy(tiled_copy_s2t_SFB, tSFBs_s2t[s], thr_tCtSFB_s2t);
        }

        for (int kb = 0; kb < size<2>(tCrA_st[0]); ++kb) {
          cute::gemm(tiled_mma.with(tiled_mma.accumulate_,
                                    tCtSFA(_, _, kb), tCtSFB(_, _, kb)),
                     tCrA_st[s](_, _, kb), tCrB_st[s](_, _, kb), tCtAcc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive_multicast_2x1SM(
            reinterpret_cast<uint64_t*>(&storage.empty_barrier[s]), kEmptyMask);
      }
      cutlass::arch::umma_arrive_multicast_2x1SM(
          reinterpret_cast<uint64_t*>(&storage.mma_barrier[0]), kPairMask);
    }

    // Per-round epilogue: ALL 128 threads of both CTAs (VERBATIM from FP8).
    cute::wait_barrier(storage.mma_barrier[0], t & 1);
#if DUAL2SM_TMA_EPI
    {
      constexpr int kDChunks = kTileN / 32;
      const int row0 = (tm * 2 + v) * 128;
      const int col0 = tn * kTileN;
      const int r = int(threadIdx.x);
      auto coord_t = make_coord(tm, tn, _);
      Tensor gD_t = local_tile(mD, mma_tiler, coord_t, Step<_1, _1, X>{});
      Tensor tCgD_t = cta_mma.partition_C(gD_t);
      Tensor tDgD = thr_t2r_copy.partition_D(tCgD_t);
      Tensor tDgDG = group_modes<1, kEpiRank>(tDgD);
#if DUAL2SM_D_HALF
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
          if (warp_idx == 0 && elect_one_thr) {
            cute::tma_store_wait<kStages - 1>();
          }
          __syncthreads();
        }
        uint4* sbuf = reinterpret_cast<uint4*>(raw_pointer_cast(storage.smem_A[c2 % kStages].begin()));
        CUTE_UNROLL
        for (int g = 0; g < 8; ++g) {
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
        cute::tma_store_fence();
        __syncthreads();
        if (warp_idx == 0 && elect_one_thr) {
          uint32_t s_ptr = cute::cast_smem_ptr_to_uint(raw_pointer_cast(storage.smem_A[c2 % kStages].begin()));
          int32_t crd_n = col0 + c2 * 64;
          int32_t crd_m = row0;
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
          if (warp_idx == 0 && elect_one_thr) {
            cute::tma_store_wait<kStages - 1>();
          }
          __syncthreads();
        }
        float4* sbuf = reinterpret_cast<float4*>(raw_pointer_cast(storage.smem_A[c % kStages].begin()));
        CUTE_UNROLL
        for (int g = 0; g < 8; ++g) {
          sbuf[r * 8 + (g ^ (r & 7))] =
              make_float4(tDrAcc(4 * g + 0), tDrAcc(4 * g + 1),
                          tDrAcc(4 * g + 2), tDrAcc(4 * g + 3));
        }
        cute::tma_store_fence();
        __syncthreads();
        if (warp_idx == 0 && elect_one_thr) {
          uint32_t s_ptr = cute::cast_smem_ptr_to_uint(raw_pointer_cast(storage.smem_A[c % kStages].begin()));
          int32_t crd_n = col0 + c * 32;
          int32_t crd_m = row0;
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
      Tensor tDgD = thr_t2r_copy.partition_D(tCgD_t);
      Tensor tDgDG = group_modes<1, kEpiRank>(tDgD);
      CUTE_UNROLL
      for (int c = 0; c < size<1>(tDtAccG); ++c) {
        Tensor tDrAcc = make_tensor<Accumulator>(shape(tDgDG(_, c)));
        copy(tiled_t2r_copy, tDtAccG(_, c), tDrAcc);
        cutlass::arch::fence_view_async_tmem_load();
        copy(tDrAcc, tDgDG(_, c));
      }
    }
#endif  // DUAL2SM_TMA_EPI
    if (t + 1 < my_tiles) {
      __syncthreads();
      cutlass::arch::ClusterBarrier::arrive(
          &storage.tmem_empty[0], leader_rank,
          uint32_t((warp_idx == 0) && elect_one_thr));
    }
#if DUAL2SM_TMA_EPI
    if (warp_idx == 0 && elect_one_thr) {
      cute::tma_store_wait<0>();
    }
#endif
  }

  __syncthreads();
  if (elect_one_warp) {
    tmem_allocator.free(tmem_base, kTmemColsAlloc);
  }
}

torch::Tensor run_dual_2sm_nvfp4_matmul(torch::Tensor a, torch::Tensor b,
                                        torch::Tensor sfa, torch::Tensor sfb) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a, b must be 2D packed-u8 [rows, k/2]");
  TORCH_CHECK(a.size(1) == b.size(1), "a, b must share packed K");
  TORCH_CHECK(a.dtype() == torch::kUInt8 && b.dtype() == torch::kUInt8,
              "NVFP4 kernel takes packed uint8 (2 e2m1 per byte)");
  TORCH_CHECK(sfa.dtype() == torch::kFloat8_e4m3fn && sfb.dtype() == torch::kFloat8_e4m3fn,
              "scales must be float8_e4m3fn (positive values == ue4m3)");
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && sfa.is_cuda() && sfb.is_cuda());

  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();
  auto sfa_contig = sfa.contiguous();
  auto sfb_contig = sfb.contiguous();
  int64_t m = a_contig.size(0);
  int64_t k = a_contig.size(1) * 2;  // logical K (elements)
  int64_t n = b_contig.size(0);

  TORCH_CHECK(m % kTileM == 0 && n % kTileN == 0 && k % kTileK == 0,
              "Size must be divisible by the pair tile (", kTileM, "x", kTileN, "x", kTileK, ")");

  auto options = a.options().dtype(torch::kFloat32);
  auto c_buffer = torch::empty({m, n}, options);  // beta=0: never read
#if DUAL2SM_D_HALF
  auto d_buffer = torch::empty({m, n}, a.options().dtype(torch::kFloat16));
#else
  auto d_buffer = torch::empty_like(c_buffer);
#endif

  auto tiled_mma = make_tiled_mma(MMA_Atom<MmaOp>{});
  auto bM = tile_size<0>(tiled_mma);                       // 256
  auto bN = tile_size<1>(tiled_mma);                       // kTileN
  auto bK = tile_size<2>(tiled_mma) * Int<kTileK / 64>{};  // atom K=64
  auto mma_tiler = make_shape(bM, bN, bK);

  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  // e2m1 K-rows are kTileK/2 BYTES: kTileK=256 -> SW128 atom, 128 -> SW64.
  using SwAtomA = cute::conditional_t<(kTileK % 256 == 0),
      UMMA::Layout_K_SW128_Atom<TypeA>, UMMA::Layout_K_SW64_Atom<TypeA>>;
  using SwAtomB = cute::conditional_t<(kTileK % 256 == 0),
      UMMA::Layout_K_SW128_Atom<TypeB>, UMMA::Layout_K_SW64_Atom<TypeB>>;
  auto sA_layout = UMMA::tile_to_mma_shape(SwAtomA{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(SwAtomB{}, mma_shape_B);

  // SF smem layouts (vendor deduction; per-CTA, per-stage).
  auto sSFA_layout = BlkScaledConfig::deduce_smem_layoutSFA(tiled_mma, mma_tiler);
  auto sSFB_layout = BlkScaledConfig::deduce_smem_layoutSFB(tiled_mma, mma_tiler);

  using SharedStorageT =
      Dual2SmNvfp4SharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout),
                                decltype(sSFA_layout), decltype(sSFB_layout)>;

  // B75 TMEM admissibility is enforced ON DEVICE (__trap on a statically
  // folded check in the kernel): host-side tmem_ptr arithmetic instantiates
  // cute::rotr as an uninlinable always_inline host function (gcc).

  static_assert(sizeof(SharedStorageT) <= 110 * 1024,
                "Shared storage too large for 2 CTAs/SM; reduce DUAL2SM_STAGES");

  // A/B as logical-element subbyte tensors (K-major).
  Tensor mA = make_tensor(
      make_gmem_ptr(recast_ptr<TypeA const>(reinterpret_cast<uint8_t const*>(a_contig.data_ptr()))),
      make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
  Tensor mB = make_tensor(
      make_gmem_ptr(recast_ptr<TypeB const>(reinterpret_cast<uint8_t const*>(b_contig.data_ptr()))),
      make_layout(make_shape(n, k), make_stride(k, Int<1>{})));
  Tensor mC = make_tensor(make_gmem_ptr(c_buffer.data_ptr<TypeC>()),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
  Tensor mD = make_tensor(make_gmem_ptr(reinterpret_cast<TypeD*>(d_buffer.data_ptr())),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));

  // SF gmem in the cuBLASLt 128x4-blocked layout (== torch._scaled_mm's).
  // This drop's tile_atom_to_shape_SFA/SFB always appends an L mode: slice
  // it off (L=0) so the kernel-side algebra stays rank-2 like mA/mB.
  auto layout_sfa_mkl = BlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k));
  auto layout_sfb_nkl = BlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k));
  TORCH_CHECK(sfa_contig.numel() >= int64_t(cosize(layout_sfa_mkl)),
              "sfa too small: need ", int64_t(cosize(layout_sfa_mkl)), " got ", sfa_contig.numel());
  TORCH_CHECK(sfb_contig.numel() >= int64_t(cosize(layout_sfb_nkl)),
              "sfb too small: need ", int64_t(cosize(layout_sfb_nkl)), " got ", sfb_contig.numel());
  Tensor mSFA_mkl = make_tensor(make_gmem_ptr(reinterpret_cast<TypeSF const*>(sfa_contig.data_ptr())),
                                layout_sfa_mkl);
  Tensor mSFB_nkl = make_tensor(make_gmem_ptr(reinterpret_cast<TypeSF const*>(sfb_contig.data_ptr())),
                                layout_sfb_nkl);
  Tensor mSFA = mSFA_mkl(_, _, 0);
  Tensor mSFB = mSFB_nkl(_, _, 0);

  auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));
  TiledMmaSF tiled_mma_sf{};
  Layout cluster_layout_sfb_vmnk = tiled_divide(make_layout(cluster_shape),
                                                make_tile(typename TiledMmaSF::AtomThrID{}));
  // SFB tile shape: full kTileN (rounded to the 128-row SF block) per CTA.
  auto tile_shape_sf = make_shape(size<0>(mma_tiler) / Int<2>{},
                                  Int<((kTileN + 127) / 128) * 128>{},
                                  size<2>(mma_tiler));

  auto tma_atom_A = make_tma_atom_A_sm100(SM100_TMA_2SM_LOAD{}, mA, sA_layout,
                                          mma_tiler, tiled_mma, cluster_layout_vmnk);
  auto tma_atom_B = make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD{}, mB, sB_layout,
                                          mma_tiler, tiled_mma, cluster_layout_vmnk);
  auto tma_atom_SFA = make_tma_atom_A_sm100<uint16_t>(SM100_TMA_2SM_LOAD{}, mSFA,
                                          sSFA_layout, mma_tiler, tiled_mma,
                                          cluster_layout_vmnk);
  auto tma_atom_SFB = make_tma_atom_B_sm100<uint16_t>(SM100_TMA_2SM_LOAD_MULTICAST{}, mSFB,
                                          sSFB_layout, tile_shape_sf, tiled_mma_sf,
                                          cluster_layout_sfb_vmnk);

  if (std::getenv("AISP_NVFP4_PRINT")) {
    // B49 host print-check: operand + SF partition shapes, smem layouts,
    // TMEM SF column spans, expect-tx bytes.
    print("partition_shape_A: "); print(mma_shape_A); print("\n");
    print("partition_shape_B: "); print(mma_shape_B); print("\n");
    print("sA_layout: "); print(sA_layout); print("\n");
    print("sB_layout: "); print(sB_layout); print("\n");
    print("sSFA_layout: "); print(sSFA_layout); print(" cosize "); print(cosize(sSFA_layout)); print("\n");
    print("sSFB_layout: "); print(sSFB_layout); print(" cosize "); print(cosize(sSFB_layout)); print("\n");
    print("layout_SFA: "); print(mSFA.layout()); print("\n");
    print("layout_SFB: "); print(mSFB.layout()); print("\n");
    // (TMEM SF column spans are checked on device; host tmem_ptr math is
    // a gcc always_inline trap -- see the kernel's __trap check.)
    print("smem/CTA bytes: "); print(int(sizeof(SharedStorageT))); print("\n");
  }

#if DUAL2SM_TMA_EPI
  CUtensorMap tma_desc_D = make_tma_desc_D(d_buffer.data_ptr(), m, n);
#endif

  int grid_pairs_m = int(m) / kTileM;
  int grid_n = int(n) / kTileN;

  dim3 dimBlock(kNumThreads);
  int smem_bytes = sizeof(SharedStorageT);

  auto* kernel_ptr = &gemm_dual_2sm_nvfp4<
      SharedStorageT, kCfg,
      decltype(mA), decltype(mB), decltype(mSFA), decltype(mSFB),
      decltype(mC), decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma), decltype(tile_shape_sf), decltype(tiled_mma_sf),
      decltype(tma_atom_A), decltype(tma_atom_B),
      decltype(tma_atom_SFA), decltype(tma_atom_SFB)>;

  AT_CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  AT_CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributePreferredSharedMemoryCarveout,
                                     cudaSharedmemCarveoutMaxShared));

  // Persistent grid: B65 static sizing = min(smem-implied, TMEM-implied).
  auto* props = at::cuda::getCurrentDeviceProperties();
  int smem_blocks = int(props->sharedMemPerMultiprocessor / size_t(smem_bytes + 1024));
  constexpr int kTmemBlocks =
      cute::TMEM::Allocator2Sm::Sm100TmemCapacityColumns / kTmemColsAlloc;
  int blocks_per_sm = smem_blocks < kTmemBlocks ? smem_blocks : kTmemBlocks;
  TORCH_CHECK(blocks_per_sm >= 1, "persistent grid: zero co-residable CTAs/SM");
  int sm_count = props->multiProcessorCount;
  int total_pair_tiles = grid_pairs_m * grid_n;
  int persist_clusters = (sm_count * blocks_per_sm) / 2;
  if (persist_clusters > total_pair_tiles) persist_clusters = total_pair_tiles;
  dim3 dimGrid(2 * persist_clusters);

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
    (void*)&mA, (void*)&mB, (void*)&mSFA, (void*)&mSFB, (void*)&mC, (void*)&mD,
    (void*)&mma_tiler, (void*)&tiled_mma, (void*)&tile_shape_sf, (void*)&tiled_mma_sf,
    (void*)&tma_atom_A, (void*)&tma_atom_B, (void*)&tma_atom_SFA, (void*)&tma_atom_SFB
#if DUAL2SM_TMA_EPI
    , (void*)&tma_desc_D
#endif
  };

  AT_CUDA_CHECK(cudaLaunchKernelExC(&launch_config, (void*)kernel_ptr, args));

#if DUAL2SM_D_HALF
  return d_buffer;
#else
  return d_buffer.to(torch::kFloat16);
#endif
}

}  // namespace dual_2sm_nvfp4_impl

torch::Tensor matmul_tcgen05_dual_2sm_nvfp4(torch::Tensor a, torch::Tensor b,
                                            torch::Tensor sfa, torch::Tensor sfb) {
  return dual_2sm_nvfp4_impl::run_dual_2sm_nvfp4_matmul(a, b, sfa, sfb);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_tcgen05_dual_2sm_nvfp4", &matmul_tcgen05_dual_2sm_nvfp4);
}
