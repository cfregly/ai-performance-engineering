// DRAFT — internal review pending
//
// Host-only print program: shows that the SM100_MMA_F16BF16_2x1SM_SS atom
// SPLITS the B operand N/2-per-CTA across the CTA pair, contradicting the
// inline print-annotation comments in CUTLASS
// examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu, which claim a
// full-N per-CTA B tile, e.g. (annotations as of CUTLASS 4.2.0..main):
//
//   print("mma_shape_B:\t"); ... // mma_shape_B:  ((_256,_16),_1,_4)
//   print("sB_layout:\t");   ... // sB_layout: Sw<3,4,3> o ... ((_256,_16),_1,_4):((_64,_1),_0,_16)
//   print("tCsB:\t");        ... // tCsB:  Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_B) o ((_256,_16),_1,_4):((_64,_1),_0,_16)
//   print("tBsB:\t");        ... // tBsB:  Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_B) o ((_16384,_1)):((_1,_0))
//
// This program builds the exact tutorial-04 TiledMMA (2x1SM 256x256 atom,
// MmaTiler 256x256x64) and prints what partition_shape_B / tile_to_mma_shape
// actually return: the B mode-0 is (_128,_16) — N/2 = 128 columns per CTA,
// 16 KiB per stage, not ((_256,_16)) / 32 KiB.
//
// Build + run (no GPU needed — everything here is host-side layout algebra):
//   nvcc -std=c++20 -arch=sm_103a -I$CUTLASS_DIR/include -o snippet snippet.cu
//   ./snippet
//
// Why it matters (measured on GB300 sm_103, CUDA 13.2): the per-CTA smem
// footprint and the mbarrier expect-tx byte budget are both derived from
// these layouts. A kernel author who trusts the full-N annotations
// over-allocates smem 1.5x per stage (48 KiB vs 32 KiB at 256x256x64) and,
// worse, over-programs the TMA transaction-byte count — an over-expected
// mbarrier NEVER fires, which is a deterministic kernel hang, not an error.

#include <cstdio>

#include <cutlass/half.h>

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>

using namespace cute;

using TypeA = cutlass::half_t;
using TypeB = cutlass::half_t;
using TypeC = float;

int main() {
  // Exactly tutorial 04: 2x1SM atom, pair tile M=256, N=256, bK=64.
  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_2x1SM_SS<TypeA, TypeB, TypeC, 256, 256,
                                 UMMA::Major::K, UMMA::Major::K>{});
  auto mma_tiler = make_shape(_256{}, _256{}, _64{});

  auto mma_shape_A = partition_shape_A(
      tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(
      tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  print("mma_shape_A:\t"); print(mma_shape_A); print("\n");
  print("mma_shape_B:\t"); print(mma_shape_B); print("\n");
  // tutorial 04 comment claims:   mma_shape_B:  ((_256,_16),_1,_4)
  // actually printed:             mma_shape_B:  ((_128,_16),_1,_4)
  //   -> the 2x1SM atom gives each CTA of the pair its own N/2 B columns.

  auto sA_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
  auto sB_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);
  print("sA_layout:\t"); print(sA_layout); print("\n");
  print("sB_layout:\t"); print(sB_layout); print("\n");

  std::printf("per-CTA smem A bytes/stage: %d\n",
              int(cosize(sA_layout) * sizeof(TypeA)));
  std::printf("per-CTA smem B bytes/stage: %d  (full-N would be %d)\n",
              int(cosize(sB_layout) * sizeof(TypeB)),
              int(2 * cosize(sB_layout) * sizeof(TypeB)));

  // ---------------------------------------------------------------------
  // The expect-tx trap (device-side; reproduced here as the verified
  // formula). With 2SM TMA loads (SM100_TMA_2SM_LOAD), BOTH CTAs of the
  // pair issue their own loads, the hardware redirects every mbarrier
  // arrival to the EVEN (leader) CTA's barrier, and ONLY the leader runs
  // set_barrier_transaction_bytes. The correct per-stage byte count is the
  // tutorial-04 formula:
  //
  //   int tma_transaction_bytes =
  //       size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tAsA))
  //     + size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tBsB));
  //
  // i.e. 2 x (per-CTA A slice + per-CTA B slice) — the pair factor
  // size<0>() == 2 and NOTHING ELSE. Two traps, both producing a barrier
  // that never fires (deterministic hang, no error):
  //   1. tma_partition's multicast slice is an OFFSET VIEW into the stage,
  //      not a shrunken tensor: sizeof(make_tensor_like(slice)) is already
  //      the full bytes DELIVERED INTO THIS CTA for that operand. Do NOT
  //      multiply by the number of multicast participants (e.g. kClusterN
  //      when A is multicast across the cluster N-mode): cta_group::2
  //      multicast counts each issue's bytes once per destination pair.
  //   2. Do NOT size the B term from the full-N tile (see prints above):
  //      the atom splits B, so the slice is N/2 wide.
  // A peer-CTA TMA completing before the leader's expect_tx is legal: the
  // mbarrier transaction count may go transiently negative.
  // ---------------------------------------------------------------------
  std::printf(
      "leader expect-tx bytes/stage = 2 * (A_slice + B_slice) = 2 * (%d + %d) = %d\n",
      int(cosize(sA_layout) * sizeof(TypeA)),
      int(cosize(sB_layout) * sizeof(TypeB)),
      int(2 * (cosize(sA_layout) * sizeof(TypeA) +
               cosize(sB_layout) * sizeof(TypeB))));
  return 0;
}
