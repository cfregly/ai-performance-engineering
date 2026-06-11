// DRAFT — internal review pending
//
// Minimal standalone reproducer: ptxas (CUDA 13.2) lowers EVERY
// SM100_TMEM_LOAD_32dp32b1x tcgen05.ld into a per-load
// LEPC + CALL.ABS.NOINC + WARPSYNC convergence-helper subroutine call.
// For a 128x256 fp32 accumulator that is 256 loads/thread = 256 helper
// calls per warp in the unrolled SASS, which dominates a small kernel's
// fixed cost. Widening the atom to SM100_TMEM_LOAD_32dp32b32x (8
// loads/thread) removes them.
//
// This file contains ONLY the TMEM allocate + tmem->register copy +
// register->gmem store epilogue (no mainloop, no MMA), so the SASS of the
// copy is easy to inspect. The TiledMMA is used solely to derive the
// canonical 128x256 TMEM accumulator tensor that make_tmem_copy partitions
// — the same construction the CuTe Blackwell tutorials use.
//
// Build (CUTLASS >= 4.2 headers, CUDA >= 13.x, sm_103a or sm_100a):
//   nvcc -std=c++20 -arch=sm_103a -I$CUTLASS_DIR/include \
//        -DT2R_ATOM=SM100_TMEM_LOAD_32dp32b1x  -o repro_1x  tmem_load_atom_repro.cu
//   nvcc -std=c++20 -arch=sm_103a -I$CUTLASS_DIR/include \
//        -DT2R_ATOM=SM100_TMEM_LOAD_32dp32b32x -o repro_32x tmem_load_atom_repro.cu
//
// SASS evidence:
//   cuobjdump -sass repro_1x  | grep -c "CALL.ABS.NOINC"   # ~256 (one per LDTM)
//   cuobjdump -sass repro_32x | grep -c "CALL.ABS.NOINC"   # ~8
//   cuobjdump -sass repro_1x  | grep -B2 -A2 "LDTM" | head # LEPC/CALL/WARPSYNC wrap
//
// Optional run (any SM100-family GPU): ./repro_1x ; ./repro_32x
// prints CUDA-event kernel time; the 1x build is several times slower for
// the identical (bit-identical destination mapping) data movement.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <cutlass/half.h>

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>

using namespace cute;

#ifndef T2R_ATOM
#define T2R_ATOM SM100_TMEM_LOAD_32dp32b1x
#endif

// Stringify the atom name for the report.
#define XSTR(s) STR(s)
#define STR(s) #s

using TypeAB = cutlass::half_t;
using Accum = float;

constexpr int kTileM = 128;
constexpr int kTileN = 256;

__global__ void tmem_epilogue_kernel(float* __restrict__ out) {
  // 1-SM 128x256 UMMA atom: used only to derive the canonical TMEM
  // accumulator layout (tmem_ptr tensor) for make_tmem_copy.
  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<TypeAB, TypeAB, Accum, kTileM, kTileN,
                           UMMA::Major::K, UMMA::Major::K>{});
  auto cta_mma = tiled_mma.get_slice(0);

  Tensor mD = make_tensor(make_gmem_ptr(out),
                          make_layout(make_shape(Int<kTileM>{}, Int<kTileN>{}),
                                      make_stride(Int<kTileN>{}, _1{})));
  Tensor tCgD = cta_mma.partition_C(mD);
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgD);  // TMEM-resident accumulator

  __shared__ uint32_t tmem_base_ptr;
  cute::TMEM::Allocator1Sm tmem_allocator{};
  if (threadIdx.x / 32 == 0) {
    tmem_allocator.allocate(
        cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns, &tmem_base_ptr);
  }
  __syncthreads();
  tCtAcc.data() = tmem_base_ptr;

  // ---- The one line under test: the TMEM_LOAD copy-atom width. ----------
  auto tiled_t2r_copy = make_tmem_copy(T2R_ATOM{}, tCtAcc);
  // -----------------------------------------------------------------------
  auto thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
  Tensor tDgD = thr_t2r_copy.partition_D(tCgD);
  Tensor tDrAcc = make_tensor<Accum>(shape(tDgD));

  copy(tiled_t2r_copy, tDtAcc, tDrAcc);  // TMEM -> registers (tcgen05.ld)
  copy(tDrAcc, tDgD);                    // registers -> gmem (keeps it live)

  __syncthreads();
  if (threadIdx.x / 32 == 0) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base_ptr,
                        cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns);
  }
}

int main() {
  float* d_out = nullptr;
  if (cudaMalloc(&d_out, sizeof(float) * kTileM * kTileN) != cudaSuccess) {
    std::printf("cudaMalloc failed (no SM100-family GPU?)\n");
    return 1;
  }

  // One warmup + error check.
  tmem_epilogue_kernel<<<1, 128>>>(d_out);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::printf("kernel failed: %s\n", cudaGetErrorString(err));
    return 1;
  }

  constexpr int kIters = 2000;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);
  for (int i = 0; i < kIters; ++i) {
    tmem_epilogue_kernel<<<1, 128>>>(d_out);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, beg, end);

  std::printf("atom=%s  kernel time: %.3f us/launch (%d launches)\n",
              XSTR(T2R_ATOM), 1000.0f * ms / kIters, kIters);
  cudaFree(d_out);
  return 0;
}
