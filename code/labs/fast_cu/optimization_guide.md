# Where fast.cu fits in this repository

The source is [fast.cu](https://github.com/pranjalssh/fast.cu) and its accompanying
[H100 worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
and [NVFP4 article](https://cudaforfun.substack.com/p/outperforming-cublas-on-nvfp4).

## Existing coverage and the added experiments

| Subject | Existing home | What this lab adds |
| --- | --- | --- |
| Arithmetic intensity, tiling, fusion | [ch09](../../ch09/README.md), including `tcgen05_tma_pipeline` and `micro_tiling_matmul` | Complete Hopper kernel progression and a cuBLAS comparison with the original kernel 12 |
| Tensor-core pipelines, TMA, warp roles, clusters | [ch10](../../ch10/README.md), [blackwell_matmul](../blackwell_matmul/README.md) | Complete plain CUDA/PTX GB300 r0-r9 ladder with separate A/B and scale queues and two-CTA MMA |
| Warp reductions and bandwidth | [ch10 atomic/DSMEM reduction](../../ch10/README.md), [memory_bandwidth_patterns](../memory_bandwidth_patterns/README.md) | CUB versus a safe adaptation of fast.cu's vectorized int32 reduction, including B200 execution |
| Stream ordering and overlap | [ch11](../../ch11/README.md) | Adapters that launch and reset outputs on the caller's CUDA stream |
| Compiler-sensitive instruction selection | [ch14](../../ch14/README.md) | `%laneid`, exact block geometry, wide stores, and an isolated B200 epilogue experiment |
| B200 block-scaled FP4 GEMM | [nvfp4_gemm](../nvfp4_gemm/README.md) | GB300 SM103/K96 implementation alongside the existing SM100 path |

Start with the chapter examples for individual techniques. Use this lab to
follow the complete kernel progression and run the benchmark comparisons.

## Every upstream example is present

- [Hopper GEMM driver](upstream/h100/matmul.cu) and
  [matmul_1.cuh through matmul_12.cuh](upstream/h100/matmul/) preserve the progression
  from elementary matrix multiplication through asynchronous, warp-specialized,
  persistent scheduling. The supplementary `cublaslt_matmul.cu` and
  `pingpong_experimental.cuh` are also retained as upstream reference experiments.
- [Hopper reduction](upstream/h100/sum.cu) preserves all four source variants and
  its CUB comparison. Both the copied source and the native adapter fix the
  original grid-wide output-reset race. The copied driver now uses bounded
  inputs and checks each result against an independent int64 reference.
- [GB300 driver](upstream/gb300/nvfp4/main.cu) and the ten headers below preserve
  the entire NVFP4 progression, scheduler controls, and correctness runner. The
  r0-r9 kernels are unchanged from upstream. They compiled and imported with
  CUDA 13.1.80, but no SM103 GPU was available for runtime validation.

The proposed upstream patches address defects found in the older H100 reduction
and scheduler examples. They do not change the newer GB300 NVFP4 kernels.

## NVFP4 ladder and what transfers to B200

NVFP4 stores E2M1 values with an FP8 scale for each group of 16 along K. Values
and scales have different layouts and traffic patterns. The source uses TMA
queues for both, FP32 accumulators in TMEM, and FP16 output. Correctness depends
on descriptor layout, queue lifetime, barrier phase, and output coverage together.
The B200 epilogue experiment changes only FP32-to-FP16 store width. It is not a
port of the full NVFP4 GEMM to B200.

| Rung | Local source | Mechanism | B200 interpretation |
| --- | --- | --- | --- |
| r0 | [gemm0.cuh](upstream/gb300/nvfp4/gemm0.cuh) | Working two-CTA NVFP4 pipeline with K192 feed | K96 instructions require SM103. Use the existing SM100 GEMM for K64 |
| r1 | [gemm1.cuh](upstream/gb300/nvfp4/gemm1.cuh) | Direct `%laneid` for single-lane work | Inspect uniform-register and convergence code generation. Do not assume a speedup |
| r2 | [gemm2.cuh](upstream/gb300/nvfp4/gemm2.cuh) | K256 TMA windows with independently released buffers | Queue/barrier design transfers. The K96/K256 decomposition does not transfer unchanged |
| r3 | [gemm3.cuh](upstream/gb300/nvfp4/gemm3.cuh) | Overlap two TMEM accumulator buffers | Relates to ch10 ping-pong pipelines. Check TMEM capacity and accumulator lifetime |
| r4 | [gemm4.cuh](upstream/gb300/nvfp4/gemm4.cuh) | 256-bit output stores | Implemented as the native `b200_epilogue` store-width pair |
| r5 | [gemm5.cuh](upstream/gb300/nvfp4/gemm5.cuh) | Avoid L1 allocation and favor early L2 eviction for output | Both B200 epilogue arms use the same policy to isolate width. This does not measure the policy's effect |
| r6 | [gemm6.cuh](upstream/gb300/nvfp4/gemm6.cuh) | Exact block shape improves compiler knowledge | Exact launch geometry must remain true and requires suitable compiler support |
| r7 | [gemm7.cuh](upstream/gb300/nvfp4/gemm7.cuh) | Remove dead tail work | Transfer only after proving coverage for the actual tile/K decomposition |
| r8 | [gemm8.cuh](upstream/gb300/nvfp4/gemm8.cuh) | Fold compatible tails into K64 MMAs | K64 is supported on SM100. The mixed K96/K64 schedule remains SM103-specific |
| r9 | [gemm9.cuh](upstream/gb300/nvfp4/gemm9.cuh) | L2-side ownership and shape-dependent visit order | Measure actual topology. GB300 cluster placement may differ from B200 |

The r9 scheduling lesson has two parts: decide which cluster population reuses a
row, then place those reuses close together in visitation order. A tiled or Hilbert
order alone does not establish ownership. Topology measurements and route coverage
checks are prerequisites. A silent raster fallback cannot count as r9 evidence.

## Lessons carried into the adapters

1. Keep the same computation when changing storage layout. Check the full output.
2. Separate buffer preparation from execution and expose the real output written
   by the timed kernel. Hashes verify source identity, not numerical correctness.
3. Treat CUDA architecture suffixes as requirements. SM100 and SM103 are not
   interchangeable simply because both are Blackwell.
4. Change one thing at a time. The B200 epilogue holds loads,
   conversion, cache policy, launch geometry, and output size constant while
   changing only store width.
5. Keep input distribution and integer range explicit. Repairing an atomic reset
   race is insufficient if the reduction can still overflow.
6. Distinguish a source author's measurements from this repository's measurements.
   Buffer rotation, order, cooldown, clocks, power, and compiler versions can
   change a small performance margin. Record those choices before comparing.

Measure each technique on the target GPU before claiming a speedup. Check the
full output, repeat paired measurements with equivalent workloads, record clocks
and software versions, and use Nsight to explain the difference.
