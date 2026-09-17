# Where fast.cu fits in this repository

The source is [fast.cu](https://github.com/pranjalssh/fast.cu) and its accompanying
[H100 worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
and [NVFP4 article](https://cudaforfun.substack.com/p/outperforming-cublas-on-nvfp4).
The supplied PDF of the NVFP4 article was consulted, including its optimization
ladder figure and measurement protocol. The PDF itself is not redistributed.

## Existing coverage and the added experiments

| Subject | Existing home | What this lab adds |
| --- | --- | --- |
| Arithmetic intensity, tiling, fusion | [ch09](../../ch09/README.md), including `tcgen05_tma_pipeline` and `micro_tiling_matmul` | Complete Hopper kernel progression and a cuBLAS comparison with the original kernel 12 |
| Tensor-core pipelines, TMA, warp roles, clusters | [ch10](../../ch10/README.md), [blackwell_matmul](../blackwell_matmul/README.md) | Complete plain CUDA/PTX GB300 r0-r9 ladder with separate A/B and scale queues and two-CTA MMA |
| Warp reductions and bandwidth | [ch10 atomic/DSMEM reduction](../../ch10/README.md), [memory_bandwidth_patterns](../memory_bandwidth_patterns/README.md) | CUB versus a safe adaptation of fast.cu's vectorized int32 reduction, including B200 execution |
| Stream ordering and overlap | [ch11](../../ch11/README.md) | Explicit current-stream native adapters and stream-ordered reduction reset |
| Compiler-sensitive instruction selection | [ch14](../../ch14/README.md) | `%laneid`, exact block geometry, wide stores, and an isolated B200 epilogue experiment |
| B200 block-scaled FP4 GEMM | [nvfp4_gemm](../nvfp4_gemm/README.md) | Distinct GB300 SM103/K96 implementation; it complements the existing SM100 path |

Existing chapter implementations remain the teaching entrypoints for common
concepts. The new lab keeps the source progression and controlled comparisons
together so that an unfamiliar reader can inspect the complete examples locally.

## Every upstream example is present

- [Hopper GEMM driver](upstream/h100/matmul.cu) and
  [matmul_1.cuh through matmul_12.cuh](upstream/h100/matmul/) preserve the progression
  from elementary matrix multiplication through asynchronous, warp-specialized,
  persistent scheduling. The supplementary `cublaslt_matmul.cu` and
  `pingpong_experimental.cuh` are also retained as upstream reference experiments.
- [Hopper reduction](upstream/h100/sum.cu) preserves all four source variants and
  its CUB comparison. The native adapter uses a corrected variant; the original
  program has a grid-wide output-reset race and weak correctness handling.
- [GB300 driver](upstream/gb300/nvfp4/main.cu) and the ten headers below preserve
  the entire NVFP4 progression, scheduler controls, and correctness runner.

## NVFP4 ladder and what transfers to B200

NVFP4 stores E2M1 values with an FP8 scale for each group of 16 along K. Values
and scales have different layouts and traffic patterns. The source uses TMA
queues for both, FP32 accumulators in TMEM, and FP16 output. Correctness depends
on descriptor layout, queue lifetime, barrier phase, and output coverage together.

| Rung | Local source | Mechanism | B200 interpretation |
| --- | --- | --- | --- |
| r0 | [gemm0.cuh](upstream/gb300/nvfp4/gemm0.cuh) | Working two-CTA NVFP4 pipeline with K192 feed | K96 instructions require SM103; use the existing SM100 GEMM for K64 |
| r1 | [gemm1.cuh](upstream/gb300/nvfp4/gemm1.cuh) | Direct `%laneid` for single-lane work | Inspect uniform-register and convergence code generation; avoid assuming a speedup |
| r2 | [gemm2.cuh](upstream/gb300/nvfp4/gemm2.cuh) | K256 TMA windows with independently released buffers | Queue/barrier design transfers; the K96/K256 decomposition does not transfer unchanged |
| r3 | [gemm3.cuh](upstream/gb300/nvfp4/gemm3.cuh) | Overlap two TMEM accumulator buffers | Relates to ch10 ping-pong pipelines; check TMEM capacity and accumulator lifetime |
| r4 | [gemm4.cuh](upstream/gb300/nvfp4/gemm4.cuh) | 256-bit output stores | Implemented as the native `b200_epilogue` store-width pair |
| r5 | [gemm5.cuh](upstream/gb300/nvfp4/gemm5.cuh) | Avoid L1 allocation and favor early L2 eviction for output | Both B200 epilogue arms use the same policy to isolate width; policy itself is not ablated |
| r6 | [gemm6.cuh](upstream/gb300/nvfp4/gemm6.cuh) | Exact block shape improves compiler knowledge | Exact launch geometry must remain true; requires suitable compiler support |
| r7 | [gemm7.cuh](upstream/gb300/nvfp4/gemm7.cuh) | Remove dead tail work | Transfer only after proving coverage for the actual tile/K decomposition |
| r8 | [gemm8.cuh](upstream/gb300/nvfp4/gemm8.cuh) | Fold compatible tails into K64 MMAs | K64 is supported on SM100; the mixed K96/K64 schedule remains SM103-specific |
| r9 | [gemm9.cuh](upstream/gb300/nvfp4/gemm9.cuh) | L2-side ownership and shape-dependent visit order | Measure actual topology; do not transplant a GB300 route/census assumption onto B200 |

The r9 scheduling lesson has two parts: decide which cluster population reuses a
row, then place those reuses close together in visitation order. A tiled or Hilbert
order alone does not establish ownership. Topology measurements and route coverage
checks are prerequisites; a silent raster fallback cannot count as r9 evidence.

## Lessons carried into the adapters

1. Preserve the entire arithmetic contract. Storage layout changes must not change
   the logical matrix or hide output elements from verification.
2. Separate buffer preparation from execution and expose the real output written
   by the timed kernel. Hashes verify source identity, not numerical correctness.
3. Treat CUDA architecture suffixes as requirements. SM100 and SM103 are not
   interchangeable simply because both are Blackwell.
4. Use a controlled ablation for a portable idea. The B200 epilogue holds loads,
   conversion, cache policy, launch geometry, and output size constant while
   changing only store width.
5. Keep input distribution and integer range explicit. Repairing an atomic reset
   race is insufficient if the reduction can still overflow.
6. Distinguish a source author's measurements from this repository's measurements.
   Buffer rotation, order, cooldown, clocks, power, and compiler versions can
   change a small performance margin. Record those choices before comparing.

No expected speedup is installed from the blog. A performance conclusion requires
equivalent workloads, full-output correctness, repeated paired measurements,
clock/provenance records, and Nsight evidence from the target hardware.
