# GB300 Speed-of-Light roofline: custom-CUDA-kernel labs

SoL-grounding of the repo custom-CUDA-kernel GEMM/compute labs on one GB300 node
(Blackwell Ultra, sm_103, cc 10.3, 152 SMs, 4 GPU). Goal: find which dominant kernels sit
far below Speed-of-Light with an addressable bottleneck (real headroom) vs which are
at-ceiling, then prototype the single highest-headroom lever with verify-pass guardrails.
Excludes nvfp4_gemv and nvfp4_group_gemm (sibling investigations own those).

GB300 per-GPU peaks (sol-ceilings.yaml gb300_nvl72): NVFP4 15.0 PFLOPS, FP8 7.5 PFLOPS,
BF16/FP16 3.75 PFLOPS, HBM3e 8.0 TB/s, NVLink5 1.8 TB/s. Max dynamic smem 227 KB/block
(228 KB/SM), measured on-node.

## Headline

1. The one hand-owned kernel with deep, addressable headroom is custom_vs_cublas
   (gemm_cluster, CuTe tcgen05, FP16 8192^3): 32.2% FP16 FLOP-SoL (ncu --set full,
   kernel-only), latency/occupancy-bound at 6.2% achieved occupancy. cuBLAS on the identical
   shape/node reaches 49.3% FP16-SoL (1.6x faster), so the gap is real, not noise.
2. blackwell_matmul reuses the same tcgen05 CuTe kernel family at 2048^3, where wall-clock is
   host-overhead-dominated (kernel approx 16 us vs approx 213 us wall). Same structural
   kernel, smaller, overhead-dominated shape.
3. nvfp4_gemm (CUTLASS NVFP4, M=128) is memory-bound and near-ceiling on its large-K shape
   (approx 65% HBM-SoL); its tiny shapes are launch/overhead-bound. Library-representative,
   not a hand kernel.
4. moe_cuda_ptx and ozaki_scheme are library-bound (torch.bmm cuBLAS / cuBLAS FP-emulation).
   The MoE PTX hand-kernel path is unimplemented (raises SKIPPED). At-ceiling for the library
   they call.

## Per-lab roofline, ranked by hand-owned addressable headroom

| Rank | Lab : dominant kernel | dtype, shape | Achieved | %SoL | Bound | Class | Grounding |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | custom_vs_cublas : gemm_cluster (CuTe tcgen05, SM100_MMA_F16BF16_SS 128x256) | FP16, 8192^3, C=A@B^T | 1207 TFLOPS @ 911 us | 32.2% FP16-FLOP | latency/occupancy (occ 6.2%, 1 CTA/SM, smem 192/227 KB cap, full-TMEM cap; top stall scoreboard-on-smem 46.5%) | HEADROOM (richest; cuBLAS 49.3% = 1.6x faster, same node) | ncu --set full |
| 2 | blackwell_matmul : tcgen05 (fullstack_cluster CuTe, shared family) | FP16, 2048^3 | approx 80 TFLOPS wall (kernel approx 16 us vs approx 213 us wall) | approx 2% wall, est approx 30% kernel | host-overhead-bound at this size | HEADROOM (same structural kernel as #1) | wall-clock (DRAFT; ncu follow-up) |
| 3 | nvfp4_gemm : CUTLASS NVFP4 GEMM (auto/stage5) | NVFP4, M=128, (N,K) = (7168,16384),(4096,7168),(7168,2048) | 2.46 PF big-K; 5.66/2.25/2.06 TB/s | 16% FLOP; approx 65/28/26% HBM | memory-bound (big-K near-ceiling); tiny shapes launch-bound | MOSTLY AT-CEILING (library, mem-bound) | binary TIME_MS |
| 4 | moe_cuda_ptx : torch.bmm (cuBLAS BF16 grouped) | BF16, 32768 tok, 8 experts, hidden 7168 | fwd 1.58 PF, bwd 0.39 PF, layer 0.65 PF | fwd 42%, bwd 10%, layer 17% BF16-FLOP | library (cuBLAS bmm); PTX hand path unimplemented | AT-CEILING (library) | harness throughput |
| 5 | ozaki_scheme : cuBLAS FP-emulation (int8 Ozaki) | FP64-emul, 4096^3, retained 4 bits | 61.3 TFLOPS effective | n/a (int8 peak not in ceilings) | library (cuBLAS emulation) | AT-CEILING (library) | binary TIME_MS |

Notes on classification:
- HEADROOM means --set full (or a measured library reference) shows real room below peak with
  an addressable bottleneck. AT-CEILING means the kernel is the library or is memory/overhead
  bound at the limit for its shape; the %SoL number is the valuable result, not a defect.
- nvfp4_gemm at M=128 is weight-streaming, so HBM-SoL (approx 65% on the large-K shape) is the
  right frame, not FLOP-SoL (16%). Its small shapes (4 to 7 us kernels) are launch/overhead
  bound, which is a property of the tiny problem, not an addressable kernel defect.
- moe_cuda_ptx and ozaki call cuBLAS; the per-shape %SoL reflects library efficiency on those
  shapes. There is no hand kernel to tune (the MoE PTX backend raises SKIPPED: not implemented).

## Top lever prototyped: eliminate the beta=0 C matrix in gemm_cluster

custom_vs_cublas computes a pure C = A @ B^T (beta = 0), yet the kernel materialized and
consumed a full C matrix for nothing:
- host: torch::zeros({8192,8192}, fp32) every call = a 256 MB memset (a separate kernel before
  the GEMM).
- epilogue: copy(tDgC, tDrC) loaded that 256 MB C from gmem, then axpby(1.0, acc, 0.0, C)
  discarded it (scaled by 0.0).

Change (labs/custom_vs_cublas/tcgen05_cluster.cu only): torch::zeros to torch::empty (C is
never read), drop the C-load and the zero-scaled axpby, write the TMEM accumulator straight
to D. Output math is identical (D = A @ B^T).

Before vs after, same node, same FP16 8192^3 shape:

| Metric | Before | After |
| --- | --- | --- |
| ncu kernel-only duration | 911 us | 904 us |
| ncu FP16 FLOP-SoL | 32.2% | 32.4% |
| official harness optimized time | 1.249 ms mean, 1.07 ms median | 1.195 ms mean, 1.05 ms median |
| official speedup vs naive baseline | 2.260x | 2.329x |
| verify (output correctness) | PASS, rel 2.6e-4 | PASS, rel 2.6e-4, bit-identical |
| probe wall-clock, 20-iter loop | 0.986 ms | 0.952 ms |

Honest read: the win is small (approx 3% wall-clock, approx 0.8% kernel) and verify-clean. It
is small precisely because the kernel is latency/occupancy-bound, not bandwidth-bound (DRAM
only 33%), so removing 256 MB of memory traffic barely moves the 904 us kernel. The wall-clock
gain is mostly the eliminated per-call host memset. It is a strict improvement (faster and
bit-identical output), so it stays applied; the unmodified kernel is backed up at
/tmp/tcgen05_cluster.cu.bak.

## Deep lever, named next (structurally capped for an incremental change)

The 32% to 50%-plus headroom requires raising occupancy/latency-hiding, which is structurally
blocked for a small edit:
- occupancy is 6.2% (1 CTA/SM). smem per CTA is approx 192 KB of a 227 KB/block limit
  (Block Limit Shared Mem = 1), so a 5th pipeline stage (240 KB) does not fit; and the kernel
  allocates the full TMEM (512 cols) for its 128x256 accumulator, so a 2nd CTA/SM cannot get
  TMEM either. Both occupancy levers are capped.
- the real fix is a warp-specialized, persistent kernel with split TMEM (256 cols/CTA) and a
  producer/consumer pipeline (the CUTLASS sm100 GEMM design). That is a kernel rewrite, not a
  flag or one-liner, so it is the named next lever rather than something to land under
  verify-pass guardrails in this pass.

The progressive variant kernels already in the lab confirm the plateau: wall-clock FP16-SoL
across cluster 29.7%, warp_parallel 31.0%, no_wait_swizzle 28.9%, no_wait 27.1%, swizzled
21.7%, 3stage 20.9%. They all share the 1-CTA/SM tcgen05 structure, so none breaks past
approx 31% without the occupancy rewrite.

## Method and caveats

- Free-GPU pick before each ncu/bench; --profile none plus a pinned CUDA_VISIBLE_DEVICES to
  avoid the sibling co-tenancy that trips the harness Foreign-CUDA-compute guard and that
  deadlocks the 2x1 cluster launch (the earlier 17-phase --profile minimal stall was this
  contention, not a kernel bug; on a free GPU the kernel runs clean).
- ncu line: /usr/local/bin/ncu --set full --launch-skip 5 --launch-count 2 --kernel-name
  regex:gemm_cluster --target-processes application-only, env set on the ncu process (no
  env VAR=v python after the separator).
- %SoL = achieved TFLOPS / FLOP peak (compute-bound) or achieved GB/s / 8.0 TB/s
  (memory-bound). custom_vs_cublas is ncu kernel-only; the other four labs are
  self-reported binary or harness throughput (weaker grounding, ncu follow-up recommended for
  nvfp4_gemm and blackwell_matmul).
- Verify gate used throughout: python -m cli.aisp bench run --targets
  labs/custom_vs_cublas:tcgen05_matmul --profile none --single-gpu must report verification
  passed with no failed_verification.