# Upstream patch / PR candidates from the GB300 campaign (B36–B58)

Findings from the 2026-06-10/11 GB300 (sm_103) optimization campaign that are
reportable or patchable OUTSIDE this repo, ranked by expected impact and
readiness. Every item carries banked evidence (runbook entry + evidence doc +
pod artifacts). In-repo fixes are already landed on main and are not listed.

## Tier 1 — ready to file (evidence complete, reproducer exists)

### 1. PyTorch Inductor: extern-kernel autotune mis-measures broadcast-bias `baddbmm`
- **Repo:** pytorch/pytorch (Inductor autotuning)
- **Finding (B56 evidence, /tmp/frontM3 on pod):** Inductor's extern-kernel
  autotune benchmarks ATEN `baddbmm` with a broadcast-bias stride `[2048,0,1]`
  benchmark artifact at 0.1403 ms when the real nvjet kernel runs 66 us — so a
  slower Triton template wins the autotune. ~40% perf left on the table at
  these shapes (32x640x512x2048 BF16 class).
- **Patch shape:** fix the extern benchmark's stride handling for broadcast
  bias (or benchmark the actual decomposed bmm+pointwise path).
- **Companion enhancement:** Inductor never considers the
  `bmm + pointwise-bias` decomposition for broadcast-bias `baddbmm`; B56
  measured the decomposition 1.31x faster end-to-end (routes to beta-zero
  nvjet). Could ship as a decomposition rule or autotune candidate.

### 2. CUTLASS: tcgen05 tutorial/example uses the pathological narrow TMEM_LOAD atom
- **Repo:** NVIDIA/cutlass (examples + cute tutorials for sm100/sm103)
- **Finding (B55, docs/gb300-capstone-f-decomposition.md):**
  `SM100_TMEM_LOAD_32dp32b1x` lowers (ptxas, CUDA 13.2) to a per-load
  LEPC + CALL.ABS.NOINC + WARPSYNC helper sequence — 256 loads/thread = 512
  helper calls dominating an entire kernel's fixed cost (7.7 us of a 24 us
  kernel). Swapping to `32dp32b32x` is a one-line 1.49x kernel win; 128x
  REGRESSES (serialized writeback). Tutorials that demonstrate the 1x atom
  teach a 1.5x perf bug.
- **Patch shape:** docs/example update recommending wide atoms (32x sweet
  spot measured) + a note on the helper-call lowering. Optionally a ptxas
  issue (via NVIDIA bug tracker) asking whether the helper-call lowering of
  narrow tcgen05.ld is intended.

### 3. CUTLASS: 2x1SM MMA atom operand-split semantics undocumented/stale prints
- **Repo:** NVIDIA/cutlass
- **Finding (B49/B57, docs/gb300-gemm-occupancy-rewrite.md):** the
  `SM100_MMA_F16BF16_2x1SM_SS` atom SPLITS B N/2-per-CTA across the pair
  (proven by smem footprints + mbarrier tx-byte balance); the tutorial's
  printed layouts suggest otherwise (stale prints). Also: `tma_partition`'s
  multicast slice is an OFFSET VIEW — expect-tx bytes must be the full
  delivered slice with NO participant multiplier; getting this wrong is a
  deterministic mbarrier deadlock (two independent fronts hit it: B53, B57).
- **Patch shape:** docs clarification + tutorial print fix + a comment in
  tutorial-04 on the expect-tx formula.

## Tier 2 — reportable findings (needs a minimal standalone reproducer first)

### 4. PyTorch: `torch._scaled_grouped_mm` refuses NVFP4
- **Repo:** pytorch/pytorch
- **Finding (B28):** the grouped scaled-mm API accepts FP8/MXFP8 only; NVFP4
  workloads are forced to 30 separate `_scaled_mm` launches, which a custom
  fused kernel beats 3.1x wall. Feature request: NVFP4 grouped support
  (sm100/sm103 hardware supports it).

### 5. PyTorch: `torch._int_mm` 13.4x slower than tf32 matmul on sm_103
- **Repo:** pytorch/pytorch
- **Finding (B20):** INT8 `_int_mm` runs 13.4x slower than the tf32 path on
  GB300 — the sm_103 INT8 GEMM dispatch appears to hit a slow kernel. Perf
  bug report with shapes + timings.

### 6. Triton: sm_103 max-autotune emits unselectable `tcgen05.wait.st`
- **Repo:** triton-lang/triton (and pytorch torch.compile)
- **Finding (toolchain section + B23-era):** `torch.compile(mode=
  "max-autotune")` on sm_103 with Triton 3.7 (and 3.5.0 on torch 2.12)
  aborts with `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st`.
  Forces every lab to fall back to reduce-overhead. Likely known upstream;
  verify against current Triton main before filing.

## Tier 3 — book/educational content (this repo's own publishing pipeline)

- The durable lesson set (B36–B58): beta=0 dead-path class; TMEM_LOAD atom
  width; hardware in-flight TMA merge vs explicit multicast; "sector waste is
  only recoverable where binding"; "traffic removal isn't a win when its
  synchronization costs more than the bytes"; graph-break removal moving the
  cudagraph capture boundary (13x/16.5x); silent capture-fallback audit rule;
  F-decomposition methodology (globaltimer probe builds); GPU lease protocol
  for multi-agent benchmarking. These are chapters/errata for the book, not
  external PRs.

## Process note
Each Tier-1 item should ship with: the minimal .cu/.py reproducer (extracted
from the lab, no repo dependencies), measured numbers from this campaign
(GB300, CUDA 13.2, torch 2.12 NGC 26.05, CUTLASS 4.3.0), and a link-free
description (internal pod paths stripped). File from a personal fork; CC the
relevant CODEOWNERS.
