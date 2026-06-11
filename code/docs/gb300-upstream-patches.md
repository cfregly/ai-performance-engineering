# Upstream patch / PR candidates from the GB300 campaign (B36–B58)

Findings from the 2026-06-10/11 GB300 (sm_103) optimization campaign that are
reportable or patchable OUTSIDE this repo, ranked by expected impact and
readiness. Every item carries banked evidence (runbook entry + evidence doc +
pod artifacts). In-repo fixes are already landed on main and are not listed.

## Tier 1 — ready to file (evidence complete, reproducer exists)

### 1. PyTorch Inductor: no `bmm + pointwise-bias` decomposition choice for broadcast-bias `baddbmm`
- **Status (2026-06-11, Front P): reproducer verified / draft ready.**
  `code/upstream/inductor-baddbmm-decomposition/{repro.py,ISSUE_DRAFT.md,README.md}`.
  repro.py ran on the pod (GPU 3, flock, foreign-proc clean): exit 0, all
  assertions passed — eager 138.0 vs 60.3 us (== banked), template choice
  reproduced (`triton_tem_fused_baddbmm` vs `nvjet_*_bz` extern), decomposition
  1.451x with the GELU consumer (banked pair: 172.8 -> 115.9 us). Standalone
  (no consumer) the decomposition is honestly 0.90-1.0x — the filing frames it
  as an AUTOTUNE-LEVEL choice, not an unconditional rewrite.
- **Repo:** pytorch/pytorch (Inductor decompositions / GEMM autotune)
- **Finding (B56 evidence — CORRECTED from the B54-era framing):** the
  0.140 ms extern-baddbmm autotune number is HONEST (`at::baddbmm_out`
  materializes the broadcast bias via a 74.6 us `direct_copy` before a beta=1
  GEMM); the Triton template correctly wins for the op as written. The bug to
  file is the MISSING `bmm + pointwise-bias` decomposition candidate: 1.31x
  end-to-end at MoE shapes (32x640x1024x2048 BF16), bias add fuses for free.
- **To file:** see README — paste ISSUE_DRAFT.md into a pytorch/pytorch issue
  (`module: inductor`, `topic: performance`), attach repro.py inline; optional
  follow-up PR branch `inductor-baddbmm-broadcast-bias-decomp`.

### 2. CUTLASS: tcgen05 tutorial/example uses the pathological narrow TMEM_LOAD atom
- **Status (2026-06-11, Front P): reproducer verified / draft ready.**
  `code/upstream/cutlass-tmem-load-atom/{tmem_load_atom_repro.cu,sass_evidence.txt,PR_DRAFT.md,README.md}`.
  Minimal standalone .cu (TMEM alloc + t2r + store only, no torch) compiled on
  the pod against BOTH CUTLASS 4.2.0 and main (4.5.2) headers: SASS shows the
  per-load LEPC + CALL.ABS.NOINC + WARPSYNC wrap — 256 vs 8 helper calls
  (1x vs 32x atom); runtime 10.91 vs 5.86 us/launch on GPU 3. All five
  Blackwell cute tutorials still demonstrate the 1x atom on current main.
- **Repo:** NVIDIA/cutlass (examples + cute tutorials for sm100/sm103)
- **Finding (B55, docs/gb300-capstone-f-decomposition.md):**
  `SM100_TMEM_LOAD_32dp32b1x` lowers (ptxas, CUDA 13.2) to a per-load
  LEPC + CALL.ABS.NOINC + WARPSYNC helper sequence — 256 loads/thread = 512
  helper calls dominating an entire kernel's fixed cost (7.7 us of a 24 us
  kernel). Swapping to `32dp32b32x` is a one-line 1.49x kernel win; 128x
  REGRESSES (serialized writeback). Tutorials that demonstrate the 1x atom
  teach a 1.5x perf bug.
- **To file:** see README — fork NVIDIA/cutlass, branch
  `docs/blackwell-tutorial-tmem-load-atom-width`, paste PR_DRAFT.md; optional
  separate ptxas question to the NVIDIA bug tracker (text at the bottom of
  PR_DRAFT.md).

### 3. CUTLASS: 2x1SM MMA atom operand-split semantics undocumented/stale prints
- **Status (2026-06-11, Front P): reproducer verified / draft ready.**
  `code/upstream/cutlass-2x1sm-docs/{snippet.cu,PR_DRAFT.md,README.md}`.
  Host-only print check (no GPU) compiled+run on the pod against BOTH CUTLASS
  4.2.0 and main (4.5.2): `partition_shape_B` actually returns
  `((_128,_16),_1,_4)` (N/2, 16 KiB B/stage — matching the banked B49 ncu
  footprints) where tutorial 04's annotations claim `((_256,_16),_1,_4)`.
  Deadlock claims cite banked B53/B57 evidence only (not re-run).
- **Repo:** NVIDIA/cutlass
- **Finding (B49/B57, docs/gb300-gemm-occupancy-rewrite.md):** the
  `SM100_MMA_F16BF16_2x1SM_SS` atom SPLITS B N/2-per-CTA across the pair
  (proven by smem footprints + mbarrier tx-byte balance); the tutorial's
  printed layouts suggest otherwise (stale prints). Also: `tma_partition`'s
  multicast slice is an OFFSET VIEW — expect-tx bytes must be the full
  delivered slice with NO participant multiplier; getting this wrong is a
  deterministic mbarrier deadlock (two independent fronts hit it: B53, B57).
- **To file:** see README — fork NVIDIA/cutlass, branch
  `docs/blackwell-tutorial04-2sm-b-split-expect-tx`, make the mechanical
  annotation edits from the PR_DRAFT table, paste PR_DRAFT.md, attach
  snippet.cu.

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
(GB300 sm_103, driver 580.159.03, CUDA 13.2, torch 2.12.0a0 NGC 26.05,
CUTLASS 4.2.0 — the version previously listed here as 4.3.0 was wrong; the
lab builds against `third_party/cutlass` whose version.h says 4.2.0, and the
CUTLASS findings were additionally reproduced against main/4.5.2), and a
link-free description (internal pod paths stripped). File from a personal
fork; CC the relevant CODEOWNERS.

Tier-1 filing packages live under `code/upstream/` (Front P, 2026-06-11):
every file is marked "DRAFT — internal review pending"; nothing has been
posted externally. Pod verification artifacts: `/tmp/frontP/` (repro
stdout/stderr, SASS dumps, ALL_DONE).
