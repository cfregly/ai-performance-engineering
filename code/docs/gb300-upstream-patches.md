# Upstream patch / PR candidates from the GB300 campaign (B36–B58)

**FILED 2026-06-11 (account `cfregly`, all five live):**
| item | link |
|---|---|
| Inductor baddbmm decomposition | https://github.com/pytorch/pytorch/issues/187093 + **code PR https://github.com/pytorch/pytorch/pull/187106** (Fixes #187093; pod-validated, test incl.) |
| `_int_mm` sm_103 Ampere dispatch | https://github.com/pytorch/pytorch/issues/187094 |
| `_scaled_grouped_mm` NVFP4 (+ sm_103 FP8 trap) | https://github.com/pytorch/pytorch/issues/187095 (+ comment on PR #174699) |
| CUTLASS TMEM_LOAD atom width (tutorials 01-05) | https://github.com/NVIDIA/cutlass/pull/3313 |
| CUTLASS tutorial-04 2SM B-split + expect-tx | https://github.com/NVIDIA/cutlass/pull/3314 |
| Triton tcgen05.wait.st | NOT FILED — already upstream (triton#8473); campaign aborts were self-inflicted (B69/B70) |


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

## Tier 2 — reportable findings (reproducers verified 2026-06-11, Front P2)

### 4. PyTorch: `torch._scaled_grouped_mm` refuses NVFP4
- **Status (2026-06-11, Front P2): reproducer verified / draft ready.**
  `code/upstream/pytorch-scaled-grouped-mm-nvfp4/{repro.py,ISSUE_DRAFT.md,README.md}`.
  repro.py ran on the pod (GPU 3, flock, foreign-proc clean): exit 0. Exact
  refusal messages captured (`ValueError: ... For MXFP8, scales must both be
  float8_e8m0fnu tensors.` / `Expected mat_a to be Float8_e4m3 matrix got
  Float4_e2m1fn_x2`); per-group NVFP4 `_scaled_mm` works (nvjet_sm103
  Avec16UE4M3 kernel, profiler-named); 30-launch fallback re-measured
  (1068 us/call GPU span, 35.6 us/GEMM). BONUS finding in the same repro:
  FP8 rowwise `_scaled_grouped_mm` is ALSO broken on sm_103 — the dispatched
  CUTLASS kernel traps ("Arch conditional MMA instruction used without
  targeting appropriate compute capability") and poisons the CUDA context,
  so NO `_scaled_grouped_mm` path works on sm_103 at all in torch 2.12.0a0.
- **Repo:** pytorch/pytorch
- **Finding (B28):** the grouped scaled-mm API accepts FP8/MXFP8 only; NVFP4
  workloads are forced to 30 separate `_scaled_mm` launches, which a custom
  fused kernel beats 3.1x wall (banked). Feature request: NVFP4 grouped
  support — capability citation verified: CUTLASS
  `examples/90_sm103_fp4_ultra_grouped_gemm` (e2m1 + ue4m3-SF, sm_103).
- **Upstream check (2026-06-11):** NVFP4 grouped is IN FLIGHT upstream —
  pytorch/pytorch#174699 (open, CuTeDSL block-scaled grouped MM with nvfp4,
  hooks `F.scaled_grouped_mm`), plus #156238 (SM100+ grouped ask) and
  #162209 (MXFP8 precedent, merged).
- **To file:** see README — consider commenting on #174699/#156238 instead
  of a new issue; if filing, pytorch/pytorch "🚀 Feature" template, paste
  ISSUE_DRAFT.md, attach repro.py; decide whether to split the sm_103
  FP8-grouped arch-trap into its own bug.

### 5. PyTorch: `torch._int_mm` 13.4x slower than tf32 matmul on sm_103
- **Status (2026-06-11, Front P2): reproducer verified / draft ready.**
  `code/upstream/pytorch-int-mm-sm103/{repro.py,ISSUE_DRAFT.md,README.md}`.
  repro.py ran on the pod (GPU 3, flock, foreign-proc clean): exit 0,
  banked numbers reproduced exactly (row-major rhs 3645.9 us = 13.4x vs
  tf32 272.3 us; 24.5x vs fp16) and the dispatched kernels captured — the
  NEW evidence: `_int_mm` lands on Ampere-era `cutlass_80_tensorop_i16832gemm`
  (col-major rhs, 1751 us = 6.4x) / `cutlass_80_wmma_..._forwardCompat`
  (row-major rhs) while fp32-tf32 gets `cutlass3x_sm100_*_2sm` and fp16
  gets `nvjet_sm103_*`. Layout-controlled: BOTH rhs layouts are several
  times slower than tf32 (the row/col 2.08x gap is the separately-tracked
  pytorch/pytorch#165230; ours is the arch-dispatch bug).
- **Repo:** pytorch/pytorch
- **Finding (B20):** INT8 `_int_mm` runs 13.4x slower than the tf32 path on
  GB300 — sm_103 INT8 dispatch hits sm_80 forward-compat kernels. Perf bug
  with shapes + timings + dispatched-kernel names.
- **To file:** see README — pytorch/pytorch bug template, paste
  ISSUE_DRAFT.md, attach repro.py; link #165230; suggest labels
  `module: linear algebra`, `topic: performance`, `oncall: quantization`.

### 6. Triton: sm_103 max-autotune emits unselectable `tcgen05.wait.st`
- **Status (2026-06-11, Front P2): DO NOT FILE — already filed upstream AND
  root-caused to an in-repo patch.**
  `code/upstream/triton-tcgen05-wait-st/{repro.py,STATUS.md,README.md}`
  (no ISSUE_DRAFT, deliberately). Upstream: the exact error string is
  triton-lang/triton#8473 (open; the genuine bug was the Triton 3.5
  release cut between breaking PR #7725 and fixing LLVM-bump PR #8299) and
  #8481 (closed dup). On THIS stack the 4-probe repro (pod, GPU 3, exit 0)
  shows vanilla Triton 3.7.0 is CLEAN on sm_103 (plain tl.dot JIT, dead
  num_warps:constexpr param, max-autotune attn+MLP compile all clean); the
  abort reproduces ONLY when the target arch is mis-set to `sm_103` without
  the `a` suffix (probe D: rc=134, exact error string) — which is precisely
  what `core/benchmark/triton_compat.py:100-109` (`ensure_triton_compat`,
  the GB10 sm_121 workaround) does to sm_103a. This resolves the B19
  "exact side effect not yet isolated" question; the campaign's 3.7 aborts
  were self-inflicted.
- **Repo:** none (in-repo fix instead): preserve the `a` suffix for all
  `sm_1xxa` in triton_compat.py, clamp only sm_121->sm_120; then re-test
  llama max-autotune with `ENABLE_TRITON_PATCH=0` and consider retiring the
  sm_103 max-autotune->default downgrade in `get_optimal_compile_mode`.
- **Finding (toolchain section + B17/B18/B19-era, REVISED):** the abort
  signature is real but its 3.7 occurrences trace to the repo's arch
  de-suffixing; the upstream 3.5-release breakage is fixed on main.


## Tier 2b — new candidates from B76 (attention serving-shapes survey, 2026-06-11)

### 7. flashinfer: `cudnn_batch_prefill_with_kv_cache` broken on sm_103 (0.6.3)
- **Repo:** flashinfer-ai/flashinfer
- **Finding (B76):** stride BAD_PARAM -> "No valid execution plans built";
  `backend="fa3"` ships sm90-only cubins; trtllm-gen/cutlass invalid on
  sm_103. Working cuDNN prefill would be worth up to ~4.3x on long-context
  prefill vs the FA2-class default (cuDNN SDPA measured at 98-106% real-SoL
  at S=8-32k on GB300). Reproducer material in /tmp/frontAT2/ (attempt
  failure evidence captured).

### 8. flashinfer: `cudnn_batch_decode_with_kv_cache` seq-len API mismatch
- **Repo:** flashinfer-ai/flashinfer
- **Finding (B76):** docstring says 1-D seq-len tensors; the cuDNN frontend
  rejects them — requires 4-D (B,1,1,1). With the workaround the path runs
  92.9% HBM-SoL, bit-identical, 1.124x over flashinfer's default decode.
  Docs fix or input normalization; small PR.

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

Tier-2 filing packages live under `code/upstream/` as well (Front P2,
2026-06-11): same DRAFT marking, nothing posted externally. Pod
verification artifacts: `/tmp/frontP2/` (per-item
.cmd/.stdout/.stderr/.exit tuples, the childD mechanism probe, ALL_DONE).
Item 6 ships a STATUS.md instead of an ISSUE_DRAFT (already filed
upstream + root-caused in-repo).
