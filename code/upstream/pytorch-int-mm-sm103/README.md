# DRAFT — internal review pending

## Upstream candidate: `torch._int_mm` sm_103 dispatch perf bug

Filing package for Tier-2 item 5 of `code/docs/gb300-upstream-patches.md`
(B20 evidence, `code/docs/gb300-runbook.md` B20 entry; lab
`code/ch13/optimized_quantization.py`).

**The claim being filed** (perf bug): on sm_103, `torch._int_mm`
dispatches Ampere-era CUTLASS 2.x kernels (`cutlass_80_tensorop_i16832gemm`
for a column-major rhs; the `cutlass_80_wmma_..._forwardCompat` WMMA
kernel for a row-major rhs) instead of any Blackwell INT8 path, landing
6.4x (col) / 13.4x (row) slower than the fp32-tf32 matmul and 11.8x /
24.5x slower than fp16 at M=8192, K=N=4096. The dispatched-kernel names
are the evidence the filing leads with.

## Files

| file | what |
|---|---|
| `repro.py` | standalone reproducer; times all variants + captures dispatched kernel names; asserts the gap |
| `ISSUE_DRAFT.md` | draft pytorch/pytorch perf-bug text |

## Verification record (2026-06-11, GB300 pod, GPU 3, flock lease, foreign-proc check clean)

`repro.py` exit 0:

```
fp32 matmul (tf32 enabled)  :  272.3 us  cutlass3x_sm100_tensorop_s256x256x8tf32gemm_..._2sm_bias_f32_relu
fp16 matmul                 :  148.7 us  nvjet_sm103_hsh_128x256_64x6_2x1_2cta_v_bz_NNT
bf16 matmul                 :  139.1 us
_int_mm, rhs column-major   : 1751.1 us  cutlass_80_tensorop_i16832gemm_s8_128x64_128x3_tn_align16
_int_mm, rhs row-major      : 3645.9 us  cutlass_80_wmma_tensorop_i161616gemm_s8_forwardCompat_128x128_32x2_nn_align4
ratios: 6.4x / 13.4x vs tf32; 11.8x / 24.5x vs fp16; row-vs-col 2.08x
```

Numbers provenance:

* re-measured this session (primary): everything in the table above. The
  row-major 3645.9 us / 13.4x-vs-tf32 reproduces the banked B20 numbers
  (3.65 ms, 13.4x) exactly; the kernel names are NEW evidence (B20 had no
  profiler capture).
* banked (cited in the draft as "we measured an INT8-quantized MLP at
  0.17x"): the ch13 lab end-to-end regression (B20 runbook entry, 12/12
  reps; quant+dequant overhead ~0.5 ms on top of the 3.65 ms GEMM).
* The col-vs-row 2.08x penalty cross-references the already-filed
  pytorch/pytorch#165230 (open; row-major-rhs layout penalty) — read this
  session; our filing is the orthogonal arch-dispatch bug and links it.

Pod evidence: `/tmp/frontP2/int_mm.{cmd,stdout,stderr,exit}`.

## What the user must do to file

1. Re-read `ISSUE_DRAFT.md` against the citation bar (all table numbers
   are verified primary this session).
2. File at github.com/pytorch/pytorch -> New issue -> bug template.
   Paste ISSUE_DRAFT.md body; attach repro.py inline (self-contained).
3. Labels to suggest in the body: `module: linear algebra`,
   `topic: performance`, `oncall: quantization` (matches #165230's triage).
4. Link #165230 in the body (already done in the draft) so triage sees
   the layout penalty is known-tracked and this is the dispatch bug.
