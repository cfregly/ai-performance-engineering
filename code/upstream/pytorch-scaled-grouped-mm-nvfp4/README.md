# DRAFT — internal review pending

## Upstream candidate: NVFP4 support for `torch._scaled_grouped_mm`

Filing package for Tier-2 item 4 of `code/docs/gb300-upstream-patches.md`
(B28 evidence, `code/labs/nvfp4_group_gemm/NCU-DIAGNOSIS.md`).

**The claim being filed** (feature request): `torch._scaled_grouped_mm`
accepts FP8/MXFP8 only and refuses NVFP4 (`float4_e2m1fn_x2`), although
the non-grouped `_scaled_mm` runs NVFP4 on the same device (cuBLASLt
`nvjet_sm103` kernel) and CUTLASS ships grouped block-scaled NVFP4 for
sm_100/sm_103 (`examples/90_sm103_fp4_ultra_grouped_gemm`, verified
e2m1 + ue4m3-SF on current main). Grouped NVFP4 workloads are forced to
one `_scaled_mm` launch per (input, group); the campaign's fused kernel
beat that loop 3.1x wall (banked B28).

**Bonus finding (this session, repro part 4):** the SUPPORTED FP8 rowwise
grouped call is ALSO broken on sm_103 — the dispatched CUTLASS kernel
traps with "Arch conditional MMA instruction used without targeting
appropriate compute capability" and poisons the CUDA context. So
`_scaled_grouped_mm` has no working path at all on sm_103 in torch
2.12.0a0. Included in the draft as "additional context" with an offer to
split it out.

## Files

| file | what |
|---|---|
| `repro.py` | standalone reproducer, no repo deps; refusal + kernel name + fallback timing + FP8 trap (child proc) |
| `ISSUE_DRAFT.md` | draft pytorch/pytorch feature-request text |

## Verification record (2026-06-11, GB300 pod, GPU 3, flock lease, foreign-proc check clean)

`repro.py` exit 0:

```
[1] per-group NVFP4 _scaled_mm: OK — kernel nvjet_sm103_oohsh_128x128_256x7_2x2_2cta_v_bz_Avec16UE4M3_Bvec16UE4M3_TNT
[2] NVFP4 refusals (exact messages):
    e4m3 block scales -> ValueError: For FP8 tensorwise and rowwise, both scales must both be
                         float32 tensors. For MXFP8, scales must both be float8_e8m0fnu tensors.
    fp32 scales       -> ValueError: Expected mat_a to be Float8_e4m3 matrix got Float4_e2m1fn_x2
[3] 30-launch fallback: 1068.1 us/call GPU span (35.60 us/GEMM), host wall 1069.6 us, 180.9 TFLOP/s
[4] FP8 rowwise grouped (child): FAILED rc=1, arch-conditional trap=yes
```

Numbers provenance:

* re-measured this session (primary): the refusal messages, the nvjet
  kernel name, the 30-launch timing (1068 us/call — slightly higher than
  the banked 701 us; this repro uses 30 distinct operand sets where the
  banked harness reused one, both launch-bound, same order), the FP8 trap.
* banked (cited, NOT re-measured): custom fused kernel 0.2235 ms vs
  cuBLAS 30x `_scaled_mm` 0.701 ms = 3.1x wall / 1.14x pure-GPU
  (`labs/nvfp4_group_gemm/NCU-DIAGNOSIS.md`, B28 runbook entry).

Pod evidence: `/tmp/frontP2/grouped_nvfp4.{cmd,stdout,stderr,exit}`.

Upstream cross-references checked this session (gh, 2026-06-11):

* pytorch/pytorch#156238 (open): "Upgrade torch._scaled_grouped_mm to SM100+"
* pytorch/pytorch#162209 (merged 2025-09): MXFP8 grouped support — dtype-extension precedent
* pytorch/pytorch#174699 (OPEN, updated 2026-06-11): CuTeDSL block-scaled
  grouped MM with an `nvfp4` format, registered into
  `torch.nn.functional.scaled_grouped_mm` — NVFP4 grouped is in flight
  upstream; the draft asks for sm_103a coverage + documents the cost
* NVIDIA/cutlass `examples/90_sm103_fp4_ultra_grouped_gemm` (main):
  `float_e2m1_t` operands + `float_ue4m3_t` SFs — the capability citation

## What the user must do to file

1. Re-read `ISSUE_DRAFT.md` against the citation bar (every number above
   is verified primary this session except the labeled banked 3.1x pair).
2. Consider FIRST commenting on #174699/#156238 instead of opening a new
   issue — the feature is visibly in flight; a new issue may be closed as
   duplicate. If filing: github.com/pytorch/pytorch -> New issue ->
   "🚀 Feature" template, paste ISSUE_DRAFT.md, attach repro.py inline.
3. Decide whether to split the sm_103 FP8-grouped arch-trap into its own
   bug report (the draft offers; maintainers usually prefer 1 bug = 1 issue).
4. Labels to suggest: `module: linear algebra`, `topic: performance`,
   `module: float8`.
