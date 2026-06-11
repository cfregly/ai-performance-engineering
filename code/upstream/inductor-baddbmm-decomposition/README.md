# DRAFT — internal review pending

## Upstream candidate: Inductor `baddbmm` broadcast-bias decomposition

Filing package for Tier-1 item 1 of `code/docs/gb300-upstream-patches.md`
(B56 evidence, `code/docs/gb300-moe-cuda-gemm1-aten.md`).

**The claim being filed** (corrected by B56 — the earlier "autotune
mis-benchmark" theory is falsified and must NOT be filed): Inductor's
autotune numbers are honest; what is missing is the `bmm + pointwise-bias`
decomposition as a choice for broadcast-bias `baddbmm`. With the bias add
fused into the adjacent activation it is 1.45x for the GEMM+activation pair
(repro, verified) and was 1.31x end-to-end on the MoE layer (banked B56
A/B, 12/12 verification-passed reps).

## Files

| file | what |
|---|---|
| `repro.py` | standalone reproducer, no repo deps; asserts template choice + the gap |
| `ISSUE_DRAFT.md` | draft pytorch/pytorch issue text |

## Verification record (2026-06-11, GB300 pod, GPU 3, flock lease, foreign-proc check clean)

`repro.py` exit 0, all assertions passed:

```
eager baddbmm (broadcast bias [B,1,N]):    138.0 us/call   (banked B56: 138.0)
eager bmm     (no bias)               :     60.3 us/call   (banked B56:  60.3)
compiled baddbmm        (bare)        :    102.2 us/call   (banked B56 template: 104.1)
compiled bmm + bias     (bare)        :    104.7 us/call   (bias-add tune lottery 41-76us, B56)
compiled gelu(baddbmm)  (as written)  :    168.1 us/call   (banked B56 pair: 172.8)
compiled gelu(bmm+bias) (decomposed)  :    115.8 us/call   (banked B56 pair: 115.9)
decomposition speedup with consumer   :    1.451x
kernels: triton_tem_fused_baddbmm_0 + triton_poi_fused_gelu_1
     vs  nvjet_..._bz_NNT + triton_poi_fused_add_gelu_0    (matches banked r09 table)
```

Pod evidence: `/tmp/frontP/repro_inductor.{stdout,stderr}` (+ the bare-op
first run in the session transcript showing the standalone 0.90x honest
negative that motivated including the consumer op).

## What the user must do to file

1. Re-read `ISSUE_DRAFT.md` against the citation bar (every number above is
   verified primary this session; the 1.31x end-to-end number cites banked
   B56 only).
2. File at github.com/pytorch/pytorch -> New issue -> "torch.compile"
   template. Paste `ISSUE_DRAFT.md` body; attach `repro.py` inline (it is
   self-contained).
3. Labels to suggest in the body: `module: inductor`, `topic: performance`.
4. Optional follow-up PR (separate decision): a decomposition in
   `torch/_inductor/decomposition.py` gated on
   `input.stride(-2) == 0` for the bias operand — suggested branch name
   `inductor-baddbmm-broadcast-bias-decomp`. The issue stands alone without
   the PR.
