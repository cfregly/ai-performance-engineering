<!-- DRAFT — internal review pending. Do NOT post externally without review. -->
<!-- Target: github.com/pytorch/pytorch issues (module: inductor) -->

# Issue title

`[inductor] baddbmm with broadcast bias: autotune never considers the bmm + pointwise-bias decomposition (1.3-1.45x left on the table at MoE expert-GEMM shapes)`

# Body

### 🐛 Describe the bug

For `baddbmm` with a **broadcast** bias (`bias[B, 1, N]`, stride `[N, 0, 1]`
after broadcasting), Inductor's max-autotune chooses between (a) Triton
templates and (b) the extern ATen kernel — but never considers the
decomposition `bmm(x, w) + bias`, which is faster than both of its existing
choices whenever the bias add can fuse into a neighboring pointwise op
(which in MLP/MoE blocks it essentially always can — there is an activation
right after the GEMM).

Why both existing choices lose at this shape
(`[32, 640, 1024] x [32, 1024, 2048] + bias[32, 1, 2048]`, bf16, sm_103):

* **Extern `baddbmm` is honestly slow** (this is *not* an autotune
  measurement bug — we verified the benchmarked number is the true per-call
  cost): cuBLAS computes `C = alpha*A@B + beta*C`, so `at::baddbmm_out`
  must first materialize the broadcast bias into the output buffer. With a
  stride-0 bias dim that is a full `[B, M, N]`-sized strided elementwise
  copy (~75 us here), followed by a beta=1 GEMM that re-reads those bytes
  (~67 us): ~138 us per call, every call.
* **The Triton template wins the autotune for the op as written** (~104-106
  us; it reads the `[B, 1, N]` bias in its epilogue for free), so this is
  what Inductor ships.
* **The missing candidate is faster than both:** `bmm` routes extern to a
  beta=0 kernel (~60-62 us — C is never read or pre-filled), and the rank-1
  bias add fuses into the adjacent pointwise. Measured with the GELU
  consumer present (the realistic MLP/MoE shape of the graph):
  `gelu(baddbmm(...))` = 168.1 us/call vs `gelu(bmm(...) + bias)` =
  115.8 us/call — **1.45x** for the GEMM+activation pair. In the full MoE
  layer this came out to **1.31x end-to-end** (0.407 ms -> 0.311 ms median,
  verified numerics).

Kernel-level view (torch profiler, per call, GB300):

| graph as written | decomposed |
|---|---|
| `triton_tem_fused_baddbmm_0` 105.9 us | `nvjet_..._bz_NNT` (extern beta=0 bmm) 61.7 us |
| `triton_poi_fused_gelu_1` 69.3 us | `triton_poi_fused_add_gelu_0` 56.0 us |

Standalone (no consumer to fuse the bias add into), the decomposition is
*not* a win (102.2 vs 104.7 us here) — which is exactly why this wants to
be an autotune-level choice (benchmarked in graph context, where the
scheduler knows whether the epilogue can fuse) or a decomposition with a
consumer-aware condition, rather than an unconditional rewrite.

Possibly related: the ATen-level cost itself (broadcast-bias `baddbmm`
paying a full output-sized materialization copy: 138.0 us vs 60.3 us for
the bare `bmm`, eager) could separately be improved in cuBLAS dispatch
(e.g. via a fused-epilogue path), but the Inductor decomposition choice is
the more general fix and is what this issue asks for.

### Repro

Standalone script attached (`repro.py`, no external deps). It compiles both
formulations with `mode="max-autotune-no-cudagraphs"`, prints per-call
timings + the kernels each compiled graph runs, and asserts the template
choice and the gap. Output on a GB300 (sm_103):

```
eager baddbmm (broadcast bias [B,1,N]):    138.0 us/call
eager bmm     (no bias)               :     60.3 us/call
compiled baddbmm        (bare)        :    102.2 us/call
compiled bmm + bias     (bare)        :    104.7 us/call
compiled gelu(baddbmm)  (as written)  :    168.1 us/call
compiled gelu(bmm+bias) (decomposed)  :    115.8 us/call
decomposition speedup with consumer   :    1.451x
```

### Suggested fix

Register a decomposition (or an additional autotune candidate) for
`baddbmm` when the bias input has a stride-0 broadcast dim:
`bmm(batch1, batch2) + input`. The pointwise add participates in normal
Inductor fusion, so wherever the GEMM output feeds a pointwise op the bias
is free; the autotuner's existing benchmarking would keep the current
behavior in the (rarer) standalone case if the candidate loses there.

### Versions

* torch 2.12.0a0 (NGC 26.05 container build), CUDA 13.2, driver 580.159.03
* GPU: NVIDIA GB300 (sm_103); shapes are MoE expert-GEMM class
  (`E=32` experts, `tokens*capacity=640`, `hidden=1024`, `ffn=2048`, bf16)
* Reproduces with caches disabled (`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`)
