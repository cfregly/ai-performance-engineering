# moe_cuda router_vectorized: GEMM1 back to the ATEN extern kernel (Front M3)

## Verdict

**WIN — 1.3055x median (0.40653 -> 0.31139 ms), verification passed on all 12
A/B reps, bit-identical output on the lab input — but via a CORRECTED
mechanism: the B54 "autotune mis-benchmark" hypothesis is FALSIFIED.**
Inductor's 0.1402 ms benchmark of the extern baddbmm choice is HONEST: with the
broadcast bias (stride `[2048, 0, 1]`), `at::baddbmm_out` materializes the bias
into the output buffer first (74.6us broadcast `direct_copy` kernel) and then
runs nvjet with beta=1 (66.9us) — ~138us real, every call. B45's "eager nvjet
does it in 66us" was kernel-only under-counting (it missed the copy). The
backend pin (`max_autotune_gemm_backends="ATEN"`) therefore measures WORSE
(369.8 vs 313.4 GPU us/call). The landed fix removes the broadcast bias from
the GEMM op instead: `baddbmm(b, x, w)` -> `bmm(x, w) + b` for BOTH expert
GEMMs; the honest autotuner then routes GEMM1 to the extern beta-zero nvjet
kernel (59.2us vs the 104us Triton template) and Inductor fuses the bias adds
into adjacent pointwise kernels for free. GEMM2's hidden 40us bias-copy (the
`direct_copy` that B51's table mislabeled "gather-back") disappears too.

- Date: 2026-06-11, pod `<gb300-pod>` (GB300 NVL72, Dell XE9712), GPU 3
  only, flock lease (`flock -w 900 /tmp/gpu3.lock` around every GPU command).
- Torch `2.12.0a0+5aff3928d8.nv26.05`. Incumbent arm: B51 compile-in-graph
  (banked 0.40292 ms; re-repro before any edit: **0.40288 ms**, verification
  passed, graph_breaks=0, 16 kernels/replay — `/tmp/frontM3/r01_b51_repro.*`).
- Evidence: `/tmp/frontM3/` on the pod (`.cmd/.stdout/.stderr/.exit` per run,
  `ab/` for the interleaved A/B, `ALL_DONE` marker).

## The bankable diagnosis: the autotuner was RIGHT, the op was wrong

Isolated reproduction (`/tmp/frontM3/r04_artifact.stdout`, GEMM1 shapes
`[32,640,1024] x [32,1024,2048] + bias[32,1,2048]`, bf16, CUDA-event timed,
200 iters):

| variant | us/call | kernel split | Inductor benchmarker |
|---|---|---|---|
| `baddbmm` broadcast bias (stride `[2048,0,1]`), out= | 138.0 | 74.6 `direct_copy` (broadcast bias -> out) + 66.9 `nvjet_..._v_badd_NNT` | 0.1392 ms |
| `baddbmm` materialized bias (contiguous) | 90.0 | 25.3 `Memcpy DtoD` + 66.1 nvjet | 0.0962 ms |
| `baddbmm` broadcast bias, allocating | 137.8 | (same as out=) | 0.1393 ms |
| pure `bmm` | 60.3 | one nvjet | 0.0651 ms |

The Inductor benchmarker reproduces its own AUTOTUNE table value (0.1392 vs
the 0.1402 ms logged choice — `/tmp/frontM3/r02_probe_control.stderr`):

```
AUTOTUNE baddbmm(32x640x2048, 32x640x1024, 32x1024x2048)
strides: [2048, 0, 1], [655360, 1024, 1], [2097152, 2048, 1]
  triton_bmm_35 0.1080 ms 100.0%  (BLOCK 128/128/64, stages=3, warps=4)
  ...
  baddbmm 0.1402 ms 77.0%
```

Mechanism: cuBLAS computes `C = alpha*A@B + beta*C`, so `at::baddbmm_out` must
pre-fill the output with the bias. With a contiguous bias that is a 25us DtoD
memcpy; with the broadcast view it is a 74.6us strided elementwise copy
(83.9MB written at ~1.1 TB/s), then the beta=1 GEMM re-reads those 83.9MB.
None of this is a benchmark artifact — it is the true per-call dataflow cost
of extern baddbmm with a broadcast bias. The 104us Triton template, which
reads the `[32,2048]` bias in its epilogue for free, was the correct autotune
winner FOR THE OP AS WRITTEN. Upstream-reportable nuance: Inductor never
considers the `bmm + pointwise-bias` decomposition (0.0651 ms extern +
fusable add) as a choice for `baddbmm`, leaving ~40% on the table here.

Corollary fix to the B51/B54 record: the 40.2us `direct_copy` in those kernel
tables is GEMM2's broadcast-bias materialization (`b2` -> out), not the
gather-back (which is fused into `triton_poi` kernels). It vanishes when GEMM2
is rewritten to `bmm + add` (probe r06) and doubles when GEMM1 also goes
extern-baddbmm (probe r03: `direct_copy x2 = 114.3us`).

## Approaches measured (probes, caches disabled, GPU 3, 200-replay CUDA events)

| probe | config | GPU us/call | kernels | GEMM1 | GEMM2 | GELU(+bias) |
|---|---|---|---|---|---|---|
| r02 control (=B51 config, fresh tune) | default backends | 313.4 | 16 | `triton_tem` 104.1 | nvjet badd 58.5 + 40.3 bias-copy | 41.0 |
| r03 backend pin | `max_autotune_gemm_backends="ATEN"` | **369.8 (WORSE)** | 17 | nvjet badd 66.3 + 74.6 bias-copy | nvjet badd 58.2 + ~39.7 bias-copy | 68.9 |
| r05 bmm rewrite, GEMM1 only | default backends | 303.2 | 16 | **nvjet bz 59.0** | nvjet badd 58.4 + 40.1 bias-copy | 75.7 (add+gelu) |
| r06 bmm rewrite, both GEMMs | default backends | **242.0** | 15 | **nvjet bz 59.2** | **nvjet bz 56.4** | 55.6 (add+gelu) |

(The add+gelu pointwise tunes between 41 and 76us across otherwise-identical
compiles — Inductor pointwise tune lottery; the B51 incumbent's disk cache
carries a 69us config. r05 vs r06 GELU delta is that lottery, not the GEMM2
change.) All probes bit-identical vs the eager baddbmm forward
(`/tmp/frontM3/r0{2,3,5,6}_probe_*.stdout`).

Chosen implementation: **r06's dataflow, gated by `AISP_MOE_CUDA_GEMM1_ATEN`**
in `GroupedTopKMoE.forward` (`use_bmm_bias_fuse`), no Inductor config override
(the changed graph changes the cache key, so arms never share artifacts).
Measured as opt-in for the A/B below, then flipped to default-ON after the
gate passed (see "Default flipped").
Per-region compile scoping (approach 2) was unnecessary: the backend pin was
refuted outright, and the rewrite needs no tuner override at all.

## A/B vs B51 incumbent (harness, interleaved, 6 reps/arm, GPU 3)

Driver `/tmp/frontM3/ab_driver.sh`; parsed `/tmp/frontM3/ab/ab_results_parsed.json`.
The A/B ran pre-flip (default still OFF): b51 arm = bare env, aten arm =
`AISP_MOE_CUDA_GEMM1_ATEN=1` (exact commands in `ab/rep*.cmd`).
Reps alternate b51/aten; per-rep `nvidia-smi --query-compute-apps -i 3`
snapshots before AND after every rep were all EMPTY (no foreign process ever
co-resident); every rep exit 0.

| arm | median ms | min | max | n | verified | kernels/replay |
|---|---|---|---|---|---|---|
| B51 incumbent (`AISP_MOE_CUDA_GEMM1_ATEN=0` semantics) | 0.40653 | 0.40511 | 0.40839 | 6 | 6/6 | 16 |
| GEMM1_ATEN (bmm + fused bias) | 0.31139 | 0.31077 | 0.31248 | 6 | 6/6 | 15 |

**speedup_median = 1.3055x** (gate >=1.05x PASSED; distributions
non-overlapping — every aten rep beat every b51 rep by >=92us).

| rep | b51 ms | aten ms |
|---|---|---|
| 1 | 0.40701 | 0.31248 |
| 2 | 0.40605 | 0.31083 |
| 3 | 0.40839 | 0.31097 |
| 4 | 0.40749 | 0.31182 |
| 5 | 0.40511 | 0.31224 |
| 6 | 0.40592 | 0.31077 |

All 12 reps: harness verification passed (rtol 0.1 / atol 1.0, untouched),
`manual_graph_active=1`, `dynamo_graph_breaks=0`, `dynamo_unique_graphs=1`;
warm compile 2.4-2.9 s/run (cold aten compile 14.3 s once, r08).

## Kernel-level proof on the exact harness path (warm disk cache, r09)

`/tmp/frontM3/r09_replay_{b51,aten}.stdout` — the lab's own `setup()` via
`get_benchmark()`, 5-replay torch profiler on the captured graph:

| kernel | B51 arm (us/replay) | aten arm (us/replay) |
|---|---|---|
| `triton_tem_fused_baddbmm_slice_unsqueeze_view_3` (GEMM1 template) | 103.6 | **— (gone)** |
| `triton_poi_fused_gelu_4` (standalone GELU, cached 69us config) | 69.2 | — |
| `nvjet_sm103_tst_256x128_64x5_4x1_2cta_v_bz_NNT` (GEMM1 extern, beta-zero) | — | **59.2** |
| `triton_poi_fused_add_gelu_unsqueeze_3` (bias1 + GELU) | — | 56.7 |
| `nvjet_sm103_tst_128x160_64x8_2x1_2cta_v_badd_NNT` (GEMM2, beta=1) | 58.1 | — |
| `nvjet_sm103_tst_128x160_64x8_2x1_2cta_v_bz_NNT` (GEMM2, beta-zero) | — | 56.4 |
| `direct_copy` elementwise (GEMM2 broadcast-bias materialization) | 40.4 | **— (eliminated)** |
| `indexFuncLargeIndex` (combine `index_add_`) | 25.3 | 25.4 |
| 11/10 small kernels (router/topk/dispatch/softmax) | 55.5 | 51.8 |
| **total / kernels per replay** | **352.0 / 16** | **249.4 / 15** |

GPU us/call over 200 replays: 342.3 -> 245.9 (delta 96.4us, matching the
harness median delta of 95.1us). GEMM1+GELU pair: 172.8 -> 115.9 (-56.9);
GEMM2 path: 98.5 -> 56.4 (-42.1). The aten arm contains ZERO `triton_tem`
kernels (loud assert armed) and ZERO `direct_copy`. Both nvjet kernels are the
beta-zero (`_bz_`) variants — C is never read: GEMM1 59.2us vs 66.3us for the
beta=1 nvjet (r03, -10.7%), GEMM2 56.4 vs 58.1 (-2.9%) — beyond what the B54
plan (extern baddbmm = beta=1 nvjet + bias copy) could have reached.

## Numerics class

**Bit-identical on the lab input** (probes r05/r06 and warm-cache r09:
`bitwise_equal_vs_eager_baddbmm: true`, `max_abs_diff: 0.0` vs the eager
baddbmm GroupedTopKMoE forward, same weights, seed-42 input). Observed
equality, not a guarantee: the rewrite changes rounding order (nvjet beta-zero
GEMM rounds to bf16, then the bias adds in a fp32-compute pointwise; baddbmm
adds bias in the fp32 accumulator) — on this input the bf16 rounding absorbs
it. Harness verification (vs eager per-token baseline, rtol 0.1 / atol 1.0 —
untouched) passed on every rep.

## Audit-rule proof metrics

As at B51/B54 (`manual_graph_active`, `compile_enabled`, `compile_seconds`,
`dynamo_graph_breaks=0`, `dynamo_unique_graphs=1`, `kernels_per_replay`), plus
NEW `router_vectorized.gemm1_aten_enabled`. Loud failures on the opt-in path:
`fullgraph=True`; setup raises if any `triton_tem*bmm*` template kernel
survives in the 5-replay profile (extern routing silently lost); mutual
exclusion with `AISP_MOE_CUDA_GELU_EPILOGUE=1` (its assert semantics assume a
GEMM1 template exists). `AISP_MOE_CUDA_COMPILE=0` kill switch untouched and
ignores this flag (eager bmm + broadcast add would cost ~150us extra; only
Inductor fuses the bias for free).

## Default flipped (clean win)

`AISP_MOE_CUDA_GEMM1_ATEN` now defaults ON; `=0` is the kill switch restoring
the exact B51 routing. Confirmation runs after the flip:

| run | env | time ms | verified | kernels | gemm1_aten |
|---|---|---|---|---|---|
| r10 | (none — new default) | 0.30710 | pass | 15 | 1 |
| r11 | `AISP_MOE_CUDA_GEMM1_ATEN=0` | 0.40256 | pass | 16 | 0 |
| r12 | `AISP_MOE_CUDA_GELU_EPILOGUE=1` | 0.39462 | pass | 15 | 0 (yielded; gelu_epi=1) |

r12 reproduces M2's fused-epilogue arm (B54 banked 0.39396 ms) with its own
loud assert passing — the default yields correctly.

An explicit `GELU_EPILOGUE=1` (M2's opt-in) auto-disables the unset default
(its loud assert needs the GEMM1 template); setting BOTH flags explicitly
raises in `setup()`.

## Files touched

- `code/labs/moe_cuda/optimized_router_vectorized.py` (local == pod, md5
  `8a883c55522d7637f385c88ae44ffc32`): `use_bmm_bias_fuse` dataflow in
  `GroupedTopKMoE.forward`, env plumbing (default ON + kill switch),
  loud-failure asserts, `gemm1_aten_enabled` metric export. B51 kill switch
  `AISP_MOE_CUDA_COMPILE=0` untouched.
- Probe/driver scripts (evidence, not lab code): pod `/tmp/frontM3/
  {probe_aten,probe_bmm,artifact_repro,replay_proof,parse_ab}.py`,
  `ab_driver.sh`.

## Pod evidence

`/tmp/frontM3/`: `r01_b51_repro.*` (pre-edit repro), `r02_probe_control.*`
(fresh-tune control + AUTOTUNE tables), `r03_probe_aten.*` (backend pin,
refuted), `r04_artifact.*` (isolated diagnosis), `r05_probe_bmm1.*`,
`r06_probe_bmm12.*`, `r07_smoke_b51.*` / `r08_smoke_aten.*` (0.40444 /
0.30578 ms), `r09_replay_{b51,aten}.*`, `r10_default_on.*` /
`r11_killswitch.*` / `r12_epilogue_yield.*` (post-flip confirms),
`ab/rep01..06_{b51,aten}.*` + `ab/ab_results_parsed.json` (authoritative A/B),
`ALL_DONE`; harness run dirs under
`code/artifacts/runs/20260611_*_targets_moe_cuda_router_vectorized/`.

## Mechanism (one paragraph)

The compiled graph's GEMM1+GELU pair cost 172.5us at B51 (104us Triton
template + 69us cached standalone GELU) because `baddbmm` with a broadcast
bias makes the extern cuBLAS choice honestly uncompetitive (bias must be
materialized into the output at 74.6us before a beta=1 GEMM) — so autotune
correctly kept the template, and pinning backends to ATEN just buys that copy
back (369.8 us/call, refuted). Rewriting both expert GEMMs as `bmm + bias`
splits the op at the right seam: the pure GEMM goes extern to nvjet's
beta-zero kernel (59.2/56.4us vs 66.3/58.1us for the beta=1 equivalents — C
is never read), and the rank-1 bias adds ride along in Inductor's pointwise
kernels (add+gelu 55.6us; GEMM2's bias folds into the gather-back) where they
are bandwidth-free. Net: 16 -> 15 kernels, ~100us/replay GPU saved vs the
incumbent arm, ~0.403 -> ~0.31 ms end-to-end.

## Named next lever

The optimized replay is now ~245us GPU for ~172 GFLOP of GEMM work: GEMM1
59.2us (86 GFLOP -> 38.7% bf16 SoL) and GEMM2 56.4us (40.6% SoL) are the
floor-setters, and the 55.6us add+gelu pointwise is 2.6x above its ~21us
memory SoL (erf ALU-bound, B54's conserved-cost finding — and its tune
lottery spans 41-76us). Named next lever: **capacity right-sizing** — the
GEMMs compute over `E*capacity = 20480` dense slots while only 8192
assignments are live (capacity 640 = 2x the observed max tokens/expert,
which is in [289, 320] for the seed-42 input, rounded up to 64); trimming
overprovision from 2x to ~1.4x (capacity 448) cuts GEMM+GELU work by 30% for
an estimated ~50us/replay (-> ~0.26 ms, another ~1.2x), at the cost of
re-validating the overflow-trash-row semantics (overflowed assignments are
silently dropped — calibration headroom is the only guard). Secondary:
pointwise-tune pinning for add+gelu (the 41 vs 76us lottery is 11% of the
replay).
