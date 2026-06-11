# moe_cuda router_vectorized: torch.compile inside the manual CUDA graph (Front M)

## Verdict

**WIN — 1.271x median (0.4996 -> 0.3932 ms), verification passed on all 12 A/B reps,
bit-identical output on the lab input.** The B45-named lever (compile the now-static
forward) holds, with one mechanism correction: the win is **dispatch-chain fusion**
(33 -> 16 kernels/replay), not GELU-epilogue fusion — GELU stayed a standalone kernel
and is actually slower compiled; the eliminated `expand/copy` materializations dominate.

- Date: 2026-06-11, pod `<gb300-pod>` (GB300 NVL72, Dell XE9712), GPU 1 only.
- Torch `2.12.0a0+5aff3928d8.nv26.05`, CUDA 13.2. Local repo HEAD `222a8ac7`.
- Target: `moe_cuda:router_vectorized` (pair `baseline_router.py` + `optimized_router_vectorized.py`).
- Evidence: `/tmp/frontM/compile/` on the pod (every run as `.cmd/.stdout/.stderr/.exit`).

## Composition choice (the B45 care point)

The lab manually captures the whole forward into one `torch.cuda.CUDAGraph` (B45).
Chosen composition: **(a) `torch.compile(mode="max-autotune-no-cudagraphs",
fullgraph=True, dynamic=False)` compiled+warmed in `setup()`, then the COMPILED
callable is captured by the existing manual graph.** `benchmark_fn()` is unchanged:
`graph.replay()`, zero Python on the hot path.

- (b) `reduce-overhead` (Inductor cudagraph trees) rejected: it re-introduces per-call
  Python dispatch (guard eval + tree bookkeeping) — exactly what B45's one-graph replay
  removed — and hands capture to machinery the harness cannot see.
- (c) epilogue-only compile rejected: bias is already folded into the GEMMs via
  `baddbmm`, and a region-compile cannot fuse into a cuBLAS GEMM outside the region.
- Grad-mode pitfall found during implementation: compile warmup and capture MUST share
  grad mode, otherwise capture triggers a Dynamo recompile whose autotune launches
  would be captured into the graph. Warmup and capture both run under `no_grad`.

## Loud-failure audit rule (B45)

No silent fallback can pass: `fullgraph=True` raises on any graph break; after warmup
`setup()` raises unless dynamo counters show `graph_breaks == 0` and
`unique_graphs >= 1` (kills the silent-eager-fallback class, e.g. `TORCH_COMPILE_DISABLE`);
capture failure on the compiled path re-raises instead of falling back (the B45
kill-switch path keeps its original semantics). Proof is exported into harness metrics
every run: `router_vectorized.manual_graph_active`, `.compile_enabled`,
`.compile_seconds`, `.dynamo_graph_breaks`, `.dynamo_unique_graphs`.

## Kill switch

`AISP_MOE_CUDA_COMPILE=0` restores B45 behavior exactly (eager forward captured by the
manual graph). Default ON. `get_config()` now sets `setup_timeout_seconds=600` to cover
the one-time compile cost (setup is untimed; cost reported via `compile_seconds`).

## Before/after (interleaved A/B, 6 reps/arm, GPU1, quiet node)

| Arm | median ms | min ms | max ms | n | vs eager baseline arm |
|---|---|---|---|---|---|
| B45 (`AISP_MOE_CUDA_COMPILE=0`) | 0.49958 | 0.49760 | 0.50228 | 6 | 188–216x (one 398x rep from baseline-arm drift) |
| compile (`=1`) | 0.39317 | 0.39094 | 0.39449 | 6 | 241–258x |
| **speedup (median/median)** | **1.2707x** | | | | |

Distributions do not overlap (compile max 0.39449 < B45 min 0.49760). Node thermal
drift note: the eager *baseline arm* drifted 94–198 ms across reps; the optimized-arm
A/B metric above is the interleaved, tight one.

Rep table (`/tmp/frontM/compile/ab/ab_results_parsed.json`; reps alternate B45/compile):

| rep | arm | optimized ms | verified | graph_breaks | compile_s |
|---|---|---|---|---|---|
| 1 | b45 | 0.50040 | pass | – | – |
| 2 | compile | 0.39449 | pass | 0 | 2.62 |
| 3 | b45 | 0.49865 | pass | – | – |
| 4 | compile | 0.39380 | pass | 0 | 2.60 |
| 5 | b45 | 0.49760 | pass | – | – |
| 6 | compile | 0.39285 | pass | 0 | 2.60 |
| 7 | b45 | 0.49901 | pass | – | – |
| 8 | compile | 0.39202 | pass | 0 | 2.58 |
| 9 | b45 | 0.50016 | pass | – | – |
| 10 | compile | 0.39094 | pass | 0 | 2.57 |
| 11 | b45 | 0.50228 | pass | – | – |
| 12 | compile | 0.39349 | pass | 0 | 2.63 |

All 12 reps: harness `verification {"passed": true, rtol 0.1, atol 1.0}`,
`manual_graph_active=1.0`; compile reps additionally `dynamo_graph_breaks=0`,
`dynamo_unique_graphs=1`. B45 repro before any edit: 0.5024 ms
(`r01_b45_repro.stdout`), matching the banked ~0.50 ms.

## Compile cost (one-time, in setup, untimed)

Cold (empty Inductor FX cache): **15.55 s** (`r02_compile_cold.stdout`). Warm (disk
cache): **2.57–2.63 s** per harness invocation. max-autotune at these shapes is cheap.

## Numerics class

**Bit-identical on the lab input** (probe `r15_probe.stdout`: `bitwise_equal: true`,
`max_abs_diff: 0.0` between B45-arm and compile-arm outputs, same seed-42 input,
fixed routing). This is an observed equality on this input, not a general guarantee —
GEMM1 changed implementation (nvjet -> Triton template) and bf16 output rounding
absorbs the fp32-accumulation differences here. Harness verification (vs the eager
per-token baseline, lab tolerance rtol=0.1/atol=1.0 — untouched) passed on all reps.

## Mechanism (kernel diff, torch profiler over 5 replays, `r15_probe.stdout`)

GPU time per replay: **456.1 us -> 347.4 us (1.313x)**; kernels per replay **33 -> 16**.
Harness delta (0.4996 -> 0.3932 ms) matches on top of ~45 us fixed harness overhead.

- Eliminated: the B45 arm's single biggest cost — 130.6 us/replay of `direct_copy`
  elementwise (3 kernels: `flat_tokens` expand materialization + gather-back copies) —
  plus ~14 small ATen launches (one-hot `eq`, `cumsum` postlude, `clamp/where/lt/mul`,
  `arange`, fills, softmax) fused into 6 `triton_poi_*` kernels.
- GEMM1 (`baddbmm` [32,cap,1024]x[32,1024,2048]): nvjet 65.9 us -> Triton template
  `triton_tem_fused_baddbmm_slice_unsqueeze_view` 104.0 us — slower in isolation, but
  autotune picked it because it absorbs the dense-slot view/slice traffic around it.
- GEMM2 unchanged (nvjet `128x160_64x8`, ~58 us). `index_add_` combine unchanged (~26 us).
- GELU: **not** fused into a GEMM epilogue; standalone `triton_poi_fused_gelu` 69.3 us
  vs eager 44.5 us (slower). The named lever's stated mechanism ("fuse bias+GELU
  epilogues") is therefore REFUTED as the win source; the win is dispatch-chain fusion.

## Files touched

- `code/labs/moe_cuda/optimized_router_vectorized.py` (local == pod,
  md5 `c029a01ca6eece2a5ec02722c117c68a`): compile-then-capture in `setup()`,
  loud-failure asserts, kill switch, evidence metrics, `setup_timeout_seconds=600`.
- Helper scripts (evidence, not lab code): local `code/ab_driver_frontM.py`,
  `code/reparse_frontM_ab.py`, `code/probe_frontM_numerics_kernels.py` = pod
  `/tmp/frontM/compile/{ab_driver,reparse_ab,probe_numerics_kernels}.py`. Safe to
  delete locally after banking (rm was permission-denied for this front).

## Pod evidence

`/tmp/frontM/compile/`: `r01_b45_repro.*`, `r02_compile_cold.*`,
`ab/rep01..rep12_{b45,compile}.*`, `ab/ab_results.json` (raw),
`ab/ab_results_parsed.json` (authoritative summary), `ab/reparse.stdout`,
`r15_probe.*`; harness run dirs under
`code/artifacts/runs/20260611_*_targets_moe_cuda_router_vectorized/`.
Pre-existing pod-only state, untouched: untracked
`code/labs/moe_cuda/expectations_4x_gb300.json` (mtime Jun 9, predates Front M).

## Named next lever

The compiled arm's remaining top costs are GEMM1's Triton template (104 us — it beat
nvjet only because it absorbs copy traffic) and the now-dominant `triton_poi_fused_gelu`
at 69.3 us (slower than eager's 44.5 us, i.e. a poorly codegen'd pointwise at
[32,cap,2048] bf16). Named next lever: **force GELU into the GEMM1 template epilogue**
(`torch._inductor.config` epilogue fusion for templates / `tanh` approx variant) or
hand-fuse `baddbmm+GELU` for GEMM1 — est. ~60–100 us/replay (another ~1.2x) if the
template epilogue matches nvjet GEMM throughput. Secondary: the 40.6 us residual
`direct_copy` (gather-back) could fold into GEMM2's prologue the same way.
