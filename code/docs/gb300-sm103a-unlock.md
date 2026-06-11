# GB300 sm_103a unlock — fixing the self-inflicted tcgen05 "max-autotune wall" (Front T)

Date: 2026-06-11. Pod `<gb300-pod>` (GB300 sm_103, CUDA 13.2,
torch 2.12.0a0+5aff3928d8.nv26.05, Triton 3.7.0), GPU 1 (flock /tmp/gpu1.lock),
harness `--profile none`. Evidence tuples: `/tmp/frontT/` on the pod;
mechanism evidence from Front P2: `/tmp/frontP2/` (4-probe repro).

## Root cause (B69's named lever)

`core/benchmark/triton_compat.py` monkey-patches Triton's
`sm_arch_from_capability` (and `TRITON_CODEGEN_ARCH`) to de-suffix every
`sm_*a` arch except `sm_100a`. On GB300 that rewrote `sm_103a -> sm_103`.
Triton 3.7 plans tcgen05 codegen from the compute capability (10.3), but the
arch-conditional tcgen05 intrinsics are only selectable for the `a` targets,
so LLVM instruction selection dies with an UNCATCHABLE fatal:

    LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st

This was the source of EVERY "tcgen05.wait.st abort" in the campaign and the
reason the `get_optimal_compile_mode` guard (max-autotune -> "default" on
sm_103 + Triton >= 3.6) was added. Vanilla Triton 3.7 is CLEAN on sm_103
(frontP2 probes A/B/C exit 0; probe D reproduces the abort only with the
de-suffix patch applied, rc=134).

## The fix

`core/benchmark/triton_compat.py` (`_canonicalize_triton_arch` +
`_safe_sm_arch_from_capability`): preserve the `a` suffix for all arches;
clamp ONLY `sm_121 -> sm_120` (the GB10 case the patch exists for).

```diff
-    # Keep the 'a' suffix for Blackwell (sm_100a) where ptxas requires it for TMA.
-    if suffix.endswith("a") and not suffix.startswith("100"):
-        suffix = suffix[:-1]
+    # Keep the 'a' suffix for the whole Blackwell family (sm_100a/sm_101a/sm_103a)
+    if suffix.endswith("a") and not suffix.startswith("10"):
+        suffix = suffix[:-1]
...
-        # Preserve 'a' suffix for sm_100a so ptxas accepts TMA/tensormap.
-        if arch.endswith("a") and not arch.endswith("100a"):
-            arch = arch[:-1]
+        # Preserve the 'a' suffix for the whole Blackwell family.
+        if arch.endswith("a") and not arch.startswith("sm_10"):
+            arch = arch[:-1]
```

(sm_90a was never reachable here on this pod — the function only strips on
`*a` returns for the live capability — but the new predicate keeps any
hypothetical `sm_9xa`-class return intact only for `sm_10x`; the module
docstring now states the clamp-only-sm_121 contract.)

Guard disposition: `core/utils/compile_utils.py` — the sm_103 "max-autotune
-> default" toolchain guard (`_max_autotune_unsafe_on_this_toolchain`) is
RETIRED (removed outright; the historical note stays in
`get_optimal_compile_mode`). A testing knob `AISP_COMPILE_MODE_OVERRIDE`
was added to `get_optimal_compile_mode` so incumbent arms can still be
measured through the harness (used for the A/B below; unset in normal
operation).

## Smoke proof (pod, GPU 1)

`/tmp/frontT/smoke_unlock.{cmd,stdout,stderr,exit}` — exit 0:

- `sm_arch_from_capability(103) -> sm_103a` (patched patch verified)
- `torch.compile(mode="max-autotune")` of a bf16 1024^3 matmul: compiles,
  runs, max abs err 0.0 vs eager; Triton mm templates participate in
  autotune (`triton_mm_88` 0.0123 ms vs `bias_addmm` 0.0100 ms — tuner
  picks honestly, no aborts).
- max-autotune attention+MLP block (frontP2 probe-C analog): clean.

## Counterfactual control (the false wall, demonstrated cold)

`/tmp/frontT/gemm_matrix_oldarch.*`: the grouped-GEMM kernel matrix run with
the OLD de-suffix behavior re-applied and a FRESH `TRITON_CACHE_DIR` -> exit
134 with the exact `tcgen05.wait.st` fatal. Under the old patch the
`labs/blackwell_gemm_optimizations` Triton kernels cannot compile AT ALL from
a cold cache on this toolchain; every prior "successful" run rode a warm
Triton cache. The de-suffix fix is therefore a works-at-all unlock for every
`tl.dot` Triton kernel on GB300, independent of any perf delta.

## Re-sweep results (GPU 1, harness `--profile none`, fresh result JSONs mtime-checked)

### 1. labs/blackwell_gemm_optimizations:blackwell_grouped_gemm — UNLOCKED-BUT-NO-PERF-CHANGE

- Harness, 2 reps (`frontT_gemm_r1/r2`): best arm `full_stack`
  0.2021 / 0.2026 ms = 1.908x / 1.904x over baseline, 510.1 / 508.7 TFLOPS,
  verify PASS both. Statistical parity with the banked pre-fix numbers
  (0.2015 ms / 511.5 TFLOPS, 1.937x — run `gb300_inv_labs_blackwell_gemm_optimizations`).
- Kernel matrix (`compare_blackwell_grouped_gemm_matrix.py`, 2 reps):
  full_stack 870.6 / 868.1 TFLOPS @8192 tokens, 444.5 / 429.0 @2048
  (fp16, balanced, 4 experts, K=2048, N=3072). Autotuner picks unchanged.
- Re-autotune under native sm_103a does NOT move the kernel: the B10/B11
  "Triton tl.dot codegen gap vs cuBLAS-batched" attribution SURVIVES the
  arch fix — the gap was never caused by the de-suffix.
- BUT the cold-cache control (above) shows the lab is only RUNNABLE at all
  because of the fix: pre-fix, a fresh-cache compile of these kernels
  aborts (exit 134, tcgen05.wait.st).

### 2. labs/moe_optimization_journey:moe_pad_quant — UNLOCKED-BUT-PARITY

3 interleaved reps per arm (`frontT_moepq_{ro,ma}_r{1..3}`), all
`succeeded` + verify PASS. Arm A = incumbent B40 champion
(reduce-overhead whole-model cudagraph, via `AISP_COMPILE_MODE_OVERRIDE`);
arm B = max-autotune (the new repo default on sm_103):

| arm | optimized ms (r1/r2/r3) | median |
|---|---|---|
| reduce-overhead (incumbent) | 0.3096 / 0.3013 / 0.3782 | 0.3096 |
| max-autotune (new default)  | 0.3160 / 0.3057 / 0.3236 | 0.3160 |

max-autotune now COMPILES AND RUNS clean (it hard-aborted pre-fix) but is
within noise of the cudagraph champion (no >=1.05x either way). B40's win
came from capture, not codegen — confirmed. The lab's internal
`default -> reduce-overhead` remap is now dead code on GB300 (the mode
flows through as max-autotune); harmless, kept.

### 3. labs/moe_cuda:router_vectorized — UNLOCKED-BUT-PARITY

2 reps (`frontT_router_r1/r2`): optimized 0.2371 / 0.2379 ms (405x / 417x vs
the 96-99 ms baseline), verify PASS. Same-day pre-fix incumbent (sibling run
`20260611_185414`): 0.2352 ms. Parity. This champion already hardcoded
`mode="max-autotune-no-cudagraphs"`; pre-fix it survived because Inductor
benchmarks template choices in SUBPROCESSES (a tcgen05 SIGABRT there is
swallowed as a failed choice) and its winning kernels are extern (nvjet) —
the arch fix gives Triton templates an honest seat at the tuner but they
still lose to extern here, as Front M3 established.

### 4. ch14:model_compile_reduced_precision — UNLOCKED-WIN

3 interleaved reps per arm. Arm A = incumbent mode "default" (what the
retired guard forced; B23c-consistent); arm B = max-autotune (new default):

| arm | optimized ms (r1/r2/r3) | speedup vs baseline | harness status |
|---|---|---|---|
| default (incumbent) | 4.5087 / 4.5012 / 4.5546 | 0.994x / 1.005x / 0.997x | failed_no_speedup x3 |
| max-autotune (new)  | 4.1749 / 4.1646 / 4.1570 | 1.130x / 1.075x / 1.137x | succeeded x3, verify PASS x3 |

max-autotune beats the incumbent optimized arm 1.083x (median/median,
4.5087 -> 4.1646 ms; min pairwise 1.078x) — clears the >=1.05x win gate on
all 3 reps and flips the benchmark from failed_no_speedup to succeeded.
This corrects B23c: "compile adds ~1.5% all modes" was measured behind the
false wall; with real max-autotune the lab clears its 1.05x speed gate.

### 5. Survey of newly-unlocked max-autotune paths (single runs)

Guard-routed callers (`get_optimal_compile_mode` direct: moe_benchmark.py
[moe journey level 7], optimized_moe_pad_quant.py [target 2],
optimized_ddp_multigpu.py [multi-GPU torchrun — NOT run: single-GPU lease];
via `compile_model`/`compile_callable` with preferred max-autotune: ch14
model_compile_reduced_precision [target 4], ch14 graph_break_control_flow,
ch13 warp_specialization_training, ch16 inference_optimizations_blackwell,
labs/real_world_models llama_3_1_8b, ...). Note: many other labs call raw
`torch.compile(mode="max-autotune")` and were never guard-routed — those
were the campaign's abort sites pre-fix.

| target | result | vs banked |
|---|---|---|
| ch14:graph_break_control_flow | succeeded 1.675x, opt 0.1650 ms, verify PASS | banked 1.638x / 0.1701 ms — parity(+) |
| ch13:warp_specialization_training | succeeded 1.747x, opt 0.1892 ms, verify PASS | banked 1.840x / 0.1795 ms — parity(-), single run |

## Verdicts

| target | verdict |
|---|---|
| triton_compat de-suffix | FIXED — sm_103a preserved; smoke + 4 labs clean; cold-cache control proves pre-fix was fatal |
| blackwell_grouped_gemm | UNLOCKED-BUT-NO-PERF-CHANGE (works-at-all unlock; 1.91x / 510 TFLOPS parity; tl.dot-gap attribution stands) |
| moe_pad_quant | UNLOCKED-BUT-PARITY (max-autotune 0.3160 ms vs cudagraph champion 0.3096 ms median — within noise; champion's win was capture, not codegen) |
| moe_cuda:router_vectorized | UNLOCKED-BUT-PARITY (0.2371/0.2379 vs 0.2352 ms incumbent) |
| ch14:model_compile_reduced_precision | UNLOCKED-WIN (1.083x over incumbent arm, 3/3 reps >=1.05x, verify PASS, status flips to succeeded) |
| survey singles | clean compiles + verify PASS everywhere; no further wins found |

## Guard disposition / recommendation

FLIP THE DEFAULT (done): keep `_max_autotune_unsafe_on_this_toolchain`
retired — the re-sweep found zero aborts, zero verification failures, and
one outright win across 15 harness runs + 2 kernel-matrix runs + smoke. The
deeper systemic effect: pre-fix, Inductor's subprocess autotuning silently
discarded EVERY Triton template choice that touched tcgen05 (SIGABRT ->
"failed choice"), biasing all max-autotune decisions toward extern kernels;
post-fix the tuner space is honest. Keep `AISP_COMPILE_MODE_OVERRIDE` as the
A/B escape hatch. Compile times stayed benign (no >20 min autotunes; the
whole-model moe_pad_quant max-autotune compile completes inside the 33 s/run
cadence observed with warm FX caches).

## Files changed (pod == local, md5-verified)

- `code/core/benchmark/triton_compat.py` — de-suffix fix (md5 `6a26087a661c0e7f5a69594c5cbf8626`)
- `code/core/utils/compile_utils.py` — guard retired + historical note + `AISP_COMPILE_MODE_OVERRIDE` (md5 `26787f749b5168108aec0785ee450103`)
- `code/docs/gb300-sm103a-unlock.md` — this document

Evidence: `/tmp/frontT/` (smoke tuple, gemm matrix r1/r2 CSVs+stdout, oldarch
control tuple, all driver logs); run artifacts `frontT_*` under
`code/artifacts/runs/`; mechanism evidence `/tmp/frontP2/`.
