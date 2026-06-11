# GB300: harness frame-overhead audit + opt-in graphed timing mode (Front H2)

**Verdict.** The default timing frame adds a measured **~22 us to every reported
iteration number** (the event-bracket floor through the real harness) plus
**~3.2 us per additional kernel launch**, and **~53-61 us of wall time per
iteration outside the bracket** (L2 clear 31, post-sync 5, Python bookkeeping
13-20, +20 force-eval when `benchmark_fn` returns a tensor). Four microsecond
labs are now >79% frame: their reported numbers are mostly ruler, not workload.
An **opt-in** graphed timing mode (`AISP_BENCH_TIMING_MODE=graphed`) now reports
the pure-GPU per-iteration figure *alongside* the untouched frame-mode number;
it reproduced B50's and B58's banked kernel figures on the first two demo labs
and refused loudly on the third (by design). The default path is bit-for-bit
unchanged with the env var unset — regression runs on three targets match
prior numbers within noise, verification PASS everywhere.

Hardware: GB300 (cc 10.3, L2 = 129.25 MB hardware-detected, flush buffer
142.2 MB), torch 2.12.0a0+nv26.05, GPU3, quiet (foreign-process prechecks on
every run). Evidence: `/tmp/frontH2/h2_01..h2_10c.{cmd,stdout,stderr,exit,precheck}`
+ `/tmp/frontH2/h2_audit.json` on the pod.

## 1. The frame-cost model (measured, medians)

```
T_reported  ~=  22us (bracket floor through real harness)
              + 3.2us x (kernel launches beyond the first)
              + fn host residual (pybind / Python inside benchmark_fn)
              + T_pureGPU
T_wall/iter ~=  T_reported + 35us (L2 zero_+sync, OUTSIDE the bracket)
              + 5us post-sync + 13-20us bookkeeping
              + 20us force_tensor_evaluation (only if fn returns a tensor)
```

Component probes (`h2_01`, exact `_run_single_iteration` replica + isolation;
`h2_02`, the real `_benchmark_custom` with empty / 1-kernel fns):

| Component | Median |
|---|---|
| idle `torch.cuda.synchronize()` | 4.70 us host |
| event bracket (2x record + full sync + elapsed) | 19.97 us host / **4.29 us GPU-reported floor** |
| L2 clear: 142.2MB `zero_()` + sync | 31.4 us wall (26.1 us GPU kernel) — *outside* the bracket |
| kernel launch, steady enqueue | 3.14 us |
| kernel launch, drained-queue marginal (8-launch chain) | ~3.3 us |
| 1 tiny kernel drained, event-reported vs graphed | 10.08 us vs 0.66 us |
| `force_tensor_evaluation` (sync + `.item()` DtoH) | 20.1 us wall |
| **real harness, empty `benchmark_fn`** | **21.9 us reported / 75.0 us wall per iter** |
| real harness, 1 tiny kernel | 28.7 us reported / 82.7 us wall per iter |

Why the reported number inflates: the per-iteration full-device sync drains the
pipe, so the event bracket measures host dispatch (Python + launch latency)
*serially in front of* the GPU work instead of overlapped with it. The L2
clear, post-bracket sync, bookkeeping and force-eval are wall-only (they slow
the run, not the reported number).

Model validation against measured arms: metric_reduction **baseline**
(1543 launches/iter, profiler-counted): 22 + 1543x4.6 + 3620 GPU ≈ 10.7 ms vs
10.6 ms measured — the graphed supplement independently measured its frame
overhead at 7.14 ms ≈ 1543 x 4.6 us. Optimized arm: 22 + 2x3.2 + ~14 pybind +
9.0 GPU ≈ 51.6 vs 55-57 measured. blackwell tcgen05: measured frame overhead
84.8 us/call = B55's "wrapper host-dispatch-bound ~85-100 us" finding.

L2-clear efficacy check (`h2_05`): the eager `zero_()`+sync **does evict** on
GB300 (25MB read: 18.4 us warm -> 22.1 us after clear ≈ the 25MB/8TB/s HBM
delta); in-graph zeros also evict (graph-interleaved sum 14.9 us vs 10.6 warm).
The default frame's cold-L2 contract is intact.

## 2. Frame-share ranking ("what would a better ruler unlock")

frame share = (frame-mode reported - pure-GPU) / frame-mode, optimized arms:

| # | Lab : target | Frame-mode | Pure-GPU | Frame share | Source |
|---|---|---|---|---|---|
| 1 | nvfp4_dual_gemm | 47.5 us | 6.6 us | **86%** | banked B38/B41 |
| 2 | fullstack_cluster:capstone_extension_tcgen05 | ~100 us | 15.2 us | **~85%** | inferred (same kernel+wrapper as #3; B44/B55) |
| 3 | blackwell_matmul:blackwell_matmul_tcgen05 | 105.2 us | 16.8 us | **84%** | fresh graphed demo `h2_09b`; B58 kernel 15.2 us |
| 4 | training_hotpath:metric_reduction_vectorized | 56.7 us | 11.7 us | **79%** | fresh graphed demo `h2_09a`; B50 8.16 us |
| 5 | software_pipelining | 124.5 us | 60.5 us | 51% | banked B41 |
| 6 | block_scaling:block_scaling | 80.5 us | 44.8 us | 44% | fresh `h2_03c`; B38/B46 kernel (graphed refuses, see below) |
| 7 | moe_cuda | 235 us | ~200 us | ~15% (est.) | B59 arm; harness floor + dispatch estimate |

Labs 1-4: further kernel wins are **invisible end-to-end** in frame mode — a
2x kernel cut moves the headline <10%. The graphed figure is the honest
denominator for kernel-level claims there.

## 3. Opt-in graphed timing mode (implemented)

`AISP_BENCH_TIMING_MODE=graphed` (knobs: `AISP_BENCH_GRAPHED_CAPTURE_ITERS`,
default 50; `AISP_BENCH_GRAPHED_REPLAYS`, default 30). In
`core/harness/benchmark_harness.py::_benchmark_custom`, **after** the default
loop and every default-path protection completes:

1. capture N iterations of `benchmark_fn` into ONE CUDA graph (side stream);
   time whole-graph replays with a single event bracket, divide by N → **warm**
   pure-GPU figure;
2. cold-L2 figure: interleaved `(l2_zero_; fn)` graph minus a clears-only graph
   (only when `clear_l2_cache` is on);
3. **capture-fidelity audit**: per-iteration kernel-launch counts (torch
   profiler) of one eager call vs one replay must match exactly, else
   **RuntimeError** — no silent fallback (B45 rule);
4. one eager `fn()` after the replays restores benchmark state onto ordinary
   allocator memory (the graphs' private pools die with the graph objects —
   without this the verification payload reads recycled memory; observed as a
   max_diff 12854 verification failure in `h2_04a` before the fix);
5. results land in `custom_metrics` (`graphed_iter_ms_median` = cold when
   available else warm, `graphed_warm_iter_ms_*`, `graphed_cold_iter_ms_median`,
   `graphed_eager_kernel_count/_ms`, `frame_iter_ms_median`,
   `frame_overhead_ms_median`) and a loud `[harness]` log line. `times_ms` —
   the banked, comparable metric — is never replaced.

Unset env ⇒ the supplement block is a no-op and `_resolve_custom_metrics` only
gains a `getattr` on an attribute that does not exist: default behavior
bit-for-bit identical.

### Demonstration (fresh, GPU3)

| Lab | Frame-mode | Graphed cold | Graphed warm | Known kernel figure | Verify |
|---|---|---|---|---|---|
| metric_reduction_vectorized | 56.7 us | **11.65 us** | 9.04 us | B50 pure-GPU 8.16 us, kernel 7.91 + fill 1.07 | PASS |
| blackwell_matmul_tcgen05 | 105.2 us | **16.79 us** | 15.04 us | B58 kernel 15.2 us | PASS |
| block_scaling | 80.5 us | — | — | **LOUD REFUSAL** (exit 1): fn replays its own internal CUDA graph; outer capture fails; no fallback | n/a (default mode unaffected) |
| metric_reduction baseline (bonus) | 10.6 ms | 3.66 ms | 3.62 ms | 1543 kernels/iter; frame overhead 7.14 ms = launch dispatch | PASS |

### Silent-kernel-drop finding (B45 class, second instance)

The first graphed run on metric_reduction reported **1.2 us/iter** — the fused
reduction kernel was launched `<<<blocks, threads>>>` on the **legacy default
stream** and was **silently dropped** from the side-stream capture: the
replayed graph contained only the `at::native` fill kernel
(profiler-of-replay + timing proof in `h2_06_probe_mr_kernel`). No CUDA error
surfaced anywhere. Fixes shipped:

- lab-side: `labs/training_hotpath/training_hotpath_kernels.cu` — all 4 launch
  sites now pass `c10::cuda::getCurrentCUDAStream()` (identical eager behavior;
  capture-legal);
- harness-side: the kernel-count capture-fidelity audit above turns this class
  into a loud refusal for any other lab that still launches on the legacy
  stream.

## 4. Default-path regression proof (env unset)

| Target | Before (h2_03) | After (h2_10) | Verification |
|---|---|---|---|
| metric_reduction_vectorized | 0.0556 ms / 191.9x | 0.0567 ms / 195.3x | PASS |
| metric_reduction_cuda (shared .cu sibling) | B50 expectation 0.0591 ms / 8.35x | 0.0558 ms / 8.83x | PASS |
| blackwell_matmul_tcgen05 | 0.1127 ms / 139.0x | 0.109 ms / 143.3x | PASS |

Zero `graphed_*` keys appear in default-mode output (grep count 0 on all three
runs). All within run-to-run noise; B50's banked range for metric_reduction was
0.0565-0.0578 ms.

## 5. Files touched (md5 local == pod, verified)

- `core/harness/benchmark_harness.py` — `f0e3ae1705920b4b89e088b7933407b6`
  (opt-in supplement + capture-fidelity audit + graphed-metrics merge in
  `_resolve_custom_metrics`; default path unchanged)
- `labs/training_hotpath/training_hotpath_kernels.cu` —
  `750507c2387aee3dac4c562f56d82204` (current-stream launches, 4 sites)

## 6. Caveats

- The graphed **cold** figure is a cross-graph subtraction; for sub-2 us
  kernels its error band is ~1 us (warm/cold can invert within noise). The
  warm figure is exact. Frame-mode numbers remain the only contract metric.
- The supplement runs `benchmark_fn` ~130 extra times after measurement;
  benchmarks must be call-idempotent (already an implicit harness requirement
  under adaptive iterations).
- First graphed run after a lab-extension edit can hit the measurement timeout
  while ninja rebuilds inside the arm's subprocess (~106 s observed in
  `h2_07a`); re-run once the build cache is warm.

## 7. Named next lever

**Repo-wide legacy-stream launch audit**: grep all `.cu`/extension launch sites
for `<<<...>>>` without an explicit stream argument and fix to
`getCurrentCUDAStream()` — the silent-drop class found here breaks any
graph-capture path (harness `enable_cuda_graph`, B45-class in-lab captures,
this mode) without an error. Then: bank dual-figure (frame + graphed) numbers
for the four >79%-frame labs so future kernel wins are measurable end-to-end;
the capstone wrapper's ~85 us host residual (pybind/checks, B55) is the
single biggest now-visible target.
