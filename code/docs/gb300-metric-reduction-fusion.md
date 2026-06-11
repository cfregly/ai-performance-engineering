# GB300: metric_reduction_vectorized single-pass fusion (Front W, B47 follow-on)

**Verdict: WIN.** The optimized arm of `labs/training_hotpath:metric_reduction_vectorized`
was three separate `(a*b).sum(dim=0)` torch passes (3x mul kernel + 3x reduce kernel +
cat) that re-read the 25.2 MB of fp32 inputs and bounced 12.6 MB temporaries through HBM
on every pass. Replaced with one fused CUDA kernel in the lab's existing extension that
reads `preds` and `targets` exactly once and produces all three per-responder sums
(`pred*pred`, `target*target`, `pred*target`) with fp32 register accumulators.

- Headline: **90.5x -> ~197x** vs baseline (harness, 3 reps), verification PASSED on all reps.
- Win gate vs prior optimized arm: **2.26x** (0.1285 ms -> 0.0568 ms median-of-3) >= 1.05x required.
- fp32 accumulation guardrail respected: output dtype/shape contract unchanged
  (`torch.float32`, `[3 * responders] = [768]`, same `torch.cat` ordering).

## Workload

`preds`/`targets`: fp32 `[32, 384, 256]` = 12.6 MB each, 25.2 MB total single-read.
Output: `[768]` fp32 (pred_sq | target_sq | covar per responder column).

## Before / after

| Measurement | Before (3-pass torch) | After (fused kernel) |
|---|---|---|
| Harness optimized arm (ms/iter) | 0.1285 (rep1) | 0.0578 / 0.0568 / 0.0565 (3 reps) |
| Harness speedup vs baseline | 90.54x | 197.29x / 197.74x / 196.77x |
| Harness verification | PASS (max_diff 0.0029) | PASS (max_diff 0.0137-0.0156, rtol/atol 1e-4 vs ~1.2e4-magnitude sums) |
| Kernel-frame, eager (CUDA events, median 300 iters) | 90.8 us | 23.4 us (3.9x) |
| Kernel-frame, CUDA-graphed (pure GPU work) | 78.9 us | 8.16 us (9.7x) |
| Main kernel GPU time (torch profiler) | n/a (6 kernels + cat) | 7.91 us |
| Effective single-read bandwidth (25.2 MB basis) | 277 GB/s (eager frame) | 3.19 TB/s (kernel), 1.07 TB/s (eager frame) |
| %HBM-SoL vs 8.0 TB/s | ~3.5% | **~40% (kernel)**, ~13% (eager frame) |

Harness-vs-kernel-frame note (B47 context): the harness adds ~33-35 us/iter of fixed
overhead here (0.0568 ms harness vs 23.4 us kernel frame), so the harness number
understates the kernel-level win (3.9x frame, 9.7x pure-GPU) — report both.

Remaining frame anatomy (eager 23.4 us): 7.9 us fused kernel + 1.2 us `torch::zeros`
fill kernel + ~14 us host (2x cudaLaunchKernel at ~4.5 us + pybind/allocator).

## Regression check (shared file)

`training_hotpath_kernels.cu` also carries the B47 chunked-grid `segment_abs_mean`
win. Re-ran `labs/training_hotpath:metric_reduction_cuda` after the edit:
0.0591 ms / 8.35x, verification PASS (max_diff 3.2e-06) — matches the recorded
expectations (0.0600 ms / 8.32x). No regression.

## Numeric cross-check (pre-harness)

Fused output vs three references, `torch.allclose(rtol=1e-4, atol=1e-4)` all True:
- vs `scalar_metric_reduction` (the baseline arm): max_abs 0.0156, max_rel 4.0e-4
- vs old `vectorized_metric_reduction`: max_abs 0.0156, max_rel 1.2e-3
- vs float64 reference: max_abs 0.0156, max_rel 8.4e-4

(Old optimized vs baseline was already max_abs 0.0029 at the same tolerance; sums have
magnitude ~1.2e4 so the rtol band is ~1.2.) Output is not bitwise-stable across calls
(atomic fold order), which the 1e-4/1e-4 verify contract permits.

## Exact change

Files (pod `/work/ai-performance-engineering/code`, mirrored to the Mac repo under
`code/labs/training_hotpath/`):

- `labs/training_hotpath/training_hotpath_kernels.cu`
  (md5 `0716d8a8abeb78a45def4b4ee63ebc61`, was B47 `3a306be38b320ed1e133e98b7dc14529`)
- `labs/training_hotpath/training_hotpath_common.py`
  (md5 `d8c6205e34b194b22204591748089be4`, was `3f31ed74fc61b9f1b427766262391f9e`)

1. `training_hotpath_kernels.cu`: added `metric_reduction_fused_kernel` — row-major
   `[num_rows, responders]`; consecutive threads cover consecutive responder columns
   (coalesced loads); each block strides rows with fp32 register accumulators
   (`fmaf`), then atomically folds partials into the zero-initialized
   `out[3 * responders]`. Host wrapper `metric_reduction_fused(preds, targets)`:
   shape/dtype checks, `torch::zeros({3*R})`, grid = `min(num_rows, sm_count*4)`
   blocks x 256 threads (SM count cached via the same static-query pattern as
   `segment_abs_mean`). New pybind def `metric_reduction_fused`.
2. `training_hotpath_common.py` (`MetricReductionVectorizedBenchmark`): optimized arm
   `setup()` loads the lab extension and warms the fused kernel; `benchmark_fn()`
   calls `extension.metric_reduction_fused(preds, targets)` instead of the 3-pass
   `vectorized_metric_reduction` (which remains as the reference function);
   `metric_reduction.uses_cuda_extension` metric now 1.0 for the optimized arm;
   `_extension` cleared in `teardown()`.

Full unified diff: pod `/tmp/frontW/fusion.diff` (169 lines). Backups of the
pre-change files: pod `/tmp/frontW/training_hotpath_kernels.cu.bak`,
`/tmp/frontW/training_hotpath_common.py.bak`.

## Session notes

- GPU 1 lease: on session start `/tmp/gpu1.lock` was held by an orphaned idle
  squatter (`flock -n /tmp/gpu1.lock -c "sleep 14400"`, PID 948889, ppid 1, zero GPU
  compute processes node-wide). Killed it per the no-idle-hold lease rule; all GPU
  work ran under `flock -n /tmp/gpu1.lock` with `CUDA_VISIBLE_DEVICES=1`.
- Logs: pod `/tmp/frontW/before_harness_run1.log`, `after_harness_run{1,2,3}.log`,
  `regression_cuda.log`; probes `/tmp/frontW/probe_metric.py`, `crosscheck.py`,
  `frame_breakdown.py`.

## Next lever

The eager frame is now ~60% host overhead (2 launches + pybind ~14 us) on 8.2 us of
GPU work. Named next lever: **single-launch fused kernel** — drop the `torch::zeros`
fill (cudaMemsetAsync or partials+last-block-final reduction on `torch::empty`) and
float4 loads with a shared-memory partial fold (kernel 7.9 us at 3.2 TB/s -> ~4-5 us
at ~5-6 TB/s). Est. eager frame 23.4 -> ~16 us, harness ~0.057 -> ~0.050 ms (~225x).
Beyond that the lab is harness-overhead-bound; the kernel-level ceiling play is
CUDA-graph capture at the harness level (pure GPU work is already 8.16 us, 9.7x over
the old arm's 78.9 us).
