# GB300 SoL Survey #3 — The Labs That Fell Through Every Prior Survey (Front S)

- **Date:** 2026-06-11 | **Pod:** <gb300-pod> (<namespace>), GPU 1 (CUDA_VISIBLE_DEVICES=1, flock /tmp/gpu1.lock)
- **Repo:** /work/ai-performance-engineering/code @ 2f7e30f9 + uncommitted GB300 fixes
- **Scope (B42 audit gap):** labs/parameterized_cuda_graphs, python_concurrency, training_hotpath, uma_memory, tcgen05_cluster_shapes, moe_parallelism (classify only), vllm-deepseek-tuning (classify only)
- **B45 rule applied:** every cudagraph/compile wrapper in scope audited for silent capture-failure→eager fallbacks.
- **GB300 peaks used:** HBM3e 8.0 TB/s; BF16/FP16 3.75, FP8 7.5, NVFP4 15.0 PFLOPS.

## Ranking table

| # | Lab : target | Baseline ms | Optimized ms | Speedup | Verify | %SoL (optimized) | Class |
|---|---|---|---|---|---|---|---|
| 1 | training_hotpath : metric_reduction_cuda | 0.4951 | 0.0610 → **0.0546** (prototyped) | 8.11x → **9.01x** (confirm 8.89x) | PASS | kernel DRAM 4.2% → 6.3% (ncu) | HEADROOM → partially harvested; remainder OVERHEAD-BOUND |
| 2 | training_hotpath : metric_reduction_vectorized | 11.9355 | 0.1275 | 93.59x | PASS | ~22% (1.73 TB/s eff., 126 MB/iter incl. materialized intermediates) | HEADROOM (addressable, named lever, not attempted) |
| 3 | training_hotpath : padding_aware_transformer | 2.1388 | 3.1745 | **0.674x speed** (goal=memory: 1243.8→312.6 MB, 3.98x) | PASS | n/a (launch-bound, not BW-bound) | SPEED-REGRESSION (memory-goal lab) |
| 4 | parameterized_cuda_graphs : parameterized_graph_launch | 2.5491 | 0.0754 | 33.83x | PASS | n/a (launch-overhead lab by design) | OVERHEAD-BOUND / AT-CEILING; B45-clean |
| 5 | tcgen05_cluster_shapes (sweep tool) | — | — | — | — | — | BROKEN-BUILD: CUTLASS header gap (see below) |
| 6 | uma_memory | — | — | — | — | — | NOT-A-BENCHMARK (helpers only; ch02 owns the reporting target) |
| 7 | python_concurrency | — | — | — | — | — | NOT-A-BENCHMARK (CPU teaching lab; scripts run fine) |
| 8 | moe_parallelism | — | — | — | — | — | NOT-A-BENCHMARK (CPU planner tool, runs fine); vLLM presets env-gapped |
| 9 | vllm-deepseek-tuning | — | — | — | — | — | ENV-GAP (no `vllm` module on pod; Makefile/script matrix, no harness targets) |

Only 4 of the 7 labs expose harness targets at all (`bench list-targets` confirms). All 4 ran and verify-passed on GB300.

## Single most important finding

**labs/training_hotpath:padding_aware_transformer's "optimized" packed-row path is 0.674x — i.e., 48% SLOWER than the dense baseline on GB300 — and only counts as a success because the lab's optimization goal is memory (3.98x memory saving).** CUDA-event re-measurement isolates it: dense 1.647 ms vs packed 2.597 ms per iteration. The prior static-sweep flag (torch::zeros in `scatter_rows`) is confirmed real and per-iteration — 18 `scatter_rows` calls/iter (4 blocks x 4 PackedLinears + in/out proj), each allocating+memsetting a full (8192, cols) fp32 tensor, ~315 MB/iter of pure memset — **but it is NOT the dominant cost**: at 8 TB/s that's only ~40 µs (+ ~18 launches ≈ ~100 µs total) of the ~950 µs regression. The packed path's real cost is ~72 extra tiny-kernel launches/iter (pack+memset+scatter per linear) plus odd-shape GEMMs (4587 active rows), which GB300's fast dense GEMMs make pointless at this workload size (16x512x256, ~56% active tokens). The packing trick only pays at much larger hidden sizes / lower active fractions.

## Prototype (the one surgical win): segment_abs_mean occupancy fix

**File:** `labs/training_hotpath/training_hotpath_kernels.cu` (single file; backup at `/tmp/frontS/training_hotpath_kernels.cu.bak`, orig md5 792d8dfd…, patched md5 3a306be3…).

**Diagnosis (ncu, GPU 1):** the optimized `segment_abs_mean_kernel` launched one block per segment → Grid 128, **Waves/SM 0.11, achieved occupancy 12.4%, DRAM 4.2% SoL**, 15.5 µs for a 5.24 MB read (128 segments x ~10.2K elems; SoL floor 0.66 µs).

**Fix (attempt 2 of ≤3):** chunked grid `(num_segments, chunks)` with `chunks = clamp(8*SM_count/num_segments, 1, 64)` (=10 on GB300 → 1280 blocks); each block reduces its slice and `atomicAdd`s `partial/length` into the zero-initialized output (numerically safe: reorder error ~1e-8 vs 1e-5 tolerance). SM count cached in a function-local static — **attempt 1 queried `cudaDeviceGetAttribute` per call and that ~10 µs host hit made the target SLOWER (77.9 µs, 6.32x); lesson: per-call driver attribute queries are not free at microsecond kernel scale.**

| Metric | Before | After |
|---|---|---|
| Harness optimized time | 61.0 µs | 54.6 µs (confirm 55.5 µs) |
| Harness speedup | 8.11x | **9.01x (confirm 8.89x), verify PASS both runs** |
| ncu kernel duration | 15.5 µs | 10.3 µs |
| Grid / Waves per SM | 128 / 0.11 | 1280 / 1.05 |
| Achieved occupancy | 12.4% | 76–84% |
| DRAM %SoL | 4.2% | 6.3% |

**Honest ceiling note:** back-to-back CUDA-event timing is unchanged (10.7 → 10.9 µs/iter) because the per-call host path (torch::zeros alloc + launch) floors at ~10.8 µs; the harness adds another ~30–55 µs/iter of per-iteration measurement overhead. The remaining kernel headroom (~10.3 → ~3-4 µs via warp-shuffle reduce + float4 loads) is real on the ncu roofline but wall-clock-invisible at this workload size. This lab is now OVERHEAD-BOUND.

## B45 audit (silent capture-fallback class)

**No silent cudagraph/compile→eager fallbacks found in any of the 7 labs.** Specifics:

- `parameterized_cuda_graphs`: the ONLY capture-path lab in scope, and it is loud-by-construction. Missing `cuda.bindings` → RuntimeError in optimized `setup()` (no eager path exists); memcpy-node binding failure → RuntimeError; every `cudaGraphExecMemcpyNodeSetParams1D` return code is checked and raises. Probe evidence that capture genuinely works on GB300: the optimized run must instantiate the graph, bind 3 memcpy nodes, patch them per-slot, and replay — all succeeded (custom metric `exec_memcpy_param_updates=3`, per-slot verification PASS, 33.83x vs the recapture baseline is consistent with graph-replay timing of ~75 µs). There is no fallback that could hide a failed capture.
- `core/utils/compile_utils.compile_model` (shared wrapper): explicitly anti-B45 — comments and code state compile failures must SKIP or raise, never silently fall back to eager. No scope lab uses torch.compile anyway.
- Harness has a `graph_capture_cheat_ratio_threshold` guard (capture-cheat detection), an additional structural defense.

## Per-lab notes / error classification

- **parameterized_cuda_graphs** (PASS, 33.83x): pinned-host request slots → device buffers → captured MLP (32x1024, fp32) → D2H. Baseline deliberately recaptures a CUDAGraph per iteration (2.55 ms); optimized patches exec memcpy-node params + replays (75 µs). Pedagogically AT-CEILING; replay path at ~75 µs for ~0.27 GFLOP + 0.5 MB H2D/D2H is latency-dominated, as intended.
- **training_hotpath** (3 targets, all PASS): see table; `metric_reduction_vectorized` optimized side runs 3 mul+sum passes materializing 3 intermediates (~126 MB traffic, 72.7 µs CUDA-event = 1.73 TB/s = ~22% SoL).
- **uma_memory**: `/proc/meminfo` + iGPU-detection helpers; consumed by ch02's `uma_memory_reporting`. Nothing to optimize here.
- **python_concurrency**: asyncio/threads/GIL teaching lab, no GPU, no harness targets; `taskrun_round1_asyncio.py` runs clean on the pod.
- **tcgen05_cluster_shapes**: BROKEN-BUILD (env/dependency, not arch): `ch10/tcgen05_warp_specialized_cutlass.cu:21: fatal error: cutlass/detail/collective/moe_stride_utils.hpp: No such file or directory` — the pod's `third_party/cutlass` checkout lacks a header the ch10 tcgen05 kernels include. This blocks the whole ch10 tcgen05 extension family, not just this lab. (sm_103a gencode itself is fine.)
- **moe_parallelism**: planner/analysis tool (CPU): `run_lab.py --scenario memory_budget` runs (needs `PYTHONPATH=.`). README is explicit: not a benchmark-pair lab. The vLLM env-preset half is unusable on this pod (no vllm).
- **vllm-deepseek-tuning**: ENV-GAP as predicted — `import vllm` → ModuleNotFoundError; lab is a Makefile/Typer matrix harness around `vllm serve`, no harness targets, results/ is startup-failure-report oriented. Classify-only, no attempt.

## Named next levers (not attempted)

1. **metric_reduction_vectorized fused single-pass** (torch-level, `vectorized_metric_reduction` in training_hotpath_common.py): compute pred_sq/target_sq/covar in ONE read of each tensor — e.g. 2x2xC gram via `einsum("anc,bnc->abc")` on stacked [pred, target], or a small Triton single-pass kernel. Traffic 126 MB → 25.2 MB; est. 72.7 → ~25 µs CUDA-event (headline ~93.6x → ~150x). Guardrail: must keep fp32 accumulation (TF32 bmm risks the 1e-4 tolerance).
2. **padding_aware packed path de-launching** (deep, do not attempt casually): persistent pre-zeroed scatter outputs (zero once in setup; padded rows never re-touched) kills the 18 memsets (~100 µs of 2.6 ms); the real lever is capturing the whole packed forward in a CUDA graph or fusing pack→GEMM→scatter, worth ~0.9 ms of launch overhead — a rewrite, not a patch. Also worth an explicit README/expectations note that on GB300 this lab is memory-goal-only (speed 0.674x is expected behavior, not a bug).
3. **segment_abs_mean residual kernel headroom**: warp-shuffle reduction + float4 loads → ~10.3 → ~3-4 µs ncu. Wall-clock-invisible (host floor 10.8 µs/call); only worth it if the lab's workload is ever scaled up (e.g. 100x segments).
4. **tcgen05_cluster_shapes unbreak**: align `third_party/cutlass` with the ch10 tcgen05 kernels' expected headers (`moe_stride_utils.hpp`) — repo dependency fix that would also revive the ch10 tcgen05 extension family on this pod.
5. **Harness microsecond-scale measurement overhead** (observation): per-iteration measured times carry ~30-55 µs of sync/bookkeeping overhead on GB300 (metric_reduction_cuda OPT: 54.6 µs harness vs 10.9 µs back-to-back CUDA-event). It is fair (both sides pay it) but compresses reported speedups for any sub-100 µs target; worth a B-series audit item if microsecond labs are ranked by harness speedup.

## Artifacts

- Runs: `artifacts/runs/20260611_072213*` (param graphs), `20260611_072444/072503/072523*` (training_hotpath pre), `20260611_073832*` + confirm (post-prototype).
- Logs: `/tmp/frontS/run_*.log`, `/tmp/frontS/tcgen05_sweep.log`, microtimer `/tmp/frontS/microtime.py`.
- Edited file (pod, uncommitted): `labs/training_hotpath/training_hotpath_kernels.cu`; backup `/tmp/frontS/training_hotpath_kernels.cu.bak`.
