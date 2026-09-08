# B200 follow-through: serving, arithmetic requirements, and profiler recovery

Status: scoped target validation complete, with no-win outcomes and an unresolved collective-replay limitation. This report extends the [September 7 results](2026-09-07-b200-review-results.md); it does not replace their retained failures or claim that every example is faster.

## Source and execution scope

DDP and the initial arithmetic matrix use `ce4681dd63740ca10a66ea3627d83a6e2d9e7a1b`;
the subsequent pipeline, cache, and FP8 checkpoint is
`aea503d9a88bd0326d93d27d5f480507febc076c`.
The final Ozaki and vLLM reruns use clean checkpoint
`2aae639f0cb10dc7ca01a1e7bbb7992f33a732fd`; their receipts bind the exact source
files and runtime used. The fresh KV envelope rerun uses clean checkpoint
`f10a5b7553825909b66b011f5c60dbb9809713b9`. Results from these checkpoints are
kept distinct below.
Execution uses one or two B200 GPUs directly, with the repository harness and an
owned, serialized process supervisor. The host reports virtualization, so these
are explicitly portable development results. They are not canonical bare-metal
qualification. CUDA, PyTorch, and the driver remain unchanged.

Receipts use the run names below under `artifacts/parallel_runs/followthrough_20260908`.
Full-output verification receipts, failed attempts, traces, and supervision receipts are retained
privately; large tensors and profiler binaries are not committed to this public repository.
The final materialized copy contains 645 files totaling 2,797,167,819 bytes.
Remote and local SHA-256 inventories are byte-identical, with inventory digest
`0ff1e4dbeb322d02360422ed4e61ff30e91ad3f2aa45851abd3967b7cb1c5e13`.

## Changes and measured disposition

| Area | Change | Current disposition |
| --- | --- | --- |
| DDP | Store sampled losses on the device and read the retained history once after training | Exact outputs pass on one and two B200s; no training-throughput win established |
| Pipeline | Amortize fill/drain boundaries across repeated fixed-weight iterations | 16 B200 runs pass exact outputs; 1.024x iteration / 1.003x process medians with mixed seed results; paired traces pass their exact mechanism contracts |
| Cache-aware inference | Remove redundant barriers for the exact 1P1D topology; report zero placement opportunities | 16 timing runs and both traces pass; observed median ratio 1.574x from reduced synchronization |
| vLLM routing and dual pools | Reuse engines; separate startup, five warmups, three wall-clock measurements, and teardown | Final dual-pool run passes exact tokens at 1.472x steady-state; dynamic routing passes exact tokens at 0.885x and is a no-win |
| TE FP8 | Expose the same explicit batch-size override in FP16 and FP8 while preserving batch 256 by default | 24 full-output observations show a workload-specific crossover: 0.718x at batch 256, 0.825x at 1024, and 1.279x at 4096 |
| Arithmetic requirements | Freeze independent numerical ceilings and require nominal, holdout, edge, and shared-reference checks | Fresh KV and Ozaki runs pass correctness; both speed goals fail without a claimed win |
| Execution hardening | Add explicit operation-placement and declared-destination write-coverage audits | Final 27 B200 tests and a real CUDA audit CLI pass, with no skips |
| Nsight Compute | Isolated newer tool versions and exactly five metrics for minimal captures | Paired 2026.2.1 selected-GEMM captures pass; application-range and coordinated collective-kernel replay time out at 180 seconds |

### DDP

The one-GPU comparison uses batch 16 and 100 training steps. The two-GPU
comparison uses batch 32 per rank and all 32 steps available from its loader.
Each topology has two fresh seeds and an ABBA order, yielding four observations
per arm. Full inputs and final outputs match exactly, and worker runtime receipts
match the workers that produced those outputs.

| Scope | Control median | Candidate median | Control / candidate |
| --- | ---: | ---: | ---: |
| One-GPU training iteration | 46.9307 ms | 46.7522 ms | 1.0038x |
| One-GPU process wall time | 13,985.3 ms | 14,012.4 ms | 0.9981x |
| Two-GPU training iteration | 84.6293 ms | 84.7961 ms | 0.9980x |
| Two-GPU process wall time | 17,533.9 ms | 17,211.8 ms | 1.0187x |

These are observed ratios, not a qualified speedup. The one-GPU control includes
a large cold-start outlier; the worker-loop results are near parity. Deferring
logging does not make its cost disappear from process wall time. The early
two-GPU drivers incorrectly expected 100 and then 64 steps; both were rejected
when actual execution reported 32, retained, and replaced by a fresh 32-step plan.
Receipts: `ddp-progress-abba/receipt.json` (completed one-GPU rows; later driver
failure retained) and `ddp-progress-abba-w2-32steps/receipt.json`.

### Pipeline and cache-aware inference

The pipeline comparison completed four seeds in ABBA order: 16 observations,
complete two-rank runtime receipts, and bitwise-equal full outputs. Median
iteration times were 15.5584 ms versus 15.1887 ms, while process wall times were
8,514.49 ms versus 8,488.87 ms. Per-seed iteration ratios ranged from 0.967x
to 1.071x. The improvement is too small and inconsistent to claim a robust win.
The change applies to this synthetic fixed-weight repeated workload; it cannot
combine iterations separated by real optimizer updates.

The baseline Nsys capture initially failed during table export because its
SQLite cache was older than the report. A separate recovery directory preserves
the original failure and uses the supported `--force-export=true` option. The
recovered baseline and fresh optimized traces both pass their exact operation-count
contracts: each contains 1,024 GEMMs, 1,024 ReLUs, and 128 NCCL sends plus 128
receives. The optimized trace reduces P2P kernels from 256 to 132 and total GPU
operations in the two measured ranges from 864 to 818. These counts explain the
mechanism; the separate ABBA timing remains the performance evidence. Receipts:
`pipeline-contiguous-abba/receipt.json`,
`pipeline-contiguous-nsys-baseline/recovery-force-export-v1/recovery-receipt.json`,
and `pipeline-contiguous-nsys-optimized/receipt.json`.

Cache-aware inference completed 16 timing runs across four seeds and two matched
Nsys captures, with complete outputs, worker/runtime PID parity, and per-rank
application-clock checks. Baseline and optimized medians were 14.0537 ms and
8.9308 ms. All four block ratios favored the candidate, ranging from 1.441x
to 1.668x. Baseline/optimized standard deviations were 1.1614/0.2731 ms.

The traces show float32 NCCL all-reduce launches dropping from 1,328 to 176 and
`cudaStreamSynchronize` calls from 1,424 to 272, while send/receive kernel counts
remain 660 in both arms. The reduction of 1,152 matches eight avoided barriers
per request across eight requests, nine warmup/measured invocations, and two
ranks. Both arms report the same cache hit rate and transfer volume. There is
only one decode rank, so neither arm has an alternative cache placement.
Receipt: `cache-aware-locked-abba/receipt.json` reports
`FULL_OUTPUT_RUNTIME_CLOCK_ABBA_NSYS_PASS` with 18 rows.

### FP8 training

Fresh matched batch-256 Nsys captures show the optimizer's four tensor-update
kernels taking about 106–107 microseconds in each arm. The FP8 capture adds
approximately 76 microseconds of quantization and scale-update kernels while
its GEMMs become shorter. A subsequent sweep used the full 67,121,152-parameter
model, compared the actual prediction and every post-step parameter, and passed the
frozen output policy for all 24 observations: two seeds, two repeats, two arms,
and three matched batches.

| Batch | Eager FP16 median | Eager TE FP8 median | FP16 / FP8 | Disposition |
| ---: | ---: | ---: | ---: | --- |
| 256 | 0.4786 ms | 0.6662 ms | 0.7183x | No measured speedup |
| 1,024 | 0.5483 ms | 0.6644 ms | 0.8252x | No measured speedup |
| 4,096 | 1.2303 ms | 0.9616 ms | 1.2795x | Candidate speedup for this workload |

The primary timing is one complete CUDA-event training update after five setup
and ten warmup updates; whole-call time is retained separately. Batch 256 remains
the default, so the 4,096 result establishes a workload-specific crossover rather
than a general default-workload speedup. Matched batch-4,096 Nsys captures also
pass for both arms. They show shorter main FP8 GEMMs together with quantization
and scale-update work; their single profiled update is diagnostic and is not used
as benchmark timing. Receipts: `te218-matched-batch-sweep-v2-receipt.json` and
`te218-batch4096-profiles-receipt.json`.

The first private sweep driver rejected an actual baseline signature because
the driver omitted the mixin's output shape and dtype from its expected fields.
The failed receipt is retained. The driver was corrected to require those fields;
the benchmark and all numerical tolerances remain unchanged.

### Reused serving engines

Three ordinary-harness runs of `labs/dynamic_router:dual_pool_vllm`
passed exact token-output checks and runtime parity. Each arm starts two engines
once, runs five warmups, then measures three complete 102-request workloads.
Prefix caching is disabled and request state resets between invocations.

| Run | Shared-pool wall time | Dual-pool wall time | Observed ratio |
| --- | ---: | ---: | ---: |
| 1 | 1,870.687 ms | 1,280.290 ms | 1.461x |
| 2 | 1,872.504 ms | 1,273.115 ms | 1.471x |
| Final source | 1,866.333 ms | 1,268.244 ms | 1.472x |

The second run reports 54.47 versus 80.12 requests/second. Startup is reported
separately at 27.21–27.64 seconds, and final lifecycle receipts retain 1.91–2.64
seconds of teardown. Including startup, warmups, measured requests, bookkeeping,
and teardown, shared-pool sessions took 47.32 and 47.11 seconds; dual-pool
sessions took 44.89 and 44.70 seconds. The steady-state ratio must not be applied
to those total session durations. All four lifecycle receipts for the first two
runs report completed disposition, no failed requests, and no shutdown errors.

The matched final-source Nsys pair passes full token equality (1,734 integers),
runtime parity, and the five-warmup/one-steady lifecycle contract. Baseline puts
all 19,222 captured kernels on device 0. The dual-pool arm splits 15,550 kernels
onto device 0 and 12,178 onto device 1; kernel-busy interval union falls from
1.812668 to 1.218329 seconds, and 99.0% of device 1 busy time overlaps device 0.
Summed GPU duration still rises 5.3% and launches rise 44.3%, while MoE BMM time
is within 1.8% and attention within 0.5%. This supports concurrent placement,
not removed arithmetic. Profiled durations are diagnostic; the repeated ordinary
runs above remain the timing evidence.

The reused dynamic-routing pair also passes exact token checks and lifecycle
separation, but is slower: 209.280 ms for the static control versus 236.455 ms
for dynamic routing, or 0.885x. That run is retained as `failed_no_speedup`, not
as a routing win. Receipts: `vllm-dual-pool-reuse-{1,2}`, their adjacent
`-lifecycle-final.json` extracts, `vllm-dual-pool-final-2aae`,
`vllm-dual-pool-nsys-2aae/{baseline,optimized}/receipt.json`, and
`vllm-dynamic-routing-reuse-2aae`.

### Numerical acceptance

The policies were fixed before the qualification runs and reject widened JSON
limits. The retained 10-case matrices passed one nominal case, two holdouts, and
two edge cases for each variant on the recorded B200 stack. The later Ozaki
ordinary-harness run also passes full correctness for both variants, while its
speed goal correctly fails: dynamic reaches 0.545x and fixed reaches 0.724x
relative to native FP64. These are no-speedup results.

| Variant | Relative-L2 ceiling | Maximum error / maximum reference ceiling |
| --- | ---: | ---: |
| KV FP8 E4M3 | 0.0625 | 0.0625 |
| KV NVFP4 E2M1 | 0.25 | 0.25 |
| Ozaki dynamic, max 16 bits, offset -56 | 0.03125 | 0.0625 |
| Ozaki fixed 12 bits | 0.000244140625 | 0.00048828125 |

KV checks every stored cache element against an unquantized BF16 PyTorch
projection. Ozaki compares complete production-size arrays with native FP64,
and the small rectangular edges additionally use a CPU long-double reference.
The x86 host ran that independent reference successfully. The limits are explicit
engineering requirements, not arbitrary-matrix error theorems or application-quality
guarantees. See the [KV requirements](../../code/labs/kv_cache_compression/ACCURACY_REQUIREMENTS.md)
and [Ozaki requirements](../../code/labs/ozaki_scheme/ACCURACY_REQUIREMENTS.md).
Receipts: `accuracy-matrix/kv-qualification.json`,
`accuracy-matrix/ozaki-qualification.json`, and `ozaki-frozen-policy-2aae`.
The updated assessors also accept all 20 retained cases while preserving their
original declared provenance. This is a reassessment of those receipts, not a
claim that their kernels reran at the later source checkpoint.

The earlier ordinary KV pair failed its secondary raw `allclose` criterion even
though both independent gates passed. Near cancellation, that local relative
criterion is inconsistent with the declared reference-normalized limits. The
replacement follows the triangle inequality: the full raw pairwise difference
must stay within `(0.0625 + 0.25) * max(abs(reference))`, with exactly matching
per-output maps in both arms. Both independent L2 and maximum-error gates remain
mandatory and unchanged. The original failed receipt remains failed. Ozaki's
ordinary runner similarly needed the baseline to expose the already-declared
secondary comparison envelope; its full-array requirements were unchanged.

The fresh KV ordinary-harness run passes input, full-output, pairwise-reference,
and runtime checks. Its shared reference has maximum magnitude 17.75; the frozen
0.3125 coefficient gives an absolute envelope of 5.546875, and the observed raw
maximum difference is 2.8125. FP8 reaches maximum relative-L2 0.040992 and
normalized-maximum error 0.044118; NVFP4 reaches 0.146155 and 0.159467, all under
the unchanged ceilings above. Baseline and optimized times are 557.582 and
544.132 ms, a 1.0247x ratio below the required 1.05x threshold. The receipt
`kv-cache-reference-envelope-final` therefore retains `failed_no_speedup` while
its correctness gates pass.

### Practical hardening

The new standalone `python -m core.harness.execution_audit` tool uses a fresh
benchmark instance outside normal timing. It observes dispatcher-visible tensor
operations and can poison explicitly declared floating-point or complex output
buffers to check that their complete logical extent was written. It records
operation/device evidence and rejects replaced destination identities.

The placement audit covers the current Python thread's PyTorch dispatcher.
Destination coverage applies to the exact declared contiguous buffers. Neither
claims to inspect arbitrary extension internals, other processes, or all
uninitialized-memory provenance. Nine stale declarations now execute real checks,
reducing the explicit missing-protection declarations from 42 to 33; those counts
are declarations, not distinct confirmed bugs.
The audit also rejects no-op and allowed-host-only callbacks that provide no
evidence of execution on the requested device, and preserves a primary failure
when teardown itself fails.

### Nsight recovery

Nsight Compute 2026.1.1 and 2026.2.1 were unpacked into private tool directories
after checking NVIDIA package sizes and SHA-256 digests. No system package hooks,
driver update, or shared CUDA/PyTorch change was used.

The old `minimal` automation path also selected NVIDIA's `basic` sections, adding
many counters beyond the requested list. The shared automation and harness now
request exactly the five minimal metrics; explicit `basic` still selects NVIDIA
sections. CLI/MCP descriptions and regression coverage agree with this behavior.

On 2026.2.1, a selected GEMM on device 0 completed with both pipeline ranks and
exactly five requested metrics in each arm. The baseline and optimized kernels
reported 164,160 and 162,848 ns, 91.62% and 92.04% SM throughput, and 15.41% and
15.52% DRAM throughput. Both receipts include the two executed rank PIDs and their
runtime provenance. This is one selected kernel's mechanism evidence, not a
complete-workload timing comparison. The earlier 2026.1.1 parser omitted the L2
counter prefix; a separate retained-artifact validation corrected that parser
error without replacing the original receipt.

Range capture rejected `cuThreadExchangeStreamCaptureMode`; on 2026.2.1 both an
application-range replay and a shared-memory NCCL attempt without NVTX timed out
after 180 seconds. Each timed-out capture was preserved and fully drained.
NVIDIA documents mandatory concurrent-kernel
coordination and replay constraints in its [Nsight Compute CLI guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#mandatory-concurrent-kernels).
The selected-kernel path is recovered; full-range replay remains unresolved.

## Checks and remaining limitations

- Final affected CPU source integration: 913 passed, 103 explicit capability or missing-declaration skips, and 30 warnings across 17 modules.
- Profiler command, CLI, MCP-document, and harness contracts: 132 passed, 1 skip.
- KV reference-envelope compatibility: 57 passed, 1 capability skip.
- Earlier B200 execution guards and deferred-progress behavior: 23 passed, no skips.
- Late-source B200 audits: 27 passed with no skips, and the real vectorization CLI passes at `f10a5b755`.
- Retained B200 arithmetic matrix: 20 of 20 cases pass; fresh KV and Ozaki ordinary-harness correctness passes with no qualified speedup.
- Syntax passes for all 48 changed Python files. Full-file Ruff has the same 75 diagnostics as the base commit and no new diagnostics; focused changed-implementation lint passes.
- All target runs are finished and their owned processes drained; artifact hash verification passes.

The first hosted CPU validation attempt reached 98% before its 30-minute job
budget expired. Its cancelled result is retained. The workflow now allows 35
minutes, preserving the entire test suite and all final audits. A prior run had
also exhausted 30 minutes after its full suite and linter passed; the added
margin addresses observed hosted-runner variability rather than skipping work.

Full collective NCU replay still needs a working tool/runtime combination; the
bounded selected-kernel captures and matched Nsys traces provide the usable
profiling paths on this stack. DDP and pipeline need a repeatable measured gain
before stronger performance claims. Small-batch FP8, dynamic routing, KV, and
Ozaki remain no-win examples for the measured workloads. The arithmetic budgets
cover these lab outputs; downstream model-quality acceptance remains a separate
requirement. The remaining 33 protection declarations require individual review,
not treatment as 33 confirmed bugs.
