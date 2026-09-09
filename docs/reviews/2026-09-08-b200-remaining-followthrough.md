# B200 follow-through: fair serving, training fixes, and NCCL capture

**Repository-wide validation remains incomplete.** The current objective is to
improve every identified area and validate each supported example against its
purpose: throughput, latency, memory savings, or scaling. Following the user's
updated direction, a universal minimum 1.05x speedup is no longer a completion
requirement. Measured tradeoffs and no-win results remain explicit; correctness,
execution coverage, and material memory/profiler failures still require work.

The earlier targeted B200 pass completed. Real-data training checks
pass on one and two B200s, the supported NCCL profiling path completes, and the
serving comparison now uses both GPUs fairly. Dedicated pools are **22.71%
slower** on total completion time for this workload, while improving short-request
time to first token. That tradeoff replaces the earlier biased speedup claim.

This pass extends the [previous results](2026-09-08-b200-followthrough-results.md).
It preserves their numerical requirements, successes, no-win results, and failed
attempts. It does not establish that every repository example is faster.

The retained targeted measurements illustrate the remaining speed gaps:

| Comparison | Recorded ratio | Disposition |
| --- | ---: | --- |
| Cache-aware 1P1D | 1.574x | Measured improvement for the recorded workload |
| FP8 training, batch 4096 | 1.279x | Improvement at this batch; smaller batches lose |
| Regular DDP training loop | Approximately 1.00x; latest private one-GPU candidate 1.035–1.038x | Modest candidate improvement; not promoted; two-GPU candidate validation pending |
| Pipeline parallelism, forward lookahead | 1.057x median ratio | Two of four blocks remain below 1.05x; process duration does not improve |
| Dynamic serving routing | 0.996x | Parity on the homogeneous workload; no throughput benefit established |
| Fair dedicated versus shared serving pools | 0.815x | Lower total throughput; short-request TTFT improves |
| KV-cache NVFP4 compute, cached weights | 1.106x | All eight cached comparisons across four A/B/B/A blocks exceed 1.05x; full profiled pair also passes |
| Ozaki dynamic / fixed versus native FP64 | 5.793x / 7.762x | Corrected public timings, independent arithmetic gates, and all-variant Nsight captures pass; scoped to the recorded workload |

These ratios retain the source, workload, timing, and qualification limits of
their individual receipts; they are not a new uniform benchmark run. The
pipeline lookahead change has mixed repeated GPU results, detailed below. The newer KV result
uses a different Transformer Engine version from the earlier 1.025x result;
the fresh old/new comparisons below use the same runtime to isolate the cache
change. Broad CI remains paused during this work.

Static discovery at `bc0ed66cd317d9d3cc3be982fe22d333ebdf68e0` finds 488 logical
baseline targets and 543 optimized entries, representing 466 unique file pairs.
Aliases can share source files while selecting different workloads. Discovery
counts do not establish execution or performance coverage, and a target's best
variant succeeding does not establish that every variant succeeds. Memory-goal
pairs also require their declared memory-saving gate; passing that gate does
not imply a 1.05x speedup. The private `performance-gap-inventory/` preserves
the source inventory and a separately scoped historical-result audit.

## Current improvement priorities

Continue bounded DDP and pipeline tuning where traces identify removable work,
then retain the measured result even if it is modest. Additional implementation
complexity must be justified by repeatable benefits on the intended workload.
One-GPU optimizer experiments do not replace the outstanding two-GPU checks.

For serving, compare shared and dedicated placement using both aggregate
throughput and short/long-request latency. The dedicated-pool tail imbalance is
a concrete next experiment; a latency benefit must retain its throughput cost.
Dynamic routing needs variable arrivals or load imbalance to assess its intended
benefit, while preserving the existing homogeneous parity result. Reuse engines
and report startup separately from steady-state requests.

For FP8, retain the small-batch regressions and the measured large-batch win as a
size-dependent crossover. Do not change the default or hide slower cases to
manufacture a universal gain. Preserve the existing cache-aware, KV-compression,
and Ozaki improvements with their workload and independent accuracy limits.

Do not waive incorrect training objectives, unresolved numerical acceptance,
uninitialized-memory reports, or missing supported-path execution. Historical
references below to the 1.05x requirement describe the original experiment goal;
they are not the current repository-wide completion gate. Broad CI remains
deferred until the implementation and runtime work is finished.

## Source and execution scope

The base is merged commit `0299307facf77bc883b9d27b4fb175ce27e50ab4`.
The regular DDP and initial routing measurements use `d289e873e`; the fair serving
and public NCCL captures use `e864b377c`. Final real-MRPC training and deferred-loss
CUDA tests use `e1f4f7e7516f58bb693ab2949a966e25d008a6be`. Subsequent CI repairs are recorded below;
serving and training workload source remains unchanged.

Runs execute directly on one or two B200s, without Slurm. The host reports
virtualization, so results are portable development evidence rather than
canonical bare-metal qualification. GPU work is serialized through owned process
supervisors. The completed stages drained naturally. No unrelated processes were
interrupted or other tasks contacted.

The normal CUDA 13.0, Torch 2.9.1+cu130, driver 580.173.02, permissions, and
credentials remain unchanged. Real MRPC and FlashAttention 2 dependencies are in
an isolated task overlay. Nsight Compute 2026.2.1 is also task-local; the system
profiler is unchanged.

## Fixes and validation

| Change | Result |
| --- | --- |
| Reuse one tokenizer in regular, FlashAttention, and compression workers | Removes a redundant construction on every rank; default dataset preparation still works without a supplied tokenizer |
| Use the existing accumulation helper in both optimized FlashAttention mains | Backward stays inside `no_sync`; partial groups use their actual divisor and perform their optimizer update |
| Avoid static-graph/no-sync incompatibility in the pinned PyTorch version | Two-rank partial accumulation completes with the expected synchronization sequence |
| Truncate real tokens before fixed-length collation | Real MRPC sequences longer than 128 tokens no longer cause mismatched tensor shapes; padding/truncation sides and aligned token fields are preserved |
| Log the original loss in four optimized DDP paths | Reported loss no longer changes merely because the accumulation divisor changes |
| Count successful admissions before choosing the next serving GPU | Shared serving distributes 102 requests 51/51; dynamic routing distributes 16 requests 8/8 |
| Remove polling that occurs only after all admissions and the fixed 10 ms serving-loop sleep | Removes unnecessary host work from both comparison arms; deferred engine execution still yields |
| Add the coordinated `core.profiling.ncu_torchrun_capture` CLI | Baseline and optimized two-rank NCCL captures complete with finite counters and natural cleanup |
| Replace four stale missing-protection declarations with behavioral tests | All four pass on actual B200s; the remaining 29 declarations are an inventory, not 29 distinct confirmed bugs |

### Actual training entrypoints and partial accumulation

All five remaining mains pass with a real TinyLlama model, the real cached
GLUE/MRPC dataset, one reused tokenizer per rank, finite losses, and the intended
eager or FlashAttention 2 implementation. These checks do not use the synthetic
dataset fallback.

| Entry point | GPUs | Native microbatches | Result |
| --- | ---: | ---: | --- |
| `baseline_ddp_flash` | 1 | 2 | PASS |
| `optimized_ddp_flash` | 1 | 2 | PASS |
| `baseline_ddp_flash_multigpu` | 2 | 2 per rank | PASS |
| `optimized_ddp_flash_multigpu` | 2 | 2 per rank | PASS |
| `ddp_compression` with compression disabled | 2 | 2 per rank | PASS |
| Optimized FlashAttention partial group | 1 | 3, accumulation 2 | PASS; 2 optimizer updates |
| Optimized FlashAttention partial group | 2 | 3 per rank, accumulation 2 | PASS; 2 optimizer updates per rank |

The two-rank partial-group run records backward synchronization as
`[false, true, true]`, one `no_sync` enter/exit pair, and static graph disabled.
Each selected-scope receipt retains its original pending-counterpart label;
the aggregate final audit checks the union of all five mains and both partial-group
runs at the same source. These are functional smokes, not a full training
convergence or throughput qualification.

Focused verification includes 26 collator/training tests with real offline
Hugging Face tokenizers, seven real CPU Gloo/reference accumulation tests, and eight real-Torch
loss-logging cases. Final deferred-metric tests pass **3/3 on B200**, including the
CUDA test: loss values stay on device until the single final host transfer.
The earlier combined regression run passed 245 tests with 84 macOS skips. These
are separate, potentially overlapping test selections, not additive unique totals.

Earlier failures remain retained: missing FA2 metadata, an overly strict private
overlay preflight that stopped before GPU launch, and the real-MRPC 131-versus-128
collation failure. The final runs use the corrected overlay, driver, and source.
The overlay preserves the normal environment and checks the core runtime identity;
FA2 intentionally supplies its overlapping module namespace only within that overlay.

### Regular DDP startup versus training

Sixteen ABBA observations across two seeds and one/two B200s passed 12 full-output
comparisons plus tokenizer diagnostics. Construction count falls from two to one
per rank. Training-loop control/candidate median ratios are **0.9982x / 0.9948x**:
no iteration-throughput win. Whole-process median ratios are **1.0080x / 1.0358x**.
The latter include setup and teardown, so they are not pure tokenizer timings.

Both two-GPU seed-balanced blocks favor the candidate process by about 619–694 ms,
and all four adjacent two-GPU comparisons favor it. One-GPU observations include
an adjacent reversal and an effect smaller than their scatter. Another independent
batch would be needed to establish a stable startup benefit of a particular size.
Those regular-DDP logs explicitly use the documented synthetic dataset fallback;
the separate real-MRPC checks above establish execution, not real-MRPC speed.

## Fair serving performance

Engines are constructed once and reused. Startup, five warmup batches, three
steady-state measurements, and teardown remain separate. Both layouts run the
same six long prompts of 4096 tokens and 96 short prompts of 128 tokens, generating
16 tokens per request. Full input/output, runtime, admissions, and lifecycle checks
pass in every pair.

| Repeat | Shared completion ms | Dedicated completion ms | Shared / dedicated |
| --- | ---: | ---: | ---: |
| 1 | 1032.122 | 1249.224 | 0.826210x |
| 2 | 1022.758 | 1256.390 | 0.814045x |
| 3 | 1024.442 | 1257.126 | 0.814908x |

The median paired ratio is **0.814908x**. Median throughput is **99.566 versus
81.185 requests/s**. Each raw CLI exit remains 1 for its failed speed goal;
reviewed disposition is `VALID_NO_WIN`, with correctness and runtime checks passing.
The earlier 1.4753x result used a shared baseline with all 102 requests on GPU 0.
It remains a historical measurement and is superseded for fair speedup claims.

Dedicated pools make a latency tradeoff:

| Request class | Shared TTFT ms | Dedicated TTFT ms | Change |
| --- | ---: | ---: | ---: |
| Short p50 | 698.669 | 357.932 | -48.77% |
| Short p95 | 776.118 | 619.340 | -20.20% |
| Long p50 | 450.276 | 718.068 | +59.47% |
| Long p95 | 624.475 | 1068.294 | +71.07% |

These are medians of three reported per-run percentiles, not pooled percentiles.
Request classes are comparable across layouts. TTFT is observed after each engine
step; per-class full-completion distributions are not available. The legacy
`tpot_tok_per_step_gpuN` metric is tokens per poll step, not time per output token;
the generic token-throughput field includes prompt and generated tokens.

Matched Nsight Systems captures identify the imbalance. Shared GPUs receive 51
mixed requests each and finish their last kernels 2.142 ms apart. Dedicated GPU 0
receives six long requests and continues **525.224 ms after GPU 1's last kernel**,
including 461.668 ms of additional GPU 0 kernel activity. Simultaneous GPU kernel
activity falls from 936.685 to 708.019 ms. BMM dominates both traces; dedicated
attention work is 203.702 ms on GPU 0 versus 15.352 ms on GPU 1.

Fixed long/short eligibility therefore leaves one GPU unable to help with the
long-request tail. This mechanism is inferred from one trace pair; ordinary runs
above establish the slowdown. Trace ranges exclude startup, and their CPU-only
SQLite export preserves both original report hashes.

Dynamic routing separately passes three full-output/runtime/lifecycle pairs with
8/8 admissions and a **0.995772x** median ratio. Its homogeneous upfront batch
provides no live-feedback placement advantage. Both Nsight Systems captures pass;
this remains parity and a failed 1.05x speed goal.

## Profiler recovery and limits

The checked-in [coordinated CLI](../tooling-and-profiling.md) runs one Nsight Compute
2026.2.1 process per rank with TCP coordination and kernel replay. It selects the
observed lockstep `NCCL@ncclGroupEnd/` and `NCCL@ncclAllReduce/` ranges. With
`--all-matching-kernels`, both public-entrypoint captures pass at `e864b377c`:

| Arm | Send/receive per rank | All-reduce per rank | Total launches, both ranks | Finite requested counter cells |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 128 | 1 | 258 | 1290/1290 |
| Optimized | 66 | 1 | 134 | 670/670 |

All 392 launches have the expected rank/device/kernel identities and all five
requested finite counters. Source, commands, helpers, tools, reports, and runtime
are bound by retained receipts. Linux ownership tracking spans detached rank
sessions; forced or unverified cleanup cannot pass. All 18 helper tests pass on
Linux, and both actual captures complete with no surviving owned processes.

Wrapper durations of 118.187 and 67.234 seconds include instrumentation/replay and
are not performance speedups. The capture excludes non-NCCL compute. Older
180-second application/range replay timeouts remain failures; the narrower supported
route now supplies the requested NCCL evidence. Unbatched-P2P initialization
warnings remain in both arms.

The NCU commands use `--clock-control none`; retained logs warn about unmodified
GPU clocks. The reviewed receipts do not establish applied application-clock
telemetry. These results qualify capture, counters, and runtime parity, not
publish-grade timing or clock control.

## Remaining guidance and evidence

Use shared serving for this workload's throughput goal. Choose dedicated pools
only with an explicit short-request latency objective and the measured long-request
cost. A future placement experiment could allow both GPUs to take long requests;
its benefit is unmeasured. Do not add KV migration based on this trace alone.

The previous pass's independently checked results remain: cache-aware 1P1D
synchronization removal measures **1.574x**; FP8 crosses over at batch 4096
(**1.279x**) while smaller batches lose; pipeline timing has no robust win.
Keep the default FP8 batch unchanged and select workload sizes explicitly.
KV compression and Ozaki retain their independent numerical ceilings. Their
newer repeated timings and completed profiler captures are recorded below;
the earlier no-win and failed-profile receipts remain preserved with their
original dispositions. The Ozaki parser correction reveals an existing speedup;
it does not accelerate the CUDA kernels.

Hardening coverage is deliberately specific. Current-thread dispatcher checks see
visible PyTorch operations, not arbitrary native/background execution. Device
identity checks detect drift at a boundary, not a switch-and-restore between
boundaries. Prioritize unexecuted paths and material memory-write gaps from the
remaining inventory rather than treating every skipped declaration as a bug.

Raw timing results, full outputs, profiler binaries, failed attempts, supervision,
source/tool manifests, and reviewed analyses are retained privately in
`/Users/admin/.codex/artifacts/ai-perf-followthrough2-20260908/`.
The final remote transfer verification covers **399 files / 1,179,540,870 bytes**
with zero size or SHA-256 mismatches; inventory digest is
`28ff674a32e882756f6877535cee1afcf4ffc759b83aa81a444d06edaf9c616a`.
The separate initial 208-file snapshot remains byte-exact and immutable.
`final-evidence-inventory.json` inventories the complete materialized package,
including review support; `final-transfer-verification.json` records remote custody.

Receipts are grouped under `validation/` for serving, profiling, and supervisors;
`training-validation/` for actual mains and partial groups; `training-provisioning/`
for isolated dependency evidence; and `review-support/` for reproducible readers,
plans, validators, and the aggregate `final-training-audit/` closure. Executed
DDP ABBA driver identity is retained separately from its unexecuted draft.

CI is reserved for the completed implementation, GPU validation, and report.
Read final CI status and the exact tested commit from
[PR #28](https://github.com/cfregly/ai-performance-engineering/pull/28); this GPU
evidence does not substitute for that integration check. The earlier CI run was cancelled as superseded when more
callers were found; its disposition remains retained.

## Repairs from the first final-integration attempt

The first integration run on `94bf95104` completed with **5,561 passed, 527
skipped, and nine failures**. Static analysis, dashboard checks, core contracts,
and all four configured CUDA architecture builds passed. The failures map to
four causes: one stale loss-logging assertion, three process-cleanup assertions,
one generated-README mismatch, and four missing-Transformers import failures.

The repairs preserve the new unscaled-loss behavior, add the pinned real
Transformers dependency to CPU CI, and synchronize the README generator with
the documented routing and accumulation behavior. All 61 generated READMEs now
match their canonical generator. The real-HF collator selection passes 14 tests,
the related Transformer selection passes 21, and repository configuration
passes 24. These are overlapping focused selections.

Profiler cleanup now snapshots PID/start-time identities before launch. It
ignores an unreadable process only when that exact identity predates the capture.
Owned, newborn, or PID-reused unreadable identities remain unverified and are
not directly signaled. This avoids rejecting captures because of an unrelated
pre-existing process while retaining conservative ownership checks. The repaired
helper passes 21 Linux CPU tests, including real detached-process cleanup. The
public receipt schema and profiler command construction are unchanged. The B200
NCU captures above remain evidence for their recorded `e864b377c` source; a fresh
GPU capture of this cleanup repair has not been claimed.

The failed integration receipt and focused repair checks are retained separately
in `ai-perf-followthrough2-20260908-final-integration/`, alongside the original
sealed GPU package. No successful full-suite rerun of these repairs is asserted
here; use the PR's actual run results.

## Further optimization work

The renewed review has produced two candidates without changing workload sizes
or accuracy limits. Pipeline commit `7dc0133d84ebbf1e2abd7d7486a109d942bee86a`
queues one independent rank-zero forward microbatch before waiting for paired
transfers. Its baseline GPipe function is byte-identical. All 88 focused CPU
checks pass, including real two-rank output comparisons; a separate three-rank
Gloo comparison also passes. Actual NCCL overlap and a repeated two-B200 speedup
remain unmeasured. The first two-B200 attempt passed baseline execution and then
failed in optimized warmup: a new check incorrectly required one completion
handle per P2P operation, while NCCL coalesces the pair into one handle. The fix
accepts the coalesced handle and still waits every returned handle, rejecting an
empty group. All 21 focused scheduling/protocol checks pass, including real
two-rank CPU output comparisons and separate coalesced/uncoalesced control-flow
cases. The failed GPU attempt drained naturally; eight files (63,976 bytes) are
retained in `pipeline-c9-failure/`, with inventory SHA-256
`8b4dda593d35ee1e8d59190df5f81690878eabe4cb4db801a03e754622e248d6`.
Corrected source `d8a533ff00e215c19dae41fdcb6be0e7bfdf6b1f` now completes
**all 16 two-B200 executions and all eight exact full-output comparisons**.
Baseline/optimized median rank-zero iteration times are **15.7713 / 14.9237 ms**
(**1.056794x**); sample standard deviations are 0.3125 / 0.4499 ms. The four
block ratios are **1.058926, 1.019520, 1.063420, and 1.021559x**, so only two
blocks exceed 1.05x. Median whole-process ratio is **0.985690x**. Preserve these
mixed results instead of treating the aggregate as universal success.

Both Nsight Systems captures pass the unchanged mechanism checks: 1,024 GEMMs,
1,024 ReLU kernels, and 128 send/receive calls per arm; P2P kernels decrease
from 256 to 132. Within the measured ranges, both devices retain 192 GEMMs each.
CUDA-launch attribution shows **12.7686 ms of rank-zero GEMM/P2P overlap** for
the optimized run versus zero for GPipe; rank one still has no such overlap.
These are instrumented interval measurements, not ordinary benchmark timings.
The driver uses harness-managed clock locking for both devices, and the batch
drains naturally. All 57 files (26,422,116 bytes) were copied and hash-verified
in `pipeline-nccl-d8/`, inventory SHA-256
`4ba3415a9e098986dafe648687890855b22b1efab775dca4505d651a284db43d`.
Further improvement must address the remaining rank-one communication exposure
and measurement spread without reducing work or weakening exact verification.

The repaired public `core.profiling.ncu_torchrun_capture` entrypoint also passes
fresh two-rank captures at the same source, using Nsight Compute 2026.2.1.
Baseline captures contain 129 selected NCCL kernels per rank; optimized captures
contain 67 per rank. All four reports contain the five required finite counters
for every selected kernel, with rank/device/process and runtime checks passing.
Both captures drain naturally, without forced cleanup or timeouts; capture
durations of 119.076 and 68.009 seconds are instrumentation overhead, not speed
measurements. The selected scope includes all matching NCCL kernels and excludes
non-NCCL compute. All 31 retained files (541,915,451 bytes) are hash-verified in
`pipeline-public-ncu-d8/`, inventory SHA-256
`4d2f876e51ccd5ec85ad2b8c087af6a6ea69194725627df78d0c71111413e17c`.

KV commit `32030e6cf31b6058f868988b13a317b5f96af3dc` refreshes packed projection
weights on the first group of each complete iteration and reuses them for the
remaining 129 groups. Both FP8 and NVFP4 arms use the same caching policy. Every
forward, activation quantization, attention operation, and full-cache write is
retained. Focused checks pass 81 tests with one existing skip.

The new KV source also passes **all ten existing B200 arithmetic qualification
cases**, with the unchanged independent limits and full-cache reference checks.
The run used Torch 2.9.1+cu130 and Transformer Engine 2.9.0+70f5366 on one isolated
B200, while an unrelated workload occupied the other device. It drained naturally
without forced cleanup. All 27 transferred files / 24,953 bytes match their
remote SHA-256 inventory. These results establish this lab's arithmetic gate;
they do not establish attention/model/task quality or a speedup. Evidence is in
`ai-perf-followthrough2-20260908-final-integration/kv-weight-cache-32030e6/`.

An ordinary harness pair at the same KV source and runtime measures **542.969 ms
FP8 versus 490.734 ms NVFP4, or 1.106445x**, with 20 timed iterations and five
warmups, harness application clocks of 1500/3996 MHz, and the explicit portable
validity profile. Full inputs, full outputs, runtime parity, and all Nsight
Systems, Nsight Compute, and PyTorch captures pass. This is one pair on one
isolated B200; separate unprofiled repetitions are reported below. The 35 retained
files / 100,745,932 bytes match their remote SHA-256 inventory in
`ai-perf-followthrough2-20260908-final-integration/kv-pair-app-range-r2-32030e6/`.

The fresh unprofiled A/B/B/A screen compares uncached `e1f4f7e75` against cached
`32030e6` on that same Torch/Transformer Engine stack, seed 42, workload,
20-iteration/five-warmup policy, UUID-selected device, and application clocks:

| Order | Source | FP8 ms | NVFP4 ms | FP8 / NVFP4 |
| --- | --- | ---: | ---: | ---: |
| A1 | Uncached | 606.933 | 620.311 | 0.978433x |
| B1 | Cached | 542.835 | 490.726 | 1.106186x |
| B2 | Cached | 543.036 | 490.814 | 1.106399x |
| A2 | Uncached | 607.026 | 620.519 | 0.978255x |

Both mirrored cache comparisons exceed 1.05x in each precision. Their geometric
means are **1.117958x for FP8** and **1.264166x for NVFP4**. The two cached
FP8/NVFP4 ratios average **1.106293x**. Every full-output, independent error,
input, runtime, source, storage, and clock gate passes. Both uncached benchmark
exits retain `failed_no_speedup`; they are valid measurements, not speed passes.
The complete four-process batch drained naturally. All 47 files / 625,329 bytes
match their remote inventory, and a local re-read independently reapplies the
validators and recalculates the comparisons. Evidence is in
`ai-perf-followthrough2-20260908-final-integration/kv-abba-v4-32030e6/`.

All four planned A/B/B/A blocks have now completed, for **16 process runs and
eight mirrored comparisons**, using seed 42 on the same B200 and runtime.
Every accuracy, full-output, source, runtime, storage, and clock check passes
when reapplied to the retained JSON. The eight cached ratios are 1.106186,
1.106399, 1.106124, 1.106006, 1.106293, 1.106128, 1.106340, and 1.106174x.
All exceed 1.05x; their mean is **1.106206x**, range **1.106006–1.106399x**,
and sample standard deviation **0.0001295x**. The matched old/new cache-effect
geometric means are **1.117964x for FP8** and **1.264127x for NVFP4**, with
every mirrored comparison above 1.05x. The uncached FP8/NVFP4 ratios remain
0.978168–0.978466x and retain their no-speedup disposition.

The final three blocks drained naturally without forced cleanup. All 141 files
(1,891,292 bytes) were copied and hash-verified in
`ai-perf-followthrough2-20260908-final-integration/kv-abba-blocks234-32030e6/`;
inventory SHA-256 is
`cd3df5dfe20d49135c2d8e5939a7acd8556cc438721031ed119fc301c369df10`.
These repeats cover one seed, device, and software environment. The older
1.025x result from Transformer Engine 2.18 is excluded from the cache-effect
calculation, and arithmetic qualification does not establish application quality.

Matched Nsight Systems captures now confirm the cache mechanism on the same
runtime. Every kernel is attributed through its CUDA launch correlation to the
single `compute_kernel:profile` range:

| Per complete iteration | Uncached | Cached |
| --- | ---: | ---: |
| Linear forwards / GEMMs, each arm | 260 / 260 | 260 / 260 |
| Attention / normalization kernels, each arm | 130 / 130 | 130 / 130 |
| Quantization ranges, each arm | 520 | 262 |
| FP8 cast kernels | 520 | 262 |
| NVFP4 weight amax / transpose / zero-amax kernels, each | 260 | 2 |
| Total FP8 kernels | 1,560 | 1,302 |
| Total NVFP4 kernels | 3,380 | 2,606 |

All other kernel-family launch counts stay unchanged. This confirms removal of
redundant weight preparation while retaining the compute workload. It is one
matched trace per arm; instrumented durations are not used for the timing claim.
The uncached control also passes all three profilers while retaining its
`failed_no_speedup` benchmark disposition. Its 38 files / 102,232,691 bytes,
including the completed profiler-batch supervision receipt, are verified in
`ai-perf-followthrough2-20260908-final-integration/kv-uncached-profile-e1f4f7e75/`.

The first attempt measured 1.106112x but both kernel-replay captures hit their
150-second limits, so its disposition remains `failed_profiler`. Its 35 files /
109,934,731 bytes are retained separately. Replaying the full selected NVTX range
with `app-range` resolves those timeouts and collects all five minimal metrics
for both arms. These counters cover the aggregate range; they do not enumerate
individual kernels, and replay duration is not an ordinary latency measurement.
The lab now prefers this replay mode while preserving an explicit CLI override.
Its selection tests pass for both variants and explicit/default modes; the full
focused file passes 30 tests. A fresh ordinary CLI invocation at `2f5bef154`,
without a replay-mode override, passes full correctness/runtime gates and all
three profilers in both arms, measuring 542.973 / 490.717 ms (1.106489x).
Both NCU reports contain all five finite counters over the complete selected
range, with no launch-count or kernel filter. The 34 files / 100,741,698 bytes
are verified in
`ai-perf-followthrough2-20260908-final-integration/kv-default-profile-2f5bef154/`.
All baseline function ASTs and the other workload/accuracy files are unchanged
from the arithmetic-qualified `32030e6` source. The profiler batch drained
naturally without forced cleanup.

The private DDP optimizer/backward overlap candidate now passes exact one-GPU
update/output checks, but its initial timing screen remains below 1.05x, as
detailed below. Fresh mechanism traces and two-GPU candidate checks remain
pending. Existing DDP, pipeline, and serving limitations remain. Earlier Ozaki no-speedup classifications are invalidated by
the timing-parser defect below; their original receipts remain preserved.
The earlier KV 1.025x result retains its original runtime-specific disposition.

An additional Ozaki candidate replaces the per-GEMM pageable-host sentinel copy
with an asynchronous device fill of the same `-1` flag. The reset and native
fallback check remain in place; arithmetic, mantissa settings, and error limits
are unchanged. CUDA documents possible staging synchronization for pageable
copies. Removing that possibility is the motivation, not a measured speedup.
The 19 focused policy/parser tests pass. Source
`5b5fb4cad48482541b271ad2297de4818a89e1ba` now compiles on B200 with CUDA 13.0.88
and cuBLAS 13.1.1.3, passes the real-CUDA sentinel/full-GEMM probe, and passes all
ten existing independent arithmetic qualification cases without changing limits.
Three Compute Sanitizer 2025.3.1 memcheck runs report zero errors: the sentinel
probe and dynamic/fixed rectangular edge cases. These checks do not establish
memory safety for every shape or application-level numerical quality. All 17
execution records and their log hashes pass local re-audit. All 23 files
(1,045,530 bytes) are retained in `ozaki-sentinel-accuracy-v2-5b5/`, inventory
SHA-256 `cda72dfd9bc7107fc10ab6f23518a1ca00b5fe92de08bf02b1abe8580450ddad`.
All 80 old/new timing executions and the separate 64/256 MiB workspace screen
complete. The sentinel change is timing-neutral: matched geometric means at
64 MiB are **1.000088x dynamic / 1.000489x fixed**. Raising workspace to 256 MiB
improves candidate timings by **1.029788x / 1.033635x**, below 1.05x and using
192 MiB more workspace. The default remains 64 MiB.
[CUDA synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).

The screen exposed a separate **shared timing-parser defect**. The C++ lab emits
scientific notation, but `parse_kernel_time_ms()` truncated the exponent:
`TIME_MS: 8.84707164764404252e-01` became 8.847 ms instead of 0.8847 ms.
Native FP64 timings had exponent zero and escaped this tenfold error. A fresh
public CLI reproduction reports native 5.132640 ms, dynamic 8.864288 ms, and
fixed 6.634400 ms, reproducing the false no-speedup classification. The patch
parses complete scientific-notation tokens in every supported time unit and
rejects malformed/non-finite default tokens. All **35 focused timing tests pass**,
and the shared parser agrees with all 80 retained CUDA-event timing logs.
Corrected source `d949f215aeacd086930f88594e1d2113fc36cea1` now reports native
**5.125280 ms**, dynamic **0.885309 ms (5.789257x)**, and fixed
**0.664301 ms (7.715300x)** through the actual public CLI on B200. All correctness
and runtime gates pass. The overall result remains **`failed_profiler`**:
Nsight Systems and the host-side PyTorch captures complete, but NCU captures no
kernels because its selected NVTX range belongs to the Python parent while the
CUDA operations run in compiled child processes. A parent PyTorch trace also
does not establish visibility into those child CUDA kernels. The batch drains
naturally; that receipt remains an incomplete profiler run.

Source `2c9f9e06f6b83d5cfd65f0df41061332a5c3cb87` builds a separate NVTX-enabled Ozaki executable and
selects its complete ten-GEMM timed loop directly in Nsight Compute. Startup,
warmup, and reference checks are outside the selected range. Ordinary timing
binaries remain separate. Its first B200 run captures all ten native kernels and
40 fixed-variant kernels, with five finite counters per kernel. The run still
returns `failed_profiler` because the minimal-mode winner branch incorrectly
counts an inapplicable PyTorch profiler as failed. All 28 files (8,405,275 bytes)
of that failed run are hash-verified in `ozaki-cli-child-2c9/`.

The follow-up at `0361e22d406e4fa2922daedc886511fa05fcf4cd` applies the same
PyTorch applicability rule to baseline, all-variant, and minimal-winner profiling.
Compiled-child execution reports the explicit reason in
`*_profiler_not_applicable`; it neither emits a misleading parent trace nor
records PyTorch as a successful required GPU profiler. All 47 focused CPU
build/dispatch/error checks pass. The actual public CLI now exits **0** in both
minimal and roofline modes, and both stages drain naturally without interruption.
Minimal profiles the winning fixed variant; roofline profiles both variants.

| Ordinary timing / capture | Native | Dynamic | Fixed |
| --- | ---: | ---: | ---: |
| Minimal run latency (ms) | 5.137859 | 0.889360 | 0.664954 |
| Minimal run speedup | 1.000x | 5.777030x | 7.726643x |
| Roofline run latency (ms) | 5.137805 | 0.886848 | 0.661939 |
| Roofline run speedup | 1.000x | 5.793332x | 7.761747x |
| Roofline selected NCU kernels | 10 | 80 | 40 |

These timings come from the separate ordinary executable runs, not profiler
durations. The workload remains 4096 cubed, seed 2026, scale 0.001, three warmups,
ten measured GEMMs, 64 MiB workspace, and the same frozen error budgets. The
harness requests 1500/3996 MHz clocks on the same portable B200 environment.
Every captured kernel carries the selected child NVTX range, and all five
required counters are finite. No kernel-name or launch-count filter narrows
the loop. The five reports across both modes contain 180 kernels and 900 finite
required counter cells. Both Nsight profilers succeed for every requested arm.
Dynamic additionally captures min/max and auxiliary GEMM/epilogue kernels;
that extra work is visible in the 80-versus-40 kernel count.
All 59 files (22,874,361 bytes) are hash-verified in `ozaki-cli-child-0361/`,
inventory SHA-256
`7b3aea1027a530425b0bc8712773ec7b11292f0778c80cfaaa05025c4b73b021`.

Compute Sanitizer 2025.3.1 then executes **12 full-size checks**: memcheck and
initcheck for native, dynamic, and fixed, each in both the ordinary and dedicated
profile build. Every command uses the unchanged default workload and accuracy
arguments, returns zero, and reports zero errors. This checks these actual
executables and inputs; it does not create a repository-wide uninitialized-memory
detector or close all remaining hardening declarations.
All 14 files (23,159 bytes) are hash-verified in `ozaki-child-memory-0361/`,
inventory SHA-256
`3d4a771ac3235a17fb6a3dfe296be44f09062ebdfd4a73b6959ae99b04de6eb9`.

The documented measurement-only shell helper also now rejects every exit code
except 2. Previously an unexpected zero exit returned success. Six real shell
checks cover child exits 0, 2, and 99 under both Bash and Zsh; all pass. Historical
README timings and process-wide trace totals are labeled separately from current
qualified evidence. No CI or push was run.

The direct CUDA-event screen at 64 MiB measures native-over-dynamic and native-over-fixed
speedup geometric means of **5.797175x / 7.738416x**, with all numerical limits
passing. These are timing-screen results, not a benefit from the sentinel patch
or a substitute for corrected public harness/profiler validation. All 85 files
(242,072 bytes) are hash-verified in `ozaki-sentinel-abba-5b5/`, inventory SHA-256
`f3e38ee1840b6c8291ce0434fa2250b9d62ba188a3f846aa738e796b84f0e0b4`.

## Private DDP optimizer-overlap screen

The next candidate groups the same fused AdamW parameter updates on a dedicated
CUDA stream after their gradients become ready. Repository runtime source stays
at `5be3ae28d09b71c63cf38fdecf44693d63ac593c`; the experimental helper is kept
outside the repository, SHA-256
`c14386a15df57b0f443480e0cfb5d7819312002c414e33e34b75d71717e0f310`.
Its independent source review finds no static correctness blocker for this
single-stream workload, without qualifying broader optimizer or DDP modes.

The three-step, one-B200 exact gate starts from identical TinyLlama weights and
uses actual MRPC data. Every step matches the synchronous fused AdamW control:
full training logits and loss, all 201 parameter tensors, all 603 optimizer-state
tensors, and full post-update logits. Every parameter receives one update per
logical step. The original probe's zero-worker prefetch configuration error is
preserved; its corrected probes pass. Eight receipt/log files (11,986 bytes) are
hash-verified in `ddp-overlap-exact-world1-v1-v3/`, inventory SHA-256
`aacad773a3dfd57ba55770b2238c2fd36600e1af50115ce665308d0ca06349b6`.

An A/B/B/A screen then invokes the actual `optimized_ddp.main` training loop with
100 steps, batch 16, accumulation 1, seed 42, real MRPC, and the same fused AdamW
settings. Both arms use the same wrappers and unchanged timer boundaries; only
the optimizer substitution differs. All four runs complete 100 updates for
every parameter. Their complete final inputs and logits match exactly. Real
MRPC produces padded sequence length 136 for the retained final batch; the
older synthetic-data timing cohort is a separate workload.

| Order | Optimizer | Training iteration (ms) | Whole process (ms) |
| --- | --- | ---: | ---: |
| A1 | Synchronous fused AdamW | 59.107510 | 12485.117 |
| B1 | Grouped overlap candidate | 57.034617 | 12230.208 |
| B2 | Grouped overlap candidate | 57.246815 | 12080.219 |
| A2 | Synchronous fused AdamW | 59.191430 | 12286.542 |

The mirrored training-loop ratios are **1.036344x and 1.033969x**, both below
1.05x. Whole-process ratios are **1.020843x and 1.017079x**; that boundary also
includes startup and the symmetric post-training output checks. These results
are an initial component screen, not an accepted baseline/optimized-pair win.
The harness manages 1500/3996 MHz clocks on the selected B200. Another workload
occupies the other GPU, so these results retain their shared-host scope. The
two-GPU occupancy guard rejects that launch before execution; no unrelated
process is interrupted. Fresh traces, a second timing seed, and exact two-GPU
checks remain pending. Accumulation, optional compilation, and other unsupported
optimizer modes require their existing synchronous paths until separately
validated. No candidate runtime change has been promoted. All 15 timing-screen
files (557,294,104 bytes), including the complete final tensors, are hash-verified
in `ddp-overlap-abba-world1-seed42-v1/`, inventory SHA-256
`bceba3519f782922320b28ee0fa57d18399770d09ba9f2ab8222aebb15814917`.

### DDP overlap mechanism and rejected follow-ups

Two paired Nsight Systems captures complete with exact final outputs. The second
adds explicit forward/backward markers, allowing CUDA runtime correlations to
separate optimizer kernels from backward kernels. Analysis below excludes the
first state-initializing update and averages the remaining 99 updates; the
ordinary timing comparisons still include all 100 updates.

| Diagnostic per steady update | Synchronous control | 50 MiB overlap |
| --- | ---: | ---: |
| Optimizer group calls | 1 | 34 |
| Host optimizer-group ranges (ms) | 0.910480 | 3.145919 |
| Range time outside CUDA runtime calls (ms) | 0.762602 | 2.780272 |
| Optimizer kernels | 55 | 107 |
| Optimizer GPU activity (ms) | 8.359493 | 9.932461 |
| Direct overlap with backward kernels (ms) | 0 | 4.826535 |
| Optimizer GPU activity after backward ends (ms) | 8.359493 | 0.953174 |

Overlap reduces the exposed optimizer tail, while smaller updates add launch
overhead and GPU activity. The non-CUDA range remainder includes interpreter and
scheduling time; it is an upper bound on possible savings, not time proven
removable from the critical path. These profiler durations explain the mechanism
and are not accepted speed ratios. The original and annotated captures are
hash-verified in `ddp-overlap-nsys-world1-seed42-v1/` and
`ddp-overlap-nsys-world1-annotated-v2/`, respectively: 11 files / 326,362,894 bytes
and 11 files / 326,343,085 bytes. Their inventory SHA-256 values are
`bf0242cf2274fd2c3cc64cd638cc420f21d9a2fdbf8345d9e94d406af18159e4`
and `da1333b86181819db6689554caf0e572ef178705889bd71768a4f8610224596d`.
The SQL, correlation script, per-iteration metrics, and report are retained in
`ddp-overlap-trace-analysis-v2/`.

Three subsequent experiments keep the same model, real data, update count, and
original timer boundaries. Each passes the three-step exact parameter/state/output
gate and all complete final-output comparisons in its four-run A/B/B/A screen.

| Experimental change | Group calls per update | Mirrored training ratios | Mirrored process ratios |
| --- | ---: | --- | --- |
| Increase capacity to 100 MiB | 19 | 1.032193x / 1.030991x | 1.016393x / 1.012292x |
| Cache 50 MiB group state lists and use functional fused AdamW | 34 | 1.035345x / 1.036996x | 1.007940x / 1.003995x |
| Reuse 50 MiB group readiness events and check the current stream directly | 34 | 1.038407x / 1.035032x | 1.024765x / 0.999778x |

None closes the 1.05x gap. The separate cohorts do not establish a significant
difference between these candidates and the original overlap helper. The cached
candidate preserves first-use state initialization and performs exactly 34
original optimizer calls followed by 3,366 cached functional calls over 100
updates. Its independent exact probe confirms that later updates exercise the
new path without changing full parameters, moments, step counters, or logits.
No runtime candidate is promoted. All 36 files / 1,114,604,750 bytes from these
exact gates and timing screens, including full final tensors, are hash-verified
in `ddp-group100-cached-screens/`, inventory SHA-256
`1f82bf5bea4601a9c6ec83e3e7789b58b976b995d8c365a33ba70b3be28f022d`.

A further annotated Nsight capture confirms that caching group state lists
reduces host optimizer-group ranges from 3.145919 to 1.923718 ms per steady
update. The non-CUDA remainder falls by about 1.199 ms, but the optimizer still
launches 107 kernels and has about 0.953 ms of GPU activity after backward.
The cached candidate's unprofiled ratios are 1.035345x / 1.036996x, compared with
the original candidate's 1.036344x / 1.033969x. These separate cohorts do not
establish a significant difference; both remain below 1.05x. Removing this host
work therefore does not close the recorded workload's remaining gap.

The separate event-reuse candidate retains gradient stream-lifetime tracking
and the original optimizer math. Its three-step exact check covers all 201
parameters, 603 optimizer-state tensors, and complete pre/post-update outputs.
Each timed candidate run performs 100 updates, reusing 34 readiness events for
3,400 event records. Full final inputs and logits match across all four timed
runs. Its receipt's `PASS` records execution and correctness; both training
ratios remain below the 1.05x performance requirement. This candidate has no
separate Nsight capture or two-GPU qualification and is not promoted.

The cached-profile and event-reuse evidence is hash-verified in
`ddp-cached-profile-stream-screens/`: 29 files / 883,747,736 bytes, inventory
SHA-256 `87732319b6490b0cee0fc09f7766bcc5044c94a3532934fa647cefcc122f7dfa`.
All three stages exit successfully and drain without forced cleanup.
The helper, exact-check driver, timing
drivers, and experiment specification are retained in
`ddp-stream-candidate-v6-drivers/`. Cached-profile analysis is preserved separately
in `ddp-cached-profile-analysis-v3/`.

The benchmark interpreter was rechecked and remains Torch 2.9.1+cu130 with CUDA
13.0. The host's default Python reports a different Torch version; it is not the
interpreter used for these runs. No package or shared-environment change was
made. These are still one-B200 component experiments on a shared host; the
candidate's two-B200 and public-harness qualification remain pending.

### Execution coverage priorities

A bounded reconciliation of the 543 logical optimized entries against one
historical snapshot maps 165 entries and leaves 378 unmapped. **Unmapped does
not mean unexecuted:** the retained ledgers span multiple sources, aliases, skips,
and later targeted reruns. This comparison cannot establish current-source
execution coverage. The next concrete gaps include FSDP2 child-produced
verification, hybrid expert parallelism, repaired persistent-decode/TMA paths,
the FA4 ALiBi provider fix, and repeated sequence-parallel validation. The ranked
file/receipt pointers and public commands are retained in
`execution-gap-priorities-422bc001a/`.

PR #28 is back in draft; broad CI and publication are deferred while this work
continues.

## FSDP2 training repair and direct B200 execution

Source `1a2a11fb40d6455af4abd6face5bfd7eaa1f64d8` repairs all four FSDP2
producers. Packed data already supplies next-token targets; passing those targets
as ordinary model labels shifted them a second time. A shared loss helper now
computes FP32 cross-entropy directly against the supplied targets. All 50 bundled
1024-token rows and all 255 bundled 128-token rows have the expected shifted
layout. Synthetic FSDP2 data now uses the same target convention.

Direct runs initialize a common seed, samplers use the active seed, and private
data generators avoid changing model initialization. Both variants consume full
per-rank batches and reject an empty loader instead of looping forever. The fast
configuration respects an explicit layer override. Positive argument validation,
matched throughput-report warmup, and synchronized maximum-rank training timing
are added; a duplicate FP8 setting is removed. `--steps` continues to mean
optimizer updates, and the intended fused optimizer remains enabled in the
optimized BF16 path.

The helper's loss, gradient, seed, and argument tests pass **18 checks**. A
separate focused run passes **66 loader, lab, and child-wrapper checks**. These
are targeted source/CPU checks, not broad CI.

On one B200, both actual entrypoints complete two optimizer updates from four
microbatches, using eight TinyLlama layers, sequence length 1024, per-rank batch
size two, accumulation two, and BF16 with FP8 disabled. Recorded training
diagnostics are **252.483 ms/update baseline** and **191.276 ms/update optimized**;
process durations are 14.742 and 14.887 seconds. This short execution check does
not establish a repeatable speedup or full training equivalence.

Both corresponding Nsight Systems runs also exit successfully. The exported
reports contain **5,346 baseline GPU kernels** and **4,539 optimized GPU kernels**.
Both direct and profiled stages drain without forced cleanup. The source bundle,
driver, CPU log, and GPU artifacts are retained under `fsdp2-source-v1/` and
`fsdp2-direct-nsys-v1/` in the final-integration artifact directory.
The latter transfer verifies 15 files / 5,225,973 bytes, inventory SHA-256
`267812afad750871422a7dadb783c069ac198d098bd2d54ed69c56fc8aa26fb5`.

Independent optimized full trained-state/output acceptance, two-B200 execution,
broader Nsight Compute coverage, and the real child-result contract remain pending. Generic FSDP2
wrapper execution therefore stays explicitly unavailable; direct-script success
is not substituted for that missing contract.

Compute Sanitizer `memcheck` subsequently completes both real entrypoints with
**zero errors** at the same source and workload. The `initcheck` baseline attempt
reports 100 uninitialized two-byte global reads in BF16 multiplication reached
from `apply_rotary_pos_emb`, then exceeds its 300-second child timeout. The
optimized initcheck arm is not reached. The supervisor drains five remaining
owned processes with SIGTERM; this is a failed, interrupted validation, not a
passing memory check. The cause remains under investigation.

Both sanitizer attempts, including the full failed report and cleanup receipt,
are preserved in `fsdp2-memory-v2/`: nine files / 854,287 bytes, inventory SHA-256
`02c8da5a35fd4fbd24ce37c5a31d9574e2b9b62ca4fbb60271273d01521e8fb2`.

The independent one-B200 training diagnostic passes the baseline exactly after
two optimizer updates. It compares all **483,428,352 model values** across 75
tensors, **966,856,779 optimizer-state values** across 225 tensors, all
**65,536,000 final logits**, the final loss, and all four training losses against
a separately instantiated and trained unsharded eager reference. Both models
remain BF16, and every checked value is finite. This validates the recorded
baseline workload; it does not establish optimized or two-GPU equivalence.

The optimized arm of that first diagnostic exceeds its 300-second timeout before
producing a final comparison receipt. Its failure and owned-process cleanup are
retained alongside the baseline pass in `fsdp2-exact-world1-v1/`: six files /
41,403 bytes, inventory SHA-256
`d3cf95ca20e548d1c9c075f274539a9af70a70f8e0d300aa2f48d1159290dbc1`.
A separate verifier revision preserves the full checks while avoiding redundant
error calculations for equal finite chunks and adding progress markers. CPU
comparison confirms identical verifier results across ten targeted cases. The
optimized arm also times out with that revision, during CPU state comparison;
its failure and cleanup receipt are retained in
`fsdp2-exact-world1-optimized-v2/`: four files / 32,176 bytes, inventory SHA-256
`b5bb297eb2f41807ffa777d90bd14eb523d140477f803ce9bdac5f7b6fce5976`.

A third verifier revision uses equivalent NumPy operations for plain CPU
floating-point tensor comparisons. It completes the optimized diagnostic in
**45.446 seconds**, with natural process drainage and no timeout. It retains
the zero-tolerance comparison and reports **FAIL** against the eager reference:

| Full comparison | Unequal values / checked values | Maximum absolute difference |
| --- | ---: | ---: |
| Final logits | 53,405,106 / 65,536,000 | 0.076171875 |
| Model parameters | 15,660,689 / 483,428,352 | 0.00048828125 |
| Optimizer state | 713,540,355 / 966,856,779 | 0.000244140625 |

The final loss differs by 0.0032787323 (0.03539% relative); all four training
losses differ, with maximum absolute difference 0.0005578995. Every compared
value is finite, and no shape, dtype, device, or state-structure mismatch is
reported. The candidate uses FlashAttention 2, while the unsharded reference
uses eager attention; both use the intended fused optimizer. This establishes
failure of bitwise equivalence, not by itself an implementation defect or an
accepted numerical error budget. Matched-attention isolation and independent
accuracy acceptance remain necessary before claiming a correctness or speed
pass for optimized FSDP2.

The complete numerical failure is preserved in
`fsdp2-exact-world1-optimized-v3/`: five files / 32,514 bytes, inventory SHA-256
`0e648bf313a9683d3aae7f1f27cf2d059445fed8adb048bf62a366c601b30c0f`.
The frozen driver and runner are retained in `fsdp2-exact-v3-driver/`.

Nsight Compute 2026.2.1 subsequently completes both actual FSDP2 entrypoints
with kernel replay, selecting the first `nvjet_tst` GEMM in each arm. Both
captures contain `nvjet_tst_128x256_64x4_4x1_v_bz_TNT` on compute capability
10.0, with **211 finite performance-counter values per arm** in the inspected
counter families. Both training processes complete two optimizer updates and
drain naturally, without timeout or forced cleanup. Clock control stays with
the harness; Nsight Compute uses `--clock-control none`.

This closes the selected-kernel replay check, not whole-model counter coverage
or numerical acceptance. The reports, CSV exports, counter check, and supervisor
receipt are retained in `fsdp2-ncu-world1-v4/`: ten files / 352,754 bytes,
inventory SHA-256
`ed0fc9d8a01670f55835a28060da2558b200b7653cad046e11657649543cd9ad`.

The reduced initcheck ladder narrows the failure further. A full q projection
followed by BF16 multiplication passes with zero errors, as does the isolated
rotary operation with initialized q/k inputs. One ordinary unsharded
`LlamaAttention` reproduces the uninitialized BF16 reads during rotary embedding,
then reaches its 180-second timeout. FSDP is therefore unnecessary to reproduce
the finding. All processes drain naturally. The ladder is preserved in
`fsdp2-initcheck-isolation-v1/`: 12 files / 1,106,485 bytes, inventory SHA-256
`5da608bff145de0e1349b6b783400ff0e175f3bad3b5411cdb2bccf878f33894`.

An additional full-state control makes both candidate and unsharded reference
use FlashAttention 2 with fused AdamW. With the default attention settings it
still differs after training: two of four training losses differ, final loss
absolute difference is 0.0012531281, and final-logit maximum absolute difference
is 0.068359375. All values are finite. This failure is preserved in
`fsdp2-exact-matched-attention-v4/`: five files / 32,747 bytes, inventory SHA-256
`0cc5119c23553e66ed80f58fec02debd60748061c9635dda1bf292bb2fdaaf5d`.

Repeating that matched-backend control with the explicit setting
`FLASH_ATTENTION_DETERMINISTIC=1` passes **every full comparison exactly**:
483,428,352 model values, 966,856,779 optimizer-state values, 65,536,000 logits,
the final loss, and all four training losses. It completes in 41.776 seconds and
drains naturally. This supports backward nondeterminism as the cause of the
matched-backend differences; it establishes the recorded FSDP2/fused-optimizer
equivalence in deterministic mode. It does not waive the separate FA2-versus-eager
accuracy question or convert default-mode timings into deterministic-mode results.
The pass is retained in `fsdp2-exact-deterministic-fa2-v5/`: five files /
21,216 bytes, inventory SHA-256
`dca7cc1f931c92abaed42e164e169a7b3ae41718b9598ffcf41f3beac2233c5d`.

## FSDP1 training repair and direct execution

Source `ddd7c8d7d63d4ad7cf3910338a0b39657ee8ca34` fixes the same second target
shift in all four FSDP1 producers, using a shared FSDP/FSDP2 loss implementation.
Synthetic targets now use the same next-token convention; data generation uses
a private RNG, direct runs bind the seed, and rank loaders reject empty work and
use full microbatches. The 50-row/two-rank/batch-two case now produces 12 full
microbatches per rank rather than overweighting a one-sample tail. Optimizer-update
step semantics and intended optimizer/attention/FP8 choices are preserved.

Thirty focused loss/data tests and 60 lab/wrapper integration tests pass. The
actual baseline and optimized single-GPU entrypoints then complete on a B200
with the full 22-layer configuration, packed sequence length 1024, microbatch
two, accumulation two, FP8 disabled, and two optimizer updates/four microbatches.
These are direct execution checks, not full training-equivalence or performance
qualification; the benchmark wrapper's larger default batch and two-GPU variant
still need their own runs.

Both source-clean runs drain naturally. Artifacts are preserved in
`fsdp1-direct-world1-v1/`: five files / 8,435 bytes, inventory SHA-256
`c2b12a07a10913696081d392d07473c240fdee3af7faa2c460635fffc0f0ff84`.
The source bundle, driver, and integration log are in `fsdp1-source-v1/`.
