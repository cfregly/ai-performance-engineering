# B200 review results

The review repaired executable failures, numerical verification, distributed
training, seed ownership, profiler cleanup, and misleading result reporting.
Available examples were exercised directly on one or two B200s. This does not
establish that every optimization is faster, or cover hardware and model assets
that are unavailable on this host.

## Validation status

The renewed completion pass starts from `6b98beed1`. It adds worker-bound
runtime receipts, expected GPU identity checks, active PyTorch profiler guards,
and real evaluation-contract tests in place of stale skips. It also tests
removing the DDP reducer for one-rank training while retaining the distributed
sampler and input order. B200 execution, the final failure regressions and
hosted CI are complete. [PR #25](https://github.com/cfregly/ai-performance-engineering/pull/25)
merged as `4b584671eb54d6aff639d39aa436cecb29f8522f`; its tree exactly matches
the CI-tested head `f32a3d012`. This completes the feasible validation and
repair pass on the available host; the limits and suggested improvements below
remain explicit. Each retained suite total and measurement identifies its
source revision.

The first expanded CPU check passed 344 cases and skipped 100, with nine
transport fixtures rejected because they omitted the child device. After
adding that explicit fixture identity, all 29 tests in the affected transport
file passed. Actual child-receipt corruption tests separately confirm that a
wrong PID or device is rejected. Runtime collection now follows measured work
and has a separate bounded grace period. New one-/two-B200 results are
recorded below; the small repeated singleton improvement remains below the
benchmark's acceptance threshold.

On `b1169c0a9`, the focused B200 suite completed **415 passed, 45 skipped,
zero failures or errors** in 124 seconds, with normal process drain. Both
normal DDP runs passed complete output verification at zero tolerance and
the new executed-runtime admission check. One GPU measured 49.200832 ms
baseline versus 47.210013 ms optimized (1.042169x); two GPUs measured
85.136544 versus 83.273424 ms (1.022374x). Both remain `failed_no_speedup`
because they did not reach the unchanged 1.05x requirement. The interleaved
old/new repeats and matched profiles are described below.

The normal `ch09:memory_bound` pair also passed on `b1169c0a9`, with complete
output and executed-runtime verification and a 26.54735x timing ratio. This
single pair result is separate from the repeated DDP optimization experiment.
The first DDP experiment attempt stopped in its reporting script because a
scalar tensor could not be viewed directly as bytes. No speedup was accepted
from that attempt. Its outputs and normal-drain receipt are retained; the
reporter now flattens tensors before byte serialization, with actual scalar,
empty, strided, BF16 and boolean CPU checks.

The corrected experiment completed all 16 one-GPU, 100-step measurements
across four interleaved ABBA blocks. All full-input, full-output and runtime
comparisons passed. Eight observations per arm had medians of 48.900048 ms
for the historical reducer path and 47.337926 ms for the singleton path;
their ranges were 48.820–48.984 and 47.168–47.499 ms, respectively. This is
about a 3.3% improvement, below the benchmark's 5% requirement. The first two-GPU experiment correctly completed
64 steps, the number of available batches per rank, and was rejected by the
experiment's incorrect 100-step assertion. Its recovery explicitly uses the
full 64-step two-GPU epoch in both arms and retains the completed one-GPU
measurements separately. The recovery completed 16 two-GPU measurements
and four matched Nsight Systems captures with full output and runtime
admission. The two-GPU control and candidate medians were 54.587270 and
54.722537 ms, respectively, with eight observations per arm. This batch-16,
64-step control intentionally exercises unchanged two-rank behavior; it is
separate from the canonical batch-32 `ddp_multigpu` example. A subsequent
Nsight Compute control capture succeeded, but its reader rejected the wide
CSV layout. The failed attempt is retained, and the repaired reader passed
against the actual captured CSV before a fresh paired attempt. Both fresh
Nsight Compute reports then passed import and counter inspection, retaining
329 finite selected counters per arm for one matched FusedAdam launch. This
single-launch scope is supporting evidence, not a whole-run speed measure.

The singleton Nsight Systems comparison removed 20,100 pointwise BF16
multiply launches and eight concatenation-copy launches over 100 steps.
That supports reduced reducer and launch overhead; there were no peer NCCL
kernels in either singleton arm. Whole-process medians were 14,662.323 and
14,693.828 ms, so no end-to-end improvement was established. The two-GPU
profiles retained identical instance counts for all 81 kernel names and
353,997 launches. Each Nsight Systems arm has one capture, separate from
the repeated unprofiled timing evidence.

A separate singleton startup experiment compared the retained reducer-free
path with lazy NCCL communicator creation. Across four seed blocks and eight
observations per arm, process-wall medians fell from 14,620.719 to 14,106.201
ms (1.036475x), with standard deviations of 99.034 and 164.733 ms. All four
block ratios improved, ranging from 1.036294x to 1.042302x. Matched Systems
captures removed one 207.720-ms initialization and one 253.913-ms destruction
range. Full inputs, outputs and runtime admission passed throughout. Commit
`5a9eeec80` applies the exact measured source: singleton NCCL initialization
omits `device_id`, while the multi-rank initialization call is unchanged.
Ten focused CPU tests passed with two explicit skips. This startup result
does not qualify the canonical benchmark's 1.05x worker-speed goal.

Further pipeline review found that `ch04:pipeline_parallel` verifies a separate
parent-process simulation before launching its distributed timing workers.
The baseline simulation retains the last microbatch, while the optimized
simulation uses the first. Its previous verification pass therefore does not
qualify the timed distributed outputs. Existing timing and profiler receipts
remain exploratory evidence; actual full-batch worker-output transport is
repaired in `5887577d0`. Both workers retain all measured microbatches
in input order and validate an independent sequential reference after timing.
The candidate uses rank-dependent warmup and paired bidirectional transfers.
The real two-rank CPU transport and related pipeline/worker tests passed
58 cases; another 14 helper-aware hygiene cases passed. The six affected
pipeline/tensor-parallel entrypoints have no static contract errors or warnings.
On `4071e06ca`, the normal full-shape two-B200 pipeline run passed actual
worker-output and runtime verification, measuring 15.502764 ms baseline and
15.307192 ms optimized (1.012776x). It remains `failed_no_speedup`. The
subsequent four-seed ABBA batch completed 16 launches and eight full-output
comparisons with exact equality throughout. Eight observations per arm had
medians of 15.941546 and 15.782274 ms (1.010092x), with standard deviations
of 0.366567 and 0.279505 ms. Per-block ratios ranged from 0.982065x to
1.024960x, so the result supports parity, not a qualifying speedup.
Commit `0613b1319` therefore tightens the pipeline's output tolerance from
`(0.1, 1.0)` to exact equality; five CPU schedule/reference tests passed,
including rejection of a small BF16 output perturbation. Its target rerun
passed on `632315c6b`. Both new Nsight Systems captures completed. Each new
kernel-replay Nsight Compute attempt timed out at its 180-second limit and
drained, leaving that profiler evidence explicitly incomplete.
The new Systems traces retain 1,024 main matrix-multiplication launches in
each arm. Paired transfers reduce NCCL SendRecv launches from 256 to 144
and aggregate SendRecv kernel time from 125.700 to 54.700 ms. This supports
the communication change without turning overlapping kernel-time sums into
an end-to-end speedup claim.
One subsequent bounded Compute probe selected only the common main GEMM
after eight launches, retaining the full two-rank workload. Its baseline also
timed out after 180 seconds and drained; the paired candidate was not launched.
The incomplete report and exact command are retained. Compute replay therefore
remains unresolved for this pipeline on the current host.

The earlier exploratory pipeline run on `b1169c0a9` measured 15.64706 ms
baseline and 23.32224 ms optimized (0.670907x). Both Nsight Systems reports
and their kernel, API and NVTX summaries were retained. Neither Nsight
Compute attempt produced a complete report; the run ended `failed_profiler`
and all owned processes drained. These captures do not repair the output
verification gap described above.

Commit `1d30adb22` additionally requires the independently retained local
worker count and complete per-rank runtime agreement. It rejects a result
with rank 1 removed from both receipt maps, as well as an incomplete or
version-mismatched rank 1. Actual CPU harness checks passed 25 cases, the
central admission suite passed 15, and the subsequent strict count-schema
check passed all seven integration cases. The target rerun on `4071e06ca`
completed 108 focused tests with one skip and no failures or errors, including
the per-rank receipt checks and final replacement CUDA version-lock case.

The unchanged TE precision pair was also rerun in the isolated TE 2.18
environment on `b1169c0a9`. FP16 measured 0.500929 ms and FP8 achieved
0.715940x, again `failed_no_speedup`, with normal process drain. Its actual
forward outputs and runtime admission passed the existing checks, but the
verification input omitted the training target and its numerical tolerance
was not calibrated. Commit `4071e06ca` includes the live training target and
all post-SGD parameter tensors in verification, retains independent CPU
outputs after timing, and fixes the baseline's FP16 metric label. All four
opt-in CUDA training-output tests passed in the isolated TE 2.18 environment,
including actual changed-target parameter updates. The earlier diagnostic
does not establish a complete training-correctness or speedup claim.
The first full-output calibration completed all four seeds but rejected its
seed-42-selected tolerance on holdout data. Review also found that its private
driver constructed a fresh harness configuration with the `performance`
backend instead of preserving the benchmark's `fp32_strict` policy. The
attempt is retained as diagnostic data; it does not establish a tolerance for
the published pair.

The corrected calibration preserved `fp32_strict`, the default 256-by-4096
workload and all 65 training updates: five setup, ten warmup and fifty measured
steps. Seed 44 selected a predeclared grouped numerical policy with a 20%
reserve; fresh holdouts 45, 1044 and 1045 all passed. Every input, training
target, prediction and all 67,121,152 parameter elements were checked, with
executed-runtime agreement. The selected `(rtol, atol)` values are `(0.4, 1.0)`
for prediction, `(0.001, 0.00075)` for weights and `(0.001, 0.00005)` for biases.
Maximum absolute differences over this cohort were 0.594971, 0.000534058 and
0.000041008, respectively. Replacing any output with zeros or introducing a
localized perturbation beyond its budget was rejected. These are numerical
bounds for this workload and configuration; FP8 timing remained slower than
FP16. Commit `9743d0588` integrates the exact-keyed policy into normal
verification, including subprocess transport and golden-output cache
comparisons. Both arms must declare the same map and cover every captured
output. Missing or changed predeclared child policies are rejected. Reused
benchmark instances discard prior child receipts before fresh execution, and
direct verification rejects invalid output-dictionary entries instead of
silently filtering them. The combined CPU check passed 37 cases with four
CUDA-only skips; neighboring transport and verification checks passed 152
cases with 15 explicit skips. On B200, this source passed 192 focused cases
and all four opt-in TE CUDA cases. The normal TE 2.18 and TE 2.9 runs preserved
the configured lifecycle, full output map and runtime agreement, with timing
ratios of 0.699952x and 0.671259x, respectively. Normal singleton and two-GPU
DDP also passed exact outputs and runtime agreement at 1.038194x and 1.011154x.
All four normal pairs remain `failed_no_speedup`. The pipeline baseline worker
completed, but the parent's optional tolerance-map lookup incorrectly required
a local payload. That integration failure is retained, and the full-suite gate
correctly held the broader run for its repair. Commit `632315c6b` makes the
optional map return `None` when a coordinator has no local payload; declared
maps and mandatory transported outputs remain checked. Nineteen focused CPU
checks passed, followed by the B200 regression and TE CUDA checks. The normal
two-B200 pipeline then passed its new exact output policy and executed-runtime
admission. The final-source pipeline measured 15.616376 ms baseline and
16.232841 ms optimized (0.962024x). The normal TE 2.18 and TE 2.9 ratios were
0.720155x and 0.696505x; singleton DDP reached 1.035264x and canonical two-GPU
DDP reached 1.009320x. These individual runs
confirm execution and preserve the no-speedup outcomes; the interleaved
experiments above remain the basis for optimization claims.

The final full B200 suite on `632315c6b` completed **5,905 passed, 68 skipped,
two failures and zero errors** across 5,975 cases in 2,347.253 seconds, followed
by complete process drain. Both failures were outside benchmark execution:
an old ZeRO2 callback test fabricated child stdout without a worker runtime
receipt, and the Chapter 13 README generator omitted the new calibration text.
Commit `8c0cce536` replaces the fabricated child with a real CPU worker;
`289dd73f7` preserves the documentation in the generator. The two complete
affected test files then passed **43 cases, zero skips, failures or errors**
on B200, again with normal drain. Benchmark and harness execution code did not
change between the broad run and these final regression checks.

The 68 full-suite skips comprise 42 explicit missing-protection declarations,
one nonbehavioral test-count summary, and 25 hardware, dependency, policy or
opt-in cases. Separate target runs cover the TE 2.18 CUDA opt-in cases as
described above. A skipped negative-dependency test means its required absent
dependency is installed; it does not indicate a failed installed runtime.

The retained test files now contain 42 explicit missing-protection declarations,
down from 58. This is a source inventory, not a claim that 16 independent
protections were qualified: several labels duplicated existing evaluation
contracts. Seven retained distributed labels also overlap existing declared
policy checks: collective algorithm, gradient-bucket bytes, barrier policy and
async-completion policy are compared, and registered workloads validate
completion and barrier receipts. Those checks do not provide general runtime
algorithm detection or instrumentation for arbitrary distributed workloads.
The new CUDA version and identity cases passed on B200; one final
duplicate version-lock test label was also replaced and passed its target
rerun. Version admission covers observed driver, CUDA, cuDNN, Python,
PyTorch and relevant installed-package versions. Installed cuBLAS package
metadata does not identify the native library actually loaded. The profiler
guard covers an enclosing PyTorch profiler; it does not detect an external
Nsight session or a profiler started and stopped entirely inside a workload.

| Check | Current result |
| --- | --- |
| Broad target execution | 486 targets attempted; unsupported and informational outcomes remain separate from passes |
| Benchmark contract scan | 936 entrypoints; zero errors and warnings |
| Repository-wide Ruff correctness checks | Passed |
| Focused GPU regressions | Earlier 813 affected-file tests and six isolated Llama tests passed; latest Colfax suites passed 32 cases in each pinned environment |
| Latest integrated GPU suite | **5,905 passed, 68 skipped, two failures, zero errors** on `632315c6b`; both failures repaired and the affected files passed **43 B200 tests** on `289dd73f7`; all processes drained |
| Standalone MoE entrypoint | Level 0 completed its normal 436.5M-parameter workload |
| Normal dual-pool vLLM runs | Exact tokens passed twice; speed target failed twice |
| Gradient-fusion NCU | Complete five-metric reports inspected; later repeats timed out or reported driver `UnknownError`, so replay remains intermittent |
| Hosted CI | [Benchmark validation](https://github.com/cfregly/ai-performance-engineering/actions/runs/34202021120) passed on `f32a3d012`: **5,442 CPU tests passed, 533 skipped, zero failures or errors** in 22 minutes 54 seconds. Static analysis, all 936 benchmark contracts, and dashboard audit/lint/build plus 21 tests passed. [Configured CUDA architecture builds](https://github.com/cfregly/ai-performance-engineering/actions/runs/34202021092) passed on the same head. |
| Main delivery | [PR #25](https://github.com/cfregly/ai-performance-engineering/pull/25) merged as `4b584671e`; its tree exactly matches the CI-tested head `f32a3d012`. It builds on [PR #24](https://github.com/cfregly/ai-performance-engineering/pull/24), merged as `6b98beed1`. Earlier repair PRs #20, #21 and #23 are also merged. |
| Explicit opt-in GPU tests | ZeRO2 passed; two Blackwell tests passed on each of two ranks; local-model vLLM passed |
| Version-specific FP8 template | The actual TE 2.18 CUDA template passed in a separate environment; one executed case, no skip |
| New Colfax GPU validation | Both full-workload deep-dive runs, both ABBA repeats, 64 opt-in cases and eight Nsight report inspections passed on `356490bd1` |

The original broad inventory included 441 zero exits, 44 exits with code 1, and
one native crash. Zero exits include skips and informational results. Later
targeted runs repaired the executable and verification defects; the complete
attempt history and remaining dispositions are in the
[repair checkpoint](2026-09-06-codebase-repair-checkpoint.md).

Earlier integrated suites exposed outdated fixtures and assertions plus an
autotune compilation timeout. Those original failures and compiler cleanup
receipts remain preserved. That earlier repair cycle's integrated suite completed
5,874 collected cases: 5,795 passed, 79 skipped and zero failures or errors, in
31 minutes 29 seconds. It exited normally and drained all owned processes.

That earlier hosted CPU suite collected 5,877 cases: 5,341 passed, 536 skipped
and zero failures or errors in 23 minutes 32 seconds.
[Benchmark validation](https://github.com/cfregly/ai-performance-engineering/actions/runs/34157897194)
and [CUDA architecture builds](https://github.com/cfregly/ai-performance-engineering/actions/runs/34157897282)
both passed. The later Colfax changes were also exercised in both dedicated
B200 environments, as recorded above; the integrated GPU suite retains its
own source revision rather than claiming a rerun on the final merge.

An earlier distributed bootstrap repair passed all eight CPU/CUDA worker tests
on B200, including one- and two-rank execution. Its two hosted workflows
then passed: [benchmark validation](https://github.com/cfregly/ai-performance-engineering/actions/runs/34143112135)
and [dual-architecture builds](https://github.com/cfregly/ai-performance-engineering/actions/runs/34143112148).
That repair's integrated GPU rerun used merged `main` with the same tested code tree.
It completed all 5,845 cases in 41 minutes 52 seconds and retained one repeated
Llama autotune timeout. Native CUTLASS compilation was still active; pytest
shutdown again needed scoped cleanup after its complete XML report was saved.
The unchanged numerical workload now runs in a fresh interpreter with owned
compiler cleanup. It passed with full collection and again inside the completed
passing integrated run. At that earlier revision, skipped hardening tests documented
detectors that were not implemented: 58 cases explicitly
name missing detectors, and one further skip rejects test-name counts as proof
of protection coverage. CPU-only negative controls and supported opt-in cases
are separate; the distributed, model and Colfax opt-ins were executed later.

The profiler builders now request exactly the five validated metrics for
application-range replay, without adding a section set. All 67 profiler checks
passed on CPU and B200. One complete full-workload baseline capture and the
repository-generated optimized capture passed strict report inspection. A later
baseline repeat timed out after ten minutes. The last full-workload repeat on
`568c9102c` reported a driver `UnknownError`; its owned profiler was stopped and
drained after preserving the error. Stable Nsight/NCCL replay remains unproven
on this stack. Both successful and incomplete attempts are retained.

## Measured outcomes

| Workload | Evidence and result |
| --- | --- |
| NanoChat | Approximately 2.00409x; repeated equivalent-workload measurements, complete bitwise outputs and Nsight Systems evidence |
| Chapter 2 transfer | Approximately 26.60845x for the full 100 MiB transfer; repeated interleaved measurements and peer-transfer trace evidence |
| Llama | Approximately 1.07956x; complete 8,388,608-element bitwise outputs, fresh inputs, repeated measurements and CUDA graph trace evidence |
| Memory-bound compilation | 31.74309x repeated ABBA observation; all 16,777,216 outputs passed at 1e-5 relative/2e-5 absolute tolerance; Nsight confirmed 128 kernels fused into one |
| Two-GPU cache-aware inference | Full 2,048-element outputs matched exactly; repeated ABBA measured 0.988428x, so the earlier single-run 1.64306x did not reproduce |
| Colfax FA4 decode | 1.173108x repeated ABBA; all 32,768 outputs matched exactly; normal deep-dive run and profiler inspection passed |
| Colfax FA4 backward | 1.080924x repeated ABBA; all 402,653,184 gradient elements matched exactly; normal deep-dive run and profiler inspection passed |
| One-/two-GPU DDP | Exact outputs passed, including partial accumulation groups; no qualifying speedup |
| Two-GPU disaggregated inference | Exact output passed; 1.00818x remained below the 1.05x speed requirement |
| Dual-pool vLLM | Exact tokens passed with batch-invariant Triton; 0.95366x and 0.97053x in normal runs including engine startup |

These are portable observations from the retained runtime. Memory regressions,
default-backend token mismatches, and no-speedup results remain visible.

The latest memory and cache-aware measurements used four fresh seeds and eight
observations per arm. Memory medians were 2.387228 ms baseline and 0.075205 ms
optimized, with standard deviations 0.006990 and 0.010427 ms. Cache-aware medians
were 14.331648 and 14.499441 ms, with standard deviations 0.470046 and 0.517233 ms.
Complete outputs were compared before and after timing in every block. Both
memory Nsight Systems and five-metric Nsight Compute reports were inspected;
both cache-aware arms also retained Systems traces. The two-GPU topology has
only one decode rank, so it cannot establish an affinity-migration benefit.

The Colfax repeats used four fresh seeds and eight observations per arm at the
unchanged default shapes. Decode medians were 1.056113/0.900269 ms, with standard
deviations 0.000819/0.000216 ms. Backward medians were 30.557050/28.269385 ms,
with standard deviations 0.398677/0.072006 ms. Both source revisions are checked
against every runtime Python file and installed Git provenance. The traces and
counters support the measured ablation; internal TMEM/barrier ordering was not
directly traced. See the [lab](../../code/labs/flashattention4/README.md).

## Improvements to prioritize

1. **Measure steady-state serving separately from startup.** The router currently
   reconstructs engines in each measured invocation. A dedicated benchmark with
   reusable engines would better isolate admission, TTFT and decode throughput;
   it must reset request state and preserve full token checks on both arms.
2. **Choose numerical acceptance budgets before accepting approximate kernels.**
   The KV-cache compression and Ozaki cases retain measured error distributions.
   Their acceptance thresholds need an independent accuracy policy.
3. **Investigate distributed bottlenecks from the retained traces.** DDP and
   disaggregated runs now have trustworthy outputs and throughput parsing.
   Their no-speedup results are useful starting points for matched profiling,
   rather than reasons to relax the speed requirement.
4. **Keep runtime combinations explicit.** The repository-built vLLM 0.16/FA4
   backport is reproducible and hash-checked. Use the documented matching
   environment and attention backend for exact router comparisons; revalidate
   after changing a model, compiler, or serving backend. The TE 2.18 FP8 template
   was also tested separately from TE 2.9; its isolated build excluded optional
   NCCL expert parallelism because PyTorch 2.9 lacks the required headers.
5. **Finish the wider hardware matrix when those resources are available.**
   Grace/GB10, four-or-more GPUs, multi-node communication, and the missing
   Phi3.5/TensorRT-LLM model engine remain outside this host's coverage.
6. **Implement the explicitly missing benchmark protections before claiming them.**
   Per-operation CPU execution placement, managed-memory events, uninitialized
   memory provenance and related skipped detectors remain engineering work. Passing arithmetic or
   test-name counts cannot establish these protections.

Ordinary examples support Python and local `torchrun`; Slurm is optional.
See the [router lab](../../code/labs/dynamic_router/README.md) for the tested
two-GPU command and the [code README](../../code/README.md) for the explicit
vLLM backport installation.
