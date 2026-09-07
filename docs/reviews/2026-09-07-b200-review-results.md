# B200 review results

The review repaired executable failures, numerical verification, distributed
training, seed ownership, profiler cleanup, and misleading result reporting.
Available examples were exercised directly on one or two B200s. This does not
establish that every optimization is faster, or cover hardware and model assets
that are unavailable on this host.

## Validation status

| Check | Current result |
| --- | --- |
| Broad target execution | 486 targets attempted; unsupported and informational outcomes remain separate from passes |
| Benchmark contract scan | 932 entrypoints; zero errors and warnings |
| Repository-wide Ruff correctness checks | Passed |
| Latest focused GPU regressions | 813 passed; all six isolated Llama tests also passed on `9d342cc79` |
| Latest integrated GPU suite | **5,795 passed, 79 skipped, zero failures or errors** on `46ba89929`; normal exit and complete process drain |
| Standalone MoE entrypoint | Level 0 completed its normal 436.5M-parameter workload |
| Normal dual-pool vLLM runs | Exact tokens passed twice; speed target failed twice |
| Gradient-fusion NCU | Complete five-metric baseline and optimized reports inspected; repeated baseline replay remains intermittent |
| Hosted CI | 5,310 CPU tests passed, 535 skipped; static analysis, dashboard and dual-architecture CUDA builds passed on `d77181e28` |
| Main delivery | [PR #21](https://github.com/cfregly/ai-performance-engineering/pull/21) merged as `814866061`; its tree matches the tested code. PR #20 is also merged |
| Explicit opt-in GPU tests | ZeRO2 passed; two Blackwell tests passed on each of two ranks; local-model vLLM passed |
| Remaining GPU validation | New Colfax decode/backward source and input-freshness repairs require their opt-in GPU tests, profiling and repeated timing runs |

The original broad inventory included 441 zero exits, 44 exits with code 1, and
one native crash. Zero exits include skips and informational results. Later
targeted runs repaired the executable and verification defects; the complete
attempt history and remaining dispositions are in the
[repair checkpoint](2026-09-06-codebase-repair-checkpoint.md).

Earlier integrated suites exposed outdated fixtures and assertions plus an
autotune compilation timeout. Those original failures and compiler cleanup
receipts remain preserved. After the repairs, the latest integrated suite ran
all 5,874 collected cases successfully, with 79 explicit skips, in 31 minutes
29 seconds. It exited normally and drained all owned processes.

The final distributed bootstrap repair passed all eight CPU/CUDA worker tests
on B200, including one- and two-rank execution. Both final hosted workflows
then passed: [benchmark validation](https://github.com/cfregly/ai-performance-engineering/actions/runs/34143112135)
and [dual-architecture builds](https://github.com/cfregly/ai-performance-engineering/actions/runs/34143112148).
The fresh integrated GPU rerun used merged `main` with the same tested code tree.
It completed all 5,845 cases in 41 minutes 52 seconds and retained one repeated
Llama autotune timeout. Native CUTLASS compilation was still active; pytest
shutdown again needed scoped cleanup after its complete XML report was saved.
The unchanged numerical workload now runs in a fresh interpreter with owned
compiler cleanup. It passed with full collection and again inside the completed
passing integrated run. Some skipped hardening tests also document detectors that are not yet
implemented, rather than demonstrating those protections.

The profiler builders now request exactly the five validated metrics for
application-range replay, without adding a section set. All 67 profiler checks
passed on CPU and B200. One complete full-workload baseline capture and the
repository-generated optimized capture passed strict report inspection. A later
baseline repeat timed out after ten minutes, so stable Nsight/NCCL replay remains
unproven on this stack. Both successful and incomplete attempts are retained.

## Measured outcomes

| Workload | Evidence and result |
| --- | --- |
| NanoChat | Approximately 2.00409x; repeated equivalent-workload measurements, complete bitwise outputs and Nsight Systems evidence |
| Chapter 2 transfer | Approximately 26.60845x for the full 100 MiB transfer; repeated interleaved measurements and peer-transfer trace evidence |
| Llama | Approximately 1.07956x; complete 8,388,608-element bitwise outputs, fresh inputs, repeated measurements and CUDA graph trace evidence |
| Memory-bound compilation | 31.74309x repeated ABBA observation; all 16,777,216 outputs passed at 1e-5 relative/2e-5 absolute tolerance; Nsight confirmed 128 kernels fused into one |
| Two-GPU cache-aware inference | Full 2,048-element outputs matched exactly; repeated ABBA measured 0.988428x, so the earlier single-run 1.64306x did not reproduce |
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
   after changing a model, compiler, or serving backend.
5. **Finish the wider hardware matrix when those resources are available.**
   Grace/GB10, four-or-more GPUs, multi-node communication, and the missing
   Phi3.5/TensorRT-LLM model engine remain outside this host's coverage.

Ordinary examples support Python and local `torchrun`; Slurm is optional.
See the [router lab](../../code/labs/dynamic_router/README.md) for the tested
two-GPU command and the [code README](../../code/README.md) for the explicit
vLLM backport installation.
