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
| Integrated GPU suite | 5,722 passed, 78 skipped, 30 failed; affected files passed after repairs in the focused rerun |
| Standalone MoE entrypoint | Level 0 completed its normal 436.5M-parameter workload |
| Normal dual-pool vLLM runs | Exact tokens passed twice; speed target failed twice |
| Baseline gradient-fusion NCU | Full-workload one-metric report captured; three-minute five-metric attempt incomplete; longer bounded probe pending |
| Hosted CI and remaining main merges | Pending final GPU validation |

The original broad inventory included 441 zero exits, 44 exits with code 1, and
one native crash. Zero exits include skips and informational results. Later
targeted runs repaired the executable and verification defects; the complete
attempt history and remaining dispositions are in the
[repair checkpoint](2026-09-06-codebase-repair-checkpoint.md).

The integrated suite ran every collected case. Its failures exposed outdated
fixtures and assertions plus an autotune compilation timeout. All affected files
passed the subsequent B200 rerun; this is a full diagnostic run followed by
focused repair validation, not a single passing integrated run. Pytest shutdown
needed scoped cleanup of surviving compiler work after the original report was
written. Both the test counts and cleanup receipt are preserved.

## Measured outcomes

| Workload | Evidence and result |
| --- | --- |
| NanoChat | Approximately 2.00409x; repeated equivalent-workload measurements, complete bitwise outputs and Nsight Systems evidence |
| Chapter 2 transfer | Approximately 26.60845x for the full 100 MiB transfer; repeated interleaved measurements and peer-transfer trace evidence |
| Llama | Approximately 1.07956x; complete 8,388,608-element bitwise outputs, fresh inputs, repeated measurements and CUDA graph trace evidence |
| Memory-bound compilation | Full output passed at 1e-5 relative/2e-5 absolute tolerance; 26.7x observed in one normal run, with repeated performance validation still needed |
| Two-GPU cache-aware inference | Full verification passed; 1.64306x observed in one normal run, with repeated performance validation still needed |
| One-/two-GPU DDP | Exact outputs passed, including partial accumulation groups; no qualifying speedup |
| Two-GPU disaggregated inference | Exact output passed; 1.00818x remained below the 1.05x speed requirement |
| Dual-pool vLLM | Exact tokens passed with batch-invariant Triton; 0.95366x and 0.97053x in normal runs including engine startup |

These are portable observations from the retained runtime. Memory regressions,
default-backend token mismatches, and no-speedup results remain visible.

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
