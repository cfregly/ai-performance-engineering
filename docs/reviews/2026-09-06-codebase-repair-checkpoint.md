# Codebase repair checkpoint

This change repairs benchmark correctness, profiler lifecycle and reporting,
distributed launch behavior, and environment diagnostics. It is a validated
repair checkpoint, not a claim that every example is maximally fast or that the
complete hardware matrix has passed.

## Main repairs

- Preserve real numerical verification and equivalent workloads in block
  scaling, KV-cache compression, attention, MoE, and distributed examples.
- Fix BF16-to-FP8 double rounding in KV-cache quantization. A one-dimensional
  FP32 scale requests FP32 multiplication while retaining the preallocated FP8
  output. The existing tolerance and timed workload are unchanged.
- Validate complete Nsight Compute application-range captures, bind their
  provenance to the actual report, and keep descriptive range duration separate
  from kernel timing and benchmark speedup across the CLI, MCP and summaries.
- Improve CUDA/NVSHMEM bounds and lifecycle checks, process cleanup, device
  visibility handling, and structured unsupported-environment diagnostics.
- Keep ordinary examples runnable through Python or local `torchrun`.
  Slurm is optional; a scheduler can reserve one allocation for a whole sweep.
  Direct all-reduce runs now use unique artifact names without a Slurm job ID.

## Retained validation

The integrated CPU run before the final small KV/portability corrections
collected 5,084 tests: 4,600 passed, 481 skipped, and three failed because the
local Torch 2.8 runtime lacks the expected precision API. Those failures remain
recorded; this is not a full-suite pass. Static checks covered 2,788 Python files
and 161 shell files. The final KV-focused local suite passed nine tests and
skipped four CUDA-only tests.

Actual execution on a shared, virtualized two-B200 host with Torch 2.9.1 and
CUDA 13 confirmed full block-scaling numerical checks and profiler/report
collection, plus both two-rank serving regressions. The standalone public CLI
application-range capture passed with all five requested metrics, the complete
selected NVTX range, exact report provenance and verified child runtimes.
Every completed stage's owned processes drained.

The original KV output check failed. A reduced real-class B200 diagnostic
reproduced 31,600 tolerance violations with identical inputs; explicit FP32
multiplication removed all violations without changing tolerance. The correction then passed all 13 append-path tests and full default-workload
input/output verification on the B200. That replay still ended as a profiler
failure: sidecar materialization incorrectly applied a 16 MiB source-file
limit to the valid 41,843,422-byte optimized NCU capture. The repair separates
streamed artifact hashing from source-worker limits and prevents failed
required captures from updating saved expectations. Retained-report
reprocessing and the complete merged-revision replay remain separate gates.
The reported memory saving and single-pair timings are not accepted
performance wins while required validation remains incomplete.

## Remaining work

### September 6 direct execution follow-up

PR #15 merged as `c40767b8294d0471082aea9cfb8f5636b8961aec`. The final
candidate's hosted CPU suite completed with 4,598 passed, 492 skipped, and no
failures or errors (5,090 collected). Static, dashboard, and native build checks
also passed. Both retained KV NCU reports were successfully reprocessed after
the large-artifact hashing repair; the original interrupted/failed runs retain
their original status.

The unfiltered GPU test suite ran directly on the two-B200 host on
September 6 starting at 10:05:44 UTC, using the clean merged source, Torch
2.9.1+cu130, and `python -m pytest tests -q --tb=short`. It completed in
1,831.28 seconds: **5,008 passed, 77 skipped, and 5 failed**. Two-rank NCCL
tests executed. One failure selected system CMake 3.28.3 instead of the
already-installed 3.31.10; four process-lifecycle checks encountered exited
children retained as zombies by the outer validation supervisor until final
drain. The supervisor ultimately drained all owned processes. All five tests
then passed on the same merged source in 26.31 seconds after selecting CMake
3.31.10 and reaping the supervisor's adopted, exited children during execution.
Both receipts are retained: the original full run remains failed, and the
targeted rerun passed and drained its processes. No Slurm allocation is required.
The subsequent example
inventory includes benchmark pairs, registered demos, and tools; the pytest
suite alone does not establish that all of those entrypoints executed.

PR #16 (`7b7728a8dd043a1e9a07a29565626dfed89c8984`) adds explicit optional
device-identity and evaluation contracts, and distributed workload metadata.
It merged as `3d7982a8860506505d1ca24b8c3754f3826fdba1` after CPU validation,
dashboard, static analysis, and dual-architecture native build checks passed.
Its 86 focused contract tests passed on the B200 host with no skips in 9.29
seconds. Declared collective algorithms are not profiler observations,
and dataset hashes do not prove semantic freedom from contamination.

The integrated local contract/harness suite passed 172 tests and skipped 24
hardware/platform cases. Evaluation receipts now originate in the actual worker,
survive subprocess serialization on failure, and finalize after teardown; source
or threshold drift rejects timing results. The default 873-file benchmark lint
scan passed without errors or warnings.

Discovery confirmed 486 benchmark targets, 29 demos, and 34 tools. Ten registered
demo/tool paths previously had factories but no invoked workload entrypoint;
they now invoke their actual standalone execution paths. Real CLI tests with
CUDA hidden reject execution with explicit diagnostics rather than exiting
successfully after import. These remain
demo/tool runs, not baseline-versus-optimized performance evidence.

The broad sweep exposed two practical continuation gaps: aggregate preflight can
reject a whole bucket, and an exception escaping a chapter can abort later
chapters. The direct validation queue therefore gives each target a separate
process and result record, continues on nonzero exits, and preserves earlier
results. Native tier-1, cluster, and fabric entrypoints run separately from this
486-target pass. A follow-up repair now isolates explicit filtered targets when
continuation is requested, records preflight failures and ordinary exceptions,
and proceeds to later units. Its 78 focused and existing regression tests passed.
Run-level aborts still propagate. Batch target resolution and contiguous-prefix
native resume accounting remain improvement items.

The first direct demo pass attempted all 29 registered demos: 23 exited
successfully and six required follow-up. One requires GB10 coherent memory;
two need explicit workload arguments; the other three exposed a TMA alignment
failure, a CUTLASS dependency preflight failure, and a distributed launcher
selecting another environment's Python through `PATH`. Tools also began
executing. Completed results were preserved before replacing the coordinator
to resume with corrected source and arguments.

The launcher repair uses the selected Python's `torch.distributed.run` for demos
and profiler workers, matching benchmark timing. A real two-worker identity test
passes even with another environment's `torchrun` first on `PATH`. The combined
launcher and TMA static contract selection passed 82 tests with two skips.
The CUDA alignment repair then passed the actual production kernel's three
full-output and canary shapes under Compute Sanitizer with zero reported errors,
and the original 2048-square demo completed. The focused B200 regression
selection passed 90 tests with no skips. The initial standalone validator
compile exposed a separate flag defect: `-arch=sm_100a` emits generic PTX as well;
the architecture-specific caller requires an explicit `compute_100a` to
`sm_100a` code-generation target. The launcher, alignment, and continuation
repairs are committed in `a738687c28c10161e3000a97ce2ed640795872b6`; they are
not yet merged. The validator flag repair follows that commit.

CI's broader `--include-unpaired --fail-on-warnings` lint scan also passed:
932 files, zero errors and warnings. The contract checker recognizes exact
registered demo/tool paths as standalone entrypoints while continuing to reject
`__main__` blocks in paired benchmark modules.

### Resumed examples and measured KV graph candidate

The resumed queue uses clean source
`57e754fe5eb87d8669e800a0dca3e05bef31a1aa`, retaining prior source identities
for reused results. All 29 registered demos and 34 tools have been attempted.
The reconciled exit counts are 25 successful demo exits and four nonzero exits,
and 32 successful tool exits and two nonzero exits. These are execution counts,
not inference-quality claims: some tools are calculators, simulations, asset
generators, or dependency probes. The GB10-only demo remains unsupported on B200.

The two-B200 reruns exposed invalid collective keyword arguments in expert
parallelism, unsafe unbatched ring transfers and fully masked softmax blocks,
and an incompatible full-graph compile request around Transformer Engine's
disabled compiler boundary. The DTensor tool also confused visible device IDs
with process ranks. Repairs are committed in `99a365de6` with 57 focused CPU
checks passing and three hardware checks skipped. Their B200 reruns remain
pending. The default dynamic-router tool now fails if requested vLLM execution
cannot initialize; explicit synthetic mode records its provenance. Its earlier
successful exit must not be reused as real vLLM evidence.

The seven-caller TMA validator compiled and found a separate odd-width defect:
the Chapter 7 copy kernel overwrote 387 row-padding values for a 129-by-385
logical tensor with leading dimension 400. The follow-up preserves TMA stores
for full tiles and bounds ordinary stores on partial tiles. Strict full-output
and canary checks are unchanged; the B200 rerun is still required.

The standalone KV experiment compared the exact 256-step FP8 body with graph
replay on the same buffers. Eight interleaved AB/BA rounds produced 80 samples
per path, after five warmups per path:

| Diagnostic | Eager FP8 body | CUDA graph replay |
|---|---:|---:|
| Median CUDA-event time | 36.7343 ms | 23.8563 ms |
| 5th–95th percentile | 36.7258–36.7649 ms | 23.8482–23.8694 ms |
| Profiled kernel executions | 8,192 | 8,192 |
| In-range CUDA runtime calls | 16,385 | 3 |

Raw FP8 bytes and FP32 scales matched bitwise before and after measurement;
the full BF16-reference check also passed at its existing tolerance. Graph
preparation cost 175.357 ms including three preruns, with 1,024 additional
allocated bytes and 4 MiB additional reserved memory. This is approximately
35.1% lower latency for the existing FP8 body, with matching kernel work in the
Nsight trace. It is a noncanonical mechanism experiment, not an overall win
against the BF16 baseline. The integrated candidate still needs its full paired
benchmark and required profiler checks.

The first two multi-GPU benchmark targets completed timing and Nsight Systems
capture but retained `failed_profiler` outcomes. The pure-copy Chapter 2 target
produced no NCU kernel report, and Chapter 4 gradient fusion lacked an explicit
timed NVTX range for direct distributed NCU capture. These failures remain
visible while the queue continues.

Practical improvement priorities are consistent standalone launch metadata,
real behavioral tests for distributed paths, explicit synthetic/unsupported
result categories, and profiler requirements that reflect the workload being
measured. The cost calculator now also ran with explicitly illustrative inputs;
those values are not measured B200 power or token throughput.

- Run the complete GPU test suite on the final merged revision.
- Complete all four sweep stages and reconcile the 486-target inventory,
  including unsupported and failed cases rather than silently dropping them.
- Finish missing protection checks and remaining attention/MoE investigations.
- Validate four-GPU, multi-node and unavailable dependency combinations on
  appropriate hardware. Two B200s do not establish those results.
- Use repeated interleaved measurements and profiler evidence before accepting
  a performance claim; virtualized-host measurements remain noncanonical.

Reproducible entrypoints are documented in [the sweep playbook](../../code/FULL_SWEEP.md).
Focused KV checks run with `python -m pytest tests/test_kv_optimization_append_paths.py`
from `code/`. Hardware and dependency requirements still apply independently of
whether an external scheduler is used.
