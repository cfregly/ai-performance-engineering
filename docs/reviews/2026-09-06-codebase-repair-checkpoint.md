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
