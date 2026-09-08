# Remaining B200 follow-through

This pass starts from merged source `0299307facf77bc883b9d27b4fb175ce27e50ab4`
and extends the [previous report](2026-09-08-b200-followthrough-results.md).
The earlier captures, failures, numerical requirements, and no-win results remain
retained. This document records new work separately.

## Changes under validation

- DDP workers construct one tokenizer and pass that same tokenizer to dataset
  preparation. The baseline and optimized workers both use this lifecycle.
  Dataset preparation still constructs its own tokenizer when none is supplied.
  This removes redundant startup work; it does not change training arithmetic.
- Dynamic routing records each successful admission before choosing a GPU for
  the next request. Previously all 16 requests saw empty queue metrics and went
  to the first GPU. Metrics collected after the final admission could not affect
  any placement decision, so that polling is removed from this workload.
- Routing and dual-pool loops stop adding a fixed 10 ms sleep after each blocking
  engine step. Both baseline and optimized variants use the same loop behavior.
  The direct engine wrapper retains its existing yield when execution is deferred.
- Four stale missing-protection declarations become behavioral tests of existing
  cross-device execution and current-device boundary checks. The remaining 29
  declarations are an inventory, not 29 distinct confirmed bugs.

## Profiler investigation

Nsight Compute 2026.2.1, the latest release listed in NVIDIA's
[release notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html),
is already installed in an isolated task directory. The
system profiler, CUDA, PyTorch, driver, permissions, and credentials are unchanged.
The new probe uses one profiler per rank with TCP coordination. Its lockstep
filters select the `NCCL` domain's `ncclGroupEnd` and `ncclAllReduce` ranges seen
in the retained Nsight Systems trace. Compute launches are not globally matched
between asymmetric pipeline ranks.

This is a new diagnosis of collective replay, with a bounded owned process group;
it is not a performance comparison. The earlier successful selected-GEMM capture
does not establish full collective replay success. The launch follows the
[documented concurrent-kernel options](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#mandatory-concurrent-kernels)
and the installed 2026.2.1 CLI help.

The first launch was refused because an unrelated workload occupied a GPU.
Its subsequent 15-minute capacity wait ended without launching any GPU work.
No unrelated processes were interrupted or other tasks contacted.

## Validation status

The combined local regression check passed 245 tests and skipped 84 on macOS,
covering execution audits, both anti-cheat inventories, tokenizer reuse, and the
vLLM API/control-flow contracts. The
four new tests require two visible CUDA devices and remain unqualified until
they pass on the target. Dispatcher checks cover visible PyTorch operations on
the current thread; they do not cover arbitrary native/background execution.
Device identity checks detect drift present at a checked boundary, not a
switch-and-restore between boundaries.

New direct B200 timing, full-output verification, and profiler results are pending.
No new performance win or completed GPU validation is claimed by this draft.
