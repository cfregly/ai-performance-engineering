# Hybrid expert-parallel remediation: CPU evidence, CUDA HOLD

The source slice addresses original W1-029 (cross-stream reusable storage), W1-075
(replica initialization), and W1-076 (zero-send collective participation). Source
review and real CPU/Gloo checks pass. CUDA/NCCL, multiple physical hosts, the
default BF16 workload, sanitizers, and performance remain **HOLD**.

## Changes

- Each world rank creates all node process groups in the same order, retaining
  the group to which it belongs. Initial replicated projection/router parameters
  are broadcast before optimizer creation; expert shards remain distinct.
- Every member participates in an applicable remote-world or same-node phase,
  including zero senders and globally empty phases. An empty local input no
  longer bypasses count exchange and the two communication directions.
- Routing and expert output buffers are separated by branch. Differentiable
  outputs use fresh storage and retain autograd history; inference still reuses
  capacity within its own branch. Side-stream inputs and returned outputs record
  their consuming streams. Same-rank compute may overlap remote communication,
  but the current stream waits before switching from world to local NCCL groups.
- Tokens and their route weights travel in one differentiable payload. The
  previous separate autograd nodes could run different backward collectives on
  zero-receive ranks. Functional list collectives now return the tensors carrying
  their backward node instead of their preallocated, detached backing buffer.
- The original expert `out=mul` path failed with trainable parameters. The
  differentiable path now builds a real CopySlices graph. Existing inference
  behavior, immutable index caches, blocking host count reads, and reusable CUDA
  events remain intact. The stream is allocated only when CUDA overlap executes;
  the benchmark still requires CUDA, while its route helpers can be validated
  with actual CPU Gloo tensors.

The group-creation, expert autograd, functional-list-return, and divergent
backward-node defects are adjacent discoveries, not additional invented audit
IDs. Root owns their final ledger classification.

## Retained failures and controls

`hybrid-ep-original-source.py` is the exact reviewed b57e4c6a9 source. Running
`hybrid-ep-original-mechanisms.py` with four real Gloo processes reproduces unequal
replicas and the expert autograd error on every rank. Ranks 0/2 return empty
without collectives, while ranks 1/3 hit actual five-second Gloo timeouts. The sole
adapter selects CPU storage in the original count-buffer property; original
route control flow and collectives are unchanged. This is explicitly not a CUDA
reproduction. The old subgroup-creation ordering was established from source
and the documented API contract; its old runtime failure was not separately
reproduced.

The first revised test exposed Gloo's functional-list implementation using
scatter, which cannot handle uneven splits in this installed PyTorch 2.8 build.
That failure is retained, not reported as a NCCL failure. Uneven and subgroup CPU
cases use the production single-buffer collective; balanced world traffic still
exercises the production functional-list path and its real backward graph.

A second attempt then aborted in real Gloo backward with mismatched message
sizes. Its source snapshot is `hybrid-ep-before-packed-payload.py`. Packing token
and weight gradients into one collective payload resolves that failure. Earlier
failures, all subsequent runs, and available rank diagnostics are retained under
the `hybrid-ep-*` prefix.

The stable focused suite has **14 passed, 9 skipped**. Four new CPU tests include
three-step baseline/optimized expert forward and gradient comparisons, branch
storage independence with retained-capacity reuse, and four-process Gloo checks.
Each rank verifies ordered node groups, replica broadcast without changing
experts, averaged replica gradients and an optimizer update, then remote uneven,
same-node uneven, globally empty, and balanced list routes. Outputs, original
input order, input gradients, route-weight gradients, and expert parameter
gradients are compared with independently assembled expert references.

Ten existing CPU wrapper tests pass. Eight existing CUDA wrapper cases and the
new four-GPU NCCL case skip because this host has no CUDA/NCCL. Source assertions
were narrowly updated so obsolete shared names and the detached-return bug do
not constrain the implementation; their existing test IDs remain.

## Required GPU acceptance

The new gated test calls actual `forward_loss`, not a stand-in route method, for
baseline, optimized without overlap, and optimized with overlap. Three routing
patterns exercise rank-varying empty branches, globally empty remote traffic,
and mixed destinations. Each variant performs three optimizer steps plus
inference with varying token counts to reuse retained capacities. Full outputs,
loss, input gradients, all sharded/replicated parameter gradients, and updates
are compared with a serial global float64 CPU oracle. The four-GPU gate executes
FP32 with TF32 disabled; default-BF16 qualification remains separate. Its
54 per-rank checks have **not run** here. The local four-GPU test uses two logical
node groups and does not establish inter-node fabric behavior.

From `code/`, on an authorized four-GPU host:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -s -rs -p no:cacheprovider tests/test_audit_wave1_hybrid_ep.py -k four_gpu_nccl --junitxml=/absolute/receipt-dir/hybrid-nccl.xml
```

The test bounds its spawned rank processes at 300 seconds and kills owned ranks
on failure/timeout. Run the same gate with Compute Sanitizer memory/synchronization
checks, preserving all subprocess logs. For an authorized two-host experiment,
the same file exposes a torchrun entry point with four world ranks and two local
ranks per host; use the environment's existing launch/rendezvous configuration:

```sh
PYTHONPATH=. torchrun --nnodes=2 --nproc-per-node=2 [authorized rendezvous options] tests/test_audit_wave1_hybrid_ep.py --nccl-worker --output-dir /absolute/receipt-dir
```

The external multi-host launcher must bound and clean up the entire job. All rank
JSON files, exact hardware/software/topology provenance, and sanitizer results
are required. Device/process host metadata is emitted; `fabric_qualified` stays
false because numerical agreement alone does not establish fabric qualification.
No GPU allocation, remote launch, performance run, or historical result rewrite
was performed in this slice.

The group and stream ordering contract is documented in
[PyTorch distributed process groups](https://docs.pytorch.org/docs/2.9/distributed.html#groups).
Actual local Gloo behavior is from PyTorch 2.8.0, not the repository's pinned GPU
stack. Root reviewed the production diff and found no further source concern;
full hybrid multi-group backward ordering remains an explicit hardware gate.
