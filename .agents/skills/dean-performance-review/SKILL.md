---
name: dean-performance-review
description: Evidence-first performance review and optimization workflow adapted from Jeff Dean and Sanjay Ghemawat's Performance Hints for this PyTorch, CUDA, Triton, and distributed benchmark repository. Use for hot-path reviews, latency or throughput regressions, optimization proposals, benchmark design, and performance-sensitive APIs.
---

# Dean/Ghemawat performance review

Use this skill when the request involves performance-sensitive code, a slow path,
an optimization, a benchmark pair, or a performance-oriented API or data structure.
Read `code/AGENTS.md` before changing files under `code/`.

The source is Jeff Dean and Sanjay Ghemawat's
[Performance Hints](https://abseil.io/fast/hints.html). Their document focuses on
single-binary performance and uses mostly C++ examples. This skill adapts the
general principles to this repository; it does not treat those examples as proof
for Python, GPU, distributed, or ML-runtime changes.

## Required posture

- Prefer a faster design at authoring time when it does not materially hurt clarity.
- Measure or estimate before accepting extra complexity. Static review identifies
  hypotheses, not measured wins.
- Preserve the workload, correctness policy, hardware/runtime context, and one
  variable under test across control and candidate runs.
- Optimize the user-visible critical path. Do not move work outside the timer and
  call that an end-to-end win.
- Keep setup, warmup, steady state, and teardown costs separate and named.
- Do not publish a speedup without the repo's clock, provenance, profiler, repeat,
  and correctness gates.

## Workflow

### 1. Define the path and cost model

Fill or update `code/templates/performance_intake.yaml`.

Record:

- the primary KPI and workload shape;
- whether the code runs during setup, once per batch, once per request, once per
  token, once per rank, or once per element;
- rough counts for bytes moved, allocations, launches, synchronizations, boundary
  crossings, and remote/storage operations;
- the expected dominant cost and the largest plausible improvement.

Use current measurements for operation costs when available. Treat published
latency tables as intuition, not constants for the active host.

### 2. Establish a trustworthy baseline

Choose the smallest representative layer from
`code/docs/benchmark_methodology.md`: micro, component, or end to end. Freeze the
workload with `code/templates/benchmark_workload_spec.yaml` and capture a baseline
with the repo harness or triage bundle.

For benchmark classes, run the static hot-path checks as part of the normal linter:

```bash
cd code
python scripts/linting/check_benchmarks.py
```

Static findings are triage inputs. An allowlist is valid only when the flagged work
is the behavior the benchmark intentionally measures.

### 3. Profile before choosing the edit

Start with the highest-level profile that can locate the cost, then narrow with NCU,
Nsight Systems, allocation profiles, hardware counters, or a focused microbenchmark.
If the profile is flat:

1. inspect loops high in the call stack;
2. look for structural work repeated at API boundaries;
3. inspect allocation count and cache/memory traffic;
4. replace overly general operations with the narrow operation actually needed;
5. consider several independently verified small wins rather than inventing one
   large speculative rewrite.

### 4. Rank candidates in this order

1. **Algorithm and work avoided:** improve asymptotic behavior; add a common-case
   fast path; precompute, defer, specialize, or cache; hoist invariant work out of
   loops; remove hot-path logging and eager stats.
2. **API shape:** batch operations to amortize Python, RPC, process, launch, or lock
   crossings; accept views when ownership is not transferred; let callers pass
   preallocated buffers or already-known metadata.
3. **Representation and locality:** compact hot state; separate hot and cold data;
   prefer contiguous/batched storage; use arrays or bitsets for dense integer
   domains; choose layouts that reduce HBM, cache-line, and transaction waste.
4. **Allocation and copying:** preallocate, reserve, reuse scratch state, avoid
   unnecessary materialization, and transfer ownership instead of copying when the
   language/runtime contract permits it.
5. **Parallelism and synchronization:** batch parallel work, overlap independent
   work, amortize locks, shorten critical sections, shard contention, and check for
   false sharing or needless context switches. Confirm that bandwidth or occupancy
   headroom exists before adding parallelism.
6. **Generated code and compilation:** keep common fast paths small; move rare slow
   paths out of line; avoid needless specialization, graph variants, template/code
   generation, or compilation cache fragmentation.

Prefer the highest item supported by the profile. Do not jump to a lower-level
micro-optimization while a higher-level structural cost dominates.

### 5. Map the principles to this repository

- **Python:** avoid per-iteration Python containers, formatting, logging, regexes,
  repeated attribute/config parsing, subprocesses, and file/network I/O unless they
  are the declared workload.
- **PyTorch:** create tensors and random inputs in `setup()` where semantics allow;
  reuse output/scratch buffers; avoid `.item()`, `.cpu()`, `.numpy()`, `.tolist()`,
  and device transfers in a GPU timed path; keep compile and graph capture outside
  steady-state replay measurements.
- **CUDA/Triton:** inspect launch count, occupancy, memory transactions, divergence,
  register pressure, synchronization, and code variants. Do not infer a kernel win
  from host timing alone.
- **Distributed:** batch collectives or control-plane calls where semantics allow;
  measure per-rank tails; distinguish useful overlap from hidden synchronization;
  never generalize the source document's single-binary claims into a distributed
  result without distributed evidence.

### 6. Implement one reversible hypothesis

Keep the public interface stable when possible. Put specialized representation or
layout changes behind a narrow module boundary. Add or update:

- correctness tests;
- a regression test for the mechanism;
- the representative benchmark or profiler path;
- structured metrics/provenance that expose the claimed mechanism.

### 7. Verify and report

Run the focused tests, syntax/import validation, the static benchmark linter, and a
real repo invocation when feasible. For performance claims, use repeated interleaved
control/candidate runs under locked clocks and report distributions, not a best run.

Report these fields:

- `Path`: exact hot path and invocation frequency.
- `Baseline`: artifact and metric distribution.
- `Cost model`: dominant operations and expected ceiling.
- `Evidence`: profiler counters or trace locations.
- `Change`: one mechanism, including complexity or API tradeoff.
- `Result`: measured delta with correctness and variance gates.
- `Boundary`: what remains unmeasured or what the result does not prove.

If target hardware is unavailable, report the code and static checks as verified but
the performance effect as unmeasured.
