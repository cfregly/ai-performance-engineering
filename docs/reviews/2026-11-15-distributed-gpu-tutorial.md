# Distributed GPU tutorial review and integration

## Decision

Add one fixed numerical application that connects communication choices, plus
focused guides that explain their contracts and costs. Existing chapters already
cover NCCL, NVSHMEM, topology, overlap, and graph-captured symmetric memory. The
increment is a shared workload, explicit grouped P2P, a staged learning path,
registration lifecycle, current NCCL device API guidance, and a separate
five-mode NVSHMEM device lab with complete-output verification.

The source is FZJ-JSC's
[SC26 tutorial at commit cd34f9d](https://github.com/FZJ-JSC/tutorial-multi-gpu/tree/cd34f9d40d600efd8d84e433b1a9e84ab5eff5f3),
dated November 15, 2026. The local comparison began at
`275419650b448e2aedc461531dc28d2e86a995c7`. Unrelated working changes were preserved.

## Review coverage

The pinned tree has 129 tracked entries, including an external KokkosComm
gitlink. The review covered the repository material and the gitlink's metadata.
The external GitLab source could not be retrieved, so its implementation is an
explicit omission. No claim below depends on that source.

| Source surface | Reviewed material | Result for this repository |
|---|---|---|
| Overview and onboarding | System topology, programming-model overview, Jupyter access, environment setup | Reuse the topology-to-placement explanation. Keep course accounts, reservations, and deployment scripts out of the local learning path. |
| MPI introduction | GPU buffers, local rank mapping, decomposition, staging, GPUDirect | Add a CUDA-aware MPI arm to a matched stencil. Explain buffer awareness separately from stream awareness and selected transport. |
| Optimization lecture and exercise 04 | Boundary-first streams, interior work, halo exchange, priorities, convergence checks | Add serialized/overlapped schedules with explicit dependencies and a fixed iteration policy. |
| Performance and debugging | Compute Sanitizer, cuda-gdb, per-rank Nsight reports, NVTX, profiling scope | Reuse existing profiling tools. Add a concrete native lab trace exercise and complete-output checks. |
| Host NCCL/NVSHMEM and exercise 07 | Grouped P2P, registration, symmetric allocation, on-stream puts and barriers | Implement these in the common workload and explain costs, completion, and reuse. |
| Device communication and exercise 09 | Graph capture, pointer orientation, NVSHMEM device operations, NCCL LSA/GIN | Add graph replay to the host NCCL workload, five runnable NVSHMEM device modes, and precise device API guidance. Do not import the defective GIN protocol. |
| Advanced summary | Kokkos, directives, Python MPI, UCC/SHARP, AMD/Intel pointers | Preserve as related references. Vendor ports, an unverified Kokkos submodule, and another SHARP tutorial are lower-value additions to this NVIDIA-focused implementation. |
| Hidden templates and notebooks | All masters, tasks, solutions, six instruction paths, generation files | Adopt staged questions and reference solutions. Avoid copying duplicate generated notebook/source trees. |
| Build, CI, metadata, course scripts | Makefiles, compile-only CI, environment/deployment scripts, licenses and citation metadata | Keep current build capability checks and source credit. Do not treat compilation as GPU correctness or performance evidence. |

All eight slide decks were extracted, totaling 298 PDF pages including animation
steps. Eighteen selected diagram, chart, and timeline pages were rendered for
visual inspection. All 46 files in exercises 04 and 07 and all 50 files in
exercise 09 were inspected, including hidden masters and task/solution variants.
The 18 notebooks contain the exercise instructions. They are not independent
executed GPU evidence.

## What was incorporated

| Addition | Location | Practical value and cost |
|---|---|---|
| A fixed-iteration Jacobi application | [Native lab](../../code/labs/multi_gpu_communication/README.md) | Compare communication without changing the mathematics. Adds native CUDA/MPI dependencies and explicit validation work. |
| CUDA-aware MPI, host NCCL, and optional host-on-stream NVSHMEM | Same native lab | Exposes distinct producer-readiness and completion contracts on one grid. Each backend still needs a supported runtime. |
| Grouped neighbor exchange | [NCCL guide](../../code/ch04/nccl_groups.md) and native source | Lets dependent P2P calls progress together and can amortize submission cost. Delaying submission to build larger batches can add latency. |
| Serialized and overlapped schedules | Native lab and Chapters 4 and 11 | Produces boundaries early and leaves interior work independent. Extra launches, events, and resource contention can erase the benefit. |
| Explicit user-buffer registration | Native NCCL option and NCCL guide | Can avoid internal copies on eligible paths. Adds setup, consistent-rank requirements, and storage/handle lifetimes. |
| Captured two-step replay and odd tail | Native NCCL graph option and Chapter 12 | Preserves pointer parity while amortizing repeated launches. Adds capture and graph-resource costs and a collective replay contract. |
| PGAS, typed puts, barriers, MPI bootstrap, and consumed acknowledgments | [NVSHMEM guide](../../code/ch04/nvshmem_communication.md) | Makes one-sided operation semantics and reuse obligations explicit. A world barrier is simpler but couples every PE's progress. |
| Scalar, block, and persistent device protocols | [NVSHMEM device lab](../../code/labs/nvshmem_device/README.md) and [exercises](../../code/labs/nvshmem_device/TASKS.md) | Implements five runnable modes with independent full-output checks. One CTA covers local synchronization in waiting kernels, at the cost of limited compute throughput and explicit remote epochs. |
| Native NCCL device API explanation | [Device communication guide](../../code/ch04/nccl_device_communication.md) | Adds LSA/GIN, symmetric windows, supported-version checks, and cumulative signal epochs. The fragments are educational, not a runnable network solver. |
| Staged exercises and scaling instructions | [Tasks](../../code/labs/multi_gpu_communication/TASKS.md) | Connects correctness, critical-path prediction, profiling, and measured scaling. Larger GPU counts and multi-node transport require separate evidence. |
| Relevant chapter and lab connections | Chapters 4, 11, 12, 13, 15 and the training/communication labs | Makes existing grouped operations discoverable without claiming their current waits achieve overlap. |

The source's most useful teaching choice is keeping Jacobi constant across APIs.
For AI engineers, that isolates ideas later used in context-parallel exchange,
pipeline activation transfer, and expert dispatch without introducing an entire
training or serving stack at the same time.

## Existing coverage that should be reused

The current [communication-stack lab](../../code/labs/nccl_nixl_nvshmem/README.md)
explicitly benchmarks selected-block host-tier handoff and probes stack
availability. It is not an application implemented through three interchangeable
communication libraries. The new numerical lab is a sibling with a different
contract, and NIXL keeps its memory-tier role.

The [native NVSHMEM pipeline](../../code/ch04/nvshmem_pipeline_patterns.cu)
already orders payload publication, completion, ready notification, consumption,
and reuse. The Python NVSHMEM pipeline documents CPU/Gloo signaling sidebands.
These are different implemented paths and should not be presented as equivalent
GPU-native signaling.

The [symmetric-memory ring](../../code/ch04/symmetric_memory_example.py) already
captures multi-rank publications and two lifetime fences on its CUDA backend.
The new graph lesson adds a numerical iteration with ping-pong output parity.
It does not establish that distributed graph capture was previously absent.

Topology discovery, collective size sweeps, arrival/contention diagnosis,
cluster readiness, and scaling models already exist. Reuse them. A fixed-global
or fixed-local-grid application curve adds empirical coverage that a planner
estimate and a bandwidth sweep do not supply.

## Source issues not carried into the adaptation

These are source-level findings. They were not reproduced on GPUs during this
review.

1. Several validators stop before the last owned row. A tolerance comparison
   based only on `fabs(error) > tolerance` can also accept NaNs. Check every
   owned element and reject nonfinite values.
2. Some sparse norm-check paths stop the wall timer before outstanding GPU
   work is synchronized. Finish all timed streams before stopping the timer.
3. The graph solution reads the original allocation regardless of completed
   iteration parity. Select the final field from actual iteration count.
4. The NCCL device solution repeatedly waits for the same small cumulative
   signal threshold. Its later waits can accept earlier notifications. Use
   advancing per-edge epochs and a consumed/reuse protocol.
5. The exact NCCL device setup calls and communicator properties need newer
   APIs than the exercise's broad 2.28 guard. The guide uses the exact 2.31
   floor for its selected setup path and checks network capability separately.
6. Some MPI starter paths omit producer-stream completion before passing GPU
   buffers into MPI. Device-buffer awareness is not implicit stream ordering.
7. One- and two-row partitions, uneven ownership, zero-owned ranks, typed
   transport precision, and symmetric destination offsets need explicit rules.
8. The lecture summary calls high-priority streams required for overlap.
   Priority is a hint. Correct dependencies, independent work, and available
   execution resources determine whether overlap can occur.

The relevant sources are the
[MPI solution](https://github.com/FZJ-JSC/tutorial-multi-gpu/blob/cd34f9d40d600efd8d84e433b1a9e84ab5eff5f3/04-H_Overlap_Communication_and_Computation_MPI/solutions/jacobi.cpp),
[host NCCL solution](https://github.com/FZJ-JSC/tutorial-multi-gpu/blob/cd34f9d40d600efd8d84e433b1a9e84ab5eff5f3/07-H_NCCL_NVSHMEM/solutions/NCCL/jacobi.cpp),
[graph solution](https://github.com/FZJ-JSC/tutorial-multi-gpu/blob/cd34f9d40d600efd8d84e433b1a9e84ab5eff5f3/09-H_Device-initiated_Communication_and_CUDA_Graphs/solutions/Using_CUDA_Graphs/jacobi.cpp),
and [device NCCL kernel](https://github.com/FZJ-JSC/tutorial-multi-gpu/blob/cd34f9d40d600efd8d84e433b1a9e84ab5eff5f3/09-H_Device-initiated_Communication_and_CUDA_Graphs/solutions/Device-initiated_Communication_with_NCCL/jacobi_kernels.cu).
The guides cite current primary API authorities beside their contracts.

## Measurement and reuse limits

The tutorial CI builds six solutions in an HPC SDK container. It does not run
their multi-GPU outputs. The slides mix machine-specific historical results
with newer examples. In particular, the device-communication deck labels some
IBGDA results as an experimental library on a separate internal cluster with a
different network. Those plots are source context, not measured wins for this
repository or an equivalent-hardware transport comparison.

The repository-level tutorial license is MIT, while derived NVIDIA source
retains its own notices. Preserve the applicable file notices and cite both
the tutorial and upstream examples when adapting material. KokkosComm source
and its independent dependencies were not retrieved or incorporated.

## Validation record

Focused CPU tests exercise the real C++ decomposition and numerical reference
with an independent Python stencil oracle. They include uneven, one-row, and
two-row partitions, invalid inputs, full cell comparisons, and odd/even step
counts. The Python entrypoint and CLI registry use real help and
unsupported-capability paths. The earlier CPU and CLI pass completed 55 tests,
with six opt-in GPU integration cases skipped before devices were probed.

Syntax and bytecode checks passed for ten Python modules. The two benchmark
wrappers passed the scoped AST linter with no errors or warnings. All 42 local
links in the new guides, lab documents, and this review resolve. The separately
checked chapter README additions also resolve. These are focused checks, not a
full repository test run.

The earlier CPU and CLI test commands were:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-project --with pytest python -m pytest -q -rs -o timeout=0 code/tests/test_multi_gpu_communication.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-project --with pytest python -m pytest -q -rs -o timeout=0 code/tests/test_tools_cli.py
```

The real `aisp tools multi-gpu-communication -- --help` entrypoint also passed.
The harness pair currently supports `--profile none`. Its generic inherited
profiler launch would not preserve the native MPI invocation, so requests for
that unsupported path are rejected explicitly. Per-rank native Nsight commands
are documented.

The subsequent GPU integration ran as Slurm job `1120` on two NVIDIA B200 GPUs
connected by NVLink on one host. It used driver `580.173.02`, CUDA 13,
NCCL `2.26.2`, NVSHMEM `3.7.2`, and an isolated CUDA-aware HPC-X/Open MPI
`4.1.7a1` runtime. The executed source was an overlay on base commit
`275419650b448e2aedc461531dc28d2e86a995c7`.

The first real build exposed two integration issues. The Makefile's default
target was corrected to build the native executable, and MPI include flags
were supplied to `nvcc`. Both fixes preceded the passing GPU integration.

The GPU selection completed with **6 passed, 55 deselected in 18.79 seconds**.
Those six cases exercised MPI, NCCL, NVSHMEM, registered buffers, graph replay,
and odd/even iteration counts across selected combinations, with complete
output checks against the stencil reference. The elapsed time is the pytest
run duration, not a per-step performance result. This single-host diagnostic
is non-canonical and does not cover every backend/mode combination or establish
multi-node behavior.

A separate two-B200 diagnostic completed ordinary-buffer graph memcheck with
zero errors on both ranks, native Nsight Systems `2025.3.2` captures for both
schedules and ranks, and Nsight Compute `2025.3.1` capture of one independent
local stencil kernel per rank. NCU replay excluded communication kernels and
is not used as an application latency measurement. The repository clock-lock
context controlled both GPUs at application clocks of 1965 MHz SM and 3996 MHz
memory throughout this stage.

The timing workload held NCCL, eager execution, ordinary buffers, two ranks,
`nx=4096`, `ny=4097`, 20 fixed steps, and two warmup steps constant. Four
unprofiled samples per schedule ran in the order serialized, overlap, overlap,
serialized, serialized, overlap, overlap, serialized. Each completed mesh passed
its CPU reference with maximum absolute error zero.

| Schedule | Median step | Minimum to maximum | Sample standard deviation |
|---|---:|---:|---:|
| Serialized | 42.997 µs | 42.507 to 43.713 µs | 0.510 µs |
| Overlapped | 28.141 µs | 27.654 to 28.349 µs | 0.309 µs |

The ratio of unrounded medians is **1.528x**. Timing starts after warmup and
finishes after both GPU streams complete, takes the maximum rank duration,
and divides by the fixed step count. Setup, process startup, and CPU reference
checks are excluded. These short diagnostic samples do not establish canonical
performance qualification, end-to-end latency, or multi-node scaling.

Nsight analysis selects the exact `compute_graph:jacobi_steady_state` range
and intersects unions of Jacobi and NCCL kernel intervals within the same
process, device, and CUDA context. It counts overlapping NCCL intervals once.
The serialized control has zero intersection on both GPUs. The overlapped
captures show 284927 ns and 279455 ns of intersection, covering **95.414%** and
**95.589%** of their respective NCCL kernel interval unions. This establishes
concurrent kernel lifetimes for the intended schedule. It does not measure
transferred bytes or active SM concurrency. The timing comparison above uses
separate unprofiled runs.

The registered-buffer graph memcheck attempt completed its numerical check but
returned exit code 99 on three CUDA API reports per rank. One originated in
UCC initialization, and two came from NCCL's initial FABRIC allocation attempts.
The exact [NCCL 2.26.2 implementation](https://github.com/NVIDIA/nccl/blob/f44ac759fee12ecb3cc6891e9e739a000f66fd70/src/init.cc#L2293-L2323)
retries `cuMemCreate` without FABRIC after `NOT_PERMITTED` or `NOT_SUPPORTED`.
There were no invalid memory access findings, but this attempt is not a clean
sanitizer pass. The later ordinary-buffer stage explicitly excluded unused
UCC and HCOLL collectives and retained ordinary NCCL allocation behavior.
Successful registration still does not prove that a zero-copy transport was
selected.

Both real Python benchmark wrappers also completed setup, native execution,
post-timing payload capture, input/output retrieval, result validation, and
teardown on the B200 pair. Their complete 67-by-64 meshes agreed exactly after
five fixed steps. This was an uncapped wrapper lifecycle smoke, not a full CLI
harness run. The existing harness requires finite stage timeouts, which conflict
with the uncapped execution instruction for this task. Its separate CLI help
and target-control paths were checked without claiming a measurement pass.

The [machine-readable B200 record](../../code/labs/multi_gpu_communication/b200_validation.json)
contains the exact samples, selected case matrix, source hashes, overlap
measurements, and hashes of the retained native profiler captures and SQLite
exports. Native reports and complete timed meshes are retained separately from
public source because captures include host metadata.

## Default-tool profiling refresh and slide coverage

A fresh two-B200 run used the existing default Nsight Systems `2025.3.2` and
Nsight Compute `2025.3.1`. It completed serialized eager, overlapped eager, and
overlapped graph captures with the same native binary and the same 4096-by-4097,
20-step workload. All completed meshes passed the independent CPU reference
with maximum absolute error zero. Application clocks were locked on both GPUs.
Default command paths and file hashes matched before and after the run.

Six native Systems reports, six SQLite exports, two native Compute reports,
and both raw counter exports are retained. The exact steady-state range shows:

| Schedule | GPU 0 NCCL interval concurrency | GPU 1 NCCL interval concurrency |
|---|---:|---:|
| Serialized eager | 0% | 0% |
| Overlapped eager | 95.449% | 92.153% |
| Overlapped graph | 95.114% | 99.425% |

These fractions intersect unions of timestamped Jacobi and NCCL kernel
lifetimes. They do not measure payload bytes, active SM overlap, or a graph
speedup. The earlier unprofiled timing samples remain a separate experiment.
The Compute captures each contain one local stencil kernel, with separate CSV
header and units rows. They exclude communication kernels.

The graph traces confirm the lifecycle rather than inferring it from a diagram.
Each rank records two begin/end capture pairs and two executable instantiations
in setup. Twenty steady solver steps use ten two-step graph launches, producing
60 Jacobi and 20 NCCL kernel records per rank. Eight distinct graph kernel nodes
execute ten times each. The two warmup graph launches stay outside the steady
range. API call counts establish this structure and do not measure time saved.

The [Chapter 12 lifecycle guide](../../code/ch12/cuda_graph_lifecycle.md) and
[graph exercises](../../code/labs/multi_gpu_communication/TASKS.md#5-capture-a-stable-schedule)
cover the supplied graph slides: operation nodes and dependencies, definition,
instantiation, replay, capture fork/join, pointer parity, executable updates,
instantiation flags, and convergence checks outside capture. They explain the
actual scope of existing examples and separate DOT structure from measured
node-level execution.

The [NVSHMEM guide](../../code/ch04/nvshmem_communication.md) connects blocking
host submission, CUDA stream ordering, device initiation, completion, receiver
notification, and safe reuse. The separate device lab below implements these
protocols. The common lab's NVSHMEM backend remains host-on-stream communication.

## Runnable NVSHMEM device follow-through

The [device lab](../../code/labs/nvshmem_device/README.md) adds five native modes:
`scalar-halo`, `persistent-halo`, `block-signal`, `persistent-signal`, and
`persistent-atomic`. The first two compute the same fixed-step stencil. The
other three compute the same directed two-edge payload and retain every
incoming element at every epoch. Stencil timing and tile timing are separate
comparisons.

The waiting kernels use one CTA per PE with a checked cooperative-launch
capacity. `nvshmemx_collective_launch` launches locally on the calling PE's GPU
and does not coordinate launches across PEs. Every required PE still runs the
matching protocol. The single CTA makes local barriers cover all producer and
consumer threads. It is not a scalable multi-CTA persistent Jacobi scheduler.
The persistent stencil acknowledges a halo only after all local stencil reads
finish and protects the alternating remote destination before reuse.

All nine real two-B200 integration cases passed, including native checks and
an independent Python oracle for the complete output. Both block-signal and
persistent-signal benchmark wrapper lifecycles passed, and their complete
histories agreed exactly. The local CPU and CLI selection passed 71 tests, with
nine opt-in GPU cases skipped locally before device probing. Eight interleaved
unprofiled tile samples retained zero-error complete outputs. These results do
not establish an overall Jacobi performance win, multi-node behavior, or
canonical performance.

Nsight Systems `2025.3.2` captures completed for both PEs in all five modes with
verified numerical output. Those earlier captures retain their executed source
and original metadata, including the field that had reported API version `1.3`
under the library-version name. The final source separates runtime/header
NVSHMEM `3.7.2` from OpenSHMEM API `1.3`. Final correctness and profiling passed
after that metadata correction in jobs `1137` and `1138`. Their raw completion
records identify the corrected native SHA-256 below. The earlier `55f2337`
snapshot preserves the initial Nsys source, not those later corrected bytes.
The corrected source is now pinned by `ed588c27`. Its native SHA-256 is
`aa0f7f0fac1c453290098aabaebd957bf4ecdf91f7b5198dad14be5ebafb5d86`.

The existing default Nsight Compute `2025.3.1` completed ten reports: one steady
kernel for each mode and each selected rank. NCU ran inside MPI on one rank,
while the peer ran the native executable directly. Each report's native API
metadata and CSV export show one profiling pass and zero profiler warmup.
All ten invocations passed native and independent Python complete-output checks
with maximum error zero. Backup-byte metrics are absent, so the record keeps
them null. No-save/restore is inferred from NVIDIA's documented single-pass
kernel-replay contract, rather than a measured zero-byte value.

These are selected-kernel durations, with scalar/block kernels representing
one step/epoch and persistent kernels containing the full fixed loop. They
are not comparable solve latencies or full hardware-counter analyses. The
earlier outer application-range attempts remain incomplete diagnostics. Both
profilers' default executable paths remained unchanged, and the repository
clock-lock context controlled both allocated GPUs. The
[tested lab commands](../../code/labs/nvshmem_device/README.md#profile-the-protocol-that-actually-ran)
include the per-rank selector, CPU report inspector, and independent full-output
check. The [device B200 record](../../code/labs/nvshmem_device/b200_validation.json)
retains the source snapshots, case matrix, versions, measurements, and native
capture hashes.

This integration is not a publication-readiness verdict or a whole-repository
pass. NCCL device API fragments, GIN/IBGDA, multi-node transport, larger rank
counts, and untested backend/mode combinations require their own runtime
evidence. Scoped independent reviews do not complete the publication-readiness
check.

## Committed-source rerun

The [November 16 record](2026-11-16-distributed-gpu-validation.json) binds a fresh
run to committed source `ed588c27`, complete test inputs, build options, exact
executables, and retained outputs. All 15 GPU cases and four real benchmark
wrapper lifecycles passed in job `1141`. The full CLI benchmark pipeline remains
unexecuted.

Jobs `1142` and `1143` retained 16 native Nsys reports with complete SQLite
exports and 12 unique NCU reports with raw CSV and native API exports. Independent
CPU analysis rechecked 20 full outputs, totaling 50,780,674 float values, with
maximum error zero. Both eager and graph schedules overlapped compute and NCCL
in every steady iteration on both ranks. The serialized control had no overlap.
These are diagnostic mechanism observations, not new speedup measurements.

The first collector failed on a Python stderr preamble after all Nsys captures
and its first NCU capture completed. The continuation reused that first report
and numerical output byte for byte, then completed the remaining captures.
Both attempts remain separately identified. No failed attempt was relabeled.

The existing default Nsight Systems `2025.3.2.474` and Nsight Compute
`2025.3.1.0` remained unchanged. Fresh capture used private namespaces and
synthetic host identity. Exact-byte privacy review covered the native assets
and every SQLite row. Static MPI vendor build metadata and GPU UUIDs remain
disclosed. Older operational captures remain private.

The native device source later received a comment-only replay-safety correction.
No native executable statement changed after these runs. The validation record keeps
both the captured and maintained hashes. Documentation fixes also correct graph
diagnostics and add explicit limits beside older unvalidated examples.

The [diagnostic release](https://github.com/cfregly/ai-performance-engineering/releases/tag/tutorial-multi-gpu-b200-20261116)
retains native reports, exports, meshes, collection sources, analysis, and hash
ledgers. The [package audit](2026-11-16-distributed-gpu-package-audit.json) records
the exact asset gate. Three absolute pytest convenience links were omitted from
the public smoke archive. All 99 manifested regular files kept their original
bytes. The original archive remains retained separately.
