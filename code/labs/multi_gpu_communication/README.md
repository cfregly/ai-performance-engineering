# One workload, several multi-GPU communication choices

This lab uses a fixed-iteration Jacobi solver to explain NCCL groups, compute and
communication overlap, reusable registered buffers, and graph replay. Compare
the same numerical work before deciding whether a more complex schedule helps.
Start with the [guided exercises](TASKS.md) or the
[NCCL groups and buffer guide](../../ch04/nccl_groups.md).

## What changes in each comparison?

| Comparison | Keep fixed | Change | Main cost or constraint |
|---|---|---|---|
| Serialized versus overlapped exchange | Backend, grid, ranks, arithmetic, iterations, allocation, graph mode | Schedule boundary, interior, and halo work | Extra streams, events, and split launches can outweigh hidden communication |
| Ordinary versus registered NCCL buffers | Workload and schedule | Allocation and explicit registration | Registration costs setup time and resources, and benefits depend on the selected NCCL path |
| Eager versus graph execution | Workload, schedule, registration, and rank mapping | Submit repeated steps through captured graphs | Capture and instantiation need reuse to repay setup and require valid pointer lifetimes |
| CUDA-aware MPI, host NCCL, and stream-ordered NVSHMEM | Numerical workload and stopping policy | Communication interface and its completion rules | MPI needs producer synchronization, while NVSHMEM needs symmetric allocation and explicit remote completion |

These are separate experiments. Changing scheduling, registration, and graph
mode together does not identify which mechanism caused a timing difference.
NCCL grouping is part of the correct P2P exchange in both scheduling variants.
An unsafe ungrouped exchange is not a useful performance baseline.

## Numerical contract

The global grid has `nx` columns and `ny` rows. The `ny - 2` owned rows are split
across ranks. The first and last global rows are periodic y halos, and the x
edges retain their initialized Dirichlet values. Every iteration applies the
same float32 five-point stencil. There is no convergence early exit or
variable-frequency norm reduction in this comparison.

Partitions may be uneven. Each rank must own at least one row. One-row and
two-row partitions need special care because their boundary regions can meet
and their interior can be empty. The implementation must not update a boundary
twice. Odd iteration counts must read the final ping-pong buffer.

The native CPU reference checks every rank's owned rows and actual received
halos. Nonfinite values are failures. The gathered dump reconstructs periodic
halo rows from gathered owned rows. Python checks of that dump validate the
mesh and reconstructed boundaries, not the final remote halo transfer
independently. Reference computation, output
verification, setup, and warmup are excluded from the reported steady-state
solve interval. That interval must include completion of all compute and halo
work. Report the slowest rank rather than treating rank zero's local duration
as the whole application.

## Run the native component

Use a Git clone and an allocated GPU environment with CUDA, an MPI C++ compiler,
and NCCL. Build receipts require Git. Build
against the same headers and libraries used at runtime. The native process uses
one GPU per rank. Set the launcher's GPU visibility and binding explicitly.
Install the MPI development headers as well as its wrapper and runtime. The
Makefile discovers Open MPI compile flags with `mpicxx --showme:compile` because
`nvcc` preprocesses CUDA before invoking the host compiler wrapper. For another
MPI implementation, supply its include flags through `MPI_COMPILE_FLAGS`.

The Python launcher requires a POSIX Python environment with `psutil` and
`waitid`/`WNOWAIT`. Linux Python provides the required process controls. SIGINT
must not be ignored, SIGCHLD must have its default disposition, and no other
component may reap the owned launcher. Unsupported
signal dispositions are rejected before launch. The launcher retains stdout,
stderr, and an `.execution.json` receipt beside the result, including after
interruption. Each MPI launch runs in its own owned process group. The Python
caller waits for that group to become quiescent before returning. If a descendant
escapes into another process group while retaining a capture pipe, the launcher
drains the available bytes and closes its read end once the owned group is
quiescent. The receipt records this limit in `execution.capture_limit`. The
launcher does not signal the escaped process or retain its later output.

From this directory:

```bash
JACOBI_RESULTS=$(mktemp -d /tmp/jacobi-lab.XXXXXX)
export JACOBI_RESULTS
make NVTX_ENABLED=1
mpirun -n 2 ./jacobi_native --backend nccl --schedule serialized \
  --nx 256 --ny 257 --iterations 5 --warmup 1 --json-out "$JACOBI_RESULTS/serialized.json"
mpirun -n 2 ./jacobi_native --backend nccl --schedule overlap \
  --nx 256 --ny 257 --iterations 5 --warmup 1 --json-out "$JACOBI_RESULTS/overlap.json"
```

The deliberately uneven grid and odd step count exercise decomposition and
buffer parity. They are a correctness smoke check, not a workload expected to
demonstrate a speedup. Larger shapes need a separate, recorded workload choice.
Do not substitute unsupported capability or failed verification with another
backend under the same result name.

Inspect the Python runner without launching a GPU process:

```bash
python -m labs.multi_gpu_communication.run_lab_multi_gpu_communication --help
python -m cli.aisp tools multi-gpu-communication -- --help
```

Run these module commands from `code/`. The runner validates the build receipt
and launches the selected native binary. Build it with the Makefile first.
Direct native runs are component diagnostics. The receipt records the Git
commit, compiler versions, build options, binary hash, and hashes of three
files: `jacobi_native.cu`, `jacobi_contract.hpp`, and `Makefile`. The runner
checks those four hashes and refuses to reuse an existing result path. It does
not hash every transitive header or build include, such as `nvtx_utils.cuh` or
`cuda_arch.mk`. Rebuild after changing those dependencies.

Use `mpirun`, `mpiexec`, or a compatible wrapper for `--launcher`. The runner
passes `-np` and does not support `srun` directly. Pass a fresh `--json-out` path
to retain results after the runner exits. A native JSON status alone is not a
runner acceptance result. Require a successful process exit and completed Python
validation. A rejected run can leave its original native JSON with `status: passed`.
The runner exits `3` for an unavailable capability and `1` for a failed run.

For CUDA-aware MPI, select `--backend mpi`. Automatic capability detection uses
Open MPI's `MPIX_Query_cuda_support` when that extension is available. For an
MPI implementation without that query, pass `--assert-cuda-aware-mpi` only after
confirming its documented GPU-buffer support and runtime configuration. The
flag records the caller's assertion. It does not enable CUDA support. That
asserted path has not been GPU-tested in the retained matrix.
MPI does not receive a CUDA stream argument here, so complete the
boundary producer before passing its buffer to MPI. The interior kernel can
continue independently during the host's halo exchange. See
[Open MPI's capability checks](https://docs.open-mpi.org/en/main/tuning-apps/accelerators/cuda.html).

The optional NVSHMEM build uses the same stencil with stream-ordered puts and
a stream-ordered all-PE barrier. From this directory, rebuild with the installed
NVSHMEM path, then select its backend:

```bash
make -B NVTX_ENABLED=1 WITH_NVSHMEM=1 NVSHMEM_HOME="$NVSHMEM_HOME"
mpirun -n 2 ./jacobi_native --backend nvshmem --schedule overlap \
  --nx 256 --ny 257 --iterations 5 --warmup 1 --json-out "$JACOBI_RESULTS/nvshmem.json"
```

Set `NVSHMEM_HOME` to the installed library prefix first. Every PE allocates the
same maximum partition size for both fields, even when its owned row count is
smaller. Halo destinations use the receiver's row count. MPI bootstraps the
NVSHMEM job, and the implementation checks that rank and PE identities agree.
This arm uses host-submitted stream operations. It does not claim in-kernel
device initiation or an IBGDA transport.

For this NVSHMEM arm, every local PE must see the same full GPU set in the same
order before selecting one GPU per local rank. Pass the same
`CUDA_VISIBLE_DEVICES` value to every PE, as required by
[current NVSHMEM guidance](https://docs.nvidia.com/nvshmem/api/latest/faq.html).
The lab compares that string and checks physical rank assignment. String
equality does not prove full visibility when a container or device cgroup masks
GPUs separately. The launcher must establish this precondition. The lab does
not compare complete visible-device UUID lists across local PEs.
Per-node UUID masks with different strings are outside this arm's supported
configuration. The MPI and NCCL arms retain ordinary explicit per-rank masks.

## Run the equivalent harness pair

The pair uses the same numerical source and workload options, changes the
schedule, and retains the complete timed mesh for verification. Each wrapper's
setup rebuilds the executable separately. The retained pair therefore used
two binaries from the same source, not one shared binary or bit-reproducible
builds. From `code/`:

```bash
python -m cli.aisp bench list-targets --chapter labs/multi_gpu_communication
python -m cli.aisp bench run --targets labs/multi_gpu_communication:jacobi --profile none \
  --suite-timeout 0 --execution-timeout-seconds 0
python -m cli.aisp bench run --targets labs/multi_gpu_communication:jacobi --profile none \
  --suite-timeout 0 --execution-timeout-seconds 0 \
  --target-extra-arg 'labs/multi_gpu_communication:jacobi=--nx 4096 --ny 4097 --steps 20 --native-warmup 2 --ranks 2'
```

The real wrapper lifecycles were tested on B200. The full `bench run` pipeline
above has not been executed for this lab. Harness overrides use `--steps` and
`--native-warmup`, while the native executable uses `--iterations` and `--warmup`.
Two ranks are the default because this is the validated topology.

These commands require both timeout flags to be explicitly set to zero. They
disable the harness suite and execution-stage deadlines for the owned isolated
subprocess. A finite outer worker timeout cannot safely clean up the native MPI
launcher's separate process group, so both benchmark variants reject finite
execution before the build starts. Standalone native launches remain supported.
The native build and launch have no time deadline. Both use owned cooperative
process groups and wait for cleanup after Ctrl-C. The caller retains bytes it
has read from stdout and stderr. Output buffered by a child but never flushed is
outside this guarantee. The generic harness profiler remains disabled for these
native MPI wrappers.
Environment, numerical, and speedup gates remain active.
See the [execution deadline contract](../../docs/api-reference.md#8-benchmark-domain).

The `256 × 257` default is a deliberate correctness-smoke exception to the
repository's usual speedup-oriented benchmark defaults. It is intended for
small correctness checks. The retained wrapper smoke used a smaller `17 × 7`
grid and did not show a win. Overlap is not guaranteed to win at either size.
Choose and record a representative workload separately before evaluating
performance.

Use the same target overrides for both variants when changing shape, native
step count, ranks, backend, registration, or graph mode. Direct native timing
reports one completed fixed-iteration solve per invocation. The harness consumes
its per-step time and actual output instead of timing process startup as GPU
work.

The current wrapper supports `--profile none`. Its inherited single-process
profiler path cannot launch this MPI workload correctly, so profiler requests
fail explicitly. Use the per-rank native Nsight command below for diagnostic
captures. Canonical performance qualification still needs the repository's
clock, provenance, repeated-measurement, verification, and required profiler
gates. The B200 diagnostic below does not complete that qualification.

## Read the overlap schedule

The serialized schedule completes local compute before exchanging halos. The
overlapped schedule produces boundary rows on the communication stream and
interior rows on a separate compute stream. The grouped NCCL receives and sends
follow the boundary producer on its stream. Join both paths before the next
iteration reads their output.

The communication stream uses the highest priority returned by CUDA's priority
query. That is a scheduling hint that favors higher-priority pending work.
It does not guarantee preemption or simultaneous execution. Correctness still
requires dependencies. A group that mixes
the independent compute stream into the NCCL phase can introduce a join and
remove overlap. The [group guide](../../ch04/nccl_groups.md) explains the exact
submission and completion boundaries.

## Test registration and capture independently

For supported NCCL builds, add `--registered-buffers` to both schedule runs.
The implementation allocates eligible buffers, registers both ping-pong
allocations, completes their last use, deregisters them, and then frees them.
Every rank must make the same registration choice. Allocation and registration
are separate operations. Successful registration does not prove the selected
transport avoided internal copies.

For NCCL graph execution, add `--graph` to otherwise equivalent runs. Captured
steps contain fixed pointer bindings. Replaying a two-step graph returns the
output to the original buffer. A separate odd tail must finish in the other
buffer. All ranks need compatible capture and replay participation. MPI calls
on the host are not silently converted into graph nodes. See
[NCCL's graph requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html).
This implementation also preserves captured node priorities and requires CUDA
11.7 or newer headers for that instantiation path.

Record allocation, communicator setup, registration, capture, warmup, and
steady-state costs separately where the artifact provides them. Fixed-step
graph results do not establish convergence-controlled solver performance.
With graph mode and zero warmup, first-launch upload is included in the timed
interval. The recorded graph profiles used warmup outside that interval.

## Verify overlap with native profiler evidence

Build with `NVTX_ENABLED=1` so the steady-state NVTX range used by the analysis
is present. Keep the build receipt with the captures. The native JSON does not
currently report this compile-time flag. Capture one report per rank using your
site's normal launcher:

```bash
JACOBI_PROFILE=$(mktemp -d /tmp/jacobi-profile.XXXXXX)
export JACOBI_PROFILE
mpirun -n 2 nsys profile --trace=cuda,nvtx,mpi \
  --cuda-graph-trace=node \
  --output="$JACOBI_PROFILE/jacobi-overlap.%q{OMPI_COMM_WORLD_RANK}" \
  ./jacobi_native --backend nccl --schedule overlap \
  --nx 4096 --ny 4097 --iterations 20 --warmup 2 --json-out "$JACOBI_PROFILE/profiled.json"
```

To inspect graph replay with the same workload, use a separate report prefix
and result path, and add `--graph` to the native command:

```bash
mpirun -n 2 nsys profile --trace=cuda,nvtx,mpi \
  --cuda-graph-trace=node \
  --output="$JACOBI_PROFILE/jacobi-graph-overlap.%q{OMPI_COMM_WORLD_RANK}" \
  ./jacobi_native --backend nccl --schedule overlap --graph \
  --nx 4096 --ny 4097 --iterations 20 --warmup 2 --json-out "$JACOBI_PROFILE/profiled-graph.json"
```

The node trace flag exposes graph kernel intervals used by the overlap
analysis. A graph-level trace alone does not supply those node intervals.

Native captures, exports, logs, and JSON can contain private paths and process
metadata. Keep them outside the source tree and audit exact bytes before sharing.
The historical captures behind the older percentages below remain private.

Inspect whether interior kernels run alongside halo communication, whether
boundary production delays communication, and whether the final dependency
protects the next iteration. Retain the native `.nsys-rep` files and source
identity. Keep these intrusive profiler timings separate from unprofiled
latency comparisons. A successful run, lower host enqueue time, or multiple
stream names does not prove GPU overlap.

## Current-source B200 verification

On November 16, all six common-lab GPU cases and both real wrapper lifecycles
passed at committed source `ed588c27`. The same run passed all nine device-lab
cases and both device wrappers. The
[current validation record](../../../docs/reviews/2026-11-16-distributed-gpu-validation.json)
binds the tests, source and binary hashes, workload arguments, and captures.
The [diagnostic release](https://github.com/cfregly/ai-performance-engineering/releases/tag/tutorial-multi-gpu-b200-20261116)
retains the audited native reports, existing exports, full outputs, and checksums.

The original MPI and NVSHMEM cases used a `17 × 5` grid with no interior-kernel
work. A later [two-case followup](../../../docs/reviews/2026-11-16-host-interior-followup.json)
passed both backends at source `96a523686` with a `17 × 67` grid, five steps,
and one warmup step. Each rank had a nonempty interior. Both complete outputs
matched the independent float32 reference with zero maximum error. That run
checks correctness of the overlap schedule. It did not collect a timeline, so
observed GPU overlap remains specific to the NCCL profiles below.

Fresh default-tool profiling retained six common-lab Nsys reports and two
single-pass NCU stencil reports, alongside the device-lab captures. The common
Nsys workload was `4096 × 4097`, 20 steps, and two warmup steps. Intersections
inside `compute_graph:jacobi_steady_state` measured:

| Schedule | Iterations with compute/NCCL overlap, each rank | NCCL interval time overlapped, ranks 0 / 1 |
| --- | ---: | ---: |
| Serialized | 0 / 20 | 0% / 0% |
| Eager overlap | 20 / 20 | 96.00% / 95.85% |
| Graph overlap | 20 / 20 | 100.00% / 98.79% |

These are intersections of GPU kernel intervals from the node-level traces.
They establish overlap for the captured workload, not a new unprofiled speedup.
NCU covers selected local stencil duration, not the full communication path.
The first collector failed on an NCU CSV preamble after completing all Nsys
captures. Its continuation passed, and both attempts remain identified in the
record. Job 1143 reused job 1142's stencil NCU report, native JSON, and output
for rank 0 byte for byte, then re-exported the report. The existing default
`nsys` and `ncu` were unchanged.

## Historical two-B200 diagnostic

Six GPU integration cases passed on two NVLink-connected NVIDIA B200 GPUs on
one host. They exercised MPI, NCCL, NVSHMEM, registered buffers, graph replay,
and odd/even iteration counts across selected combinations. See the
[integration review](../../../docs/reviews/2026-11-15-distributed-gpu-tutorial.md#validation-record)
for runtime versions, source identity, build fixes, and validation scope. The
[machine-readable record](b200_validation.json) retains exact samples and native
profiler hashes.

A separate NCCL comparison used `nx=4096`, `ny=4097`, 20 fixed steps, and two
warmup steps. Both schedules used eager execution and ordinary unregistered
buffers. Four interleaved unprofiled samples per schedule produced these median
step times:

| Schedule | Median step time |
| --- | ---: |
| Serialized | 43.0 µs |
| Overlapped | 28.1 µs |

The reported median ratio was **1.528x**. The displayed medians are rounded.
These are native solve timings with setup and the CPU reference excluded, not
whole-process latency. Four samples per schedule are a small diagnostic, not
canonical performance qualification or a multi-host scaling result.

Nsight Systems `2025.3.2` captured both schedules with a report per rank. In the
overlapped captures, Jacobi ran concurrently with **95.4136%** of the time covered
by NCCL kernel intervals on GPU 0 and **95.5889%** on GPU 1. Overlapping NCCL
intervals were counted once. The serialized control showed **0%** on both GPUs.
This supports the intended overlap mechanism for those captures.
The unprofiled timings above are separate from the intrusive profiler run.

Compute Sanitizer memcheck reported zero errors on both ranks for the
ordinary-buffer graph path. The registered-buffer graph run exited `99` after
an API error during UCC initialization and the expected NCCL `cuMemCreate`
fabric-retry reports. It reported no illegal memory access, but it is not a clean
sanitizer pass. NCU `2025.3.1` completed a local stencil-kernel capture only.
That does not qualify the complete communication path.

The clean historical memcheck record names unused UCC and HCOLL MPI collectives
as excluded, but does not retain the exact exclusion command. The recipe below
uses Open MPI's component selection for ordinary buffers and the same workload.
It has not been rerun as written. With the site's normal MPI environment,
retain a separate log for each rank:

```bash
JACOBI_MEMCHECK=$(mktemp -d /tmp/jacobi-memcheck.XXXXXX)
export JACOBI_MEMCHECK
OMPI_MCA_coll='^ucc,hcoll' mpirun -n 2 compute-sanitizer --tool memcheck \
  --error-exitcode 99 --log-file "$JACOBI_MEMCHECK/memcheck.%q{OMPI_COMM_WORLD_RANK}.log" \
  ./jacobi_native --backend nccl --schedule overlap --nx 17 --ny 7 \
  --iterations 5 --warmup 1 --graph --json-out "$JACOBI_MEMCHECK/result.json" \
  --dump-output "$JACOBI_MEMCHECK/output.f32"
```

This recipe is specific to an Open MPI setup with those optional components.
It does not suppress CUDA memory findings or turn the registered-buffer
attempt into a clean pass.

Both Python benchmark wrappers also passed a real uncapped lifecycle smoke,
including native execution and post-timing payload capture. Their complete
outputs agreed exactly. This is separate from a full CLI harness measurement.

These results establish the behavior observed in the selected single-host
cases. Registration-path selection, other backend/mode combinations, other
hardware and runtimes, and multi-host behavior need separate evidence.

The default-tool profiling refresh also retained node-level traces for graph
replay. Each rank instantiated the pair and odd-tail executables once during
setup. Its 20 steady solver steps submitted ten pair-graph launches and executed
60 stencil kernels plus 20 NCCL kernels. The two graph warmup launches were
outside the timed range. This verifies replay structure without treating
submission counts or intrusive profiler timings as a graph speedup.
The [lifecycle guide](../../ch12/cuda_graph_lifecycle.md) explains the distinction
and maps the requested node types to actual repository examples.

## Extend the lesson without duplicating existing examples

The [NVSHMEM guide](../../ch04/nvshmem_communication.md) explains symmetric
allocation, host/stream/device initiation, completion, and safe slot reuse.
Chapter 4 already has native NVSHMEM queue and pipeline examples. Its Python
pipeline uses documented CPU/Gloo sidebands and must not be described as native
GPU signaling.

The [NCCL device guide](../../ch04/nccl_device_communication.md) adds LSA/GIN,
symmetric windows, signal epochs, and consumed acknowledgments. Its fragments
are educational. They do not constitute a validated GIN Jacobi implementation.
The existing [CUDA symmetric-memory ring](../../ch04/symmetric_memory_example.py)
already demonstrates captured multi-rank peer writes and both lifetime fences.

For strong scaling, hold the global grid and iteration count fixed as rank
count grows. For weak scaling, hold owned rows and columns per rank fixed and
set `ny = ranks * owned_rows + 2`. Record rank mapping and actual topology in
both experiments. Keep these application curves distinct from communication
microbenchmarks and planner estimates. Reuse the
[cluster readiness and provenance workflow](../../cluster/README.md).
The native lab requires at least two ranks. Label a two-rank scaling reference
as such, and normalize efficiency relative to that reference rather than
inventing a one-GPU timing.

## Source credit and evidence limits

The learning sequence comes from
[Efficient Distributed GPU Programming for Exascale](https://github.com/FZJ-JSC/tutorial-multi-gpu/tree/cd34f9d40d600efd8d84e433b1a9e84ab5eff5f3)
by Andreas Herten, Lena Oden, Simon Garcia de Gonzalo, Jiri Kraus, and Oded Green
([year-independent DOI](https://doi.org/10.5281/zenodo.5745504))
and its upstream
[NVIDIA programming-model examples](https://github.com/NVIDIA/multi-gpu-programming-models).
Keep upstream license notices with any adapted source. The tutorial's MIT
metadata does not replace NVIDIA notices carried by its derived code.

The B200 results above are diagnostic and non-canonical. They do not establish
publication readiness, a whole-repository pass, a clean registered-graph
sanitizer result, or performance on other systems. Keep those limits with any
reuse of the measured ratio or overlap figures, and retain the native evidence
referenced by the integration review.
