# NVSHMEM device communication

This lab implements five GPU communication modes: two stencil exchanges and
three computed-tile exchanges. Device initiation can remove a host round trip
between production and communication.
It also makes the application responsible for remote visibility, notification,
buffer reuse, and GPU progress. Use the
[host-on-stream communication lab](../multi_gpu_communication/README.md) when a
regular host-enqueued communication phase is sufficient.

The learning sequence builds on the
[FZJ-JSC multi-GPU tutorial](https://github.com/FZJ-JSC/tutorial-multi-gpu/tree/cd34f9d40d600efd8d84e433b1a9e84ab5eff5f3).
The [integration review](../../../docs/reviews/2026-11-15-distributed-gpu-tutorial.md)
records the adaptations and remaining coverage limits.

All five modes' user kernels intentionally use one CTA per PE. The persistent
examples demonstrate a complete repeated protocol, not a multi-CTA persistent
Jacobi solver.
`scalar-halo` and `persistent-halo` compute the same stencil and final output.
The three tile modes compute and retain every epoch's received values. Their
timings do not compare identical work with the stencil solvers.

**Validation state:** All nine two-GPU integration cases passed on two B200
GPUs, with native and independent Python checks of the complete output. The
block-signal and persistent-signal benchmark wrappers also completed, and their
full histories matched exactly. Nsight Systems traces were collected for both
PEs in all five modes. Ten Nsight Compute reports cover one steady kernel per
mode and rank, each with one actual profiling pass and verified complete
outputs. Final validation also passed after correcting version metadata. The
[B200 validation record](b200_validation.json) separates the executed snapshots,
commands, and capture hashes. These checks do not establish an overall Jacobi
performance win or full hardware-counter coverage.

## Choose a mode

| Mode | GPU work | Synchronization and cost |
|---|---|---|
| `scalar-halo` | Each stencil thread computes a value and uses scalar `nvshmem_float_p` for a boundary value. | A host-enqueued on-stream barrier separates steps. Fine-grained puts add operations, and the host still submits each step. |
| `persistent-halo` | One CTA per PE computes all owned stencil cells and exchanges full boundary rows in a kernel loop. | Ready epochs protect halo reads, and delayed consumed acknowledgments protect the reused remote halo buffers. One CTA limits available compute throughput. |
| `block-signal` | A pure `prepare_payload` kernel computes two outgoing tiles, followed by one exchange CTA per PE per epoch. | All CTA threads participate in block puts. Ready and consumed notifications protect each reusable inbox. Repeated launches remain. |
| `persistent-signal` | One CTA per PE computes, exchanges, and records every epoch in a kernel loop. | Cumulative 64-bit ready and consumed signals replace per-epoch host control. The CTA holds execution resources while waiting. |
| `persistent-atomic` | The same persistent tile work uses ordinary integer atomic counters and `wait_until_all`. | Both incoming counters must reach the current epoch. Counter range, ownership, and progress remain application responsibilities. |

The three tile modes use the same payload and history layout. The runner
defaults to two MPI ranks, with one PE and one GPU per rank. The native
source accepts more ranks when the workload and hardware meet its checks. In a
two-PE periodic ring, previous and next are the same peer. The two logical edges
still need separate payload, ready, and consumed slots.

## Build and run

Use Linux with two supported, mutually reachable GPUs, a compatible CUDA toolkit,
MPI, and NVSHMEM. Set `NVSHMEM_HOME` to an installed NVSHMEM distribution. The
build must link the actual NVSHMEM host and device libraries with relocatable
device code. The source queries the supported collective launch size before
launching a kernel that waits for peers. A one-CTA design does not waive that
check. See the [NVSHMEM installation requirements](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html)
and [collective launch contract](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html).

The waiting kernels use `nvshmemx_collective_launch` on each PE. Despite the
name, this call launches only on the calling PE's GPU. It does not synchronize
or coordinate launches across PEs. Its local cooperative launch and capacity
check support resident execution and local CTA synchronization. Every required
PE must still launch its kernel and follow the remote ready/consumed protocol.

From the repository root, select the architecture for the actual GPUs:

```bash
make -C code/labs/nvshmem_device NVSHMEM_HOME=/path/to/nvshmem ARCH=sm_100
python code/labs/nvshmem_device/run_lab_nvshmem_device.py --help
```

The same runner is available through the tools registry. From `code/`:

```bash
python -m cli.aisp tools nvshmem-device -- --help
```

The [Makefile](Makefile) accepts `NVSHMEM_INCLUDE` and `NVSHMEM_LIBDIR` overrides
for installations with different layouts. MPI include discovery uses Open MPI's
`mpicxx --showme:compile`, or explicit `MPI_COMPILE_FLAGS` for another toolchain.
Read the [native implementation](nvshmem_device_native.cu) and
[runner](run_lab_nvshmem_device.py) alongside the protocol below.
Use the intended installation's `nvcc`, `mpicxx`, and `mpirun` on `PATH`, with
the matching MPI and NVSHMEM shared libraries available to every PE. Supply the
site's MPI environment exports and binding arguments explicitly.

Run each protocol with complete output retained:

```bash
python code/labs/nvshmem_device/run_lab_nvshmem_device.py --mode scalar-halo --ranks 2 --nx 17 --ny 7 --iterations 4 --warmup 1 --json-out /tmp/scalar-halo.json --dump-output /tmp/scalar-halo.f32
python code/labs/nvshmem_device/run_lab_nvshmem_device.py --mode persistent-halo --ranks 2 --nx 17 --ny 7 --iterations 4 --warmup 1 --json-out /tmp/persistent-halo.json --dump-output /tmp/persistent-halo.f32
python code/labs/nvshmem_device/run_lab_nvshmem_device.py --mode block-signal --ranks 2 --elements 32 --iterations 4 --warmup 1 --json-out /tmp/block-signal.json --dump-output /tmp/block-signal.f32
python code/labs/nvshmem_device/run_lab_nvshmem_device.py --mode persistent-signal --ranks 2 --elements 32 --iterations 4 --warmup 1 --json-out /tmp/persistent-signal.json --dump-output /tmp/persistent-signal.f32
python code/labs/nvshmem_device/run_lab_nvshmem_device.py --mode persistent-atomic --ranks 2 --elements 32 --iterations 4 --warmup 1 --json-out /tmp/persistent-atomic.json --dump-output /tmp/persistent-atomic.f32
```

Use the runner's explicit `--launcher` and repeatable `--launcher-arg` options for
the site's MPI launch requirements. Use `--binary` only with a matching build
receipt. Missing libraries, stale artifacts, unsupported launch capacity, and
incorrect rank counts must produce explicit failures or capability skips.
The runner must not substitute a CPU or host communication path for a requested
device mode.

Choose fresh artifact paths for each run. The runner refuses to overwrite an
existing result, dump, or log. Without explicit paths, it retains artifacts in
`code/artifacts/nvshmem_device`. The final JSON includes the independent Python
check. The raw native result is retained as `<json-out>.native.json`, with
`<json-out>.stdout.log` and `<json-out>.stderr.log` beside it.

## Trace a tile epoch before changing it

Every PE allocates matching symmetric objects in the same order and size. An
NVSHMEM destination pointer describes an offset in the caller's symmetric
allocation, paired with a destination PE. It is not a raw remote CUDA pointer.
Read the [symmetric-memory guide](../../ch04/nvshmem_communication.md#pgas-does-not-mean-interchangeable-pointers)
before changing the layout.

For outgoing edge 0, a PE writes the previous PE's incoming edge 1. Outgoing edge
1 writes the next PE's incoming edge 0. The exchange then follows this cycle:

1. Compute both outgoing tiles for epoch `e`.
2. Every CTA thread calls both nonblocking block puts with matching arguments.
3. Synchronize the CTA so all issuers have posted their operations. An elected
   thread calls device quiet before publishing both ready notifications.
4. Wait for both incoming ready values to reach `e`, then synchronize the CTA
   before its threads read the inboxes.
5. Copy every incoming element to this epoch's history, then synchronize before
   an elected thread acknowledges both consumed tiles.
6. Wait for both consumed acknowledgments for `e`. Synchronize before reusing
   the payload and inbox storage for the next epoch.

Quiet completes preceding operations issued by this PE. It does not tell a
remote consumer that data is ready and does not mean the consumer has finished.
The ready and consumed phases provide those separate facts. The CTA barriers
make the elected-thread quiet and acknowledgments cover all participating
threads. These choices follow the
[ordering contract](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html).

Signal mode uses dedicated symmetric `uint64_t` signal objects. Atomic mode
uses separate ordinary `int` counters, incremented once per edge and epoch.
Its `nvshmem_int_wait_until_all(counter, 2, nullptr, NVSHMEM_CMP_GE, e)` includes
both counters because a null status mask excludes neither. Do not mix those
counter objects with the signal API or reset a live counter during a protocol.
Waiting for the fixed value one would accept an earlier epoch. See
[signals](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html) and
[point-to-point waits](https://docs.nvidia.com/nvshmem/api/latest/gen/api/sync.html).

All threads must execute a `_block` call with the same arguments and control
flow. A block call inside `if (threadIdx.x == 0)` is invalid. A single CTA also
lets each local barrier cover the complete local producer and consumer. Adding
CTAs requires a new ownership and progress design. It is not a launch-size edit.
NVSHMEM's [CUDA execution model](https://docs.nvidia.com/nvshmem/api/latest/cuda-interactions.html)
explains why a peer wait can deadlock when the matching producer cannot run.

## Keep stencil halo consumption separate

`persistent-halo` has a different consumption point from the tile history copy.
At zero-based step `s`, its one CTA reads the current field and halo epoch `s`
while computing the next owned field. A CTA barrier finishes all those reads
before an elected thread acknowledges incoming epoch `s`, when `s > 0`.
Before overwriting a remote halo in the alternating target buffer, the sender
waits for consumed epoch `s - 1`, when `s >= 2`.

Every CTA thread then participates in the two block puts. The preceding PE's
lower halo uses that receiver's owned-row count. CTA synchronization and device
quiet finish the transfers before ready epoch `s + 1` is published. The CTA
waits for both incoming ready epochs before all threads swap the current and
next pointers. The initial halo needs no consumed acknowledgment. After `I`
steps, halo `I` has arrived but has not been read by another stencil step. The
terminal consumed epoch is therefore `I - 1`, not `I`.

This protection concerns reuse of the remote halo destination. A completed put
has already copied its local source values. A peer's later consumption of that
copy does not keep the sender's owned-row storage live. Host completion still
precedes output verification and freeing symmetric buffers.

Both stencil modes produce the same complete numerical output. Their messages
differ: scalar mode puts each interior boundary element, while persistent mode
puts full rows, including fixed x boundary cells. The persistent version removes
per-step host launches and global on-stream barriers but adds point-to-point
ready/reuse bookkeeping. Its single CTA is a correctness and progress design,
not an efficient multi-block decomposition for large grids.

## Verify the computed work

Both stencil modes check the complete final `float32` grid with periodic y halos
and fixed x boundaries against an independent CPU stencil implementation. The
tile modes retain every incoming element for every timed epoch. For source PE
`p`, source edge `s`, zero-based element `i`, and one-based epoch `e`, the value is:

```text
1 + (((e - 1) * ranks + p) * 2 + s) * elements + i
```

The supported configuration requires
`max(iterations, warmup) * ranks * 2 * elements <= 2^24`, keeping these integers
exactly representable as `float32`. Warmup completes before protocol state and
history are reset for the timed run. This makes a stale epoch, swapped edge,
wrong peer, missing element, or overwritten history observable. A sum or final
inbox alone would lose that evidence.

Tile dumps are row-major `float32` arrays ordered as
`[rank][epoch - 1][incoming_edge][element]`. Incoming edge 0 expects the previous
PE's outgoing edge 1, and incoming edge 1 expects the next PE's outgoing edge 0.
Stencil dumps contain the complete final `[ny][nx]` grid. The Python runner
checks the native output with an independent CPU oracle and rejects nonfinite
or incomplete output. Keep the JSON result, binary dump, and matching source
and build provenance together.

## Compare the benchmark pair

The baseline uses `block-signal`, and the optimized wrapper uses
`persistent-signal`. Both compute and retain the same complete history, with
263 elements per edge and 32 timed epochs by default. From `code/`:

```bash
python -m cli.aisp bench list-targets --chapter labs/nvshmem_device
python -m cli.aisp bench run --targets labs/nvshmem_device:nvshmem_device --profile none
```

Use the native profiling recipe below for communication kernels. The wrappers
reject the generic harness profiler path because MPI execution and live peer
state require explicit collection control. B200 validation exercised both real
wrapper lifecycles and compared their complete outputs. It did not execute the
full `bench run` pipeline, whose mandatory finite stage timeouts conflict with
this validation run's uncapped execution instruction.

## Profile the protocol that actually ran

The validated tools are the existing default Nsight Systems `2025.3.2` and
Nsight Compute `2025.3.1`. Their default executable paths were unchanged. Use
Nsight Systems for all five modes to inspect the real kernel schedule and
host submissions. Retain native reports and identify the steady
NVTX range, `compute_graph:nvshmem_device_steady_state`, in the default NVTX
domain. It includes the run, stream-ordered quiet, and stream completion. A
persistent kernel is one long-lived kernel per PE. A single kernel interval
does not expose the duration of each internal epoch or prove
network transport progress.

Run the profiling commands under the repository's clock-lock context on both
allocated GPUs. From the repository root, define this helper in a Python
environment with the harness dependencies. It restores the previous clock state
when the command exits and does not impose a workload timeout:

```bash
nvdev_locked_run() {
  python - "$@" <<'PY'
from contextlib import ExitStack
import subprocess
import sys
sys.path.insert(0, "code")
from core.harness.benchmark_harness import lock_gpu_clocks
with ExitStack() as locks:
    for device in (0, 1):
        locks.enter_context(lock_gpu_clocks(device=device))
    subprocess.run(sys.argv[1:], check=True)
PY
}
```

Keep its clock telemetry with the results. Use the existing selected `nsys` and
`ncu` on `PATH` and retain their versions. Add the site's normal MPI arguments
and library environment exports to each invocation below.

For a single-host Open MPI launch, collect one Systems report per rank from the
repository root. Add the site's normal MPI arguments where required:

```bash
NVDEV_NSYS_DIR=$(mktemp -d /tmp/nvshmem-device-nsys.XXXXXX)
export NVDEV_NSYS_DIR
nvdev_locked_run mpirun -n 2 -x NVDEV_NSYS_DIR bash -c '
  exec nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    -o "$NVDEV_NSYS_DIR/persistent-halo.${OMPI_COMM_WORLD_RANK}" \
    code/labs/nvshmem_device/nvshmem_device_native \
    --mode persistent-halo --nx 33 --ny 11 --iterations 5 --warmup 1 \
    --json-out "$NVDEV_NSYS_DIR/native.json" \
    --dump-output "$NVDEV_NSYS_DIR/output.f32"
'
```

The native program writes the shared numerical artifacts only on rank zero.
Keep both rank reports and the full output. The dump still needs the independent
Python check shown below.

Do not replay a signaling kernel multiple times against a peer that has already
advanced. The successful Compute recipe launches NCU inside MPI on one selected
rank while its peer runs the native executable directly. The
[tested rank selector](profile_rank.sh) requests duration and profiler metadata,
selects the first matching steady kernel, and lets the remaining workload
finish. It uses no metric set or sections and leaves cache and clock control
to the recorded execution setup. The
[2025.3 CLI](https://docs.nvidia.com/nsight-compute/2025.3/NsightComputeCli/index.html)
documents selected-rank MPI wrappers and kernel/NVTX filters.

```bash
NVDEV_NCU_DIR=$(mktemp -d /tmp/nvshmem-device-ncu.XXXXXX)
export NVDEV_NCU_DIR
nvdev_locked_run mpirun -n 2 bash code/labs/nvshmem_device/profile_rank.sh \
  0 "$NVDEV_NCU_DIR/persistent-halo-p0" persistent_halo \
  "$PWD/code/labs/nvshmem_device/nvshmem_device_native" \
  --mode persistent-halo --nx 33 --ny 11 --iterations 5 --warmup 1 \
  --json-out "$NVDEV_NCU_DIR/native.json" \
  --dump-output "$NVDEV_NCU_DIR/output.f32"
```

Repeat in a fresh output directory with selector argument `1` to profile the
other GPU. Keep both GPUs visible in the same order to both PEs. Only the
selected rank is under NCU. This is one normal MPI invocation per capture.

| Mode | Kernel-name pattern | Captured scope |
|---|---|---|
| `scalar-halo` | `scalar_halo` | One steady stencil step, excluding its following on-stream barrier |
| `persistent-halo` | `persistent_halo` | One kernel containing all fixed stencil steps |
| `block-signal` | `block_exchange_signal` | One steady exchange epoch, excluding `prepare_payload` |
| `persistent-signal` | `persistent_exchange` | One kernel containing all fixed tile epochs |
| `persistent-atomic` | `persistent_exchange` | One kernel containing all fixed tile epochs |

The actual report, not the metric list or `--launch-count 1`, must establish the
pass count. All ten retained reports show `profiler__replayer_passes = 1` and
zero profiler warmup passes through both raw export and native-report API
inspection. Backup-byte metrics are absent, so the validation record stores
`backup_bytes: null`. No-save/restore is inferred from the
[documented single-pass kernel-replay contract](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html),
not a measured zero-byte value. Every profiled invocation completed and passed
the native and independent Python full-output checks with maximum error zero.

Inspect the real report on a CPU with the selected tool's matching
`extras/python` directory:

```bash
python code/labs/nvshmem_device/inspect_ncu_replay_metrics.py \
  "$NVDEV_NCU_DIR/persistent-halo-p0.ncu-rep" \
  --ncu-python /path/to/NsightCompute-2025.3.1/extras/python \
  --output "$NVDEV_NCU_DIR/replay-metadata.json"
```

The [inspector](inspect_ncu_replay_metrics.py) preserves absent metrics as absent.
It does not convert an unreported backup-byte metric to zero.

Check a persistent-halo dump independently from the repository root, using the
output directory from either example:

```bash
python - "$NVDEV_NCU_DIR/output.f32" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "code")
from labs.nvshmem_device.run_lab_nvshmem_device import NativeWorkload, validate_full_output
workload = NativeWorkload(mode="persistent-halo", nx=33, ny=11, iterations=5, warmup=1)
print(validate_full_output(Path(sys.argv[1]), workload))
PY
```

Use `NVDEV_NSYS_DIR` instead for the Systems example and match the workload
parameters when changing modes. One-pass instrumentation can still perturb
scheduling and peer waits. These reports provide selected-kernel duration,
not an unprofiled solve latency or a full hardware-counter analysis. A separate
capture of pure `prepare_payload` is safe for broader kernel-replay analysis,
but its counters cover preparation alone. The earlier outer-launcher
application-range attempts stalled and remain incomplete diagnostics.

Separate setup, warmup, steady execution, verification, and teardown. Compare
block and persistent tile modes only with identical payloads, epochs, output
history, and verification. The persistent mode saves repeated launches but adds
GPU-resident synchronization and holds a CTA while waiting. Do not infer a
speedup from a shorter host submission trace. Continue with the
[targeted exercises](TASKS.md).
