# NVSHMEM: addressing, progress, and safe reuse

Use NVSHMEM when a GPU needs to decide when and where to move data, especially
when communication follows fine-grained computation. It offers symmetric memory,
one-sided operations, and device-callable synchronization. The benefit is less
host orchestration. The cost is explicit ownership, ordering, progress, and
buffer-lifetime management. A host-driven NCCL exchange is often enough when
communication is already a small, regular phase. Compare those choices in the
[common communication lab](../labs/multi_gpu_communication/README.md) and
[NCCL group guide](nccl_groups.md).

For runnable device initiation, use the
[NVSHMEM device lab](../labs/nvshmem_device/README.md). Its five modes cover
scalar halo puts, a fused persistent stencil, block signaling, persistent
signaling, and persistent ordinary atomic counters. The synchronized modes use
one CTA per PE and check the supported cooperative launch size. This is a
complete teaching protocol with full-output verification, not a scalable
multi-CTA persistent scheduler.

## PGAS does not mean interchangeable pointers

Each processing element, or PE, owns a local part of a partitioned global address
space. An operation identifies remote storage with a local symmetric address
and a destination PE. The runtime translates that pair. A pointer obtained on
another PE is not a portable NVSHMEM address, and symmetric allocation does not
make arbitrary remote pointers directly load/store accessible. See the
[memory model](https://docs.nvidia.com/nvshmem/api/latest/gen/mem-model.html).

`nvshmem_malloc` is collective. Every PE must make matching allocations with the
same size and order. Each receives its own local allocation. Freeing is also
collective. Finish communication and consumers before freeing storage.
NVSHMEM's [allocation API](https://docs.nvidia.com/nvshmem/api/latest/gen/api/memory.html)
defines the matching requirements.

For directly accessible peers, `nvshmem_ptr(local_symmetric_address, peer)` can
return a local pointer that addresses the peer's storage. It can also return
null. That result is a direct-access pointer, not a symmetric address to pass
back into an RMA operation. Check the
[pointer-query contract](https://docs.nvidia.com/nvshmem/api/latest/gen/api/setup.html#nvshmem-ptr)
instead of assuming NVLink or network peers share one usable pointer value.

### Equal allocations can hold unequal amounts of work

A row decomposition need not divide evenly. Suppose `global_rows >= npes`, all
PEs agree on the grid and pitch, and each allocation reserves two halo rows.
The following host fragment illustrates storage and offsets after NVSHMEM
initialization and device selection. It is not a complete solver.

```cpp
const int base = global_rows / npes;
const int remainder = global_rows % npes;
auto rows_at = [=](int pe) { return base + (pe < remainder); };
const int owned_rows = rows_at(mype);
const int max_rows = base + (remainder != 0);
const size_t allocation_rows = static_cast<size_t>(max_rows) + 2;
float* grid = static_cast<float*>(
    nvshmem_malloc(allocation_rows * pitch * sizeof(float)));
```

Check allocation success collectively before proceeding. Local owned rows are
`1..owned_rows`. Their halos are row `0` and row `owned_rows + 1`. All PEs reserve
the same capacity, but the lower halo's logical position depends on its owner.
For periodic neighbors, the corresponding destination addresses are:

```cpp
const int previous = (mype + npes - 1) % npes;
const int next = (mype + 1) % npes;
float* previous_lower_halo = grid + (rows_at(previous) + 1) * pitch;
float* next_upper_halo = grid;
```

Use `previous_lower_halo` with destination PE `previous`, and `next_upper_halo`
with destination PE `next`. These are addresses within the caller's symmetric
allocation that describe offsets in the peer's corresponding allocation.
Using the sender's row count for the previous PE's lower halo can write the
wrong row when ownership differs. Fixed physical boundaries need their own
neighbor and boundary rules. Equal allocation size does not imply equal work.

## Choose where communication starts

### Initialize inside an MPI job

MPI and NVSHMEM can coexist. Initialize MPI first, choose the GPU assigned to
each local process, then initialize NVSHMEM with the communicator. This host
fragment assumes `selected_device` came from the launcher's explicit mapping.
`MPI_CHECK` and `CUDA_CHECK` stop the job on failure.

```cpp
MPI_CHECK(MPI_Init(&argc, &argv));
MPI_Comm comm = MPI_COMM_WORLD;
CUDA_CHECK(cudaSetDevice(selected_device));
nvshmemx_init_attr_t attributes = NVSHMEMX_INIT_ATTR_INITIALIZER;
attributes.mpi_comm = &comm;
const int status = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attributes);
if (status != 0) MPI_Abort(comm, status);

int rank = 0;
int size = 0;
MPI_CHECK(MPI_Comm_rank(comm, &rank));
MPI_CHECK(MPI_Comm_size(comm, &size));
if (rank != nvshmem_my_pe() || size != nvshmem_n_pes()) MPI_Abort(comm, 1);
```

The attribute supplies the address of an MPI communicator handle during
initialization. NVIDIA requires `nvshmem_finalize()` before the MPI communicator
is destroyed. This concerns the communicator object, not the lexical scope of
a local handle copy. With `MPI_COMM_WORLD`, finalize NVSHMEM before
`MPI_Finalize`. Keep the attribute and its pointer valid through the
initialization call. Complete work and free symmetric storage before finalizing
NVSHMEM. MPI bootstrap does not make `MPI_Barrier` a completion operation for
GPU-issued puts. [MPI bootstrap lifetime contract](https://docs.nvidia.com/nvshmem/api/latest/gen/api/setup.html#nvshmemx-init-attr)

An OpenSHMEM job uses `NVSHMEMX_INIT_WITH_SHMEM` after initializing OpenSHMEM and
must finalize NVSHMEM before `shmem_finalize`. This is a separate bootstrap
choice. It does not merge the CPU and GPU symmetric heaps. See the
[initialization and lifecycle API](https://docs.nvidia.com/nvshmem/api/latest/gen/api/setup.html).

### Match the initiating scope to the work

| Form | What executes it | Responsibility |
| --- | --- | --- |
| Host `nvshmem_float_put` or `nvshmem_float_get` | A CPU thread invokes an RMA operation | Make the source ready and use host-scope completion where required. |
| `nvshmemx_*_on_stream` | The host enqueues work ordered on a CUDA stream | Order producers, transfers, waits, and consumers on that stream or with events. Return to the host is not device completion. |
| Device scalar API | A GPU thread issues communication | Avoid accidentally issuing the same transfer from every thread. |
| Device `_warp` or `_block` API | The entire named scope participates | Every thread in that scope must call with matching arguments. Do not place a block call inside a thread-zero branch. |

The [API overview](https://docs.nvidia.com/nvshmem/api/latest/api/overview.html)
defines these scopes. More participating threads can help drive a peer copy,
but they also consume GPU resources. A put issued by a CUDA thread is
device-initiated even when the network transport later uses a CPU proxy.

A put moves data to a peer without requiring a matching receive. A get fetches
data from a peer. Neither tells the application that the peer has finished
using a destination. A direct blocking put permits source reuse when it returns
in the calling execution context, but remote delivery can still be pending.
An on-stream call returning to the CPU does not mean its queued transfer has
finished. Nonblocking operations need an appropriate completion operation. See the
[RMA contracts](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html).

### A host put is different from an on-stream put

A blocking host-initiated schedule follows this sequence:
`GPU compute → host cudaStreamSynchronize → host shmem_put → host wait`.
Returning control to the CPU and submitting the transfer adds latency between
the producer and its dependent work. In a fixed-size problem spread across more
GPUs, local computation shrinks while that control cost remains. In the lab's
row decomposition, each halo row also keeps the same width. Waiting for a whole
boundary or bulk transfer can delay work that could use an earlier completed
tile. Larger messages amortize per-message work, while finer transfers add
more operations and notification traffic. Neither is a universal scaling win.

Both typed calls below transfer `count` **float elements**, not `count` bytes.
`inbox` describes the corresponding symmetric destination on `peer`. The local
`source` must satisfy the API's local-operand requirements. It can be symmetric
storage or eligible private storage registered with NVSHMEM. Do not assume every
source must be symmetric, or that any unregistered private GPU pointer is valid.
The [local-buffer registration rules](https://docs.nvidia.com/nvshmem/api/latest/gen/api/memory.html#memory-registration)
describe the latter case.

For a direct host call, explicitly finish any CUDA producer first. The NVSHMEM
call is not ordered behind a kernel merely because that kernel was launched
earlier by the same CPU thread:

```cpp
produce<<<producer_grid, producer_block, 0, stream>>>(source, count);
CUDA_CHECK(cudaStreamSynchronize(stream));
nvshmem_float_put(inbox, source, count, peer);
nvshmem_quiet();
```

Here, the host quiet completes this host-issued transfer. The receiver still
needs the agreed notification or phase synchronization before consuming data.
`cudaStreamSynchronize(stream)` waits for that stream. Independent work on
another stream can continue if dependencies and GPU resources permit it. The
[native MPI exchange](../labs/multi_gpu_communication/jacobi_native.cu#L328-L382)
uses this distinction: boundary production finishes before the host calls
`MPI_Sendrecv`, while interior work has already been submitted separately.
See CUDA's [asynchronous execution model](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

For a stream-based call, keep the producer and transfer on the same stream:

```cpp
produce<<<producer_grid, producer_block, 0, stream>>>(source, count);
nvshmemx_float_put_on_stream(inbox, source, count, peer, stream);
nvshmemx_quiet_on_stream(stream);
```

The `nvshmemx` prefix identifies an NVSHMEM extension. The `_on_stream` suffix
adds CUDA stream ordering. The host returns before this sequence completes.
Use an event dependency for consumers on another stream, or synchronize the
stream before host access or teardown. Check CUDA launches and errors in real
code. The [API overview](https://docs.nvidia.com/nvshmem/api/latest/api/overview.html)
and [typed put contract](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html#nvshmem-put)
define the distinction. The `putmem` variants instead express their count in bytes.

The same native source implements host-enqueued NCCL exchange and NVSHMEM
on-stream puts, with events joining communication and interior work before the
next iteration. The lab's [two-B200 profiles](../labs/multi_gpu_communication/README.md#observed-two-b200-diagnostic)
show actual overlap for host NCCL in the measured case. Device initiation can
publish smaller pieces as a kernel produces them, at the cost of GPU execution
resources, transport requirements, and a more complex protocol. NVSHMEM still
needs symmetric destinations, visibility ordering, and a safe consumption/reuse
protocol. Network progress may still use a [CPU proxy](#cpu-proxy-and-ibgda-are-transport-choices).

### Publish a halo from producing CUDA threads

Thread-level publication moves the transfer into the producer kernel. This
educational fragment computes each updated stencil value before sending it. `nvshmem_float_p(destination, value, peer)` sends one float **by
value**. The pointer/count `nvshmem_float_put` form sends an array. Both forms
use a symmetric remote destination. See the [scalar RMA contract](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html#nvshmem-p).

Every PE must allocate both grids collectively, in the same order and with the
same maximum capacity from the earlier decomposition. Initialize input halos
and fixed boundary columns correctly in both buffers before the first step.
Assume `nx >= 3` and at least one owned row per PE. Only columns `1..nx-2` are
updated. `previous_rows` is the **receiving**
previous PE's owned-row count, not the sender's count.

```cpp
__global__ void jacobi_publish_scalar(
    float* next_grid, const float* current_grid, int nx, int owned_rows,
    int previous_rows, int previous_pe, int next_pe) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    if (x >= size_t(nx - 1) || y > size_t(owned_rows)) return;
    const size_t i = y * size_t(nx) + x;
    const float value = 0.25f * (current_grid[i - 1] + current_grid[i + 1] +
                                current_grid[i - nx] + current_grid[i + nx]);
    next_grid[i] = value;
    if (y == 1) {
        const size_t remote_row = size_t(previous_rows) + 1;
        nvshmem_float_p(next_grid + remote_row * size_t(nx) + x, value, previous_pe);
    }
    if (y == size_t(owned_rows)) {
        nvshmem_float_p(next_grid + x, value, next_pe);
    }
}
```

For a one-row partition, both branches intentionally publish the same computed
value to the two halo destinations. Each step then queues this host-side phase
on every PE, using a grid covering its owned cells:

```cpp
jacobi_publish_scalar<<<launch_grid, launch_block, 0, stream>>>(
    next_grid, current_grid, nx, owned_rows, rows_at(previous), previous, next);
CUDA_CHECK(cudaGetLastError());
nvshmemx_barrier_all_on_stream(stream);
std::swap(current_grid, next_grid);
```

The extension's actual name includes the `x`: `nvshmemx_barrier_all_on_stream`.
It follows the entire kernel on the same stream and completes prior GPU-issued
updates when all PEs reach the phase. The next step on that stream therefore
reads ready halos. Kernel completion alone does not establish remote delivery.
The host can return before the queued barrier finishes. This whole-step
protocol also protects the double-buffer roles. Keep the barrier every step,
or replace it with a complete readiness and consumption protocol, before
allowing reuse. NVIDIA's [kernel example](https://docs.nvidia.com/nvshmem/api/latest/using.html#example-nvshmem-program)
uses this kernel-then-stream-barrier pattern. It still couples all PEs to the
slowest step.

A thread can publish when its value is ready, but scheduling and transport
determine when movement actually starts. Per-cell RMA can cost more than
coalesced warp, block, or bulk operations. NVLink access depends on actual peer
reachability and the selected transport, not a DGX label. Device calls also need
the NVSHMEM device library and relocatable device code, such as `-rdc=true`,
in addition to the host library. Follow the [build contract](https://docs.nvidia.com/nvshmem/api/latest/using.html#compiling-nvshmem-programs)
and [device-lab Makefile](../labs/nvshmem_device/Makefile). Start with the runnable
device lab's [`scalar-halo` mode](../labs/nvshmem_device/README.md#choose-a-mode). It implements
this kernel-then-stream-barrier pattern and checks the complete final stencil.
The [common Jacobi lab](../labs/multi_gpu_communication/README.md) retains its
separate host-on-stream NVSHMEM backend. The inline fragment above explains the
operation and does not replace the lab's initialization, checks, and teardown.

## Separate ordering, completion, and notification

| Mechanism | What it establishes | What remains to do |
| --- | --- | --- |
| `nvshmem_fence` | Delivery order for covered earlier and later operations to the same PE | It does not complete delivery or notify the receiver. |
| `nvshmem_quiet` | Completion of covered operations issued in the applicable CPU or GPU scope | It does not wait for a remote consumer. |
| `nvshmem_barrier` / `nvshmem_barrier_all` | Team/world participation plus covered update completion | All required PEs must arrive. Unrelated ranks can become a bottleneck. |
| Put-with-signal | Its payload is delivered before its associated signal is observed | Other transfers are not implicitly covered, and delivery is not consumption. |
| Signal wait/test | The local synchronization object satisfies its condition | The application still owns consumer ordering and slot reuse. |

The [ordering API](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html)
defines fence and quiet. CPU ordering calls cover CPU-issued communication.
They do not drain GPU-issued operations. Use GPU-side completion or an ordered
stream completion operation, then synchronize that stream when the host needs
the result. Synchronize issuing GPU threads before an elected thread orders
their operations.

For separately issued nonblocking payloads, the protocol is:
`payload → GPU quiet → ready epoch → receiver wait → consume → consumed acknowledgement → reuse`.
Quiet completes prior operations in its CPU or GPU scope. It does not notify a
receiver or prove that the receiver has consumed the payload. Fence orders
covered operations but does not complete them. An appropriate NVSHMEM barrier
is the alternative that combines update completion with participant rendezvous.

The [barrier and sync contracts](https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html)
also distinguish `sync` from `barrier`. `nvshmem_sync_all` is not a replacement
for completing pending NVSHMEM RMA. Use a barrier for a simple whole-team phase
boundary. Use neighbor signals when independent neighbors can proceed without
waiting for the entire team.

### Host barrier versus on-stream barrier

| Call | Producer readiness | Completion and consumer placement |
| --- | --- | --- |
| Host `nvshmem_barrier_all()` | Finish a CUDA producer before issuing a host put that reads its output | Blocks the calling CPU thread until the world participates and covered host-issued updates complete. It does not complete GPU-issued RMA. |
| `nvshmemx_barrier_all_on_stream(stream)` | Earlier producers and on-stream puts on this stream are ordered before the barrier | Enqueues a GPU-side phase boundary. A later consumer on this stream waits for it. The CPU can continue before the barrier completes. |

For host-issued puts, the simple phase is:

```cpp
produce<<<producer_grid, producer_block, 0, stream>>>(source, count);
CUDA_CHECK(cudaStreamSynchronize(stream));
nvshmem_float_put(inbox, source, count, peer);
nvshmem_barrier_all();
consume<<<consumer_grid, consumer_block, 0, stream>>>(inbox, output, count);
```

Every PE must execute its matching phase, even when its send peer differs from
its receive peer. For stream-issued puts, keep that phase on the stream:

```cpp
produce<<<producer_grid, producer_block, 0, stream>>>(source, count);
nvshmemx_float_put_on_stream(inbox, source, count, peer, stream);
nvshmemx_barrier_all_on_stream(stream);
consume<<<consumer_grid, consumer_block, 0, stream>>>(inbox, output, count);
```

Replacing the last stream barrier with a host barrier would lose the completion
contract for the GPU-issued put. A consumer on another stream needs an event
dependency. The host needs stream synchronization before reading results or
tearing down storage. The [barrier API](https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html#nvshmem-barrier-all)
defines both scopes. World barriers simplify this phase but make every PE pay
for the slowest arrival.

### Publish, consume, then acknowledge

The following fragments illustrate one producer and one consumer sharing one
reusable destination slot. Both PEs have corresponding symmetric allocations
for `inbox`, `ready`, and `consumed`. The producer's `payload` is a valid local
operand. Signals start at zero, generations start at one and increase without
wraparound, and all initialization has completed before communication starts.
Only this producer writes the slot. All CUDA launches and initialization calls
need normal error checking.

The producer queues the payload and its ready signal, then waits for the
consumer's acknowledgement before another iteration can reuse the slot:

```cpp
nvshmemx_float_put_signal_on_stream(
    inbox, payload, count, ready, generation,
    NVSHMEM_SIGNAL_SET, consumer_pe, stream);
nvshmemx_signal_wait_until_on_stream(
    consumed, NVSHMEM_CMP_EQ, generation, stream);
```

The consumer waits, runs its complete consumer kernel, then acknowledges on
the same stream. The acknowledgement kernel contains no wait or collective:

```cpp
__global__ void acknowledge(uint64_t* consumed, uint64_t generation, int producer_pe) {
    nvshmemx_signal_op(consumed, generation, NVSHMEM_SIGNAL_SET, producer_pe);
}

// Consumer host thread, using an initialized CUDA stream.
nvshmemx_signal_wait_until_on_stream(
    ready, NVSHMEM_CMP_EQ, generation, stream);
consume<<<consumer_grid, consumer_block, 0, stream>>>(inbox, output, count);
acknowledge<<<1, 1, 0, stream>>>(consumed, generation, producer_pe);
```

`consume` represents the application's kernel, whose reads must finish before
it returns. Stream order places the acknowledgement after that kernel. For
final teardown, finish GPU-issued operations with a stream-ordered quiet,
synchronize the stream, and let all PEs reach the agreed teardown phase before
collectively freeing allocations. See the
[signaling API](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html)
and [wait API](https://docs.nvidia.com/nvshmem/api/latest/gen/api/sync.html).

With multiple slots or incoming neighbors, assign distinct notification state
to each slot and writer. Generation counters avoid treating an old ready value
as a new message. Double buffering gives more room for overlap, but it does not
replace the acknowledgement that protects a slot from premature overwrite.

### Cooperate within a block, then publish readiness

`nvshmemx_float_put_nbi_block(destination, source, nelems, peer)` transfers
`nelems` **float elements**, using pointer operands. `_nbi` does not make either
buffer immediately reusable. The remote destination must be symmetric and the
local source must satisfy the RMA operand contract. Every CTA thread must call
with identical arguments in uniform control flow. A thread-zero call or a call
made only by edge threads is invalid. Warp variants require the same agreement
within their participating warp. See [nonblocking RMA](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html#nvshmem-put-nbi)
and [cooperative API scopes](https://docs.nvidia.com/nvshmem/api/latest/api/overview.html).

This device fragment assumes the CTA has produced a valid contiguous tile.
Each directed tile/edge has its own symmetric `uint64_t` ready and consumed
words, initialized to zero before first use. One producer owns each ready word.
Epochs start at one, increase without wrapping, and are not reset while live:

```cpp
const bool leader = threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
if (leader && epoch > 1) {
    nvshmem_signal_wait_until(consumed_here, NVSHMEM_CMP_GE, epoch - 1);
}
__syncthreads();
nvshmemx_float_put_nbi_block(destination, source, nelems, peer);
__syncthreads();
if (leader) {
    nvshmem_quiet();
    nvshmemx_signal_op(ready_at_peer, epoch, NVSHMEM_SIGNAL_SET, peer);
}
__syncthreads();
```

After the collective put has returned and the CTA has synchronized, the elected
thread's `nvshmem_quiet` can complete the preceding operations issued by that
CTA. Its scope is the calling PE's prior GPU operations, with the required
CUDA ordering between issuing threads. This CTA barrier establishes that order
for this CTA, not for arbitrary other CTAs. There is no need to invent a
`quiet_block` operation. The current [quiet contract](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html#nvshmem-quiet)
explicitly describes this pattern.

The receiver waits for its local ready word with
`nvshmem_signal_wait_until(ready_here, NVSHMEM_CMP_GE, epoch)`. If an elected
thread waits on behalf of a CTA, synchronize the CTA before other threads read
the payload. Complete every consumer read before acknowledging that epoch back
to the sender. The producer's consumed wait then protects the next overwrite.
These inline fragments require the launch and progress rules below. The
runnable [`block-signal` and `persistent-signal` modes](../labs/nvshmem_device/README.md#trace-a-tile-epoch-before-changing-it)
implement the complete two-edge protocol and retain every received element at
every epoch. Their tile workload is separate from the stencil modes.

`NVSHMEM_SIGNAL_SET` writes the chosen epoch. `NVSHMEM_SIGNAL_ADD` adds a
contribution. A shared count does not identify which neighbor or tile arrived.
If a protocol uses increments, derive the expected count from exact producers
and contributions for that epoch. Prefer separate edge flags when that makes
the obligation clear. [Signal semantics](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html)

### Wait on local flags without losing peer identity

Wait routines inspect local symmetric objects that peers update. They do not
poll an arbitrary remote pointer, send notifications, or complete unrelated
RMA. For ordinary integer flags, these device calls illustrate one entry and
an array of two incoming edge flags:

```cpp
nvshmem_int_wait_until(edge_flags + previous_edge, NVSHMEM_CMP_GE, epoch);
nvshmem_int_wait_until_all(edge_flags, 2, nullptr, NVSHMEM_CMP_GE, epoch);
```

Here `epoch` has type `int` and stays within its supported range. Each entry has
one agreed writer. The `_all` count is in elements and uses one comparison and
epoch for every included entry. A null `status` includes all entries. In a
non-overlapping local mask array, zero includes an entry and nonzero excludes
it. The mask selects flags, not a collective PE list. These routines accept
their documented integer/signal types, not the full RMA half/float/double list.
See the [wait contracts](https://docs.nvidia.com/nvshmem/api/latest/gen/api/sync.html#nvshmem-wait-until-all).
An empty wait set proves no readiness.

An ordinary symmetric AMO counter is separate from a dedicated `uint64_t`
signal object. For example, `nvshmem_int_atomic_inc(counter, peer)` is a
non-fetching integer increment. Use `nvshmem_int_atomic_fetch_inc` when the old
value is needed. Select a supported AMO type rather than copying the RMA type
table. Neither increment replaces quiet before publishing payload completion.
The [AMO contracts](https://docs.nvidia.com/nvshmem/api/latest/gen/api/amo.html#nvshmem-atomic-inc)
define both forms. The existing [native pipeline](nvshmem_pipeline_patterns.cu) uses stream-ordered
integer ready and consumed flags through `nvshmemx_int_p_on_stream`. Those RMA
stores are distinct from AMO calls and dedicated signal-word operations.

## Preserve forward progress

Fusing several Jacobi steps into a persistent kernel also introduces local
inter-block dependencies. `__syncthreads()` synchronizes one CTA, not the grid
or a peer GPU. Before a tile advances, every local and remote predecessor it
reads must have published the right epoch. Before reuse, every consumer must
have finished. A valid tile wavefront or a proven global-step protocol is
required. In a two-dimensional launch, `blockDim.y` counts threads per CTA and
`gridDim.y` counts CTA rows. Neither a guessed grid multiplier nor one shared
counter establishes all of those dependencies.

For kernels containing device-side NVSHMEM waits or PE collectives, use
`nvshmemx_collective_launch`. Query
`nvshmemx_collective_launch_query_gridsize` for that kernel, block shape,
arguments, and shared-memory requirement. Check both return codes and keep the
total block count within the returned capacity. This path requires CUDA
cooperative-launch support. This synchronized design assumes one PE per GPU
and exclusive GPU ownership. See NVIDIA's [launch API](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html)
and [execution constraints](https://docs.nvidia.com/nvshmem/api/latest/using.html#communication-model).

NVSHMEM uses CUDA cooperative launch locally. Calling
`cudaLaunchCooperativeKernel` alone does not replace NVSHMEM's prescribed launch
path or a remote protocol. The current
[launch API contract](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html#nvshmemx-collective-launch)
states that `nvshmemx_collective_launch` launches only on the calling PE's GPU
and does not coordinate launches across PEs. Every required PE must launch and
follow matching synchronization rules. Launch residency helps
waiting CTAs make progress. It does not establish remote completion, readiness,
or safe slot reuse.

The scalar halo fragment above uses only RMA inside its kernel and puts the
barrier on the host-enqueued stream afterward. That pattern can use an ordinary
CUDA kernel launch. The block fragment's in-kernel waits make it a different
launch case. Persistent-kernel launch savings cost more protocol state, GPU
residency, and portability requirements. A kernel per step with a stream
barrier is often the simpler starting point.

The runnable [`persistent-halo` mode](../labs/nvshmem_device/README.md#keep-stencil-halo-consumption-separate)
keeps all local stencil dependencies within one CTA. It acknowledges halo epoch
`s` only after every local stencil read finishes, retains consumed waits before
overwriting the alternating target halo, and publishes ready epoch
`s + 1` after the row puts complete. The final received halo has not been read
by a subsequent step, so its consumed epoch is `iterations - 1`. All nine
two-B200 integration cases passed native and independent Python complete-output
checks across the five modes. The one-CTA design limits compute throughput and
does not establish a large-grid Jacobi speedup or a general multi-block protocol.

In this strict schedule, the preceding ready wait has already established the
consumed condition: a peer sends its acknowledgment before quiet and the ready
notification observed at the preceding step's end. The consumed waits therefore
cannot block. They keep the reuse obligation explicit as a second guard. After
startup, that bookkeeping costs two remote acknowledgment signals and two local
consumed-wait calls per step. Consumed acknowledgments begin at zero-based step
one and consumed waits at step two. Ready notifications occur at every step.
If a future boundary-first schedule publishes readiness earlier, those waits
may become necessary. Relaxing the ready phase needs a new dependency proof,
not an assumption that the current redundancy carries over.

A waiting kernel can occupy resources needed by the kernel that would satisfy
its wait. Separate streams alone do not guarantee forward progress. Put a
producer before its dependent wait using stream order or a CUDA event. Avoid
cycles where a barrier blocks the operation needed to release another PE's
wait. The [CUDA interaction guide](https://docs.nvidia.com/nvshmem/api/latest/cuda-interactions.html)
shows these hazards. Device-driven control saves host round trips only when its
waiting work does not prevent the rest of the application from running.

## CPU proxy and IBGDA are transport choices

Separate the **data plane**, which moves payload bytes, from the **control
plane**, which submits requests and tracks completion. [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)
lets a NIC access GPU memory without staging the payload through CPU memory.
It does not establish who submits work to the NIC. With a CPU-proxy transport
such as IBRC, a device call can follow `GPU request → CPU proxy → NIC queue`
while the payload follows `GPU memory ↔ NIC`. The CPU handles submission and
completion without copying the payload. GPU initiation therefore does not
imply CPU-free network progress. Local peer traffic can use a different path.
NVIDIA's [proxy and IBGDA walkthrough](https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/)
traces these two paths.

IBGDA can submit network work from the GPU. The [current setup guide](https://docs.nvidia.com/nvshmem/api/latest/using.html#using-gpudirect-async-kernel-initiated-communication)
lists the supported Mellanox NIC, OFED, driver, and peer-memory or DMA-BUF
requirements. It enables IBGDA with `NVSHMEM_IB_ENABLE_IBGDA=1`. The
[environment reference](https://docs.nvidia.com/nvshmem/api/latest/gen/env.html)
also exposes `NVSHMEM_IBGDA_NIC_HANDLER`, whose automatic selection can use a CPU
handler when GPU handling is unavailable. Verify the selected transport and
handler in runtime diagnostics. An enable flag records intent, not observed
execution. CPU-assisted IBGDA is also distinct from the traditional IBRC proxy:
the GPU can prepare NIC work while the CPU rings its doorbell. NVIDIA's
[NVSHMEM 3.0 explanation](https://developer.nvidia.com/blog/enhancing-application-portability-and-compatibility-across-new-platforms-using-nvidia-magnum-io-nvshmem-3-0/)
describes this intermediate mode. Record the handler as well as the transport
when comparing results. Current documentation lists `ucx` as a selectable
remote transport. Check the installed build and version rather than carrying
forward the transport-overview slide's historical “experimental” label for UCX.

GPU submission can help many small requests proceed in parallel. It adds GPU
instruction and queue-management work. Queue pairs consume resources, and more
queues can increase fence/quiet cost. CPU proxying adds a control handoff and
needs CPU progress resources and appropriate NUMA placement. Bulk transfers
can amortize that cost. Measure message sizes and concurrency before tuning
queue counts or replacing a working path.
NVIDIA's [performance guide](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/performance.html)
and [CPU-affinity setting](https://docs.nvidia.com/nvshmem/api/latest/gen/env.html)
describe these tradeoffs. Optimizing an internal request ring or its fences
does not remove the application's quiet, notification, or reuse obligations.
The lab's two-B200 NVLink observations do not establish network selection,
GPUDirect RDMA or IBGDA performance, or multi-node correctness.

### Read the GH200 Jacobi comparison with its topology and footnote

The tutorial's [NVSHMEM comparison, slide 31 at the reviewed commit](https://github.com/FZJ-JSC/tutorial-multi-gpu/blob/cd34f9d40d600efd8d84e433b1a9e84ab5eff5f3/08-L_CUDA_Graphs_and_Device-initiated_Communication_with_NVSHMEM/slides.pdf#page=31)
uses NVSHMEM 3.1.7 on JUPITER GH200 120 GB GPUs for a 17408 × 17408 Jacobi grid.
NVLink uses one node with up to four GPUs. CPU proxy and IBGDA use up to four
nodes with one GPU per node. That placement changes both the interconnect and
the node layout.

The CPU-proxy bars extend beyond the plotted 2-second range. Read their labels:
12.303 seconds at two GPUs, 12.502 at three, and 12.421 at four. Treating their
visible height as about 2 seconds would misstate the result. The separate
`IBGDA [1]` series uses experimental NVSHMEM on an NVIDIA internal GH200 480 GB
cluster with NDR400, as the slide's footnote states. It is a different cohort.

This example motivates checking network request overhead for the Jacobi
schedule. It does not establish a universal IBGDA speedup, a matched comparison
with this repository's B200 NVLink runs, or a result for another message size
and concurrency. Before choosing a transport for an application, retain its
actual placement, NIC and link, NVSHMEM build, selected handler, correctness
result, and comparable timings. Keep experimental and production cohorts
separate.

## Read and run the existing examples

| Existing path | What it demonstrates | Evidence limit |
| --- | --- | --- |
| [NVSHMEM device lab](../labs/nvshmem_device/README.md) and [native source](../labs/nvshmem_device/nvshmem_device_native.cu) | Five runnable scalar, block, persistent stencil, signal, and atomic-counter modes with independent complete-output checks | Nine two-B200 integration cases, all five modes' per-rank Nsys captures, and ten one-pass selected-kernel NCU duration reports passed. The waiting kernels use one CTA per PE. This is not full hardware-counter or multi-node coverage. |
| [Common native Jacobi exchange](../labs/multi_gpu_communication/jacobi_native.cu) | Equal-capacity symmetric allocations, unequal owned rows, neighbor-specific halo offsets, and host-enqueued stream puts followed by an on-stream world barrier | This is the stream API, not communication issued by the stencil kernel. The world barrier couples all PEs. GPU validation is separate from this guide. |
| [Native queue and stream pipeline](nvshmem_pipeline_patterns.cu) | Queue payload before tail publication using fence, then a stream pipeline with put, quiet, ready, and consumed flags | The pipeline checks the first value of each chunk. Its no-NVSHMEM educational build does not perform communication. |
| [Legacy device RMA sketch](nvshmem_ibgda_microbench.cu) | Scalar and block-scoped API shapes | Unvalidated rates. Timing excludes GPU-side RMA completion, block bytes count one CTA, and scalar MOPS counts launched rather than active threads. Do not use its rates for a transport comparison. |
| [Legacy scalar API sketch](nvshmem_multigpu_examples.cu) | Individual scalar put calls | Its whole-array put comment is not a block-scoped API example. The host barrier does not establish GPU-issued RMA completion, and it prints samples without rejecting incorrect results. |
| [Legacy advanced sketches](nvshmem_advanced_multigpu.cu) | Ring and recursive-exchange structures | The single-inbox ring and recursive paths do not protect consumption before the next remote overwrite, including the ring phase boundary. They are not validated protocol references. |
| [Python pipeline channel](nvshmem_pipeline_parallel_multigpu.py) | Symmetric-allocation remote copies with ready and consumed protocols | The targeted PyTorch path uses CPU/Gloo sidebands because its signal methods are stubs. It is not GPU-native NVSHMEM signaling. |
| [Symmetric-memory ring](symmetric_memory_example.py) | Batched NCCL P2P and a captured CUDA symmetric-memory comparison | The named CUDA-backend graph path and the NVSHMEM-default helper have different completion paths. |

The legacy IBGDA sketch records its stop event before any GPU-side quiet. Its
later host barrier is outside that interval and does not establish the missing
GPU completion contract. The printed GB/s counts one block's bytes instead of
all blocks, while MOPS counts threads that return without issuing a put. At the
defaults, the byte count is 128 bytes rather than 1024 per iteration, and 2048
launched threads are counted although only 256 issue scalar puts. Its
`--proxy-penalty-ms` sleep occurs after CUDA-event timing and is not measured
proxy overhead. Enabling IBGDA is not transport-selection proof, and checksum
verification is not full payload acceptance. Read the file only as an
unvalidated API sketch.

The legacy advanced file's single receive buffer can be overwritten by a
neighbor's next send before its consumer finishes. This includes the
reduce-scatter to all-gather boundary, even with two PEs, and the recursive
exchange path. A barrier before consumption does not order the next remote
write after that consumption. A repair needs another consumption/reuse phase
or buffer separation that covers phase boundaries, adding synchronization or
storage. The file's separate double-buffered pattern does not validate the
single-inbox paths. Use the new device lab for a complete, checked protocol.

Do not use the legacy scalar file's commented `nvshmem_float_put` as a
block-scoped optimization. Each calling thread would issue the whole array
transfer. The collective block API is a different operation. That sketch also
uses device synchronization followed by a host barrier instead of establishing
GPU-scoped RMA completion, then prints a sample without rejecting mismatches.
Use the lab's `scalar-halo` path for runnable scalar publication.

Build the new device lab against the installed runtime, then run its complete
persistent stencil from the repository root:

```bash
make -C code/labs/nvshmem_device NVSHMEM_HOME=/path/to/nvshmem ARCH=sm_100
python code/labs/nvshmem_device/run_lab_nvshmem_device.py --mode persistent-halo --ranks 2 --nx 17 --ny 7 --iterations 5 --warmup 1 --json-out /tmp/persistent-halo.json --dump-output /tmp/persistent-halo.f32
```

Use fresh output paths. The [lab instructions](../labs/nvshmem_device/README.md#build-and-run)
provide all five commands, library-path overrides, and launcher requirements.

On an allocated two-GPU B200 system with the NVSHMEM package layout expected by
the [chapter Makefile](Makefile), the existing native pipeline target is:

```bash
make -C code/ch04 nvshmem_pipeline_patterns_sm100 ARCH=sm_100
nvshmrun -np 2 code/ch04/nvshmem_pipeline_patterns_sm100
```

`nvshmrun` is NVSHMEM's standalone launcher. Use the compatible launcher
installed for the selected runtime. Confirm the build actually enabled NVSHMEM
rather than accepting educational-mode output as a communication test.

From `code/`, the existing Python pipeline pair has a separate harness path:

```bash
python -m cli.aisp bench run --targets ch04:nvshmem_pipeline_parallel_multigpu --profile none
```

Its [baseline wrapper](baseline_nvshmem_pipeline_parallel_multigpu.py) selects
NCCL, while the [candidate wrapper](optimized_nvshmem_pipeline_parallel_multigpu.py)
selects symmetric-memory movement with the Gloo control protocol described
above. Keep that transport distinction in any report. Use the
[collective diagnosis guide](collective_diagnosis.md) and retained per-rank
profiles when deciding whether arrival, progress, or resource contention is the
actual problem.

This guide was checked against current primary API documentation and local
source. The inline fragments explain API contracts. Runtime evidence applies
to the linked lab implementations and selected tested cases. The
[B200 record](../labs/nvshmem_device/b200_validation.json) retains exact source
and capture provenance. NCU validated selected-kernel duration with one observed
pass and zero profiler warmup. Backup-byte metrics are unreported, so
no-save/restore is a documented inference from the single-pass contract.
Single-host results do not establish full hardware-counter analysis,
multi-node transport behavior, or canonical performance.
