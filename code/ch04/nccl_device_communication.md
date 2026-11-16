# NCCL device communication: LSA, CFT, GIN, and buffer lifetimes

Use device-initiated communication when a kernel produces data that another GPU needs and moving the transfer into that kernel removes useful scheduling work. The payoff can be fewer host launches and earlier publication of a completed tile or halo. The cost is a communication protocol inside your application: topology, memory visibility, completion, and buffer reuse become your responsibility. Keep host NCCL as the baseline when its existing schedule is sufficient.

This guide explains the device API and its connection to CUDA Graphs, building on [Hands-On 9 in the JSC multi-GPU tutorial](https://github.com/FZJ-JSC/tutorial-multi-gpu/tree/cd34f9d40d600efd8d84e433b1a9e84ab5eff5f3/09-H_Device-initiated_Communication_and_CUDA_Graphs). Its C++ fragments illustrate individual operations. They omit complete initialization, kernels, launch coordination, and error cleanup, and do not constitute a runnable GIN solver. The native [multi-GPU communication lab](../labs/multi_gpu_communication/README.md) implements host MPI/NCCL scheduling and NVSHMEM on-stream transfers separately.

## Choose the API for the work

| Mechanism | Who initiates communication? | What the application controls |
|---|---|---|
| `ncclGroupStart` / `ncclGroupEnd` around `ncclSend` / `ncclRecv` | Host code submits NCCL operations | Group compatible operations, then express compute and communication dependencies through streams and events |
| CUDA Graph capture of host NCCL operations | Host code launches a previously captured schedule | Keep addresses and the schedule valid across replay and coordinate participating ranks |
| NCCL device API | User CUDA code calls communication primitives | Choose an accessible peer path and implement publication, consumption, and reuse ordering |

Grouping host calls does not turn them into user-kernel communication. It also does not prove compute/communication overlap. The actual stream dependencies and hardware execution determine that. NCCL permits capture of collective, point-to-point, and grouped operations starting with NCCL 2.9 and CUDA 11.3. Capture participation and graph launch must match across the relevant ranks. [NCCL groups](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html), [NCCL graph requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)

Host initiation and host waiting are separate choices. The CPU-Initiated Communication slide's blocking compute/synchronize/put/wait sequence adds a host handoff to the dependent path. The [NVSHMEM guide](nvshmem_communication.md#a-host-put-is-different-from-an-on-stream-put) explains its wait scope and strong-scaling tradeoff. Host-enqueued stream operations can overlap independent GPU work. The [native NCCL exchange](../labs/multi_gpu_communication/jacobi_native.cu#L324-L377) expresses those dependencies, and the lab's [two-B200 profiles](../labs/multi_gpu_communication/README.md#observed-two-b200-diagnostic) observed that overlap. Moving initiation into a kernel is useful when finer publication or less host orchestration justifies its protocol and GPU resource cost.

## Read the device API stack by capability

The tutorial's [device API stack, slide 32 at the reviewed commit](https://github.com/FZJ-JSC/tutorial-multi-gpu/blob/cd34f9d40d600efd8d84e433b1a9e84ab5eff5f3/08-L_CUDA_Graphs_and_Device-initiated_Communication_with_NVSHMEM/slides.pdf#page=32)
places custom operations, optimized collectives, and compute/communication
fusion above the same device API. It names Reshard (RL), DeepEPv2, NCCL EP,
UB-X, TRT-LLM, CuTe DSL, and FlashInfer as ecosystem examples. This is an
architecture map, not a release-by-release support matrix or a performance
comparison. Check the particular application's implementation and build before
assuming it uses a given device API path.

| Module | Capability | What to establish before using it |
|---|---|---|
| LSA, or load/store accessible memory | Access a peer window with GPU loads and stores over CUDA P2P | Query the LSA team. NVLink and some PCIe P2P peers qualify. NVLink SHARP multicast is a separate hardware capability exposed through Multimem. |
| CFT, currently named Compute Fabric Transport | Use CUDA fabric logical endpoints for fabric transfers, reductions, and barriers | Request CFT resources and use the CFT team and endpoint queries. Hardware, driver, and toolkit must support the fabric instructions. |
| GIN, or GPU-Initiated Networking | Issue network operations from device code | Establish GIN connectivity, backend support, context resources, and signal requirements. A backend can still use CPU progress. |

The slide labels CFT as CUDA Fabric Transport and places it over large-scale
NVLink. Current NVIDIA documentation calls it **Compute Fabric Transport** and
dates its NCCL helpers to 2.31. Its current runnable barrier example requires
CUDA Toolkit 13.3 and SM_100. An NVLink connection or a Blackwell GPU alone is
insufficient evidence that CFT is usable. The CFT setup must succeed for every
participating rank. [NCCL module overview](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html),
[CFT requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cft.html),
[CFT resource and team contracts](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_cft.html)

This guide's code below uses LSA and GIN. It does not implement or validate CFT,
Multimem, or the ecosystem integrations shown on the slide. A two-GPU NVLink
run can establish behavior for its observed local path. It cannot establish
network progress, CFT support, or framework-level integration.

## LSA and GIN reach different peers

An LSA team contains peers whose memory is accessible through GPU loads and stores. NVLink is a common case, and some PCIe P2P connections also qualify. GIN sends data over a supported networking backend. A process being on the same node does not establish LSA membership. Query the team rather than deriving the transport from rank arithmetic. GIN can use a CPU proxy backend, so GPU initiation does not imply that all network progress occurs without a CPU. [NCCL device communication and requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)

The network data plane moves the payload. The control plane submits requests
and tracks completion. GPUDirect RDMA can move payloads directly between GPU
memory and a NIC while a CPU proxy still handles requests. GIN's GDAKI backend
moves network initiation into GPU code on supported systems. Keep backend
selection separate from the call's location in a CUDA kernel. The
[NVSHMEM transport discussion](nvshmem_communication.md#cpu-proxy-and-ibgda-are-transport-choices)
explains the same distinction for CPU proxy, IBGDA, and CPU-assisted IBGDA.

Within device code, translate a world rank before calling an accessor that expects an LSA rank:

```cpp
const ncclTeam world = ncclTeamWorld(dev_comm);
const ncclTeam lsa = ncclTeamLsa(dev_comm);
const bool peer_is_lsa = ncclTeamRankIsMember(lsa, world, peer_world_rank);

if (peer_is_lsa) {
    const int peer_lsa_rank = ncclTeamRankToTeam(lsa, world, peer_world_rank);
    float* peer_row = static_cast<float*>(
        ncclGetLsaPointer(output_window, row_offset_bytes, peer_lsa_rank));
    // This fragment assumes a one-dimensional block and one value per thread.
    if (threadIdx.x < count) peer_row[threadIdx.x] = produced_values[threadIdx.x];
}
```

The offset is in bytes. The pointer identifies the peer's corresponding registered allocation. These stores still need a publication protocol before the peer consumes the row. For a cooperative LSA barrier, reserve distinct barrier indices for independently participating thread groups, and have every member of that barrier's team participate. A block barrier alone does not synchronize another GPU or another CUDA block. [Memory accessors and LSA barriers](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_memory.html)

## Register a window, then request the device resources

Three registration mechanisms serve different interfaces:

| Registration | Handle and lifetime | Purpose |
|---|---|---|
| Local user-buffer registration | `ncclCommRegister` returns a handle released by `ncclCommDeregister` | Let host NCCL operations use eligible application buffers directly |
| Capture-based registration | Eligible captured NCCL operations manage registration through graph lifetime | Support registration for documented algorithms and allocators during graph capture |
| Symmetric window registration | `ncclCommWindowRegister` returns `ncclWindow_t`, released by `ncclCommWindowDeregister` | Expose corresponding memory windows to device communication and other window-aware NCCL paths |

Allocation alone does not register a buffer. Window registration requires `ncclMemAlloc` or a qualified VMM allocator. For host NCCL, register both source and destination and use registered buffers consistently across participating ranks. Registration may reduce copies or communication resource use, but the effective path depends on allocator, topology, algorithm, and transport. A successful registration call is not a measured speedup. [NCCL user-buffer registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html)

For an uneven stencil decomposition, reserve the same maximum local allocation size on every rank. Owned row counts can differ. Halo offsets must still refer to the receiving rank's layout. In a ping-pong solver, keep each pointer paired with its registered window when swapping roles. [Window registration API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommwindowregister)

This host-side fragment illustrates a full-GIN setup. `CHECK` must stop the distributed run on failure. `comm` is an initialized host communicator, and `max_owned_rows` is the same value on every rank.

```cpp
#if NCCL_VERSION_CODE < 23100
#error "This example's capability queries require NCCL 2.31 or newer"
#endif

ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
CHECK(ncclCommQueryProperties(comm, &properties));
if (!properties.deviceApiSupport ||
    properties.ginConnectionType != NCCL_GIN_CONNECTION_FULL) {
    fail_all_ranks("Full GIN connectivity is required by this variant");
}

const size_t bytes = size_t(nx) * size_t(max_owned_rows + 2) * sizeof(float);
float* output = nullptr;
ncclWindow_t output_window;
CHECK(ncclMemAlloc(reinterpret_cast<void**>(&output), bytes));
CHECK(ncclCommWindowRegister(
    comm, output, bytes, &output_window, NCCL_WIN_COLL_SYMMETRIC));

ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
requirements.lsaBarrierCount = 1;
requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
requirements.ginSignalCount = 4;
requirements.ginStrongSignalsRequired = true;
ncclDevComm dev_comm;
CHECK(ncclDevCommCreate(comm, &requirements, &dev_comm));
```

The four signal slots are an example resource reservation for two ready notifications and two consumption acknowledgments. Their mapping must be defined consistently at both ends of each directed transfer. An LSA-only variant can omit GIN resources. Rail connectivity is insufficient for arbitrary world-rank destinations. `ncclDevCommCreate` is collective, and requirements must cover every barrier and signal that the kernels use. [Device setup API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html)

The version floor above belongs to these exact APIs. Device communication began in 2.28, but `ncclCommQueryProperties` arrived in 2.29, the requirements `ginConnectionType` field in 2.29.7, and the communicator-properties `ginConnectionType` field in 2.31. GIN device compilation also requires CUDA 12.2 or newer, with backend-specific driver and networking requirements. Compile host and device code against matching NCCL headers, and rebuild GIN kernels when upgrading NCCL. [API version annotations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html), [GIN compatibility requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#cross-version-compatibility)

Keep allocations, windows, and communicators alive until all work using them has completed. Destroy graphs and executable graphs before releasing buffers they reference. Deregister a window before freeing its allocation, and destroy the device communicator before its parent host communicator. `ncclDevCommDestroy` requires that no kernel using that device communicator can still be running. [Resource lifetime requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#nccldevcommdestroy)

## Advance signal epochs and wait for consumption

A notification has two jobs that should be explicit: publish completed data and establish when the next producer may reuse storage. A ready notification establishes visibility for a consumer. It does not say the consumer has finished reading.

For one directed transfer per epoch, start `epoch` at one and increment it once per transfer. Give each directed edge its own ready and consumed signal slots. Two halo messages sent to the same rank still need unambiguous channels. Keep a channel on the same GIN context, or track epochs separately for each context. Keep signal storage across kernel launches, and initialize it only when starting a new, fully synchronized protocol.

The following sender fragment uses one issuing device thread after its producer kernel has completed on the same stream. Its receiver consumes the entire previous epoch before acknowledging it:

```cpp
ncclGin gin(dev_comm, context_index);
const ncclTeam world = ncclTeamWorld(dev_comm);

if (epoch > 1) {
    gin.waitSignal(ncclCoopThread{}, local_consumed_slot, epoch - 1);
}
gin.put(world, peer_world_rank,
        destination_window, destination_offset_bytes,
        source_window, source_offset_bytes, payload_bytes,
        ncclGin_StrongSignalInc{ready_slot_at_peer});
gin.flush(ncclCoopThread{});
```

The matching receiver first waits for readiness. It sends its acknowledgment only after the consumer has finished every read:

```cpp
gin.waitSignal(ncclCoopThread{}, local_ready_slot, epoch);
// Execute the consumer and complete its reads before the following signal.
gin.signal(world, sender_world_rank,
           ncclGin_StrongSignalInc{consumed_slot_at_sender});
```

These fragments express phases, not a complete cross-kernel implementation. If consumption occurs in another kernel, order the acknowledgment after that kernel on the stream. If many threads issue puts, their source-completion operation must cover those issuing threads and contexts. A control thread cannot substitute its own `flush` for arbitrary other producers. The one-thread `flush` above makes its own source safe to reuse. It does not prove remote consumption.

`waitSignal` compares a cumulative signal value with a threshold and does not consume the count. Waiting for the fixed value `1` every iteration can therefore accept an old notification. A shared count can also lose the distinction between neighbors. Use increasing per-edge epochs and explicit acknowledgments, or a documented collective barrier with suitable visibility and reuse guarantees. Resetting a live signal can race with incoming increments. Strong signals order preceding puts to the same peer on the same context. Weak signals cover only their attached payload. [GIN signal, flush, and wait semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html)

NVSHMEM has the same underlying obligations. A symmetric allocation provides corresponding storage, while GPU puts and stream-ordered barriers or quiet operations establish completion. A CPU barrier does not automatically complete GPU-issued communication. Scalar `nvshmem_float_p` is simple, while block puts aggregate contiguous data with fewer application calls. The transport determines the actual transactions. Every thread in the participating block must follow the block operation's contract. [NVSHMEM ordering](https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html), [NVSHMEM block-put examples](https://docs.nvidia.com/nvshmem/api/latest/examples.html)

The [NVSHMEM guide](nvshmem_communication.md#publish-a-halo-from-producing-cuda-threads)
explains scalar halo publication, cooperative nonblocking puts, quiet before
readiness, and consumption acknowledgements. The runnable
[NVSHMEM device lab](../labs/nvshmem_device/README.md) implements those ideas in
five modes: scalar and persistent stencil halos, block signaling, persistent
signaling, and persistent ordinary atomic counters. Its waiting kernels use
one CTA per PE, checked against the local cooperative-launch capacity. Each
required PE still launches and participates in the protocol. This is not a
launch barrier across GPUs or a multi-CTA persistent solver. Nine two-B200
integration cases, per-rank Nsys captures for all five modes, and ten
selected-rank single-kernel NCU duration reports passed. Each actual NCU report
shows one pass and zero profiler warmup. The
[validation record](../labs/nvshmem_device/b200_validation.json) distinguishes
this duration scope from full hardware counters and records unreported backup
bytes. These NVSHMEM results do not validate the NCCL GIN fragments above.

## Capture the dependency graph and select the final buffer

A halo-exchange iteration can compute boundary rows, publish them, compute interior rows beside communication, and join both paths before the next iteration. Events express that graph during stream capture. Stream priority can influence scheduling but provides no correctness guarantee. `cudaGraphInstantiateFlagUseNodePriority` preserves captured node priorities where supported. [CUDA graph instantiation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)

Capture records the pointer values and control path used during capture. A host `std::swap` during later execution does not change those recorded addresses. A captured epoch argument is fixed too. A replayed device communication protocol needs a device-maintained epoch or a valid kernel-parameter update before launch. For a solver that alternates input/output buffers and optionally checks a norm, one approach captures four variants:

| Input/output orientation | Norm check | Captured work |
|---|---|---|
| A → B | No | Compute and halo exchange |
| B → A | No | Compute and halo exchange |
| A → B | Yes | Compute, halo exchange, norm accumulation, and readback |
| B → A | Yes | Compute, halo exchange, norm accumulation, and readback |

Select both graph orientation and final readback from the number of completed iterations. With initial input in A, an odd number of iterations finishes in B. Another approach captures two iterations returning to A and uses a separate odd tail. Keep all ranks on compatible graph variants and check schedules. A host norm reduction still costs synchronization and communication on every check. Fixed-iteration experiments avoid convergence-policy differences when comparing schedules.

Separate capture, instantiation, upload, and warmup from steady-state replay timing, and report their cost when deciding how much reuse is needed. More graph variants cost setup time and memory. Replay helps when repeated launch overhead matters enough to amortize that setup. An eager schedule can be sufficient for long kernels or rapidly changing work.

## Connect this to the existing examples

The [Chapter 4 symmetric-memory ring](symmetric_memory_example.py) already captures peer publications and both lifetime fences using the CUDA symmetric-memory backend. It reads the final parity slot after replay. This guide adds the native NCCL device interface and its explicit transport and protocol choices. The [parameterized graph lab](../labs/parameterized_cuda_graphs/README.md) covers executable memcpy-node updates for changing host bindings. That does not establish that captured NCCL peer operations can be mutated arbitrarily.

The [NVSHMEM scalar/block microbenchmark](nvshmem_ibgda_microbench.cu) and [advanced NVSHMEM examples](nvshmem_advanced_multigpu.cu) provide related communication code. Use the [device lab](../labs/nvshmem_device/README.md#choose-a-mode) for the new complete-output device protocols. The [host communication lab](../labs/multi_gpu_communication/README.md) keeps the stencil workload and complete-output reference fixed while comparing serialized exchange, overlap, and NCCL registration and graph replay. Its NVSHMEM backend uses host on-stream transfers. The NCCL device API snippets in this guide remain educational material. They are not a compiled or validated GIN implementation.

For a future GIN implementation, validation must include odd/even iteration counts, uneven and one-row partitions, all owned rows, finite values, and a deliberately delayed neighbor. Exercise LSA-only, mixed LSA/GIN, and forced-GIN paths separately where supported. A clean LSA run cannot validate the network protocol, and static review cannot establish a performance win.
