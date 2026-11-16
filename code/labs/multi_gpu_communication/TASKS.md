# Guided communication exercises

Use the complete native source as the reference solution. Change one mechanism
at a time and keep the numerical contract fixed. Before each run, predict which
part of the iteration's critical path should change and why.

## 1. Check decomposition and output first

Use a grid with an uneven number of owned rows across two ranks. Check the CPU
reference at zero iterations, then run the native solver for one, two, and five
iterations. Native timing requires at least one step. Then try one-row and
two-row partitions.

Explain which allocation holds the final result for each iteration count.
Verify every owned row and the periodic halos against the CPU reference.
Confirm that invalid shapes and ranks with no owned rows fail explicitly.
These checks detect mistakes that an even step count or a sampled checksum
can hide.

## 2. Identify the communication group

Find `ncclGroupStart` and `ncclGroupEnd` in the native halo exchange. Map each
send to its matching receive. Repeat that mapping for two ranks, where the top
and bottom neighbor are the same peer.

Explain why messages to that peer must match in order and why a wait inside the
open group is unsafe. Draw the wait cycle an ungrouped all-send-first exchange
could create. Do not run a deliberately hanging variant as a timing baseline.
Distinguish the group of submissions from the communicator's set of ranks.

## 3. Expose independent work

Compare serialized and overlapped schedules with the same backend and inputs.
Find the event that makes the previous iteration's input ready, the boundary
producer, the independent interior kernel, the halo exchange, and the join
before the next iteration.

Explain why the boundary rows are ready for sending before the whole interior
finishes. Check empty-interior partitions. Explain why increasing stream
priority cannot repair a missing event or create hardware capacity.

Use a retained Nsight Systems report to identify actual simultaneous work.
If overlap appears but the full step is slower, inspect resource contention
and the cost of splitting kernels and adding events. Report that outcome.

## 4. Separate registration from scheduling

Keep schedule and graph mode fixed. Compare ordinary and explicitly registered
NCCL buffers on a supported runtime. Find both registrations and their teardown.
Confirm that all ranks use the same mode and both send and receive allocations
remain alive through completion.

Report setup cost separately. Use NCCL diagnostics and profiler evidence to
investigate the selected path. Explain why avoiding an internal copy does not
remove network bytes or guarantee a shorter iteration.

## 5. Capture a stable schedule

Read the [graph lifecycle guide](../../ch12/cuda_graph_lifecycle.md). Identify
which operations become graph nodes and which CPU operations only define the
capture. Locate template creation, executable instantiation, optional upload,
warmup, replay, and destruction in the linked implementations.

Compare eager and graph execution with identical registration and schedule.
Follow two captured steps through A to B and back to A. Then follow an odd
tail. Verify the final result after both odd and even counts.
Explain why a host pointer swap affects later captured arguments but does not
change earlier nodes. The native pair and tail are separate rank-local graphs.
Identify the MPI operations that remain outside capture.

Draw the event fork from compute into communication and the join back to compute
before `cudaStreamEndCapture`. In the guide's norm-check illustration, explain
why async memset precedes both reduction branches and why the readback follows
the join. Explain why a captured event wait is allowed while a host stream or
device synchronization during capture is not. `Global` capture mode does not
mean a global MPI graph.

Find the lab's `UseNodePriority` instantiation flag. Explain its effect and why
`AutoFreeOnLaunch` would not free this solver's ordinary work buffers. Check the
installed API signature and its error diagnostics instead of copying an older
slide's instantiation call unchanged.

Predict two launches of one executable into two streams, then compare that
prediction with two independent executable instances and separate buffers.
Distinguish serialization between launches from internal DAG parallelism.
Explain why the dependent A → B → A solver steps cannot simply be made
independent by creating more executable instances.

Identify which startup costs are excluded from replay timing. Estimate the
reuse needed to repay measured capture cost using measured per-step savings.
If replay does not improve the step, do not invent a break-even point.
Inspect CPU submission gaps and GPU intervals in the retained profile. A
node-level graph trace can show individual kernels. A graph-level trace or a
DOT drawing alone cannot establish their timing or overlap.

For a structural debugging exercise, identify where a checked
`cudaGraphDebugDotPrint` export would go after capture, using one output per rank
and pair/tail variant. The current CLI has no DOT-export option. Explain what a
Graphviz rendering can verify and what still requires a native profiler capture.

Finally, compare this fixed-step solver with a proposed convergence loop that
selects parity and norm/no-norm graph variants. Put completion before host norm
access and `MPI_Allreduce`, and make all ranks agree on continuation. Explain how
frequent checks change the critical path without claiming that convergence is
implemented here. Mark CPU host-function nodes, graph-owned memory nodes,
external semaphores, and ordinary child graphs as unsupported exercises in this
lab rather than describing them as existing runnable variants.

## 6. Compare communication models

Run the same numerical work with CUDA-aware MPI and NCCL. Identify where MPI
requires the host to wait for the CUDA producer and where NCCL inherits stream
ordering. Distinguish GPU-buffer awareness from stream awareness and GPUDirect
transport selection.

Read the native NVSHMEM examples linked from the
[NVSHMEM guide](../../ch04/nvshmem_communication.md). Work out why equal
symmetric allocation size does not require equal owned work and why a remote
halo offset belongs to the receiver's layout.

For the [NCCL device API](../../ch04/nccl_device_communication.md), trace a
ready notification and a consumed acknowledgment through two epochs. Explain
why repeatedly waiting for the fixed signal value one can accept an old
message, and why an LSA-only run cannot validate GIN networking.

Trace the device scalar halo and cooperative block-put fragments in the guide.
The scalar `p` call carries one value, while typed block puts carry an element
count. Explain why every thread in a participating block must enter the same
block operation with identical arguments. Compare the cost of one operation
per boundary cell with a batched boundary transfer. The current native NVSHMEM
backend remains a host-enqueued stream implementation.

Draw the point-to-point protocol from payload through `quiet`, ready signal,
receiver wait, consumption, and consumed acknowledgment. Then draw the
applicable NVSHMEM barrier alternative. Identify which step completes the
transfer, which notifies the receiver, and which permits reuse. Keep host,
CUDA stream, thread-block, and PE synchronization scopes distinct.

For a proposed persistent device kernel, identify every local and remote
producer needed by the next step. Explain why `__syncthreads` does not join the
whole CUDA grid and why waiting blocks can prevent another block from running.
Check collective-launch capability and grid-size requirements before using
device waits or collectives. A collective launch still needs a correct data
publication and reuse protocol.

Give each neighbor an advancing epoch and trace two buffer reuses. Explain
why one shared count cannot identify a missing neighbor. For an array wait,
identify the active entries and mask semantics. Distinguish ordinary integer
atomic counters from dedicated signal words and use each API's supported type.
Relate the chosen transport's control path to its data path. A GPU-issued
operation or a two-GPU NVLink run does not establish CPU-free network progress
or validate an inter-node IBGDA configuration.

## 7. Explain scaling with an application curve

Run strong and weak scaling as separate experiments. Label what remains fixed,
record actual GPU/rank mapping, and retain complete-output checks. Relate local
work to halo bytes and launch overhead. Small local domains can have too little
interior work to hide communication.

Use unprofiled repeated, interleaved measurements under the repository's
benchmark validity gates for performance claims. A short smoke check validates
only the behavior observed. A planner estimate, bandwidth microbenchmark, or
single best timing is not a measured application scaling curve.
