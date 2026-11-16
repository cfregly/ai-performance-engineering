# NVSHMEM device exercises

## 1. Follow scalar halo ownership

Run `scalar-halo` with odd and even iteration counts and an uneven row
decomposition. Identify the single owner of each outgoing boundary value and
the receiver's logical halo offset. Explain how a device scalar put differs
from a host-on-stream put and why the next stencil step still needs the
on-stream barrier. Verify the full final grid, including the fixed x boundaries.

## 2. Keep two edges when both neighbors are one peer

For each PE in a two-PE ring, write down where outgoing edges 0 and 1 land.
Trace incoming edge 0 and edge 1 through the independent CPU oracle. Run one,
two, and several epochs. Check every history element instead of comparing only
the final inbox. Explain which defects the source, edge, epoch, and element
terms in the payload formula reveal.

## 3. Separate completion, readiness, and consumption

Draw the tile protocol in the [README](README.md#trace-a-tile-epoch-before-changing-it).
For each arrow, identify the local thread, local CTA, or remote PE it orders.
Explain why the CTA must synchronize before elected-thread quiet and before
the consumed acknowledgment. Explain why quiet alone does not notify the
receiver and ready alone does not permit inbox reuse.

Compare monotonic `uint64_t` signals with ordinary integer counters. Explain why
`wait_until_all(..., 2, nullptr, GE, epoch)` includes both incoming edges. Work
through an old ready value of one at epoch two. Do not remove acknowledgments
or mutate synchronization state in a live run to demonstrate the failure.

## 4. Account for the participating threads

Find every `_block` API in the native exchange. Confirm that all CTA threads
reach each call with identical arguments. Find the collective-launch capacity
query and rejection path. Explain what one CTA allows each `__syncthreads()`
to cover and why adding a second CTA would require a different protocol.

Read the [launch API](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html).
Explain why `nvshmemx_collective_launch` launches only on the calling PE's GPU
and does not synchronize the PE launches. Identify the peer participation and
ready/consumed operations that remain necessary after the local capacity check.

Use the [NVSHMEM CUDA model](https://docs.nvidia.com/nvshmem/api/latest/cuda-interactions.html)
to explain a wait that occupies the resources needed by its matching producer.
Distinguish GPU initiation from a guarantee of a particular network transport.

## 5. Compare equivalent tile work

Run `block-signal`, `persistent-signal`, and `persistent-atomic` with identical
elements, timed epochs, and warmup. Retain all dumps and compare every element
across modes and with the CPU oracle. Report launch count and the additional
protocol state beside timing. Do not compare these tile timings directly with
the scalar stencil's different numerical work.

Use repeated, interleaved, unprofiled runs under the repository's validity rules
before making a performance claim. Keep setup and teardown visible, even when
the intended comparison times only steady execution. A correctness smoke run
is evidence for the observed case, not a scaling result.

## 6. Follow persistent stencil reuse

Run `scalar-halo` and `persistent-halo` for the same grid, odd and even steps,
and warmup. Include one-row, two-row, and uneven two-rank decompositions. Compare
every final grid element with the independent CPU oracle and across modes.

At zero-based step `s`, mark the last read of halo epoch `s`, its acknowledgment,
and the overwrite of the alternating remote halo destination. Explain the
`consumed >= s - 1` threshold and why the final consumed epoch is
`iterations - 1`. Distinguish local RMA source reuse from remote destination
reuse. Compare scalar interior-element puts with full-row block puts, including
fixed boundary values. Explain why one CTA provides local barrier coverage but
limits the stencil's compute throughput.

## 7. Match profiler evidence to its scope

Inspect all five modes in retained Nsight Systems reports. Identify the actual
steady range and explain the difference between a repeated exchange kernel and
one persistent kernel. State what the timeline reveals and what it cannot
separate inside the persistent loop.

In `block-signal`, select only the pure `prepare_payload` for Nsight Compute
kernel replay. Confirm its source has no remote write, atomic, or peer wait.
Keep its metrics labeled as preparation. Explain why replaying the exchange or
persistent kernel against an advanced peer is unsafe.

Follow the [tested profiling recipe](README.md#profile-the-protocol-that-actually-ran)
under the repository clock-lock context on both allocated GPUs. Profile one
rank with `profile_rank.sh` while its peer runs directly, then repeat in a fresh
job for the other rank. Use the mode's actual kernel-name pattern and the
independent full-output check.

Inspect the ten retained results in the [B200 record](b200_validation.json).
Explain why one requested duration metric does not prove one pass and why
`--launch-count 1` does not control the number of profiler passes. Use the
[native-report inspector](inspect_ncu_replay_metrics.py) to check the observed
one pass and zero profiler warmup. Backup-byte metrics are absent in these
reports. Explain the difference between that absence and a measured zero, and
why the documented single-pass contract supports the no-save/restore inference.

Compare a scalar step, a block exchange epoch, and a complete persistent
kernel's scope before comparing their durations. Keep those durations separate
from unprofiled solve latency and from full hardware-counter evidence. The
earlier whole-launcher application-range attempts remain incomplete diagnostics.
