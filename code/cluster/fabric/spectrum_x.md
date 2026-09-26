# Spectrum-X / RoCE Track

Use the [fabric diagnosis tool](diagnostics.md) for runnable collection of route,
QoS and timed port evidence. ECMP distributes flows over paths. ECN marks and CNP
report congestion to rate-control mechanisms. PFC pauses a configured link
priority. These are different signals and should remain separate in reports.
Correlate deltas with the workload interval, selected rail and application trace.
Configured QoS alone does not establish that congestion control worked during a run.

## What This Track Answers

Use this track when the operator question is: "Is the Ethernet/RDMA fabric configured the way an AI workload expects, and does runtime behavior agree with that configuration?"

This track is organized around two answers:

- is the fabric ready from a RoCE, adaptive-routing, and route-health perspective?
- do multi-node AI runtime artifacts agree with the switch-side evidence?

## Scenario: Fabric Readiness

Start here before treating a slow NCCL or all-to-all run as an application problem.

Host-side hints:

- `ethtool`
- `ibstat`
- `rdma link`
- `ibv_devinfo`

Switch-side checks:

- `nv show router adaptive-routing`
- `nv show qos roce`
- `nv show vrf default router bgp neighbor`
- `vtysh -c "show bgp ipv4 unicast summary"`
- `vtysh -c "show ip route vrf default bgp"`

What to record:

- whether high-speed Ethernet is visible from the host
- whether RDMA-over-Ethernet signals are present
- whether RoCE QoS is visible on the switches
- whether adaptive routing is configured
- whether BGP neighbor state and route visibility are available

Interpretation:

- RoCE QoS plus adaptive routing plus BGP visibility forms the minimum publish-grade readiness story.
- Healthy runtime with missing switch access is still only `runtime_verified`, not `full_stack_verified`.
- Route visibility problems should be treated as control-plane degradation even if the workload still moves packets.

## Scenario: Runtime Correlation For AI Workloads

This repo correlates Spectrum-X readiness with the same workload artifacts used everywhere else:

- `<run_id>_2nodes_nccl.json`
- `<run_id>_2nodes_alltoall_nccl_alltoall.json`
- `<run_id>_torchrun_connectivity_probe.json`
- optionally train-step, vLLM, and fio artifacts when those are part of the run

What to look for:

- multi-node NCCL algbw vs single-node NCCL algbw
- all-to-all behavior for exchange-heavy patterns
- world-size and payload success from torchrun connectivity
- whether train-step or vLLM latency knees line up with route/QoS/adaptive-routing evidence

Interpretation rules:

- low multi-node NCCL / single-node NCCL ratio: inspect RoCE QoS, adaptive routing, BGP route visibility, and host RDMA binding
- healthy connectivity probe but weak multi-node throughput: the path exists, but congestion handling or route health still needs work
- healthy switch evidence plus weak application throughput: shift suspicion toward launch shape, queue depths, or model/runtime choices

## Required Management Inputs

Pass these when you want publish-grade Spectrum-X / Cumulus verification:

- `--cumulus-hosts <host1,host2>`
- optionally `--cumulus-user <user>`
- optionally `--cumulus-ssh-key <path>`

If switch access is missing, the repo must report that explicitly. It must not silently claim congestion-feature coverage.

## Typical Failure Shapes

- high-speed Ethernet detected but no switch access: runtime may be green while congestion features remain unverified
- BGP or adaptive-routing checks failing: treat as control-plane degradation, not as a soft warning
- weak multi-node NCCL/all-to-all with Spectrum-X present: correlate QoS, route state, and adaptive-routing evidence before tuning workloads
- healthy control plane plus weak AI workloads: move attention toward host binding, queueing, or application-level contention
