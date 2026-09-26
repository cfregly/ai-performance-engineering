# InfiniBand Track

Use the [fabric diagnosis tool](diagnostics.md) to retain counters across the
selected ports and paths, then use [transport isolation](transport.md) to compare
host buffers, GPU buffers and collectives. Receiver credits and credit-wait
counters describe a different mechanism from Ethernet PFC. A cumulative counter
snapshot cannot establish that a workload encountered congestion.

## What This Track Answers

Use this track when the operator question is: "Is the lossless multi-node fabric actually healthy enough for AI workloads?"

This track is organized around three answers:

- can I see the HCAs, switches, hosts, and subnet paths I expect?
- can I verify routing and counters from the IB control plane?
- do NCCL, all-to-all, and torch distributed behavior agree with what the control plane says?

## Scenario: Capacity And Path Visibility

This is the first operator pass before blaming workloads.

Host-side inventory:

- `ibstat`
- `rdma link`
- `ibv_devinfo`

Management-plane inventory:

- `ibswitches`
- `ibhosts`
- `iblinkinfo`
- `ibnetdiscover`
- `saquery`

What to record:

- which `mlx5_*` HCAs are visible
- whether `Link layer: InfiniBand` is reported
- how many switch and host records appear from `ibswitches` and `ibhosts`
- whether `iblinkinfo`, `ibnetdiscover`, and `saquery` all return path/subnet visibility

Interpretation:

- HCAs present plus active IB link-layer signals means the host likely belongs on the IB path.
- `ibswitches` + `ibhosts` + `ibnetdiscover` together answer the "what is attached to what?" question.
- If host inventory is green but management inventory is absent, the repo should report `runtime_verified` or `present_unverified`, not a silent pass.

## Scenario: Routing And Counter Verification

Once the basic path is visible, move to route and counter correctness.

Primary commands:

- `ibdiagnet -r`
- `saquery`
- `ibaddr`
- `ibtracert`
- `ibroute`
- `perfquery`

What to confirm:

- `ibdiagnet -r` completes cleanly enough to trust subnet visibility
- `saquery` returns expected subnet-manager information
- `ibtracert` and `ibroute` can explain how traffic should traverse the fabric
- `perfquery` exposes counters you can compare against runtime symptoms

Interpretation:

- `ibdiagnet` is the broadest single "is the subnet healthy?" check in this track.
- Route tools answering cleanly but weak runtime scaling usually means congestion, queueing, or binding still needs attention.
- Route tools failing means you do not yet have publish-grade control-plane evidence, even if workloads still run.

## Scenario: Runtime Correlation For AI Workloads

The repo ties InfiniBand verification to the same runtime artifacts used by the rest of AI Systems Performance:

- `<run_id>_2nodes_nccl.json`
- `<run_id>_2nodes_alltoall_nccl_alltoall.json`
- `<run_id>_torchrun_connectivity_probe.json`
- optionally train-step, vLLM, and fio artifacts when those workloads are in scope

What to look for:

- multi-node NCCL algbw vs single-node NCCL algbw
- all-to-all behavior for path-sensitive exchange patterns
- torchrun world-size and payload probe success
- whether train-step or vLLM knees line up with routing/counter anomalies

Interpretation rules:

- low multi-node NCCL / single-node NCCL ratio: inspect routing, congestion, HCA/interface binding, and subnet state before tuning the model stack
- healthy `torchrun` probe but weak NCCL/all-to-all: the path exists, but the performance shape still points to routing, congestion, or placement
- healthy control plane plus weak AI workloads: shift suspicion toward launch shape, host binding, queue depths, or application behavior

## Required Management Inputs

Pass these when you want publish-grade IB control-plane verification:

- `--ib-mgmt-host <host>`
- optionally `--ib-mgmt-user <user>`
- optionally `--ib-mgmt-ssh-key <path>`

If they are missing, the repo must report that explicitly. It must not silently downgrade the control plane.

## Typical Failure Shapes

- HCAs present but management host missing: `present_unverified` or `runtime_verified`, never a silent skip
- `ibdiagnet` or `saquery` failing: treat as control-plane visibility loss, not as a runtime pass
- weak NCCL/all-to-all with healthy route visibility: look for congestion, host binding mistakes, or oversubscription
- weak NCCL plus weak path visibility: fix control-plane confidence first, then rerun the workload correlation
