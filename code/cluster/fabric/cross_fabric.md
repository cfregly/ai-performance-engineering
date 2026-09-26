# Cross-Fabric Interpretation

## Why This Exists

Fabric validation in this repo is not a separate networking lab. It is tied directly to AI Systems Performance artifacts:

- NCCL all-reduce
- NCCL all-to-all
- torchrun connectivity
- train-step
- vLLM concurrency and request-rate sweeps
- nvbandwidth

The fabric question is always: what does the control plane say, what does the runtime say, and how does that line up with the workload outcome?

## Completeness Model

Each family lands in one of four states:

- `not_present`
- `present_unverified`
- `runtime_verified`
- `full_stack_verified`

Management endpoints that are missing or inaccessible produce structured `not_configured` or `unavailable` states. They do not disappear from the report.

## Fast Interpretation Rules

Use [cross-layer diagnosis](../../core/analysis/cross_layer_diagnosis.md) to test
whether workload symptoms overlap measured counter changes on a matching path.
The tool requires interval deltas and a shared clock or a measured clock-skew
bound. Peak bandwidth ratios alone do not distinguish rank waiting, contention
and lost overlap.


- `runtime_verified` + weak AI workload scaling usually means the runtime path is alive but something about routing, congestion, queueing, or placement still needs work.
- `full_stack_verified` + weak scaling shifts suspicion toward host/runtime choices, launch shape, or application behavior.
- `present_unverified` means the hardware is there but the evaluation still lacks enough evidence for a publish-grade claim.

## Core Questions To Answer

- Is the topology what the operator expects?
- Is the management plane visible and healthy?
- Are the runtime collectives and connectivity probes healthy?
- Do train-step or vLLM results show a knee or collapse that matches the fabric evidence?

## Scenario Reading Order

Read the fabric families in this order:

1. NVLink / NMX
   - topology capacity planning
   - tenant partition readiness
   - switch and cable telemetry
2. InfiniBand
   - capacity and path visibility
   - routing and counter verification
   - runtime correlation against NCCL and all-to-all
3. Spectrum-X / RoCE
   - fabric readiness (RoCE QoS, adaptive routing, BGP)
   - runtime correlation against NCCL and all-to-all

The point is not to keep three disconnected networking playbooks. The point is to answer the same operator questions regardless of fabric family.

## Artifact Reading Order

1. `<run_id>_fabric_capability_matrix.json`
2. `<run_id>_fabric_verification.json`
3. `<run_id>_fabric_ai_correlation.json`
4. `<run_id>_fabric_scorecard.json`
5. `<run_id>_cluster_scorecard.json`

That sequence moves from "what is present" to "what was verified" to "what it means for AI workloads".

## How To Interpret Weak Scaling

- NVLink weak scaling with strong NMX topology/telemetry usually pushes suspicion toward placement, host runtime, or the workload itself.
- InfiniBand weak scaling with weak `ibdiagnet`, `saquery`, `ibtracert`, or `perfquery` signals points toward routing, subnet health, or counter-visible congestion.
- Spectrum-X weak scaling with weak QoS, adaptive-routing, or BGP route visibility points toward Ethernet/RDMA fabric readiness before model tuning.
- When a family is only `runtime_verified`, treat the workload as "working but not fully explained" until the management plane is visible too.
