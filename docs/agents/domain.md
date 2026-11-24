# Domain docs

Read the relevant section of the [repository glossary](../../GLOSSARY.md) when exploring a domain, reviewing a design, or drafting a ticket. Use the owning reference for behavior and validation rules.

| Work area | References |
| --- | --- |
| Benchmark targets, workloads, timing, verification, or expectations | [Benchmark methodology](../../code/docs/benchmark_methodology.md) and the matching branch of the [code instruction router](../../code/AGENTS.md#scoped-authority-map) |
| Harness organization or configuration precedence | [ADR 0001](../../code/docs/adr/0001-harness-organization.md) |
| CLI or MCP entry points and tool contracts | [ADR 0002](../../code/docs/adr/0002-cli-mcp-modularity-and-security.md) and the [MCP tool catalog](../../code/docs/mcp_tools.md) |
| Metric units, run identity, or retained results | [Benchmark artifact contract](../../code/docs/benchmark_methodology.md#artifact-contract) and [warehouse event schema](../../code/docs/performance_warehouse.md#stable-event-schema) |
| Capture bundles and operator replay | [Retained operator replay guide](../../code/labs/trace_replay/README.md) |
| Cluster readiness, canonical runs, or publication | [Two-node run contract](../../code/cluster/docs/canonical_2node_inference_surface.md) and [cluster promotion rules](../../code/cluster/README.md#promotion-rules) |

Keep adopted decisions with the existing [architecture decision records](../../code/docs/adr/README.md) when their rationale and tradeoffs will matter to future work. Flag any proposal that reopens a recorded decision.
