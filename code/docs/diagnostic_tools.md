# Network, fabric and serving diagnosis

These tools accompany Chapters 3, 4, 15 and 17. They collect and analyze evidence
for performance questions that an isolated kernel benchmark cannot answer.

See the [validation record](diagnostic_tools_validation.md) for the executed
checks and the remaining hardware qualification.

| Question | Command | Guide |
| --- | --- | --- |
| Why is this TCP connection slow or failing? | `aisp tools network-diagnose` | [Network diagnosis](../ch03/network_diagnosis.md) |
| What changed on the selected RoCE or InfiniBand path? | `aisp tools fabric-diagnose` | [Fabric diagnosis](../cluster/fabric/diagnostics.md) |
| Is the limit in host RDMA, GPU RDMA or collectives? | `aisp tools transport-diagnose` | [Transport diagnosis](../cluster/fabric/transport.md) |
| Are collectives waiting, contending or losing overlap? | `aisp tools collective-diagnose` | [Collective diagnosis](../ch04/collective_diagnosis.md) |
| Which engine and serving architecture meets the workload's latency objectives? | `aisp tools serving-compare` | [Serving comparison](../labs/serving_comparison/README.md) |
| How do I expose the supported SGLang build's native failure counters before its first failure? | `aisp tools serving-prepare-runtime` | [Runtime preparation](../labs/serving_comparison/README.md#prepare-sglang-failure-counters) |
| Does a workload symptom coincide with fabric evidence? | `aisp tools cross-layer-diagnose` | [Cross-layer diagnosis](../core/analysis/cross_layer_diagnosis.md) |

Run from `code/`. Inspect a tool's exact arguments through the same public entrypoint:

```bash
python -m cli.aisp tools network-diagnose -- --help
python -m cli.aisp tools fabric-diagnose -- --help
python -m cli.aisp tools transport-diagnose -- --help
python -m cli.aisp tools collective-diagnose -- --help
python -m cli.aisp tools serving-compare -- --help
python -m cli.aisp tools serving-prepare-runtime -- --help
python -m cli.aisp tools cross-layer-diagnose -- --help
```

The MCP `tools_diagnostics` tool dispatches the same command registry. For example:

```json
{"tool": "network-diagnose", "args": ["--help"], "timeout_seconds": 60}
```

Use an existing serving deployment or explicitly selected host pair for load
tests. State collection and offline analysis do not configure switches. The tools
retain failures instead of substituting a CPU or alternate transport result.

Every measurement needs its workload shape, path, version and collection interval.
Guides explain the required hardware and the limits of each result. CPU-only
tests exercise parsers, calculations, protocol handling and command dispatch.
They do not establish GPU correctness, fabric health or engine performance.

Diagnostics are deliberately separate from benchmark optimization pairs. To claim
a speedup, use the repository's correctness, clock, repeated-measurement and
profiling gates with an equivalent workload. A diagnostic run can identify the
next experiment without satisfying those publication requirements.
