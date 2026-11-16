# Performance Warehouse

## Goal
Treat benchmarking as a first-class product surface instead of a pile of ad hoc runs, screenshots, and dashboards.

The warehouse exists to answer two questions quickly and credibly:

- what changed?
- what raw evidence proves it?

Use this with:

- [`docs/benchmark_methodology.md`](./benchmark_methodology.md) for the run methodology and evidence policy
- [`templates/benchmark_run.yaml`](../templates/benchmark_run.yaml) for the declarative run contract
- [`templates/performance_warehouse_contract.yaml`](../templates/performance_warehouse_contract.yaml) for a concrete schema and retention template
- [`cluster/docs/kubernetes_benchmark_service.md`](../cluster/docs/kubernetes_benchmark_service.md) for the Kubernetes-native control loop

## Product Stance
Performance reporting is split into two separate acts:

- optimization work
- claim making

Optimization can be exploratory. Claims cannot. Claims need:

- written methodology
- pinned artifacts
- repeat runs
- distributions and confidence
- provenance and lineage to raw evidence
- an explicit statement when the number is unstable or not representative

## Storage Split
Keep raw artifacts and analytical tables separate.

| Layer | Contents | Storage expectation | Why it exists |
| --- | --- | --- | --- |
| Raw evidence | logs, traces, profiler reports, manifests, exported metrics, scheduler events, job metadata, request traces | cheap immutable object storage | preserves the unit of record and makes audits possible |
| Curated analytics | run facts, workload dimensions, topology dimensions, serving outcomes, training outcomes, telemetry slices, regressions | columnar analytical store | supports comparisons, dashboards, regression detectors, and release views |
| Hot operational metrics | reduced-cardinality counters and time series | time-series store | supports active debugging and alerting |

Do not force raw traces into the analytical store. Do not force dashboards to scrape raw traces on demand.

## Stable Event Schema
Use one stable event and dimension model across frameworks and hardware generations.

### Core identifiers
Every row should be attributable through stable keys:

- `run_id`
- `benchmark_case_id`
- `workload_spec_digest`
- `artifact_manifest_digest`
- `query_spec_digest` for published rollups
- `scheduler_run_id`
- `job_uid`
- `pod_uid`
- `node_name`
- `gpu_uuid`
- `rank_id` for distributed paths
- `request_id` and `trace_id` for serving paths

### Required dimensions
Curated tables should expose at least these dimensions:

- run identity and cadence
- software versions: image digest, driver, CUDA, NCCL, framework, runtime, server version
- hardware topology: GPU model, GPU count, PCIe layout, NVLink topology, GPU-to-NIC affinity, NUMA layout
- cluster context: cluster, region, availability zone, node pool, scheduler path, resource flavor, tenant mode
- workload parameters: model, precision, prompt and output distributions, concurrency, batching policy, protocol, deployment model

### Required outcomes
Curated tables should expose at least these outcome groups:

- inference: TTFT, inter-token latency, p99, jitter, throughput, queueing delay, errors, cost per request, cost per token
- training: step time, tokens per second, time to train, MFU, scaling efficiency, reliability
- infrastructure: GPU utilization, clocks, power, thermals, PCIe width and speed, NVLink health, NIC loss, storage throughput, node health, scheduler timing

## Canonical Curated Tables
The exact engine can vary. The logical tables should not.

| Table | Grain | Notes |
| --- | --- | --- |
| `benchmark_run_fact` | one row per run | top-level run identity, intent, scheduler path, publication-vs-realism mode |
| `serving_outcome_fact` | one row per run x workload point x concurrency point | TTFT, TPOT, p50/p95/p99, jitter, throughput, error rate, queueing delay, cost |
| `training_outcome_fact` | one row per run x phase x scale point | step time, MFU, scaling efficiency, reliability |
| `telemetry_slice_fact` | one row per run x time window x node/rank/request scope | reduced-cardinality infrastructure and service metrics used for comparisons |
| `software_version_dim` | one row per unique software bundle | image digest and runtime stack |
| `hardware_topology_dim` | one row per unique hardware + topology bundle | GPU, NIC, PCIe, NVLink, NUMA, storage fabric |
| `cluster_region_dim` | one row per cluster/region/zone/scheduler slice | region, zone, node pool, queue, resource flavor |
| `workload_dim` | one row per frozen workload spec | model, prompt and completion distributions, batching, protocol, deployment model |
| `artifact_lineage_dim` | one row per run | raw URIs, manifest digests, signing state, report packet refs |

## Cardinality And Retention
Do not push every high-cardinality label into hot metrics.

### Hot operational metrics
Good labels:

- `run_id`
- `benchmark_case_id`
- `cluster`
- `region`
- `node_pool`
- `gpu_model`
- `scheduler_path`
- coarse `concurrency_bucket`

Bad labels for hot metrics:

- `request_id`
- `trace_id`
- full prompt hash
- arbitrary user tags

Move high-cardinality identifiers into raw logs, traces, or the analytical store.

### Suggested retention tiers
| Tier | Typical contents | Example retention |
| --- | --- | --- |
| Hot | alerting dashboards, short-window operational metrics | 30 days |
| Warm | curated run facts and telemetry rollups | 180 days |
| Cold | immutable raw evidence and long-term warehouse snapshots | 730 days or policy-driven archive |

## Serving Benchmark Matrix
Benchmark serving frameworks with one workload matrix and one service contract.

Keep these fixed unless they are the variable under test:

- prompt and output token distributions
- concurrency levels
- batching policy
- precision
- GPU type
- protocol surface
- deployment model

Frameworks that fit this matrix include:

- `llm-d`
- `vLLM`
- Triton-based paths
- KServe-style serving stacks

If protocol or deployment model changes, record it explicitly. Otherwise the comparison is polluted by API or control-plane semantics.

## Telemetry Join Model
The warehouse has to make this pivot cheap:

`run got slower -> affected workload points -> abnormal nodes/ranks/requests -> raw evidence`

Required telemetry families:

- service metrics and request traces
- GPU metrics via `dcgm-exporter`
- NVLink metrics via `nvlink-exporter`
- PCIe metrics via `node-pci-exporter`
- loss and latency probes via `ping-exporter`
- node health via `node-problem-detector`
- cluster validation probes via `hpc-verification`
- scheduler and admission timing from Kubernetes, Kueue, and SLINKY

When possible, normalize these signals onto the same run identity and time windows instead of relying on fuzzy timestamp joins alone.

## Scenario Playbooks
The system should make these scenarios debuggable by default.

### Scenario 1: Inference p99 doubled, average latency barely changed
Interpretation: tail amplification, not raw compute regression.

Check first:

- request queue time
- pod CPU throttling
- autoscaling, restart, or cold-start activity
- node-local DNS or service-resolution anomalies
- whether a dedicated node pool removes the p99 regression

### Scenario 2: Training throughput dropped 20 percent and GPU utilization is low
Interpretation: GPUs are waiting.

Check first:

- CPU throttling
- dataloader or preprocessing saturation
- storage throughput
- network stalls
- cached-input or microbenchmark reruns to isolate model code from the input path

### Scenario 3: Distributed training has step spikes and one node seems slower
Interpretation: classic straggler behavior.

Check first:

- rank-level collective latency and bandwidth
- PCIe width and speed
- NVLink health
- thermal signals
- packet loss or NIC errors
- whether removing or cordoning the suspect node restores the distribution

### Scenario 4: Jobs are pending too long and cluster utilization is worse than expected
Interpretation: scheduler or quota pressure.

Check first:

- Kueue `ClusterQueue` pressure and admission behavior
- Kubeflow or CRD admission state if applicable
- SLINKY segment or topology fragmentation
- over-constrained NFD labels, affinities, or topology exclusivity

## Published Number Lineage
Every published number should resolve through:

1. the dashboard or report row
2. the analytical query or materialized view definition
3. the `run_id`, `workload_spec_digest`, and `artifact_manifest_digest`
4. the raw artifact set in immutable storage

If a number cannot be walked back through those steps, it is not publication-safe.

## Stakeholder Views
The same evidence should be rendered differently for each audience.

| Audience | What they need |
| --- | --- |
| Engineering | traces, metrics, narrow hypothesis, minimal repro, proposed experiment |
| Product | impact on latency, throughput, reliability, or cost; expected user impact; confidence and next step |
| Partner or OSS | what is ours, what is upstream, minimal repro, smallest actionable change |

## Repo Mapping
The repo now has the pieces for this contract even if it does not yet ship the full production warehouse:

- [`templates/benchmark_run.yaml`](../templates/benchmark_run.yaml) carries warehouse sinks and observability joins
- [`core/scripts/validate_benchmark_run.py`](../core/scripts/validate_benchmark_run.py) enforces the minimum contract locally and in CI
- [`cluster/docs/kubernetes_benchmark_service.md`](../cluster/docs/kubernetes_benchmark_service.md) describes how a future operator would populate the warehouse
- [`templates/performance_warehouse_contract.yaml`](../templates/performance_warehouse_contract.yaml) is the concrete schema and retention template

Thin interface surfaces are also exposed now:

- CLI: `python -m cli.aisp tools benchmark-contracts`
- MCP: `benchmark_contracts`
