# Benchmark Methodology

## Purpose
This repo already had benchmark harnesses, manifests, tier-1 suites, and cluster evaluation bundles. What it lacked was one explicit methodology that says how those pieces should be used when the output needs to stand up in engineering reviews, release gates, customer escalations, analyst briefings, or benchmark publications.

This document is that contract.

Use it with:

- [`templates/performance_intake.yaml`](../templates/performance_intake.yaml) for KPIs, constraints, and the variable under test.
- [`templates/benchmark_workload_spec.yaml`](../templates/benchmark_workload_spec.yaml) for the frozen workload definition and measurement policy.
- [`templates/benchmark_run.yaml`](../templates/benchmark_run.yaml) for the declarative `BenchmarkRun` contract.
- [`docs/performance_warehouse.md`](./performance_warehouse.md) and [`templates/performance_warehouse_contract.yaml`](../templates/performance_warehouse_contract.yaml) for the raw-versus-curated warehouse design, telemetry joins, retention tiers, and lineage policy.
- [`docs/perf_intake_and_triage.md`](./perf_intake_and_triage.md) for the fast intake + first-pass collection flow.
- [`cluster/docs/kubernetes_benchmark_service.md`](../cluster/docs/kubernetes_benchmark_service.md) and [`cluster/configs/benchmarkrun-crd.yaml`](../cluster/configs/benchmarkrun-crd.yaml) for the Kubernetes-native service direction already being sketched in this repo.

## The Three-Layer Stack
Use all three layers together. Do not skip straight to end-to-end numbers and then guess at root cause.

| Layer | Goal | Typical repo entrypoints | Questions it answers |
| --- | --- | --- | --- |
| Microbenchmarks | Isolate subsystem limits and regressions. | `python -m cli.aisp benchmark memory|cache|roofline|pcie|tc|disk|nccl|p2p|speed`, `cluster/scripts/run_nccl_all_reduce.sh`, `cluster/scripts/run_fio_bench.sh`, `cluster/scripts/run_gemm_bench.sh` | Is the problem compute, communication, storage, or a specific kernel/runtime path? |
| Component benchmarks | Measure serving, dataloaders, job startup, and scheduler paths with pinned workload semantics. | `python -m cli.aisp bench run --targets ... --profile minimal|deep_dive`, [`core/scripts/profiling/perf_triage_bundle.sh`](../core/scripts/profiling/perf_triage_bundle.sh), `cluster/scripts/run_vllm_bench.sh`, `cluster/scripts/run_torchrun_transformer_train_step.sh` | Which subsystem regresses the user-visible path? |
| End-to-end benchmarks | Validate realistic workflows after lower layers are understood. | `python -m cli.aisp bench run-tier1 --single-gpu --profile minimal`, `python -m cli.aisp cluster common-eval --preset common-answer-fast|modern-llm`, end-to-end labs under `labs/real_world_models` | What does the user or trainer actually feel after all interactions and queueing are included? |

The methodology program should model all three layers even if a single `BenchmarkRun` only enables a subset.

## Workload Freeze Rule
Before comparing anything, freeze the workload definition:

- same model
- same sequence-length mix
- same precision
- same batching policy
- same concurrency model
- same dataset or prompt corpus
- same correctness policy

The human-editable freeze contract lives in [`templates/benchmark_workload_spec.yaml`](../templates/benchmark_workload_spec.yaml). The declarative/operator-facing version lives in [`templates/benchmark_run.yaml`](../templates/benchmark_run.yaml) and can be checked with [`core/scripts/validate_benchmark_run.py`](../core/scripts/validate_benchmark_run.py).

## Comparison Rule
Compare one variable at a time. Valid candidates include:

- hardware generation
- runtime version
- scheduler path
- control-plane path
- driver stack
- network topology
- storage stack

If more than one variable moves, the run is exploratory, not publication-grade.

## Dean/Ghemawat Optimization Loop

This repo adapts Jeff Dean and Sanjay Ghemawat's [Performance Hints](https://abseil.io/fast/hints.html) into an evidence loop for Python, PyTorch, CUDA, Triton, and distributed workloads. Their source document is centered on single-binary performance and mostly C++ examples; the categories transfer, but GPU and distributed claims still require repo-native measurements.

1. **Estimate:** state invocation frequency and approximate bytes, allocations, launches, synchronizations, locks, and boundary crossings. Reject ideas whose best-case ceiling cannot move the primary KPI.
2. **Measure:** capture a representative baseline and profile. A static code smell is a hypothesis, not proof that it matters.
3. **Choose the highest-leverage class:** algorithm or work avoided; bulk API shape; compact/local representation; fewer allocations and copies; less synchronization or better parallelism; smaller generated code.
4. **Change one mechanism:** preserve the frozen workload and keep the public interface stable when the optimization can live behind an encapsulation boundary.
5. **Verify:** pass correctness first, then repeated paired measurements, profiler/counter attribution, and the artifact/provenance contract.
6. **Bound the claim:** distinguish micro, component, and end-to-end effects and state excluded setup, warmup, compile, graph-capture, or teardown costs.

The executable agent workflow lives in [`../../.agents/skills/dean-performance-review/SKILL.md`](../../.agents/skills/dean-performance-review/SKILL.md). The intake fields live in [`../templates/performance_intake.yaml`](../templates/performance_intake.yaml).

## Metrics By Workload Type
### Training
Track these as primary metrics:

- time to train
- MFU
- scaling efficiency
- training reliability

### Inference
Track these as primary metrics:

- TTFT
- tokens per second
- p99 latency
- jitter
- cost per token or request

Secondary metrics such as request goodput, p50 latency, GPU utilization, NIC throughput, and power are diagnostic, not substitutes for the user-facing outcome.

## Artifact Contract
Treat benchmark artifacts like release artifacts.

Every serious run must capture:

- pinned workload spec
- image digests, not mutable image tags
- driver, CUDA, NCCL, framework, and runtime versions
- exact hardware topology
- immutable raw artifacts
- audit trail linking the claim back to the run

Every serious benchmark program should also preserve:

- raw logs, traces, and profiler outputs in immutable low-cost storage
- curated fact and dimension tables in a columnar analytical store
- retention tiers for hot, warm, and cold data
- a lineage path from any published number back to the raw artifact manifest
- cardinality budgets so operational metrics remain usable

For publication-grade work, also require signed provenance. The repo template models this through `spec.provenance.signing`, even if the attestation backend is provided by your deployment environment rather than by the harness itself.

The current harness already captures manifests, profiler artifacts, raw timings, and artifact hashes. Cryptographic provenance signing is therefore a policy requirement that still depends on deployment or CI infrastructure today; do not imply it exists if the attestation path was not actually used.

Existing repo primitives that already support this:

- `core/benchmark/run_manifest.py`
- `cluster/runs/<run_id>/manifest.json`
- tier-1 history packages under `artifacts/history/tier1/<run_id>/`
- canonical cluster packages under `cluster/published/current/`

## Trial Policy
Do not report a single run.

- run multiple trials
- keep raw distributions
- report confidence
- apply explicit outlier policy
- decompose performance shifts into compute, communication, storage, and orchestration overhead

The repo `BenchmarkRun` validator enforces:

- `minReplicates >= 3`
- confidence level must be declared
- outlier policy must be declared
- distributions and confidence intervals must be reported
- rank-level outliers must be surfaced for distributed runs

## Bottleneck Taxonomy
Classify the bottleneck before choosing instrumentation.

| Bottleneck | What to inspect first | Repo-native signal |
| --- | --- | --- |
| Compute-bound | GPU counters, NCU, kernel timelines, model-server metrics | `profile_ncu`, tier-1 deep dive, Nsight artifacts |
| Comm-bound | NCCL traces, RDMA probes, NVLink/NIC health | cluster all-reduce, all-gather, stability suites |
| Input-bound | storage probes, network probes, dataloader metrics | fio bundles, data pipeline sweeps, triage bundle |
| Control-plane-bound | scheduler timing, queue depth, job startup timing | readiness runs, startup benchmarks, suite steps |

Do not over-index on GPU utilization alone. Low utilization can mean dataloader starvation, CPU throttling, queueing, network stalls, slow collectives, or bad topology placement.

## Distributed Rule: The Slowest Rank Wins
Distributed training and serving are gated by the slowest rank or node.

Always collect:

- rank-level collective latency and bandwidth
- rank or node outliers
- RDMA path validation
- GPU-to-NIC affinity
- PCIe width and speed
- NVLink health
- thermal throttling

If one GPU or one node is consistently slow, isolate it and check whether the cluster returns to baseline. In Kubernetes environments, make node cordon or removal an explicit remediation path, not an ad hoc decision.

Relevant repo surfaces:

- `cluster/scripts/run_allreduce_stability.sh`
- `cluster/scripts/run_gemm_sanity_all_nodes.sh`
- `cluster/analysis/plot_mamf.py`
- `cluster/common-eval --preset modern-llm`

## Serving Stack Comparison Rule
When benchmarking `llm-d`, `vLLM`, Triton, KServe-style serving paths, or equivalent stacks, keep the serving contract fixed unless that contract is the variable under test.

Freeze:

- protocol surface
- deployment model
- prompt and output token distributions
- concurrency sweep points
- batching policy
- precision
- GPU type

If protocol behavior or deployment shape changes between candidates, record that explicitly. Otherwise you are often measuring framework API behavior or control-plane differences more than serving efficiency.

## Publication-Grade Versus Realism-Grade
These are different test modes and should stay different.

| Mode | Isolation policy | Why it exists |
| --- | --- | --- |
| Publication-grade | Dedicated nodes, stable background load, fixed topology, topology-exclusive scheduling when appropriate | Hard regression gates, public claims, RFPs, analyst or benchmark publications |
| Realism-grade | Multi-tenant scenarios included, cluster context preserved, background load tolerated and recorded | Customer-experience testing and operator reality |

Both modes must record enough cluster context to explain outliers later.

## Declarative BenchmarkRun
The repo now exposes a portable `BenchmarkRun` spec:

- Template: [`templates/benchmark_run.yaml`](../templates/benchmark_run.yaml)
- Validator: [`core/scripts/validate_benchmark_run.py`](../core/scripts/validate_benchmark_run.py)
- Kubernetes CRD: [`cluster/configs/benchmarkrun-crd.yaml`](../cluster/configs/benchmarkrun-crd.yaml)

Validate a spec locally:

```bash
python -m core.scripts.validate_benchmark_run --file templates/benchmark_run.yaml
```

Use this spec as the stable interface between:

- workload owners
- performance engineers
- cluster operators
- CI/CD schedulers
- analytical storage

The `BenchmarkRun` contract also models:

- observability join keys that tie cluster, job, request, rank, and hardware telemetry together
- raw artifact versus curated warehouse sinks
- retention tiers and lineage requirements
- scenario playbooks for tail latency, starvation, stragglers, and scheduler backpressure

## Repo Mapping
Use the declarative spec to choose the right execution path instead of inventing a new one per workload.

| Need | Repo command |
| --- | --- |
| Fast benchmark health signal | `python -m cli.aisp bench run-tier1 --single-gpu --profile minimal` |
| Performance intake + quick instrumentation bundle | follow [`docs/perf_intake_and_triage.md`](./perf_intake_and_triage.md) |
| Common cluster evaluation | `python -m cli.aisp cluster common-eval --preset common-answer-fast` |
| Full modern LLM cluster evaluation | `python -m cli.aisp cluster common-eval --preset modern-llm` |
| Multi-node contract validation before workloads | `python -m cli.aisp cluster common-eval --preset multinode-readiness` |
| Canonical 2-node inference serving lane | `cluster/docs/canonical_2node_inference_surface.md` + `python -m cli.aisp cluster common-eval --preset multinode-readiness` followed by `--preset modern-llm` |

## Kubernetes Service Design
The declarative service design requested in the benchmark methodology lives in [`cluster/docs/kubernetes_benchmark_service.md`](../cluster/docs/kubernetes_benchmark_service.md).

The warehouse design that backs those runs lives in [`docs/performance_warehouse.md`](./performance_warehouse.md).

Thin interface surfaces for these contracts are exposed through:

- `python -m cli.aisp tools benchmark-contracts`
- dashboard API `GET /api/benchmark/contracts`
- MCP tool `benchmark_contracts`

That design makes the service Kubernetes-native without pretending the current repo already ships a full operator:

- `BenchmarkRun` is the declarative workload and measurement contract
- an operator resolves it into SLINKY and Kueue execution primitives
- hot metrics land in time-series storage
- long-term regression data lands in an analytical warehouse
- canary, nightly, and pre-release runs are scheduled automatically
