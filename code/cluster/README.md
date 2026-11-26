# Cluster Evaluation Harness

This folder has one contract that matters for new work:

```text
cluster/runs/<run_id>/
  manifest.json
  structured/
  raw/
  figures/
  reports/
```

If you are running benchmarks, consuming results from MCP/CLI/dashboard, or handing the output to another tool, use the run directory as the unit of record.

## Start Here
Use this for the common "evaluate this system" ask:

```bash
python -m cli.aisp cluster common-eval \
  --preset common-answer-fast \
  --run-id <run_id> \
  --hosts localhost \
  --labels localhost \
  --ssh-user "$(id -un)" \
  --primary-label localhost \
  --extra-arg --skip-render-localhost-report
```

Presets:
- `common-answer-fast`: the fastest useful answer bundle; NCCL, vLLM concurrency, vLLM request-rate, GEMM, GPU STREAM, fio, quick nvbandwidth, scorecard, coverage, MLPerf alignment.
- `core-system`: the standard full single-node/system bundle.
- `modern-llm`: the canonical full bundle with advanced LLM/distributed signals.
- `fabric-systems`: `modern-llm` plus the fabric capability matrix, verification ladder, AI-correlation artifact, and fabric scorecard. This preset defaults to capability-aware partial completion instead of strict canonical completeness gating.
- `multinode-readiness`: checks contract and environment only; no workloads.

Canonical 2-node serving lane:
- use `multinode-readiness` first as the dry-run/preflight gate for the exact host, label, and interface contract
- then use `modern-llm` on exactly 2 hosts to capture the canonical multinode vLLM + communication evidence bundle
- operator contract, example spec files, and required artifacts live in `cluster/docs/canonical_2node_inference_surface.md`

Dedicated fabric entrypoint:

```bash
python -m cli.aisp cluster fabric-eval \
  --run-id <run_id> \
  --hosts <h1,h2> \
  --labels <l1,l2> \
  --ssh-user <user> \
  --ssh-key <key>
```

The dedicated fabric path defaults to capability-aware partial completion. Use `--extra-arg --strict-canonical-completeness` only when you explicitly want the broader publish-grade completeness gates.

Localhost fabric default:
- single-node localhost `cluster fabric-eval` now auto-resolves to the canary vLLM profile when you do not provide explicit vLLM overrides
- current canary contract is `openai-community/gpt2`, `TP=1`, `ISL=64`, `OSL=32`, concurrency `1`, repeats `1`, with request-rate sweep disabled by default
- heavier localhost serving remains available through the existing explicit overrides (`--extra-arg --model ...`, `--extra-arg --isl ...`, `--extra-arg --osl ...`, `--extra-arg --concurrency-range ...`, and request-rate flags)

Fabric handbook:
- `cluster/fabric/README.md`
- `cluster/fabric/fabric_command_catalog.json`

## Declarative BenchmarkRuns
When you need the benchmarking service contract instead of just the scripts:

- freeze the workload first with `templates/performance_intake.yaml` and `templates/benchmark_workload_spec.yaml`
- start from `templates/benchmark_run.yaml`
- validate it with `python -m core.scripts.validate_benchmark_run --file templates/benchmark_run.yaml`
- use `cluster/configs/benchmarkrun-crd.yaml` as the Kubernetes CRD shape
- read `cluster/docs/kubernetes_benchmark_service.md` for the intended SLINKY + Kueue operator mapping

That path is for declarative scheduling, CI automation, and publication-vs-realism policy. The existing `common-eval` presets remain the current execution backend.

## What To Open
- `cluster/runs/<run_id>/`: new run results.
- `cluster/published/current/`: the current published canonical package.
- `cluster/field-report.md`: published report package.
- `cluster/field-report-notes.md`: published evidence ledger.
- `cluster/docs/advanced-runbook.md`: operator reference for running, promoting, and validating canonical packages.
- `cluster/docs/field-report-template.md`: promotion template, not the entrypoint.

## What To Ignore Most Of The Time
- `cluster/published/current/structured/`: published structured artifacts for the current canonical package.
- `cluster/published/current/figures/`: published figures for the current canonical package.
- `cluster/published/current/raw/`: raw logs when the publication includes them.

That published package is for readers and stakeholder deliverables. It is not the forward runtime contract.
Use `cluster/archive/runs/<run_id>/` for independently retained historical baselines.

## Current Published State

The published field report covers `2026-03-17_localhost_fabric_canonical`, a historical single-GPU localhost canary. The [snapshot manifest](published/current/manifest.json) lists the structured evidence and figures linked by the report and notes. Private identifiers and selected operator output were removed. Of 180 original inventory entries, 25 are in this snapshot, 95 from the historical publication are outside its reader scope, and 60 raw logs were unavailable. Four additional linked files bring the snapshot to 29. Five retained files already differed from the source-run hashes in the historical publication, including the vLLM and GEMM CSVs. Operator activity logs stay under the run's `progress/activity/` directory.

## Promotion Rules
When a run should become the new published package:
1. Run `common-eval` or the raw suite and inspect `cluster/runs/<run_id>/`.
2. Promote the localhost package into `cluster/published/current/` explicitly:

```bash
python -m cli.aisp cluster promote-run \
  --run-id <run_id> \
  --label localhost \
  --allow-run-id <historical_baseline_if_any> \
  --cleanup
```

Promotion checks the candidate manifest, suite records, and available artifact hashes. It renders
and validates the report against the staged package before installing it. With `--cleanup`,
it applies a checked cleanup plan after the publication succeeds.
If promotion is interrupted, run `python3 cluster/scripts/promote_run.py --recover` from `code/` to restore an unfinished publication.

3. If this run also becomes the stakeholder report package, sync:
   - `cluster/field-report.md`
   - `cluster/field-report-notes.md`
   - `cluster/docs/field-report-template.md`
   - `cluster/docs/advanced-runbook.md`
4. Re-run `cluster/scripts/validate_field_report_requirements.sh`.

## Multi-Node
Before the first real multi-node run, use the readiness-only preset:

```bash
python -m cli.aisp cluster common-eval \
  --preset multinode-readiness \
  --run-id <run_id> \
  --hosts <h1,h2> \
  --labels <l1,l2> \
  --ssh-user <user> \
  --ssh-key <key> \
  --extra-arg --oob-if \
  --extra-arg <iface> \
  --extra-arg --socket-ifname \
  --extra-arg <iface>
```

That writes:
- `cluster/runs/<run_id>/structured/<run_id>_multinode_readiness.json`

Use `modern-llm` only after readiness is green.
Use `fabric-systems` or `cluster fabric-eval` when the primary goal is NVLink, InfiniBand, or Spectrum-X characterization rather than generic system bring-up.

## Canonical 2-Node Inference Surface
Use this when the question is specifically "is the core 2-node inference lane healthy, comparable, and ready for baseline-vs-optimization work?"

- Contract doc: `cluster/docs/canonical_2node_inference_surface.md`
- Workload spec example: `templates/canonical_2node_inference_workload_spec.yaml`
- BenchmarkRun example: `templates/canonical_2node_inference_benchmark_run.yaml`

The canonical flow intentionally reuses the existing surfaces instead of adding a new launcher:

```bash
python -m cli.aisp cluster common-eval \
  --preset multinode-readiness \
  --run-id <run_id> \
  --hosts <leader,worker> \
  --labels <leader_label,worker_label> \
  --ssh-user <user> \
  --oob-if <iface> \
  --socket-ifname <iface>
```

Then, after readiness is green:

```bash
python -m cli.aisp cluster common-eval \
  --preset modern-llm \
  --run-id <run_id> \
  --hosts <leader,worker> \
  --labels <leader_label,worker_label> \
  --ssh-user <user> \
  --ssh-key <path> \
  --oob-if <iface> \
  --socket-ifname <iface> \
  --primary-label <leader_label> \
  --extra-arg --gpu-hourly-cost-usd \
  --extra-arg <usd_per_gpu_hour> \
  --extra-arg --vllm-multinode-image \
  --extra-arg <repo@sha256:...>
```

## Optional Export
`python -m cli.aisp cluster build-canonical-package ...` is optional. Most users can ignore it. It only exists to export already-produced runs into a clean shareable bundle after collection/promotion are already done.
