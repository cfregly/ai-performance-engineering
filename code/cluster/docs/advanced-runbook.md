# Cluster Evaluation Harness: Advanced Runbook

This is the operator reference.
If you just want to run or read benchmarks, start with `cluster/README.md`.

## Canonical Contract
There are only two artifact surfaces that matter:
- `cluster/runs/<run_id>/`: the real run package for new work.
- `cluster/field-report*.md` plus `cluster/published/current/structured/` and `cluster/published/current/figures/`: the published canonical package.

Current published package:
- Canonical localhost package: `2026-03-17_localhost_fabric_canonical`
- Historical localhost baseline mentioned in earlier reports: `2026-03-08_localhost_modern_profile_r32_full20b`

Interpretation:
- `2026-03-17_localhost_fabric_canonical` is what the report package points at.
- `2026-03-08_localhost_modern_profile_r32_full20b` requires separately retained evidence for historical comparisons. The snapshot in this checkout covers the March 17 run.

## 1) Routine Single-Node Evaluation
Use this for the usual "tell me how this system behaves" ask:

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

What you get in `cluster/runs/<run_id>/`:
- `structured/`: JSON/CSV summaries
- `raw/`: logs
- `figures/`: plots
- `reports/`: run-local report package when rendering is enabled
- `manifest.json`: machine-readable inventory

## 2) Canonical Full Localhost Package
Use this when you want the full published-quality localhost package rather than the faster answer bundle:

```bash
python -m cli.aisp cluster common-eval \
  --preset modern-llm \
  --run-id <run_id> \
  --hosts localhost \
  --labels localhost \
  --ssh-user "$(id -un)" \
  --primary-label localhost
```

This includes the advanced signals that `common-answer-fast` intentionally skips:
- `allreduce_stability`
- `allreduce_latency_comp`
- `allgather_control_plane`
- `nccl_alltoall`
- `nccl_algo_comparison`
- `train_step_workload`

GPU benchmark validity rules:
- clock locking is mandatory
- missing clock-lock metadata invalidates the run
- profiling and structured artifacts must be non-empty and semantically valid, not just present

## 2b) Canonical Fabric Package
Use this when the cluster question is specifically about NVLink, InfiniBand, or Spectrum-X / RoCE:

```bash
python -m cli.aisp cluster fabric-eval \
  --run-id <run_id> \
  --hosts <h1,h2> \
  --labels <l1,l2> \
  --ssh-user <user> \
  --ssh-key <key> \
  --primary-label <l1>
```

Default behavior is capability-aware partial completion. For fabric-only bring-up, the entrypoint records `not_present` and `not_configured` states structurally without inheriting the broader strict canonical completeness gate.

Optional publish-grade gate:

```bash
python -m cli.aisp cluster fabric-eval \
  --run-id <run_id> \
  --hosts <h1,h2> \
  --labels <l1,l2> \
  --ssh-user <user> \
  --ssh-key <key> \
  --primary-label <l1> \
  --nmx-url https://<your-nmx-host> \
  --require-management-plane
```

Expected fabric outputs:
- `<run_id>_fabric_command_catalog.json`
- `<run_id>_fabric_capability_matrix.json`
- `<run_id>_fabric_verification.json`
- `<run_id>_fabric_ai_correlation.json`
- `<run_id>_fabric_scorecard.json`
- `<run_id>_fabric_scorecard.md`

Management-plane inputs:
- `--nmx-url https://<your-nmx-host>`
- `--nmx-token <token>`
- `--ib-mgmt-host <host>`
- `--ib-mgmt-user <user>`
- `--ib-mgmt-ssh-key <path>`
- `--cumulus-hosts <host1,host2>`
- `--cumulus-user <user>`
- `--cumulus-ssh-key <path>`

If those endpoints are missing, the run records `not_configured` instead of silently dropping the control plane.

## 3) Multi-Node Readiness Before First Workload Launch
Do this before the first real multi-node run:

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

Expected artifact:
- `cluster/runs/<run_id>/structured/<run_id>_multinode_readiness.json`

For the current core serving lane, pair that readiness gate with the canonical 2-node inference contract in `cluster/docs/canonical_2node_inference_surface.md`. That document freezes the current vLLM-based 2-node surface, required artifact bundle, and the exact handoff commands used to unblock downstream optimization work.

Fail fast conditions:
- missing labels
- empty hosts/labels
- duplicate labels
- missing interface binding for multi-node
- modern multi-node path requested without the required workload toggles

## 4) Promote, Clean, Validate
Promotion is a separate act from merely running benchmarks.

### 4a) Promote a Run
A run becomes canonical only when you intentionally sync the published package:
- `cluster/field-report.md`
- `cluster/field-report-notes.md`
- `cluster/field-report-localhost.md`
- `cluster/field-report-localhost-notes.md`
- `cluster/docs/field-report-template.md`
- `cluster/docs/advanced-runbook.md`

### 4b) Clean Superseded Flat Artifacts
After promotion:

```bash
cluster/scripts/cleanup_run_artifacts.sh \
  --canonical-run-id <canonical_run_id> \
  --allow-run-id <baseline_run_id> \
  --plan-file <unused_path>
```

Review the cleanup plan, then rerun the same command with `--apply` to execute it. Cleanup refuses candidates that contain symlinks or hold evidence linked by the published reports. Use `--allow-run-id` to retain a linked comparison run.

Use additional `--allow-run-id` entries only for intentionally retained historical baselines or verification runs.

### 4c) Validate the Published Package
```bash
cluster/scripts/validate_field_report_requirements.sh \
  --report cluster/field-report.md \
  --notes cluster/field-report-notes.md \
  --template cluster/docs/field-report-template.md \
  --runbook cluster/docs/advanced-runbook.md \
  --canonical-run-id <canonical_run_id>
```

For a localhost package, also validate:

```bash
cluster/scripts/validate_field_report_requirements.sh \
  --report cluster/field-report-localhost.md \
  --notes cluster/field-report-localhost-notes.md \
  --canonical-run-id <canonical_run_id>
```

### 4d) Understand Scorecard vs Run Success
These are different:
- Suite success means the run executed correctly.
- Canonical completeness means the run met the full coverage policy.

Example:
- A `common-answer-fast` run can be fully green operationally.
- It can still fail canonical completeness because `common-answer-fast` omits advanced signals that `modern-llm` includes.

### 4e) Required Current Outputs
For a useful current run, expect at minimum:
- `*_node1_nccl.json`
- `*_vllm_serve_sweep.csv`
- `*_vllm_serve_request_rate_sweep.csv`
- `*_gemm_gpu_sanity.csv`
- `*_gpu_stream.json`
- `*_fio.json`
- `*_nvbandwidth.json`
- `*_cluster_scorecard.json`
- `*_benchmark_coverage_analysis.json`
- `*_mlperf_alignment.json`

### 4f) Quick Friction Checks (Required For Canonical Runs)
Canonical published runs must include quick friction evidence.

Required artifact family:
- `*_quick_friction.json`

Expected use:
- classify install, import, and operational friction before treating a package as publishable
- surface friction in the report, notes, and operator_checks_dashboard outputs

### 4g) Monitoring Expectations Snapshot (Required For Canonical Runs)
Canonical published runs must include monitoring expectations evidence.

Required artifact family:
- `*_monitoring_expectations.json`
- `*_operator_checks_dashboard.json`
- `*_operator_checks_dashboard.png`

Expected use:
- prove the node exposes the observability surface the operator story claims
- keep the report grounded in operator_checks_dashboard outputs rather than anecdote alone

## 5) What Not To Do
- Do not consume new runs by scanning `cluster/published/current/structured/` first.
- Do not leave superseded flat artifacts on disk after promotion.
- Do not treat a green fast preset as equivalent to a full canonical package.
- Do not accept clock-unlocked GPU benchmarks.
- Do not keep adding side docs for one-off cleanup or hardening notes; fold them into this runbook or `cluster/README.md`.
