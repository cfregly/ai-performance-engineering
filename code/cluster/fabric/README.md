# Fabric Evaluation Handbook

This directory is the maintained fabric surface for the repo.

The archived NVIDIA course exports under `nvidia-advanced-networking-for-ai-infra` stay as source material only. The operator-facing entrypoints are:

- [nvlink.md](nvlink.md)
- [infiniband.md](infiniband.md)
- [spectrum_x.md](spectrum_x.md)
- [cross_fabric.md](cross_fabric.md)
- [fabric_command_catalog.json](fabric_command_catalog.json)

## Entry Points

Use the dedicated fabric path when the question is "is the fabric healthy, correctly configured, and fast enough for AI workloads?":

```bash
python -m cli.aisp cluster fabric-eval \
  --run-id <run_id> \
  --hosts localhost \
  --labels localhost \
  --ssh-user "$(id -un)" \
  --primary-label localhost \
  --nmx-url https://<your-nmx-host>
```

This path defaults to capability-aware partial completion. It keeps `not_present`, `not_configured`, and `unavailable` signals visible in structured outputs instead of failing on broader publish-grade completeness gates.

Use the lab-only NMX helper when the question is "how would I carve this NVLink domain into Alpha/Beta partitions without hand-building the commands?":

```bash
python -m cli.aisp cluster nmx-partition-lab \
  --nmx-url https://<your-nmx-host> \
  --alpha-name AlphaPartition \
  --beta-name BetaPartition
```

This helper is inventory-driven and read-only. It returns the suggested Alpha/Beta seed locations, the borrow/rebalance flow, and the exact `curl` commands for create/update/delete/poll/verify, but does not execute any mutating NMX partition calls.

Use the preset when staying inside the common-eval surface:

```bash
python -m cli.aisp cluster common-eval \
  --preset fabric-systems \
  --run-id <run_id> \
  --hosts <h1,h2> \
  --labels <l1,l2> \
  --ssh-user <user> \
  --ssh-key <key> \
  --nmx-url https://<your-nmx-host>
```

## Structured Outputs

Fabric runs emit these artifacts under `cluster/runs/<run_id>/structured/`:

- `<run_id>_fabric_command_catalog.json`
- `<run_id>_fabric_capability_matrix.json`
- `<run_id>_fabric_verification.json`
- `<run_id>_fabric_ai_correlation.json`
- `<run_id>_fabric_scorecard.json`
- `<run_id>_fabric_scorecard.md`

These artifacts are also folded into:

- `manifest.json`
- `<run_id>_cluster_scorecard.json`
- `reports/field-report-localhost*.md` when localhost report rendering is enabled

For NMX-backed NVLink domains, the verification payload now answers the operator scenarios directly:

- topology and capacity-planning counts for compute nodes, GPUs, switch ASICs, switch trays, and ports
- chassis counts and chassis serial-number coverage
- GPU-to-node mapping and candidate Alpha/Beta 4-GPU allocations
- sample GPU, switch, and chassis objects with `DeviceID`, `DomainUUID`, health, port count, and `LocationInfo`
- partition inventory, default-partition membership, and unassigned capacity
- telemetry endpoint coverage for switch temperature, throughput, physical errors, and cable diagnostics

## Management-Plane Inputs

The fabric evaluator is capability-aware. If management endpoints are missing, the run records `not_configured` instead of silently skipping.

Preferred CLI arguments:

- `--nmx-url https://<your-nmx-host>`
- `--nmx-token <token>`
- `--ib-mgmt-host <host>`
- `--ib-mgmt-user <user>`
- `--ib-mgmt-ssh-key <path>`
- `--cumulus-hosts <host1,host2>`
- `--cumulus-user <user>`
- `--cumulus-ssh-key <path>`
- `--require-management-plane`

Use `--require-management-plane` for publish-grade runs when missing management access should fail the fabric step instead of downgrading completeness.

## Design Rules

- Canonical evaluation is read-only.
- Mutating lab commands remain documented in the catalog as `lab_only=true`.
- NVLink, InfiniBand, and Spectrum-X must be interpreted alongside NCCL, all-to-all, train-step, vLLM, and nvbandwidth artifacts.
- The run directory remains the only contract surface that matters.
