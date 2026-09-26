# Fabric Diagnostics Guide

`cluster.fabric.diagnostics` collects read-only RoCE and InfiniBand evidence before and after a bounded interval. It parses cumulative counters, calculates deltas and rates, and keeps every command response as raw evidence.

The tool does not change switch configuration or reset counters. Its results are diagnostic and noncanonical.

## RoCE Snapshot

Provide every switch and interface that carries the workload traffic.

```bash
python -m cluster.fabric.diagnostics \
  --run-id 2026-09-26_fabric_diag \
  --run-dir cluster/runs/2026-09-26_fabric_diag \
  --family roce \
  --cumulus-hosts <leaf01>,<leaf02> \
  --switch-interfaces swp1,swp2 \
  --cumulus-user <user> \
  --cumulus-ssh-key <key> \
  --interval-seconds 10
```

The snapshot includes RoCE QoS, adaptive routing, BGP summary, BGP routes, and interface counters. It tries the current `nv show interface <id> counters qos` and egress queue forms, plus the older `qos roce counters` form. A command that is absent on the target software release is recorded as `unsupported`. A failed command is recorded as `error`. Aligned priority group, traffic class, receive, and transmit tables receive qualified counter names so rows cannot collapse.

Remote collection requires an existing known-host entry because SSH uses `StrictHostKeyChecking=yes`. It preserves the user's SSH agent configuration. Every remote command runs under the target's `timeout` utility. If that utility is unavailable, the command is `unsupported` and the collector does not start it.

## InfiniBand Snapshot

Counter endpoints use `[<hca>@]<lid>:<port>`. The HCA name is optional metadata. `perfquery` still targets the LID and port.

```bash
python -m cluster.fabric.diagnostics \
  --run-id 2026-09-26_ib_diag \
  --run-dir cluster/runs/2026-09-26_ib_diag \
  --family infiniband \
  --ib-mgmt-host <ufm-or-mgmt-host> \
  --ib-mgmt-user <user> \
  --ib-mgmt-ssh-key <key> \
  --ib-endpoints mlx5_0@11:1,mlx5_1@12:1 \
  --ib-route-lids 11,12 \
  --ib-switch-lids 21,22 \
  --interval-seconds 10
```

The tool reads `ibstat`, `saquery`, `ibdiagnet -r`, `perfquery -x`, `ibtracert`, and the requested `ibroute` tables. It never uses `perfquery -R` because resetting counters would destroy evidence and change management state.

## Bound a Workload

An explicit workload command replaces the idle interval. The command runs locally through the same bounded command runner. Its stdout, stderr, return code, duration, and stable workload ID are retained.

```bash
python -m cluster.fabric.diagnostics \
  --run-id 2026-09-26_fabric_workload \
  --run-dir cluster/runs/2026-09-26_fabric_workload \
  --family roce \
  --cumulus-hosts <leaf01>,<leaf02> \
  --switch-interfaces swp1,swp2 \
  --workload-id <serving-profile-id> \
  --workload-command '<bounded workload command>' \
  --timeout-seconds 180
```

Only pass a command whose effects and target scope you have already reviewed. The fabric collector remains read-only, but the supplied command has the behavior of that command.

Use `--workload-id` to match the workload ID in serving, application, or transport signal exports. The supplied ID takes precedence over the fallback command hash. It can also tag a timed interval that has no `--workload-command`, including a workload launched by another orchestrator. The tag is a declared identity and does not prove that the selected workload used every sampled fabric path.

Local commands run in a new process group. On timeout, the runner sends TERM and then KILL only to that owned group. It does not use a global process-name kill. Raw evidence records the timeout and termination steps. If an SSH transport timeout prevents confirmation of remote cancellation, the result records that cancellation was not verified and remains an error.

## Analyze Retained Snapshots

Use retained mode when snapshots already exist. No switch or host command runs.

```bash
python -m cluster.fabric.diagnostics \
  --run-id 2026-09-26_retained \
  --run-dir cluster/runs/2026-09-26_retained \
  --before <before-snapshot.json> \
  --after <after-snapshot.json>
```

The after timestamp must be later than the before timestamp. A lower cumulative value is marked `reset_or_wrap`. The tool does not guess a wrapped delta because counter width and reset provenance may be unknown. A zero delta is valid measured evidence and remains distinct from an unavailable counter.

Both source snapshots must carry the same source run ID. The retained output uses the requested run ID and records the source IDs so its file name and structured payload agree.

## Output Contract

| Path | Contents |
| --- | --- |
| `raw/<run_id>_fabric_diagnostics_<phase>_<index>_<family>_<command>.json` | Exact command, host, stdout, stderr, return code, timing, and parsed fields |
| `raw/<run_id>_fabric_diagnostics_workload.json` | Optional workload command evidence |
| `structured/<run_id>_fabric_diagnostics_before.json` | Before snapshot with normalized counters |
| `structured/<run_id>_fabric_diagnostics_after.json` | After snapshot with normalized counters |
| `structured/<run_id>_fabric_counter_deltas.json` | Matched counter deltas, rates, timestamps, reset states, and normalized signals |
| `structured/<run_id>_fabric_diagnostics.json` | Complete diagnostic result and artifact references |
| `structured/<run_id>_fabric_diagnostics_manifest.json` | File sizes and SHA-256 values for this diagnostic package |
| `reports/<run_id>_fabric_diagnostics.md` | Human-readable evidence table |

The tool also merges a noncanonical reference under `tools.fabric_diagnostics` in the run directory's `manifest.json`. Existing manifest keys are preserved. The merge uses the shared `.diagnostic-bundle.lock` and an atomic replacement so concurrent diagnostic writers do not lose each other's entries.

The normalized `signals` array contains `pfc`, `ecn`, `link_error`, and `throughput` records when matching parsed fields exist. Cumulative counters export `role=counter`, `semantics=delta`, and the interval delta in the original unit. Their rate remains available as `rate_per_second` with a separate rate unit. Gauges export `role=measurement` and never claim delta semantics. Every signal includes its metric, value, unit, per-field start and end Unix timestamps, collector clock domain, scope, and raw evidence references. The clock domain contains a hash of the collector kernel hostname so unrelated collector clocks cannot be treated as one timeline. Retained snapshots must have the same clock domain. A local target uses the same hashed kernel hostname in `scope.host`. A remote target uses `scope.endpoint_alias_hash`. The separate `scope_identity.kernel_hostname_verified` field is false because an endpoint alias does not prove the remote kernel hostname. Workload-tagged signals also include `scope.workload_id`.

When at least two selected ports have comparable byte or packet counter rates, `path_balance` reports each sampled rate and the observed range. Unequal rates describe traffic distribution. They do not prove an ECMP collision or identify its cause. Duplicate normalized counter names are rejected because the parser cannot safely infer a missing priority hierarchy.

The CLI exits 0 for usable `ok` evidence and for `partial` evidence whose remaining gaps are unsupported. It exits 2 when no measurement is supported and 1 for invalid data, any command error, or a failed workload. Missing management access and unsupported commands remain visible in the structured result.
