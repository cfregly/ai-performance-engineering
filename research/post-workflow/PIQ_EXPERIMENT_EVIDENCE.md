# PIQ current diagnostic evidence

This public source snapshot contains validators, schemas, an empty fixed
admissions anchor, and synthetic fixtures only. It contains no real execution,
admission, host, device, account, package, qualification, or publication data.

PIQ diagnostic envelope version 2 is the only active PIQ experiment-evidence
contract in this repository. It has three closed experiment-kind variants:
`checkpoint_write_batching_stage0`, `cpu_two_host_functional`, and
`te_mxfp8_gpu_training`. Only the TE variant can bind measurement receipt v3.
The other two remain diagnostic and cannot establish package readiness. PIQ
envelope v1, its separate supplements, reporters, importers, and scaffolds are
retired and are not accepted.

This is a pre-GA current-contract-only boundary. There is no compatibility
promise, migration adapter, automatic conversion, or evidence backfill for an
older envelope. Retained packages that still reference an older contract need
an explicit deletion or new-execution decision. Changing a version number does
not make old evidence valid.

The validator and importer are source-only tools. They do not contact PIQ,
launch a workload, reserve a host, allocate a GPU, qualify a scientific result,
create a post package, or authorize publication.

## Commands

```sh
python3 tools/piq_gpu_training_evidence.py check /absolute/path/to/v2-bundle
python3 tools/piq_gpu_training_evidence.py import \
  /absolute/path/to/v2-bundle /new/local/destination
python3 -m unittest -v tools.test_piq_gpu_training_evidence
```

The check command accepts only a real directory. For example, the TE variant's
`envelope.json` declares:

```json
{
  "schema_version": 2,
  "kind": "piq_experiment_evidence",
  "scope": "diagnostic",
  "scientific_disposition": "DIAGNOSTIC_ONLY",
  "experiment_kind": "te_mxfp8_gpu_training",
  "workload_family": "gpu_training"
}
```

`experiment_kind` and `workload_family` are an exact closed pair. Unknown
kinds, cross-kind fields, and mixed censuses fail rather than falling through
to a compatibility branch.

The envelope cannot claim qualified scope or a qualified scientific
disposition. A valid diagnostic export still does not establish qualification
or package readiness.

## Stage 0 variant

`checkpoint_write_batching_stage0` binds exactly one run with two ordered
attempts: baseline then candidate. Both attempts must succeed and be accepted.
Their raw identities, input hash, and output hash bind one exact independent
oracle. Both output hashes must equal the oracle output. The variant also binds
terminal cleanup plus immutable custody, readback, and export documents for the
same run and attempt census, all verified after the objective ends. It admits
no detached object and no qualification receipt.

## CPU two-host functional variant

`cpu_two_host_functional` binds one accepted attempt with exactly ranks 0 and
1 on two distinct physical-host hashes and two distinct boot-identity hashes.
Each host has one local rank and one positive worker PID, and each indexed rank
output repeats the same host, boot, PID, run, attempt, and rank identity.

The partition/rehash census must conserve every input record through two
nonempty partitions, rehash, correctness, and output. The TCP evidence must
bind one worker PID directly to the other worker PID, their distinct physical
host identities, and a positive byte count. Terminal cleanup must name both
worker PIDs, confirm both workers terminated, and confirm transport closed.
Custody, readback, and export bind the same run and attempt and are verified
after the objective.

A named owner and authority attest the exact two-host/rank/PID tuple before the
objective. A separate current admission binds the owner-attestation hash and
same study/run/attempt, is not revoked, and covers the complete objective
window. Neither bundle document authenticates itself. The exact study, owner,
authority, envelope, attestation, admission, run, and attempt tuple must also
appear once in the repository-owned
`trust/piq-v2-two-host-admissions.json` after an independent source review.
That review must follow owner attestation and objective completion. The fixed
anchor is empty until a real callback is reviewed. There is no package,
environment, or command-line override. This is an execution-time admission
rule plus an out-of-band source-review trust boundary, not a wall-clock
freshness guess during later validation. Even a valid row remains
diagnostic-only and has no qualification receipt or report.

## TE GPU-training variant and fixed census

One envelope binds the prospective design and exactly five executions in this
order:

```text
admission-01: BF16  -> MXFP8
pair-01:      BF16  -> MXFP8
pair-02:      MXFP8 -> BF16
pair-03:      MXFP8 -> BF16
pair-04:      BF16  -> MXFP8
```

Each execution has one distinct PIQ run ID and exactly two ordered attempts.
The complete envelope therefore has ten attempts. Failed and unaccepted
attempts remain in place with their reason. They cannot be omitted, replaced,
retried under the same census, or reordered. Admission is diagnostic and never
enters the four accepted scientific pairs.

Every attempt binds the immutable arm configuration and has exactly two CUDA
rank rows on one observed host. Ranks and local ranks are 0 and 1, with distinct
positive PIDs, distinct admitted device UUIDs, exact indexed output references,
and the expected arm-configuration hash.

## Assignment and identity

Each execution binds requested and observed one-node, two-accelerator,
two-process resources. Its assignment evidence records the scheduler job,
owner, node, session or cgroup, device UUIDs, source, runtime, executable, plan,
and ordered UTC window. Every raw attempt has an ordered UTC start/end interval
contained by that group assignment window. The execution, both raw attempts,
and rank rows must agree with that immutable assignment.

The envelope binds:

- the fixed study ID and complete registered design,
- the resolved first-party source/import closure and exact hashes,
- the data revision, train, validation, and sampler identities,
- the initial detached checkpoint identity,
- PIQ, runtime, hardware, and complete stack-layer identities, and
- the complete-job objective window containing every assignment and attempt.

A guessed module, same-named request file, preloaded module, missing import, or
identity drift fails closed.

## Workload evidence and detached objects

The required workload roles are execution, quality, checkpoint/restart,
telemetry summary, telemetry raw-custody index, cleanup, readback, and export.
The declared stack observes host, accelerator, fabric, storage, orchestration,
runtime, and workload layers.

Large checkpoints and raw telemetry remain detached from Git. Each detached
object still binds its object ID, role, SHA-256, byte count, media type,
immutable custody record, local durable readback, PIQ readback, export readback,
and retained availability. Each readback is verified after the objective ends
and must match the same complete object census. A locator or digest alone is
not custody.

## Strict artifact boundary

The artifact index is closed and byte-exact. Every indexed file must exist with
its declared lowercase SHA-256 and size. Every non-envelope file in the bundle
must be indexed. Unsafe or case-colliding paths, symlinks, unexpected files,
changed bytes, duplicate JSON keys, nonfinite JSON, malformed JSONL, empty
records, and unsupported fields fail.

Import validates the source, copies only the declared indexed artifacts plus
the envelope into a new destination, validates the copy, and writes a
nonpublishable diagnostic import receipt. It refuses overwrite, symlinks, an
existing destination, and a destination inside the source. Import never copies
detached payload bytes or upgrades the scientific disposition.

## Qualification and package binding

Only [measurement receipt v3](QUALIFIED_GPU_TRAINING_EVIDENCE.md) can qualify
the `te_mxfp8_gpu_training` variant. A measured package must declare:

```json
{
  "content_state": "measured",
  "measurement_contract": "qualified_gpu_training_v3",
  "measurement_receipt": "evidence/measurement-receipt.json",
  "piq_experiment_evidence": "evidence/piq-study/envelope.json"
}
```

The package gate validates the exact PIQ v2 bundle. For the TE variant it
requires the valid v3 receipt to bind that envelope's path, hash, study
identity, and full run/attempt census. A missing receipt, wrong version, wrong
variant, wrong envelope, omitted attempt, synthetic design, or non-qualified
deterministic report keeps the package blocked. Stage 0 and CPU two-host
variants always retain the explicit diagnostic readiness block. A successful
check or import never substitutes for measurement qualification.

The diagnostic finding binds the same v2 envelope and exact run/attempt order.
For TE it also binds all five run IDs, ten attempt IDs, the exact v3 receipt,
and its qualification report. Stage 0 and CPU two-host findings instead bind
their exact variant evidence roles and must leave qualification receipt and
report null. Private run, host, boot, device, job, PID, transport, cgroup,
owner, admission, assignment, raw telemetry, checkpoint, custody, and review
data do not enter a public projection.

See [QUALIFIED_GPU_TRAINING_EVIDENCE.md](QUALIFIED_GPU_TRAINING_EVIDENCE.md)
for the deterministic evaluator, private package boundary, and public-safe
summary boundary.
