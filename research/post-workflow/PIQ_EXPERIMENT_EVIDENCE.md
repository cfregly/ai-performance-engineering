# Import PIQ diagnostic experiments

This public projection checks retained evidence bytes only. It contains no
real execution callback or admission. The production admissions anchor is
empty, and synthetic fixtures provide no execution, physical-host, qualification,
package-readiness or publication proof.

PIQ exports can retain workload evidence for private analysis without inventing
GPU metadata for CPU work. Version 1 is a diagnostic contract, with a shared
envelope and small workload adapters. It does not qualify a performance result,
create a measurement receipt, or make a post package ready.

```sh
python3 tools/piq_experiment_evidence.py check /absolute/path/to/export
python3 tools/piq_experiment_evidence.py import /absolute/path/to/export internal-experiments/experiment-id
python3 -m unittest -v tools.test_piq_experiment_evidence tools.test_check_package
```

Imports preserve the envelope and indexed artifact bytes. They refuse an existing
destination, copy only declared evidence, verify the copy, and retain an
`import-receipt.json` with `publishable: false`, the envelope hash, experiment ID
and import time. No integration, network access or third-party library is needed.
Keep real private evidence outside public projections. The importer never creates
articles, images or social drafts.

## Common contract

[The JSON schema](schemas/piq-experiment-evidence-v1.schema.json) defines the
versioned structure; [the example](examples/piq-diagnostic/envelope.json) is a
complete **synthetic fixture**. Its attempts, outputs, identities and times do
not describe a PIQ run. Use the Python checker for file, identity and cross-field
checks that JSON Schema cannot perform.

Every bundle has `envelope.json`, `schema_version: 1`,
`kind: "piq_experiment_evidence"`, `scope: "diagnostic"`, and a nonempty
`workload_family`. It contains:

- `experiment`: exact ID, question, hypothesis, workload scope, hashed immutable
  design and source files. The design is retained before execution.
- `stack`: full PIQ Git revision, exact runtime and hardware identifiers, and
  named layer identifiers. Use PIQ layer names `host`, `accelerator`, `fabric`,
  `storage`, `orchestration`, `runtime`, `workload`; other sublayers may be added.
  All families require `host`, `runtime`, `workload`. Omit nonapplicable layers.
- `objective`: metric, unit, minimize/maximize direction, and actual UTC start/end
  window. Numbers remain in the native raw-result payload; import does not
  independently reproduce or scientifically judge them.
- `attempts`: distinct attempt IDs and real run IDs, baseline/candidate roles, succeeded/failed
  status, explicit accepted boolean, hashed raw JSON and failure reason (null
  for success). Several native attempts may share one real run ID; do not invent
  a separate PIQ run per arm. Retain failed attempts; accepted means usable diagnostic data.
- `comparison`: every baseline/candidate attempt ID in envelope order, scientific
  disposition (`DIAGNOSTIC_ONLY`, `INCONCLUSIVE`, or `REJECTED`) and reason.
- `correctness`: explicit boolean and hashed retained JSON. A failed correctness
  result cannot coexist with accepted attempts.
- `workload_evidence`: declared role-to-hashed-file map. Every family requires
  input/output. `cpu_data_query` also requires `query_plan` and a `storage` layer.
  Register future extra roles in `FAMILY_REQUIREMENTS`; unknown families can
  import the common diagnostic contract, with qualification unsupported.
- `artifacts`: the complete path/hash index of retained evidence selected for
  import. Every common-contract reference must appear with its exact hash.
  Additional declared native outputs, API confirmations, custody records and
  readbacks are allowed; each receives the same path, byte-hash and symlink
  validation and is preserved by import. Unindexed files are not imported.

All artifact paths are nonempty POSIX paths within the bundle. No absolute paths,
traversal, symlinks, empty files, duplicate artifact paths or conflicting hashes
are accepted. The envelope and importer receipt are not indexed artifacts.

## Bind the plan and run identities

The hashed design JSON must contain values equal to the envelope:

```json
{
  "experiment_id": "same-as-experiment.id",
  "workload_family": "cpu_data_query",
  "source_artifacts": [{"path": "source/query.sql", "sha256": "..."}],
  "input": {"path": "evidence/input.db", "sha256": "..."},
  "objective": {"metric": "query_elapsed", "unit": "ms", "direction": "minimize"},
  "required_evidence_roles": ["input", "output", "query_plan"]
}
```

`required_evidence_roles` is the sorted, exact role list. Output hashes are
retained after execution; the design binds their planned roles. Keep all other
fixed workload parameters and acceptance criteria in this immutable design.
Each raw-result JSON must contain matching `experiment_id`, `attempt_id`,
`run_id`, `role`, `design_sha256` and `input_sha256`, alongside the original
results. Correctness/output JSON must bind `experiment_id`; correctness must
also match `accepted`. Output `run_ids` must equal succeeded run IDs in first-occurrence
order, deduplicated when native attempts share a run. CPU query-plan JSON also binds `experiment_id` and may retain
per-run EXPLAIN QUERY PLAN rows. Input may be a database, CSV or other immutable
source file. Additional workload roles retain their native opaque file formats.

An optional design `input_artifacts` list binds 1–16 additional opaque input
files as `{path, sha256}` references. Paths use `evidence/inputs/<filename>`
with portable filenames; duplicates, case collisions, reserved device names
and collisions with other evidence are refused. Each input, including the main
input, must be nonempty and at most 8 MiB; their combined size is at most 64 MiB.
Every additional file must appear in the envelope artifact index with its exact
hash and in the indexed `evidence/stack-identity.json` workload's identical
`input_artifacts` list. When the original `executable-manifest.json` is present,
it must be indexed and its `files` must bind the complete additional input list,
main input, design and stack identity. Its workload identity hash must match the
retained workload object. Omitted parts and changed bytes fail validation.
The importer preserves these files without interpreting a main input index,
archive format or adapter algorithm. Historical bundles without the optional
field keep their existing shape.

These checks establish internal consistency and detect changed bytes. They
cannot prove when an external producer created its plan, that a claimed runtime
executed, or that no omitted attempt exists. Preserve PIQ's authoritative run
records and pre-execution plan timestamps for those questions.

## Formal correctness evidence

New finite-workload exports keep the same v1 envelope. Their retained
`correctness` JSON adds this check, alongside `experiment_id` and `accepted`:

```json
{
  "check": {
    "configuration": {"oracle": "json-numeric-tolerance", "atol": "0.01", "rtol": "0"},
    "reference": {"path": "oracle/reference.json", "sha256": "..."},
    "candidate": {"path": "oracle/candidate.json", "sha256": "..."},
    "report": {"path": "oracle/report.json", "sha256": "..."},
    "verification": "producer-asserted"
  }
}
```

The immutable design must declare the identical `correctness` configuration.
The envelope's `oracle_reference`, `oracle_candidate` and `correctness_report`
workload roles must exactly match those three references. Their files receive
the ordinary path, hash, index and import-preservation checks.

Supported configuration metadata follows PIQ's formal oracle contract:

- `output-sha256-equality`: only `oracle`.
- `json-numeric-tolerance`: `oracle`, `atol` and `rtol`. Both tolerances are
  nonnegative plain decimal strings of at most 64 characters; nonfinite values,
  exponents, leading zeroes, numeric JSON values and extra fields are rejected.
- `json-keyed-record-set-equality`: `oracle` and a `recordKey` string of
  1–128 characters.

The report must have schema `piq.finite-correctness-report.v1`, the same oracle,
`status` (`pass` or `fail`), a bounded lowercase `reasonCode`, a nonnegative
integer `comparedValues`, and exact `configurationSha256`, `referenceSha256`
and `candidateSha256` bindings. Its status must agree with both retained and
envelope `accepted` values. The configuration hash uses
`piq_verify.canonical.canonical_sha256`: sorted compact JSON encoded as UTF-8,
with non-ASCII characters retained. This differs from the older arm
configuration hash domain, which escapes non-ASCII characters.

The importer validates metadata and retained bytes. It does not implement or
rerun numerical tolerance or keyed-record comparisons, independently certify a
producer's decision, or accept a forged `server-verified` label. PIQ's shared
oracle helper owns the comparison algorithm. If a retained failed attempt
prevents an operand pair/report, the formal check and its three roles may be
absent only with `accepted: false` and no accepted attempts. Historical v1
bundles without formal correctness configuration remain readable.

## Rank, device and resource evidence

Ranked finite exports add an `execution` role pointing to retained JSON with
`experiment_id`, `run_id`, canonical plan `resources`, and `attempts`. The
immutable design binds the same requested `resources` and an `arms` list containing
`arm_id`, `configuration` and `configuration_sha256`. Each execution attempt
binds `attempt_id`, `arm_id`, `expected_process_count` and `rank_census` to the
raw result's original `native_attempt` and `arm_configuration`. The expected
count comes from arm `processCount`, otherwise resource `processCount`,
otherwise one. Arm configuration hashes retain PIQ's ingestion identity JSON
encoding, distinct from the correctness hash encoding described above.

The canonical local CPU runner records a single process directly and omits a
rank census. When no execution role or rank records are declared, an exact
two-arm design may use that shape only with one local process, one node and
zero accelerators. The checker still binds resource bounds, each arm's
configuration and hash, native attempt identity/state, positive child PID,
exit status, indexed output bytes, decimal elapsed time and an ordered UTC
window within the experiment window. It derives no rank or execution artifact.
Malformed resources, multiple processes or nodes, any GPU, and grafted rank
records keep the explicit execution-evidence requirement.

Each census row retains `rank`, `localRank`, `host`, `processId`, `deviceType`
(`cpu` or `cuda`), `deviceUuid`, and an output `{artifactName, sha256}`.
`artifactName` resolves directly within the bundle and must be indexed with its
exact hash. A succeeded attempt requires every rank from zero through the
declared process count minus one, with distinct output artifacts, distinct
process IDs per host, distinct local ranks per host, and no more hosts than the
declared node count. These identities are compared within each attempt;
sequential attempts may reuse a process ID.

CPU rows require a null device UUID. CUDA rows require GPU resources and a
nonempty device UUID; a declared GPU attempt cannot silently substitute CPU
rows. GPU evidence also requires an `accelerator` stack layer. The current
finite GPU shape supports one node, one GPU and one process, with each arm
bounded to 30 seconds. Completed or ranked GPU attempts require a retained
`resources.assignmentEvidence` reference whose `observation` has schema
`piq.finite-slurm-observation.v1`. An unstarted or failed preflight with no
observed ranks may retain its missing assignment without inventing one.

Assignment observations are recorded during execution. They are excluded from
the immutable requested resources in the design and plan hash. Their artifact
must retain the original ingestion identity JSON bytes plus a newline. The
checker binds its original Slurm job ID, owner, node, GPU UUID, process ID,
cgroup memory cap and historical time window to the retained ranks and indexed
`executable-manifest.json`. Its source, input, runtime, executable and plan
hashes, plus requested resources, must match that original manifest. The retained `jobBinding`
projection, its hash and the scheduler's `piq-finite:<hash>` job comment must
agree. The observation's process ID must match an observed native rank; all GPU
ranks must report the same assigned host and device. A serving `leaseContext`
cannot substitute for this Slurm observation.

These are private diagnostic consistency checks. Retained process/device labels
and historical Slurm observations do not independently prove execution or
authorize a new launch. The importer does not check current allocations or
freshness. GPU execution remains refused until an existing owner provides a
supported execution seam with current authority. Import never reserves a
resource or starts a process, GPU workload or service.

### Supplementary two-host functional receipt

`nodeCount` is requested capacity, not a statement that every requested host
participated. Generic envelope validation therefore keeps its upper-bound rule
and does not require the number of reported host labels to equal `nodeCount`.

Manifest schema 2 packages that discuss a functional two-host run must
separately declare `piq_two_host_functional_receipt` at
`evidence/two-host-functional-receipt.json`. The
[version-1 receipt schema](schemas/piq-two-host-functional-evidence-v1.schema.json)
binds a second diagnostic envelope, its exact execution and correctness files,
one run, selected accepted attempts, each rank-output-to-aggregate correctness
projection, and distinct run/envelope-bound readback, export, private-import and
cleanup projections. For
each selected attempt, the checker requires a succeeded, accepted attempt with
an explicit complete two-rank census that reports two distinct nonempty host
labels. Hosts are evaluated within each attempt. Two one-host attempts on
different labels do not pass. Assigned CPU evidence also binds exactly two
declared target IDs and distinct host-identity hashes through
`executor: "assigned-cpu-hosts"`.

This receipt is deliberately `authority: "retained_producer_assertion"` and
`qualification: "not_established"`. It closes a package-consistency hole. It
does not independently prove that two physical machines ran, that the ranks
were concurrent, or that inter-host traffic occurred. Declaring it therefore
blocks package readiness unless the exact
[two-host admitted owner-attested contract](PIQ_TWO_HOST_OWNER_AUTHORITY.md)
also validates. That optional schema-2 declaration is fixed at
`piq_two_host_owner_authority_receipt: "evidence/two-host-owner-authority-receipt.json"`.
It binds producer bytes and the exact envelope/run/attempt census to retained
owner handback, the objective and native attempt windows, declared physical-host
hashes, canonical boot/session/PID identity, TCP-family inter-host transport,
and cleanup after launch, both ranks and transport. The exact tuple
must also appear in the fixed repository admissions anchor after independent
source review. The committed production anchor is empty until a real owner
callback is reviewed. Hashes provide byte integrity, not origin
authentication. Admission is an out-of-band source-review trust boundary, not
a cryptographic signature. Use a separately reviewed signed version 2 if that
property becomes necessary. An admitted receipt may remove only this one
readiness block. The producer stays diagnostic, and neither receipt can replace
or promote the qualified envelope named by `piq_experiment_evidence`.

Check the retained consistency boundary directly with:

```sh
python3 tools/check_piq_two_host_functional.py /absolute/path/to/package
```

The command reports `NOT_DECLARED` when the package has not opted in. With only
the producer receipt, a passing standalone check states that producer-assertion
consistency passed while package readiness remains blocked and exits 0 as the
documented consistency-only API. A declared owner receipt that is invalid or
unadmitted exits nonzero. An admitted owner-attested receipt checks the narrower
retained functional execution/custody assertions against the fixed reviewed
tuple. It does not independently verify physical machines, execution or traffic,
and scientific qualification remains unestablished.

## Qualification and publication stay separate

Version 1 refuses `scope: "qualified"` and `scientific_disposition: "QUALIFIED"`.
The separate [qualified CPU receipt](QUALIFIED_CPU_EVIDENCE.md) enforces its
predeclared baseline, fixed gain, repeatability, correctness and latency criteria
for the CPU study's claimed scope. Its deterministic report is recomputed from
the complete retained raw census and bound to named baseline and scientific
reviews. It does not change a v1 envelope's diagnostic scope. Serving
claims still require end-to-end gains over the established optimized serving
baseline. A diagnostic smoke, speedup-looking sample or unknown workload family
cannot satisfy that gate.

The existing `evidence/measurement-receipt.json` schema version 1 and its GPU
requirements are unchanged. A CPU diagnostic does not fabricate power or
profiler records to fit that schema. A separate qualified CPU receipt contract
uses schema version 2 with `state: "measured"` and `execution_class: "cpu"`.
It is limited to the supported CPU study protocol. It cannot waive GPU v1
requirements or qualify GPU or end-to-end serving results. Import is not a
shortcut around qualification.

If a content manifest declares `piq_experiment_evidence` pointing to
`evidence/<bundle>/envelope.json`, the package checker validates the bundle and
blocks package readiness unless a fully valid CPU v2 measurement receipt binds
that exact current envelope path and hash, its study identity, and its complete
run/attempt/raw-result census. A missing or invalid receipt, wrong envelope,
omitted attempt, or unqualified report keeps the diagnostic block. A successful
import or accepted correctness boolean alone never removes it. All
existing image, review, Typefully, measured-receipt and publication requirements
remain in force. Valid evidence import is not a completed content package.
