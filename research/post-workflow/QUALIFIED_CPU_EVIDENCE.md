# Qualify one prospective CPU codec study

Measurement receipt v2 supports a fixed, prospective, lossless artifact-codec
study. It does not promote an earlier diagnostic or qualify a GPU, serving,
physical-disk, energy, cost, or multi-host result. GPU receipt v1 and PIQ
diagnostic envelope v1 retain their existing meaning. In particular, the old
gzip level-9/level-1 pair remains diagnostic.

Choose and review the design before execution. Retain a primary-source review
explaining why the exact baseline is an established optimized implementation
for flexible lossless artifact storage. Fix the real input corpus, source,
installed toolchain, Python runtime, hardware, affinity, timed boundary, schedule,
criteria, time budget and cleanup before running. Never adjust them after
examining results. The intended study compares installed pigz with `-6 -p 2 -n
-c` against installed zstd with `-3 -T2 --no-progress -q -c`, with their matching
CLI decompression paths. The design, source and review must bind the actual
binaries, versions, libraries and arguments. Neither codec is a correctness
reference for the other; exact original bytes are the oracle.

## Public Python API

```python
from tools.qualified_cpu_study import (
    CRITERIA, TIMED_STEPS, validate_cpu_design, normalize_cpu_sample,
    evaluate_cpu_study, validate_cpu_telemetry, validate_cpu_receipt,
    cpu_receipt_binds_envelope,
)

errors = validate_cpu_design(design)  # Before execution; no measurements read.
sample = normalize_cpu_sample(design, group, envelope, raw, observations,
                              design_sha256=design_file_sha256)
errors = validate_cpu_telemetry(native_bundle_path, envelope, raw, observations)
report = evaluate_cpu_study(design, all_samples_in_declared_order,
                            design_sha256=design_file_sha256)
errors = validate_cpu_receipt(package_path, receipt)
```

Only after the last call returns `[]` may a package gate use
`cpu_receipt_binds_envelope(package_path, receipt, envelope_path)` to test exact
path/hash membership. Membership alone does not qualify evidence. Malformed
receipts fail closed; the evaluator returns `INCONCLUSIVE` for malformed data.
The receipt schema is [measurement-receipt-v2.schema.json](schemas/measurement-receipt-v2.schema.json).
Python additionally verifies retained bytes and cross-file semantics.

## Fixed design

The hashed design JSON has `schema_version: 1`,
`kind: "qualified_cpu_codec_design"`, nonempty `study_id`, `workload_scope`,
`execution_owner`, `runtime_identifier` and `hardware_identifier`, a UTC
`created_at`, `execution_class: "cpu"` and integer `accelerator_count: 0`.
It contains:

- `input`: the main manifest `{path, sha256}`; `input_artifacts`: ordered
  `{path, sha256}` references to every actual corpus part, excluding intent or
  ledger metadata; and `input_bytes`: their exact combined retained byte size.
- `source_artifacts`: ordered `{path, sha256}` references to stable executed
  implementation files. Every group envelope must contain each stable source
  hash exactly once. Per-group source plans may have additional distinct hashes;
  registration separately binds every exact native group design and its sources.
- `python_runtime`: the exact JSON value later retained at
  `observations.runtime.python`; `host_observations`: advance observed values for
  at least `hostname`, `cpu_models` and `cpu_affinity`. Every declared host field
  must match every attempt. Runtime identifiers include the pinned tool versions
  and hashes; an identifier alone cannot replace the observations.
- `arms.baseline` and `arms.candidate`: distinct `id`, plain `implementation`,
  exact `configuration` text, and `native_configuration`. For this producer,
  native configuration includes `codec: "pigz-gzip" | "zstd"`, `toolchain` and
  `telemetryBundle` (the latter two are canonical JSON strings). Toolchain kind is
  `piq.codec-toolchain.v1`, with thread count,
  actual binary paths/hashes/versions and loader/library pins. The raw native arm
  configuration and `observations.runtime.toolchain` must match those pins.
  Both arms bind the identical telemetry reference
  `{artifactName: "inputs/finite-cpu-telemetry.zip", sha256, sizeBytes}`. This
  additional immutable native input contains the canonical observer source
  closure. It is not a corpus part and does not contribute to `input_bytes`.
- `format_policy: "flexible_lossless"`,
  `correctness_oracle: "exact_input_sha256_roundtrip"`,
  `timed_boundary: "complete_fresh_process"`, `timed_steps` exactly equal to
  exported `TIMED_STEPS`, and `criteria` exactly equal to exported `CRITERIA`.
- `groups`, in this exact order: `pair-01`, `control-01`, `pair-02`, `control-02`,
  `pair-03`, `control-03`, `pair-04`, `control-04`. Each item is
  `{id, kind: "pair" | "control", order: ["baseline", "candidate"]}`. Treatment
  pair orders alternate AB, BA, AB, BA. Both roles in every control use the
  baseline configuration; the declared role order is baseline then candidate.

The protocol requires four fresh matched treatment pairs and four neutral A/A
groups: sixteen actual process attempts, with eight actual PIQ group run IDs.
There are no pilots or automatic selective reruns. Each pair must have
baseline/candidate complete-process ratio at least `1.10`, and candidate combined
archive bytes no greater than baseline. Candidate observed maximum process
latency across treatment pairs must not exceed baseline observed maximum.
Every neutral baseline/candidate ratio must be within `[0.95, 1.05]`. All input
parts must round-trip exactly, and no failed, missing or unaccepted attempt is
permitted. Ratios use integer nanoseconds and deterministic decimal formatting.
The observed maximum is not a p95 estimate or a serving-tail result.

The complete parent-runner process window includes process creation/imports,
metadata, input reads/hashes, compression, archive writes and file fsync,
archive readback, decompression, exact verification, evidence serialization and
teardown. `TIMED_STEPS` identifies the required workload operations; it does not
replace the parent's complete-process timing. Do not time only compression or
add independent phase durations to synthesize the primary result.

## Native evidence and exact census

Each native PIQ design preserves normal finite local CPU resources: executor
`local-process`, zero accelerators, one node and one process. Its native arms
and configuration hashes remain unchanged. Add a pre-execution
`qualification_study` object:

```json
{
  "study_id": "the-global-study-id",
  "design_sha256": "the-global-design-file-hash",
  "group_id": "pair-01",
  "group": {"id": "pair-01", "kind": "pair", "order": ["baseline", "candidate"]}
}
```

The adapter specification supplies that object at
`provenanceMetadata.qualification_study`; the exporter preserves it at
`design.qualification_study`. PIQ checks provenance identity without interpreting
the scientific rules. Native `executionOrder` uses its existing
`reference`/`candidate` arm labels; study provenance uses `baseline`/`candidate`
roles. Native execution order and actual process windows must agree with the
study's declared AB/BA sequence.

Every export keeps the original PIQ diagnostic envelope, raw attempts and
artifacts. The CPU validator runs the unchanged PIQ bundle checker. It derives
primary elapsed nanoseconds exactly from `raw.native_attempt.elapsedSeconds`,
PID from `childPid`, window from `startedAt`/`completedAt`, and success from
`state`, `exitCode` and `failureReason`. It uses the raw result's real
`run_id`, `attempt_id` and `role`. It never trusts an analysis-added duration or
an adapter's self-reported PIQ run identity. An optional `raw.study_sample`
projection is ignored.

The adapter's indexed `native-evidence.json` is referenced by
`raw.adapter_result.native_evidence.observations` using `{artifactName, sha256}`,
relative to the directory of `raw.native_attempt.output.artifactName`. It retains:

- `runtime.toolchain`, `runtime.python`, `host.process_id` and all planned host
  observations. Adapter PID must equal the parent's actual child PID.
- `input_manifest_sha256`, `input_bytes` and exact `timed_steps`.
- Ordered `parts`, each with `input: {sha256, sizeBytes}`,
  `archive: {artifactName, sha256, sizeBytes}`, `roundtrip_sha256` and
  `roundtrip_bytes`. Archive paths are relative to the observation directory.
  Every archive is a nonempty indexed retained file with matching hash/size.
  Every part's roundtrip hash and byte count must equal its input.

## Required CPU telemetry

CPU receipt v2 requires the finite observer's actual retained summary and raw
bytes. This collection is separate from optional profiler claims. An empty
`profile_evidence` list does not waive required CPU telemetry.

`raw.adapter_result.native_evidence` must contain exact `cpu_telemetry_summary`
and `cpu_telemetry_raw` references to `finite-cpu-telemetry.json` and
`finite-cpu-telemetry.raw`, alongside `observations`. Both are indexed artifacts
in the attempt output directory. The summary must equal
`observations.telemetry` as JSON. Its schema is `piq.finite-cpu-telemetry.v1`,
with canonical `host-procfs` source identity and producer ID
`finite-cpu-observer`. Its `source.producerSha256` is `sha256:<bundle hash>`;
that hash must match the full native `telemetryBundle` configuration, the
identical `observations.runtime.telemetry_bundle`, and the actual indexed ZIP's
hash and byte size. The ZIP must appear in the immutable native design's
additional input list. Retain its source and original plan bindings.

The gate checks all of the following, rather than trusting a complete flag:

- `terminal_coverage.state` is `complete`, raw/sample overflow is false, and
  dropped-error/gap counts are zero. Every error and gap remains explicit.
  Cgroup or host context failure cannot be relabeled a child-exit race.
- Every `raw_sources` record has its actual sequence, contiguous byte offset,
  size, exact retained/source digest and untruncated status. `raw_file` binds
  the complete file size and digest, with no omitted trailing bytes. Zero-length
  native records, including an empty `io.max`, retain their entry and the empty
  byte digest; they are not missing records.
- The adapter registration equals the canonical parent runner's child PID and
  `observations.host.process_id`. The codec registrations and owning-Popen
  terminal observations conserve every compression and decompression PID from
  every part, including its actual returncode. Raw `/proc` stat snapshots bind
  PID, starttime and actual parent PID; raw cgroup and membership records bind
  the registered cgroup identity. No CUDA/device identity is synthesized.
- Normalized samples reference their exact retained raw-source entry and
  canonical procfs-file framing. Their raw stat PID/starttime, process labels,
  source names and units must agree with the registered CPU process. Each
  registered process has an actual sample; the adapter has start/stop samples.
  A legitimate terminal process-identity race needs a prior successful sample,
  the same real owning-Popen terminal identity, and a bounded unobserved exit
  interval. Arbitrary missing samples cannot inherit a complete verdict.
- Required start/stop host and cgroup context is actually retained. The host
  sources include scheduling, memory, pressure, disk/network context, mount and
  filesystem records. Cgroup sources include CPU quota/accounting, effective
  cpuset, memory, I/O, PID limits and membership. The backing mount binds its
  retained native record. Missing required availability remains a blocker.
- Observation boundaries lie inside the actual parent process window. Native
  collection overhead remains explicit in nanoseconds. Sampling and raw-byte
  limits remain explicit and cannot hide dropped data.

Complete means complete for this required collection and its declared scope.
Filesystem/block counters remain shared-device context; network is host context
with the workload unexercised; fabric and accelerators remain unexercised.
RSS samples do not prove peaks or aggregate host memory, cgroup membership does
not imply exclusivity, and finite sampling cannot prove unseen descendants.
Retain the source's coverage and limitations. This collection establishes no
physical-media bandwidth, network performance, GPU execution or causal speedup
attribution. A synthetic fixture's complete flag says nothing about availability
on an actual execution host.

The normalizer rejects inconsistent/partial telemetry metadata. The public
`validate_cpu_telemetry` helper additionally verifies the indexed bytes and raw
record/PID bindings; the full receipt validator calls it for every native
attempt. A producer may call it immediately after export to stop on missing
observability before admitting another group. It does not launch or probe work.

The aggregate hashed `raw_census` JSON binds `study_id`, `design_sha256` and an
`attempts` list in the exact declared group/arm order. Each row is
`{group_id, role, run_id, attempt_id, envelope: {path, sha256}, raw_result:
{path, sha256}}`, using package-relative paths. It must conserve every attempt
from all eight envelopes, including failures. A run contains the two actual
attempts for one group; never invent a separate run per arm.

## Receipt and reviews

Receipt v2 declares `state: "measured"`, `execution_class: "cpu"`, and exact
design-matching `workload_scope`, `runtime_identifier`, `hardware_identifier`
and `source_artifacts`. It binds hashed `design`, `registration`, `raw_census`,
`correctness`, `qualification_report`, `baseline_review`, `scientific_review`
and all eight `piq_envelopes`.

Registration retains `study_id`, `design_sha256`, `registered_at`, a nonempty
`authority` and `record_id` identifying the authoritative advance record, and
`group_design_sha256s`, an exact eight-entry mapping from every declared group ID
to its pre-execution native `evidence/design.json` byte hash. This freezes the
per-group source specifications without a circular global-design dependency.
Design creation precedes registration. Registration and baseline review precede
every process; scientific review follows every process.

Each review is a separate retained JSON data artifact with `study_id`,
`design_sha256`, a named `reviewer`, `independent: true`, `verdict: "PASS"`, UTC
`reviewed_at`, and concrete `reasoning`. Reviewer cannot equal execution owner.
The baseline review also identifies `baseline_id`, `optimized_standard: true`
and nonempty HTTPS `primary_sources`; reasoning must explain its suitability.
The scientific review binds `qualification_report_sha256` and assesses source,
controls, correctness, timing boundary, scope and limitations. Booleans alone
do not establish an independent assessment.

`qualification_report` must equal deterministic evaluator output as JSON, with
verdict `QUALIFIED`; an edited verdict or ratio cannot pass. `correctness` binds
`study_id`, `design_sha256`, `accepted: true`, and exact `attempts` records
`{run_id, attempt_id, parts}` from every normalized sample. `accepted_repeats`
contains all four ordered `{id: "pair-01", accepted: true, raw_result:
<that group's hashed envelope>}` records. The envelope indexes the pair's
unchanged native raw results. `failed_attempts` retains every native failure as
`{run_id, attempt_id, failure_reason}`; any such entry blocks qualification.
Keep incomplete and failed studies privately rather than deleting entries.

Each claim has distinct `id`, hashed `source`, all four ordered
`accepted_repeat_ids`, and explicit boolean `profile_claim`. Empty
`profile_evidence` is valid for unprofiled claims. A profiler claim requires a
retained `{kind: "cpu_profile", capture_performed: true, study_id,
design_sha256, artifact: {path, sha256}}` entry. GPU/power placeholders are
forbidden. This CPU receipt cannot cover a CUDA rank or substitute for GPU v1.

## Capture-first product workflow

`tools/run_cpu_study.py` has three explicit stages: `prepare`, `capture`, and
`retain`. All three bind the same source, runtime, corpus, method, API origin,
review records, CPU affinity, and immutable registration. `capture` is the only
stage that launches scientific children. It refuses PIQ API credentials, makes
no PIQ client or network call, and writes the canonical local submission through
`piq_sdk.capture_manifest`. Its successful handback remains explicitly
`UNRETAINED` and does not evaluate qualification.

After the native process and its cleanup have ended, `retain` uses the configured
PIQ API. A fresh exact run readback of 404 selects `resume_capture`. An existing
run selects `complete_capture_custody`, which performs no run POST. Any other
read outcome is unknown and stops before a retention write. Retention never
launches or measures a child, and a prior retention start is reconciled from its
original handback instead of repeated. Only eight successfully retained,
correct, method-matching group envelopes may reach the unchanged evaluator.

## Limits and checks

These checks establish retained-byte consistency and fixed-rule arithmetic.
They cannot authenticate the execution producer, independently prove registration
time or reviewer independence, discover attempts omitted from every authoritative
record, or prove that a producer executed its source. Preserve original PIQ
records, custody, source snapshots and advance-plan timestamps for those questions.
No CPU qualification bypasses the complete content, repository, review, preview
or unscheduled Typefully gates. Publication still requires authorization.

```sh
python3 -m unittest -v tools.test_run_cpu_study tools.test_qualified_cpu_study tools.test_check_package tools.test_piq_experiment_evidence
```

`build_cpu_fixture(package)` in the test module creates only small, explicitly
synthetic evidence/receipt files. Its invented identities, timings, reviews and
archives describe no execution and must never be used as research evidence.
Its telemetry summary, source bundle, raw records, process IDs and availability
are also explicitly synthetic. They do not model or repair a real host's missing
cgroup/controller fields.
