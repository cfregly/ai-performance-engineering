# Qualify the fixed two-B200 Transformer Engine training study

This public source snapshot contains validators, schemas, an empty fixed
admissions anchor, and synthetic fixtures only. It contains no real execution,
admission, host, device, account, package, qualification, or publication data.

Measurement receipt v3 is the only qualification contract for the fixed
Transformer Engine MXFP8-versus-BF16 training design described here. It does
not accept GPU receipt v1, CPU receipt v2, or PIQ diagnostic envelope v1.
There is no compatibility promise, migration adapter, or backfill path for
those retired contracts. PIQ diagnostic envelope v2 retains the study. It does
not qualify it.

This is a retained-evidence and deterministic-arithmetic contract. Its source,
schemas, tests, and synthetic examples do not establish that a workload ran.
They do not authorize PIQ mutation, a B200 allocation, a package, a public
projection, a Typefully draft, scheduling, or publication.

## Public Python API

```python
from tools.qualified_gpu_training import (
    CRITERIA,
    MEASUREMENT_SEMANTICS,
    REQUIRED_PHASES,
    validate_gpu_design,
    normalize_gpu_attempt,
    validate_gpu_checkpoint,
    validate_gpu_telemetry,
    evaluate_gpu_study,
    validate_gpu_receipt,
    gpu_receipt_binds_envelope,
    build_gpu_public_summary,
)
```

The JSON contracts are:

- [measurement receipt v3](schemas/measurement-receipt-v3.schema.json),
- [PIQ diagnostic envelope v2](schemas/piq-experiment-evidence-v2.schema.json),
- [public-safe summary v1](schemas/gpu-training-public-summary-v1.schema.json).

Python performs the cross-file, path, byte, custody, identity, timing, and
scientific checks that JSON Schema cannot express. Every new JSON reader rejects
duplicate keys at any depth and `NaN`/infinity. Artifact paths must remain
inside their declared package or bundle, may not collide by case, and may not
cross a symlink. Hashing streams retained files.

## Fixed study and one changed variable

The baseline and candidate use the same two-rank, one-host DDP job, TE modules,
BF16 resident parameters, FP32 master weights and Adam moments, dataset bytes,
sampler, initial checkpoint, validation oracle, checkpoint/restart path,
telemetry, custody, and cleanup. The only change is:

```text
baseline:  te.autocast(enabled=False)
candidate: te.autocast(enabled=True,
                       recipe=MXFP8BlockScaling(fp8_format=Format.E4M3))
```

The immutable design freezes the 24-block, 2,048-hidden, 8,192-MLP byte model,
192 updates, batch and accumulation policy, pinned WikiText-103 revision and
converted bytes, BF16 quality oracle, TE 2.20.2, PyTorch 2.15.0, CUDA Toolkit
13.4.3, cuBLAS 13.8.2, loaded cuDNN/NCCL/driver identities, fused attention,
DDP/NCCL topology, device order, deterministic policy, complete clock boundary,
retention, cleanup, budget, and every criterion.

The design also binds, separately, the admitted and actually resolved absolute
path and SHA-256 for the runner, adapter, verifier, evaluator, checkpoint,
telemetry, and helper modules. A request-selected same-named module, ambient
`sys.modules` entry, guessed directory, or different resolved digest fails.

Start from
[the result-free design template](templates/gpu-training-qualification-design.json).
Every `UNFILLED` value and zero digest/limit must be replaced and independently
reviewed before registration. The template is not an execution design.

## Ratified machine semantics

The evaluator constants were fixed only after these five definitions were made
machine-exact. The immutable design must embed exported
`MEASUREMENT_SEMANTICS` without changes.

1. **Update p95.** Retain every update latency. The secondary population is
   exactly updates 13–95 and 98–192. Updates 1–12, checkpoint transition 96,
   and first-resumed transition 97 remain in raw evidence but are excluded.
   Sort the 178 eligible integer nanosecond values and select one-based rank
   `ceil(95*n/100)`. No interpolation is used.
2. **Coefficient of variation.** Use population, not sample, standard
   deviation across the four complete times for each arm. The 5% gate has no
   rounded operand: for count `n`, sum `S`, and times `x_i`, require
   `400 * sum((n*x_i-S)^2) <= n*S^2`. Display rounding is half-even only after
   the decision.
3. **Intervals and gaps.** Every span is a half-open integer monotonic interval
   `[start_ns,end_ns)`. Reject spans outside the primary clock. Merge touching
   or overlapping spans before deriving a phase's wall duration across ranks.
   Cross-phase overlap is allowed. Phase durations must never be added into a
   synthetic total. Each rank's low-overhead coverage union must equal the
   complete primary clock. Any positive uncovered nanosecond is an unexplained
   gap. Nothing is clipped.
4. **Quality operands.** Retain total NLL as integer nanonats and oracle target
   bytes as an integer. Retain byte accuracy as exact correct-byte and
   target-byte counts. Gate NLL and percentage-point differences with rational
   cross-multiplication and no rounding. Decimal rendering is explanatory only:
   NLL uses nine places and accuracy six, half-even.
5. **Power observation.** The design records the expected power-limit value for
   each admitted device in integer milliwatts. Every attempt retains exactly
   two read-only observations per device, in device/start/end order, from the
   named `nvidia-smi` power-limit query. Each observation must be within 1,000
   mW of its device's expected value. A setter, mutation attempt, device swap,
   missing sample, or different observer fails.

Boundary tests cover both sides of each definition. Changing one requires a
new criteria/semantics version and a new prospective design. It cannot repair
observed results.

## Fixed attempt census

The design has five groups and ten actual attempts in this exact order:

```text
admission-01: BF16  -> MXFP8   (12 updates per arm, never a measured repeat)
pair-01:      BF16  -> MXFP8
pair-02:      MXFP8 -> BF16
pair-03:      MXFP8 -> BF16
pair-04:      BF16  -> MXFP8
```

Each group has one real PIQ run ID and two real attempt IDs. Admission proves
only bounded support, finite values, two-rank/backend parity, and budget fit.
It cannot enter accepted repeats, a claim, gain/quality arithmetic, or a public
summary. Full attempts contain exactly 192 updates and the complete
checkpoint/restart job. No retry, replacement, reordering, omitted failure, or
invented run per arm is allowed. Any failed, timed-out, OOM, unsupported,
unaccepted, or incomplete attempt makes the deterministic report
`INCONCLUSIVE`.

## Complete job and exact gates

The primary clock starts before the initial checkpoint is opened and stops only
after final cleanup and evidence-manifest readback. Every full arm must retain:

- 192 optimizer updates, 768 accumulation rounds, 1,536 rank-local
  microbatches, and 12,582,912 target bytes,
- finite losses and gradients, with no overflow, skipped update, NCCL/CUDA
  error, Xid, ECC error, or fallback warning,
- one durable checkpoint at update 96, worker/launcher exit, a fresh two-rank
  restart, exact state reload, matching next batch and dry forward, update 97
  exactly once, a durable final checkpoint, and a clean-process final load,
- final independent BF16 evaluation on the same 512-window oracle, and
- continuous low-overhead full-stack telemetry, custody, local/PIQ/export
  readback, and terminal cleanup.

The evaluator uses integer nanoseconds/counts and `Fraction` cross-products. It
requires every pair gain to be at least 1.05, median gain at least 1.10, gain
spread at most 0.10, and within-arm population CV at most 0.05. The median
candidate/BF16 p95 ratio must be at most 0.95 and each pair at most 1.02.

Every arm must improve BF16-evaluated NLL by at least 0.10 nat per byte from the
shared initial oracle. Candidate NLL may be no more than 0.03 nat/byte worse in
any pair and 0.02 at the median. Candidate byte accuracy may be no more than
0.75 percentage point worse in any pair and 0.50 at the median.

For data open, NCCL initialization, periodic save, checkpoint load,
restart-to-first-update, final save, validation, evidence flush, and cleanup,
the candidate wall-time union must be no greater than
`1.10 * baseline + 2,000,000,000 ns` in every pair. These phase gates do not
weaken the complete-time gate.

## Checkpoint and detached-object custody

Periodic and final checkpoints bind nonempty object IDs, content SHA-256, and
byte counts. The state census includes model, FP32 master weights, both Adam
moments, scheduler, global step, sampler cursor, and both ranks' RNG state.
Pre-exit and post-load digests must match exactly. The saved and resumed next
batch digests must match. Dry-forward loss may differ by no more than 1,000
nanonats. Step 96 must load as 96 and advance to 97 exactly once. No arm-specific
external TE state may be omitted.

Large checkpoint and raw-telemetry payloads remain detached from Git. A
detached object is valid only with its content hash, byte count, media/role,
immutable custody record, local durable readback, PIQ readback, export
readback, and retained availability. A locator or digest without all three
readbacks is insufficient. A private package builder may copy compact manifests
and proofs only. It must reject unindexed payload bytes.

## PIQ diagnostic envelope v2

The v3 receipt accepts only envelope v2 experiment kind
`te_mxfp8_gpu_training`, with `kind: "piq_experiment_evidence"`,
`scope: "diagnostic"`, and `workload_family: "gpu_training"`. It refuses
`scope: "qualified"` and a qualified scientific disposition. The current v2
schema also represents Stage 0 and CPU two-host diagnostic variants, but those
can never bind or qualify this receipt. One TE envelope binds all five
executions and ten attempts.

Every attempt has two complete CUDA rank rows on one observed host: ranks and
local ranks 0 and 1, distinct PIDs, distinct admitted device UUIDs, exact rank
outputs, and the exact arm-configuration hash. Each execution binds requested
and observed one-node/two-accelerator/two-process resources plus scheduler job,
owner, node, session/cgroup, device, source, runtime, executable, plan, and
time-window assignment evidence. Each raw attempt's ordered UTC interval must
be contained by its exact assignment window. The stack must observe host,
accelerator, fabric, storage, orchestration, runtime, and workload.

Required workload roles are execution, quality, checkpoint/restart, telemetry
summary, telemetry raw/custody index, cleanup, readback, and export. The
artifact index is complete: unindexed files, missing bytes, case collisions,
changed bytes, symlinks, unsafe paths, malformed JSON, and nonfinite JSON fail.
Failed attempts remain in order. Import copies only declared evidence into a
new destination, validates it again, and writes a nonpublishable diagnostic
receipt. Import never launches or reserves anything.

```sh
python3 tools/piq_gpu_training_evidence.py check /absolute/path/to/v2-bundle
python3 tools/piq_gpu_training_evidence.py import /absolute/path/to/v2-bundle /new/local/destination
```

## Receipt, registration, and reviews

Receipt v3 has a closed top-level field set and exact references for the
design, registration, source/import closure, baseline review, telemetry review,
raw census, correctness/quality, checkpoint/restart, telemetry coverage,
custody/readback/cleanup, qualification report, scientific review, sources,
one PIQ v2 envelope, four accepted pairs, complete failures, optional real
profiles, and claims.

The immutable design is registered, and independent optimized-baseline and
telemetry/semantics reviews are retained before execution. Registration binds
the design hash and all five native group-plan hashes. Registration and both
pre-execution reviews precede every attempt. The independent scientific review
follows every attempt and covers source identity, controls, correctness/quality,
checkpoint/restart, timing boundary, telemetry, custody/cleanup, scope, and
limitations, and binds the deterministic report hash. Reviewers must be named,
substantive, independent, and different from the execution owner.

The stored qualification report must equal evaluator output as an object and
must also use canonical sorted, indented UTF-8 JSON bytes. An edited verdict,
number, key, or whitespace fails recomputation. Every claim binds the exact
report and all four pair IDs. A profile claim additionally requires a nonempty
real capture and parser acceptance. Ordinary telemetry is not a profile.

## Evidence-to-package and public boundary

Private package construction remains outside this reusable public snapshot.
The private source must contain exactly the validated evidence closure. Any
private builder must reject synthetic or inconclusive evidence, symlinks,
unexpected files, prewritten diagnostic interpretation, changed copied bytes,
and an existing or concurrently claimed destination. It must revalidate copied
evidence, leave review `PENDING`, and never materialize detached tensors or
profiler payloads. No package scaffold is projected here.

The private receipt, census, envelope, assignments, host/boot/device/job/PID/
cgroup/owner/window identities, raw telemetry, checkpoint state, custody
records, and reviews stay private. A qualified receipt may produce only the
deterministic allowlisted summary:

```sh
python3 tools/project_gpu_training_public_summary.py \
  /absolute/private-package /new/public-summary.json
```

The projector refuses overwrite and any non-qualified input. It exposes the
public-safe method and complete boundary, aggregate gain, repeatability,
latency, quality, phase, correctness, and checkpoint verdicts, public repeat
ordinals, report and design hashes, scope, exclusions, tradeoff, and unmeasured
limits. The projection gate rejects mappings of private evidence and scans the
summary for retained private execution identities.

## Limits and checks

A passing synthetic fixture establishes validator behavior only. Hashes provide
retained-byte integrity, not producer authentication. The validator cannot
prove that an external producer ran its source or that an authoritative system
did not omit activity from every retained record. Preserve original advance
registration, PIQ records, owner handback, and custody for those questions.

```sh
python3 -m unittest -v \
  tools.test_qualified_gpu_training \
  tools.test_piq_gpu_training_evidence \
  tools.test_gpu_training_public_summary
```

No qualified receipt bypasses content, repository, review, image, rendered
preview, unscheduled Typefully, or publication-authorization gates.
