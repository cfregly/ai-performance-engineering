# Audit remediation status

The goal remains active. All 128 original findings have source dispositions and
evidence; the completed second-wave report is still required. Changes remain
uncommitted on `codex/audit-remediation-20260830`. No merge, deployment or GPU
qualification is claimed.

## Completed local work

- **3,948 passed, 449 explicitly skipped, zero failures** in ordinary `code/tests`
  (456.78 seconds). All 4,391 previous case identities remain; six cases were added.
  The obsolete HTTP case changed from false success to an explicit unsupported
  skip. No CLI test-file exclusions were used.
- All 321 recorded source/documentation/test hashes and the tracked diff stayed
  unchanged through the run. Syntax and fatal-name checks passed for 212 Python
  files. The audit plan, mutable status/ledger and evidence are tracked separately.
- Triton now uses 3.5.1, as required by the pinned Torch Linux wheel. CPU CI also
  explicitly installs `requests` and `tokenizers`. The corrected CUDA core pair
  resolves 26 packages; the CPU package set resolves 58 with verified CPU Torch
  and PyPI sources. These are Linux-target metadata results, not installations.
- Independent review accepted the dependency corrections and verified 121
  package-artifact hashes. The HTTP package's 27 artifact hashes also match.
  Earlier evidence remains intact; the old mutable status page was preserved
  byte-for-byte before its update.
- Original ledger: 52 `source_fixed`, 76 `awaiting_runtime`; no untriaged or
  in-progress original items. The separate refutation and 26 adjacent discoveries
  remain recorded. These statuses do not certify GPU execution.

## Material limits

The local run used macOS/arm64, Python 3.12.2 and CPU PyTorch 2.8.0, without CUDA,
nvcc or Triton. It does not verify execution with the new Linux package pins.
Capability and unsupported-policy skips are not passing coverage. The existing
quick benchmark-correctness selection was retained, not an all-target sweep.

The 61 generic distributed-training wrappers reject harness execution and
verification because the former parent-side toy model did not verify child
training. Direct scripts and factory configuration remain available; restoration
requires actual child-produced results and an independent reference. The three
legacy HTTP microbenchmark/export routes also remain absent. New HTTP checks
exercise the actual ASGI server and failure handling, not those retired features
or the production `serve` CLI.

Actual pinned Linux installation/CI, full staged CUDA provisioning, supported
builds, GPU/NCCL/stream/sanitizer checks, reviewed numerical policies and new
performance evidence remain pending. The binary-only full-requirements probe
stopped at a source-only GPUtil distribution; it establishes neither full graph
success nor a normal pip/setup version conflict. Historical numbers are not
requalified. `HANDOFF.md` still assigns both B200 GPUs to another task; no GPU
probe or job was launched. The 32-case attention gate retains all 115 source
hashes, but its actual CUDA acceptance remains HOLD.

Docker Desktop started and resumed nine existing containers through their restart
policies. None was modified or stopped by this task. The audit image pull failed
at the macOS Keychain credential helper (`-25293`); no audit container was created
and no bypass attempted. Docker remains running. A successful user-side credential
unlock is needed before retrying that local Linux route.

The supplied audit artifact still contains only the original 128 findings. Its
future-coverage footer is not a second-wave report. Missing input is not treated
as zero findings.

## Reviewable artifacts

- [Plan](../../../AUDIT_REMEDIATION_PLAN.md)
- [Issue ledger](remediation-ledger.json)
- [Current local receipt](evidence/integration/pinned-linux-integration/receipt.json)
- [Current source manifest](evidence/integration/pinned-linux-integration/source-before-full-cpu.json)
- [Complete current test output](evidence/integration/pinned-linux-integration/full-cpu.txt)
- [Dependency evidence](evidence/integration/pinned-linux-readiness/README.md)
- [Remaining acceptance](evidence/integration/pinned-linux-integration/runtime-update.md)
- [Earlier preserved local checkpoint](evidence/integration/final-local/receipt.json)

Nothing has been staged or published. A later review bundle or commit must include
new source files and dated evidence as well as tracked modifications.
