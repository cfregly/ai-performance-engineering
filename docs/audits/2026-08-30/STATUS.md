# Audit remediation status

The goal remains active. All 128 original first-wave findings have dispositions,
but 76 still require supported runtime evidence. The completed second-wave report
has not been supplied. Changes remain uncommitted on
`codex/audit-remediation-20260830`; no merge, deployment, GPU qualification, or
performance result is claimed.

## Current source checkpoint

- Original ledger: **52 `source_fixed`, 76 `awaiting_runtime`**. There are no
  untriaged or in-progress original entries.
- Adjacent discoveries: **42 total — 30 `source_fixed`, 12
  `awaiting_runtime`**. LOCAL-038 through LOCAL-042 cover the tensor-parallel
  attention defects exposed by the preserved broad diagnostic.
- Final tensor-parallel source acceptance: **18 focused tests passed** and **2
  related hygiene tests passed**. Directed coverage includes poisoned padded KV
  state, cache-offset causality, heterogeneous and zero input lengths, exact
  length validation, monotonic workspace growth, bounded mask fallback, and
  repeated autograd. The assigned fixer also ran a 160-case independent oracle
  sweep; the file-backed JUnit receipt remains the acceptance evidence.
- Repository gates: `make lint` passes across **932 benchmark files** with zero
  contract errors or warnings; fatal repository Ruff, focused Ruff, focused
  mypy, targeted compile/static checks, and the import-edge checker pass.
- Per the user-directed verification policy now recorded in `code/AGENTS.md`, the
  full CPU suite was **not rerun** after the final tensor-parallel fixes. The
  earlier run is retained only as diagnostic evidence: **4,181 passed, 450
  skipped, 1 failed**. That failure led to the final fixes and is not a current
  integration result.
- The focused checkpoint binds **392 changed, new, or deleted source paths** to
  HEAD `b57e4c6a9e261c09ac09208705d040c81b03d35e` and tracked-diff SHA-256
  `2ec6c0ac1a634b54742171b221c2d81152a8e1573597f625a66575fed3fef719`.

## Other completed source work

- Offline profiling, metadata, Grace-provider, chapter-profiling, extension, and
  Chapter 3/4 seams pass **173 focused CPU tests with 6 capability skips**.
  Independent read-only review passed 141 seam tests and adversarial negative
  controls. Real Nsight, CUDA/NVCC, Grace hardware, and performance evidence are
  separate runtime gates.
- Four ZeRO factories now consume fresh child-produced training state and pass
  the final CPU/Gloo protocol: **79 passed, 2 CUDA/NCCL skips**, plus **6
  critical negative controls**, a four-factory matrix, and an independent probe
  observing six completed reduce-scatter and six completed all-gather calls per
  rank. The remaining **57 generic wrappers** still refuse unsupported harness
  verification.
- All **90 direct dependency specifications** resolve to a **327-package**
  CPython 3.12 x86_64-manylinux metadata graph under one reviewed GPUtil source
  distribution exception. The CPU workflow has 20 matching pins, a 56-package
  exact-source union, and a 49-package first-PyPI closure. This is metadata
  resolution, not Linux installation or CUDA qualification.
- Broad import validation on Darwin CPU remains an expected **594/600**. The six
  failures require Triton, cuda-python, or FlashInfer from the pinned target
  runtime. The import validator preserves their exception classes and complete
  diagnostics; no lazy fallback was added.

## Material limits

The local checks used macOS/arm64, Python 3.12.2, and CPU PyTorch 2.8.0. Actual
pinned Linux installation and CI, CUDA imports/builds, two-GPU CUDA/NCCL
correctness, Nsight capture, sanitizer checks, Grace observations, reviewed
numerical policies, and performance measurements remain pending. Capability
skips are not passing GPU coverage.

`HANDOFF.md` still assigns both B200 GPUs to another task, so this task has not
probed them, changed clocks, or launched a job. The Docker Linux route remains
blocked by the macOS Keychain credential helper (`-25293`); no bypass was
attempted.

At the 2026-08-31T03:11:40Z intake check, the supplied artifact still reported
the original 128 findings. Its “Not yet reviewed” footer remained a
future-coverage inventory, not a completed second wave. The user reported that
Wave 2 was on hold for one more hour, and a one-time task follow-up was scheduled.
Missing Wave 2 input is not treated as zero findings.

## Reviewable artifacts

- [Execution plan](../../../AUDIT_REMEDIATION_PLAN.md)
- [Issue ledger](remediation-ledger.json)
- [Focused source-gate receipt](evidence/integration/source-gate-followup/receipt.json)
- [Focused source manifest](evidence/integration/source-gate-followup/source-targeted-final.json)
- [Semantic seam receipt](evidence/integration/semantic-seam-closure/receipt.json)
- [ZeRO supplement](evidence/integration/zero2-child-protocol/supplement-receipt.json)
- [Dependency review](evidence/integration/full-dependency-review/receipt.json)
- [Remaining runtime acceptance](evidence/integration/pinned-linux-integration/runtime-update.md)
- [Latest Wave 2 artifact and task check](evidence/intake/second-wave-check-20260831T031140Z.json)

Nothing has been staged, committed, published, merged, or deployed.
