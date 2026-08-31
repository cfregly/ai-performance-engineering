# Retained hosted CPU closure

[Benchmark Validation run 33391774956](https://github.com/cfregly/ai-performance-engineering/actions/runs/33391774956)
already completed successfully at source revision
`3316e0efe985040745ffd926c5f76a6bd4436aff`. This reconciliation did not rerun
it.

The GitHub-hosted Ubuntu 24.04, CPython 3.12.14 x64 job explicitly installed
`requests==2.34.2` and `tokenizers==0.22.2`, ran the complete `tests/` directory,
and uploaded JUnit evidence for 4,807 cases: 4,346 passed, 461 explicit skips,
zero failures, and zero errors. The retained JUnit contains 49 Nanochat cases;
41 passed on CPU and eight CUDA cases skipped explicitly.

The 461 skips are heterogeneous. They include GPU/runtime holds as well as 62
known non-capability scope skips: 57 labeled `Missing production protection`,
three labeled `Missing protection`, one protection-coverage summary, and one
absent legacy dashboard-route check. They are retained as skips and are not
presented as passing protection coverage.

This fully closes adjacent discovery `LOCAL-025`, whose only remaining contract
was clean Linux installation, collection, and execution with those dependencies.
The same run closes W1-006's Linux CPU full-directory CI sub-gate. W1-006 still
requires an explicit supported-GPU runner to execute the CUDA protection and
verification cases, so its overall runtime acceptance remains pending.

The exact uploaded ZIP, extracted JUnit, and source workflow are retained under
[`vendor/run-33391774956`](vendor/run-33391774956). The `vendor/` segment marks
these immutable external files as evidence rather than live project manifests.
Structured facts and file hashes are in [`receipt.json`](receipt.json).
