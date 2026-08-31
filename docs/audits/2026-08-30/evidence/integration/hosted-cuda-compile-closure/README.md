# Retained hosted CUDA 13 compile reconciliation

[Dual-Architecture Compare Builds run 33391774950](https://github.com/cfregly/ai-performance-engineering/actions/runs/33391774950)
already completed successfully at source revision
`3316e0efe985040745ffd926c5f76a6bd4436aff`. This reconciliation did not rerun
it.

The GitHub-hosted job used `nvidia/cuda:13.0.0-devel-ubuntu22.04`, CUTLASS
`v4.5.2`, and the repository's real Make entrypoints. It compiled and linked the
configured chapter targets for `sm_100`, `sm_103`, `sm_120`, and `sm_121`, then
reported that all configured builds completed successfully. The complete job log,
workflow source, and compare script are retained under
[`vendor/run-33391774950`](vendor/run-33391774950).

The exact row mapping is in [`compile-targets.json`](compile-targets.json):

- 14 pending Wave 1 rows have direct-translation-unit four-target compile/link
  evidence: `W1-001`, `W1-008`, `W1-011`–`W1-016`, `W1-045`, `W1-048`–`W1-051`,
  and `W1-103`. `W1-001` has both normal and `-DVERIFY=1` binaries.
- Three pending Wave 1 header rows have bounded consumer evidence: `W1-009`,
  `W1-046`, and `W1-104`. The map names every consumer that remains unbuilt; this
  is not all-consumer closure.
- Nine pending Wave 2 rows have exact-source four-target compile/link evidence:
  `W2-002`, `W2-015`, `W2-039`, `W2-050`, `W2-052`, `W2-100`, `W2-101`,
  `W2-102`, and `W2-119`.
- `W2-078` has a narrower header-through-consumer compile result for the four
  configured targets. Its H100/`sm_90` cell was not compiled.
- The same job supplies supporting compiler evidence for the exact configuration
  closures `W1-007`, `W1-052`, `W1-057`, and `W1-111`.

The job had no GPU or driver. None of these results executes a binary or qualifies
outputs, device ordering, numerical behavior, sanitizers, profilers, hardware,
or performance. Every pending row retains its exact remaining gate in the target
map and remediation ledger.
