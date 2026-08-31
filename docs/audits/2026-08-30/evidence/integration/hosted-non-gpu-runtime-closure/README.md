# Retained hosted non-GPU runtime reconciliation

This package reconciles already completed hosted evidence against the exact
remaining ledger contracts. It did not rerun the full suite.

[Benchmark Validation run 33391774956](https://github.com/cfregly/ai-performance-engineering/actions/runs/33391774956)
ran final source `3316e0efe985040745ffd926c5f76a6bd4436aff` on GitHub-hosted
Ubuntu 24.04 with CPython 3.12. Its retained JUnit records 4,807 cases: 4,346
passed, 461 explicit skips, zero failures, and zero errors. The skips are
heterogeneous and are not converted to passing capability or protection coverage.

Seven Wave 1 rows are fully verified because their exact findings concern host
configuration behavior rather than device execution:

- `W1-007`: hardware-alias architecture selection;
- `W1-052`: compare-loop failure propagation;
- `W1-055`: compute-capability/product metadata identity;
- `W1-057`: GB200 versus SM120 label and build routing;
- `W1-067`: the custom-vs-cuBLAS Make `ARCH` knob;
- `W1-111`: TMA helper root and target selection; and
- `W1-112`: warp-specialization wrapper root resolution.

[`w1-junit-subset.json`](w1-junit-subset.json) retains every exact passing JUnit
case used for those dispositions. The same file records the bounded selected-
Python ABI-query subgate for `W1-115`, whose pinned cu130 extension build/import
remains pending. Applicable real-compiler support is kept in the separate
[hosted CUDA compile receipt](../hosted-cuda-compile-closure/receipt.json).

[`wave2-runtime-audit.json`](wave2-runtime-audit.json) maps all 48 Wave 2
`awaiting_runtime` rows to 61 row-to-test references covering 56 unique exact
passing final-source JUnit nodes. It also
records nine exact four-target CUDA 13 compile/link subgates and the narrower
`W2-078` header compile result. None of the 48 whole rows is verified because the
required target-hardware acceptance cells have not run.

The reconciliation also advances bounded subgates for `LOCAL-006` (all four
fresh CPU/Gloo ZeRO child paths), `LOCAL-024` (Torch CPU plus Triton 3.5.1 Linux
co-install/import), and `LOCAL-027` (the bounded 20-pin/56-distribution Linux CPU
install plus final-source hosted CPU CI). Their full CUDA/NCCL or 90/327 graph
contracts remain pending.

B200 custody was unavailable. No GPU was probed or launched, and no numerical,
sanitizer, profiler, hardware, or performance gate is inferred from this package.
