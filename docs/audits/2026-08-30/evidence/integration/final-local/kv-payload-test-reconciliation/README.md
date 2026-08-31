# KV payload regression-test reconciliation

The two existing parameterizations now pass against the repaired production contract. The whole runtime-gating file and numerics regressions passed together: **60 passed, one actual-CUDA skip** in 6.96 seconds (24 runtime-gating cases; 36 numerics cases passed and one skipped).

Only `test_kv_cache_verification_output_is_built_after_benchmark` changed. Its name, parameterization and signature remain; all other top-level imports, classes and functions are AST-identical to the preserved original.

The test runs actual CPU BF16 `KVCacheAttention` with ordinary PyTorch Linear/LayerNorm modules and real cache writes. Twelve distinct balanced sign rows, zero-epsilon LayerNorm, diagonal projections and distinct channel biases have an independent exact analytic K/V reference. The test compares all 192 nonzero cache elements, refuses the missing accuracy policy, uses an exact zero policy only for this CPU fixture, and checks non-aliasing plus corruption at the last K and V element. The real production reference and accuracy comparator remain active.

The original post-timing control remains: stack/cat operations fail only around the readiness marker, then are restored before capture. No CUDA/TE setup or benchmark execution is simulated. This fixture does not calibrate or qualify FP8/NVFP4, CUDA numerics, performance or hardware behavior.

Preserved attempts include the parent full-suite failures and the first edited test run, which failed because the test treated the tuple-returning tolerance API as a dataclass. The tuple assertion was corrected; no production code changed. See `receipt.json`, `before-observation.json`, `focused-attempt-1.txt`, `combined-attempt-1.txt` and `scope-proof.json`.
