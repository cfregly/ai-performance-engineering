# Ordinary hygiene collection and stale-contract integration

Final local result: **552 passed, 1 explicit Linux-runtime skip, 1 existing version warning** in 8.59 seconds. All 553 tests collected through ordinary pytest; no AST execution substitute, import mock, global dependency whitelist, test deselection or GPU fallback was used. The tested environment was Darwin arm64, Python 3.12.2, PyTorch 2.8.0 with no CUDA or Triton. This is CPU/source validation, not the pinned CUDA stack or any kernel qualification.

## Changes

Only `code/tests/test_benchmark_hygiene_regressions.py` was edited in this package. The two occupancy factory imports now live in their one dependent test. Its original 30 assertions are preserved across the metadata test and a separate source test. Both execute normally on CPU after the independently owned production lazy-import repair. All original test IDs remain; one source test was added by this split.

The whole-file run exposed assertions that still required behavior intentionally removed during Wave 1. Thirteen existing test functions were reconciled, in addition to the occupancy split:

- Custom GEMM verification now exercises the real per-element validator against finite corruption, NaN and infinity instead of requiring the retired global-maximum normalization.
- NVFP4 grouped GEMM checks the complete canonical group outputs against independent references before copying them into the preallocated payload (W1-086).
- CuTe timing retains explicit current-stream events and requires positive workload/iteration arguments after the layout/validation repair (W1-074).
- KV-cache tests require calibrated full K/V snapshots and shared inherited capture; parameter counts and metadata remain cached outside capture (W1-030 and associated policy repairs).
- Router tests require cumulative-token deltas and shared V0/V1 output consumption, preventing double counting (W1-028).
- Nanochat timing checks require device synchronization before and after the work, with the host clock enclosing submission and all streams (W1-036/W1-085). The SDPA test reflects the intentional CUDA-dependent Flash selection and a list copy of cached non-Flash backends.
- Persistent-decode tests require all sequences and full outputs instead of the old eight-value view or capped item count (W1-039/W1-088). Thread-level async copies are named accurately (W1-090).
- A newly observed adjacent prefill gap was reported to the owner: four prefill/decode variants still captured only part of decode output and omitted prefill results. The production owner added combined full decode/prefill payloads, independent validation and copy-only baseline/optimized workload parity. The existing hygiene IDs now require those full buffers and validation before capture. This new gap is adjacent evidence, not an invented original finding ID.

No production code was changed by this package. The source repairs belong to their existing owners and receipts. Tests do not restore old incorrect code to obtain a pass.

## Platform boundary

`test_run_benchmarks_reaps_current_run_descendants` exercises the production Linux `/proc` collector. Its original real child-process attempt failed on Darwin because `/proc` does not exist; the test's cleanup terminated that child. The exact runtime test now explicitly skips on non-Linux. On Linux, missing `/proc/self/stat` is an assertion failure before launching a child. All CPU/source process-table fixtures remain enabled. No portable process-cleanup implementation or Linux runtime acceptance is claimed.

The existing `placement_sim` warning about its preferred Torch/CUDA version remains in the preserved output. No CUDA workload was substituted with a CPU benchmark under a GPU claim.

## Evidence lineage

- `collection-before.txt`: original import failure, exit 2, zero tests collected.
- `collection-after.txt`: ordinary collection of 553 tests, exit 0.
- `cpu-source-tests-attempt-1.txt`: five independent CPU/source checks pass.
- `cpu-source-tests-attempt-2.txt`: eight CPU/source checks pass, including real occupancy factory construction, without Triton and without skips.
- `whole-file-diagnostic-attempt-1.txt`: ordinary `--maxfail=10` run, 278 pass / 10 fail. This was diagnostic while other modules could change.
- `reconciliation/whole-file-attempt-1.txt`: complete ordinary run after the first repair group, 550 pass / 3 fail.
- `reconciliation/whole-file-attempt-2.txt`: complete final run, 552 pass / 1 Linux-runtime skip / 1 warning; the test file and all five prefill dependencies were byte-stable during this invocation.
- Corresponding command JSON files retain commands, working directories, environment overrides, return codes and relevant source hashes.
- `scope-and-assertion-preservation.json` proves the initial occupancy split retained all 30 assertions. `reconciliation/final-scope.json` records all final changed functions and preserves every original test ID. Independent review of the import split is retained separately; it is not presented as review of all later assertion changes.
- Exact original/import-slice sources and every failed attempt remain. The earlier import-slice receipt is immutable and historical. An intermediate proposed missing-Triton skip was never executed, was removed before validation, and remains only as a clearly labeled historical diff.

Reproduce the final CPU/source run from `code/`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/miniconda3/bin/python -m pytest -q -rs -p no:cacheprovider tests/test_benchmark_hygiene_regressions.py
```

Actual CUDA/Triton compilation, GPU outputs, sanitizer runs, GPU timing/throughput, default-scale benchmarks and Linux process reaping remain separate acceptance gates. A green hygiene source check cannot close them.
