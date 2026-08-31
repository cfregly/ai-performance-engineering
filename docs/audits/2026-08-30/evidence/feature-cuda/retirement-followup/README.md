# Retirement follow-up

The earlier immutable P04 receipt recorded removal of the incomplete, unreferenced experiment. That attempt and its exact original source remain preserved. This follow-up restores the original path as a `#error` retirement stub to comply with the no-file-deletion rule at code/AGENTS.md line 21, without restoring nonexistent APIs or claiming that the separate experiment is qualified. The source test now requires explicit retirement rather than deletion.

`host-tests.txt`: 50 tests passed. `preprocessor-failure.txt`: the real host C++ preprocessor rejected the stub with its explicit retirement message (expected exit 1). This is a host control, not a CUDA build. The separate `tcgen05_tma_multicast.cu` source stayed unchanged; CUDA compile/full-output/sanitizer acceptance remains HOLD.

Earlier `validation-receipts.json`, `source-manifest.json`, originals and all attempt logs were left untouched. The earlier deletion record is historical and superseded for the two changed source paths by this receipt.
