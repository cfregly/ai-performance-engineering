# Audit artifact integration review

Snapshot: `2026-08-30T23:37:25.677587+00:00`; base `b57e4c6a9e261c09ac09208705d040c81b03d35e`. This is an artifact and source-identity review, not a new test or GPU acceptance run.

The snapshot contains 512 evidence files. Checks of 66 receipt/manifest/report metadata files and 884 explicit file/hash observations found no missing hashed files, changed evidence hashes, unresolved file-like JSON values, or metadata changes during the scan. All five Markdown file links resolve.

Required follow-up:

- Preserve the existing package receipts as historical epochs. Parent will capture the final source hashes and run the available CPU integration checks after all writers stop.
- Record the final hashes and generator/documentation checks for `code/AGENTS.md`, `code/core/scripts/refresh_readmes.py`, and `code/labs/moe_optimization_journey/README.md`. These are expected subsequent edits; the existing receipts do not cover their current bytes.
- Include the dated audit tree and its scoped `.gitignore` in the eventual patch or commit. No audit files were tracked at this snapshot; `git diff` alone would omit these untracked artifacts. This review did not stage anything.

The other 23 paths with historical hash differences have their current hashes recorded by later package receipts or preflight reports. That establishes source identity only. In particular, a CUDA HOLD/preflight report cannot establish runtime correctness or substitute for the final CPU rerun. The shared hygiene timing test function still exactly matches its retained function hash; no whole-file test result is inferred from that check.

Portability: the scoped `!*.log` exception makes all 59 existing audit logs includable. All six ignored CPU profile files have byte-identical nonignored archive copies and extractor input copies. The only remaining ignored file without a copy is disposable Python bytecode. Original profile paths and historical receipts were not changed.

Ledger: original findings and package assignments each contain exactly `W1-001` through `W1-128` once; original title/location/severity fields are preserved. Counts remain 5 critical, 37 high, 58 medium, and 28 low. The source capture hash is `474779ab49b67c5c888e5f689b2400204b6d0f46f304219cedd0773694b6e1ba` and matches the ledger. Wave 2 remains required and `awaiting_report`.

Files: `scan.json` records the metadata snapshot and per-path drift; `all-hash-references.json` records the resolved observations; `review.json` records interpretation and limits. `audit_artifacts.py` is the scanner used for this snapshot. Do not rerun it in this frozen directory; a future scan should use a new evidence directory.

Boundary: no source, old receipt, ledger, Git index, tests, GPU workloads, or installation were changed or run. This scan checks explicit metadata references; it does not validate every prose claim, reconstruct all historical trees, or qualify CUDA correctness/performance.
