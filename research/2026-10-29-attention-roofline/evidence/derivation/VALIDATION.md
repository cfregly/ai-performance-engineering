# Analytical calculation validation

Python 3.14.6 standard-library checks on October 29, 2026:

- All 16 unit tests passed, including exact byte counts, unit conversions, capacity boundaries, padded tiles, batch volumes, target-equality boundaries and invalid inputs.
- The three generated JSON/CSV files reproduced byte for byte.
- Integer and dyadic floating-point toy reductions matched an independent output-cell oracle with unchanged reduction order.

October 31, 2026: all 19 current unit tests passed, including evidence-hash binding and the CLI check/reject test. The 16-test result above records the October 29 validation.

The three README commands also passed on October 31 from a clean checkout of public commit `86fb8f0f5e1ad9452474a32fadb93ce7b052150d`. The roofline calculator, its tests and retained arithmetic data are unchanged in this revision. Selected figure rendering is checked separately. The generated JSON and CSV files matched the retained derivation byte for byte. The retained reproduction receipt records the checkout identity, commands and exit codes.

No GPU execution or timing was performed. These checks establish the declared arithmetic and tiny CPU toy equivalence. They do not establish attention-derivative correctness, implementation fit, attainable runtime or speedup.

Run from the companion directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s code -v
PYTHONDONTWRITEBYTECODE=1 python3 code/roofline.py --output-dir evidence/derivation --check
```

The retained `SHA256SUMS` is historical: it records the October 29 snapshot, including the earlier README, tests and validation note. It is not a checksum command for this maintained tree. The maintained files are checked by the current tests and reproduction command above. Current source hashes are recorded in `SHA256SUMS-current`, which intentionally excludes this note and both checksum ledgers to avoid a self-reference.

Check the maintained source snapshot from the companion directory:

```sh
shasum -a 256 -c evidence/derivation/SHA256SUMS-current
# On systems with GNU coreutils instead:
sha256sum -c evidence/derivation/SHA256SUMS-current
```
