# Analytical calculation validation

Python 3.14.6 standard-library checks on October 29, 2026:

- All 16 unit tests passed, including exact byte counts, unit conversions, capacity boundaries, padded tiles, batch volumes, target-equality boundaries and invalid inputs.
- The three generated JSON/CSV files reproduced byte for byte.
- Integer and dyadic floating-point toy reductions matched an independent output-cell oracle with unchanged reduction order.

No GPU execution or timing was performed. These checks establish the declared arithmetic and tiny CPU toy equivalence. They do not establish attention-derivative correctness, implementation fit, attainable runtime or speedup.

Run from the companion directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s code -v
PYTHONDONTWRITEBYTECODE=1 python3 code/roofline.py --output-dir evidence/derivation --check
shasum -a 256 -c evidence/derivation/SHA256SUMS
```
