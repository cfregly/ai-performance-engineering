# Post package checks

This is the committed source snapshot of the post-package validator and tests.
It checks content, six audience variants, image and evidence hashes, review
receipts, exact Typefully readback, and unscheduled draft state. It does not
qualify GPU performance or publish content.

Run from this directory:

```bash
python3 -m unittest tools.test_check_package
python3 tools/check_package.py ../2026-10-22_nested-graph-breaks
```

## Clear, runnable examples

Every performance post must link to its runnable example in this repository.
Include the command, runtime requirements, inputs, and observed run evidence.
An image or source-only check does not prove that an example ran on the GPU.
Commit and push the code, tests, scientific images, and shareable captures before
calling a related draft ready. Preserve completed runs and failed controls.

Name comparisons in ordinary language within each standalone post or image.
For the nested-graph study, say “with nested graph-break support” and “without
it”. Keep exact identifiers in code and methods references. Enabling support
does not remove graph breaks from the workload.

The wording check covers public article, social copy, and image captions. It
leaves code examples, raw measurements, profiler exports, and archived sources
intact. The local posts workspace uses this same byte-identical validator.
Update this snapshot and tests whenever that validator changes.
