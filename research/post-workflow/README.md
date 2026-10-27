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

## Scientific profile previews

Declare `scientific_profiles: true` in a package manifest and follow [PROFILE_CONVENTIONS.md](PROFILE_CONVENTIONS.md). Run `python3 research/post-workflow/tools/check_package.py /absolute/package/path`. The preview gate binds every media ID to its source file, checks all document pages and rejects crop or overlay failures. Review the final Typefully editor and phone-width preview after upload. Keep source, final draft, schedule and publication as separate states.

## Configurable shared pipeline

`tools/pipeline_config.py` validates an owner profile from `pipeline.example.json`: private drafts repository, public repository and subdirectory, local research hours and IANA timezone, GPU host and theme. It performs no external actions. The same configuration and profile-preview tools are proposed upstream so authors can use different profiles with common code. `check_public_projection.py` validates an explicit staged public copy against retained sources, including scientific captures, without exporting or publishing files. New private drafts may require the claim-scope template before their final check.
