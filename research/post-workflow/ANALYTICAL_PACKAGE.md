# One explicitly authorized analytical package

The normal performance workflow still requires its retained measured or
source-reported evidence. This optional reader-visual schema supports a human
exception for one named package that explains synthetic arithmetic. It does
not turn estimates into measurements, establish a speedup, authorize GPU work,
or authorize publication. A new package needs its own explicit user instruction.

Use this contract only when the user explicitly authorizes an analytical
performance package without a measured speedup. Retain the actual user message
reference and a verbatim excerpt. An assistant's proposal, inferred preference,
or this document cannot supply that authorization. The receipt records the
instruction rather than independently proving its authenticity.

The ordinary [reader standard](READER_STANDARD.md), copy/chart review, actual
platform attachment and phone-width preview checks remain in force. For this
exception, replace the measured comparison with clearly introduced analytical
values, assumptions, units and limits. Keep internal authorization and review
bookkeeping outside public copy. Public labels such as "hypothetical example"
or "analytical lower bound" explain the claim itself and belong beside it.

## Manifest and reader declaration

Every file reference below has this exact shape:

```json
{"path": "package/relative/file", "sha256": "64 lowercase hexadecimal characters"}
```

Paths must stay inside the package and cannot use symlinks. Each file must be
nonempty and match its retained digest. The example paths below are relative to
the package root, so do not include the directory name in their actual values.

The manifest must declare these fields:

```json
{
  "content_state": "hypothetical",
  "locally_measured_performance_claims": false,
  "local_gpu_execution": "not_run",
  "reader_visuals_required": true,
  "analytical_authorization": {"path": "evidence/analytical-authorization.json", "sha256": "..."},
  "reader_visuals_review": {"path": "evidence/reader-visual-review.json", "sha256": "..."},
  "reader_visuals": {
    "schema_version": 2,
    "analytical_basis": {"path": "evidence/analytical-basis.json", "sha256": "..."},
    "independent_review": {"path": "evidence/independent-review.json", "sha256": "..."},
    "scientific_chart": {"path": "images/comparison.png", "sha256": "..."},
    "chart_text": {"path": "images/comparison.txt", "sha256": "..."},
    "preview_receipt": {"path": "social/profile-preview.json", "sha256": "..."},
    "review_receipt": "evidence/reader-visual-review.json",
    "copy": {
      "article": {"path": "README.md", "sha256": "...", "sequence": []},
      "x": {"path": "social/x.txt", "sha256": "...", "sequence": []},
      "linkedin": {"path": "social/linkedin.txt", "sha256": "...", "sequence": []}
    }
  }
}
```

The ellipses and empty sequences are placeholders, not passing values. Every
sequence must contain `example`, `mechanism`, `analytical_comparison`, and
`scope` in that order, each with a nonempty verbatim `excerpt` found in the
current copy and saved draft. Chart, screenshot, attachment and saved-copy
checks are identical to schema 1. No `measurement_receipt`,
`source_reported_receipt`, or reader declaration `measurement_basis` is allowed.
Schema 2 is valid only for `hypothetical`. Schema 1 keeps its existing measured
or source-reported receipt and `measured_comparison` stage.

## Authorization receipt

```json
{
  "schema_version": 1,
  "authorization_kind": "explicit_user_instruction",
  "permission": "single_analytical_package_without_measured_speedup",
  "package_directory": "the-exact-package-directory-name",
  "claimed_scope": "The precise analytical claim this instruction permits.",
  "authorized_by": "The human who authorized it",
  "authorized_at": "2026-10-29T12:00:00-07:00",
  "source": {
    "kind": "user_message",
    "reference": "An exact conversation and message reference",
    "excerpt": "A verbatim excerpt of the actual analytical exception."
  }
}
```

The directory must match the package being checked. The claim scope must equal
the basis scope. The authorizing name, reference and excerpt must be nonempty,
and the time must be a completed ISO time with a timezone. Record the actual
time and provenance instead of copying this illustrative receipt.

## Derivation basis and arithmetic results

```json
{
  "schema_version": 1,
  "result_kind": "synthetic_arithmetic",
  "package_directory": "the-exact-package-directory-name",
  "author": "The actual derivation author",
  "claimed_scope": "The precise analytical claim this instruction permits.",
  "assumptions": ["Specific inputs, units and model assumptions."],
  "limitations": ["Specific costs, effects or behavior this model excludes."],
  "code": {"path": "code/derive.py", "sha256": "..."},
  "results": {"path": "evidence/analytical-results.json", "sha256": "..."}
}
```

Assumptions and limitations are nonempty lists of distinct statements. Retain
UTF-8 derivation source and its actual computed results. The results JSON has
`schema_version: 1`, repeats `result_kind`, `claimed_scope`, `assumptions`,
`limitations` and the exact `code` reference from the basis, and contains
`values`, a nonempty list of distinct named finite numeric values with units:

```json
{"name": "transfer_lower_bound", "value": 2, "unit": "s"}
```

These values are arithmetic outputs, not execution timing observations. The
validator never executes retained code. An independent reviewer must actually
recalculate the values and assess the assumptions and claimed scope.

## Independent recalculation review

Retain a JSON receipt with `schema_version: 1`, `status: "PASS"`, a nonempty
`reviewer` different from the basis author, completed `reviewed_at` no earlier
than authorization, and `method: "independent_recalculation"`. Bind its exact
current inputs through these four hash references:

- `analytical_authorization`: the manifest's authorization reference
- `analytical_basis`: the reader declaration's basis reference
- `code`: the basis code reference
- `results`: the basis results reference

Its `checks` object must contain each following key with `status: "PASS"` and
specific nonempty `notes`. Retain actual findings instead of boilerplate.

- `independent_recalculation`
- `arithmetic_and_units_correct`
- `assumptions_explicit`
- `claimed_scope_supported`
- `limitations_explicit`
- `no_measured_performance_implication`
- `no_serving_speedup_inference`

## Reader review

Keep the existing schema 1 reader review format, named reviewer, completed time
after the platform preview, `declaration_sha256` and screenshot-bound platform
findings. Compute the declaration digest with `declaration_sha256` from
`tools/check_reader_visuals.py`. Schema 2 requires these `checks` keys, each
with `status: "PASS"` and specific nonempty `notes`:

- `example_before_jargon`
- `example_explains_mechanism`
- `analytical_comparison_visible`
- `analytical_values_match_derivation`
- `assumptions_and_limits_visible`
- `scope_and_conditions_visible`
- `no_measured_performance_implication`
- `scientific_chart_selected`
- `phone_readable`

The manifest's `reader_visuals_review` hashes this final reader receipt and must
name the same path as `reader_visuals.review_receipt`. Keeping that hash outside
the declaration avoids a circular digest. Build receipts in this order:
authorization, code/results, basis, independent review, copy/chart/preview
declaration, reader review, manifest. Any changed input needs updated bindings
and an actual renewed review of the change.

Hashes prove which bytes a receipt refers to. They do not prove arithmetic,
human intent, independence, comprehension or readable pixels. Those remain
review judgments with retained provenance. Passing this optional contract does
not replace the remaining package stages or the full package check.
