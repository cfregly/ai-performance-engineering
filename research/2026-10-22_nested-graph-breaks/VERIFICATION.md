# Verification

## Source and claim review

The public claim is limited to the depth-8 compiler microbenchmark on one
NVIDIA B200. The original retained cohort measured a 4.531795x steady
geometric-mean ratio. Independent fresh run 022 measured 4.541758x. The two
cohorts are named separately in the article, claims file, sources, and social
copy.

The startup boundary remains visible. The original candidate was 33.2 percent
slower across the first ten compiled-call intervals, and the original startup
control failed. Fresh run 022 also failed the startup measurement rule, while
its startup control passed. No text describes startup as qualified.

## GPU stage

This is a measured package. The original retained 24-cell B200 matrix and the
independent fresh022 24-cell B200 matrix are complete. Each cohort passed all
retained correctness checks and its steady-state qualification. Fresh022 also
retains accepted Nsys and NCU evidence. No GPU workload was launched to create
this package metadata, copy, or receipt projection.

The measurement receipt binds both accepted cohorts, the fixed source, the
fresh022 run guard, correctness, native profiler evidence, and exact retained
power-limit telemetry. The receipt does not combine their samples.

## Profiler stage

The fresh022 native `.nsys-rep` and `.ncu-rep` files, SQLite databases, CSV
exports, Chrome traces, hashes, and qualification receipts are retained under
`profiles/fresh022/`. The Nsys view uses graph-level tracing. Internal CUDA
Graph node kernels are not reconstructed. NCU retained one matching fused
kernel per arm and used replay-based counter collection.

Normal unprofiled workers establish latency. The profiles provide supporting
mechanism evidence and are not timing acceptance measurements.

## Content review

The article, claims, sources, retrospective analysis map, follow-up answers,
and dual-platform package-review copy were checked against the retained JSON
evidence. Human-facing text uses plain language, keeps the original and repeat
separate, and contains no serving, training, model-quality, or cross-GPU win.

`analysis-plan.md` is explicitly retrospective. It does not claim that a new
pre-experiment plan existed or modify the fixed `protocol.json` gates.

## Image stage

The package now contains three explainers, six distinct audience variants,
landscape and square cover images, text sidecars, 400-pixel previews, and contact
sheets. The cover is a deterministic adaptation of the recovered original
Gamma export. Its source export and derivation are recorded separately.

An independent reviewer inspected the landscape and square assets and previews.
`images/review.json` records PASS and binds each asset to its hash. The shipping
files are exact copies of the selected reviewed cover.

## Typefully stage

The five reader-specific replies remain separate unscheduled drafts, with their
Nsys and NCU images and repository links. Their prose now names nested graph-break
support rather than referring to unexplained Boolean values.

The single dual-platform package review is draft 11497664:
https://typefully.com/?a=337207&d=11497664

Its exact X and LinkedIn text, three attached images, and unscheduled state were
read back. The images are the selected cover plus the Nsys and NCU views.
`social/typefully-receipt.json` binds the texts, media IDs, and shipping-image hash.
The copies under `typefully/` and `social/` are byte-identical.

## Publication state

Nothing was published or scheduled by this work. The five reader replies and
new package-review draft remain unscheduled. The original published post is
separate from these drafts.

## Package gate

The package passes `make check SLUG=2026-10-22_nested-graph-breaks`.
The wording-check tests pass 42 cases, the existing scientific CPU suite passes
35 cases, the visual tests pass 7 cases, and the Zymtrace controller tests pass
20 cases. These checks validate their declared source and artifact scopes.
They do not rerun or re-qualify the retained GPU measurements.

The public-copy gate rejects unexplained Boolean comparisons and the removed
retention boilerplate. It leaves exact code examples and profiler data intact.

## Fresh Zymtrace capture

The fresh capture is still in progress. Its bounded controller and source pins
are committed. A launch or injected worker alone will not be counted as a
capture. Completion requires positive backend events for both worker PIDs,
the assigned GPU and stream IDs, retained exports, and verified cleanup.
