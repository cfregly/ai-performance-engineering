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
25 cases. These checks validate their declared source and artifact scopes.
They do not rerun or re-qualify the retained GPU measurements.

The public-copy gate rejects unexplained Boolean comparisons and the removed
retention boilerplate. It leaves exact code examples and profiler data intact.

## Fresh Zymtrace capture

Both intended PyTorch 2.14.1 workers completed in a fresh custodian-approved
B200 idle window on October 26, 2026. The controller ran for 43 seconds from
20:21:10 to 20:21:53 UTC. Both workers passed correctness and runtime checks.
The [capture directory](zymtrace/captures/torch2141-b200-20261026/) retains the
raw event and flamegraph exports, audit results, worker results, and cleanup.

One earlier B200 worker ran under system PyTorch 2.13 because the plan resolved
the virtualenv interpreter symlink. That attempt was rejected and retained
separately. Its backend export contains 348 CUDA records with matching PID,
GPU, and stream fields. It does not satisfy the intended runtime comparison.

The launcher now preserves the virtualenv invocation path and performs a
CPU-only Torch identity check before GPU work. The installed virtualenv was
read back as Torch 2.14.1+cu130, CUDA 13.0, with the correct executable and
prefix. The rejected worker, profiler, and temporary rule were cleaned up.

Backend readback accepted 104 matching CUDA records without nested graph-break
support and 188 with support. The second raw export includes the earlier
worker because the time filters overlap. The retained audit identifies each
worker by PID, GPU, and stream. Both owned worker groups were empty, the
profiler exited cleanly, the temporary rule was removed at version 12, and
immediate GPU readback was empty. No startup, serving, or new performance
qualification is claimed from the Zymtrace visibility work.

The social drafts also include CUDA stack views derived from the retained
Zymtrace event exports. Each view filters records by the worker's PID and
assigned GPU before counting stack occurrences. Width represents exported
record count, not time. The renderer, input hashes, and full stack exports
are retained in [zymtrace/views/](zymtrace/views/). The original Nsys and NCU
images remain attached alongside these views.

## October 26 GPU-time flamegraph correction

The current Zymtrace images use GPU-time percentages from the API's event-time
field, filtered to the exact worker and GPU. Percent normalization avoids an
unsupported absolute-unit conversion. Both images and phone-width previews
passed independent review. Full stacks, time weights, fractions, source hashes,
unit authority and renderer tests are retained beside the images. Count-based
images are preserved as superseded diagnostics.

The package gate now rejects count weights, guessed absolute units, wrong
worker/GPU filters, invalid weights, stale image/sidecar hashes and incorrect
plotted fractions. The committed validator and guard passed 64 tests, and the
renderer passed five tests. Completed benchmark and capture files are unchanged.
All six existing Typefully drafts and four native LinkedIn composers remain
unsent and unscheduled, with their reply parents and mentions verified again.
See [the current draft receipt](social/time-weighted/receipt.json).

## October 26 final rendered profile check

The six active Typefully drafts were read back after the final media edits. All remain drafts with no schedule or publication time. X profile posts each attach one image. The main LinkedIn draft attaches a four-page PDF with 175px margins on all sides of each 1750px square page. Actual preview geometry and visible controls were checked on every page. Four LinkedIn reply copy banks and native unsent composers use one 1400×1750 portrait comparison each. Exact reply parents and real person mentions were rechecked.

[social/profile-preview.json](social/profile-preview.json) binds every final media ID to its committed source hash and saved readback. It records the geometry, full-profile visibility, label review and PDF control clearance. The pipeline checks this receipt and rejects multi-image profile grids, material cropping, distortion, mismatched media and stale draft review. Small stack labels remain available in the linked full-size files and interactive captures. This changes presentation only, with zero GPU calls or changes to completed run evidence.
