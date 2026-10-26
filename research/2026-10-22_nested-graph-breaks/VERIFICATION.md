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

The six audience variants, three concepts, cover, previews, contact sheet,
sidecars, shipping image, and independent image review are owned by the image
lane. This file does not predeclare that review's verdict. The package remains
incomplete until those retained files and the hash-bound `images/review.json`
are present.

## Typefully stage

Five existing reader-specific reply drafts were read back as unscheduled and
unpublished, with two retained Nsys and NCU images each. They remain intact and
are not represented as one draft.

A new dual-platform master review draft is specified under `typefully/`, with
byte-identical package copies under `social/`. Its required title is `Nested
graph breaks | package review | DRAFT ONLY`. This verification does not claim
that the new draft exists. The Typefully lane must run the guard, create the
single unscheduled draft, read back exact X and LinkedIn text plus both media
attachments, and then write `social/typefully-receipt.json`.

## Publication state

No package content was scheduled or published by this work. The retained five
reader drafts remain unscheduled. The new master review draft is pending the
Typefully lane's guarded creation and exact readback.

## Package gate

The retained package CPU suite passed 35 tests:

```text
python -m unittest test_analyze test_counter_capture test_publication
Ran 35 tests
OK
```

The latest package gate reached the image and Typefully lanes with no content,
manifest, source, measurement-receipt, or hash error. Its seven remaining
items were:

```text
images/README.md
images/audiences.md
image.png
image-square.png
image.txt
images/review.json
social/typefully-receipt.json
```

The final package gate must run again after those files are present. A passing
artifact gate does not independently judge visual quality or re-qualify the
GPU measurements.
