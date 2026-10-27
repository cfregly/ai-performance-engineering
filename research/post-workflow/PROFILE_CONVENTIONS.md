# Public performance profile conventions

Use the profiler's established visual and unit conventions in public posts.
A performance flamegraph uses CPU or GPU time for width. Time-proportional
samples are appropriate when the sampling method establishes that relationship.
Kernel-launch counts and exported row counts do not establish time spent.
Keep count-based stack inventories in internal capture diagnostics.

Before uploading a scientific image:

1. Retain the exact field definition and primary source for its quantity and
   units. Verify any conversion. If an absolute unit is undocumented, use a
   justified relative-time share and state that normalization.
2. Select the exact process, GPU, event kind, stream, and workload phases before
   aggregation. Never mix another worker into one condition's profile.
3. Preserve raw captures. Retain source hashes, filtered counts, weights,
   normalization, and a reproducible renderer beside the image.
4. Distinguish inclusive and self time, GPU activity and elapsed wall time,
   and GPU work attributed to a host stack from CPU execution time.
5. Inspect full-size images and phone-width previews. Check the final platform
   text, image attachments, reply target, person tags, links, and unscheduled
   state after the final edit.

The package check invokes `tools/check_profile_conventions.py` for active
Zymtrace views. It verifies retained GPU-time meaning, dimensionless per-worker
fractions, exact source hashes and filters, positive finite weights, normalized
totals, and active image paths. Its regression tests reject count weights and
guessed absolute units. Independent scientific and visual review remains
required. The validator cannot establish an external source's truth by itself.

For the nested-graph study, the retained API defines `JsonEvent.c` as event
time. The public images use `100 * sum(c for stack) / sum(c for worker)` and
show percentage of captured GPU time. The API's unspecified absolute base unit
is not converted to milliseconds. These whole-capture profiles support stack
attribution. They do not replace the separate steady-state latency experiment.

## Render the platform, not only the source canvas

Detailed scientific profiles use one full-width image per X post or one profile
per LinkedIn PDF page. Use a single fitted comparison image for image-only
replies. Do not put detailed landscape profiles into multi-image grids that
crop the headings or shrink all labels.

Retain 400-pixel previews, readable full key-function labels, source bounds and
source hashes. After upload, inspect the actual Typefully editor and realistic
preview for every platform. Check the complete profile, headings and key labels,
and measure the displayed image or page aspect ratio. Account for carousel
controls and overlays. A valid source PNG alone is insufficient.

The package gate runs `check_social_preview.py`. Its saved preview receipt binds
source hashes and media IDs to exact draft readbacks, checks aspect-preserving
single-image or document-page presentation, rejects more than 0.5% cover crop,
and requires a visual readability review. Preserve exact reply parents,
resolved mentions, dormant promotion settings and unscheduled state after edits.

Declare `scientific_profiles: true` in each profiler package manifest. The gate
requires a retained attachment source for every media ID and checks every PDF
page against the document page count. Each page needs its own overlay and
readability review. Do not certify a whole document from its first page.
