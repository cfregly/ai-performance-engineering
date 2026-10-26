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
