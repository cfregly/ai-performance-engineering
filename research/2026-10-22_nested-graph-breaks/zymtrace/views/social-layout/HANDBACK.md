# Social profile image handback

Objective: remove social-preview cropping while keeping complete time-weighted mechanism profiles and readable function names.

Status: COMPLETE for local image rendering. No external account changes, uploads, commits, or GPU execution were performed by this renderer owner.

Outputs:

- `zymtrace-without-nested-graph-breaks-square.png`: 1400 × 1400, one image for its own X post.
- `zymtrace-with-nested-graph-breaks-square.png`: 1400 × 1400, one image for its own X post.
- `zymtrace-nested-graph-breaks-comparison-portrait.png`: 1400 × 1750, one comparison image for LinkedIn.
- Matching SVGs, byte-identical per-arm stack sidecars, comparison source references, 400px and 520px previews, and `layout-manifest.json`.

Every accepted row is retained: 104 without support and 188 with support. Every mechanism frame selected by the existing `FOCUS` definition is retained. Bars use exported `c / sum(c)` within each worker. No absolute time conversion was introduced. Full unfiltered SVGs in `../time-weighted/` remain unchanged.

The source cards have 56px headlines and 44px key function names and percentages. The LinkedIn comparison uses 44px arm labels. Flamegraph labels use 22px text in 30px X rows and 26px LinkedIn rows. Narrow labels use ellipses, with full names and percentages retained in SVG tooltips and the source sidecars. Each image states the captured phases and limits of the comparison.

Validation: all three unit checks passed. They check event/frame retention and weight normalization, exact copied sidecars, and every text/bar bounding box within the canvas and profile bounds. Pango supplies font ink extents. Full images and 520px previews were inspected, then legends were enlarged to 44px and the final 400px and 520px previews were inspected. No chart clipping was observed. Complete function names in the legends remain readable at 400px. Full flamegraph labels do not all fit at mobile width, as disclosed by the linked full stacks.

Blockers: none for local rendering. Actual Typefully preview geometry and account state require the parent owner's browser readback.

Next step: parent owner runs the package/profile convention checks, commits and pushes the images and source, replaces draft media, and verifies one full-width square image per X post plus one 4:5 LinkedIn image.
