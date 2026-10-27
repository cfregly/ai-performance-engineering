# Repository instructions

## Writing

- Always edit human-facing text before sending, saving, committing, or publishing it. This includes replies, comments, docstrings, docs, UI copy, commit messages, and PR titles and bodies.
- Use plain language, concrete details, and direct sentences. Cut filler, canned openings, hype, repeated caveats, needless jargon, and invented compound labels. Explain what changed and why.
- Do not use semicolons in prose or comments. Split the sentence or rewrite it. Preserve punctuation required by code, commands, data formats, and exact quotations.
- Read the final text for clarity and accuracy. Preserve technical meaning and state testing limits plainly.

## Code

For work under `code/`, also follow [code/AGENTS.md](code/AGENTS.md), which routes to the relevant implementation and validation rules.

## Research posts

- Keep the benchmark, analysis code, tests, evidence summaries, and scientific images for a post in this repository.
- Commit and push the code and referenced images before calling a social draft ready. Verify the remote commit contains them.
- Include links to the code and images in `cfregly/ai-performance-engineering` in every related post or reply draft. Prefer commit-pinned links for the evidence used in a claim.
- Preserve completed benchmark files. Distinguish the source that executed a recorded run from a maintained successor and document any reader correction.
- When sharing profiler evidence, include the retained native Nsys/NCU captures and existing exports, with cohort labels, hashes, and download links. A graph diagram does not substitute for the captured timeline or report. Keep profile views tied to their source files and disclose graph-level versus node-level tracing.
- Public performance flamegraphs must follow profiler conventions. Width represents verified CPU or GPU time, including time-proportional samples when the sampling method supports that interpretation. Do not substitute launch counts or exported row counts. Retain the field-to-unit authority, exact worker/GPU filters, captured phases, normalization, and source hashes. Host frames in a GPU profile inherit descendant GPU time and do not measure host CPU execution. A count-based stack inventory belongs in internal capture diagnostics, not the public performance draft.
- Run the profile-convention guard as part of the package check before uploading scientific images. Review the resulting image and draft against those units and filters. Preserve original captures and mark superseded derived images clearly.

- Verify scientific images in the actual platform preview after upload. Use crop-safe single-image posts or document pages, retain readable key labels at phone width, and require the rendered-preview gate before declaring a draft ready. Declare `scientific_profiles: true` in the package manifest for any profiler package, including Nsys, NCU and Zymtrace.
