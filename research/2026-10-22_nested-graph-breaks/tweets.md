# Social copy

The exact master package-review inputs are `typefully/x.txt` and
`typefully/linkedin.txt`. Their copies under `social/` are byte-identical for
the package receipt. The draft title is **Nested graph breaks | package review
| DRAFT ONLY**. This is a new review draft, not a replacement for the original
post or the five reader-specific reply drafts.

## X

B200 microbenchmark: nested graph-break support on/off measured 4.53× steady originally and 4.54× in independent fresh022. Startup gate failed. Timing was unprofiled. Nsys/NCU show structure. Not serving.

Code: https://github.com/cfregly/ai-performance-engineering/tree/e57c5bb31ec492094b4fd9f73a615195aceab936/research/2026-10-22_nested-graph-breaks/profiles
Image: https://github.com/cfregly/ai-performance-engineering/blob/e57c5bb31ec492094b4fd9f73a615195aceab936/research/2026-10-22_nested-graph-breaks/profiles/views/nsys-first-post-warmup-invocation.png

## LinkedIn

Two B200 runs now support the same narrow compiler result.

The original retained microbenchmark measured a 4.531795× geometric-mean steady-state speedup with nested graph-break support compared with the same workload without it across four paired seeds. Independent fresh run 022 measured 4.541758× under the same depth-8 scope. These are separate results. The second does not replace or rewrite the first.

Neither run qualified startup. In the original run, the first ten compiled-call intervals were 33.2% slower with nested graph-break support on, and the original no-break startup control also failed. That cost is part of the result.

Normal unprofiled workers establish latency. Separate Nsys and NCU captures explain execution structure and add profiling overhead, so their timeline widths and single-kernel durations are not speedup measurements. The retained Nsys view shows 16 graph executions with nested graph-break support off and 2 with it on for the selected post-warmup call. NCU saved one fused Triton kernel per arm, at 6.400 µs with support off and 20.992 µs with it on. The support-on kernel does more work per launch.

This is a controlled PyTorch 2.14.1+cu130 compiler microbenchmark on one NVIDIA B200. It is not a serving, model-throughput, training, or cross-GPU claim.

Attached: retained Nsys and NCU views.

Code: https://github.com/cfregly/ai-performance-engineering/blob/e57c5bb31ec492094b4fd9f73a615195aceab936/research/2026-10-22_nested-graph-breaks/benchmark.py
Native captures and exports: https://github.com/cfregly/ai-performance-engineering/tree/e57c5bb31ec492094b4fd9f73a615195aceab936/research/2026-10-22_nested-graph-breaks/profiles
Nsys view: https://github.com/cfregly/ai-performance-engineering/blob/e57c5bb31ec492094b4fd9f73a615195aceab936/research/2026-10-22_nested-graph-breaks/profiles/views/nsys-first-post-warmup-invocation.png
NCU view: https://github.com/cfregly/ai-performance-engineering/blob/e57c5bb31ec492094b4fd9f73a615195aceab936/research/2026-10-22_nested-graph-breaks/profiles/views/ncu-single-kernel-metrics.png

## Alternate openings

1. Turning nested graph-break support on changed how the same 64 operations were grouped. Two separate B200 runs measured about 4.5× better steady latency, with a startup cost that still failed its gate.
2. The profiler explains why this B200 microbenchmark uses fewer launches. It does not measure the speedup. Separate unprofiled workers do that.
3. The useful result is narrow: fewer compiled regions improved steady latency on one B200, while startup remained a failure.

## Headlines

1. Fewer graph breaks made steady calls faster, but startup still lost
2. One compiler option grouped the same work into fewer launches
3. What the B200 profiles explain, and what they do not

## Existing reader follow-up drafts

These five drafts remain separate, unscheduled, and unchanged:

- LinkedIn portability response: https://typefully.com/?d=11481014&a=337207
- LinkedIn break-even response: https://typefully.com/?d=11481015&a=337207
- LinkedIn memory response: https://typefully.com/?d=11481016&a=337207
- LinkedIn training-default response: https://typefully.com/?d=11481017&a=337207
- X profiler-attribution response: https://typefully.com/?d=11481018&a=337207

Their retained final readbacks and guards are indexed in
`evidence/existing-typefully-drafts.json`.
