# Social copy

The current update is Typefully draft 11497664, titled **NEW UPDATE · Oct 26 Zymtrace · Standalone X + LinkedIn**. It has six X posts with one image per illustrated post and one LinkedIn post with a four-page profile PDF. The copies under `typefully/` and `social/` match the saved final readback. All remain unscheduled. Reader-specific replies are separate drafts.

## X

With nested graph-break support, B200 steady calls were 4.53× faster than without it. An independent run measured 4.54×. Startup was slower and failed its gate. Unprofiled timing. Compiler microbenchmark.

Runnable example: https://github.com/cfregly/ai-performance-engineering/blob/e57c5bb31ec492094b4fd9f73a615195aceab936/research/2026-10-22_nested-graph-breaks/benchmark.py
Image: https://github.com/cfregly/ai-performance-engineering/blob/96bdb5940a8f67238c07238caac34ab80fe28e84/research/2026-10-22_nested-graph-breaks/image.png

---

With nested graph-break support, Nsys shows 2 graph executions instead of 16. Internal nodes are hidden. Unprofiled repeated calls measured the speedup. Profiling adds overhead. NCU used 9 replay passes. This image shows one call after warmup.

---

NCU recorded one fused Triton kernel: 6.400 µs without nested graph-break support and 20.992 µs with it. With support, each launch does more work. These kernel durations do not measure the full-call speedup.

---

Zymtrace: without nested graph-break support. The full profile is shown here. Bar width is this worker's share of captured GPU time (%). Includes setup, warmup, timing and correctness. Full stacks and source data are linked below.

---

Zymtrace: with nested graph-break support. Each worker is normalized to 100% of its captured GPU time. These profiles show GPU work attributed to launch stacks. The steady-state speedup was measured separately.

---

Run the example and inspect the captures:
Code: https://github.com/cfregly/ai-performance-engineering/blob/2a4e3fd00dadf2380a5c413af33bb46c4727fce2/research/2026-10-22_nested-graph-breaks/benchmark.py
Capture controller: https://github.com/cfregly/ai-performance-engineering/blob/2a4e3fd00dadf2380a5c413af33bb46c4727fce2/research/2026-10-22_nested-graph-breaks/zymtrace/run_capture.py
Raw events + flamegraphs: https://github.com/cfregly/ai-performance-engineering/tree/7f9e981c45d5448aeacd727eabdcb8a308023732/research/2026-10-22_nested-graph-breaks/zymtrace/captures/torch2141-b200-20261026
Image renderer + PNGs: https://github.com/cfregly/ai-performance-engineering/tree/c807ebb08d06e14575dfa51ffd8180695de2c5cf/research/2026-10-22_nested-graph-breaks/zymtrace/views

Profiling: @zymtrace. B200: @verdacloud.

## LinkedIn

Two B200 runs support the same compiler result. We now also have Zymtrace captures from both example workers.

With nested graph-break support, the depth-8 microbenchmark measured a 4.531795× geometric-mean steady-state speedup across four paired seeds. Independent fresh run 022 measured 4.541758×. The runs remain separate.

Neither run qualified startup. In the original run, the first ten compiled-call intervals were 33.2% slower with nested graph-break support, and the startup control also failed.

Nsys shows 16 graph executions without nested graph-break support and 2 with it for one call after warmup. NCU captured one fused Triton kernel per arm: 6.400 µs without support and 20.992 µs with it. With support, each launch does more work. Profiling adds overhead. The published speedup comes from separate unprofiled repeated calls.

Zymtrace 26.10.1 captured both depth-8 workers on B200, PyTorch 2.14.1+cu130 and CUDA 13.0. Both passed correctness checks. The attached flamegraphs show GPU time attributed to the code that launched it. Each worker is filtered separately, and width shows its share of captured GPU time (%). These captures include setup, warmup, timing and correctness. They do not measure host CPU execution or establish an exact CPU/GPU causal percentage.

Swipe through Nsys, NCU, and both Zymtrace profiles, with one complete profile per page.

This is a compiler microbenchmark. It does not establish a training, serving, or cross-GPU gain.

Code: https://github.com/cfregly/ai-performance-engineering/blob/2a4e3fd00dadf2380a5c413af33bb46c4727fce2/research/2026-10-22_nested-graph-breaks/benchmark.py
Nsys/NCU reports and images: https://github.com/cfregly/ai-performance-engineering/tree/e57c5bb31ec492094b4fd9f73a615195aceab936/research/2026-10-22_nested-graph-breaks/profiles
Zymtrace raw events and flamegraphs: https://github.com/cfregly/ai-performance-engineering/tree/7f9e981c45d5448aeacd727eabdcb8a308023732/research/2026-10-22_nested-graph-breaks/zymtrace/captures/torch2141-b200-20261026
Zymtrace renderer and images: https://github.com/cfregly/ai-performance-engineering/tree/5817499c29603afc0fbc1a79ac1b3f5d325f3042/research/2026-10-22_nested-graph-breaks/zymtrace/views

Profiling: @[zymtrace](urn:li:organization:105228663). B200 hardware: @[Verda](urn:li:organization:107473669).
Profiles PDF (one per page): https://github.com/cfregly/ai-performance-engineering/blob/5817499c29603afc0fbc1a79ac1b3f5d325f3042/research/2026-10-22_nested-graph-breaks/zymtrace/views/social-layout/linkedin-profiles.pdf

## Alternate openings

1. Turning nested graph-break support on changed how the same 64 operations were grouped. Two separate B200 runs measured about 4.5× better steady latency, with a startup cost that still failed its gate.
2. The profiler explains why this B200 microbenchmark uses fewer launches. It does not measure the speedup. Separate unprofiled workers do that.
3. The useful result is narrow: fewer compiled regions improved steady latency on one B200, while startup remained a failure.

## Headlines

1. Fewer graph breaks made steady calls faster, but startup still lost
2. One compiler option grouped the same work into fewer launches
3. What the B200 profiles explain, and what they do not

## Existing reader follow-up drafts

These five drafts remain separate and unscheduled. Each now includes the new
Zymtrace profile images and capture links alongside Nsys and NCU:

- LinkedIn portability response: https://typefully.com/?d=11481014&a=337207
- LinkedIn break-even response: https://typefully.com/?d=11481015&a=337207
- LinkedIn memory response: https://typefully.com/?d=11481016&a=337207
- LinkedIn training-default response: https://typefully.com/?d=11481017&a=337207
- X profiler-attribution response: https://typefully.com/?d=11481018&a=337207

Their retained final readbacks and guards are indexed in
`evidence/existing-typefully-drafts.json`.
