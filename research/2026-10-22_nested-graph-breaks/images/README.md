# Nested graph-break support visual package

This directory contains three conceptually different explainers, one cover,
and six audience-specific visuals for the historical B200
`nested_break_depth8` result. Every family has a 1600×900 landscape export, a
1080×1080 square export, a 400×225 phone-width preview, a 400×400 square
preview, a visible-copy sidecar, and structured prompt and provenance data.

The public labels use **support off** and **support on**. The exact PyTorch flag
identifier stays in the JSON provenance rather than the drawings. Both arms run
the same workload with explicit graph breaks. Support changes how those breaks
are compiled and grouped. It does not remove the graph breaks from the
workload.

## Evidence boundary

All claim-bearing visuals use the original historical cohort:

- one NVIDIA B200
- PyTorch 2.14.1+cu130
- synthetic depth-8 workload with 1,048,576 FP32 elements
- support off: 16 emitted regions, 16 replay launches per profiled invocation,
  and 407.567 µs synchronized steady median
- support on: 2 emitted regions, 2 replay launches per profiled invocation,
  and 89.935 µs synchronized steady median
- same 64 captured ATen operations in both arms
- 4.531795× steady geometric-mean speedup across four paired seeds
- first ten compiled-call intervals: 2411.177 ms with support off and
  3211.677 ms with support on
- support on took 33.2% longer over the first ten intervals, so the startup
  route remains **FAIL**

Fresh run 022 measured a separate 4.541758× steady result. It is an independent
capture cohort and is not the source of the 4.531795× result shown here. Native
profiler images are retained elsewhere in the package and are not substituted
for audience artwork.

## Visual families

| Stem | Job | Main visual idea |
| --- | --- | --- |
| [`a`](a.png) | Mechanism | 16 regions × 4 operations become 2 regions × 32 operations while the total stays 64. |
| [`b`](b.png) | Numerical result | Separate panels show the passing steady result and the failing startup result on their own scales. |
| [`c`](c.png) | How to spot it | A trace-like diagram shows 16 versus 2 `cudaGraphLaunch` events inside one invocation. It is explicitly not a profiler screenshot. |
| [`fun`](fun.png) | Cover | The Gamma-origin bar-chart concept shows the exact synchronized steady medians with plain support labels. |
| [`aud-curious`](aud-curious.png) | Curious non-specialist | The same work moves in 16 small trips or 2 larger trips. |
| [`aud-ml-engineer`](aud-ml-engineer.png) | ML/AI engineer | Compiler shape changes from 16 regions and launches to 2 while workload code stays fixed. |
| [`aud-performance-engineer`](aud-performance-engineer.png) | Performance and capacity engineer | A four-seed dot plot shows the range and descriptive geometric mean. |
| [`aud-kernel-engineer`](aud-kernel-engineer.png) | CUDA and hardware engineer | Launch count falls while the captured operation total stays fixed. |
| [`aud-platform-operations`](aud-platform-operations.png) | Platform operations | Warmup, steady measurement, and cold-path monitoring stay separate. |
| [`aud-business-capacity`](aud-business-capacity.png) | Business and capacity reader | Measured compiler-loop results are separated from unmeasured serving, cost, and capacity outcomes. |

The audience rationales and alt text are in [audiences.md](audiences.md).

## Cover origin

The cover comes from the verified existing Gamma document
[`4.53× faster at steady state`](https://gamma.app/docs/453-faster-at-steady-state-u5q8j0ibrd2qhu1?mode=doc).
Its immutable three-card PNG export is retained at
[`sources/gamma-453-faster-at-steady-state-original.zip`](sources/gamma-453-faster-at-steady-state-original.zip)
with SHA-256
`89aeb6360afbb8ce289896d8530d5927bb0caf78b78613db947cdb57794d5f63`.
The first source card is retained unchanged at
[`sources/gamma-original-export/1_453-faster-at-steady-state.png`](sources/gamma-original-export/1_453-faster-at-steady-state.png)
with SHA-256
`d43959205955cc07b463d05f5ad2805efeb16e9d275202b6a5b575f4eb375be6`.

The source card is portrait and uses rounded numbers plus contextless
`False`/`True` labels. The shipping `fun` family is therefore a deterministic
landscape and square scientific-chart adaptation. It preserves the card's
single steady-latency comparison, horizontal-bar concept, and pixel-sampled
palette. It replaces the ambiguous labels with **Support off** and **Support
on** and reads the exact values from retained JSON evidence. It is Gamma-origin
but is not represented as a direct Gamma bitmap export. The complete transform
and output hashes are in
[`gamma-cover-derivation.json`](gamma-cover-derivation.json); the stage boundary
is in [`gamma-cover-status.json`](gamma-cover-status.json).

The built-in image generator produced one text-free candidate during recovery.
It drew 15 small tiles instead of the required 16, so it was rejected. The
unused candidate remains in
[`sources/unused-imagegen-cover-candidate.png`](sources/unused-imagegen-cover-candidate.png)
with its rejection reason in [visual-provenance.json](visual-provenance.json).

## Index and provenance

- [Landscape contact sheet](contact-sheet.png)
- [Square contact sheet](contact-sheet-square.png)
- [HTML gallery](gallery.html)
- [Visual provenance](visual-provenance.json)
- [Render receipt](render-receipt.json)
- [Review inventory](reviewable-inventory.json)
- [Image hashes](final-image-hashes.txt)
- [Filled cover prompt](fun-prompt.txt)
- [Gamma cover derivation receipt](gamma-cover-derivation.json)

The renderer is [render_whiteboard_visuals.py](render_whiteboard_visuals.py).
It reads retained public JSON files, asserts every displayed result, and does
not execute the workload. Run it with the Codex bundled Python environment:

```text
/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  images/render_whiteboard_visuals.py
```

## Review state

Rendering and mechanical evidence assertions pass. Independent visual review
is still required after the final cover bytes are in place. The review must
inspect every full-size, square, and native 400 px export for exact copy,
clipping, overlap, readable hierarchy, and distinct audience composition.

