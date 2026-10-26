# Visual brief

## One sentence

Nested graph-break support grouped the same 64 captured operations into 2
replay regions instead of 16, which produced a 4.531795× synchronized steady
speedup on the historical depth-8 B200 microbenchmark while the first-ten
startup route remained a failure.

## Three explainer jobs

- **A, mechanism:** show the same operation total grouped into 16 × 4 and
  2 × 32 regions.
- **B, numerical result:** show steady and startup results in separate panels
  with separate units and verdicts.
- **C, how to spot it:** count `cudaGraphLaunch` events within one invocation.
  Keep the drawing explicitly separate from a raw profiler screenshot.

## Cover job

Adapt the verified existing Gamma first card's single horizontal-bar comparison
to landscape and square layouts. Show the exact 407.567 and 89.935 microsecond
steady medians and the 4.531795× result. Public labels should say **Support
off** and **Support on** after the clear subtitle **Nested graph-break support:
off versus on**. Keep the internal PyTorch flag identifier in provenance only.

## Visual system

- warm white `#FBFBF8`
- dark marker `#202326`
- blue `#3478BA` for support off
- orange `#E87823` for support on
- hand-drawn marker lines, light hatching, large direct labels, and generous
  open space
- no glossy rendering, fake profiler UI, logos, avatars, or unretained claims

## Required limits

- The 4.531795× result belongs to the historical cohort.
- Fresh run 022 measured 4.541758× independently and must stay separate.
- The startup route failed.
- The workload is a synthetic compiler microbenchmark, not model serving.
- Cost, capacity, revenue, model quality, and serving latency were not measured.

