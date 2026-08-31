# Prefill and decode complete-output follow-up

CPU/source validation passes; actual CUDA execution remains **HOLD**. This integration discovery is recorded separately from the original 128 audit IDs.

All four wrappers now snapshot every decode element and every prefill destination element into preallocated payload storage. The actual prefill source is part of the input payload. Independent full decode validation and exact prefill-copy validation execute outside timing. Both peers now perform copy-only prefill; the baseline previously added the source again. FULL graph replay refreshes q/k/v and both optimized paths wait for caller-stream input production before scheduling side-stream work.

The preserved old methods reproduce the missing-coverage defect on actual CPU tensors: a [3,11,16] decode result becomes a [1,8,16] payload, and corrupting the final decode and prefill elements neither changes that payload nor fails the old validator. The preserved baseline copy method produces 2*source. These are CPU payload/copy controls, not CUDA execution or simulated GPU evidence.

- New focused controls: **32 passed, 7 CUDA skips**.
- Combined prefill, attention, shared-wrapper and decode-config tests: **102 passed, 33 CUDA skips** (4.43 s); exact commands and JUnit results are retained.
- Actual standalone capability preflight: **HOLD**, exit 3, zero CUDA checks.
- All six edited Python files compile. AST comparisons preserve the four decode math helpers, descriptor-extension loader and existing decode validator exactly. Scoped diff check passes.
- Independent parent review accepted the source with the inherited tolerance limitation. Agent A's whole hygiene suite reports 552 passed and one Linux-only skip; its receipt hashes match the five production files here. Source assertions are not numerical CUDA proof.

Seven bounded real CUDA cases are prepared: the two baselines, optimized native copy, and FULL, PIECEWISE, FULL_AND_PIECEWISE-full and FULL_AND_PIECEWISE-fallback descriptor paths. Each case uses three changed-input iterations on a nondefault caller stream, real producer delay without a prelaunch synchronization, full retained outputs, independent references and final-element negative controls. These cases have not executed locally. Their fixed FP32 dyadic inputs support exact comparison but do not calibrate arbitrary-data or lower-precision numerical accuracy.

The inherited decode budget remains rtol=0.1/atol=1.0 and is **not newly calibrated**. Target CUDA/toolchain builds, actual graph/copy/stream execution, numerical-budget calibration, sanitizer checks and fresh timing remain required. Historical reports are preserved and do not establish a speedup for the repaired workload. Root updated the README and generator; this slice did not modify them.
