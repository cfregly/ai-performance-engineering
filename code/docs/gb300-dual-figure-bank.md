# GB300: dual-figure bank for the frame-bound labs (Front A2, B64 named lever)

**Verdict.** Six graphed-mode dual figures are now banked (frame-mode + pure-GPU,
capture-fidelity-audited) covering all remaining frame-bound labs including the two
capstone targets; every default-mode run PASSED verification inside (or faster than)
its banked band, zero regressions. Two loud refusals, both root-caused: (1)
software_pipelining initially refused with a **NEW B45-class instance (#4): an
EXPLICIT-but-DEFAULT stream** — `at::cuda::getDefaultCUDAStream()` passed as the launch
stream is invisible to side-stream capture exactly like a raw `<<<>>>` legacy launch,
and the B64 parser audit classified it "explicit" (it only flagged missing/zero stream
args); a 1-line `getCurrentCUDAStream()` fix (B62 precedent) unlocked the lab's dual
figure, and the graphed warm figure **60.73 us reproduces B41's banked 60.5 us kernel**
end-to-end. (2) moe_cuda's optimized arm refuses by design (its B45 one-graph dispatch
cannot nest inside an outer capture — same class as block_scaling). Headroom result:
with honest denominators, **metric_reduction_vectorized now has the largest TRUE
kernel-side gap** (9.02 us pure-GPU vs a 3.15 us single-read HBM floor, 2.9x), while
the tcgen05 GEMMs are ~1.25x off a structurally-capped floor and software_pipelining
is closed.

Hardware: GB300 (cc 10.3), GPU3 quiet (foreign-process precheck per run, flock'd,
no idle holds), pod `<gb300-pod>`, pod repo md5-reconciled against local HEAD
`41f50dd9` (12/12 files matched) before any run. Evidence:
`/tmp/frontA2/a2_01..a2_16.{cmd,stdout,stderr,exit,precheck}` + `a2_summary.json`
+ `ALL_DONE` on the pod. Default-mode harness flags: `--profile none --single-gpu`.

## 1. THE DUAL-FIGURE TABLE (the bankable artifact)

Pure-GPU = graphed-mode optimized arm (cold = L2-cleared interleaved subtraction,
warm = exact whole-graph replay / N). Frame share = `frame_overhead_ms_median /
frame_iter_ms_median` from the same graphed run. Fidelity = `graphed_eager_kernel_count`
(eager kernels/iter == replayed kernels/iter, enforced by the harness audit).

| Lab : target | Banked frame figure (band cited) | This-session frame (default mode) | Pure-GPU cold / warm | Frame share | Fidelity | Verdict |
|---|---|---|---|---|---|---|
| fullstack_cluster : cluster_gemm_tcgen05 | 0.0887–0.0972 ms (B58/Y2 harness x3, gb300-capstone-f-decomposition.md) | 0.0903 ms / 1.113x PASS (a2_07) | **17.87 / 15.58 us** (a2_08) | 79.0% | 1 | DUAL FIGURE BANKED; warm = B61 kernel 14.98 us + ~0.6 us epilogue store + wrapper-in-capture |
| fullstack_cluster : cluster_gemm_tcgen05_cta2 | 0.0868–0.0908 ms (same Y2 table) | 0.0892 ms / 1.159x PASS (a2_09) | **17.89 / 15.54 us** (a2_10) | 78.7% | 1 | DUAL FIGURE BANKED; 2SM pair lands at the same pure-GPU figure as 1SM |
| blackwell_matmul : blackwell_matmul_tcgen05 | 0.104–0.163 ms (B58 harness x3); 0.109–0.113 ms (B62) | 0.1071 ms / 146.5x PASS (a2_05) | **16.78 / 15.04 us** (a2_06) | 83.2% | 1 | DUAL FIGURE BANKED; warm reproduces the B62 demo (15.04) and B58's 15.2 us kernel |
| training_hotpath : metric_reduction_vectorized | 0.0565–0.0578 ms (B50); 0.0556→0.0567 (B62 regression pair) | 0.0561 ms / 188.8x PASS (a2_01) | **11.64 / 9.02 us** (a2_02) | 78.5% | 2 | DUAL FIGURE BANKED; warm reproduces B62's 9.04 / B50's 8.16+fill |
| training_hotpath : metric_reduction_cuda | 0.0558 ms / 8.83x (B62); B50 expectation 0.0591 / 8.35x | 0.0497 ms / 9.95x PASS (a2_03; graphed sibling run 0.0569 / 8.61x in-band) | **7.70 / 5.87 us** (a2_04) | 85.9% | 2 | DUAL FIGURE BANKED (first pure-GPU figure for this target); default run sits 11% FASTER than band — positive-direction frame-median jitter, not a regression |
| software_pipelining : tile_pipeline | 0.123–0.126 ms (B41 harness) | 0.1246 ms / 2.57x PASS (a2_13; pre-fix a2_11 0.1272 / 2.47x) | **63.63 / 60.73 us** (a2_14) | 45.8% | 1 | DUAL FIGURE BANKED after 1-line stream fix; warm reproduces B41's banked 60.5 us kernel |
| moe_cuda : router_vectorized | 0.23491 ms (B59; lifetime ~35x) | 0.2352 ms / 401.7x PASS (a2_15) | optimized: **LOUD REFUSAL** (expected class); baseline graphed 23.46 ms warm, 49 kernels | baseline 1.85%; optimized est ~10–15% (B62) | 49 (baseline) | DEFAULT FIGURE RE-CONFIRMED; optimized arm structurally uncapturable (see notes) |

All 8 default-mode runs: verification PASS, `failed_verification: 0`. All graphed runs
that completed: verification PASS (the post-replay eager-restore contract held).

## 2. Refusals (loud, recorded — both are results)

- **a2_12 software_pipelining graphed (exit 1), pre-fix**: `CAPTURE-FIDELITY VIOLATION —
  one eager benchmark_fn() launches 1 kernel(s), but the captured 50-iteration graph
  replays 0 (expected 50)`. Root cause `labs/software_pipelining/software_pipelining_kernels.cu:402`:
  `const cudaStream_t stream = at::cuda::getDefaultCUDAStream();` — an explicit stream
  argument, so B64's "45/45 labs chevron launches already explicit" sweep passed it, but
  the default stream is exactly what side-stream capture cannot see. **B45-class instance
  #4, new sub-class: explicit-but-default.** Fixed to `getCurrentCUDAStream()` (eager
  behavior identical — current == default outside capture); default-mode regression
  re-run PASS in-band (a2_13), graphed dual figure unlocked (a2_14) and lands on B41's
  banked kernel number.
- **a2_16 moe_cuda graphed (exit 1), optimized arm**: `CUDA graph capture of benchmark_fn
  FAILED ... cudaErrorStreamCaptureUnsupported / cudaErrorStreamCaptureInvalidated` —
  the optimized arm replays its own internal one-graph dispatch (B45) and
  `cudaGraphLaunch` is not permitted inside an outer stream capture. Same class as
  block_scaling's banked refusal (B62); no silent fallback, default mode unaffected.
  The BASELINE arm graphed cleanly first (23.46 ms warm vs 23.90 ms frame, 49
  kernels/iter, 1.85% frame) — confirming moe_cuda numbers are ~all GPU already.
- block_scaling was NOT re-run: its refusal (internal-graph replay) is already banked in
  B62 and its mechanism is identical to a2_16's.

## 3. Remaining explicit-but-default sites (repo grep, unfixed)

The new sub-class exists in two more labs (in-process torch extensions, capture-eligible
= DANGEROUS-LATENT by B64's own taxonomy):
- `labs/persistent_decode/tma_extension.py:84` and `persistent_decode_ext.cu:94`
- `labs/memory_bandwidth_patterns/bandwidth_patterns_kernels.cu:187,205,232,245,261`

These should get the same 1-line fix + the B64 smoke (eager + capture + kernel-count).

## 4. Per-lab notes: what the pure-GPU figure reveals (true headroom vs B61 real ceilings)

Denominators: dense FP16 ~1.9 PF effective (B61 reframing; cuBLAS asymptote), HBM 8 TB/s.

| Rank | Lab | Pure-GPU warm | True floor | Gap | Status |
|---|---|---|---|---|---|
| **1** | **metric_reduction_vectorized** | **9.02 us** (7.91 kernel + ~1.1 fill) | 25.2 MB single-read / 8 TB/s = **3.15 us** | **2.9x (~5.9 us)** | ADDRESSABLE — kernel runs at 3.19 TB/s = ~40% HBM-SoL (B50 ncu basis) |
| 2 | metric_reduction_cuda | 5.87 us (2 kernels) | latency-bound (kernel DRAM 6.3%, sol-roofline-labs3); ~1–2 us graphed 2-kernel floor | ~3x relative, ~4 us absolute | overhead-bound math kernel; small absolute EV |
| 3 | blackwell_matmul_tcgen05 | 15.04 us | ~12.1 us (B61 bit-identity floor, 128 tiles/128 SMs) | 1.24x (~2.9 us) | mainloop CLOSED at ~87% real SoL (B61); residual = epilogue store + capture-visible wrapper; beyond is CONTRACT-level (split-K, shapes) |
| 3= | cluster_gemm_tcgen05 / _cta2 | 15.58 / 15.54 us | same ~12.1 us | 1.29x (~3.5 us) | same kernel + ~0.5 us more wrapper inside capture than the blackwell target — the ~85–100 us host dispatch (B55) is now PROVEN to be 79% frame, not kernel |
| 5 | software_pipelining tile_pipeline | 60.73 us | 50.3 us naive (403 MB / 8 TB/s, B41); practical parity ceiling 86.8–87.1% (torch.add, B41) ≈ 58 us | 1.05x vs practical | CLOSED (B41 verdict re-confirmed end-to-end by an independent ruler) |
| 6 | moe_cuda router_vectorized | ~0.235 ms frame ≈ GPU | GEMM-dominated; B59 floor = 20% residual padding | — | not graphable; next lever stays B59's sub-64 capacity rounding |

The headline reframing this table buys: **the capstone "~85–100 us host dispatch" (B55)
is now measured at 79% frame share with a 15.5 us pure-GPU residual** — i.e. the wrapper
host residual is harness-frame, not workload, and the capstone kernel story is closed at
B61's ~87% real SoL. The lab where a kernel win is still both real and measurable is
metric_reduction_vectorized.

## 5. Named next lever (largest true kernel-side gap)

**metric_reduction_vectorized single-launch + float4 lever (B50-named, now measurable):**
drop the `torch::zeros` fill (partials + last-block-final reduction on `torch::empty`)
and float4 loads with smem partial fold — B50 estimates kernel 7.9 → ~4–5 us at
~5–6 TB/s. In frame mode this win is invisible (~8% headline, inside jitter); against
the now-banked 9.02 us warm figure it is a measurable ~1.8–2x pure-GPU win with the
graphed dual figure as the contract. Gate any such change on: default-mode band
0.0556–0.0578 ms intact + graphed fidelity count staying 2 → 1 (the launch drop is
itself the proof).

Secondary: fix the two remaining explicit-but-default labs (section 3) before any
capture-based work touches them.

## 6. Files touched

- `labs/software_pipelining/software_pipelining_kernels.cu` — md5
  `a8cfcf000408c7053259f23dc0e99d7c` → `1b28e65095cc1547d8026902387dc4dc` (1-line
  stream fix + comment; local == pod verified). Default-mode regression: a2_13 PASS
  in-band, baseline arm untouched semantics (same stream in eager).
- `docs/gb300-dual-figure-bank.md` — this report.

## 7. Caveats

- The cold figures carry the ~1 us cross-graph-subtraction error band (B62 caveat);
  warm figures are exact and are the comparison-grade numbers above.
- Frame-mode medians of the microsecond labs jitter run-to-run by up to ~15% within a
  session (a2_03 0.0497 vs a2_04 0.0545 same target, same session, std ~0.009 ms);
  band checks above use the result-JSON `time_ms` and were evaluated per-run.
- moe_cuda's optimized frame share (~10–15%) remains an estimate (B62 frame model:
  ~22 us bracket + 1 graph launch on a 235 us workload); it cannot be measured by this
  mode by construction.
