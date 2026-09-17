# B200 NVFP4 results

The full K64 port runs on B200 and shows a small lead over cuBLASLt at 8192³.
The profiled harness measured **1.032×**. Five alternating timing rounds had a
**1.030× median ratio**, with one round essentially tied. These are exploratory
results from a virtualized host, below the repository's unchanged 1.05× speed-goal
threshold. Full-output correctness passed.

The port adapts r5 from Pranjal Shankhdhar's
[Outperforming cuBLAS on NVFP4](https://cudaforfun.substack.com/p/outperforming-cublas-on-nvfp4).
It is based on fast.cu revision `2dfe5e26aecfd9e5f27bf9d5837deea01acda24b`.
The tested code is commit `21ba7efa47a90daa3b8cede27169f3aea62861e9`.
[Structured results](sm100_validation.json) contain source hashes, raw round times,
profiler summaries, and artifact digests. Tests ran on 2026-09-17 UTC.

## What changed for SM100

B200 needs K64 MMAs in place of SM103's K96 instructions. The port changes the
operand descriptors, scale layout, TMA transfer sizes, and barrier byte counts
together. A K512 group contains eight K64 MMAs, two A/B windows, and four scale
slots. TMA zero fill covers the final partial group.

The kernel keeps r5's two-CTA pipeline, separate operand and scale queues, two
TMEM accumulator buffers, and cache-hinted 256-bit FP16 stores. A ragged output
row exposed an unaligned 128-bit fallback store during testing. The port now
checks the destination alignment before using that store.

The original SM103 r0-r9 examples remain available. The B200 port does not include
r9's topology-specific scheduler. See the [optimization guide](optimization_guide.md)
for the mapping to chapters and other labs.

## Correctness and compatibility

| Check | Result |
| --- | --- |
| Local fast.cu tests | 66 passed, 12 GPU checks skipped |
| B200 small and partial-tile tests | 9 passed, full-size test skipped |
| B200 8192³ test | 1 passed |
| Standalone driver at 8192³ | All 67,108,864 output elements bitwise equal to cuBLASLt |
| Compute Sanitizer on 129×257×1136 and 259×513×160 | 0 errors |
| Benchmark lint | 10 files, 0 errors, 0 warnings |
| Original SM103 r0 and r9 with the updated adapter | CUDA 13.1.80 compile, link, and import passed |

Small tests compare against an independent CPU FP32 calculation. They check
poisoned output overwrite, prefix/suffix guards, deterministic repeated launches,
and a nondefault CUDA stream. Both benchmark arms use identical packed E2M1
inputs, UE4M3 VEC16 scale buffers, and full FP16 outputs.

The B200 builds used CUDA 13.0.88, driver 580.173.02, PyTorch 2.9.1+cu130, and
cuBLAS 13.0.0.19. No SM103 GPU was available for runtime testing. All original
r0-r9 headers remain unchanged from the earlier integration.

## Measurements

The run used one B200 with application clocks set to 1965 MHz SM and 3996 MHz
memory through the harness. No foreign workload overlapped the accepted batch.
The application clocks were restored, and persistence mode was returned to its
original disabled state after reconciling the earlier attempts.

| Measurement | cuBLASLt | B200 port | Baseline / port |
| --- | ---: | ---: | ---: |
| Harness reported latency | 197.423 µs | 191.392 µs | 1.032× |
| Five graph rounds, median latency | 220.702 µs | 214.343 µs | 1.030× |
| Nsight Systems kernel time | 178.976 µs | 173.184 µs | 1.033× |
| Nsight Compute kernel time | 179.550 µs | 176.480 µs | 1.017× |

The baseline uses the first usable cuBLASLt heuristic, matching the inherited
driver policy. The harness used 5 warmups and 20 iterations under the strict validity profile.
Both arms passed input/output verification and captured Nsight Systems, Nsight
Compute, and PyTorch profiles. The harness returned `failed_no_speedup` because
the measured ratio missed 1.05×. This was an acceptance-threshold failure.

The separate graph experiment used 10 warmups and 1,000 replays per arm in each
of five rounds, alternating arm order. The paired ratios were 0.9994, 1.0167,
1.0316, 1.0299, and 1.0298. Their median was 1.0298 and sample standard deviation
was 0.0137. No round was discarded. CUDA-event graph timings include the spacing
between host-issued replays, so they are distinct from isolated kernel times.

Nsight Compute reported SM throughput of 85.94% for cuBLASLt and 86.59% for the
port, with DRAM throughput of 14.82% and 14.38%. This does not suggest a simple
HBM-bandwidth bottleneck. It does not isolate the benefit of each pipeline change.

These measurements reuse one resident operand set. They do not reproduce the
article's rotating-input protocol or establish results for other shapes. The
last idle SM-clock read was 1312 MHz despite unchanged application clocks, so
continuous active-clock residency is not established. Earlier queue attempts
were excluded because process ownership was ambiguous. Raw logs and profiler
files are retained outside this public repository and identified by digest in
the structured results.

## Reproduce

From `code/`, select an idle B200 with `CUDA_VISIBLE_DEVICES` if needed:

```bash
AISP_RUN_FAST_CU_NVFP4_SM100_GPU_TEST=1 python -m pytest tests/test_fast_cu_nvfp4_sm100.py -q
AISP_RUN_FAST_CU_NVFP4_SM100_FULL_GPU_TEST=1 python -m pytest tests/test_fast_cu_nvfp4_sm100.py -q -k full
python -m cli.aisp bench run --targets labs/fast_cu:nvfp4_sm100 --profile deep_dive --single-gpu --iterations 20 --warmup 5
PYTHONPATH=. python labs/fast_cu/checks/measure_nvfp4_sm100.py --output /tmp/nvfp4-sm100-paired.json
PYTHONPATH=. compute-sanitizer --tool memcheck --error-exitcode 99 python -c 'from labs.fast_cu.nvfp4_sm100 import load_sm100_extension; m = load_sm100_extension(); m.validate_against_host(129, 257, 1136, 20260916); m.validate_against_host(259, 513, 160, 20260917)'
```

The small GPU suite includes K values 32, 64, 96, 128, 160, 1136, and 1152.
The native setup also runs the inherited host-reference gates before timing.
