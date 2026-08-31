# Lab - MoE Optimization Journey

## Summary
Packages a staged MoE optimization story from naive execution to quantized/padded fast paths so you can measure which step is actually doing the work.

## Shared level mapping and compatibility names
| Shared level | Actual requested path |
| --- | --- |
| 0 | Naive expert loops |
| 1 | Batched expert computation |
| 2 | Fused SiLU and multiplication |
| 3 | Intermediate-buffer reuse |
| 4 | Sorted per-expert GEMMs |
| 5 | Padded batched GEMMs |
| 6 | CUDA graph replay of the expert path |
| 7 | torch.compile on the graph-friendly model |

The shared benchmark casts models to BF16. Legacy names are compatibility aliases: level2_fp8/streams/sorted/permuted request level2 fusion; level3_grouped/sorted/fp8 request level3 buffer reuse; level4_parallel requests grouped dispatch; optimized_moe_expert_parallel requests level5 BMM. None of these names enables FP8 or multi-stream/distributed execution. Runtime availability and capture/compile metrics must confirm actual kernel paths. The separate level6_native_fp8 experiment is not one of these shared wrappers. Its native E4M3 path includes activation scaling in timing, combines all routed outputs, and checks full outputs against original BF16 weights. Normal setup requires AISP_NATIVE_FP8_ACCURACY_POLICY; no accuracy or speedup is accepted without reviewed limits and actual target evidence. Collect errors with python -m labs.moe_optimization_journey.calibrate_native_fp8 --output /tmp/native-fp8-unique-attempt.json. This calibration records failures and never grants acceptance; retained CPU reference weights are part of its memory cost.

## Incomplete Triton experiments withdrawn
The incomplete standalone Triton MoE/FFN and raw grouped-GEMM experiments were withdrawn; legacy `triton_fused_moe` calls fail explicitly and emit no performance result. The active SiLU-times-up helper uses differentiable PyTorch whenever either input needs gradients, and also uses explicit PyTorch on CPU or without Triton. Only eligible CUDA inference launches the elementwise Triton kernel, including any required contiguous copies in the call. This is activation fusion, not a fused full expert FFN. Actual CUDA numerical, device, stream and memcheck acceptance remains HOLD.

## Problem
MoE optimization is often told as a narrative, not a benchmarked sequence. This lab keeps the sequence explicit so you can see which stage of the journey is providing the real win.

## Baseline Path
- naive MoE execution path
- simple correctness reference
- useful for showing how expensive unstructured expert execution can be

## Optimized Path
- staged optimized MoE path with batching/layout/scheduling improvements
- separate padded/quantized route for a more production-like fast path
- designed to attribute wins to concrete optimization steps

## Historical Delta (not requalified by this audit)
Historical results, preserved without recertifying their source/accuracy contracts, from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `moe` | `41.938 ms` | `1.217 ms` | `34.47x` |
| `moe_pad_quant` | `4.681 ms` | `1.790 ms` | `2.62x` |

These records do not establish that legacy wrappers named FP8, streams, or expert parallelism executed those techniques. Fresh correctness and profiler evidence is required before attributing or repeating these speedup claims.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/moe_optimization_journey:moe --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/moe_optimization_journey:moe_pad_quant --profile deep_dive --single-gpu
```

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/moe_optimization_journey
python -m cli.aisp bench run --targets labs/moe_optimization_journey --profile minimal
```

## Learning Goals
- Show a stepwise MoE optimization story with measured deltas instead of vague progression.
- Keep the naive path, batched path, and padded/quantized path benchmarked under one roof.
- Make it obvious which optimization stage is worth carrying forward.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_moe.py`, `baseline_moe_pad_quant.py` | Naive/reference entrypoints. |
| `level0_naive.py` through `level6_full_stack.py` | Incremental optimization stages used by the journey, including a real CUDA-graph replay stage before the compiled finale. |
| `moe_benchmark.py` | Shared benchmark harness layer for the staged MoE path. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/moe_optimization_journey
python -m cli.aisp bench run --targets labs/moe_optimization_journey --profile minimal
```
- Targets follow the `labs/moe_optimization_journey:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/moe_optimization_journey:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/moe_optimization_journey --profile minimal` should keep both the core MoE and pad/quant targets green.
- Deep-dive runs should make the kernel/layout win attributable to the staged path rather than only to end-to-end timing.
- The Level 6 CUDA-graphs entrypoint should report graph capture/replay instead of silently falling back to the Level 5 fused path.

## Notes
- This lab is a good example of how the repo should teach optimization: staged, benchmarked, and profiler-backed.
