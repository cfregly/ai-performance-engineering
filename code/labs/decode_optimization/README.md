# Lab - Decode Optimization

## Summary
Decode-focused microbenchmarks that isolate serving-side wins such as pinned memory, streams, compile/graphs, FP8/FP4, warp specialization, and HuggingFace cache policy changes without dragging full attention stacks into every comparison.

## Problem
Decode paths die by a thousand cuts: host staging, stream orchestration, cache policy, compile overhead, and kernel schedule all matter. This lab keeps those costs as separate targets so you can see what actually moves TTFT, TPOT, and total decode latency.

## Baseline Path
- eager decode on pageable inputs and conservative cache policy
- straightforward correctness reference
- enough host and launch overhead to make serving optimizations visible

## Optimized Path
- pinned inputs and dual-stream decode variants
- `torch.compile` and CUDA Graph decode paths
- setup-time prefill-state reuse for static-prefix serving paths
- FP8/FP4 and warp-specialized kernels where the hardware supports them
- static-cache HuggingFace loop for the cache-policy pair

## Measured Delta
Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `decode` | `9.845 ms` | `2.441 ms` (`ultimate`) | `4.03x` |
| `decode_hf_cache` | `288.157 ms` | `39.843 ms` | `7.23x` |
| `decode_streams` | `27.391 ms` | `23.753 ms` | `1.15x` |
| `decode_warp_specialized` | `38.386 ms` | `14.963 ms` | `2.57x` |
| `decode_double_buffer_tma` | `0.173 ms` | `0.081 ms` | `2.14x` |
| `decode_device_resident` | `11.487 ms` | `2.340 ms` | `4.91x` |
| `decode_prefix_state_cache` | `1.761 ms` | `1.018 ms` | `1.73x` |
| `decode_candidate_logits` | `8.866 ms` | `6.335 ms` | `1.40x` |

The `decode_device_resident` row is from `artifacts/runs/codex_decode_device_resident_20260630_204000/`; the optimized path reports zero prompt/payload copies per iteration. The `decode_prefix_state_cache` row is from the B200/Zymtrace run `artifacts/runs/20260701_132419__bench__profile_none_targets_labs_decode_optimization_decode_prefix_state_cache/`; the optimized path reports zero prefill computes per iteration and reduces TTFT from about `0.91 ms` to `0.12 ms`. The `decode_candidate_logits` row is from `artifacts/runs/codex_decode_candidate_logits_forced_20260630_210000/`; the optimized path reduces the effective logits vocabulary from `131072` to `1`. This is the useful shape of the lab: some decode optimizations are huge, some are modest, and the lab keeps them separated instead of averaging them into a fake single story.

## Control Surfaces
Treat `decode_pinned` as a supplementary local-contract speed benchmark. The pair now uses a dedicated pageable-vs-pinned baseline on the same transfer-heavy workload (`host_payload_mb=512`, no stream overlap) so the pinned-memory comparison is measurable on its own. More broadly, the standalone pinned-memory stepping stones remain non-headline benchmarks, while the lab's canonical host-overhead benchmark stays on `decode_streams`, where the workload keeps that same large host payload and adds copy/compute overlap.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/decode_optimization:decode --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/decode_optimization:decode_device_resident --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/decode_optimization:decode_prefix_state_cache --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/decode_optimization:decode_candidate_logits --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/decode_optimization:decode_hf_cache --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/decode_optimization:decode_warp_specialized --profile deep_dive --single-gpu
```

Those targets cover the most useful slices: general decode orchestration, device-resident serving contracts, real decoder-loop cache policy, and the fused Triton kernel path.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/decode_optimization
python -m cli.aisp bench run --targets labs/decode_optimization --profile none
python -m cli.aisp demos labs-decode-multigpu --nproc-per-node 4 -- --iters 4 --warmup 1
```

## Learning Goals
- Contrast eager vs pinned/streamed vs compiled/graph decode paths on the same workload.
- Measure FP8/FP4 tensor-core benefits relative to FP16/BF16 baselines.
- Validate Triton warp-specialized decode kernels against Python math and harness expectations.
- Observe NVLink-C2C behavior by scaling the decode loop across available GPUs.
- Show when a prefix-cache/device-resident request path can remove recurring prompt-side H2D staging.
- Show when a static-prefix request path can reuse prefill state instead of recomputing it per decode request.
- Show when guided decoding can avoid full-vocabulary logits by scoring only legal candidates.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_decode.py`, `optimized_decode_pinned.py`, `optimized_decode_streams.py`, `optimized_decode_compile.py`, `optimized_decode_graph.py`, `optimized_decode_graph_full.py`, `optimized_decode_ultimate.py` | Serving-path decode variants that isolate host, stream, compile, and graph effects. |
| `baseline_decode_device_resident.py`, `optimized_decode_device_resident.py` | Prefix-cache-style serving variant that seeds prompt-side inputs once and skips recurring H2D staging in the decode hot path. |
| `baseline_decode_prefix_state_cache.py`, `optimized_decode_prefix_state_cache.py` | Static-prefix serving variant that keeps prompt tensors resident and reuses the setup-time prefill state during short decode. |
| `baseline_decode_candidate_logits.py`, `optimized_decode_candidate_logits.py` | Guided/constrained decode variant that compares full-vocabulary scoring plus candidate filtering with direct candidate-only projection. |
| `baseline_decode_hf_cache.py`, `optimized_decode_hf_cache.py` | Real HuggingFace decoder-loop comparison: dynamic cache + per-step EOS sync vs static cache + compiled decode + batched EOS polling. |
| `baseline_decode_fp8.py`, `optimized_decode_fp8.py`, `baseline_decode_fp4.py`, `optimized_decode_fp4.py` | Prefill-focused low-precision decode comparisons on hardware that supports them, including the intentional BF16/nn.Linear versus FP8/Transformer Engine TELinear path. |
| `baseline_decode_warp_specialized.py`, `optimized_decode_warp_specialized.py` | Warp-specialized decode path plus its eager correctness reference. |
| `baseline_decode_double_buffer_tma.py`, `optimized_decode_double_buffer_tma.py`, `decode_common.py`, `decode_multigpu_demo.py` | CUDA double-buffer/TMA path, shared helpers, and the multi-GPU NVLink-C2C demo. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/decode_optimization
python -m cli.aisp bench run --targets labs/decode_optimization --profile minimal
```
- Targets follow the `labs/decode_optimization:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/decode_optimization:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- Baseline vs pinned/streams shows improved TTFT and TPOT with lower host wait time.
- Compile/graph variants emit fewer kernels and higher tokens/sec than the baseline in harness output.
- FP8/FP4 runs use a prefill-focused workload (`decode_tokens=0`) to surface tensor-core benefits; outputs remain within tolerance.
- Warp-specialized Triton kernel is validated against a workload-matched eager baseline; the expectation file stays green.
- `decode_device_resident` emits zero prompt/payload copies per iteration on the optimized path while preserving the same model output.
- `decode_prefix_state_cache` emits zero prefill computes per iteration on the optimized path while preserving the same short-decode output.
- `decode_candidate_logits` emits the same constrained-token decode output while reducing the effective logits vocabulary from the full vocabulary to the candidate set.
- The multi-GPU demo exercises NVLink-C2C without graph-capture failures when launched via `torchrun`.

## Notes
- All targets emit TTFT, TPOT mean, decode time, total time, and tokens/sec in `custom_metrics` for easy diffing.
- `decode_pinned` is a supplementary local-contract stepping-stone target that now isolates pageable vs pinned staging on the same large host payload; use `decode_streams` when you want the lab's canonical pinned-host plus overlap speed claim.
- `decode_device_resident` is a larger serving-contract optimization: it applies when routing and prefix-cache policy keep prompt-side buffers resident on the GPU between decode iterations.
- `decode_prefix_state_cache` is a larger serving-contract optimization: it applies when a stable prompt prefix can be prefetched once and reused for low-latency short decode requests.
- `decode_candidate_logits` models grammar, schema, or router-constrained serving where the legal next-token set is known before the lm_head projection.
- FP4 requires NVFP4-capable Blackwell hardware; unsupported platforms fail fast.
- The HF cache pair reproduces the main idea from Chaim Rand's token-generation optimization write-up while keeping the harness contract intact.
- `decode_fp8` is intentionally a BF16/`nn.Linear` baseline versus FP8/Transformer Engine `TELinear`, because Transformer Engine is the supported FP8 linear path in this lab.
