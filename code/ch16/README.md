# Chapter 16 - Production Inference Optimization

## Summary
Focuses on real-world inference services: paged attention, Flash SDP, FP8 serving, telemetry hooks, schedulers, and Blackwell-friendly load-test harnesses.

For compiled serving, use the [Chapter 14 recompilation guide](../ch14/recompilation.md)
to define warmup coverage and guard-miss policy before measuring service tail latency.

## Problem
Chapter 16 is where "serving optimization" stops being a collection of tricks and becomes a latency budget. The chapter is most useful when it proves which serving-path changes actually improve steady-state latency, scheduling efficiency, or memory behavior under the shared harness.

## Baseline Path
- straightforward serving loops with conservative attention and scheduling choices
- little or no graph capture, cache-aware staging, or backend specialization
- easier to debug, but usually too expensive for production latency targets

## Optimized Path
- Flash SDP, block-sparse attention, and scheduler-aware execution where they help
- selective graph/compilation techniques for steady-state serving paths
- the same benchmark harness contract as the rest of the repo, so the gains are comparable and reproducible

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `flash_sdp` | `0.322 ms` | `0.198 ms` | `1.63x` | Flash SDP path |
| `flashinfer_block_sparse` | `0.941 ms` | `0.239 ms` | `3.94x` | block-sparse attention path |
| `runtime_scheduler` | `112.762 ms` | `63.425 ms` | `1.78x` | scheduler/runtime coordination |

The good chapter-level read is "which serving-path changes help enough to matter?" rather than trying to average these into one generic serving number.

## Profiler Evidence
Use deep-dive runs when you want Nsight-backed evidence for backend selection and scheduling behavior:

```bash
python -m cli.aisp bench run --targets ch16:flash_sdp --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch16:flashinfer_block_sparse --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch16:runtime_scheduler --profile deep_dive --single-gpu
```

Those targets answer different questions:
- `flash_sdp`: better attention backend choice
- `flashinfer_block_sparse`: structured sparsity payoff
- `runtime_scheduler`: queueing and scheduling overhead reduction

This chapter now also includes a first-class post-training quantization target family:
- `awq_gptq_smoothquant`: reference dense serving path versus explicit AWQ, GPTQ, and SmoothQuant benchmark variants

## Repro Commands
```bash
python -m ch16.compare
python -m cli.aisp bench list-targets --chapter ch16
python -m cli.aisp bench run --targets ch16 --profile minimal
python -m cli.aisp bench run --targets ch16:flash_sdp --profile deep_dive --single-gpu
```

## Token-fair scheduling and serving traces
[fair_scheduler.py](fair_scheduler.py) adds Virtual Token Counter admission to
the scheduling material: counter lift on client re-entry, input-token charges,
and charges for output tokens actually produced. `peek()` allows memory admission
before queue mutation; the caller supplies decode feedback and completion.
`fair_scheduler` compares linear and heap selection over identical complete
admission sequences. It measures CPU policy work, without claiming LLM execution.

```bash
python -m cli.aisp bench run --targets ch16:fair_scheduler --profile minimal
python -m pytest tests/test_serving_policies.py tests/test_serving_trace.py -q
python -m cli.aisp tools serving-trace -- --trace requests.jsonl --ttft-ms 250 --tpot-ms 50 --output metrics.json
```

The CPU policy is usable without CUDA; the main benchmark CLI still skips chapters
without CUDA, and strict timing validity requires Linux. Existing runtime scheduler
and vLLM monitoring measurements keep their own event/Prometheus definitions.
[Serving trace analysis](../core/analysis/serving_trace.md) computes request SLOs from
observed timestamps and preserves failure details and partial outputs.
Source: [Virtual Token Counter](https://arxiv.org/abs/2401.00588).

## Learning Goals
- Profile large decoder workloads to spot hotspots before deploying models.
- Adopt paged attention, Flash SDP, and piecewise compilation to hit latency targets.
- Integrate FP8 quantization, symmetric memory, and cache monitoring in serving loops.
- Simulate production loads (multi-node, MoE) while validating accuracy via perplexity checks.

## Directory Layout
| Path | Description |
| --- | --- |
| `inference_optimizations_blackwell.py`, `inference_profiling.py`, `inference_server_load_test.py`, `inference_serving_multigpu.py` | Top-level orchestration scripts for profiling and load testing multi-GPU inference deployments. |
| `baseline_flash_sdp.py`, `optimized_flash_sdp.py`, `baseline_dense_attention_flash.py`, `optimized_dense_attention_flash.py`, `optimized_dense_attention_flash_blackwell_variant.py` | Attention kernels that compare naive implementations versus Flash backends, including an explicit non-canonical hardware variant for the Blackwell-tagged dense-attention path. |
| `baseline_piece_graphs.py`, `optimized_piece_graphs.py`, `baseline_regional_compilation.py`, `optimized_regional_compilation.py` | Piecewise graph capture and regional compilation for stable low-latency decode, with explicit steady-state replay metrics such as `regional_compilation.capture_ms`, `regional_compilation.graph_bucket_count`, and `regional_compilation.steady_state_only`. |
| `baseline_awq_gptq_smoothquant.py`, `optimized_awq_gptq_smoothquant_awq.py`, `optimized_awq_gptq_smoothquant_gptq.py`, `optimized_awq_gptq_smoothquant_smoothquant.py`, `awq_gptq_smoothquant_benchmarks.py` | First-class post-training quantization benchmark family comparing a dense reference MLP against explicit AWQ, GPTQ, and SmoothQuant serving-style transforms. |
| `fp8_transformer_engine.py`, `test_fp8_quantization_real.py`, `symmetric_memory_inference.py`, `multi_gpu_validation.py` | Serving-time FP8 and symmetric-memory validations to guarantee accuracy and NVLink efficiency. |
| `moe_performance_benchmark.py`, `synthetic_moe_inference_benchmark.py`, `moe_workload.py` | MoE inference harnesses that stress router placement and per-expert batching. |
| `cache_monitoring.py`, `dcgm_prometheus_exporter.py`, `scheduler.py`, `perplexity_eval.py` | Telemetry, scheduling, and accuracy utilities wired into the inference pipeline. |
| `compare.py`, `requirements.txt`, `Makefile`, `expectations_{hardware_key}.json` | Harness entry and dependencies for inference-focused verification. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch16.compare
python -m cli.aisp bench list-targets --chapter ch16
python -m cli.aisp bench run --targets ch16 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python optimized_dense_attention_flash.py --profile minimal` yields fewer page faults and improved throughput relative to the baseline script.
- `python symmetric_memory_inference.py --validate` confirms NVLink-backed KV replicas stay in sync with negligible skew.
- `python inference_server_load_test.py --duration 120` exercises the scheduler and should report stable TTFT/TPOT metrics after warm-up.
- `python -m cli.aisp bench run --targets ch16:regional_compilation --profile minimal --single-gpu` should emit `regional_compilation.capture_ms`, `regional_compilation.graph_bucket_count`, and `regional_compilation.steady_state_only=1` so replay-only timing stays auditable.

## Notes
- `dcgm_prometheus_exporter.py` emits per-GPU metrics consumable by Prometheus/Grafana without extra setup.
- `cache_monitoring.py` can be run standalone to sanity-check allocator health between runs.
- `optimized_dense_attention_flash_blackwell_variant.py` is an explicit non-canonical hardware variant of `baseline_dense_attention_flash.py`; keep it out of canonical expectation coverage unless it grows real hardware-specific behavior.
