# Chapter 18 - Advanced Attention & Decoding

## Summary
Collects modern decoder techniques-FlexAttention, FlexDecoding, speculative and paged attention workflows-implemented in both PyTorch and CUDA/Triton so you can iterate quickly while validating kernels on real hardware.

## Problem
Chapter 18 is the "does decoder complexity actually buy you anything?" checkpoint. It puts flexible masking, speculative decoding, tensor-core kernels, and serving integration on the same chapter surface so you can see which tricks reduce latency and which ones only add engineering cost.

## Baseline Path
- straightforward FlexAttention / decode execution
- conservative serving integration without aggressive caching or graph replay
- good correctness anchor, but usually too much launch and data-movement overhead

## Optimized Path
- FlexDecoding, tensor-core-specialized kernels, and cache-aware paths
- graph replay and serving-integrated decode paths where they help
- still benchmarked through the shared harness instead of one-off scripts

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `flexdecoding` | `161.596 ms` | `81.980 ms` | `1.97x` | baseline masks the full KV cache; optimized slices to the active decode window |
| `eos_early_exit` | `11.694 ms` | `2.481 ms` | `4.71x` | optimized stops after 16/128 decode steps once all rows are EOS |
| `tensor_cores` | `3.805 ms` | `0.243 ms` | `15.65x` | tensor-core decode kernel |
| `rope_q_cache` | `106.429 ms` | `4.523 ms` | `23.53x` | cache-aware rope/Q-path reuse |

The `eos_early_exit` row is from `artifacts/runs/codex_ch18_eos_early_exit_light_20260630_214000/`. The chapter has a mix of "moderate but real" improvements and "big kernel-level" improvements. Treat those as different stories rather than averaging them together into one headline number. `flexdecoding` is the chapter-native work-reduction story: the baseline scores the full KV cache with a sliding-window mask, while the optimized path trims the decode step down to the active window before attention. Re-measure it on your hardware before treating the chapter numbers as a decision threshold.

## Profiler Evidence
Use deep-dive harness runs when you want Nsight evidence for cache reuse, launch count, and kernel selection:

```bash
python -m cli.aisp bench run --targets ch18:flexdecoding --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch18:eos_early_exit --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch18:tensor_cores --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch18:rope_q_cache --profile deep_dive --single-gpu
```

For serving integration, use the chapter-specific vLLM path only after the direct benchmark targets are clean, because the chapter harness gives you the more trustworthy baseline/optimized comparison.

## Repro Commands
```bash
python -m ch18.compare
python -m cli.aisp bench list-targets --chapter ch18
python -m cli.aisp bench run --targets ch18 --profile minimal
python -m cli.aisp bench run --targets ch18:flexdecoding --profile deep_dive --single-gpu
```

## Learning Goals
- Prototype FlexAttention/FlexDecoding workloads with custom masks, score mods, and KV-cache integration.
- Evaluate speculative decoding pipelines that trade extra compute for lower latency.
- Test tensor-core optimized attention kernels tailored for Blackwell tmem limits.
- Validate integration points with serving frameworks (vLLM) using the provided runners.
- Separate EOS polling overhead from early-exit work reduction when a batch finishes before the max decode budget.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flexdecoding.py`, `optimized_flexdecoding.py`, `optimized_flexdecoding_graphs.py`, `v1_engine_loop.py`, `v1_engine_loop_common.py` | FlexDecoding benchmarks plus a V1 polling-loop correctness tool (not a benchmark pair). |
| `baseline_eos_sync_polling.py`, `optimized_eos_sync_polling.py`, `baseline_eos_early_exit.py`, `optimized_eos_early_exit.py`, `eos_early_exit_common.py` | EOS synchronization and early-exit decode-loop comparisons, including a lightweight fixed-buffer benchmark for skipped tail decode work. |
| `baseline_paged_attn_backend.py`, `optimized_paged_attn_backend.py`, `baseline_paged_attn_layout.py`, `optimized_paged_attn_layout.py`, `paged_attn_split_common.py` | Split paged-attention comparisons: dense math-versus-flash backend selection and dense masked decode versus block-table-driven FlexAttention sparse kernels. |
| `baseline_tensor_cores.py`, `optimized_tensor_cores.py`, `flashmla_kernel.cu`, `warp_specialized_triton.py` | Tensor-core attention kernels plus Triton equivalents for rapid validation. |
| `flex_attention_native.py`, `flex_attention_enhanced.py`, `flex_attention_large_model.py`, `kv_cache_integration_example.py` | FlexAttention examples ranging from toy sizes to large models with KV-cache reuse. |
| `baseline_vllm_v1_integration.py`, `optimized_vllm_v1_integration.py`, `baseline_vllm_decode_graphs.py`, `optimized_vllm_decode_graphs.py`, `configs/`, `spec_configs/`, `workload_config.py` | Serving integrations and config presets for pushing workloads through vLLM or custom harnesses. |
| `speculative_decode/spec_config_sweep.py` | Tooling to sweep speculative-decoding configs and summarize latency/throughput tradeoffs. |
| `compare.py`, `expectations_{hardware_key}.json`, `test_flex_attention.py` | Harness entry, regression thresholds, and pytest coverage for FlexAttention APIs. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch18.compare
python -m cli.aisp bench list-targets --chapter ch18
python -m cli.aisp bench run --targets ch18 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python -m ch18.compare` runs the chapter baseline/optimized sweep through the shared harness.
- `python -m cli.aisp bench run --targets ch18:vllm_v1_integration --profile minimal` completes with accuracy parity vs the native FlexAttention path.
- `python -m pytest -q ch18/test_flex_attention.py` passes locally, confirming mask/score-mod helpers are wired correctly.
- `ch18:eos_early_exit` reports fewer decoded steps on the optimized path while preserving the generated-token verification buffer.

## Notes
- `flex_attention` scripts accept env vars like `BLOCK_SIZE`, `DOC_SPAN`, and `SEQ_LEN` so you can sweep shapes without editing code.
- `flashmla_kernel.cu` includes the Blackwell-specific tensor memory guard to keep compilation healthy on SM121 hardware.
- `paged_attn_backend` isolates SDPA backend choice on a dense layout, while `paged_attn_layout` converts a real per-batch block table into both a dense reference mask and a fused FlexAttention block-mask kernel.
