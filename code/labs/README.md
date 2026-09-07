# Labs

## Summary
Labs are where the repo stops being chapter-by-chapter pedagogy and starts telling complete optimization stories.
Some labs are strict baseline/optimized benchmark pairs. Others are playbooks or matrix harnesses that need a different, more honest doc shape.

## How To Read This Directory
There are two useful lab classes in this repo:

- **Benchmark-pair labs**: these expose harness targets, keep correctness gates, and are the right place to make performance claims.
- **Playbook / matrix labs**: these package workflows, scenario drills, or large tuning matrices. They are still valuable, but they should not pretend to be a single baseline/optimized benchmark when they are not.

## What Counts As A Good Lab Here
A strong public lab should make three things obvious:

- what the baseline path is
- what changed in the optimized or alternative path
- what measured artifact proves the claim

If a lab cannot answer those questions yet, the doc should say so directly instead of faking a benchmark pair.

## Learning Goals
- Help readers find the right lab quickly.
- Separate benchmark-pair labs from playbook/matrix labs honestly.
- Point contributors toward the repo's expected lab quality bar.

## Directory Layout
| Path | Description |
| --- | --- |
| `labs/block_scaling`, `labs/blackwell_matmul`, `labs/blackwell_gemm_optimizations`, `labs/flashattention4`, `labs/memory_bandwidth_patterns`, `labs/nccl_nixl_nvshmem`, `labs/persistent_decode`, `labs/parameterized_cuda_graphs`, `labs/training_hotpath` | Benchmark-pair labs with strong kernel/perf narratives. Measured claims require retained artifacts; the FlashAttention-4 Colfax diary pairs are source-grounded studies whose local target-GPU qualification is pending. |
| `labs/decode_optimization`, `labs/kv_optimization`, `labs/moe_cuda`, `labs/moe_optimization_journey` | Serving-path and MoE labs where the benchmark pair is part of a broader optimization story. |
| `labs/moe_decode_blackwell_matrix`, `labs/nanochat_fullstack`, `labs/python_concurrency`, `labs/vllm-deepseek-tuning` | Larger workflow-oriented and matrix/playbook labs that need a richer doc model than a simple pair benchmark. |
| `labs/nvfp4_*` | Low-precision kernel labs where verification discipline matters as much as the timing win. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/block_scaling
python -m cli.aisp bench list-targets --chapter labs/decode_optimization
python -m cli.aisp bench list-targets --chapter labs/memory_bandwidth_patterns
python -m cli.aisp bench list-targets --chapter labs/nccl_nixl_nvshmem
python -m cli.aisp bench list-targets --chapter labs/moe_cuda
```
- Use `list-targets` first; the benchmark-pair labs expose clean harness targets, while the playbook/matrix labs often have their own scripts or Makefiles.
- Use `labs/memory_bandwidth_patterns` when you want a narrow coalescing/vectorization/shared-memory bandwidth lab instead of a broader serving-path workflow.
- Use `labs/nccl_nixl_nvshmem` when you want communication-stack tradeoffs and capability probing instead of a broader distributed-training or serving workflow.
- If a lab does not have a clean baseline/optimized target yet, do not invent one in documentation.

## Validation Checklist
- Benchmark-facing labs should expose reproducible harness targets or clearly document why they are still workflow/matrix labs.
- Public lab READMEs should prefer measured artifact-backed claims over generic feature descriptions.

## Lab Index

| Lab | Summary | Suggested Chapters |
| --- | --- | --- |
| `labs/nvfp4_gemv/` | GPUMODE `nvfp4_gemv` challenge workspace | ch06, ch10 |
| `labs/nvfp4_gemm/` | GPUMODE `nvfp4_gemm` challenge workspace | ch06, ch09, ch10 |
| `labs/async_input_pipeline/` | Async CPU->GPU input overlap | ch02, ch05, ch11 |
| `labs/block_scaling/` | Blackwell hardware-supported block scaling with direct CUTLASS vs PyTorch microbenchmarks | ch06, ch09 |
| `labs/blackwell_gemm_optimizations/` | Blackwell grouped/MoE GEMM optimization journey | ch10, ch15 |
| `labs/blackwell_matmul/` | Matmul suite focused on Blackwell | ch06, ch09, ch10 |
| `labs/cudnn_sdpa_bench/` | cuDNN SDPA benchmarking | ch10, ch18 |
| `labs/custom_vs_cublas/` | Custom kernel vs cuBLAS parity | ch06, ch09 |
| `labs/cache_aware_disagg_inference/` | Cache-aware disaggregated inference scheduling lab | ch17, ch19 |
| `labs/cutlass_profiler_kernel_selector/` | CUTLASS profiler-based kernel selection | ch06, ch09 |
| `labs/decode_optimization/` | Decoder hot-path optimization | ch18, ch19 |
| `labs/dynamic_router/` | Dynamic prefill/decode routing | ch17, ch19 |
| `labs/flashattention4/` | FlashAttention-4 pipeline co-design, including a [Colfax diary guide](flashattention4/colfax_optimization_diaries.md) to decode S/P ping-pong and backward hdim-64 TMEM de-aliasing | ch10, ch11, ch18 |
| `labs/flashattention_gluon/` | FlashAttention experimentation | ch18 |
| `labs/flashinfer_attention/` | FlashInfer block-sparse attention lab | ch16 |
| `labs/flexattention/` | FlexAttention harness and sweeps | ch18 |
| `labs/fullstack_cluster/` | Full-stack cluster + DSMEM workflows | ch10 |
| `labs/kv_cache_compression/` | KV-cache compression/quantization | ch18, ch19 |
| `labs/kv_optimization/` | KV-cache performance optimization | ch15, ch18, ch19 |
| `labs/memory_bandwidth_patterns/` | Measured bandwidth patterns: coalesced vs strided access, vectorized copy, shared-memory transpose, and cp.async checkpoints | ch02, ch10, ch11 |
| `labs/moe_cuda/` | CUDA MoE decode toolkit | ch06, ch10, ch15 |
| `labs/moe_decode_blackwell_matrix/` | Blackwell MoE decode scenario matrix for routing locality, persistent scheduling, and graph-launch tradeoffs | ch10, ch15, ch18, ch19 |
| `labs/moe_optimization_journey/` | MoE optimization narrative | ch15, ch19 |
| `labs/moe_parallelism/` | MoE parallelism planning | ch04, ch15 |
| `labs/nccl_nixl_nvshmem/` | Communication-stack tradeoffs with an honest single-GPU analogue and NCCL/NIXL/NVSHMEM stack probing | ch04, ch17, ch19 |
| `labs/nanochat_fullstack/` | End-to-end inference stack (NanoChat) | ch16 |
| `labs/occupancy_tuning/` | Triton occupancy/schedule sweeps | ch08, ch14 |
| `labs/parameterized_cuda_graphs/` | PyTorch-first parameterized CUDA Graph replay with stable buffers and executable memcpy-node updates | ch10, ch11 |
| `labs/persistent_decode/` | Persistent decode + TMA prefill | ch10, ch11 |
| `labs/python_concurrency/` | Python concurrency control-plane playbook (`asyncio`, retries, idempotency, hybrid pipelines) | ch03, ch11, ch16 |
| `labs/real_world_models/` | Real-world model optimization playbook | ch20 |
| `labs/speculative_decode/` | Speculative decoding | ch15, ch18 |
| `labs/training_hotpath/` | Training hot-path supporting examples for reduction fusion and padding-aware projections | ch12, ch14 |
| `labs/trtllm_phi_3_5_moe/` | TensorRT-LLM Phi-3.5-MoE comparison | ch16, ch18 |
| `labs/train_distributed/` | Distributed training workflows | ch03, ch04 |
| `labs/uma_memory/` | UMA / unified memory diagnostics | ch02, ch07 |

## Notes
- Labs now intentionally support both benchmark-pair docs and honest workflow/component docs. The distinction is part of the quality bar, not an exception to it.
