# Wave 2 remaining runtime gates

This is the finding-specific runtime remainder after source revision `3316e0efe985040745ffd926c5f76a6bd4436aff`. All 48 rows have source repairs and focused host/source evidence; none is counted as runtime-verified. The final hosted compare workflow compiled 14 CUDA chapters for `sm_100`, `sm_103`, `sm_120`, and `sm_121`, but it had no GPU and did not run the device or numerical checks below. B200 custody is still unavailable to this task, and several gates also require Grace, pinned vLLM/Transformer Engine, NVML, or supported compiler/runtime combinations.

## CUDA graph, CUTLASS, TMA, Triton, and compiler execution (18)

| ID | Severity | Location | Remaining acceptance |
| --- | --- | --- | --- |
| W2-001 | critical | `code/ch12/cuda_extensions/cuda_graphs_kernels.cu:99` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-002 | critical | `code/ch12/optimized_graph_conditional_runtime.cu:282` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-003 | critical | `code/labs/custom_vs_cublas/cutlass_gemm/cutlass_gemm.cu:125` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-015 | high | `code/ch09/baseline_cublas_gemm_fp4_perchannel.cu:196` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-039 | medium | `code/ch02/memory_transfer_nvlink_demo.cu:296` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-050 | medium | `code/ch09/baseline_cublas_gemm_fp4_perchannel.cu:58` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-052 | medium | `code/ch09/optimized_cutlass_gemm_fp4_all_concepts.cu:151` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-053 | medium | `code/ch09/tcgen05_basic.cu:179` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-078 | medium | `code/core/common/headers/arch_detection.cuh:71` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-079 | medium | `code/core/scripts/utilities/probe_hardware_capabilities.py:250` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-083 | medium | `code/labs/cutlass_profiler_kernel_selector/run_triton_matmul.py:75` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-091 | medium | `code/labs/software_pipelining/software_pipelining_kernels.cu:50` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-100 | low | `code/ch07/async_prefetch_2d_demo.cu:134` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-101 | low | `code/ch07/optimized_tma_bulk_tensor_2d.cu:154` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-102 | low | `code/ch07/optimized_tma_copy.cu:450` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-110 | low | `code/ch12/cuda_extensions/cuda_graphs_kernels.cu:9` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-119 | low | `code/core/common/headers/cuda13_demos.cuh:225` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |
| W2-137 | low | `code/labs/top_k_kernel/top_k_kernel_common.py:744` | Source repair and host contracts pass; compile on a supported toolkit/driver and compare target-device outputs, including the applicable non-square and tail cases. |

## CUDA asynchronous ordering, allocator lifetime, and FP8 numerics (9)

| ID | Severity | Location | Remaining acceptance |
| --- | --- | --- | --- |
| W2-010 | high | `code/ch05/optimized_ai.py:112` | Source repair and host contracts pass; real asynchronous stream/allocator ordering and target-device numerical validation remain required. |
| W2-020 | high | `code/ch16/inference_optimizations_blackwell.py:294` | Source repair and host contracts pass; real asynchronous stream/allocator ordering and target-device numerical validation remain required. |
| W2-024 | high | `code/ch19/fp8_compiled_matmul.py:103` | Source repair and host contracts pass; real asynchronous stream/allocator ordering and target-device numerical validation remain required. |
| W2-025 | high | `code/ch19/optimized_memory_double_buffering.py:234` | Source repair and host contracts pass; real asynchronous stream/allocator ordering and target-device numerical validation remain required. |
| W2-045 | medium | `code/ch04/multi_node_blackwell.py:533` | Source repair and host contracts pass; real asynchronous stream/allocator ordering and target-device numerical validation remain required. |
| W2-073 | medium | `code/ch19/baseline_dynamic_quantized_cache.py:311` | Source repair and host contracts pass; real asynchronous stream/allocator ordering and target-device numerical validation remain required. |
| W2-075 | medium | `code/ch20/optimized_pipeline_sequential.py:141` | Source repair and host contracts pass; real asynchronous stream/allocator ordering and target-device numerical validation remain required. |
| W2-129 | low | `code/labs/kv_optimization/optimized_kv_standard.py:370` | Source repair and host contracts pass; real asynchronous stream/allocator ordering and target-device numerical validation remain required. |
| W2-132 | low | `code/labs/occupancy_tuning/triton_matmul_schedules.py:235` | Source repair and host contracts pass; real asynchronous stream/allocator ordering and target-device numerical validation remain required. |

## NCCL, distributed, and multi-GPU behavior (11)

| ID | Severity | Location | Remaining acceptance |
| --- | --- | --- | --- |
| W2-008 | high | `code/ch04/multi_node_blackwell.py:375` | Source repair and host contracts pass; torchrun/NCCL on the required topology must verify cross-rank correctness, ordering, timing, and device/cache identity as applicable. |
| W2-022 | high | `code/ch16/inference_server_load_test.py:290` | Source repair and host contracts pass; torchrun/NCCL on the required topology must verify cross-rank correctness, ordering, timing, and device/cache identity as applicable. |
| W2-043 | medium | `code/ch04/ddp_nvlink_overlap.py:223` | Source repair and host contracts pass; torchrun/NCCL on the required topology must verify cross-rank correctness, ordering, timing, and device/cache identity as applicable. |
| W2-047 | medium | `code/ch04/nvls_collectives.py:50` | Source repair and host contracts pass; torchrun/NCCL on the required topology must verify cross-rank correctness, ordering, timing, and device/cache identity as applicable. |
| W2-056 | medium | `code/ch15/baseline_disaggregated_inference_multigpu.py:194` | Source repair and host contracts pass; torchrun/NCCL on the required topology must verify cross-rank correctness, ordering, timing, and device/cache identity as applicable. |
| W2-057 | medium | `code/ch15/expert_parallelism.py:194` | Source repair and host contracts pass; torchrun/NCCL on the required topology must verify cross-rank correctness, ordering, timing, and device/cache identity as applicable. |
| W2-066 | medium | `code/ch17/optimized_pipeline_parallelism.py:146` | Source repair and host contracts pass; torchrun/NCCL on the required topology must verify cross-rank correctness, ordering, timing, and device/cache identity as applicable. |
| W2-067 | medium | `code/ch17/optimized_pipeline_parallelism.py:280` | Source repair and host contracts pass; torchrun/NCCL on the required topology must verify cross-rank correctness, ordering, timing, and device/cache identity as applicable. |
| W2-068 | medium | `code/ch17/optimized_prefill_decode_disagg.py:159` | Source repair and host contracts pass; torchrun/NCCL on the required topology must verify cross-rank correctness, ordering, timing, and device/cache identity as applicable. |
| W2-099 | low | `code/ch04/dist_allreduce.py:60` | Source repair and host contracts pass; torchrun/NCCL on the required topology must verify cross-rank correctness, ordering, timing, and device/cache identity as applicable. |
| W2-123 | low | `code/labs/custom_vs_cublas/autotune.py:40` | Source repair and host contracts pass; torchrun/NCCL on the required topology must verify cross-rank correctness, ordering, timing, and device/cache identity as applicable. |

## Transformer Engine 2.x (1)

| ID | Severity | Location | Remaining acceptance |
| --- | --- | --- | --- |
| W2-017 | high | `code/ch13/optimized_precisionfp8_te.py:103` | Source/API repair passes host contracts; pinned Transformer Engine 2.x setup and FP8 execution remain required. |

## Pinned vLLM 0.16 (4)

| ID | Severity | Location | Remaining acceptance |
| --- | --- | --- | --- |
| W2-028 | high | `code/labs/dynamic_router/vllm_runner.py:232` | Pinned vLLM 0.16 source contracts were checked; real engine/request/token-parity execution on the target GPU remains required. |
| W2-072 | medium | `code/ch18/v1_bucketed_decode_loop.py:203` | Pinned vLLM 0.16 source contracts were checked; real engine/request/token-parity execution on the target GPU remains required. |
| W2-085 | medium | `code/labs/dynamic_router/vllm_runner.py:232` | Pinned vLLM 0.16 source contracts were checked; real engine/request/token-parity execution on the target GPU remains required. |
| W2-127 | low | `code/labs/dynamic_router/baseline_dynamic_router_vllm.py:50` | Pinned vLLM 0.16 source contracts were checked; real engine/request/token-parity execution on the target GPU remains required. |

## B200 BF16 and full-model behavior (2)

| ID | Severity | Location | Remaining acceptance |
| --- | --- | --- | --- |
| W2-089 | medium | `code/labs/real_world_models/llama_3_1_8b_optimization.py:104` | Source/workload repair passes host contracts; declared Llama depth, BF16/output, memory behavior, and full-model execution on B200-class hardware remain required. |
| W2-090 | medium | `code/labs/real_world_models/llama_3_1_8b_optimization.py:171` | Source/workload repair passes host contracts; declared Llama depth, BF16/output, memory behavior, and full-model execution on B200-class hardware remain required. |

## Live NVML and Prometheus telemetry (1)

| ID | Severity | Location | Remaining acceptance |
| --- | --- | --- | --- |
| W2-096 | medium | `code/monitoring/prometheus_exporter.py:89` | Exporter source and focused tests pass; a live device-wide NVML/Prometheus telemetry scrape remains required. |

## Grace and non-Grace hardware identity (2)

| ID | Severity | Location | Remaining acceptance |
| --- | --- | --- | --- |
| W2-098 | low | `code/ch02/cpu_gpu_topology_aware.py:148` | Detection source contracts pass; actual Grace identity/coherency and a non-Grace negative probe remain required. |
| W2-120 | low | `code/core/scripts/utilities/probe_hardware_capabilities.py:234` | Detection source contracts pass; actual Grace identity/coherency and a non-Grace negative probe remain required. |

## Reconciliation

- Runtime-gated rows: **48**
- Source-fixed rows: **89**
- Already fixed with current evidence: **4**
- Untriaged rows: **0**
- Goal complete: **no**
