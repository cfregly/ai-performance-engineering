# B200 autoresearch handoff

Updated: 2026-08-16

## Goal

Apply the evidence discipline from [Sankalp's autoresearch writeup](https://sankalp.bearblog.dev/autoresearch/) and [Mike's QR v2 writeup](https://ml-mike.com/writing/qr_v2/) to this repository. Validate every discovered benchmark target on the remote B200 host. Keep correctness, provenance, per-case regression limits, and promotion evidence bound to each result.

The commit that contains this file is the authoritative checkpoint. Run `git rev-parse HEAD` after checkout and record that commit in every validation artifact.

## GPU yield rule

If another task asks for the GPUs, stop this sweep at the next safe target boundary.

1. Stop the active orchestrator and wait for its child benchmark or profiler process to exit.
2. Do not kill unrelated jobs.
3. Run the normalized status command below.
4. Record the active target, completed targets, and remaining targets in this file.
5. Commit the updated handoff and source changes.
6. Push the work branch.
7. Merge it into `main`.
8. Push `main` and verify both remote branch tips.
9. Yield the GPUs.

Run with `--no-auto-resume`. This prevents a stopped sweep from restarting while another task owns the GPUs.

## Current resume point

The strict full sweep has not started. No target is partially complete. All 486 discovered targets remain.

The dry-run inventory contains 464 single-GPU lane targets and 22 two-GPU lane targets. Some target names contain `multigpu` but remain in the single-GPU lane because their benchmark contract explicitly supports one visible GPU. The lane recorded below is authoritative.

A CPU-hidden full test run at public commit `50dd8b2a13c0e01b5840c18ebfae855e3a004ee1` completed before the latest focused fixes:

- 2,703 passed
- 516 skipped
- 5 failed
- 41 percent line coverage

The five failures exposed two deterministic MCP fixture defects and three order-dependent subprocess failures. A clean first-failure probe then identified the exact remaining order defect. A repository hygiene test left `LD_PRELOAD` pointing at a deleted fake NCCL library. The dynamic loader warning replaced the expected CUDA benchmark skip reason.

Post-run fixes now use isolated MCP fixture data, prevent generic cluster promotion from changing tracked files, run the generic fabric tool in readiness-only mode, report truthful readiness-only progress, defer chapter 18 vLLM environment setup until benchmark setup, and restore `LD_PRELOAD` after the explicit system-policy test. The complete CPU-hidden suite still needs one clean rerun from the next published checkpoint.

## Completed validation

- Repository benchmark audit checked 932 files with 0 errors and 0 warnings.
- The integrated post-fix CPU-safe regression set passed 144 tests and skipped 26 capability-limited tests.
- Focused campaign, evidence, queue, promotion, dashboard, setup, benchmark, and generated-code boundary tests passed in their owned batches.
- Dashboard dependency audit reported 0 findings. Lint, component tests, and production build passed.
- Root workflow YAML parsed. All workflow run blocks and setup shell scripts passed syntax checks.
- `make lint` passed when Make used the configured Python interpreter.
- Chapter 8 and Chapter 10 tcgen05 source audits checked 80 files with 0 errors and 0 warnings.
- The complete MCP catalog passed 157 tests with GPUs hidden. Every generic tool call now checks that tracked Git state is unchanged.
- The latest integrated cluster, progress, environment, chapter 18 import-isolation, CUDA wrapper, and profiler set passed 35 tests with GPUs hidden.
- The exact leaking environment-test plus CUDA-wrapper order passed 2 tests after the fix.
- The repository benchmark audit still checks 932 files with 0 errors and 0 warnings after the latest changes.
- B200 Chapter 9 FP8 CUTLASS baseline, optimized, and verification builds compiled for SM100a. Verification checksums matched.
- B200 Chapter 19 focused GPU tests passed.
- One Chapter 9 timing probe was observed. It is not enough evidence for a performance claim.

## Environment contract

Use environment variables or local shell aliases for transport details. Do not commit hostnames, private paths, keys, tokens, or remote instance identifiers.

Required placeholders:

- `$B200_JUMP`: jump host
- `$B200_HOST`: B200 host reached through the jump host
- `$B200_CANDIDATE_WORKTREE`: candidate checkout
- `$B200_PYTHON`: validated Python interpreter
- `$RUN_ID`: stable run identifier

The validated hardware contract is two NVIDIA B200 GPUs with CUDA 13, PyTorch with CUDA 13 support, Triton, Nsight Systems, and Nsight Compute. The host has NVLink between both GPUs. RDMA was not visible, so fabric results may be truthful partial results.

## Preflight after checkout

```bash
git status --short
git submodule status --recursive
"$B200_PYTHON" -m core.scripts.linting.check_benchmarks --include-unpaired --fail-on-warnings
make lint
CUDA_VISIBLE_DEVICES= COVERAGE_FILE=/tmp/aiperf-coverage \
  "$B200_PYTHON" -m pytest tests/ -v --tb=short -p no:cacheprovider \
  --cov=core --cov-report=term-missing
```

Do not continue to GPU timing if collection fails. Fix collection and rerun the CPU-hidden suite first.

## Strict sweep command

Run the command inside `$B200_CANDIDATE_WORKTREE/code`.

```bash
"$B200_PYTHON" -m cli.aisp bench run-e2e \
  --run-id "$RUN_ID" \
  --run-full-sweep \
  --run-fabric \
  --cluster-preset common-answer-fast \
  --validity-profile strict \
  --profile minimal \
  --iterations 5 \
  --warmup 1 \
  --no-auto-resume
```

Do not set GPU clocks outside the harness. Run one benchmark or profiler owner at a time.

## Status and pause capture

```bash
"$B200_PYTHON" -m cli.aisp bench run-e2e-status --run-id "$RUN_ID"
```

The normalized status is the preferred source. Capture these run files before yielding:

- `artifacts/e2e_runs/$RUN_ID/manifest.json`
- `artifacts/e2e_runs/$RUN_ID/progress.json`
- `artifacts/e2e_runs/$RUN_ID/checkpoint.json`
- `artifacts/e2e_runs/$RUN_ID/target_inventory.json`
- `artifacts/e2e_runs/$RUN_ID/events.jsonl`
- `artifacts/e2e_runs/$RUN_ID/summary.json`
- `artifacts/e2e_runs/$RUN_ID/summary.md`

Generated run artifacts remain ignored by Git. Record their hashes and terminal status in the campaign ledger or this handoff. Do not force-add raw profiler output or generated binaries.

## Resume command

Use the same `$RUN_ID` after reviewing the normalized status and confirming no other task owns the GPUs.

```bash
"$B200_PYTHON" -m cli.aisp bench run-e2e \
  --run-id "$RUN_ID" \
  --resume \
  --no-auto-resume
```

## Remaining validation phases

1. Rerun the full CPU-hidden pytest suite with coverage.
2. Fix every collection or test failure that reproduces in isolation.
3. Run the strict 486-target sweep below.
4. Run the two-GPU lane only when both GPUs are free.
5. Record honest `passed`, `failed`, `skipped`, or `partial` status for every target.
6. Compare candidate and control on representative Chapter 9 and Chapter 19 cases with repeated trials.
7. Audit manifests, hashes, environment evidence, profiler evidence, and no-regression gates.
8. Update this handoff with the exact completed and remaining target sets.
9. Commit, push, merge to `main`, push `main`, and verify remote tips.

"100 percent coverage" means every discovered target has a terminal status and every reachable code path has measured coverage. It does not permit turning unsupported hardware or missing fabric into a false pass.

## Complete unrun target inventory

This list came from the strict remote dry run on 2026-08-16. Every listed target remains unrun in the full sweep.

```text
ch01:gemm [1 GPU lane]
ch01:gemm_batched [1 GPU lane]
ch01:gemm_strided [1 GPU lane]
ch01:nvfp4_mlp [1 GPU lane]
ch01:performance [1 GPU lane]
ch01:performance_fp16 [1 GPU lane]
ch01:performance_fusion [1 GPU lane]
ch02:cublas [1 GPU lane]
ch02:grace_coherent_memory [1 GPU lane]
ch02:memory_transfer [1 GPU lane]
ch03:double_buffered_batch_provisioning [1 GPU lane]
ch03:gemm [1 GPU lane]
ch03:pageable_copy [1 GPU lane]
ch03:pinned_prefetch_mlp [1 GPU lane]
ch03:rack_prep [1 GPU lane]
ch04:bandwidth_benchmark_suite [1 GPU lane]
ch04:bandwidth_benchmark_suite_multigpu [1 GPU lane]
ch04:continuous_batching [1 GPU lane]
ch04:continuous_batching_multigpu [1 GPU lane]
ch04:cpu_reduction [1 GPU lane]
ch04:dataparallel [1 GPU lane]
ch04:dataparallel_multigpu [1 GPU lane]
ch04:disaggregated [1 GPU lane]
ch04:disaggregated_multigpu [1 GPU lane]
ch04:grace_blackwell_locality [1 GPU lane]
ch04:gradient_compression_fp16 [1 GPU lane]
ch04:gradient_compression_fp16_comm_only [1 GPU lane]
ch04:gradient_compression_fp16_comm_only_multigpu [1 GPU lane]
ch04:gradient_compression_fp16_multigpu [1 GPU lane]
ch04:gradient_compression_int8 [1 GPU lane]
ch04:gradient_compression_int8_comm_only [1 GPU lane]
ch04:gradient_compression_int8_comm_only_multigpu [1 GPU lane]
ch04:gradient_compression_int8_multigpu [1 GPU lane]
ch04:gradient_fusion [1 GPU lane]
ch04:nccl [1 GPU lane]
ch04:nixl_tier_handoff [1 GPU lane]
ch04:no_overlap [1 GPU lane]
ch04:nvlink_multigpu [1 GPU lane]
ch04:nvlink_topology_aware [1 GPU lane]
ch04:nvlink_topology_aware_multigpu [1 GPU lane]
ch04:nvshmem_ibgda_microbench [1 GPU lane]
ch04:nvshmem_ibgda_microbench_multigpu [1 GPU lane]
ch04:nvshmem_pipeline_parallel [1 GPU lane]
ch04:nvshmem_training_example [1 GPU lane]
ch04:nvshmem_training_patterns [1 GPU lane]
ch04:nvshmem_vs_nccl_benchmark [1 GPU lane]
ch04:pcie_staging [1 GPU lane]
ch04:reinit_comm [1 GPU lane]
ch04:reinit_comm_multigpu [1 GPU lane]
ch04:symmetric_memory [1 GPU lane]
ch04:symmetric_memory_perf [1 GPU lane]
ch05:ai [1 GPU lane]
ch05:decompression [1 GPU lane]
ch05:distributed_multigpu [1 GPU lane]
ch05:host_staged_reduction [1 GPU lane]
ch05:storage_cpu [1 GPU lane]
ch05:vectorization [1 GPU lane]
ch06:adaptive [1 GPU lane]
ch06:add [1 GPU lane]
ch06:add_cuda [1 GPU lane]
ch06:attention_ilp [1 GPU lane]
ch06:autotuning [1 GPU lane]
ch06:bank_conflicts [1 GPU lane]
ch06:elementwise_ilp [1 GPU lane]
ch06:launch_bounds [1 GPU lane]
ch06:launch_bounds_cuda [1 GPU lane]
ch06:quantization_ilp [1 GPU lane]
ch06:warp_divergence_ilp [1 GPU lane]
ch07:async_prefetch [1 GPU lane]
ch07:copy_scalar [1 GPU lane]
ch07:copy_scalar_vectorized [1 GPU lane]
ch07:copy_uncoalesced [1 GPU lane]
ch07:copy_uncoalesced_coalesced [1 GPU lane]
ch07:float4_vector [1 GPU lane]
ch07:hbm_copy [1 GPU lane]
ch07:hbm_peak [1 GPU lane]
ch07:lookup [1 GPU lane]
ch07:matmul [1 GPU lane]
ch07:matmul_tiled [1 GPU lane]
ch07:memory_access [1 GPU lane]
ch07:tma_bulk_tensor_2d [1 GPU lane]
ch07:tma_copy [1 GPU lane]
ch07:transpose [1 GPU lane]
ch07:transpose_padded [1 GPU lane]
ch08:ai_optimization [1 GPU lane]
ch08:hbm [1 GPU lane]
ch08:hbm_cuda [1 GPU lane]
ch08:hbm_cuda_vectorized [1 GPU lane]
ch08:loop_unrolling [1 GPU lane]
ch08:nvfp4_mlp [1 GPU lane]
ch08:occupancy_tuning [1 GPU lane]
ch08:tcgen05_custom_vs_cublas [1 GPU lane]
ch08:threshold [1 GPU lane]
ch08:thresholdtma [1 GPU lane]
ch08:tiling [1 GPU lane]
ch08:tiling_tcgen05 [1 GPU lane]
ch09:compute_bound [1 GPU lane]
ch09:cublas_gemm_fp4_perchannel [1 GPU lane]
ch09:cublaslt_gemm [1 GPU lane]
ch09:cublaslt_gemm_fp16 [1 GPU lane]
ch09:cublaslt_gemm_fp4 [1 GPU lane]
ch09:cublaslt_gemm_fp8 [1 GPU lane]
ch09:cute_dsl_nvfp4_gemm [1 GPU lane]
ch09:cutlass_gemm [1 GPU lane]
ch09:cutlass_gemm_fp16 [1 GPU lane]
ch09:cutlass_gemm_fp4 [1 GPU lane]
ch09:cutlass_gemm_fp4_all_concepts [1 GPU lane]
ch09:cutlass_gemm_fp4_perchannel [1 GPU lane]
ch09:cutlass_gemm_fp8 [1 GPU lane]
ch09:fused_l2norm [1 GPU lane]
ch09:memory_bound [1 GPU lane]
ch09:micro_tiling_matmul [1 GPU lane]
ch09:sdpa_attention [1 GPU lane]
ch09:tcgen05_tma_pipeline [1 GPU lane]
ch09:triton [1 GPU lane]
ch10:atomic_reduction [1 GPU lane]
ch10:attention [1 GPU lane]
ch10:batch [1 GPU lane]
ch10:cluster_group [1 GPU lane]
ch10:cluster_group_no_dsmem [1 GPU lane]
ch10:cluster_group_single_cta [1 GPU lane]
ch10:cluster_multicast [1 GPU lane]
ch10:cooperative_persistent [1 GPU lane]
ch10:double_buffered_pipeline [1 GPU lane]
ch10:dsmem_reduction [1 GPU lane]
ch10:dsmem_reduction_cluster_atomic [1 GPU lane]
ch10:dsmem_reduction_v3 [1 GPU lane]
ch10:dsmem_reduction_warp_specialized [1 GPU lane]
ch10:flash_attention [1 GPU lane]
ch10:flash_attn_tma_micro_pipeline [1 GPU lane]
ch10:flashattention3_pipeline [1 GPU lane]
ch10:matmul_tcgen05_epilogue [1 GPU lane]
ch10:matmul_tcgen05_pipelined [1 GPU lane]
ch10:matmul_tcgen05_vs_cublas [1 GPU lane]
ch10:persistent_matmul_tma [1 GPU lane]
ch10:pipeline_3stage [1 GPU lane]
ch10:tcgen05_cluster_pipeline [1 GPU lane]
ch10:tcgen05_warp_specialization [1 GPU lane]
ch10:tcgen05_warp_specialization_cutlass [1 GPU lane]
ch10:tcgen05_warpgroup_specialization [1 GPU lane]
ch10:tma_2d_pipeline [1 GPU lane]
ch10:warp_spec_pingpong [1 GPU lane]
ch10:warp_specialized_cluster_pipeline [1 GPU lane]
ch10:warp_specialized_pipeline [1 GPU lane]
ch10:warp_specialized_pipeline_enhanced [1 GPU lane]
ch11:adaptive_streams [1 GPU lane]
ch11:distributed_streams [1 GPU lane]
ch11:gemm_streams [1 GPU lane]
ch11:stream_ordered [1 GPU lane]
ch11:stream_ordered_kv_cache [1 GPU lane]
ch11:streams [1 GPU lane]
ch11:tensor_cores_streams [1 GPU lane]
ch11:warp_specialization_multistream [1 GPU lane]
ch11:warp_specialized_multistream [1 GPU lane]
ch11:warp_specialized_two_pipelines_driver [1 GPU lane]
ch11:warp_specialized_two_pipelines_multistream [1 GPU lane]
ch12:cuda_graphs [1 GPU lane]
ch12:cuda_graphs_conditional [1 GPU lane]
ch12:cuda_graphs_conditional_enhanced [1 GPU lane]
ch12:cuda_graphs_router [1 GPU lane]
ch12:dynamic_parallelism_device [1 GPU lane]
ch12:dynamic_parallelism_host [1 GPU lane]
ch12:graph_bandwidth [1 GPU lane]
ch12:graph_conditional_runtime [1 GPU lane]
ch12:kernel_fusion [1 GPU lane]
ch12:kernel_fusion_llm_dedicated_stream_and_prefetch_for_blackwell [1 GPU lane]
ch12:kernel_fusion_llm_persistent_buffer_and_stream_friendly_setup [1 GPU lane]
ch12:kernel_fusion_llm_reuse_static_tensor_and_simplify_setup [1 GPU lane]
ch12:kernel_launches [1 GPU lane]
ch12:nvfp4_mlp [1 GPU lane]
ch12:uneven_partition [1 GPU lane]
ch12:uneven_static [1 GPU lane]
ch12:work_queue [1 GPU lane]
ch13:arithmetic_intensity [1 GPU lane]
ch13:attention_sliding_window [1 GPU lane]
ch13:attention_standard [1 GPU lane]
ch13:autograd_standard [1 GPU lane]
ch13:bandwidth_naive [1 GPU lane]
ch13:dataloader_default [1 GPU lane]
ch13:fp4_perchannel [1 GPU lane]
ch13:fp8_perchannel [1 GPU lane]
ch13:fp8_static [1 GPU lane]
ch13:kv_cache_naive [1 GPU lane]
ch13:kv_cache_naive_flash_blockwise [1 GPU lane]
ch13:kv_cache_naive_pool [1 GPU lane]
ch13:long_context_attention [1 GPU lane]
ch13:matmul_pytorch [1 GPU lane]
ch13:memory_profiling [1 GPU lane]
ch13:precisionfp8 [1 GPU lane]
ch13:precisionfp8_pad_inner [1 GPU lane]
ch13:precisionfp8_pad_inner_matmul [1 GPU lane]
ch13:precisionfp8_rowwise [1 GPU lane]
ch13:precisionfp8_rowwise_gw_hp [1 GPU lane]
ch13:precisionfp8_te [1 GPU lane]
ch13:precisionmixed [1 GPU lane]
ch13:quantization [1 GPU lane]
ch13:regional_compile [1 GPU lane]
ch13:torchao_quantization [1 GPU lane]
ch13:torchao_quantization_compiled [1 GPU lane]
ch13:training_speed [1 GPU lane]
ch13:training_standard [1 GPU lane]
ch13:warp_specialization_training [1 GPU lane]
ch14:attention_eager_sdpa [1 GPU lane]
ch14:cublas_vs_cutlass [1 GPU lane]
ch14:cuda_python [1 GPU lane]
ch14:flex_attention_sparse [1 GPU lane]
ch14:graph_break_control_flow [1 GPU lane]
ch14:model_compile_reduced_precision [1 GPU lane]
ch14:nccl_quantization [1 GPU lane]
ch14:regional_triton [1 GPU lane]
ch14:sliding_window [1 GPU lane]
ch14:triton_persistent [1 GPU lane]
ch15:allreduce_rmsnorm [1 GPU lane]
ch15:continuous_batching [1 GPU lane]
ch15:continuous_batching_multigpu [1 GPU lane]
ch15:dep2_parallel [1 GPU lane]
ch15:greedy_sampler [1 GPU lane]
ch15:guided_decoding [1 GPU lane]
ch15:inference_monolithic [1 GPU lane]
ch15:inference_placement [1 GPU lane]
ch15:kv_cache_management [1 GPU lane]
ch15:kv_cache_nvlink_pool [1 GPU lane]
ch15:kv_cache_nvlink_pool_multigpu [1 GPU lane]
ch15:medusa_eagle_speculative [1 GPU lane]
ch15:medusa_eagle_speculative_eagle [1 GPU lane]
ch15:medusa_eagle_speculative_medusa [1 GPU lane]
ch15:moe_comm_exchange [1 GPU lane]
ch15:moe_comm_exchange_hierarchical [1 GPU lane]
ch15:moe_comm_exchange_overlap [1 GPU lane]
ch15:moe_dispatch [1 GPU lane]
ch15:moe_inference [1 GPU lane]
ch15:moe_overlap [1 GPU lane]
ch15:moe_overlap_local_route [1 GPU lane]
ch15:moe_overlap_shared_expert [1 GPU lane]
ch15:moe_routing_topology_aware [1 GPU lane]
ch15:nvfp4_mlp [1 GPU lane]
ch15:prefill_decode_disagg [1 GPU lane]
ch15:prefill_decode_disagg_multigpu [1 GPU lane]
ch15:single_gpu_kv_handoff [1 GPU lane]
ch15:speculative_decoding [1 GPU lane]
ch15:wide_ep [1 GPU lane]
ch16:awq_gptq_smoothquant [1 GPU lane]
ch16:awq_gptq_smoothquant_awq [1 GPU lane]
ch16:awq_gptq_smoothquant_gptq [1 GPU lane]
ch16:awq_gptq_smoothquant_smoothquant [1 GPU lane]
ch16:dense_attention_flash [1 GPU lane]
ch16:dense_attention_flash_blackwell_variant [1 GPU lane]
ch16:flash_sdp [1 GPU lane]
ch16:flashinfer_block_sparse [1 GPU lane]
ch16:nvfp4_mlp [1 GPU lane]
ch16:piece_graphs [1 GPU lane]
ch16:regional_compilation [1 GPU lane]
ch16:runtime_scheduler [1 GPU lane]
ch17:dynamic_routing [1 GPU lane]
ch17:inference_full [1 GPU lane]
ch17:memory [1 GPU lane]
ch17:moe_router_local_capacity [1 GPU lane]
ch17:moe_router_uniform [1 GPU lane]
ch17:moe_router_uniform_topology [1 GPU lane]
ch17:nvfp4_mlp [1 GPU lane]
ch17:pipeline_parallelism [1 GPU lane]
ch17:prefill_decode_disagg [1 GPU lane]
ch17:prefill_decode_disagg_batched_multigpu [1 GPU lane]
ch17:prefill_decode_disagg_overlap_multigpu [1 GPU lane]
ch17:prefill_decode_disagg_tpot_long [1 GPU lane]
ch17:prefill_decode_disagg_tpot_long_multigpu [1 GPU lane]
ch17:prefill_decode_disagg_ttft [1 GPU lane]
ch17:prefill_decode_disagg_ttft_multigpu [1 GPU lane]
ch17:routing_static [1 GPU lane]
ch18:cudagraph_bucketing [1 GPU lane]
ch18:eos_early_exit [1 GPU lane]
ch18:eos_sync_polling [1 GPU lane]
ch18:flexattention_sliding_window [1 GPU lane]
ch18:flexdecoding [1 GPU lane]
ch18:flexdecoding_graphs [1 GPU lane]
ch18:paged_attn_backend [1 GPU lane]
ch18:paged_attn_layout [1 GPU lane]
ch18:rope_q_cache [1 GPU lane]
ch18:tensor_cores [1 GPU lane]
ch18:tiny_gemm_fused [1 GPU lane]
ch18:vllm_decode_graphs [1 GPU lane]
ch18:vllm_v1_integration [1 GPU lane]
ch19:adaptive_parallelism [1 GPU lane]
ch19:dynamic_precision [1 GPU lane]
ch19:dynamic_quantized_cache [1 GPU lane]
ch19:dynamic_quantized_cache_coalesced [1 GPU lane]
ch19:fp4_hardware_kernel [1 GPU lane]
ch19:fp4_weight_quantization [1 GPU lane]
ch19:kv_prefetch_overlap [1 GPU lane]
ch19:memory_double_buffering [1 GPU lane]
ch19:mxfp8_moe [1 GPU lane]
ch19:nvfp4_training [1 GPU lane]
ch19:vectorization_memory [1 GPU lane]
ch20:autotuning [1 GPU lane]
ch20:bf16_mlp [1 GPU lane]
ch20:end_to_end_bandwidth [1 GPU lane]
ch20:integrated_kv_cache [1 GPU lane]
ch20:memory_standard [1 GPU lane]
ch20:moe [1 GPU lane]
ch20:nvfp4_mlp [1 GPU lane]
ch20:pipeline_sequential [1 GPU lane]
ch20:training_single [1 GPU lane]
labs/async_input_pipeline:async_input_pipeline [1 GPU lane]
labs/blackwell_gemm_optimizations:blackwell_grouped_gemm [1 GPU lane]
labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_full_stack [1 GPU lane]
labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_large_tiles [1 GPU lane]
labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_persistent [1 GPU lane]
labs/blackwell_matmul:blackwell_matmul [1 GPU lane]
labs/blackwell_matmul:blackwell_matmul_cluster [1 GPU lane]
labs/blackwell_matmul:blackwell_matmul_pipeline [1 GPU lane]
labs/blackwell_matmul:blackwell_matmul_tcgen05 [1 GPU lane]
labs/blackwell_matmul:blackwell_matmul_tma [1 GPU lane]
labs/block_scaling:block_scaling [1 GPU lane]
labs/cache_aware_disagg_inference:cache_aware_disagg [1 GPU lane]
labs/cache_aware_disagg_inference:cache_aware_disagg_multigpu [1 GPU lane]
labs/cudnn_sdpa_bench:flash_sdp [1 GPU lane]
labs/custom_vs_cublas:tcgen05_matmul [1 GPU lane]
labs/decode_optimization:decode [1 GPU lane]
labs/decode_optimization:decode_candidate_logits [1 GPU lane]
labs/decode_optimization:decode_compile [1 GPU lane]
labs/decode_optimization:decode_device_resident [1 GPU lane]
labs/decode_optimization:decode_double_buffer_tma [1 GPU lane]
labs/decode_optimization:decode_fp4 [1 GPU lane]
labs/decode_optimization:decode_fp8 [1 GPU lane]
labs/decode_optimization:decode_graph [1 GPU lane]
labs/decode_optimization:decode_graph_full [1 GPU lane]
labs/decode_optimization:decode_hf_cache [1 GPU lane]
labs/decode_optimization:decode_pinned [1 GPU lane]
labs/decode_optimization:decode_prefix_state_cache [1 GPU lane]
labs/decode_optimization:decode_streams [1 GPU lane]
labs/decode_optimization:decode_ultimate [1 GPU lane]
labs/decode_optimization:decode_warp_specialized [1 GPU lane]
labs/dynamic_router:dual_pool_vllm [1 GPU lane]
labs/dynamic_router:dynamic_router_vllm [1 GPU lane]
labs/flashattention4:best_available_attention [1 GPU lane]
labs/flashattention4:best_available_attention_alibi [1 GPU lane]
labs/flashattention4:best_available_attention_alibi_windowed [1 GPU lane]
labs/flashattention4:best_available_attention_causal [1 GPU lane]
labs/flashattention4:best_available_attention_dense [1 GPU lane]
labs/flashattention4:best_available_attention_softcap [1 GPU lane]
labs/flashattention4:best_available_attention_windowed [1 GPU lane]
labs/flashattention4:flashattention4 [1 GPU lane]
labs/flashattention4:flashattention4_alibi [1 GPU lane]
labs/flashattention4:flashattention4_alibi_windowed [1 GPU lane]
labs/flashattention4:flashattention4_causal [1 GPU lane]
labs/flashattention4:flashattention4_dense [1 GPU lane]
labs/flashattention4:flashattention4_softcap [1 GPU lane]
labs/flashattention4:flashattention4_windowed [1 GPU lane]
labs/flashattention_gluon:flashattention_gluon [1 GPU lane]
labs/flashinfer_attention:flashinfer_attention [1 GPU lane]
labs/flexattention:flex_attention [1 GPU lane]
labs/fullstack_cluster:cluster_gemm [1 GPU lane]
labs/fullstack_cluster:cluster_gemm_tcgen05 [1 GPU lane]
labs/fullstack_cluster:cluster_gemm_tcgen05_cta2 [1 GPU lane]
labs/fullstack_cluster:moe_hybrid_ep [1 GPU lane]
labs/fullstack_cluster:moe_hybrid_ep_multigpu [1 GPU lane]
labs/kv_cache_compression:kv_cache [1 GPU lane]
labs/kv_cache_compression:kv_cache_nvfp4 [1 GPU lane]
labs/kv_optimization:kv_standard [1 GPU lane]
labs/memory_bandwidth_patterns:bandwidth_patterns [1 GPU lane]
labs/moe_cuda:decode_attention [1 GPU lane]
labs/moe_cuda:decode_kernel [1 GPU lane]
labs/moe_cuda:kv_transfer [1 GPU lane]
labs/moe_cuda:kv_transfer_direct [1 GPU lane]
labs/moe_cuda:kv_transfer_direct_graphs [1 GPU lane]
labs/moe_cuda:kv_transfer_graphs [1 GPU lane]
labs/moe_cuda:moe_backend_selection [1 GPU lane]
labs/moe_cuda:router [1 GPU lane]
labs/moe_cuda:router_vectorized [1 GPU lane]
labs/moe_cuda_ptx:moe_grouped_gemm_bwd [1 GPU lane]
labs/moe_cuda_ptx:moe_grouped_gemm_fwd [1 GPU lane]
labs/moe_cuda_ptx:moe_layer [1 GPU lane]
labs/moe_cuda_ptx:moe_quant [1 GPU lane]
labs/moe_optimization_journey:moe [1 GPU lane]
labs/moe_optimization_journey:moe_batched [1 GPU lane]
labs/moe_optimization_journey:moe_bmm_fusion [1 GPU lane]
labs/moe_optimization_journey:moe_compiled [1 GPU lane]
labs/moe_optimization_journey:moe_cuda_graphs [1 GPU lane]
labs/moe_optimization_journey:moe_expert_parallel [1 GPU lane]
labs/moe_optimization_journey:moe_fp8 [1 GPU lane]
labs/moe_optimization_journey:moe_fused [1 GPU lane]
labs/moe_optimization_journey:moe_grouped [1 GPU lane]
labs/moe_optimization_journey:moe_memefficient [1 GPU lane]
labs/moe_optimization_journey:moe_pad_quant [1 GPU lane]
labs/moe_optimization_journey:moe_parallel [1 GPU lane]
labs/moe_optimization_journey:moe_permuted [1 GPU lane]
labs/moe_optimization_journey:moe_sorted [1 GPU lane]
labs/moe_optimization_journey:moe_streams [1 GPU lane]
labs/moe_optimization_journey:moe_triton [1 GPU lane]
labs/nanochat_fullstack:nanochat_inference [1 GPU lane]
labs/nccl_nixl_nvshmem:tier_handoff [1 GPU lane]
labs/nvfp4_dual_gemm:nvfp4_dual_gemm [1 GPU lane]
labs/nvfp4_gemm:nvfp4_gemm [1 GPU lane]
labs/nvfp4_gemv:nvfp4_gemv [1 GPU lane]
labs/nvfp4_group_gemm:nvfp4_group_gemm [1 GPU lane]
labs/nvfp4_group_gemm:nvfp4_group_gemm_g2_n3072_k4096 [1 GPU lane]
labs/nvfp4_group_gemm:nvfp4_group_gemm_g2_n4096_k1536 [1 GPU lane]
labs/nvfp4_group_gemm:nvfp4_group_gemm_g8_n4096_k7168 [1 GPU lane]
labs/nvfp4_group_gemm:nvfp4_group_gemm_g8_n7168_k2048 [1 GPU lane]
labs/occupancy_tuning:proton_matmul [1 GPU lane]
labs/occupancy_tuning:proton_matmul_bm128_bn128_bk32_nw8 [1 GPU lane]
labs/occupancy_tuning:proton_matmul_bm128_bn256_bk64 [1 GPU lane]
labs/occupancy_tuning:proton_matmul_bm256_bn256_bk64 [1 GPU lane]
labs/occupancy_tuning:proton_matmul_bm64_bn256_bk32 [1 GPU lane]
labs/occupancy_tuning:proton_matmul_bm64_bn64_bk32_nw2 [1 GPU lane]
labs/ozaki_scheme:ozaki_scheme [1 GPU lane]
labs/ozaki_scheme:ozaki_scheme_dynamic [1 GPU lane]
labs/ozaki_scheme:ozaki_scheme_fixed [1 GPU lane]
labs/parameterized_cuda_graphs:parameterized_graph_launch [1 GPU lane]
labs/persistent_decode:native_tma_prefill_decode [1 GPU lane]
labs/persistent_decode:nvlink_offload [1 GPU lane]
labs/persistent_decode:paged_kv_offload [1 GPU lane]
labs/persistent_decode:paged_kv_offload_prefetch [1 GPU lane]
labs/persistent_decode:persistent_decode [1 GPU lane]
labs/persistent_decode:persistent_decode_cuda [1 GPU lane]
labs/persistent_decode:persistent_decode_full_and_piecewise [1 GPU lane]
labs/persistent_decode:persistent_decode_graphs [1 GPU lane]
labs/persistent_decode:persistent_decode_triton [1 GPU lane]
labs/persistent_decode:right_sized_decode [1 GPU lane]
labs/persistent_decode:tma_prefill_decode [1 GPU lane]
labs/real_world_models:llama_3_1_8b [1 GPU lane]
labs/recsys_sequence_ranking:sequence_ranking [1 GPU lane]
labs/software_pipelining:tile_pipeline [1 GPU lane]
labs/speculative_decode:speculative_decode [1 GPU lane]
labs/speculative_decode:speculative_decode_transition_table [1 GPU lane]
labs/speculative_decode:speculative_decode_trusted [1 GPU lane]
labs/top_k_kernel:top_k_kernel [1 GPU lane]
labs/top_k_kernel:top_k_kernel_cuda [1 GPU lane]
labs/train_distributed:ddp [1 GPU lane]
labs/train_distributed:ddp_compression [1 GPU lane]
labs/train_distributed:ddp_compression_int8 [1 GPU lane]
labs/train_distributed:ddp_compression_multigpu_int8 [1 GPU lane]
labs/train_distributed:ddp_compression_multigpu_powersgd [1 GPU lane]
labs/train_distributed:ddp_compression_powersgd [1 GPU lane]
labs/train_distributed:ddp_flash [1 GPU lane]
labs/train_distributed:ddp_flash_multigpu [1 GPU lane]
labs/train_distributed:ddp_multigpu [1 GPU lane]
labs/train_distributed:fsdp [1 GPU lane]
labs/train_distributed:fsdp2 [1 GPU lane]
labs/train_distributed:fsdp2_multigpu [1 GPU lane]
labs/train_distributed:fsdp_multigpu [1 GPU lane]
labs/train_distributed:pipeline_1f1b [1 GPU lane]
labs/train_distributed:pipeline_1f1b_multigpu [1 GPU lane]
labs/train_distributed:pipeline_1f1b_to_gpipe_multigpu [1 GPU lane]
labs/train_distributed:pipeline_dualpipe [1 GPU lane]
labs/train_distributed:pipeline_dualpipe_multigpu [1 GPU lane]
labs/train_distributed:pipeline_dualpipev [1 GPU lane]
labs/train_distributed:pipeline_dualpipev_multigpu [1 GPU lane]
labs/train_distributed:pipeline_gpipe [1 GPU lane]
labs/train_distributed:pipeline_gpipe_multigpu [1 GPU lane]
labs/train_distributed:pipeline_gpipe_to_dualpipe_multigpu [1 GPU lane]
labs/train_distributed:pipeline_gpipe_to_dualpipev_multigpu [1 GPU lane]
labs/train_distributed:symmem_training [1 GPU lane]
labs/train_distributed:symmem_training_multigpu [1 GPU lane]
labs/train_distributed:zero1 [1 GPU lane]
labs/train_distributed:zero1_multigpu [1 GPU lane]
labs/train_distributed:zero2 [1 GPU lane]
labs/train_distributed:zero2_multigpu [1 GPU lane]
labs/train_distributed:zero3 [1 GPU lane]
labs/train_distributed:zero3_multigpu [1 GPU lane]
labs/training_hotpath:metric_reduction_cuda [1 GPU lane]
labs/training_hotpath:metric_reduction_vectorized [1 GPU lane]
labs/training_hotpath:padding_aware_transformer [1 GPU lane]
labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe [1 GPU lane]
ch02:memory_transfer_multigpu [2 GPU lane]
ch04:gradient_fusion_multigpu [2 GPU lane]
ch04:nvshmem_pipeline_parallel_multigpu [2 GPU lane]
ch04:nvshmem_training_example_multigpu [2 GPU lane]
ch04:nvshmem_training_patterns_multigpu [2 GPU lane]
ch04:nvshmem_vs_nccl_benchmark_multigpu [2 GPU lane]
ch04:pipeline_parallel [2 GPU lane]
ch04:pipeline_parallel_1f1b [2 GPU lane]
ch04:pipeline_parallel_multigpu [2 GPU lane]
ch04:pipeline_parallel_multigpu_1f1b [2 GPU lane]
ch04:symmetric_memory_multigpu [2 GPU lane]
ch04:symmetric_memory_perf_multigpu [2 GPU lane]
ch04:tensor_parallel [2 GPU lane]
ch04:tensor_parallel_allgather_multigpu [2 GPU lane]
ch04:tensor_parallel_async [2 GPU lane]
ch04:tensor_parallel_multigpu [2 GPU lane]
ch04:torchcomms [2 GPU lane]
ch04:torchcomms_multigpu [2 GPU lane]
ch13:context_parallel_multigpu [2 GPU lane]
ch13:expert_parallel_multigpu [2 GPU lane]
ch13:sequence_parallel_multigpu [2 GPU lane]
ch15:disaggregated_inference_multigpu [2 GPU lane]
```
