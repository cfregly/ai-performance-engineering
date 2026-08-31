# Wave 1 remediation inventory

Status: implementation in progress. All 128 external findings are retained. The ledger records source repairs, actual CPU evidence, pending reviews and required GPU checks separately; no overall completion is claimed.

[Plan](../../../AUDIT_REMEDIATION_PLAN.md) · [Ledger](remediation-ledger.json) · [Complete source](wave-1-source.json)

Counts: 5 critical, 37 high, 58 medium, 28 low. Source severity is preserved; verifier qualifications are in the source capture.

The second wave is required and pending; its count is unknown. The separately refuted timeout allegation is W1-R001, outside this table.

## P01: Bootstrap and build architecture (11)

| ID | Source severity | Location at reviewed revision | Finding |
| --- | --- | --- | --- |
| W1-005 | critical | `code/setup.sh:1924` | setup.sh installs the pinned stable torch==2.9.1+cu130 from the nightly-only cu130 index, so setup always aborts |
| W1-007 | high | `code/ch01/Makefile:41` | b200/b300/gb10/gb300 targets cannot change the compile architecture — target-specific ARCH is set after cuda_arch.mk already chose the gencode |
| W1-025 | high | `code/labs/custom_vs_cublas/cutlass_gemm/CMakeLists.txt:50` | Manual -gencode 100a/103a conflicts with CMAKE_CUDA_ARCHITECTURES=100 under separable compilation — the SM100a/SM103a kernels never reach the device link |
| W1-052 | medium | `code/ch19/Makefile:41` | ch16/ch18/ch19/ch20 'compare' loops drop per-architecture build failures — exit status is the last arch only |
| W1-057 | medium | `code/core/common/cuda_arch.mk:90` | sm_120 mislabeled 'Grace-Blackwell GB200 (CC 12.0)' — GB200 GPUs are CC 10.0; following the label builds binaries that cannot run on GB200 |
| W1-067 | medium | `code/labs/custom_vs_cublas/Makefile:12` | ARCH knob is dead: default sm_100 makes the auto-detect unreachable and the gencode is hardcoded to compute_100a regardless of ARCH |
| W1-111 | low | `code/core/scripts/build_tma_demos.sh:34` | build_tma_demos.sh resolves the repo root to code/core, so both builds fail unconditionally (and the make targets don't exist anyway) |
| W1-112 | low | `code/core/scripts/run_warp_specialization_ci.sh:7` | run_warp_specialization_ci.sh computes REPO_ROOT as code/core, so the CI test wrapper always fails with file-not-found |
| W1-115 | low | `code/labs/custom_vs_cublas/cutlass_gemm/CMakeLists.txt:52` | -D_GLIBCXX_USE_CXX11_ABI=0 is wrong for the pinned torch 2.9.1+cu130 wheels (CXX11 ABI=1) |
| W1-124 | low | `code/requirements_latest.txt:18` | README quickstart 'pip install -r requirements_latest.txt' cannot resolve torch==2.9.1+cu130 — the cu130 index line is commented out |
| W1-125 | low | `code/setup.sh:3897` | Final summary claims cuDNN 9.15.1.9-1 was installed, contradicting the script's own 9.16.0.29-1 install-and-hold |

## P02: Harness, anti-cheat and CI (13)

| ID | Source severity | Location at reviewed revision | Finding |
| --- | --- | --- | --- |
| W1-006 | high | `.github/workflows/benchmark-validation.yml:95` | CI runs only ~28 of 267 test files; all anti-cheat/verification suites are unwired |
| W1-021 | high | `code/core/benchmark/comparison.py:578` | Chapter 'speedup' metric mapping flips timing.mean_ms to HIGHER_IS_BETTER, inverting regression detection |
| W1-040 | high | `code/tests/test_anti_cheat_protections.py:139` | Anti-cheat tests assert the config values the test itself just set |
| W1-041 | high | `code/tests/test_protection_effectiveness.py:337` | 'Protection catches attack' tests are tautologies that never invoke the protections |
| W1-054 | medium | `code/core/benchmark/comparison.py:643` | Chapter metric 'target' VALUES are misused as regression-threshold PERCENTAGES |
| W1-061 | medium | `code/core/harness/benchmark_harness.py:3506` | Harness fabricates Gaussian timing samples and reports them as measured raw times |
| W1-062 | medium | `code/core/harness/benchmark_harness.py:4887` | UnboundLocalError: allowed_antipatterns referenced when sync-detection is disabled but antipattern-detection is enabled |
| W1-063 | medium | `code/core/harness/benchmark_harness.py:2387` | torchrun tokens/s parser regex requires a literal backslash — throughput never extracted |
| W1-093 | medium | `code/tests/test_anti_cheat_edge_cases.py:1314` | 43 of 124 edge-case tests contain no assertions; several bodies are literally 'pass' |
| W1-095 | medium | `code/tests/test_blackwell_stack.py:105` | Blackwell 'validation suite' swallows every exception with print and can never fail |
| W1-126 | low | `code/tests/test_benchmark_regime_configs.py:19` | Module-wide CUDA skip disables pure-config regression tests that need no GPU |
| W1-127 | low | `code/tests/test_symmetric_memory_inference.py:61` | Gloo demo tests silently return (reporting PASS) unless symmetric memory and 2 GPUs are present |
| W1-128 | low | `code/tests/test_validate_benchmark_pairs_tools.py:448` | nvshmem pair-validation test asserts nothing on multi-GPU hosts |

## P03: Chapter CUDA, TMA and stream timing (20)

| ID | Source severity | Location at reviewed revision | Finding |
| --- | --- | --- | --- |
| W1-001 | critical | `code/ch10/tma_multicast_cluster.cu:168` | TMA multicast GEMM loads the B tile from the transposed location, producing wrong results |
| W1-008 | high | `code/ch08/baseline_thresholdtma.cu:74` | Baseline timing measures only host enqueue: events on legacy stream, kernels on non-blocking stream |
| W1-009 | high | `code/ch08/threshold_async_kernel.cuh:133` | cp.async wait_group off-by-one: each block's last tile is consumed before its copy is guaranteed complete |
| W1-010 | high | `code/ch10/tcgen05_warp_specialized.cu:81` | Tile swizzle drops/duplicates output tiles whenever grid_n is not a multiple of 8 |
| W1-011 | high | `code/ch11/streams_ordered_demo.cu:118` | Timed region records events on default stream while kernels run on non-blocking streams — measures launch overhead only |
| W1-012 | high | `code/ch11/streams_ordered_demo.cu:90` | No inter-stream dependencies: H2D copy, compute, and D2H copy race across three independent streams |
| W1-013 | high | `code/ch11/streams_overlap_demo.cu:221` | Vectorization 'speedup' compares a baseline doing 2 kernel launches/iter against optimized doing 1 — ~2x inflated |
| W1-014 | high | `code/ch11/streams_warp_specialized_demo.cu:186` | Same wrong-stream event timing: stop event does not wait for kernels launched on non-blocking streams |
| W1-015 | high | `code/ch12/optimized_cuda_graphs.cu:107` | Graph-replay benchmark records events on default stream while replays run on non-blocking stream |
| W1-045 | medium | `code/ch06/occupancy_api.cu:62` | Occupancy queried with 0 dynamic shared memory but kernel launched with block_size*4 bytes; divergent __syncthreads in partial block |
| W1-046 | medium | `code/ch06/optimized_ilp_low_occupancy_vec4_impl.cuh:191` | Host verification uses a 4-wide coefficient pattern against the kernel's 8-wide pattern, so the demo always reports Incorrect and exits 1 |
| W1-047 | medium | `code/ch10/tma_2d_pipeline_blackwell.cu:11` | Header and summary claim 128B swizzle and L2 promotion, but the code encodes SWIZZLE_NONE and L2_PROMOTION_NONE |
| W1-048 | medium | `code/ch11/streams_ordered_demo.cu:73` | Chunking mismatch: array split into 8 pipeline chunks but only 3 are ever transferred or processed |
| W1-049 | medium | `code/ch11/streams_overlap_demo.cu:278` | scale_kernel_async is launched with the Float8-sized grid on Blackwell, so it only processes half the array and reports ~2x inflated bandwidth |
| W1-050 | medium | `code/ch11/streams_overlap_demo.cu:193` | 'Three-way pipeline' overlap benchmark runs copy and compute on the same stream because stream2 aliases stream1 |
| W1-051 | medium | `code/ch11/streams_overlap_demo.cu:296` | Default-stream cudaMemcpy races with kernels still running on the non-blocking stream |
| W1-102 | low | `code/ch02/memory_transfer_pcie_demo.cu:98` | PCIe 5.0 "~128 GB/s theoretical max" is unreachable for this benchmark's sequential-transfer metric |
| W1-103 | low | `code/ch07/fp8_32byte_loads_demo.cu:105` | Comment claims 0x3C initializes FP8 buffer to 1.0, but 0x3C is 1.5 in E4M3 |
| W1-104 | low | `code/ch08/threshold_tma_kernel.cuh:50` | aligned_size_t<16> constructed with a size that is not a multiple of 16 for ragged tails (undefined behavior) |
| W1-105 | low | `code/ch10/optimized_dsmem_reduction_warp_specialized.cu:11` | Header claims 'No atomics needed - single writer per cluster' and an 8-CTA cluster; the kernel uses atomicAdd from all 4 CTAs |

## P04: Blackwell kernel and feature correctness (12)

| ID | Source severity | Location at reviewed revision | Finding |
| --- | --- | --- | --- |
| W1-002 | critical | `code/labs/custom_vs_cublas/tcgen05_cluster.cu:211` | Pipelined mainloop reuses smem stage without waiting on empty_barrier: TMA overwrites operands still being read by in-flight tcgen05 MMA |
| W1-019 | high | `code/core/benchmark/blackwell_optimizations/blackwell_optimizations/test_all_features.cu:275` | Same 4096-thread block bug: the 'comprehensive Blackwell feature' benchmark always fails to launch; feature claims are also unbacked |
| W1-020 | high | `code/core/benchmark/blackwell_optimizations/blackwell_optimizations/test_tma.cu:255` | Launch config uses a 64x64 = 4096-thread block, exceeding the 1024 threads/block hardware limit: every kernel launch fails |
| W1-026 | high | `code/labs/custom_vs_cublas/tcgen05_cluster.cu:309` | Odd grid_m is padded up for the cluster launch but the kernel has no tile bounds guard: padded CTAs write a full tile out of bounds |
| W1-027 | high | `code/labs/custom_vs_cublas/tcgen05_gemm.cu:177` | matmul_tcgen05_bias_silu never applies the bias: it computes silu(A@B^T), not silu(A@B^T + bias) |
| W1-065 | medium | `code/labs/blackwell_matmul/grace_blackwell_kernels.cu:529` | Dead #ifdef on enum attributes leaves TMA gated at CC >= 10, wrongly refusing Hopper (TMA is SM90) |
| W1-068 | medium | `code/labs/custom_vs_cublas/experimental/tcgen05_tma_multicast.cu:178` | TMA multicast issued with no fence_barrier_init/cluster_sync after mbarrier init and no cross-CTA empty synchronization |
| W1-069 | medium | `code/labs/custom_vs_cublas/run_lab.py:220` | False hardware claim sells the race as an optimization: 'MMA hardware handles dependencies internally' and '+43% improvement' |
| W1-070 | medium | `code/labs/custom_vs_cublas/tcgen05_cluster.cu:6` | File header and lab stage labels claim TMA multicast, but the kernel uses plain SM90_TMA_LOAD for both operands — no multicast exists |
| W1-109 | low | `code/core/benchmark/blackwell_optimizations/blackwell_optimizations/test_tma.cu:214` | Wrong hardware claim: 'TMA requires Blackwell (SM 10.0+)' — TMA was introduced on Hopper (SM 9.0) |
| W1-116 | low | `code/labs/custom_vs_cublas/experimental/tcgen05_multicast.cu:77` | Uses a nonexistent 'cyclic_multicast' namespace — the file cannot compile — and its 'multicast' path is dead code anyway |
| W1-117 | low | `code/labs/custom_vs_cublas/run_lab.py:463` | Arithmetic-intensity figure is wrong by 4x: 4096^3 fp16 GEMM is ~1365 FLOPs/byte, not 5461 |

## P05: Nanochat engine and benchmarks (6)

| ID | Source severity | Location at reviewed revision | Finding |
| --- | --- | --- | --- |
| W1-003 | critical | `code/labs/nanochat_fullstack/nanochat/engine.py:584` | CUDA-graph decode replays a stale KV-cache write position, producing wrong tokens after the first replay |
| W1-034 | high | `code/labs/nanochat_fullstack/nanochat/engine.py:491` | Persistent-decode path runs forward on a side stream with no synchronization against the consumer stream |
| W1-035 | high | `code/labs/nanochat_fullstack/nanochat/engine.py:919` | non_blocking D2H copy into pinned memory is read by the host immediately, without synchronization |
| W1-036 | high | `code/labs/nanochat_fullstack/scripts/bench_b200_flags.py:125` | 'persistent' mode decode is timed with events on the default stream while the work runs on the engine's side stream |
| W1-085 | medium | `code/labs/nanochat_fullstack/benchmark_incremental_optimizations.py:159` | '+ enable_persistent_decode' and '+ use_cuda_graphs' benchmark rows are no-ops: the flags gate Engine paths but the benchmark never uses the Engine |
| W1-121 | low | `code/labs/nanochat_fullstack/test_new_optimizations.py:87` | CTA clustering 'validation' asserts nothing and prints tests-passed unconditionally |

## P06: Distributed training and MoE (17)

| ID | Source severity | Location at reviewed revision | Finding |
| --- | --- | --- | --- |
| W1-004 | critical | `code/labs/train_distributed/optimized_zero2_multigpu.py:147` | Optimized ZeRO-2 never runs the optimizer: overlap_with_ddp=True makes step() a no-op |
| W1-028 | high | `code/labs/dynamic_router/vllm_runner.py:526` | Router fed tokens-per-step EMA as its ttft_ms metric in the vLLM routing demo |
| W1-029 | high | `code/labs/fullstack_cluster/moe_hybrid_ep_common.py:727` | Overlap mode shares reuse-buffers between the comm stream and the default stream — cross-stream data race corrupts expert outputs |
| W1-032 | high | `code/labs/moe_optimization_journey/level4_triton.py:285` | Async pinned D2H copy of expert counts read without synchronization (race) |
| W1-033 | high | `code/labs/moe_parallelism/scenarios.py:217` | Five of seven planner scenarios are sized for a 128-GPU DGX-A100 cluster but bound to the 576-GPU GB200 cluster |
| W1-075 | medium | `code/labs/fullstack_cluster/moe_hybrid_ep_common.py:1112` | 'Replicated' router/projection parameters are never synchronized at init, so averaging their gradients is incoherent |
| W1-076 | medium | `code/labs/fullstack_cluster/moe_hybrid_ep_common.py:857` | Collective participation is gated on per-rank data-dependent token counts — asymmetric zero counts hang all ranks |
| W1-078 | medium | `code/labs/moe_cuda/optimized_kv_transfer.py:130` | Overlapped KV-transfer pipeline lacks the compute-after-copy dependency (WAR race across iterations) |
| W1-079 | medium | `code/labs/moe_cuda_ptx/moe_cuda_ptx_common.py:1260` | moe_layer verification tolerance loosened 10x based on a quantization step the code explicitly does not perform |
| W1-080 | medium | `code/labs/moe_optimization_journey/level2_fp8.py:2` | Journey wrapper benchmarks advertise optimizations (FP8, streams, sorting, grouping) their level never enables |
| W1-081 | medium | `code/labs/moe_optimization_journey/level6_native_fp8.py:195` | FP8 hidden activations saturate with scale=1.0, and utilization is measured against the BF16 peak |
| W1-082 | medium | `code/labs/moe_optimization_journey/triton_fused_moe.py:102` | Fused MoE Triton kernel computes a tiny slice of the math yet reports full-GEMM TFLOPS |
| W1-083 | medium | `code/labs/moe_optimization_journey/triton_kernels.py:140` | Demo Triton kernels compute wrong math: FFN kernel reuses one hidden tile for all N blocks; grouped GEMM has no per-group pointer offsets |
| W1-084 | medium | `code/labs/moe_parallelism/plan.py:379` | EP all-to-all cost charged even at EP=1, and hotspot text hardcodes HDR100 on GB200 clusters |
| W1-091 | medium | `code/labs/train_distributed/optimized_zero2.py:77` | Optimized single-GPU ZeRO-2 requires a DDP comm hook that does not exist in any PyTorch release |
| W1-114 | low | `code/labs/cache_aware_disagg_inference/cache_aware_disagg_multigpu_common.py:367` | Peer KV handoff sends a non-contiguous slice for batch_size > 1 |
| W1-120 | low | `code/labs/moe_cuda/optimized_router.py:88` | AdaptiveTopKMoE applies expert 0's FC2 weights to every routed expert |

## P07: Attention, decode and async input (12)

| ID | Source severity | Location at reviewed revision | Finding |
| --- | --- | --- | --- |
| W1-016 | high | `code/ch18/flashmla_kernel.cu:27` | FlashMLA book sample computes softmax of per-thread partial dot products - the attention math is wrong |
| W1-022 | high | `code/core/common/async_input_pipeline.py:110` | Prefetcher allocates batches on the copy stream but never record_stream()s them: cross-stream use-after-free race |
| W1-039 | high | `code/labs/persistent_decode/persistent_decode_ext.cu:38` | Shared-memory reduction race: smem[0] read without a barrier before buffer reuse |
| W1-066 | medium | `code/labs/cudnn_sdpa_bench/baseline_flash_sdp.py:51` | --backend cudnn never pins cuDNN: it builds the same fallback preference list as auto, so the cuDNN-vs-Flash comparison can silently measure Flash twice |
| W1-071 | medium | `code/labs/flashattention4/README.md:27` | 'Current validated' 14.45x claim contradicted by the repo's latest recorded B200 run |
| W1-072 | medium | `code/labs/flashattention4/flashattention4_common.py:659` | TFLOPs inflated ~2x for alibi/softcap modes: FLOPs counted dense, kernels run causal |
| W1-073 | medium | `code/labs/flashattention_gluon/flashattention_gluon_common.py:102` | Flash-attention kernel masks K/V loads but not scores; padded keys get softmax weight |
| W1-074 | medium | `code/labs/flexattention/flex_attention_cute.py:61` | flash-attn CuTe called with [B, H, S, D] tensors but expects [B, S, H, D] |
| W1-088 | medium | `code/labs/persistent_decode/optimized_persistent_decode_triton.py:83` | Triton persistent decode silently drops sequences when num_programs < batch |
| W1-089 | medium | `code/labs/persistent_decode/paged_kv_offload_common.py:278` | Pinned staging buffer overwritten while non-blocking H2D copy may still be in flight |
| W1-090 | medium | `code/labs/persistent_decode/tma_extension.py:58` | "Native TMA" extension never uses the TMA engine — 4-byte thread-scope cp.async at best |
| W1-118 | low | `code/labs/decode_optimization/baseline_decode_warp_specialized.py:1` | "Warp-specialized Triton decode" pair contains no Triton and no warp specialization |

## P08: FP4/FP8 numerics and verification (10)

| ID | Source severity | Location at reviewed revision | Finding |
| --- | --- | --- | --- |
| W1-030 | high | `code/labs/kv_cache_compression/optimized_kv_cache_nvfp4.py:79` | Verification tolerance rtol=1.0, atol=10.0 accepts any output including all zeros |
| W1-031 | high | `code/labs/kv_cache_compression/optimized_kv_cache_nvfp4.py:29` | NVFP4BlockScaling called with kwargs the TE recipe does not accept — TypeError on any machine with Transformer Engine |
| W1-037 | high | `code/labs/nvfp4_gemm/local_eval_submission.py:86` | Correctness verify compares the output tensor with itself for case0/case2 (aliasing tautology) |
| W1-038 | high | `code/labs/nvfp4_gemv/local_eval.py:169` | NameError: `sys` is never imported on the module execution path that actually runs |
| W1-077 | medium | `code/labs/kv_cache_compression/optimized_kv_cache_nvfp4.py:90` | kv_cache.compression_ratio metric (4x/2x) is fabricated — the KV cache is BF16 in both paths |
| W1-086 | medium | `code/labs/nvfp4_group_gemm/baseline_nvfp4_group_gemm.py:40` | Baseline and optimized group-GEMM targets run the same custom kernel — verification is self-referential |
| W1-087 | medium | `code/labs/ozaki_scheme/README.md:48` | "Verification-clean within rtol=1e-2, atol=1e-2" claim is vacuous for this workload's output scale |
| W1-119 | low | `code/labs/kv_cache_compression/baseline_kv_cache.py:1` | Baseline docstrings claim MXFP8 block scaling but the code uses per-tensor DelayedScaling |
| W1-122 | low | `code/labs/nvfp4_gemv/optimized_submission.py:146` | Packed-scale cache keyed on data_ptr/_version can silently serve stale scales after allocator address reuse |
| W1-123 | low | `code/labs/nvfp4_group_gemm/README.md:56` | Directory Layout lists cutlass_extension.py, which does not exist |

## P09: Hardware specifications and metric math (8)

| ID | Source severity | Location at reviewed revision | Finding |
| --- | --- | --- | --- |
| W1-018 | high | `code/core/benchmark/benchmark_peak.py:200` | Peak HBM (and L2) bandwidth undercounts copy traffic by 2x — measured 'peak' then becomes the validation target |
| W1-024 | high | `code/docs/gb300-sol-roofline.md:11` | GB300 FP8/BF16 dense peak specs inflated 1.5x across all roofline docs |
| W1-053 | medium | `code/core/benchmark/benchmark_peak.py:360` | FP4 'peak TFLOPS' builds an invalid recipe: NVFP4BlockScaling passed as a DelayedScaling kwarg |
| W1-055 | medium | `code/core/benchmark/detect_sm.py:31` | Architecture table labels CC 12.x as 'Grace-Blackwell GB200/GB300' — wrong hardware spec, contradicted elsewhere in the repo |
| W1-056 | medium | `code/core/benchmark/metrics.py:57` | B200 peak FLOPS specs wrong (FP8 2500 'sparse', FP16 tensor 1250) — roofline efficiency percentages inflated ~1.8x |
| W1-058 | medium | `code/core/diagnostics/microbench.py:344` | tensor_core_bench fp8 path always crashes on modern PyTorch (randn/matmul unsupported for float8) |
| W1-094 | medium | `code/tests/test_benchmark_metrics.py:48` | CI-gated B200 spec test is too loose to catch wrong hardware constants it currently blesses |
| W1-108 | low | `code/core/benchmark/benchmark_peak.py:701` | NVLink bandwidth mixes GiB and GB — result underreported by ~7.4% |

## P10: CLI, LLM, parsing and monitoring (10)

| ID | Source severity | Location at reviewed revision | Finding |
| --- | --- | --- | --- |
| W1-017 | high | `code/cli/aisp.py:267` | Default `aisp` invocation imports nonexistent module cli.tui and crashes |
| W1-023 | high | `code/core/llm.py:209` | max_tokens is floored at 131072, breaking every OpenAI/Anthropic call and ignoring all overrides |
| W1-059 | medium | `code/core/engine.py:978` | ai.ask imports nonexistent core.book.get_book_citations; ImportError silently swallowed so citations are always empty |
| W1-060 | medium | `code/core/engine.py:1030` | ai.explain summary/bullet regexes double-escaped — explanation is the whole citation blob, key_points never parse bullets |
| W1-064 | medium | `code/core/perf_core_base.py:1313` | get_nvlink_status regexes are double-escaped and never match — NVLink always reported as 0 links / 0 GB/s |
| W1-092 | medium | `code/monitoring/prometheus_exporter.py:139` | Exporter emits duplicate # HELP/# TYPE lines per GPU series - Prometheus rejects the whole scrape on any multi-GPU host |
| W1-106 | low | `code/cli/commands/profiling.py:419` | `aisp profile flame` claims to generate a flame graph but is a no-op |
| W1-107 | low | `code/core/analysis/distributed_analysis.py:126` | NVLink-active check compares capitalized 'Active' against a lowercased string — branch can never run |
| W1-110 | low | `code/core/perf_core_base.py:1421` | CUTLASS version parse uses r"\\d+" — always IndexError, version reported as None with a spurious warning |
| W1-113 | low | `code/core/whatif.py:22` | FP8 what-if scenario ships an invalid torch.autocast code example |

## P11: Documentation and operational commands (9)

| ID | Source severity | Location at reviewed revision | Finding |
| --- | --- | --- | --- |
| W1-042 | high | `docs/environment.md:12` | Environment doc prescribes CUDA 12.9/cu129 while the repo requires CUDA 13 |
| W1-043 | medium | `CONTRIBUTING.md:57` | B300 mislabeled as SM100; B300/Blackwell Ultra is compute capability 10.3 (sm_103) |
| W1-044 | medium | `CONTRIBUTING.md:314` | CONTRIBUTING claims MIT License; the repo is Apache 2.0 |
| W1-096 | medium | `docs/appendix.md:856` | NCCL_NTHREADS and NCCL_BUFFSIZE described with wrong semantics and wrong defaults |
| W1-097 | medium | `docs/appendix.md:539` | cp.async attributed to Hopper/Blackwell; it is an Ampere (CC 8.0) feature |
| W1-098 | medium | `docs/appendix.md:191` | NVL72's 130 TB/s is aggregate NVLink bandwidth, not bisection bandwidth |
| W1-099 | medium | `docs/tooling-and-profiling.md:45` | ncu/nsys commands use metric and trace names that do not exist |
| W1-100 | medium | `docs/tooling-and-profiling.md:14` | Every scripted workflow in the profiling guide references nonexistent files |
| W1-101 | low | `CONTRIBUTING.md:36` | Setup and test commands in CONTRIBUTING reference nonexistent paths |

## Source refutation

**W1-R001:** The source review refuted the claim that the configured pytest timeout is inert because the external plugin is absent. It identified built-in timeout support in `code/conftest.py`. Preserve that finding and rationale; do not install a plugin as an assumed fix. This planning task did not separately run timeout verification.

## Closure evidence

Use the ledger to record reproduction, final fix design, changed files/revision, exact commands, attempts, outputs, remaining hardware gates, and independent review. Source verification, GPU correctness, performance acceptance and documentation are distinct. Consult each source record for its original suggested fix and verifier notes; the suggestions are not pre-approved implementations.
