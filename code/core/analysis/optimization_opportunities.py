"""Rank benchmark results by next optimization opportunity.

The benchmark harness already proves whether an optimization worked. This module
answers the follow-up question: which target deserves the next profiling or
experiment slot?
"""

from __future__ import annotations

import copy
import json
import math
import re
import shlex
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_MIN_SPEEDUP = 1.10
DEFAULT_TARGET_SPEEDUP = 1.50
DEFAULT_MIN_MEMORY_SAVINGS_PCT = 10.0
DEFAULT_SLOW_BASELINE_MS = 100.0
DEFAULT_PORTFOLIO_BUDGET = 5
PRIORITY_RANK = {"high": 3, "medium": 2, "low": 1}
FRONTIER_CATALOG_KEYS = ("target_catalog", "available_targets", "benchmark_targets")

OPTIMIZATION_PRIMITIVE_SPECS = [
    {
        "primitive": "cuda_graph_replay",
        "terms": ["cuda graph", "cudagraph", "graph replay", "make_graphed_callables"],
        "why": "CUDA Graph replay can remove launch overhead from stable-shape hot loops.",
        "transfer_question": "Can the recipient target expose a warm static-shape region with identical correctness checks?",
    },
    {
        "primitive": "torch_compile_reduce_overhead",
        "terms": ["torch.compile", "reduce-overhead", "inductor", "compile"],
        "why": "Compilation can fuse framework overhead or small kernels when shapes are stable enough.",
        "transfer_question": "Does the recipient have repeated Python/framework overhead before the dominant kernel work?",
    },
    {
        "primitive": "tma_pipeline",
        "terms": ["tma", "tensor memory accelerator", "async copy", "cp.async"],
        "why": "TMA and async copy pipelines can hide memory movement behind compute.",
        "transfer_question": "Does the recipient move tile-shaped data with enough reuse to amortize staging?",
    },
    {
        "primitive": "fp8_fp4_precision_path",
        "terms": ["fp8", "fp4", "nvfp4", "mxfp8", "block scale", "block_scaled"],
        "why": "Emerging precision paths can unlock Tensor Core throughput when tolerances allow.",
        "transfer_question": "Can the recipient preserve numerical tolerance under the same shape family?",
    },
    {
        "primitive": "cutlass_or_cublaslt_tiling",
        "terms": ["cutlass", "cublaslt", "tile shape", "tcgen"],
        "why": "Library tiling and Tensor Core scheduling choices often transfer across GEMM-like targets.",
        "transfer_question": "Does the recipient share the same matrix-shape family and precision constraints?",
    },
    {
        "primitive": "pinned_nonblocking_transfer",
        "terms": ["pinned", "non_blocking", "nonblocking", "pin_memory", "prefetch"],
        "why": "Pinned and nonblocking host/device transfers can reduce input stalls and overlap copies.",
        "transfer_question": "Does the recipient expose host-device transfer time separately from compute?",
    },
    {
        "primitive": "communication_overlap",
        "terms": ["overlap", "allreduce", "reduce_scatter", "allgather", "nccl", "nvshmem"],
        "why": "Communication overlap can turn exposed fabric time into useful compute time.",
        "transfer_question": "Can the recipient prove lower exposed collective time without numerical drift?",
    },
    {
        "primitive": "persistent_kernel",
        "terms": ["persistent", "persistent kernel", "warp specialized", "warp_specialized"],
        "why": "Persistent kernels can reduce launch/scheduling overhead for repeated device-side work.",
        "transfer_question": "Does the recipient have enough steady-state work to justify persistent residency?",
    },
    {
        "primitive": "kv_cache_layout",
        "terms": ["kv cache", "kv_cache", "paged", "page size", "prefix", "decode"],
        "why": "KV-cache layout changes often transfer across decode and attention-heavy targets.",
        "transfer_question": "Can the recipient measure HBM transactions and decode correctness under the same prompt shape?",
    },
    {
        "primitive": "vectorized_memory_access",
        "terms": ["vectorized", "coalesced", "float4", "int4", "contiguous", "stride"],
        "why": "Vectorized/coalesced memory access can improve useful bytes per memory transaction.",
        "transfer_question": "Does the recipient show memory-transaction waste or uncoalesced access in profiler counters?",
    },
]

COMPOUND_PRIMITIVE_SPECS = [
    {
        "compound": "graph_stabilized_kv_decode",
        "name": "Graph-stabilized KV decode loop",
        "primitives": ["kv_cache_layout", "cuda_graph_replay"],
        "rationale": "Decode targets with KV-cache pressure and stable replay regions can turn layout work into a lower-launch-overhead steady-state loop.",
        "acceptance_gate": "Accept only if the target shows clean evidence for both primitives, output verification passes, and launch/HBM guardrail metrics do not regress.",
    },
    {
        "compound": "precision_tile_co_design",
        "name": "Precision and tile co-design",
        "primitives": ["fp8_fp4_precision_path", "cutlass_or_cublaslt_tiling"],
        "rationale": "Emerging precision paths usually need tile and library scheduling choices to convert lower precision into measured throughput.",
        "acceptance_gate": "Accept only if both primitives are isolated first, numerical tolerance passes, and the combined variant beats the verified control.",
    },
    {
        "compound": "staged_vectorized_memory_movement",
        "name": "Staged and vectorized memory movement",
        "primitives": ["tma_pipeline", "vectorized_memory_access"],
        "rationale": "Targets that already stage data or vectorize accesses are candidates for combining async movement with cleaner memory transactions.",
        "acceptance_gate": "Accept only if profiler counters show better memory-transaction efficiency and wall time improves without correctness drift.",
    },
    {
        "compound": "host_fabric_overlap",
        "name": "Host/fabric overlap pipeline",
        "primitives": ["communication_overlap", "pinned_nonblocking_transfer"],
        "rationale": "Pinned host transfers and overlapped collectives can be paired when input staging and exposed fabric time sit in the same critical path.",
        "acceptance_gate": "Accept only if exposed transfer or collective time shrinks while end-to-end step output verification remains clean.",
    },
    {
        "compound": "persistent_graph_steady_state",
        "name": "Persistent graph steady state",
        "primitives": ["persistent_kernel", "cuda_graph_replay"],
        "rationale": "Repeated device-side work may benefit from combining persistent residency with host-side graph replay after shapes are proven stable.",
        "acceptance_gate": "Accept only if persistent occupancy, launch overhead, and correctness guardrails are all captured for the same workload contract.",
    },
    {
        "compound": "compiled_graph_hotpath",
        "name": "Compiled static hotpath with graph replay",
        "primitives": ["torch_compile_reduce_overhead", "cuda_graph_replay"],
        "rationale": "Compilation and graph replay attack different layers of overhead and should be tested together only after their static-shape assumptions match.",
        "acceptance_gate": "Accept only if compile warmup is excluded, both primitives are visible in artifacts, and the replayed variant beats a compiled-only control.",
    },
    {
        "compound": "decode_precision_tile_stack",
        "name": "Decode precision/tile stack",
        "primitives": [
            "kv_cache_layout",
            "fp8_fp4_precision_path",
            "cutlass_or_cublaslt_tiling",
        ],
        "rationale": "Attention/decode targets that touch KV layout and low-precision GEMM paths may need tile choices, cache layout, and tolerance checks designed together.",
        "acceptance_gate": "Accept only if each primitive is independently reversible and the combined stack passes decode correctness plus numerical tolerance gates.",
    },
    {
        "compound": "precision_tma_tile_stack",
        "name": "Precision-aware TMA tile stack",
        "primitives": [
            "tma_pipeline",
            "fp8_fp4_precision_path",
            "cutlass_or_cublaslt_tiling",
        ],
        "rationale": "Blackwell-style precision work can require async tile movement and library tile schedules to line up before Tensor Core throughput appears.",
        "acceptance_gate": "Accept only if the combined path improves kernel time after precision guardrails, TMA staging evidence, and tile-shape controls are recorded.",
    },
]

CROSS_LANE_BRIDGE_SPECS = [
    {
        "bridge": "decode_fabric_overlap",
        "name": "Decode/fabric overlap bridge",
        "signals": ["serving_decode_hotpath", "distributed_fabric"],
        "primitives": ["communication_overlap", "kv_cache_layout"],
        "rationale": "Distributed serving targets can hide collective or placement cost only when decode/KV timing and exposed fabric time are measured together.",
        "experiments": [
            "Separate prefill, decode, and exposed collective time in the same profile.",
            "Sweep overlap start point or rank placement while holding the decode prompt shape fixed.",
            "Compare TTFT, TPOT, and exposed collective time before changing model-parallel policy.",
        ],
        "acceptance_gate": "Accept only if decode metrics improve and exposed fabric time drops without output drift or worse p95 latency.",
    },
    {
        "bridge": "precision_attention_memory",
        "name": "Precision-aware attention memory bridge",
        "signals": ["attention_or_kv", "emerging_precision", "memory_movement"],
        "primitives": [
            "kv_cache_layout",
            "fp8_fp4_precision_path",
            "cutlass_or_cublaslt_tiling",
        ],
        "rationale": "Attention and KV-cache targets that also touch emerging precision need cache layout, tile shape, and numerical tolerance explored as one design space.",
        "experiments": [
            "Run a KV page/layout sweep under BF16 control before enabling FP8/FP4 variants.",
            "Pair each precision mode with one tile-shape family and one cache layout.",
            "Track HBM transactions, tokens per second, and numerical tolerance as co-primary metrics.",
        ],
        "acceptance_gate": "Accept only if the bridge beats BF16/control on latency or throughput while memory and numerical guardrails pass.",
    },
    {
        "bridge": "graph_captured_input_pipeline",
        "name": "Graph-captured input pipeline bridge",
        "signals": ["input_or_storage", "runtime_launch", "memory_movement"],
        "primitives": [
            "pinned_nonblocking_transfer",
            "cuda_graph_replay",
            "torch_compile_reduce_overhead",
        ],
        "rationale": "Input-path targets can waste GPU time even after kernel tuning; pairing pinned/nonblocking transfers with stable launch paths exposes whether input staging or host overhead dominates.",
        "experiments": [
            "Measure GPU idle time with pinned/nonblocking transfer enabled and disabled.",
            "Compare eager, compiled, and graph replay launch paths after input tensors are staged.",
            "Sweep queue depth and prefetch window with identical sample ordering.",
        ],
        "acceptance_gate": "Accept only if GPU idle time falls and verified outputs remain stable under the same input-order contract.",
    },
    {
        "bridge": "control_plane_decode_warm_pool",
        "name": "Control-plane decode warm-pool bridge",
        "signals": ["control_plane_or_disaggregation", "serving_decode_hotpath"],
        "primitives": ["cuda_graph_replay", "kv_cache_layout"],
        "rationale": "Serving systems with scheduler or disaggregation signals should connect startup/readiness timing to TTFT and steady decode behavior instead of optimizing those paths separately.",
        "experiments": [
            "Measure queue wait, readiness, TTFT, and steady decode in one run contract.",
            "Compare warm-pool and cold-start variants with the same prompt and batch shape.",
            "Keep graph replay or cache warmup cost visible instead of folding it into steady state.",
        ],
        "acceptance_gate": "Accept only if startup or queueing improves without masking TTFT, TPOT, or correctness regressions.",
    },
    {
        "bridge": "fabric_precision_tile_bridge",
        "name": "Fabric precision/tile bridge",
        "signals": ["distributed_fabric", "emerging_precision"],
        "primitives": [
            "communication_overlap",
            "fp8_fp4_precision_path",
            "cutlass_or_cublaslt_tiling",
        ],
        "rationale": "Multi-GPU precision wins can disappear when collective shape, tile shape, and overlap are not tuned against the same tensor-parallel contract.",
        "experiments": [
            "Compare BF16 and FP8/FP4 tensor-parallel runs with the same rank topology.",
            "Sweep collective bucket size and tile shape as separate variables before combining them.",
            "Track effective bandwidth, collective latency, kernel time, and numerical drift together.",
        ],
        "acceptance_gate": "Accept only if precision/tile speedup survives the distributed guardrails and exposed collective time does not grow.",
    },
]

MOTIF_SPECS = [
    {
        "motif": "attention_kv_layout",
        "title": "Transfer attention and KV-cache layout experiments",
        "terms": ["attention", "prefill", "decode", "kv"],
        "reason": "Decode, prefill, and cache-heavy targets often share paging, prefix reuse, and memory-layout bottlenecks.",
        "transfer_experiments": [
            "Sweep KV page/block size and cache layout across the support targets.",
            "Compare Flash/FlexAttention, prefix-aware reuse, and decode-specific fused kernels under the same workload contract.",
            "Measure token throughput and HBM transaction efficiency before changing model-level policy.",
        ],
        "acceptance_gate": "At least one support target improves >=5% with identical output verification and no regression on the prototype target.",
    },
    {
        "motif": "moe_router_grouped_gemm",
        "title": "Transfer MoE routing and grouped-GEMM experiments",
        "terms": ["moe", "expert", "router"],
        "reason": "MoE targets frequently move together when routing, padding, capacity factor, or grouped-GEMM batching changes.",
        "transfer_experiments": [
            "Sweep capacity factor and top-k routing with a fixed token distribution.",
            "Compare grouped GEMM batching, expert sorting, and padding/quantization interactions.",
            "Validate load-balance metrics alongside latency so routing wins do not hide quality or tail regressions.",
        ],
        "acceptance_gate": "Prototype routing changes must improve one latency metric and preserve expert-load balance on every support target.",
    },
    {
        "motif": "communication_overlap",
        "title": "Transfer communication-overlap experiments",
        "terms": ["nccl", "allreduce", "gradient", "comm", "nvlink", "nvshmem"],
        "reason": "Gradient, collective, and topology targets often share bucket sizing, overlap, and protocol selection tradeoffs.",
        "transfer_experiments": [
            "Sweep bucket size, reduce-scatter/all-gather decomposition, and overlap start point.",
            "Compare NCCL protocol choices against NVSHMEM or topology-aware variants where available.",
            "Track exposed communication time, not just end-to-end step time.",
        ],
        "acceptance_gate": "Communication optimization must reduce exposed collective time without increasing verified step output drift.",
    },
    {
        "motif": "memory_layout_transfer",
        "title": "Transfer memory-layout and prefetch experiments",
        "terms": ["copy", "memory", "hbm", "prefetch", "tma", "lookup", "cache"],
        "reason": "Memory-bound targets often respond to the same transaction-width, prefetch, and layout-pretranspose ideas.",
        "transfer_experiments": [
            "Compare vectorized loads/stores, layout pretransposition, and TMA or async prefetch variants.",
            "Measure requested bytes, actual bytes, and cache-hit behavior before accepting wall-clock wins.",
            "Try a read-mostly layout and a write-coalesced layout as separate one-variable experiments.",
        ],
        "acceptance_gate": "A layout transfer is accepted only when profiler counters show lower memory waste and timing improves.",
    },
    {
        "motif": "precision_tile_autotune",
        "title": "Transfer precision and tile-autotune experiments",
        "terms": ["gemm", "matmul", "fp4", "fp8", "nvfp4", "tcgen", "cutlass"],
        "reason": "GEMM-like targets often have non-obvious winning tile shapes and precision modes that transfer across nearby kernels.",
        "transfer_experiments": [
            "Sweep cublasLt, CUTLASS, or Triton tile shapes with the same input shape family.",
            "Compare BF16, FP8, FP4/NVFP4, and block-scaled variants where correctness tolerances allow.",
            "Keep rank-0 and best-of-N candidate timing separate so autotune overhead does not pollute claims.",
        ],
        "acceptance_gate": "Publish only candidates that pass correctness and beat the current best median, not just the first autotune rank.",
    },
    {
        "motif": "launch_graph_persistence",
        "title": "Transfer launch, graph, and persistence experiments",
        "terms": ["compile", "graph", "cuda graph", "launch", "persistent"],
        "reason": "Small kernels and decode loops often share framework overhead, CUDA Graph replay, and persistent-kernel tradeoffs.",
        "transfer_experiments": [
            'Compare eager, `torch.compile(mode="reduce-overhead")`, CUDA Graph replay, and persistent-kernel variants.',
            "Separate host-frame timing from pure GPU timing before claiming kernel improvements.",
            "Check cold-start and warm steady-state runs so graph capture does not hide initialization cost.",
        ],
        "acceptance_gate": "A launch-path transfer must improve warm steady-state and report any cold-start penalty explicitly.",
    },
    {
        "motif": "input_pipeline_overlap",
        "title": "Transfer input-pipeline overlap experiments",
        "terms": ["dataloader", "storage", "gds", "fio", "input"],
        "reason": "Input and storage targets often share queue depth, pinned memory, and overlap bottlenecks.",
        "transfer_experiments": [
            "Sweep queue depth, prefetch factor, pinned memory, and async copy overlap.",
            "Compare local storage, GDS, and networked storage under the same batch-shape contract.",
            "Track idle GPU time so storage wins are tied to useful compute overlap.",
        ],
        "acceptance_gate": "Input-pipeline changes must lower GPU idle time and preserve sample ordering or reproducibility expectations.",
    },
    {
        "motif": "control_plane_scheduler",
        "title": "Transfer scheduler and control-plane experiments",
        "terms": ["scheduler", "kueue", "slinky", "control"],
        "reason": "Scheduler-path targets often share queueing, readiness, and startup timing bottlenecks that are invisible in kernel-only profiles.",
        "transfer_experiments": [
            "Measure queue wait, image/startup delay, readiness time, and first-token/first-step time separately.",
            "Compare scheduler path, placement constraints, and warm-pool behavior as one-variable experiments.",
            "Keep service-level latency and utilization together so faster startup does not degrade steady state.",
        ],
        "acceptance_gate": "Scheduler transfers must improve one startup or queueing metric without hurting steady-state benchmark output.",
    },
]

FRONTIER_SIGNAL_SPECS = [
    {
        "signal": "serving_decode_hotpath",
        "bonus": 13.0,
        "terms": [
            "decode",
            "prefill",
            "continuous_batching",
            "inference",
            "serving",
            "vllm",
            "sglang",
            "trtllm",
        ],
        "reason": "Serving and token hot paths tend to turn first evidence into directly useful latency or throughput experiments.",
    },
    {
        "signal": "distributed_fabric",
        "bonus": 12.0,
        "terms": [
            "multigpu",
            "nccl",
            "nvlink",
            "nvshmem",
            "allreduce",
            "allgather",
            "tensor_parallel",
            "pipeline_parallel",
            "fsdp",
            "ddp",
        ],
        "reason": "Multi-GPU and fabric targets expose overlap and topology wins that rarely show up in single-kernel evidence.",
    },
    {
        "signal": "attention_or_kv",
        "bonus": 11.0,
        "terms": ["attention", "flexattention", "flash", "prefill", "decode", "kv"],
        "reason": "Attention and KV-cache probes often reveal reusable layout, paging, and prefix-reuse experiments.",
    },
    {
        "signal": "emerging_precision",
        "bonus": 10.0,
        "terms": ["fp4", "nvfp4", "fp8", "mxfp8", "tcgen", "blackwell", "cutlass"],
        "reason": "New precision paths can unlock hardware-specific wins not represented by older benchmark evidence.",
    },
    {
        "signal": "memory_movement",
        "bonus": 9.0,
        "terms": [
            "memory",
            "hbm",
            "bandwidth",
            "copy",
            "prefetch",
            "tma",
            "cache",
            "symmetric_memory",
            "grace",
        ],
        "reason": "Memory movement probes are good early candidates because profiler counters make bottlenecks concrete.",
    },
    {
        "signal": "runtime_launch",
        "bonus": 8.0,
        "terms": ["compile", "cuda_graph", "graph", "persistent", "launch"],
        "reason": "Launch and graph probes can convert flat small-kernel timings into reusable runtime patterns.",
    },
    {
        "signal": "input_or_storage",
        "bonus": 7.0,
        "terms": ["storage", "gds", "dataloader", "fio", "input"],
        "reason": "Input-path probes can surface GPU idle time that kernel-only scans miss.",
    },
    {
        "signal": "control_plane_or_disaggregation",
        "bonus": 7.0,
        "terms": ["scheduler", "kueue", "slinky", "disaggregated", "rack"],
        "reason": "Control-plane and disaggregation probes connect benchmark output to deployment bottlenecks.",
    },
]

FRONTIER_BLUEPRINT_SPECS = {
    "serving_decode_hotpath": [
        {
            "name": "prefill_decode_phase_split",
            "hypothesis": "Separate prefill and decode timing, then optimize the phase that dominates TTFT or TPOT instead of averaging them together.",
            "knobs": [
                "prefill/decode batch mix",
                "KV cache residency policy",
                "continuous batching window",
            ],
            "primary_metrics": ["ttft_ms", "tpot_ms", "tokens_per_second"],
            "profiler_tools": ["nsys", "pytorch", "zymtrace"],
            "guardrails": ["output_correctness", "p95_latency", "gpu_utilization"],
        },
        {
            "name": "decode_graph_replay",
            "hypothesis": "Capture the steady decode loop with CUDA Graph replay after first-token setup to reduce launch overhead without hiding warmup cost.",
            "knobs": ["graph capture boundary", "static decode shape", "warmup iterations"],
            "primary_metrics": ["steady_state_decode_ms", "cuda_launch_count"],
            "profiler_tools": ["nsys"],
            "guardrails": ["cold_start_ms", "output_correctness"],
        },
    ],
    "distributed_fabric": [
        {
            "name": "topology_aware_overlap",
            "hypothesis": "Move collectives earlier and choose topology-aware placement so exposed fabric time overlaps with useful compute.",
            "knobs": ["bucket_size_mb", "overlap_start_layer", "rank_topology"],
            "primary_metrics": ["exposed_collective_ms", "step_time_ms", "nvlink_throughput"],
            "profiler_tools": ["nsys", "hta", "zymtrace"],
            "guardrails": ["numerical_drift", "gpu_idle_time"],
        },
        {
            "name": "nccl_nvshmem_protocol_compare",
            "hypothesis": "Compare NCCL protocol choices against NVSHMEM-style handoff for the same tensor shape before changing model parallel policy.",
            "knobs": ["collective_protocol", "message_size", "pipeline_depth"],
            "primary_metrics": ["collective_latency_ms", "effective_bandwidth_gbps"],
            "profiler_tools": ["nsys"],
            "guardrails": ["output_correctness", "rank_synchronization"],
        },
    ],
    "attention_or_kv": [
        {
            "name": "kv_page_layout_sweep",
            "hypothesis": "Sweep KV page/block size and cache layout to reduce HBM waste while preserving decode correctness.",
            "knobs": ["page_size", "block_size", "prefix_reuse_policy"],
            "primary_metrics": ["tokens_per_second", "hbm_transactions", "cache_hit_rate"],
            "profiler_tools": ["ncu", "nsys", "zymtrace"],
            "guardrails": ["output_correctness", "memory_capacity"],
        },
        {
            "name": "flex_flash_attention_compare",
            "hypothesis": "Compare Flash/FlexAttention and fused decode kernels under the same shape contract before optimizing model-level policy.",
            "knobs": ["attention_backend", "sequence_length", "causal_mask_shape"],
            "primary_metrics": ["attention_kernel_ms", "dram_throughput_pct"],
            "profiler_tools": ["ncu", "zymtrace"],
            "guardrails": ["numerical_tolerance", "peak_memory_gb"],
        },
    ],
    "emerging_precision": [
        {
            "name": "fp8_fp4_block_scale_sweep",
            "hypothesis": "Sweep FP8/FP4 block scaling and tile shape together because precision wins can disappear with the wrong memory layout.",
            "knobs": ["precision", "block_scale_shape", "tile_shape"],
            "primary_metrics": ["median_wall_time_ms", "tensor_core_utilization"],
            "profiler_tools": ["ncu", "zymtrace"],
            "guardrails": ["numerical_tolerance", "fallback_kernel_count"],
        },
    ],
    "memory_movement": [
        {
            "name": "transaction_width_prefetch_sweep",
            "hypothesis": "Increase useful bytes per transaction with vectorized access, layout pretranspose, or prefetch before changing algorithmic structure.",
            "knobs": ["vector_width", "layout_order", "prefetch_distance"],
            "primary_metrics": ["dram_bytes_per_output", "l2_hit_rate", "median_wall_time_ms"],
            "profiler_tools": ["ncu", "nsys", "zymtrace"],
            "guardrails": ["output_correctness", "peak_memory_gb"],
        },
    ],
    "runtime_launch": [
        {
            "name": "compile_graph_persistence_matrix",
            "hypothesis": "Compare eager, torch.compile, CUDA Graph replay, and persistent kernels while reporting cold-start separately from warm steady state.",
            "knobs": ["runtime_mode", "capture_boundary", "persistent_kernel"],
            "primary_metrics": ["warm_latency_ms", "cold_start_ms", "cuda_launch_count"],
            "profiler_tools": ["nsys", "pytorch", "zymtrace"],
            "guardrails": ["output_correctness", "compile_time_s"],
        },
    ],
    "input_or_storage": [
        {
            "name": "input_overlap_queue_depth",
            "hypothesis": "Sweep queue depth, pinned memory, and async copy overlap to reduce GPU idle time without changing sample order.",
            "knobs": ["queue_depth", "prefetch_factor", "pinned_memory"],
            "primary_metrics": ["gpu_idle_pct", "input_wait_ms", "samples_per_second"],
            "profiler_tools": ["nsys", "pytorch"],
            "guardrails": ["sample_ordering", "host_memory_gb"],
        },
    ],
    "control_plane_or_disaggregation": [
        {
            "name": "placement_warm_pool_probe",
            "hypothesis": "Measure queue wait, image/startup delay, readiness time, and first-token latency separately before optimizing scheduler policy.",
            "knobs": ["placement_constraint", "warm_pool_size", "handoff_boundary"],
            "primary_metrics": ["queue_wait_ms", "readiness_ms", "first_token_ms"],
            "profiler_tools": ["nsys"],
            "guardrails": ["steady_state_throughput", "resource_utilization"],
        },
    ],
}

LANE_PREFERRED_SIGNALS = {
    "attention_kv_layout": "attention_or_kv",
    "moe_router_grouped_gemm": "serving_decode_hotpath",
    "communication_overlap": "distributed_fabric",
    "memory_layout_transfer": "memory_movement",
    "precision_tile_autotune": "emerging_precision",
    "launch_graph_persistence": "runtime_launch",
    "input_pipeline_overlap": "input_or_storage",
    "control_plane_scheduler": "control_plane_or_disaggregation",
}


@dataclass(frozen=True)
class BenchmarkCandidate:
    target: str
    chapter: str
    example: str
    status: str
    optimization_goal: str
    baseline_time_ms: float | None = None
    optimized_time_ms: float | None = None
    best_speedup: float | None = None
    memory_savings_pct: float | None = None
    category: str = ""
    rationale: str = ""
    best_optimization: str = ""
    baseline_file: str = ""
    artifact_count: int = 0
    story_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class OptimizationOpportunity:
    rank: int
    target: str
    priority: str
    opportunity_type: str
    score: float
    status: str
    optimization_goal: str
    baseline_time_ms: float | None
    optimized_time_ms: float | None
    best_speedup: float | None
    memory_savings_pct: float | None
    frontier_motif: str | None
    frontier_signals: list[str]
    frontier_score_breakdown: list[dict[str, Any]]
    source_terms: list[str]
    source_delta_terms: list[str]
    source_files: list[str]
    catalog_source: str
    optimization_primitives: list[dict[str, Any]]
    experiment_blueprints: list[dict[str, Any]]
    evidence: list[str]
    rationale: str
    recommended_experiments: list[str]
    next_command: str
    benchmark_run: dict[str, Any]


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _status_bucket(status: Any) -> str:
    normalized = str(status or "").strip().lower()
    aliases = {
        "ok": "succeeded",
        "success": "succeeded",
        "error": "failed",
        "skip": "skipped",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized.startswith("failed"):
        return "failed"
    if normalized.startswith("skipped"):
        return "skipped"
    return normalized or "unknown"


def _split_target(target: str) -> tuple[str, str]:
    if ":" not in target:
        return target, ""
    chapter, example = target.split(":", 1)
    return chapter.strip(), example.strip()


def _normal_goal(value: Any) -> str:
    goal = str(value or "speed").strip().lower()
    if goal in {"performance", "latency"}:
        return "speed"
    return goal or "speed"


def _optimized_time_ms(
    row: dict[str, Any], baseline_time_ms: float | None, speedup: float | None
) -> float | None:
    explicit = _to_float(row.get("best_optimized_time_ms") or row.get("optimized_time_ms"))
    if explicit is not None:
        return explicit
    if baseline_time_ms is not None and speedup is not None and speedup > 0.0:
        return baseline_time_ms / speedup
    return None


def _artifact_count(row: dict[str, Any]) -> int:
    artifacts = row.get("artifacts")
    if isinstance(artifacts, dict):
        return len([value for value in artifacts.values() if value])
    return sum(
        1
        for key, value in row.items()
        if isinstance(value, str) and value and key.endswith(("_rep", "_trace", "_json"))
    )


def _candidate_from_tier1_target(row: dict[str, Any]) -> BenchmarkCandidate:
    target = str(row.get("target") or "")
    chapter, example = _split_target(target)
    speedup = _to_float(row.get("best_speedup"))
    baseline = _to_float(row.get("baseline_time_ms"))
    story = row.get("story_metadata") if isinstance(row.get("story_metadata"), dict) else {}
    return BenchmarkCandidate(
        target=target,
        chapter=chapter,
        example=example,
        status=_status_bucket(row.get("status")),
        optimization_goal=_normal_goal(row.get("optimization_goal")),
        baseline_time_ms=baseline,
        optimized_time_ms=_optimized_time_ms(row, baseline, speedup),
        best_speedup=speedup,
        memory_savings_pct=_to_float(
            row.get("best_memory_savings_pct") or row.get("memory_savings_pct")
        ),
        category=str(row.get("category") or ""),
        rationale=str(row.get("rationale") or ""),
        best_optimization=str(row.get("best_optimization") or ""),
        baseline_file=str(row.get("baseline_file") or ""),
        artifact_count=_artifact_count(row),
        story_metadata=story,
    )


def _candidate_from_benchmark(chapter: str, row: dict[str, Any]) -> BenchmarkCandidate:
    example = str(row.get("name") or row.get("example") or "")
    target = str(row.get("target") or f"{chapter}:{example}")
    target_chapter, target_example = _split_target(target)
    chapter = target_chapter or chapter
    example = target_example or example
    speedup = _to_float(row.get("speedup") if "speedup" in row else row.get("best_speedup"))
    baseline = _to_float(row.get("baseline_time_ms"))
    story = row.get("story_metadata") if isinstance(row.get("story_metadata"), dict) else {}
    return BenchmarkCandidate(
        target=target,
        chapter=chapter,
        example=example,
        status=_status_bucket(row.get("status")),
        optimization_goal=_normal_goal(row.get("optimization_goal")),
        baseline_time_ms=baseline,
        optimized_time_ms=_optimized_time_ms(row, baseline, speedup),
        best_speedup=speedup,
        memory_savings_pct=_to_float(
            row.get("best_memory_savings_pct") or row.get("memory_savings_pct")
        ),
        category=str(row.get("category") or ""),
        rationale=str(row.get("rationale") or ""),
        best_optimization=str(row.get("best_optimization") or ""),
        baseline_file=str(row.get("baseline_file") or ""),
        artifact_count=_artifact_count(row),
        story_metadata=story,
    )


def _candidate_from_catalog_entry(entry: Any) -> BenchmarkCandidate:
    if isinstance(entry, str):
        row: dict[str, Any] = {"target": entry}
    elif isinstance(entry, dict):
        row = dict(entry)
    else:
        row = {}

    target = str(row.get("target") or "")
    if not target and (row.get("chapter") or row.get("example")):
        target = f"{row.get('chapter', '')}:{row.get('example', '')}"
    chapter, example = _split_target(target)
    story = row.get("story_metadata") if isinstance(row.get("story_metadata"), dict) else {}
    story = dict(story)
    story["frontier"] = True
    for key in (
        "source_terms",
        "source_delta_terms",
        "frontier_signal_matches",
        "optimization_primitives",
        "optimized_files",
        "catalog_source",
    ):
        if key in row:
            story[key] = row[key]
    return BenchmarkCandidate(
        target=target,
        chapter=chapter,
        example=example,
        status="frontier",
        optimization_goal=_normal_goal(row.get("optimization_goal")),
        category=str(row.get("category") or ""),
        rationale=str(
            row.get("rationale")
            or "Discovered runnable target has no evidence in this benchmark artifact."
        ),
        best_optimization=str(row.get("best_optimization") or ""),
        baseline_file=str(row.get("baseline_file") or ""),
        artifact_count=0,
        story_metadata=story,
    )


def _catalog_entries_from_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in (*FRONTIER_CATALOG_KEYS, "targets"):
            nested = value.get(key)
            if isinstance(nested, list | dict):
                return _catalog_entries_from_value(nested)
        entries: list[Any] = []
        for target, metadata in value.items():
            if isinstance(metadata, dict):
                entry = dict(metadata)
                entry.setdefault("target", target)
                entries.append(entry)
            else:
                entries.append({"target": target, "rationale": str(metadata or "")})
        return entries
    return []


def _frontier_candidates(
    payload: dict[str, Any], measured: list[BenchmarkCandidate]
) -> list[BenchmarkCandidate]:
    measured_targets = {candidate.target for candidate in measured if candidate.target}
    seen = set(measured_targets)
    frontier: list[BenchmarkCandidate] = []
    for key in FRONTIER_CATALOG_KEYS:
        for entry in _catalog_entries_from_value(payload.get(key)):
            candidate = _candidate_from_catalog_entry(entry)
            if not candidate.target or candidate.target in seen:
                continue
            frontier.append(candidate)
            seen.add(candidate.target)
    return frontier


def normalize_candidates(payload: dict[str, Any]) -> list[BenchmarkCandidate]:
    """Normalize supported benchmark artifact shapes into candidates.

    Supported shapes:
    - tier-1 `summary.json` with a top-level `targets` list
    - transformed analyzer data with a top-level `benchmarks` list
    - raw `benchmark_test_results.json` with nested `results[].benchmarks[]`
    - optional `target_catalog`, `available_targets`, or `benchmark_targets`
      entries for runnable targets missing from the evidence artifact
    """
    candidates: list[BenchmarkCandidate] = []

    targets = payload.get("targets")
    if isinstance(targets, list):
        for row in targets:
            if isinstance(row, dict):
                candidates.append(_candidate_from_tier1_target(row))
    elif isinstance(payload.get("benchmarks"), list):
        benchmarks = payload["benchmarks"]
        for row in benchmarks:
            if isinstance(row, dict):
                candidates.append(_candidate_from_benchmark(str(row.get("chapter") or ""), row))
    elif isinstance(payload.get("results"), list):
        results = payload["results"]
        for chapter_result in results:
            if not isinstance(chapter_result, dict):
                continue
            chapter = str(chapter_result.get("chapter") or "")
            for row in chapter_result.get("benchmarks", []) or []:
                if isinstance(row, dict):
                    candidates.append(_candidate_from_benchmark(chapter, row))

    candidates.extend(_frontier_candidates(payload, candidates))
    return candidates


def _contains_any(candidate: BenchmarkCandidate, terms: Iterable[str]) -> bool:
    story = candidate.story_metadata or {}
    source_terms = " ".join(str(item) for item in story.get("source_terms", []) or [])
    source_delta_terms = " ".join(str(item) for item in story.get("source_delta_terms", []) or [])
    signal_terms = " ".join(
        " ".join(
            [str(match.get("signal") or "")]
            + [str(term) for term in match.get("matched_terms", []) or []]
        )
        for match in story.get("frontier_signal_matches", []) or []
        if isinstance(match, dict)
    )
    primitive_terms = " ".join(
        " ".join(
            [str(primitive.get("primitive") or "")]
            + [str(term) for term in primitive.get("matched_terms", []) or []]
            + [str(term) for term in primitive.get("introduced_terms", []) or []]
        )
        for primitive in story.get("optimization_primitives", []) or []
        if isinstance(primitive, dict)
    )
    haystack = " ".join(
        [
            candidate.target,
            candidate.category,
            candidate.rationale,
            candidate.best_optimization,
            candidate.baseline_file,
            source_terms,
            source_delta_terms,
            signal_terms,
            primitive_terms,
        ]
    ).lower()
    return any(term in haystack for term in terms)


def _row_contains_any(row: dict[str, Any], terms: Iterable[str]) -> bool:
    haystack = " ".join(
        [
            str(row.get("target") or ""),
            str(row.get("opportunity_type") or ""),
            str(row.get("optimization_goal") or ""),
            str(row.get("rationale") or ""),
            " ".join(str(item) for item in row.get("source_terms", []) or []),
            " ".join(str(item) for item in row.get("source_delta_terms", []) or []),
            " ".join(str(item) for item in row.get("frontier_signals", []) or []),
            " ".join(
                " ".join(
                    [str(primitive.get("primitive") or "")]
                    + [str(term) for term in primitive.get("matched_terms", []) or []]
                    + [str(term) for term in primitive.get("introduced_terms", []) or []]
                )
                for primitive in row.get("optimization_primitives", []) or []
                if isinstance(primitive, dict)
            ),
            " ".join(str(item) for item in row.get("evidence", []) or []),
        ]
    ).lower()
    return any(term in haystack for term in terms)


def _slug(value: str, *, max_len: int = 63) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:max_len].strip("-") or "target"


def _term_variants(term: str) -> set[str]:
    normalized = term.strip().lower()
    return {normalized, normalized.replace(" ", "_"), normalized.replace("_", " ")}


def _text_contains_term(text: str, term: str) -> bool:
    return any(variant and variant in text for variant in _term_variants(term))


def _read_catalog_source(paths: Iterable[Path], *, max_chars_per_file: int = 12000) -> str:
    chunks: list[str] = []
    for path in paths:
        chunks.append(path.name)
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace")[:max_chars_per_file])
        except OSError:
            continue
    return "\n".join(chunks).lower()


def _read_catalog_file(path: Path, *, max_chars: int = 12000) -> str:
    try:
        return (
            f"{path.name}\n{path.read_text(encoding='utf-8', errors='replace')[:max_chars]}".lower()
        )
    except OSError:
        return path.name.lower()


def _catalog_signal_matches(text: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for spec in FRONTIER_SIGNAL_SPECS:
        terms = [str(term) for term in spec["terms"] if _text_contains_term(text, str(term))]
        if terms:
            matches.append(
                {
                    "signal": spec["signal"],
                    "matched_terms": sorted(set(terms)),
                    "bonus": spec["bonus"],
                    "reason": spec["reason"],
                }
            )
    matches.sort(key=lambda row: (-float(row["bonus"]), str(row["signal"])))
    return matches


def _catalog_source_terms(text: str) -> list[str]:
    terms: set[str] = set()
    for spec in [*MOTIF_SPECS, *FRONTIER_SIGNAL_SPECS, *OPTIMIZATION_PRIMITIVE_SPECS]:
        for term in spec["terms"]:
            term_text = str(term)
            if _text_contains_term(text, term_text):
                terms.add(term_text)
    return sorted(terms)


def _catalog_delta_terms(
    baseline_text: str, optimized_text: str, source_terms: Iterable[str]
) -> list[str]:
    return sorted(
        {
            str(term)
            for term in source_terms
            if _text_contains_term(optimized_text, str(term))
            and not _text_contains_term(baseline_text, str(term))
        }
    )


def _catalog_optimization_primitives(
    baseline_text: str, optimized_text: str
) -> list[dict[str, Any]]:
    primitives: list[dict[str, Any]] = []
    for spec in OPTIMIZATION_PRIMITIVE_SPECS:
        matched_terms = [
            str(term) for term in spec["terms"] if _text_contains_term(optimized_text, str(term))
        ]
        if not matched_terms:
            continue
        introduced_terms = [
            term for term in matched_terms if not _text_contains_term(baseline_text, term)
        ]
        primitives.append(
            {
                "primitive": spec["primitive"],
                "matched_terms": sorted(set(matched_terms)),
                "introduced_terms": sorted(set(introduced_terms)),
                "introduced": bool(introduced_terms),
                "why": spec["why"],
                "transfer_question": spec["transfer_question"],
            }
        )
    primitives.sort(
        key=lambda row: (
            not bool(row.get("introduced")),
            str(row.get("primitive")),
        )
    )
    return primitives


def _catalog_motif(text: str) -> str:
    best: tuple[int, str] = (0, "single_target")
    for spec in MOTIF_SPECS:
        score = sum(1 for term in spec["terms"] if _text_contains_term(text, str(term)))
        if score > best[0]:
            best = (score, str(spec["motif"]))
    return best[1]


def _catalog_goal(text: str) -> str:
    if any(
        _text_contains_term(text, term)
        for term in ("memory", "hbm", "bandwidth", "copy", "storage", "cache")
    ):
        return "memory"
    return "speed"


def discover_benchmark_target_catalog(
    bench_root: Path | None = None,
) -> dict[str, Any]:
    """Mine runnable benchmark pairs into a frontier target catalog."""

    from core.discovery import chapter_slug, discover_all_chapters, discover_benchmarks

    root = (
        Path(bench_root).expanduser().resolve()
        if bench_root
        else Path(__file__).resolve().parents[2]
    )
    entries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_targets: set[str] = set()
    motif_counts: Counter[str] = Counter()
    signal_counts: Counter[str] = Counter()

    for chapter_dir in discover_all_chapters(root, bench_roots=[root]):
        chapter_id = chapter_slug(chapter_dir, root, bench_root=root)
        try:
            pairs = discover_benchmarks(chapter_dir, warn_missing=False)
        except Exception as exc:
            errors.append({"chapter": chapter_id, "error": str(exc)})
            continue
        for baseline_file, optimized_files, example in pairs:
            target = f"{chapter_id}:{example}"
            if target in seen_targets:
                continue
            seen_targets.add(target)
            source_paths = [baseline_file, *optimized_files]
            baseline_text = _read_catalog_file(baseline_file)
            optimized_text = "\n".join(_read_catalog_file(path) for path in optimized_files)
            source_text = _read_catalog_source(source_paths)
            signals = _catalog_signal_matches(source_text)
            source_terms = _catalog_source_terms(source_text)
            source_delta_terms = _catalog_delta_terms(baseline_text, optimized_text, source_terms)
            optimization_primitives = _catalog_optimization_primitives(
                baseline_text, optimized_text
            )
            motif = _catalog_motif(source_text)
            motif_counts[motif] += 1
            signal_counts.update(str(match["signal"]) for match in signals)
            signal_names = [str(match["signal"]) for match in signals[:4]]
            term_preview = source_terms[:8]
            rationale = (
                "Source-mined paired benchmark target"
                + (f" with frontier signals {', '.join(signal_names)}" if signal_names else "")
                + (f"; matched terms {', '.join(term_preview)}" if term_preview else "")
                + "."
            )
            entries.append(
                {
                    "target": target,
                    "chapter": chapter_id,
                    "example": example,
                    "category": motif,
                    "optimization_goal": _catalog_goal(source_text),
                    "rationale": rationale,
                    "baseline_file": str(baseline_file.relative_to(root)),
                    "optimized_files": [str(path.relative_to(root)) for path in optimized_files],
                    "source_terms": source_terms,
                    "source_delta_terms": source_delta_terms,
                    "frontier_signal_matches": signals,
                    "optimization_primitives": optimization_primitives,
                    "best_optimization": "source_mined_frontier_candidate",
                    "catalog_source": "benchmark_tree",
                }
            )

    entries.sort(
        key=lambda row: (
            -sum(float(match.get("bonus") or 0.0) for match in row["frontier_signal_matches"]),
            str(row["target"]),
        )
    )
    return {
        "source": str(root),
        "target_count": len(entries),
        "motif_counts": dict(motif_counts),
        "signal_counts": dict(signal_counts),
        "errors": errors,
        "targets": entries,
    }


def _frontier_motif(candidate: BenchmarkCandidate) -> str | None:
    if candidate.status != "frontier":
        return None
    for spec in MOTIF_SPECS:
        if _contains_any(candidate, spec["terms"]):
            return str(spec["motif"])
    return None


def _frontier_score_breakdown(candidate: BenchmarkCandidate) -> list[dict[str, Any]]:
    if candidate.status != "frontier":
        return []

    breakdown: list[dict[str, Any]] = []
    for spec in FRONTIER_SIGNAL_SPECS:
        if _contains_any(candidate, spec["terms"]):
            breakdown.append(
                {
                    "signal": spec["signal"],
                    "bonus": spec["bonus"],
                    "reason": spec["reason"],
                }
            )
    if candidate.category:
        breakdown.append(
            {
                "signal": "catalog_category",
                "bonus": 2.0,
                "reason": "Catalog metadata narrows the expected workload family.",
            }
        )
    if candidate.rationale and "no evidence" not in candidate.rationale.lower():
        breakdown.append(
            {
                "signal": "catalog_rationale",
                "bonus": 2.0,
                "reason": "Catalog rationale explains why this unmeasured target deserves first evidence.",
            }
        )
    return breakdown


def _frontier_signals(candidate: BenchmarkCandidate) -> list[str]:
    return [str(item["signal"]) for item in _frontier_score_breakdown(candidate)]


def _frontier_score_bonus(candidate: BenchmarkCandidate) -> float:
    return sum(float(item["bonus"]) for item in _frontier_score_breakdown(candidate))


def _candidate_source_terms(candidate: BenchmarkCandidate) -> list[str]:
    story = candidate.story_metadata or {}
    terms = [str(item) for item in story.get("source_terms", []) or [] if item]
    if not terms and candidate.status == "frontier":
        terms = [
            str(term)
            for match in story.get("frontier_signal_matches", []) or []
            if isinstance(match, dict)
            for term in match.get("matched_terms", []) or []
            if term
        ]
    return sorted(set(terms))


def _candidate_source_delta_terms(candidate: BenchmarkCandidate) -> list[str]:
    story = candidate.story_metadata or {}
    return sorted({str(item) for item in story.get("source_delta_terms", []) or [] if item})


def _candidate_source_files(candidate: BenchmarkCandidate) -> list[str]:
    story = candidate.story_metadata or {}
    files = []
    if candidate.baseline_file:
        files.append(candidate.baseline_file)
    files.extend(str(path) for path in story.get("optimized_files", []) or [] if path)
    return sorted(set(files))


def _candidate_catalog_source(candidate: BenchmarkCandidate) -> str:
    story = candidate.story_metadata or {}
    return str(story.get("catalog_source") or "")


def _candidate_optimization_primitives(candidate: BenchmarkCandidate) -> list[dict[str, Any]]:
    story = candidate.story_metadata or {}
    primitives = story.get("optimization_primitives", []) or []
    return [dict(item) for item in primitives if isinstance(item, dict)]


def _recommended_experiments(candidate: BenchmarkCandidate, opportunity_type: str) -> list[str]:
    experiments: list[str] = []
    if opportunity_type == "novel_frontier_probe":
        experiments.append(
            "Run the target with `--profile minimal --verify-output` to establish first clean evidence before optimizing it."
        )
        experiments.append(
            "If the first run is clean, rerun with `--profile deep_dive` and compare its dominant bottleneck against nearby motif clusters."
        )
    if opportunity_type in {"restore_benchmark_evidence", "repair_regression"}:
        experiments.append(
            "Run the target with `--profile minimal --verify-output` to restore a clean baseline/optimized comparison."
        )
        experiments.append(
            "If it still fails, run `bench verify` on the baseline and optimized wrappers before changing expectations."
        )

    if _contains_any(candidate, ["attention", "prefill", "decode", "kv"]):
        experiments.append(
            "Profile attention and KV-cache kernels with Nsight Compute, then test paged/prefix-aware cache layout or Flash/FlexAttention variants."
        )
    if _contains_any(candidate, ["moe", "expert", "router"]):
        experiments.append(
            "Try topology-aware routing, grouped GEMM batching, and capacity-factor sweeps before changing model-level routing policy."
        )
    if _contains_any(candidate, ["nccl", "allreduce", "gradient", "comm", "nvlink", "nvshmem"]):
        experiments.append(
            "Measure communication overlap and test bucket fusion, reduce-scatter, compression, or NVSHMEM/NCCL protocol variants."
        )
    if _contains_any(candidate, ["copy", "memory", "hbm", "prefetch", "tma", "lookup", "cache"]):
        experiments.append(
            "Check memory-transaction efficiency, then test vectorized loads, pinned/nonblocking copies, TMA prefetch, or layout pretransposition."
        )
    if _contains_any(candidate, ["gemm", "matmul", "fp4", "fp8", "nvfp4", "tcgen", "cutlass"]):
        experiments.append(
            "Sweep cublasLt/CUTLASS/Triton tile shapes and precision modes, including block-scaled FP8/FP4 where the hardware supports it."
        )
    if _contains_any(candidate, ["compile", "graph", "launch", "persistent"]):
        experiments.append(
            'Compare eager, `torch.compile(mode="reduce-overhead")`, CUDA Graph replay, and persistent-kernel launch paths.'
        )

    if not experiments:
        experiments.append(
            "Run a deep-dive profile, identify the dominant stall class, and create one narrow baseline/optimized pair for that bottleneck."
        )

    deduped: list[str] = []
    for item in experiments:
        if item not in deduped:
            deduped.append(item)
    return deduped[:4]


def _classify(
    candidate: BenchmarkCandidate,
    *,
    min_speedup: float,
    target_speedup: float,
    min_memory_savings_pct: float,
) -> str:
    status = candidate.status
    speedup = candidate.best_speedup
    memory_savings = candidate.memory_savings_pct

    if status == "frontier":
        return "novel_frontier_probe"
    if status in {"failed", "missing", "skipped"}:
        return "restore_benchmark_evidence"
    if speedup is not None and speedup < 0.95:
        return "repair_regression"
    if candidate.optimization_goal == "memory" and (
        memory_savings is None or memory_savings < min_memory_savings_pct
    ):
        return "memory_pressure_probe"
    if speedup is None:
        return "restore_benchmark_evidence"
    if speedup < min_speedup:
        return "rework_flat_optimization"
    if speedup < target_speedup:
        return "compound_stack_candidate"
    return "profile_remaining_headroom"


def _score(
    candidate: BenchmarkCandidate,
    opportunity_type: str,
    *,
    min_speedup: float,
    target_speedup: float,
    slow_baseline_ms: float,
) -> float:
    speedup = candidate.best_speedup or 0.0
    baseline = candidate.baseline_time_ms or 0.0
    story = candidate.story_metadata or {}

    base_by_type = {
        "restore_benchmark_evidence": 88.0,
        "repair_regression": 92.0,
        "novel_frontier_probe": 70.0,
        "memory_pressure_probe": 76.0,
        "rework_flat_optimization": 82.0,
        "compound_stack_candidate": 64.0,
        "profile_remaining_headroom": 34.0,
    }
    score = base_by_type.get(opportunity_type, 25.0)

    if speedup > 0.0:
        if speedup < min_speedup:
            score += min(16.0, (min_speedup - speedup) * 60.0)
        elif speedup < target_speedup:
            score += min(14.0, (target_speedup - speedup) * 25.0)
        else:
            score += max(0.0, 12.0 - min(speedup, 12.0))

    if baseline > 0.0:
        score += min(18.0, math.log10(baseline + 1.0) * 6.0)
        if baseline >= slow_baseline_ms:
            score += 8.0

    if candidate.artifact_count == 0:
        score += 4.0
    if opportunity_type == "novel_frontier_probe":
        score += _frontier_score_bonus(candidate)
    if (
        story.get("compound_optimization") is False
        and opportunity_type == "compound_stack_candidate"
    ):
        score += 6.0

    return round(min(score, 100.0), 2)


def _priority(score: float) -> str:
    if score >= 80.0:
        return "high"
    if score >= 55.0:
        return "medium"
    return "low"


def _evidence(candidate: BenchmarkCandidate) -> list[str]:
    evidence = [f"status={candidate.status}"]
    if candidate.best_speedup is not None:
        evidence.append(f"best_speedup={candidate.best_speedup:.2f}x")
    if candidate.baseline_time_ms is not None:
        evidence.append(f"baseline={candidate.baseline_time_ms:.3f} ms")
    if candidate.optimized_time_ms is not None:
        evidence.append(f"optimized={candidate.optimized_time_ms:.3f} ms")
    if candidate.memory_savings_pct is not None:
        evidence.append(f"memory_savings={candidate.memory_savings_pct:.1f}%")
    if candidate.status == "frontier":
        motif = _frontier_motif(candidate)
        signals = _frontier_signals(candidate)
        if motif:
            evidence.append(f"frontier_motif={motif}")
        if signals:
            evidence.append(f"frontier_signals={','.join(signals)}")
    evidence.append(f"artifacts={candidate.artifact_count}")
    return evidence


def _rationale(candidate: BenchmarkCandidate, opportunity_type: str) -> str:
    if opportunity_type == "novel_frontier_probe":
        signals = _frontier_signals(candidate)
        if signals:
            return (
                "The target is runnable but absent from current evidence; frontier signals "
                f"({', '.join(signals)}) make it a high-leverage first-evidence probe."
            )
        return "The target is runnable according to the supplied catalog but absent from current evidence; run a minimal verified pass before treating it as an optimization lead."
    if opportunity_type == "restore_benchmark_evidence":
        return "The target does not currently provide clean succeeded evidence, so optimization claims and comparisons are blocked."
    if opportunity_type == "repair_regression":
        return "The optimized path is slower than baseline; this should be fixed before new speedup claims are made."
    if opportunity_type == "memory_pressure_probe":
        return "The target is memory-focused but has little recorded memory relief, so cache layout or allocation pressure likely deserves attention."
    if opportunity_type == "rework_flat_optimization":
        return "The optimized path is effectively flat; a deep profile should identify whether launch, memory, compute, or communication dominates."
    if opportunity_type == "compound_stack_candidate":
        return "The target has a real but modest win, making it a good candidate for stacking another compatible optimization layer."
    return "The target already has a win, but its runtime and theme suggest remaining headroom worth profiling after higher-priority gaps."


def _profile_command(target: str, profile: str) -> str:
    return f"python -m cli.aisp bench run --targets {target} --profile {profile} --verify-output"


def _profiler_recipe(signal: str, tools: Iterable[str]) -> dict[str, Any]:
    tool_list = [str(tool) for tool in tools]
    preflight_checks: list[str] = ["target has a clean minimal-profile control run"]
    if any(tool in {"nsys", "ncu", "hta"} for tool in tool_list):
        preflight_checks.append("GPU is visible via nvidia-smi before deep profiling")
    if "nsys" in tool_list or "hta" in tool_list:
        preflight_checks.append("Nsight Systems is available for timeline and launch analysis")
    if "ncu" in tool_list:
        preflight_checks.append("Nsight Compute is available for kernel-counter analysis")
    if "hta" in tool_list:
        preflight_checks.append(
            "multi-GPU trace includes communication events when the target is distributed"
        )
    if "pytorch" in tool_list:
        preflight_checks.append(
            "PyTorch profiler can write a trace directory under the isolated artifact root"
        )
    if "zymtrace" in tool_list:
        preflight_checks.append(
            "Zymtrace CUDA injection library resolves via CUDA_INJECTION64_PATH, ZYMTRACE_CUDA_INJECTION64_PATH, or the standard profiler install"
        )

    artifact_expectations = [
        "profile summary JSON",
        "raw profiler artifact references",
        "dominant bottleneck note",
        "guardrail metric comparison",
    ]
    if "zymtrace" in tool_list:
        artifact_expectations.append("zymtrace_launch_manifest.json with CUDA injection path")

    return {
        "profile_mode": "deep_dive",
        "tools": tool_list,
        "preflight_checks": preflight_checks,
        "artifact_expectations": artifact_expectations,
        "trace_focus": signal,
    }


def _frontier_blueprints_for_row(
    row: dict[str, Any], *, limit: int = 4, preferred_signal: str | None = None
) -> list[dict[str, Any]]:
    target = str(row.get("target") or "")
    if not target:
        return []

    blueprints: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    signals = [str(signal) for signal in row.get("frontier_signals", [])]
    if preferred_signal and preferred_signal in signals:
        signals = [preferred_signal, *(signal for signal in signals if signal != preferred_signal)]
    for signal in signals:
        for spec in FRONTIER_BLUEPRINT_SPECS.get(signal, []):
            name = str(spec.get("name") or signal)
            if name in seen_names:
                continue
            seen_names.add(name)
            tools = [str(tool) for tool in spec.get("profiler_tools", [])]
            blueprints.append(
                {
                    "id": _slug(f"{target}-{name}", max_len=72),
                    "signal": signal,
                    "target": target,
                    "name": name,
                    "hypothesis": spec.get("hypothesis"),
                    "experiment_knobs": list(spec.get("knobs", [])),
                    "primary_metrics": list(spec.get("primary_metrics", [])),
                    "guardrails": list(spec.get("guardrails", [])),
                    "smoke_command": _profile_command(target, "minimal"),
                    "profile_command": _profile_command(target, "deep_dive"),
                    "profiler_recipe": _profiler_recipe(signal, tools),
                    "promotion_gate": "Promote only after the smoke command succeeds, profiler artifacts identify the bottleneck, and the candidate beats the verified control.",
                }
            )
            if len(blueprints) >= limit:
                return blueprints

    if not blueprints:
        blueprints.append(
            {
                "id": _slug(f"{target}-generic-frontier-profile", max_len=72),
                "signal": "general_frontier",
                "target": target,
                "name": "generic_frontier_profile",
                "hypothesis": "Use first evidence and a deep-dive profile to classify the dominant bottleneck before proposing code changes.",
                "experiment_knobs": ["profile_mode", "input_shape", "artifact_isolation"],
                "primary_metrics": ["verified_status", "median_wall_time_ms"],
                "guardrails": ["output_correctness", "artifact_paths_present"],
                "smoke_command": _profile_command(target, "minimal"),
                "profile_command": _profile_command(target, "deep_dive"),
                "profiler_recipe": _profiler_recipe("general_frontier", ["nsys", "ncu"]),
                "promotion_gate": "Promote only after first evidence exists and the profile identifies one concrete bottleneck.",
            }
        )
    return blueprints[:limit]


def _next_command(target: str, opportunity_type: str) -> str:
    profile = (
        "minimal"
        if opportunity_type
        in {"restore_benchmark_evidence", "repair_regression", "novel_frontier_probe"}
        else "deep_dive"
    )
    return _profile_command(target, profile)


def _infer_workload_type(candidate: BenchmarkCandidate) -> str:
    if _contains_any(
        candidate,
        ["decode", "prefill", "kv", "inference", "serving", "vllm", "sglang", "trt"],
    ):
        return "inference"
    if _contains_any(
        candidate,
        ["train", "training", "gradient", "ddp", "fsdp", "optimizer", "backward"],
    ):
        return "training"
    return "mixed"


def _infer_comparison_variable(candidate: BenchmarkCandidate) -> str:
    if _contains_any(candidate, ["nccl", "allreduce", "nvlink", "nvshmem", "ib", "rdma"]):
        return "network_topology"
    if _contains_any(candidate, ["compile", "graph", "cuda graph", "launch", "persistent"]):
        return "runtime_version"
    if _contains_any(candidate, ["dataloader", "storage", "gds", "fio", "input"]):
        return "storage_stack"
    if _contains_any(candidate, ["scheduler", "kueue", "slinky", "control"]):
        return "scheduler_path"
    return "runtime_version"


def _infer_precision(candidate: BenchmarkCandidate) -> str:
    if _contains_any(candidate, ["nvfp4", "fp4"]):
        return "nvfp4"
    if _contains_any(candidate, ["fp8", "mxfp8"]):
        return "fp8"
    if _contains_any(candidate, ["fp16"]):
        return "fp16"
    return "bf16"


def _benchmark_run_payload(candidate: BenchmarkCandidate, opportunity_type: str) -> dict[str, Any]:
    workload_type = _infer_workload_type(candidate)
    benchmark_class = (
        "realism_grade"
        if opportunity_type
        in {"restore_benchmark_evidence", "repair_regression", "novel_frontier_probe"}
        else "publication_grade"
    )
    cadence = (
        "canary"
        if opportunity_type
        in {"restore_benchmark_evidence", "repair_regression", "novel_frontier_probe"}
        else "nightly"
    )
    overrides = {
        "name": f"opportunity-{_slug(candidate.target, max_len=51)}",
        "benchmarkClass": benchmark_class,
        "workloadType": workload_type,
        "schedulerPath": "slinky-kueue",
        "cadence": cadence,
        "model": "openai/gpt-oss-20b" if workload_type != "training" else "training-hotpath-model",
        "precision": _infer_precision(candidate),
        "batchingPolicy": "continuous" if workload_type == "inference" else "static",
        "concurrencyModel": "closed_loop",
        "comparisonVariable": _infer_comparison_variable(candidate),
    }
    overrides_json = json.dumps(overrides, separators=(",", ":"))
    return {
        "overrides": overrides,
        "render_command": (
            f"python -m cli.aisp tools benchmark-run-render -- --overrides-json '{overrides_json}'"
        ),
        "mcp_tool": "render_benchmark_run",
        "api_route": "/api/benchmark/contracts/render-run",
    }


def _build_execution_plan(opportunities: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a deterministic phased runbook from ranked opportunities."""
    phase_specs = [
        {
            "name": "restore_evidence",
            "title": "Restore trustworthy evidence first",
            "types": {"restore_benchmark_evidence", "repair_regression"},
            "profile": "minimal",
            "stop_condition": "Every selected target has a succeeded minimal-profile run with output verification enabled.",
        },
        {
            "name": "explore_frontier",
            "title": "Establish first evidence for frontier targets",
            "types": {"novel_frontier_probe"},
            "profile": "minimal",
            "stop_condition": "Every frontier target has one verified minimal-profile run before deep profiling or code changes.",
        },
        {
            "name": "deep_profile_headroom",
            "title": "Deep-profile the highest-headroom targets",
            "types": {
                "memory_pressure_probe",
                "rework_flat_optimization",
                "compound_stack_candidate",
                "profile_remaining_headroom",
            },
            "profile": "deep_dive",
            "stop_condition": "Each selected target has profiler artifacts that identify the dominant stall class before code changes begin.",
        },
    ]

    phases: list[dict[str, Any]] = []
    for spec in phase_specs:
        items = [
            {
                "rank": row.get("rank"),
                "target": row.get("target"),
                "priority": row.get("priority"),
                "opportunity_type": row.get("opportunity_type"),
                "score": row.get("score"),
                "command": _profile_command(str(row.get("target")), str(spec["profile"])),
                "benchmark_run": row.get("benchmark_run"),
                "stop_condition": spec["stop_condition"],
            }
            for row in opportunities
            if row.get("opportunity_type") in spec["types"]
        ]
        if items:
            phases.append(
                {
                    "name": spec["name"],
                    "title": spec["title"],
                    "profile": spec["profile"],
                    "parallelism_policy": "Run serially by default; parallelize only when each command has isolated GPU visibility and artifact directories.",
                    "items": items,
                }
            )

    compound_items = [
        {
            "rank": row.get("rank"),
            "target": row.get("target"),
            "priority": row.get("priority"),
            "experiments": row.get("recommended_experiments", []),
            "validation_command": _profile_command(str(row.get("target")), "minimal"),
            "benchmark_run": row.get("benchmark_run"),
        }
        for row in opportunities
        if row.get("opportunity_type")
        in {"memory_pressure_probe", "rework_flat_optimization", "compound_stack_candidate"}
    ]
    if compound_items:
        phases.append(
            {
                "name": "implement_and_validate",
                "title": "Implement one narrow experiment per target, then re-measure",
                "profile": "minimal",
                "parallelism_policy": "Keep one variable under test per run so expectation refreshes remain auditable.",
                "items": compound_items,
            }
        )

    next_commands = [
        item["command"]
        for phase in phases[:2]
        for item in phase.get("items", [])
        if item.get("command")
    ]
    return {
        "phase_count": len(phases),
        "phases": phases,
        "next_commands": next_commands[:10],
        "completion_gate": "Re-run benchmark_opportunities on the new summary and require no high-priority restore_evidence or repair_regression items before publishing speedup claims.",
    }


def _build_innovation_hypotheses(opportunities: list[dict[str, Any]]) -> dict[str, Any]:
    """Cluster ranked opportunities into transfer experiments.

    A single benchmark result can be noisy or local. Clusters identify motifs where
    one prototype experiment may transfer to nearby targets.
    """
    clusters: list[dict[str, Any]] = []
    for spec in MOTIF_SPECS:
        support = [
            row
            for row in opportunities
            if _row_contains_any(row, spec["terms"])
            and row.get("opportunity_type") != "restore_benchmark_evidence"
        ]
        if not support:
            continue
        support.sort(key=lambda row: (-float(row.get("score") or 0.0), str(row.get("target"))))
        prototype = support[0]
        priority = max(
            (str(row.get("priority") or "low") for row in support),
            key=lambda value: PRIORITY_RANK.get(value, 0),
        )
        clusters.append(
            {
                "motif": spec["motif"],
                "title": spec["title"],
                "priority": priority,
                "support_count": len(support),
                "prototype_target": prototype.get("target"),
                "support_targets": [row.get("target") for row in support[:5]],
                "reason": spec["reason"],
                "prototype_command": prototype.get("next_command"),
                "transfer_experiments": spec["transfer_experiments"],
                "validation_commands": [
                    _profile_command(str(row.get("target")), "minimal") for row in support[:3]
                ],
                "acceptance_gate": spec["acceptance_gate"],
            }
        )

    clusters.sort(
        key=lambda row: (
            -PRIORITY_RANK.get(str(row.get("priority")), 0),
            -int(row.get("support_count") or 0),
            str(row.get("motif")),
        )
    )
    return {
        "cluster_count": len(clusters),
        "clusters": clusters,
        "portfolio_guidance": (
            "Start with the highest-priority prototype, then run its validation commands "
            "across support targets before promoting the idea as a reusable optimization."
        ),
    }


def _build_source_transfer_map(opportunities: list[dict[str, Any]]) -> dict[str, Any]:
    """Map source-mined optimization patterns to likely recipient targets."""

    synthetic_signals = {"catalog_category", "catalog_rationale"}
    rows_by_signal: dict[str, list[dict[str, Any]]] = {}
    rows_by_primitive: dict[str, list[dict[str, Any]]] = {}
    for row in opportunities:
        signals = [
            str(signal)
            for signal in row.get("frontier_signals", []) or []
            if str(signal) not in synthetic_signals
        ]
        for signal in signals:
            rows_by_signal.setdefault(signal, []).append(row)
        for primitive in row.get("optimization_primitives", []) or []:
            if not isinstance(primitive, dict):
                continue
            primitive_name = str(primitive.get("primitive") or "")
            if primitive_name:
                rows_by_primitive.setdefault(primitive_name, []).append(row)

    patterns: list[dict[str, Any]] = []
    for signal, rows in rows_by_signal.items():
        rows.sort(key=lambda row: (-float(row.get("score") or 0.0), str(row.get("target"))))
        source = rows[0]
        recipients = [row for row in rows[1:] if row.get("target") != source.get("target")]
        terms = sorted(
            {
                str(term)
                for row in rows[:6]
                for term in row.get("source_terms", []) or []
                if str(term)
            }
        )
        blueprints = _frontier_blueprints_for_row(
            {"target": source.get("target"), "frontier_signals": [signal]},
            preferred_signal=signal,
            limit=2,
        )
        patterns.append(
            {
                "pattern": signal,
                "pattern_id": _slug(f"source-transfer-{signal}"),
                "motif": source.get("frontier_motif") or "single_target",
                "source_target": source.get("target"),
                "source_terms": terms[:12],
                "source_files": source.get("source_files", []),
                "recipient_targets": [row.get("target") for row in recipients[:5]],
                "recipient_count": len(recipients),
                "blueprint_ids": [blueprint.get("id") for blueprint in blueprints],
                "transfer_blueprints": blueprints,
                "prototype_command": _profile_command(str(source.get("target")), "minimal"),
                "recipient_validation_commands": [
                    _profile_command(str(row.get("target")), "minimal") for row in recipients[:3]
                ],
                "adoption_gate": (
                    "Transfer only after source and recipient targets both have clean minimal-profile evidence, "
                    "then run one source-pattern variant with unchanged guardrail metrics."
                ),
            }
        )

    for primitive_name, rows in rows_by_primitive.items():
        rows.sort(
            key=lambda row: (
                -sum(
                    1
                    for primitive in row.get("optimization_primitives", []) or []
                    if isinstance(primitive, dict)
                    and primitive.get("primitive") == primitive_name
                    and primitive.get("introduced")
                ),
                -float(row.get("score") or 0.0),
                str(row.get("target")),
            )
        )
        source = rows[0]
        recipients = [row for row in rows[1:] if row.get("target") != source.get("target")]
        source_primitive = next(
            (
                primitive
                for primitive in source.get("optimization_primitives", []) or []
                if isinstance(primitive, dict) and primitive.get("primitive") == primitive_name
            ),
            {},
        )
        preferred_signal = next(
            (
                str(signal)
                for signal in source.get("frontier_signals", []) or []
                if str(signal) not in synthetic_signals
            ),
            primitive_name,
        )
        blueprints = _frontier_blueprints_for_row(
            {"target": source.get("target"), "frontier_signals": [preferred_signal]},
            preferred_signal=preferred_signal,
            limit=2,
        )
        introduced_terms = sorted(
            {
                str(term)
                for row in rows[:6]
                for primitive in row.get("optimization_primitives", []) or []
                if isinstance(primitive, dict) and primitive.get("primitive") == primitive_name
                for term in primitive.get("introduced_terms", []) or []
                if term
            }
        )
        patterns.append(
            {
                "pattern": f"primitive:{primitive_name}",
                "pattern_id": _slug(f"source-transfer-primitive-{primitive_name}"),
                "pattern_type": "optimization_primitive",
                "primitive": source_primitive,
                "motif": source.get("frontier_motif") or "single_target",
                "source_target": source.get("target"),
                "source_terms": introduced_terms or source.get("source_delta_terms", []),
                "source_files": source.get("source_files", []),
                "recipient_targets": [row.get("target") for row in recipients[:5]],
                "recipient_count": len(recipients),
                "blueprint_ids": [blueprint.get("id") for blueprint in blueprints],
                "transfer_blueprints": blueprints,
                "prototype_command": _profile_command(str(source.get("target")), "minimal"),
                "recipient_validation_commands": [
                    _profile_command(str(row.get("target")), "minimal") for row in recipients[:3]
                ],
                "adoption_gate": source_primitive.get("transfer_question")
                or "Validate the primitive on source and recipient with clean guardrail metrics before promotion.",
            }
        )

    patterns.sort(
        key=lambda row: (
            -int(row.get("recipient_count") or 0),
            str(row.get("pattern_type") or "frontier_signal"),
            str(row.get("pattern")),
            str(row.get("source_target")),
        )
    )
    return {
        "pattern_count": len(patterns),
        "primitive_pattern_count": sum(
            1 for pattern in patterns if pattern.get("pattern_type") == "optimization_primitive"
        ),
        "source_target_count": len(
            {str(row.get("source_target")) for row in patterns if row.get("source_target")}
        ),
        "recipient_target_count": len(
            {
                str(target)
                for row in patterns
                for target in row.get("recipient_targets", []) or []
                if target
            }
        ),
        "policy": (
            "Use source-mined patterns to choose transfer experiments only after first evidence exists; "
            "a shared signal is a hypothesis, not a claim."
        ),
        "patterns": patterns,
    }


def _primitive_names(row: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for primitive in row.get("optimization_primitives", []) or []:
        if isinstance(primitive, dict):
            name = str(primitive.get("primitive") or "")
        else:
            name = str(primitive or "")
        if name:
            names.add(name)
    return names


def _compound_validation_commands(target: str, support_rows: list[dict[str, Any]]) -> list[str]:
    commands = [_profile_command(target, "minimal")]
    commands.extend(
        _profile_command(str(row.get("target")), "minimal")
        for row in support_rows[:3]
        if row.get("target")
    )
    deduped: list[str] = []
    for command in commands:
        if command not in deduped:
            deduped.append(command)
    return deduped


def _build_compound_primitive_hypotheses(
    opportunities: list[dict[str, Any]], *, max_per_compound: int = 6
) -> dict[str, Any]:
    """Propose unmeasured primitive combinations from source-mined evidence."""

    rows = [row for row in opportunities if row.get("target")]
    primitives_by_target = {
        str(row.get("target")): _primitive_names(row) for row in rows if row.get("target")
    }
    hypotheses: list[dict[str, Any]] = []

    for spec in COMPOUND_PRIMITIVE_SPECS:
        required = {str(item) for item in spec.get("primitives", []) if item}
        if len(required) < 2:
            continue

        candidate_rows = []
        for row in rows:
            target = str(row.get("target") or "")
            present = primitives_by_target.get(target, set()) & required
            missing = required - present
            if not present or not missing:
                continue
            support_has_missing = any(
                other_target != target and bool(other_primitives & missing)
                for other_target, other_primitives in primitives_by_target.items()
            )
            if support_has_missing:
                candidate_rows.append(row)

        candidate_rows.sort(
            key=lambda row: (
                -len(primitives_by_target.get(str(row.get("target") or ""), set()) & required),
                len(required - primitives_by_target.get(str(row.get("target") or ""), set())),
                -float(row.get("score") or 0.0),
                str(row.get("target") or ""),
            )
        )

        for row in candidate_rows[:max_per_compound]:
            target = str(row.get("target") or "")
            target_primitives = primitives_by_target.get(target, set())
            present = sorted(target_primitives & required)
            missing = sorted(required - target_primitives)
            missing_set = set(missing)
            support_rows = [
                support
                for support in rows
                if str(support.get("target") or "") != target
                and bool(
                    primitives_by_target.get(str(support.get("target") or ""), set()) & missing_set
                )
            ]
            support_rows.sort(
                key=lambda support: (
                    -len(
                        primitives_by_target.get(str(support.get("target") or ""), set())
                        & missing_set
                    ),
                    -len(
                        primitives_by_target.get(str(support.get("target") or ""), set()) & required
                    ),
                    -float(support.get("score") or 0.0),
                    str(support.get("target") or ""),
                )
            )
            if not support_rows:
                continue

            support_targets = [str(support.get("target")) for support in support_rows[:5]]
            support_primitives = sorted(
                {
                    primitive
                    for support in support_rows[:5]
                    for primitive in primitives_by_target.get(
                        str(support.get("target") or ""), set()
                    )
                    & required
                }
            )
            compound_score = (
                float(row.get("score") or 0.0)
                + len(present) * 10.0
                + min(len(support_rows), 5) * 2.0
                - len(missing) * 1.5
            )
            hypotheses.append(
                {
                    "compound_id": _slug(f"compound-{spec.get('compound')}"),
                    "compound": spec.get("compound"),
                    "name": spec.get("name"),
                    "target": target,
                    "priority": row.get("priority"),
                    "compound_score": round(compound_score, 3),
                    "primitives": sorted(required),
                    "present_primitives": present,
                    "missing_primitives": missing,
                    "support_targets": support_targets,
                    "support_count": len(support_rows),
                    "support_primitives": support_primitives,
                    "rationale": spec.get("rationale"),
                    "prototype_command": _profile_command(target, "minimal"),
                    "validation_commands": _compound_validation_commands(target, support_rows),
                    "experiment_steps": [
                        "Verify the target and support targets with minimal profiles before changing code.",
                        "Add the missing primitive as a single-variable variant and keep the current primitive as the control.",
                        "Run the combined primitive variant only after the single-primitive guardrails pass.",
                    ],
                    "acceptance_gate": spec.get("acceptance_gate"),
                }
            )

    hypotheses.sort(
        key=lambda row: (
            -float(row.get("compound_score") or 0.0),
            len(row.get("missing_primitives") or []),
            str(row.get("compound_id") or ""),
            str(row.get("target") or ""),
        )
    )
    return {
        "hypothesis_count": len(hypotheses),
        "compound_count": len({row.get("compound") for row in hypotheses}),
        "policy": (
            "Compound primitives are experiment hypotheses, not claims. Isolate each primitive first, "
            "then measure the combined variant with unchanged correctness and guardrail metrics."
        ),
        "hypotheses": hypotheses,
    }


def _known_compound_primitive_pairs() -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for spec in COMPOUND_PRIMITIVE_SPECS:
        primitives = sorted(str(item) for item in spec.get("primitives", []) or [] if item)
        for left_index, left in enumerate(primitives):
            for right in primitives[left_index + 1 :]:
                pairs.add(tuple(sorted((left, right))))
    return pairs


def _primitive_pair_reason(left: str, right: str) -> str:
    pair = {left, right}
    if "kv_cache_layout" in pair and "persistent_kernel" in pair:
        return "KV-cache layout may expose a steadier decode loop that persistent residency can exploit."
    if "communication_overlap" in pair and "persistent_kernel" in pair:
        return "Persistent device-side work may hide more exposed fabric time than host-driven overlap alone."
    if "fp8_fp4_precision_path" in pair and "vectorized_memory_access" in pair:
        return "Lower-precision elements can make vectorized memory movement a larger fraction of useful work."
    if "tma_pipeline" in pair and "pinned_nonblocking_transfer" in pair:
        return "Host/device staging and device-side async movement may form one end-to-end copy pipeline."
    if "torch_compile_reduce_overhead" in pair and "persistent_kernel" in pair:
        return "Compilation can remove framework overhead while persistent kernels attack remaining launch and residency overhead."
    return (
        "The pair is source-backed but absent from the fixed compound catalog, so it is worth one isolated synthesis probe."
    )


def _build_primitive_pair_synthesis_plan(
    opportunities: list[dict[str, Any]], *, limit: int = 20
) -> dict[str, Any]:
    """Propose source-backed primitive pairs not covered by fixed compound specs."""

    rows = [row for row in opportunities if row.get("target")]
    primitive_specs = {
        str(spec.get("primitive")): spec
        for spec in OPTIMIZATION_PRIMITIVE_SPECS
        if spec.get("primitive")
    }
    primitive_names = sorted(primitive_specs)
    known_pairs = _known_compound_primitive_pairs()
    primitives_by_target = {
        str(row.get("target")): _primitive_names(row) for row in rows if row.get("target")
    }
    support_by_primitive: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for primitive in _primitive_names(row):
            support_by_primitive.setdefault(primitive, []).append(row)

    syntheses: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        target = str(row.get("target") or "")
        present_primitives = sorted(primitives_by_target.get(target, set()))
        if not target or not present_primitives:
            continue
        for present in present_primitives:
            for candidate in primitive_names:
                if candidate in present_primitives:
                    continue
                pair = tuple(sorted((present, candidate)))
                if pair in known_pairs:
                    continue
                key = (target, pair[0], pair[1])
                if key in seen:
                    continue
                support_rows = [
                    support
                    for support in support_by_primitive.get(candidate, [])
                    if str(support.get("target") or "") != target
                ]
                if not support_rows:
                    continue
                support_rows.sort(
                    key=lambda support: (
                        -float(support.get("score") or 0.0),
                        str(support.get("target") or ""),
                    )
                )
                source_support_rows = [
                    support
                    for support in support_by_primitive.get(present, [])
                    if str(support.get("target") or "") != target
                ]
                synthesis_score = (
                    float(row.get("score") or 0.0)
                    + min(len(support_rows), 5) * 3.0
                    + min(len(source_support_rows), 5) * 1.5
                    + len(_row_signal_set(row)) * 1.2
                )
                seen.add(key)
                syntheses.append(
                    {
                        "synthesis_id": _slug(
                            f"synthesis-{target}-{present}-{candidate}", max_len=96
                        ),
                        "target": target,
                        "priority": row.get("priority"),
                        "synthesis_score": round(synthesis_score, 3),
                        "pair": list(pair),
                        "present_primitive": present,
                        "candidate_primitive": candidate,
                        "pair_state": "not_in_known_compound_specs",
                        "support_targets": [
                            str(support.get("target")) for support in support_rows[:5]
                        ],
                        "support_count": len(support_rows),
                        "frontier_signals": sorted(_row_signal_set(row)),
                        "hypothesis": _primitive_pair_reason(pair[0], pair[1]),
                        "candidate_transfer_question": primitive_specs.get(candidate, {}).get(
                            "transfer_question"
                        ),
                        "prototype_command": _profile_command(target, "minimal"),
                        "experiment_steps": [
                            "Re-run the target control and the present primitive as the unchanged baseline.",
                            "Introduce the candidate primitive as a one-variable isolated variant.",
                            "Run the combined pair only after both single-primitive measurements are clean.",
                        ],
                        "acceptance_gate": (
                            "Accept only if the combined pair beats the best isolated primitive on the same workload contract, with correctness and guardrail metrics clean."
                        ),
                        "claim_boundary": (
                            "Claim this only as an untried primitive-pair hypothesis until a full validation queue captures control, candidate, profile, and review evidence."
                        ),
                    }
                )

    syntheses.sort(
        key=lambda row: (
            -float(row.get("synthesis_score") or 0.0),
            str(row.get("target") or ""),
            str(row.get("present_primitive") or ""),
            str(row.get("candidate_primitive") or ""),
        )
    )
    selected = syntheses[:limit]
    return {
        "synthesis_count": len(selected),
        "available_synthesis_count": len(syntheses),
        "target_count": len({item.get("target") for item in selected}),
        "primitive_pair_count": len({tuple(item.get("pair", [])) for item in selected}),
        "policy": (
            "Use primitive-pair synthesis to find source-backed compound ideas beyond the fixed compound catalog. Every item is advisory until isolated single-primitive evidence and a combined-variant gate pass."
        ),
        "syntheses": selected,
    }


def _signal_bonus(signal: str) -> float:
    for spec in FRONTIER_SIGNAL_SPECS:
        if spec.get("signal") == signal:
            return float(spec.get("bonus") or 0.0)
    return 0.0


def _row_signal_set(row: dict[str, Any]) -> set[str]:
    synthetic_signals = {"catalog_category", "catalog_rationale"}
    return {
        str(signal)
        for signal in row.get("frontier_signals", []) or []
        if str(signal) and str(signal) not in synthetic_signals
    }


def _bridge_validation_commands(
    prototype_target: str, support_rows: list[dict[str, Any]]
) -> list[str]:
    commands = [
        _profile_command(prototype_target, "minimal"),
        _profile_command(prototype_target, "deep_dive"),
    ]
    commands.extend(
        _profile_command(str(row.get("target")), "minimal")
        for row in support_rows[:3]
        if row.get("target") and str(row.get("target")) != prototype_target
    )
    deduped: list[str] = []
    for command in commands:
        if command not in deduped:
            deduped.append(command)
    return deduped


def _build_cross_lane_bridge_map(opportunities: list[dict[str, Any]]) -> dict[str, Any]:
    """Find multi-lane experiment bridges from source-mined frontier signals."""

    rows = [row for row in opportunities if row.get("target")]
    bridges: list[dict[str, Any]] = []
    for spec in CROSS_LANE_BRIDGE_SPECS:
        required_signals = {str(signal) for signal in spec.get("signals", []) if signal}
        primitive_hints = {str(item) for item in spec.get("primitives", []) if item}
        if len(required_signals) < 2:
            continue

        direct_rows = [row for row in rows if required_signals <= _row_signal_set(row)]
        adjacent_rows = [
            row
            for row in rows
            if row not in direct_rows
            and len(_row_signal_set(row) & required_signals) >= len(required_signals) - 1
            and bool(_primitive_names(row) & primitive_hints)
        ]
        if not direct_rows and not adjacent_rows:
            continue

        support_rows = direct_rows or adjacent_rows
        support_rows.sort(
            key=lambda row: (
                -len(_primitive_names(row) & primitive_hints),
                -float(row.get("score") or 0.0),
                str(row.get("target") or ""),
            )
        )
        prototype = support_rows[0]
        prototype_target = str(prototype.get("target") or "")
        present_primitives = sorted(
            {
                primitive
                for row in support_rows[:8]
                for primitive in _primitive_names(row)
                if primitive in primitive_hints
            }
        )
        missing_primitives = sorted(primitive_hints - set(present_primitives))
        signal_strength = sum(_signal_bonus(signal) for signal in required_signals)
        bridge_score = (
            signal_strength
            + min(len(direct_rows), 10) * 3.0
            + min(len(adjacent_rows), 10) * 1.5
            + len(present_primitives) * 4.0
            - len(missing_primitives) * 1.5
        )
        bridges.append(
            {
                "bridge_id": _slug(f"bridge-{spec.get('bridge')}"),
                "bridge": spec.get("bridge"),
                "name": spec.get("name"),
                "signals": sorted(required_signals),
                "primitive_hints": sorted(primitive_hints),
                "present_primitives": present_primitives,
                "missing_primitives": missing_primitives,
                "support_count": len(support_rows),
                "direct_support_count": len(direct_rows),
                "adjacent_support_count": len(adjacent_rows),
                "prototype_target": prototype_target,
                "support_targets": [str(row.get("target")) for row in support_rows[:6]],
                "bridge_score": round(bridge_score, 3),
                "rationale": spec.get("rationale"),
                "experiments": list(spec.get("experiments", [])),
                "prototype_command": _profile_command(prototype_target, "minimal"),
                "validation_commands": _bridge_validation_commands(prototype_target, support_rows),
                "acceptance_gate": spec.get("acceptance_gate"),
                "evidence_contract": (
                    "Bridge experiments must keep one workload contract, isolate each variable before combining lanes, "
                    "and report all guardrail metrics next to the primary metric."
                ),
            }
        )

    bridges.sort(
        key=lambda row: (
            -float(row.get("bridge_score") or 0.0),
            -int(row.get("direct_support_count") or 0),
            str(row.get("bridge_id") or ""),
        )
    )
    return {
        "bridge_count": len(bridges),
        "policy": (
            "Cross-lane bridges are source-mined experiment hypotheses. Use them when a target spans multiple high-value frontier signals, then validate each lane before combining changes."
        ),
        "bridges": bridges,
    }


def _sample_targets(rows: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (-float(row.get("score") or 0.0), str(row.get("target") or "")),
    )
    return [
        {
            "target": row.get("target"),
            "score": row.get("score"),
            "command": _profile_command(str(row.get("target")), "minimal"),
        }
        for row in sorted_rows[:limit]
        if row.get("target")
    ]


def _primitive_row_matches(
    row: dict[str, Any], primitive_name: str, *, introduced: bool | None = None
) -> bool:
    for primitive in row.get("optimization_primitives", []) or []:
        if not isinstance(primitive, dict):
            continue
        if str(primitive.get("primitive") or "") != primitive_name:
            continue
        if introduced is None or bool(primitive.get("introduced")) is introduced:
            return True
    return False


def _build_coverage_gap_map(opportunities: list[dict[str, Any]]) -> dict[str, Any]:
    """Find negative-space experiment leads in the current benchmark catalog."""

    rows = [row for row in opportunities if row.get("target")]
    synthetic_signals = {"catalog_category", "catalog_rationale"}
    signal_counts = Counter(
        str(signal)
        for row in rows
        for signal in row.get("frontier_signals", []) or []
        if str(signal) and str(signal) not in synthetic_signals
    )
    primitive_counts = Counter(
        str(primitive.get("primitive") or "")
        for row in rows
        for primitive in row.get("optimization_primitives", []) or []
        if isinstance(primitive, dict) and primitive.get("primitive")
    )
    introduced_counts = Counter(
        str(primitive.get("primitive") or "")
        for row in rows
        for primitive in row.get("optimization_primitives", []) or []
        if isinstance(primitive, dict)
        and primitive.get("primitive")
        and primitive.get("introduced")
    )
    primitives_by_target = {
        str(row.get("target")): _primitive_names(row) for row in rows if row.get("target")
    }

    total_targets = len(rows)
    signal_floor = max(3, min(12, math.ceil(total_targets * 0.02)))
    introduced_floor = max(2, min(8, math.ceil(total_targets * 0.01)))
    min_introduction_ratio = 0.25

    signal_coverage: list[dict[str, Any]] = []
    signal_gaps: list[dict[str, Any]] = []
    for spec in FRONTIER_SIGNAL_SPECS:
        signal = str(spec.get("signal") or "")
        count = int(signal_counts.get(signal, 0))
        matching_rows = [
            row for row in rows if signal in set(row.get("frontier_signals", []) or [])
        ]
        if count == 0:
            coverage_state = "missing"
        elif count < signal_floor:
            coverage_state = "thin"
        else:
            coverage_state = "covered"
        coverage = {
            "signal": signal,
            "target_count": count,
            "target_floor": signal_floor,
            "coverage_state": coverage_state,
            "reason": spec.get("reason"),
            "terms": list(spec.get("terms", []))[:8],
            "sample_targets": _sample_targets(matching_rows, limit=3),
        }
        signal_coverage.append(coverage)
        if coverage_state == "covered":
            continue
        gap_score = float(spec.get("bonus") or 1.0) * (1.0 + (signal_floor - count) / signal_floor)
        signal_gaps.append(
            coverage
            | {
                "gap_id": _slug(f"coverage-signal-{signal}"),
                "gap_type": "frontier_signal",
                "gap_score": round(gap_score, 3),
                "recommended_action": (
                    "Create or catalog runnable benchmark pairs for these terms before optimizing inside already-dense lanes."
                    if count == 0
                    else "Run the best current smoke target, then add nearby benchmark pairs until this lane has enough diversity."
                ),
                "first_probe_command": (
                    _profile_command(str(matching_rows[0].get("target")), "minimal")
                    if matching_rows
                    else None
                ),
                "acceptance_gate": "A gap closes only after runnable targets have clean minimal-profile evidence and the lane has at least the target_floor count.",
            }
        )

    primitive_coverage: list[dict[str, Any]] = []
    primitive_gaps: list[dict[str, Any]] = []
    for spec in OPTIMIZATION_PRIMITIVE_SPECS:
        primitive_name = str(spec.get("primitive") or "")
        matched_count = int(primitive_counts.get(primitive_name, 0))
        introduced_count = int(introduced_counts.get(primitive_name, 0))
        introduction_ratio = introduced_count / matched_count if matched_count else 0.0
        matched_rows = [row for row in rows if _primitive_row_matches(row, primitive_name)]
        not_introduced_rows = [
            row
            for row in matched_rows
            if not _primitive_row_matches(row, primitive_name, introduced=True)
        ]
        if matched_count == 0:
            coverage_state = "missing"
        elif introduced_count == 0:
            coverage_state = "not_introduced"
        elif introduced_count < introduced_floor or introduction_ratio < min_introduction_ratio:
            coverage_state = "thin_introduced"
        else:
            coverage_state = "covered"
        coverage = {
            "primitive": primitive_name,
            "matched_target_count": matched_count,
            "introduced_target_count": introduced_count,
            "introduced_floor": introduced_floor,
            "introduction_ratio": round(introduction_ratio, 3),
            "coverage_state": coverage_state,
            "why": spec.get("why"),
            "terms": list(spec.get("terms", []))[:8],
            "sample_targets": _sample_targets(not_introduced_rows or matched_rows, limit=3),
        }
        primitive_coverage.append(coverage)
        if coverage_state == "covered":
            continue
        missing_intro = max(0, introduced_floor - introduced_count)
        ratio_gap = max(0.0, min_introduction_ratio - introduction_ratio)
        gap_score = 16.0 + missing_intro * 2.0 + ratio_gap * 20.0
        first_target_rows = not_introduced_rows or matched_rows
        primitive_gaps.append(
            coverage
            | {
                "gap_id": _slug(f"coverage-primitive-{primitive_name}"),
                "gap_type": "optimization_primitive",
                "gap_score": round(gap_score, 3),
                "recommended_action": (
                    "Add a source-mined benchmark pair that exercises this primitive, then require a baseline-vs-optimized delta term."
                    if matched_count == 0
                    else "Pick a matched target where the primitive is not introduced, add one optimized variant that introduces it, and re-run guardrails."
                ),
                "first_probe_command": (
                    _profile_command(str(first_target_rows[0].get("target")), "minimal")
                    if first_target_rows
                    else None
                ),
                "acceptance_gate": spec.get("transfer_question")
                or "The gap closes only after the primitive is introduced in optimized source and validated with clean guardrail metrics.",
            }
        )

    compound_coverage: list[dict[str, Any]] = []
    compound_gaps: list[dict[str, Any]] = []
    for spec in COMPOUND_PRIMITIVE_SPECS:
        required = {str(item) for item in spec.get("primitives", []) if item}
        if len(required) < 2:
            continue
        complete_rows = [
            row
            for row in rows
            if required <= primitives_by_target.get(str(row.get("target") or ""), set())
        ]
        partial_rows = [
            row
            for row in rows
            if primitives_by_target.get(str(row.get("target") or ""), set()) & required
            and not required <= primitives_by_target.get(str(row.get("target") or ""), set())
        ]
        complete_count = len(complete_rows)
        partial_count = len(partial_rows)
        if partial_count and complete_count == 0:
            coverage_state = "missing_complete"
        elif partial_count >= max(4, complete_count * 3) and complete_count < introduced_floor:
            coverage_state = "thin_complete"
        else:
            coverage_state = "covered"
        coverage = {
            "compound": spec.get("compound"),
            "name": spec.get("name"),
            "primitives": sorted(required),
            "complete_target_count": complete_count,
            "partial_target_count": partial_count,
            "coverage_state": coverage_state,
            "sample_targets": _sample_targets(partial_rows or complete_rows, limit=3),
        }
        compound_coverage.append(coverage)
        if coverage_state == "covered":
            continue
        gap_score = 12.0 + min(partial_count, 12) - complete_count * 2.0
        compound_gaps.append(
            coverage
            | {
                "gap_id": _slug(f"coverage-compound-{spec.get('compound')}"),
                "gap_type": "compound_primitive",
                "gap_score": round(gap_score, 3),
                "recommended_action": "Select one partial target, isolate the missing primitive first, then test the combined stack only after guardrails pass.",
                "first_probe_command": (
                    _profile_command(str(partial_rows[0].get("target")), "minimal")
                    if partial_rows
                    else None
                ),
                "acceptance_gate": spec.get("acceptance_gate"),
            }
        )

    gaps = sorted(
        [*signal_gaps, *primitive_gaps, *compound_gaps],
        key=lambda row: (
            -float(row.get("gap_score") or 0.0),
            str(row.get("gap_type") or ""),
            str(row.get("gap_id") or ""),
        ),
    )
    return {
        "target_count": total_targets,
        "gap_count": len(gaps),
        "signal_gap_count": len(signal_gaps),
        "primitive_gap_count": len(primitive_gaps),
        "compound_gap_count": len(compound_gaps),
        "policy": (
            "Coverage gaps are negative-space leads. Close them by adding or validating evidence before treating them as optimization wins."
        ),
        "signal_coverage": signal_coverage,
        "primitive_coverage": primitive_coverage,
        "compound_coverage": compound_coverage,
        "gaps": gaps,
    }


def _lead_target_from_samples(samples: list[dict[str, Any]]) -> str | None:
    for sample in samples:
        if isinstance(sample, dict) and sample.get("target"):
            return str(sample.get("target"))
    return None


def _build_novelty_queue(
    *,
    frontier_map: dict[str, Any],
    transfer_map: dict[str, Any],
    compound_hypotheses: dict[str, Any],
    coverage_gap_map: dict[str, Any],
    cross_lane_bridge_map: dict[str, Any],
    limit: int = 30,
) -> dict[str, Any]:
    """Merge the radar's novelty surfaces into one ranked action queue."""

    leads: list[dict[str, Any]] = []

    for item in frontier_map.get("diversity_queue", []) or []:
        target = str(item.get("target") or "")
        if not target:
            continue
        leads.append(
            {
                "lead_id": _slug(f"novelty-frontier-{target}"),
                "lead_type": "frontier_probe",
                "source": "frontier_discovery_map",
                "title": f"First-evidence probe for {target}",
                "target": target,
                "novelty_score": round(30.0 + float(item.get("score") or 0.0) / 10.0, 3),
                "command": item.get("command"),
                "why": f"Frontier lane `{item.get('lane')}` needs first clean evidence before deeper optimization.",
                "evidence_gate": "Minimal profile succeeds with output verification and artifact paths before any speedup claim.",
                "related_ids": list(item.get("blueprint_ids", []) or []),
            }
        )

    for pattern in transfer_map.get("patterns", []) or []:
        recipient_count = int(pattern.get("recipient_count") or 0)
        if recipient_count <= 0:
            continue
        pattern_name = str(pattern.get("pattern") or "")
        source_target = str(pattern.get("source_target") or "")
        leads.append(
            {
                "lead_id": _slug(f"novelty-transfer-{pattern_name}-{source_target}"),
                "lead_type": "source_transfer",
                "source": "source_transfer_map",
                "title": f"Transfer {pattern_name} from {source_target}",
                "target": source_target,
                "novelty_score": round(
                    28.0
                    + min(16.0, math.log2(recipient_count + 1.0) * 3.0)
                    + (5.0 if pattern.get("pattern_type") == "optimization_primitive" else 0.0),
                    3,
                ),
                "command": pattern.get("prototype_command"),
                "why": f"Source-mined pattern has {recipient_count} recipient targets.",
                "evidence_gate": pattern.get("adoption_gate"),
                "related_targets": list(pattern.get("recipient_targets", []) or []),
                "related_ids": list(pattern.get("blueprint_ids", []) or []),
            }
        )

    for hypothesis in compound_hypotheses.get("hypotheses", []) or []:
        target = str(hypothesis.get("target") or "")
        if not target:
            continue
        leads.append(
            {
                "lead_id": _slug(
                    f"novelty-compound-{hypothesis.get('compound_id')}-{target}",
                    max_len=96,
                ),
                "lead_type": "compound_primitive",
                "source": "compound_primitive_hypotheses",
                "title": str(hypothesis.get("name") or hypothesis.get("compound_id")),
                "target": target,
                "novelty_score": round(
                    34.0 + float(hypothesis.get("compound_score") or 0.0) / 5.0, 3
                ),
                "command": hypothesis.get("prototype_command"),
                "why": (
                    "Target has "
                    f"{', '.join(str(item) for item in hypothesis.get('present_primitives', []) or [])}; "
                    "test missing "
                    f"{', '.join(str(item) for item in hypothesis.get('missing_primitives', []) or [])}."
                ),
                "evidence_gate": hypothesis.get("acceptance_gate"),
                "related_targets": list(hypothesis.get("support_targets", []) or []),
                "related_ids": [str(hypothesis.get("compound_id"))],
            }
        )

    for gap in coverage_gap_map.get("gaps", []) or []:
        samples = [item for item in gap.get("sample_targets", []) or [] if isinstance(item, dict)]
        label = (
            gap.get("name")
            or gap.get("primitive")
            or gap.get("signal")
            or gap.get("compound")
            or gap.get("gap_id")
        )
        leads.append(
            {
                "lead_id": _slug(f"novelty-gap-{gap.get('gap_id')}"),
                "lead_type": "coverage_gap",
                "source": "coverage_gap_map",
                "title": str(label),
                "target": _lead_target_from_samples(samples),
                "novelty_score": round(26.0 + float(gap.get("gap_score") or 0.0), 3),
                "command": gap.get("first_probe_command"),
                "why": f"{gap.get('gap_type')} is {gap.get('coverage_state')}.",
                "evidence_gate": gap.get("acceptance_gate"),
                "related_targets": [
                    str(sample.get("target")) for sample in samples if sample.get("target")
                ],
                "related_ids": [str(gap.get("gap_id"))],
            }
        )

    for bridge in cross_lane_bridge_map.get("bridges", []) or []:
        target = str(bridge.get("prototype_target") or "")
        if not target:
            continue
        leads.append(
            {
                "lead_id": _slug(f"novelty-bridge-{bridge.get('bridge_id')}-{target}", max_len=96),
                "lead_type": "cross_lane_bridge",
                "source": "cross_lane_bridge_map",
                "title": str(bridge.get("name") or bridge.get("bridge_id")),
                "target": target,
                "novelty_score": round(32.0 + float(bridge.get("bridge_score") or 0.0) / 2.0, 3),
                "command": bridge.get("prototype_command"),
                "why": (
                    "Bridge signals "
                    f"{', '.join(str(item) for item in bridge.get('signals', []) or [])} "
                    f"with primitives {', '.join(str(item) for item in bridge.get('present_primitives', []) or [])}."
                ),
                "evidence_gate": bridge.get("acceptance_gate"),
                "related_targets": list(bridge.get("support_targets", []) or []),
                "related_ids": [str(bridge.get("bridge_id"))],
            }
        )

    deduped: dict[str, dict[str, Any]] = {}
    for lead in leads:
        lead_id = str(lead.get("lead_id") or "")
        if not lead_id:
            continue
        existing = deduped.get(lead_id)
        if existing is None or float(lead.get("novelty_score") or 0.0) > float(
            existing.get("novelty_score") or 0.0
        ):
            deduped[lead_id] = lead

    sorted_leads = sorted(
        deduped.values(),
        key=lambda row: (
            -float(row.get("novelty_score") or 0.0),
            str(row.get("lead_type") or ""),
            str(row.get("title") or ""),
            str(row.get("target") or ""),
        ),
    )
    buckets: dict[str, list[dict[str, Any]]] = {}
    for lead in sorted_leads:
        buckets.setdefault(str(lead.get("lead_type") or "unknown"), []).append(lead)

    type_order = sorted(
        buckets,
        key=lambda lead_type: (
            -float(buckets[lead_type][0].get("novelty_score") or 0.0),
            lead_type,
        ),
    )
    ranked: list[dict[str, Any]] = []
    while len(ranked) < limit and any(buckets.values()):
        for lead_type in type_order:
            if not buckets.get(lead_type):
                continue
            ranked.append(buckets[lead_type].pop(0))
            if len(ranked) >= limit:
                break
    for index, lead in enumerate(ranked, start=1):
        lead["queue_rank"] = index
    return {
        "lead_count": len(ranked),
        "available_lead_count": len(deduped),
        "lead_type_counts": dict(Counter(str(row.get("lead_type")) for row in ranked)),
        "policy": (
            "Use the novelty queue to choose the next experiment lead; it is score-ranked within each lead type and interleaved for diversity. Every item must still pass its evidence gate before promotion."
        ),
        "leads": ranked,
    }


def _novelty_validation_stage_name(lead_type: str) -> str:
    if lead_type == "frontier_probe":
        return "first_evidence"
    if lead_type == "source_transfer":
        return "transfer_variant"
    if lead_type == "compound_primitive":
        return "compound_variant"
    if lead_type == "coverage_gap":
        return "gap_closure"
    if lead_type == "cross_lane_bridge":
        return "bridge_variant"
    return "candidate_variant"


def _novelty_validation_job_ids(lead_id: str, lead_type: str) -> dict[str, str]:
    stage_name = _novelty_validation_stage_name(lead_type)
    return {
        "stage_name": stage_name,
        "control": _slug(f"{lead_id}-control", max_len=96),
        "candidate": _slug(f"{lead_id}-{stage_name}", max_len=96),
        "profile": _slug(f"{lead_id}-deep-dive", max_len=96),
        "review": _slug(f"{lead_id}-review", max_len=96),
    }


def _playbook_profiler_tools(lead: dict[str, Any]) -> list[str]:
    text = " ".join(
        str(value)
        for value in [
            lead.get("title"),
            lead.get("lead_type"),
            lead.get("why"),
            lead.get("target"),
        ]
    ).lower()
    tools = ["nsys"]
    if any(term in text for term in ("memory", "hbm", "cache", "tma", "tile", "precision")):
        tools.append("ncu")
    if any(term in text for term in ("decode", "launch", "graph", "compile", "precision")):
        tools.append("zymtrace")
    if any(term in text for term in ("fabric", "distributed", "allreduce", "nccl", "nvlink")):
        tools.append("hta")
    return sorted(set(tools))


def _playbook_metric_profile(lead_type: str) -> tuple[list[str], list[str]]:
    if lead_type == "frontier_probe":
        return (
            ["verified_status", "median_wall_time_ms", "dominant_bottleneck"],
            ["output_correctness", "artifact_paths_present", "baseline_optimized_pair_exists"],
        )
    if lead_type == "source_transfer":
        return (
            ["recipient_median_wall_time_ms", "source_pattern_metric", "transfer_delta_pct"],
            ["output_correctness", "same_workload_contract", "source_target_control_clean"],
        )
    if lead_type == "compound_primitive":
        return (
            ["combined_variant_ms", "best_isolated_primitive_ms", "guardrail_delta_pct"],
            ["output_correctness", "each_primitive_isolated", "memory_savings_pct"],
        )
    if lead_type == "coverage_gap":
        return (
            ["coverage_state", "introduced_term_count", "verified_status"],
            ["source_delta_present", "output_correctness", "artifact_paths_present"],
        )
    if lead_type == "cross_lane_bridge":
        return (
            ["primary_lane_metric", "secondary_lane_metric", "median_wall_time_ms"],
            ["both_lane_metrics_present", "output_correctness", "p95_latency_or_drift"],
        )
    return (
        ["median_wall_time_ms", "verified_status"],
        ["output_correctness", "artifact_paths_present"],
    )


def _playbook_variables(lead_type: str) -> list[str]:
    if lead_type == "frontier_probe":
        return ["profile_mode", "input_shape", "first bottleneck classification"]
    if lead_type == "source_transfer":
        return ["source pattern", "recipient target", "single transferred variable"]
    if lead_type == "compound_primitive":
        return ["present primitive", "missing primitive", "combined primitive stack"]
    if lead_type == "coverage_gap":
        return ["coverage state", "source delta term", "gap closure candidate"]
    if lead_type == "cross_lane_bridge":
        return ["first lane variant", "second lane variant", "combined bridge variant"]
    return ["candidate variable", "input shape", "profile mode"]


def _playbook_variant_ladder(lead: dict[str, Any]) -> list[dict[str, Any]]:
    target = str(lead.get("target") or "")
    lead_type = str(lead.get("lead_type") or "")
    control_command = _profile_command(target, "minimal") if target else lead.get("command")
    candidate_command = lead.get("command") or control_command
    deep_command = _profile_command(target, "deep_dive") if target else None
    if lead_type == "cross_lane_bridge":
        middle = [
            {
                "variant": "lane_a_isolated",
                "command": candidate_command,
                "purpose": "Measure the first signal path without combining the second lane.",
            },
            {
                "variant": "lane_b_isolated",
                "command": candidate_command,
                "purpose": "Measure the second signal path with the same workload contract.",
            },
            {
                "variant": "bridge_combined",
                "command": candidate_command,
                "purpose": "Combine both lanes only after isolated metrics are captured.",
            },
        ]
    elif lead_type == "compound_primitive":
        middle = [
            {
                "variant": "present_primitive_control",
                "command": control_command,
                "purpose": "Re-measure the primitive already present on the target.",
            },
            {
                "variant": "missing_primitive_isolated",
                "command": candidate_command,
                "purpose": "Introduce or simulate the missing primitive as a one-variable change.",
            },
            {
                "variant": "combined_stack",
                "command": candidate_command,
                "purpose": "Run the combined primitive stack only after both isolated variants pass.",
            },
        ]
    elif lead_type == "coverage_gap":
        middle = [
            {
                "variant": "gap_candidate",
                "command": candidate_command,
                "purpose": "Create evidence for the missing or under-introduced coverage area.",
            },
            {
                "variant": "coverage_recompute",
                "command": None,
                "purpose": "Rerun the radar and confirm the coverage state changed or was disproven.",
            },
        ]
    elif lead_type == "source_transfer":
        middle = [
            {
                "variant": "source_pattern_control",
                "command": control_command,
                "purpose": "Keep source-pattern evidence separate from recipient validation.",
            },
            {
                "variant": "recipient_transfer_candidate",
                "command": candidate_command,
                "purpose": "Apply one source-mined pattern to the recipient target.",
            },
        ]
    else:
        middle = [
            {
                "variant": "frontier_minimal_smoke",
                "command": candidate_command,
                "purpose": "Establish first clean evidence before optimization.",
            }
        ]
    ladder = [
        {
            "variant": "control_verified_current",
            "command": control_command,
            "purpose": "Capture a clean control with output verification before testing novelty.",
        },
        *middle,
    ]
    if deep_command:
        ladder.append(
            {
                "variant": "deep_profile_followup",
                "command": deep_command,
                "purpose": "Capture bottleneck evidence after the candidate path is clean.",
            }
        )
    return ladder


def _build_novelty_experiment_playbooks(
    novelty_queue: dict[str, Any], *, limit: int = 30
) -> dict[str, Any]:
    """Build experiment-design playbooks for the highest-priority novelty leads."""

    playbooks: list[dict[str, Any]] = []
    for lead in list(novelty_queue.get("leads") or [])[:limit]:
        lead_id = str(lead.get("lead_id") or "")
        if not lead_id:
            continue
        lead_type = str(lead.get("lead_type") or "")
        primary_metrics, guardrails = _playbook_metric_profile(lead_type)
        playbooks.append(
            {
                "playbook_id": _slug(f"playbook-{lead_id}", max_len=96),
                "lead_id": lead_id,
                "queue_rank": lead.get("queue_rank"),
                "lead_type": lead_type,
                "title": lead.get("title"),
                "target": lead.get("target"),
                "hypothesis": lead.get("why"),
                "variables": _playbook_variables(lead_type),
                "variant_ladder": _playbook_variant_ladder(lead),
                "primary_metrics": primary_metrics,
                "guardrail_metrics": guardrails,
                "profiler_tools": _playbook_profiler_tools(lead),
                "expected_artifacts": [
                    "control run metadata and logs",
                    "candidate run metadata and logs",
                    "deep profile trace or explicit not-applicable note",
                    "manual review checklist with evidence references",
                ],
                "stop_conditions": [
                    "control fails output verification",
                    "candidate changes more than one declared variable",
                    "guardrail metric regresses without an attached explanation",
                ],
                "promotion_gate": lead.get("evidence_gate"),
            }
        )
    return {
        "playbook_count": len(playbooks),
        "policy": (
            "Playbooks turn novelty leads into controlled experiments. Run variants in ladder order and keep claims blocked until the promotion gate and stop conditions are reviewed."
        ),
        "playbooks": playbooks,
    }


def _mutation_templates_for_lead(
    lead: dict[str, Any], playbook: dict[str, Any]
) -> list[dict[str, Any]]:
    lead_type = str(lead.get("lead_type") or "")
    target = str(lead.get("target") or playbook.get("target") or "")
    minimal_command = _profile_command(target, "minimal") if target else lead.get("command")
    deep_command = _profile_command(target, "deep_dive") if target else None
    if lead_type == "frontier_probe":
        return [
            {
                "operator": "profile_mode_escalation",
                "variable": "profile_mode",
                "command": deep_command or minimal_command,
                "implementation_hint": (
                    "After first clean evidence, rerun the same target with deep profiling and classify the dominant bottleneck."
                ),
                "expected_signal": "dominant_bottleneck",
                "guardrail": "minimal smoke has already passed with output verification",
            },
            {
                "operator": "shape_stress_sweep",
                "variable": "input_shape",
                "command": minimal_command,
                "implementation_hint": (
                    "Probe one larger and one smaller shape while keeping dtype, seed, and verification fixed."
                ),
                "expected_signal": "shape_sensitive_runtime_or_memory_delta",
                "guardrail": "only one shape dimension changes per run",
            },
            {
                "operator": "nearby_motif_probe",
                "variable": "frontier_motif",
                "command": minimal_command,
                "implementation_hint": (
                    "Compare the target against the nearest motif lane before committing to a specialized implementation."
                ),
                "expected_signal": "motif_specific_bottleneck",
                "guardrail": "do not claim speedup from motif classification alone",
            },
        ]
    if lead_type == "source_transfer":
        return [
            {
                "operator": "recipient_shape_transfer",
                "variable": "recipient_target",
                "command": minimal_command,
                "implementation_hint": (
                    "Apply the source pattern to one recipient shape while preserving the recipient control path."
                ),
                "expected_signal": "transfer_delta_pct",
                "guardrail": "source pattern is replayed separately from recipient evidence",
            },
            {
                "operator": "sham_transfer_control",
                "variable": "transferred_pattern",
                "command": minimal_command,
                "implementation_hint": (
                    "Run all setup changes without the source pattern to prove the transferred variable explains the delta."
                ),
                "expected_signal": "sham_delta_below_candidate_delta",
                "guardrail": "candidate and sham runs use the same workload contract",
            },
            {
                "operator": "recipient_guardrail_sweep",
                "variable": "guardrail_metric",
                "command": minimal_command,
                "implementation_hint": (
                    "Check the recipient's correctness, memory, and latency guardrails before widening transfer scope."
                ),
                "expected_signal": "clean_recipient_guardrails",
                "guardrail": "claim remains scoped to the measured recipient shape",
            },
        ]
    if lead_type == "compound_primitive":
        return [
            {
                "operator": "missing_primitive_isolation",
                "variable": "missing_primitive",
                "command": minimal_command,
                "implementation_hint": "Introduce or simulate exactly one missing primitive before testing the stack.",
                "expected_signal": "isolated_primitive_delta",
                "guardrail": "best isolated primitive is recorded before combined-stack review",
            },
            {
                "operator": "primitive_order_swap",
                "variable": "primitive_order",
                "command": minimal_command,
                "implementation_hint": (
                    "Swap the order of primitive application to detect whether the claimed interaction is order-sensitive."
                ),
                "expected_signal": "order_sensitive_delta",
                "guardrail": "only primitive order changes",
            },
            {
                "operator": "single_primitive_revert",
                "variable": "primitive_toggle",
                "command": minimal_command,
                "implementation_hint": (
                    "Disable one primitive at a time after the combined variant passes to prove the interaction."
                ),
                "expected_signal": "delta_drop_when_primitive_removed",
                "guardrail": "combined stack must pass correctness before reverts are interpreted",
            },
        ]
    if lead_type == "coverage_gap":
        return [
            {
                "operator": "gap_candidate_probe",
                "variable": "gap_closure_candidate",
                "command": minimal_command,
                "implementation_hint": (
                    "Create one candidate artifact that introduces the missing signal, primitive, or compound stack."
                ),
                "expected_signal": "coverage_state_changes",
                "guardrail": "candidate artifact must contain the missing evidence, not just renamed taxonomy",
            },
            {
                "operator": "taxonomy_only_recompute",
                "variable": "coverage_taxonomy",
                "command": None,
                "implementation_hint": (
                    "Rerun the radar without new artifacts to prove the gap does not close by taxonomy drift alone."
                ),
                "expected_signal": "gap_remains_open_without_new_artifacts",
                "guardrail": "no performance claim is allowed from taxonomy-only changes",
            },
            {
                "operator": "sample_target_rotation",
                "variable": "sample_target",
                "command": minimal_command,
                "implementation_hint": (
                    "Rotate to another sample target if the first one cannot expose the missing primitive cleanly."
                ),
                "expected_signal": "alternate_target_exposes_gap",
                "guardrail": "each sample target keeps an isolated artifact directory",
            },
        ]
    if lead_type == "cross_lane_bridge":
        return [
            {
                "operator": "lane_a_only",
                "variable": "first_lane_variant",
                "command": minimal_command,
                "implementation_hint": "Run the first bridge lane without the second lane enabled.",
                "expected_signal": "lane_a_delta",
                "guardrail": "lane A alone must not explain the full combined effect",
            },
            {
                "operator": "lane_b_only",
                "variable": "second_lane_variant",
                "command": minimal_command,
                "implementation_hint": "Run the second bridge lane without the first lane enabled.",
                "expected_signal": "lane_b_delta",
                "guardrail": "lane B alone must not explain the full combined effect",
            },
            {
                "operator": "bridge_interaction_toggle",
                "variable": "combined_bridge_variant",
                "command": minimal_command,
                "implementation_hint": (
                    "Turn on both lanes only after isolated lane metrics exist, then measure the interaction delta."
                ),
                "expected_signal": "combined_delta_exceeds_isolated_lanes",
                "guardrail": "both lane metrics and output correctness remain clean",
            },
        ]
    return [
        {
            "operator": "single_variable_candidate",
            "variable": "candidate_variable",
            "command": minimal_command,
            "implementation_hint": "Change exactly one candidate variable and preserve the workload contract.",
            "expected_signal": "candidate_control_delta",
            "guardrail": "candidate uses the same verification contract as control",
        }
    ]


def _build_novelty_mutation_plan(
    novelty_queue: dict[str, Any],
    novelty_playbooks: dict[str, Any],
    *,
    max_mutations_per_lead: int = 3,
) -> dict[str, Any]:
    """Expand novelty leads into bounded one-variable mutation candidates."""

    lead_by_id = {
        str(lead.get("lead_id")): lead
        for lead in novelty_queue.get("leads", []) or []
        if lead.get("lead_id")
    }
    lead_mutations: list[dict[str, Any]] = []
    mutations: list[dict[str, Any]] = []
    for playbook in novelty_playbooks.get("playbooks", []) or []:
        lead_id = str(playbook.get("lead_id") or "")
        lead = lead_by_id.get(lead_id, {})
        if not lead_id or not lead:
            continue
        templates = _mutation_templates_for_lead(lead, playbook)[:max_mutations_per_lead]
        lead_items: list[dict[str, Any]] = []
        for index, template in enumerate(templates, start=1):
            mutation_id = _slug(
                f"mutation-{lead_id}-{template.get('operator')}-{index}",
                max_len=96,
            )
            mutation = {
                "mutation_id": mutation_id,
                "lead_id": lead_id,
                "queue_rank": lead.get("queue_rank"),
                "lead_type": lead.get("lead_type"),
                "title": lead.get("title"),
                "target": lead.get("target"),
                "novelty_score": lead.get("novelty_score"),
                "playbook_id": playbook.get("playbook_id"),
                "operator": template.get("operator"),
                "variable": template.get("variable"),
                "command": template.get("command"),
                "implementation_hint": template.get("implementation_hint"),
                "expected_signal": template.get("expected_signal"),
                "guardrail": template.get("guardrail"),
                "promotion_gate": (
                    "Promote only after the mutation improves or explains the expected signal while every playbook guardrail remains clean."
                ),
                "isolation_rule": (
                    "Run as a one-variable mutation against the verified control; do not combine with another mutation until isolated evidence exists."
                ),
                "source_variant_ladder": [
                    item.get("variant")
                    for item in playbook.get("variant_ladder", []) or []
                    if item.get("variant")
                ],
            }
            lead_items.append(mutation)
            mutations.append(mutation)
        lead_mutations.append(
            {
                "lead_id": lead_id,
                "lead_type": lead.get("lead_type"),
                "title": lead.get("title"),
                "target": lead.get("target"),
                "mutation_count": len(lead_items),
                "mutations": lead_items,
            }
        )
    operator_counts = Counter(str(item.get("operator") or "unknown") for item in mutations)
    return {
        "lead_count": len(lead_mutations),
        "mutation_count": len(mutations),
        "operator_counts": dict(operator_counts),
        "max_mutations_per_lead": max_mutations_per_lead,
        "policy": (
            "Use mutation candidates to widen novelty search without overclaiming: each mutation changes one variable, keeps the playbook guardrails, and must pass isolated evidence before it can be combined or promoted."
        ),
        "lead_mutations": lead_mutations,
        "mutations": mutations,
    }


def _mutation_cost_units(mutation: dict[str, Any]) -> int:
    operator = str(mutation.get("operator") or "")
    lead_type = str(mutation.get("lead_type") or "")
    base_by_operator = {
        "profile_mode_escalation": 3,
        "shape_stress_sweep": 2,
        "nearby_motif_probe": 1,
        "recipient_shape_transfer": 3,
        "sham_transfer_control": 2,
        "recipient_guardrail_sweep": 2,
        "missing_primitive_isolation": 3,
        "primitive_order_swap": 2,
        "single_primitive_revert": 2,
        "gap_candidate_probe": 2,
        "taxonomy_only_recompute": 1,
        "sample_target_rotation": 2,
        "lane_a_only": 2,
        "lane_b_only": 2,
        "bridge_interaction_toggle": 3,
    }
    cost = base_by_operator.get(operator, 2)
    if lead_type in {"compound_primitive", "cross_lane_bridge"}:
        cost += 1
    if mutation.get("command") and "--profile deep_dive" in str(mutation.get("command")):
        cost += 1
    return cost


def _mutation_risk_flags(mutation: dict[str, Any]) -> list[str]:
    operator = str(mutation.get("operator") or "")
    lead_type = str(mutation.get("lead_type") or "")
    command = mutation.get("command")
    risks: list[str] = []
    if command is None:
        risks.append("manual_or_taxonomy_step")
    if operator in {"bridge_interaction_toggle", "single_primitive_revert", "primitive_order_swap"}:
        risks.append("interaction_interpretation")
    if lead_type == "source_transfer":
        risks.append("transfer_scope_drift")
    if lead_type == "coverage_gap" and operator != "taxonomy_only_recompute":
        risks.append("coverage_artifact_required")
    if command and "--profile deep_dive" in str(command):
        risks.append("profiler_runtime_cost")
    return risks


def _mutation_information_gain(mutation: dict[str, Any]) -> float:
    operator = str(mutation.get("operator") or "")
    base_by_operator = {
        "profile_mode_escalation": 7.0,
        "shape_stress_sweep": 6.0,
        "nearby_motif_probe": 5.0,
        "recipient_shape_transfer": 7.0,
        "sham_transfer_control": 6.5,
        "recipient_guardrail_sweep": 5.0,
        "missing_primitive_isolation": 7.0,
        "primitive_order_swap": 6.0,
        "single_primitive_revert": 6.5,
        "gap_candidate_probe": 6.0,
        "taxonomy_only_recompute": 5.5,
        "sample_target_rotation": 5.0,
        "lane_a_only": 6.0,
        "lane_b_only": 6.0,
        "bridge_interaction_toggle": 7.5,
    }
    novelty_component = min(float(mutation.get("novelty_score") or 0.0) / 10.0, 6.0)
    command_bonus = 1.0 if mutation.get("command") else 0.0
    return round(base_by_operator.get(operator, 5.0) + novelty_component + command_bonus, 3)


def _mutation_required_evidence(mutation: dict[str, Any]) -> list[str]:
    evidence = [
        "verified control remains unchanged for this mutation",
        "mutation artifact directory is isolated from the lead validation queue",
        "only the declared mutation variable changes",
        "expected signal and guardrail are recorded together",
    ]
    operator = str(mutation.get("operator") or "")
    if operator == "taxonomy_only_recompute":
        evidence.append("coverage state is recomputed without new candidate artifacts")
    if operator in {"lane_a_only", "lane_b_only", "bridge_interaction_toggle"}:
        evidence.append("bridge lane metrics are captured separately before interpreting interaction")
    if operator in {"sham_transfer_control", "recipient_shape_transfer"}:
        evidence.append("source and recipient workload contracts are recorded separately")
    return evidence


def _mutation_budget_card(mutation: dict[str, Any]) -> dict[str, Any]:
    cost_units = _mutation_cost_units(mutation)
    risk_flags = _mutation_risk_flags(mutation)
    information_gain = _mutation_information_gain(mutation)
    expected_value = round(information_gain - cost_units * 1.1 - len(risk_flags) * 1.3, 3)
    return {
        "mutation_id": mutation.get("mutation_id"),
        "lead_id": mutation.get("lead_id"),
        "queue_rank": mutation.get("queue_rank"),
        "lead_type": mutation.get("lead_type"),
        "title": mutation.get("title"),
        "target": mutation.get("target"),
        "playbook_id": mutation.get("playbook_id"),
        "operator": mutation.get("operator"),
        "variable": mutation.get("variable"),
        "command": mutation.get("command"),
        "implementation_hint": mutation.get("implementation_hint"),
        "expected_signal": mutation.get("expected_signal"),
        "guardrail": mutation.get("guardrail"),
        "promotion_gate": mutation.get("promotion_gate"),
        "isolation_rule": mutation.get("isolation_rule"),
        "source_variant_ladder": list(mutation.get("source_variant_ladder", []) or []),
        "information_gain_score": information_gain,
        "expected_value_score": expected_value,
        "cost_units": cost_units,
        "risk_flags": risk_flags,
        "required_evidence": _mutation_required_evidence(mutation),
    }


def _build_novelty_mutation_budget_plan(
    novelty_mutation_plan: dict[str, Any],
    novelty_budget_plan: dict[str, Any],
    *,
    budget_slots: int = DEFAULT_PORTFOLIO_BUDGET,
    max_cost_units: int | None = None,
) -> dict[str, Any]:
    """Select a small, diverse mutation batch from the expanded mutation plan."""

    selected_lead_ids = {
        str(item.get("lead_id"))
        for item in novelty_budget_plan.get("selected", []) or []
        if item.get("lead_id")
    }
    cards = [
        _mutation_budget_card(mutation)
        for mutation in novelty_mutation_plan.get("mutations", []) or []
        if mutation.get("mutation_id")
    ]
    for card in cards:
        card["lead_selection_state"] = (
            "selected_lead" if str(card.get("lead_id") or "") in selected_lead_ids else "backlog_lead"
        )
    cards.sort(
        key=lambda row: (
            0 if row.get("lead_selection_state") == "selected_lead" else 1,
            -float(row.get("expected_value_score") or 0.0),
            int(row.get("cost_units") or 0),
            int(row.get("queue_rank") or 9999),
            str(row.get("operator") or ""),
            str(row.get("mutation_id") or ""),
        )
    )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_leads: set[str] = set()
    selected_operators: set[str] = set()
    total_cost = 0
    cost_cap = max_cost_units if max_cost_units is not None else budget_slots * 4

    def add(card: dict[str, Any], *, selection_phase: str, allow_repeat: bool = False) -> None:
        nonlocal total_cost
        mutation_id = str(card.get("mutation_id") or "")
        lead_id = str(card.get("lead_id") or "")
        operator = str(card.get("operator") or "")
        if not mutation_id or mutation_id in selected_ids or len(selected) >= budget_slots:
            return
        if not allow_repeat and (lead_id in selected_leads or operator in selected_operators):
            return
        cost = int(card.get("cost_units") or 0)
        if total_cost + cost > cost_cap and selected:
            return
        selected.append(card)
        selected_ids.add(mutation_id)
        if lead_id:
            selected_leads.add(lead_id)
        if operator:
            selected_operators.add(operator)
        total_cost += cost
        card["selection_phase"] = selection_phase
        card["selection_reason"] = (
            "Selected as a high-information one-variable mutation with bounded cost and isolated evidence requirements."
        )

    selected_cards = [card for card in cards if card.get("lead_selection_state") == "selected_lead"]
    backlog_cards = [card for card in cards if card.get("lead_selection_state") != "selected_lead"]
    for card in selected_cards:
        add(card, selection_phase="selected_lead_operator_diversity")
    for card in selected_cards:
        add(card, selection_phase="selected_lead_expected_value_fill", allow_repeat=True)
    for card in backlog_cards:
        add(card, selection_phase="backlog_discovery_probe", allow_repeat=True)

    def defer(card: dict[str, Any]) -> dict[str, Any]:
        deferred = dict(card)
        lead_id = str(card.get("lead_id") or "")
        operator = str(card.get("operator") or "")
        cost = int(card.get("cost_units") or 0)
        if card.get("lead_selection_state") != "selected_lead":
            reason = "lead_not_in_selected_budget"
            unlock = "Promote the parent lead or run this mutation only as an isolated backup probe."
        elif lead_id in selected_leads and operator in selected_operators:
            reason = "lead_and_operator_already_represented"
            unlock = "Run the selected mutation evidence before spending another slot on this lead/operator."
        elif total_cost + cost > cost_cap:
            reason = "mutation_cost_cap_exceeded"
            unlock = "Increase max_cost_units or choose a lower-cost mutation."
        elif len(selected) >= budget_slots:
            reason = "mutation_slots_exhausted"
            unlock = "Complete or reject a selected mutation before adding this one."
        else:
            reason = "lower_information_gain"
            unlock = "Rerank after validation feedback changes cost, risk, or expected signal."
        deferred["deferral_reason"] = reason
        deferred["next_unlock"] = unlock
        return deferred

    backlog = [
        defer(card) for card in cards if str(card.get("mutation_id") or "") not in selected_ids
    ]
    return {
        "budget_slots": budget_slots,
        "max_cost_units": cost_cap,
        "mutation_count": len(cards),
        "selected_count": len(selected),
        "backlog_count": len(backlog),
        "selected_cost_units": total_cost,
        "selected_lead_count": len(selected_leads),
        "selected_operator_count": len(selected_operators),
        "selection_phase_counts": dict(
            Counter(str(card.get("selection_phase") or "unknown") for card in selected)
        ),
        "deferral_reason_counts": dict(
            Counter(str(card.get("deferral_reason") or "unknown") for card in backlog)
        ),
        "next_mutation": selected[0] if selected else None,
        "selection_policy": (
            "Select only a small mutation batch: prefer selected parent leads, preserve operator and lead diversity first, then fill by risk-adjusted information gain. Deferred mutations keep explicit unlock conditions."
        ),
        "selected": selected,
        "backlog": backlog,
    }


def _novelty_cost_units(lead: dict[str, Any], playbook: dict[str, Any] | None = None) -> int:
    lead_type = str(lead.get("lead_type") or "")
    base_by_type = {
        "frontier_probe": 2,
        "source_transfer": 3,
        "coverage_gap": 3,
        "compound_primitive": 4,
        "cross_lane_bridge": 5,
    }
    cost = base_by_type.get(lead_type, 3)
    tools = set((playbook or {}).get("profiler_tools", []) or [])
    if "hta" in tools:
        cost += 2
    if "ncu" in tools:
        cost += 1
    if "zymtrace" in tools:
        cost += 1
    return cost


def _novelty_risk_flags(lead: dict[str, Any], playbook: dict[str, Any] | None = None) -> list[str]:
    lead_type = str(lead.get("lead_type") or "")
    text = " ".join(
        str(value) for value in [lead.get("title"), lead.get("target"), lead.get("why")] if value
    ).lower()
    tools = set((playbook or {}).get("profiler_tools", []) or [])
    risks: list[str] = []
    if lead_type in {"compound_primitive", "cross_lane_bridge"}:
        risks.append("multi_variable_isolation")
    if lead_type == "coverage_gap":
        risks.append("source_change_required")
    if "hta" in tools or any(
        term in text for term in ("distributed", "fabric", "allreduce", "nccl")
    ):
        risks.append("distributed_environment")
    if "zymtrace" in tools:
        risks.append("zymtrace_injection_required")
    if not lead.get("command"):
        risks.append("missing_command")
    return risks


def _novelty_risk_mitigation_steps(risk_flags: Iterable[str]) -> list[str]:
    mitigations = {
        "multi_variable_isolation": "Run each declared variable as an isolated variant before combining them.",
        "source_change_required": "Attach the candidate diff or explicit not-applicable note before review.",
        "distributed_environment": "Capture GPU count, NCCL/fabric environment, and topology before comparing timings.",
        "zymtrace_injection_required": "Verify CUDA_INJECTION64_PATH or ZYMTRACE_CUDA_INJECTION64_PATH before profiling.",
        "missing_command": "Create an explicit reproducible command before spending a validation slot.",
    }
    return [mitigations[flag] for flag in risk_flags if flag in mitigations]


def _novelty_risk_required_evidence(risk_flags: Iterable[str]) -> list[str]:
    required = {
        "multi_variable_isolation": "isolated-variable evidence exists for every variable in the combined novelty claim",
        "source_change_required": "source diff, generated artifact, or explicit disproval note is attached",
        "distributed_environment": "distributed topology and NCCL/fabric environment are captured with the run artifacts",
        "zymtrace_injection_required": "Zymtrace launch manifest or explicit not-applicable note is present",
        "missing_command": "reproducible candidate command is recorded before manual approval",
    }
    return [required[flag] for flag in risk_flags if flag in required]


def _novelty_null_hypothesis(lead_type: str) -> str:
    if lead_type == "frontier_probe":
        return "The target is not a new optimization surface; it either lacks runnable evidence or duplicates an existing benchmark lane."
    if lead_type == "source_transfer":
        return (
            "The source-mined pattern does not improve the recipient beyond the verified control."
        )
    if lead_type == "compound_primitive":
        return "The combined primitive stack adds no value beyond the best isolated primitive."
    if lead_type == "coverage_gap":
        return (
            "The coverage gap is taxonomy noise, not a missing benchmarkable optimization surface."
        )
    if lead_type == "cross_lane_bridge":
        return "The bridge is not real; one lane explains the result or the combined variant regresses another lane."
    return "The candidate does not improve a verified control under the declared workload contract."


def _novelty_claim_boundary(lead_type: str) -> str:
    if lead_type == "frontier_probe":
        return "Claim only first clean evidence and the observed bottleneck, not a speedup."
    if lead_type == "source_transfer":
        return "Claim transferability only for the measured recipient shape and source pattern."
    if lead_type == "compound_primitive":
        return "Claim a compound effect only if the combined stack beats every isolated primitive."
    if lead_type == "coverage_gap":
        return "Claim coverage closure only after the radar recomputes the gap state."
    if lead_type == "cross_lane_bridge":
        return "Claim a bridge only if both lane metrics are present and guardrails stay clean."
    return "Claim only the measured candidate/control delta under the captured workload contract."


def _novelty_falsification_checks(lead: dict[str, Any]) -> list[str]:
    lead_type = str(lead.get("lead_type") or "")
    checks = [
        "control run fails correctness, artifact, or reproducibility checks",
        "candidate result is within measurement noise of the verified control",
        "candidate changes undeclared variables or workload shape",
    ]
    if lead_type == "frontier_probe":
        checks.extend(
            [
                "equivalent optimized evidence already exists for the target",
                "deep profile cannot identify a dominant bottleneck after first evidence",
            ]
        )
    elif lead_type == "source_transfer":
        checks.extend(
            [
                "source pattern fails to reproduce on the source target control",
                "recipient improves equally without applying the source-mined pattern",
            ]
        )
    elif lead_type == "compound_primitive":
        checks.extend(
            [
                "combined stack does not beat the best isolated primitive",
                "one primitive explains the full observed delta without the compound stack",
            ]
        )
    elif lead_type == "coverage_gap":
        checks.extend(
            [
                "gap closes only by renaming taxonomy terms without new evidence",
                "candidate artifact does not introduce the missing signal or primitive",
            ]
        )
    elif lead_type == "cross_lane_bridge":
        checks.extend(
            [
                "only one bridged lane has a captured metric",
                "combined bridge improves one lane while regressing the other lane",
            ]
        )
    return checks


def _build_novelty_falsification_plan(
    novelty_budget_plan: dict[str, Any],
    novelty_playbooks: dict[str, Any],
) -> dict[str, Any]:
    """Declare how selected novelty leads can be disproven before claims are allowed."""

    playbook_by_lead = {
        str(playbook.get("lead_id")): playbook
        for playbook in novelty_playbooks.get("playbooks", []) or []
        if playbook.get("lead_id")
    }
    lead_checks: list[dict[str, Any]] = []
    for lead in novelty_budget_plan.get("selected", []) or []:
        lead_id = str(lead.get("lead_id") or "")
        if not lead_id:
            continue
        checks = _novelty_falsification_checks(lead)
        playbook = playbook_by_lead.get(lead_id, {})
        lead_checks.append(
            {
                "lead_id": lead_id,
                "lead_type": lead.get("lead_type"),
                "title": lead.get("title"),
                "target": lead.get("target"),
                "playbook_id": lead.get("playbook_id") or playbook.get("playbook_id"),
                "null_hypothesis": _novelty_null_hypothesis(str(lead.get("lead_type") or "")),
                "falsification_checks": checks,
                "required_counterevidence": [
                    "resolved falsification checklist with artifact references",
                    "control and candidate logs use the same workload contract",
                    "manual reviewer records why the null hypothesis was rejected or retained",
                ],
                "claim_boundary": _novelty_claim_boundary(str(lead.get("lead_type") or "")),
                "stop_on_failure": (
                    "Stop validation, keep claim_allowed false, and either narrow the hypothesis or move the lead back to backlog."
                ),
            }
        )
    return {
        "lead_count": len(lead_checks),
        "check_count": sum(len(item["falsification_checks"]) for item in lead_checks),
        "policy": (
            "Every selected novelty lead declares how it can be disproven. Manual review must resolve these checks before any novelty claim is allowed."
        ),
        "lead_checks": lead_checks,
    }


def _novelty_ablation_controls(lead: dict[str, Any]) -> list[dict[str, Any]]:
    lead_id = str(lead.get("lead_id") or "novelty")
    lead_type = str(lead.get("lead_type") or "")
    target = str(lead.get("target") or "")
    control_command = _profile_command(target, "minimal") if target else lead.get("command")
    candidate_command = lead.get("command") or control_command
    controls = [
        {
            "control_id": _slug(f"{lead_id}-same-workload-replay", max_len=96),
            "control_type": "same_workload_replay",
            "command": control_command,
            "purpose": "Re-run the verified control under the exact candidate workload contract.",
            "rejects_claim_if": "control/candidate workload shape, seed, dtype, or verification contract differs",
        }
    ]
    if lead_type == "frontier_probe":
        controls.extend(
            [
                {
                    "control_id": _slug(f"{lead_id}-duplicate-lane-check", max_len=96),
                    "control_type": "duplicate_lane_check",
                    "command": None,
                    "purpose": "Search existing measured lanes for equivalent evidence before claiming novelty.",
                    "rejects_claim_if": "equivalent benchmark evidence already covers the target or motif",
                },
                {
                    "control_id": _slug(f"{lead_id}-minimal-before-deep", max_len=96),
                    "control_type": "minimal_before_deep_profile",
                    "command": control_command,
                    "purpose": "Keep first evidence separate from deep profiling interpretation.",
                    "rejects_claim_if": "deep profile is interpreted without a passing minimal smoke",
                },
            ]
        )
    elif lead_type == "source_transfer":
        controls.extend(
            [
                {
                    "control_id": _slug(f"{lead_id}-source-pattern-replay", max_len=96),
                    "control_type": "source_pattern_replay",
                    "command": control_command,
                    "purpose": "Confirm the source pattern still reproduces before transferring it.",
                    "rejects_claim_if": "source-side replay is stale or fails correctness",
                },
                {
                    "control_id": _slug(f"{lead_id}-sham-transfer", max_len=96),
                    "control_type": "sham_transfer",
                    "command": candidate_command,
                    "purpose": "Run the recipient with all non-pattern setup changes but without the transferred pattern.",
                    "rejects_claim_if": "recipient improves equally without the transferred pattern",
                },
            ]
        )
    elif lead_type == "compound_primitive":
        controls.extend(
            [
                {
                    "control_id": _slug(f"{lead_id}-best-isolated-primitive", max_len=96),
                    "control_type": "best_isolated_primitive",
                    "command": candidate_command,
                    "purpose": "Measure each primitive alone before measuring the combined stack.",
                    "rejects_claim_if": "the combined stack does not beat the best isolated primitive",
                },
                {
                    "control_id": _slug(f"{lead_id}-single-primitive-revert", max_len=96),
                    "control_type": "single_primitive_revert",
                    "command": candidate_command,
                    "purpose": "Turn off one primitive at a time to verify the claimed interaction.",
                    "rejects_claim_if": "removing one primitive leaves the full measured delta intact",
                },
            ]
        )
    elif lead_type == "coverage_gap":
        controls.extend(
            [
                {
                    "control_id": _slug(f"{lead_id}-taxonomy-only-recompute", max_len=96),
                    "control_type": "taxonomy_only_recompute",
                    "command": None,
                    "purpose": "Recompute coverage without new evidence to detect taxonomy-only closure.",
                    "rejects_claim_if": "the gap closes by term renaming without new benchmark artifacts",
                },
                {
                    "control_id": _slug(f"{lead_id}-artifact-absent-check", max_len=96),
                    "control_type": "artifact_absent_check",
                    "command": None,
                    "purpose": "Verify the missing signal or primitive is actually present in the candidate artifact.",
                    "rejects_claim_if": "candidate artifacts do not contain the missing signal or primitive evidence",
                },
            ]
        )
    elif lead_type == "cross_lane_bridge":
        controls.extend(
            [
                {
                    "control_id": _slug(f"{lead_id}-lane-a-only", max_len=96),
                    "control_type": "lane_a_only",
                    "command": candidate_command,
                    "purpose": "Measure the first bridge lane without the second lane enabled.",
                    "rejects_claim_if": "lane A alone explains the full observed delta",
                },
                {
                    "control_id": _slug(f"{lead_id}-lane-b-only", max_len=96),
                    "control_type": "lane_b_only",
                    "command": candidate_command,
                    "purpose": "Measure the second bridge lane without the first lane enabled.",
                    "rejects_claim_if": "lane B alone explains the full observed delta",
                },
            ]
        )
    return controls


def _build_novelty_ablation_plan(novelty_budget_plan: dict[str, Any]) -> dict[str, Any]:
    """Build negative controls that isolate selected novelty claims."""

    lead_controls: list[dict[str, Any]] = []
    for lead in novelty_budget_plan.get("selected", []) or []:
        lead_id = str(lead.get("lead_id") or "")
        if not lead_id:
            continue
        controls = _novelty_ablation_controls(lead)
        lead_controls.append(
            {
                "lead_id": lead_id,
                "lead_type": lead.get("lead_type"),
                "title": lead.get("title"),
                "target": lead.get("target"),
                "control_count": len(controls),
                "controls": controls,
                "required_evidence": [
                    "ablation control summary with pass/fail result for each control",
                    "explicit not-applicable note for any skipped ablation control",
                    "claim text narrowed when an ablation explains the observed delta",
                ],
            }
        )
    return {
        "lead_count": len(lead_controls),
        "control_count": sum(int(item.get("control_count") or 0) for item in lead_controls),
        "policy": (
            "Ablation controls isolate novelty claims. Manual review must show that simpler controls do not explain the observed effect."
        ),
        "lead_controls": lead_controls,
    }


def _novelty_reproducibility_profile(lead: dict[str, Any]) -> dict[str, Any]:
    lead_type = str(lead.get("lead_type") or "")
    risk_flags = set(lead.get("risk_flags", []) or [])
    base_repeats = 3
    if lead_type in {"compound_primitive", "cross_lane_bridge"}:
        base_repeats = 5
    if "distributed_environment" in risk_flags:
        base_repeats += 2
    if "zymtrace_injection_required" in risk_flags:
        base_repeats += 1
    profile_mode = "minimal_then_deep"
    if lead_type == "coverage_gap":
        profile_mode = "minimal_then_recompute"
    elif lead_type == "frontier_probe":
        profile_mode = "minimal_first_evidence"
    variance_threshold = 0.08 if lead_type in {"compound_primitive", "cross_lane_bridge"} else 0.12
    if "distributed_environment" in risk_flags:
        variance_threshold = 0.15
    return {
        "repeat_count": base_repeats,
        "profile_mode": profile_mode,
        "stability_metrics": [
            "median_wall_time_ms",
            "interquartile_range_pct",
            "output_correctness_pass_rate",
        ],
        "variance_threshold_pct": round(variance_threshold * 100.0, 2),
        "replication_gate": (
            "Pass only if control and candidate repeat distributions stay within the variance threshold and every repeat passes output verification."
        ),
    }


def _build_novelty_reproducibility_plan(novelty_budget_plan: dict[str, Any]) -> dict[str, Any]:
    """Define repeatability and stability gates for selected novelty leads."""

    lead_profiles: list[dict[str, Any]] = []
    for lead in novelty_budget_plan.get("selected", []) or []:
        lead_id = str(lead.get("lead_id") or "")
        if not lead_id:
            continue
        profile = _novelty_reproducibility_profile(lead)
        lead_profiles.append(
            {
                "lead_id": lead_id,
                "lead_type": lead.get("lead_type"),
                "title": lead.get("title"),
                "target": lead.get("target"),
                **profile,
                "required_evidence": [
                    "repeat-run manifest with control and candidate artifact paths",
                    "variance summary for the declared stability metrics",
                    "all repeats pass output verification or the claim remains blocked",
                ],
            }
        )
    return {
        "lead_count": len(lead_profiles),
        "repeat_count_total": sum(int(item.get("repeat_count") or 0) for item in lead_profiles),
        "policy": (
            "Novelty claims require repeated control/candidate evidence. Single-run wins stay advisory until variance and correctness are stable."
        ),
        "lead_profiles": lead_profiles,
    }


def _instrumentation_preflight_checks(tools: Iterable[str]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = [
        {
            "check": "benchmark_command_resolves",
            "command": "python -m cli.aisp bench list --json",
            "required": True,
            "evidence": "benchmark target catalog or explicit target-not-found note",
        }
    ]
    for tool in sorted({str(item) for item in tools if item}):
        if tool == "nsys":
            checks.append(
                {
                    "check": "nsys_available",
                    "command": "command -v nsys && nsys --version",
                    "required": True,
                    "evidence": "nsys path and version recorded in profiler preflight manifest",
                }
            )
        elif tool == "ncu":
            checks.append(
                {
                    "check": "ncu_available",
                    "command": "command -v ncu && ncu --version",
                    "required": False,
                    "evidence": "ncu path/version or not-applicable reason recorded",
                }
            )
        elif tool == "hta":
            checks.append(
                {
                    "check": "hta_trace_ready",
                    "command": "python - <<'PY'\nimport importlib.util; raise SystemExit(0 if importlib.util.find_spec('hta') else 1)\nPY",
                    "required": False,
                    "evidence": "HTA import/version or distributed-trace not-applicable reason recorded",
                }
            )
        elif tool == "zymtrace":
            checks.append(
                {
                    "check": "zymtrace_cuda_injection_ready",
                    "command": 'test -n "${CUDA_INJECTION64_PATH:-${ZYMTRACE_CUDA_INJECTION64_PATH:-}}"',
                    "required": True,
                    "evidence": "CUDA injection library path and zymtrace launch manifest recorded",
                }
            )
        else:
            checks.append(
                {
                    "check": f"{tool}_available",
                    "command": f"command -v {shlex.quote(tool)}",
                    "required": False,
                    "evidence": f"{tool} availability or explicit skip reason recorded",
                }
            )
    return checks


def _instrumentation_required_evidence(tools: Iterable[str]) -> list[str]:
    tool_set = {str(item) for item in tools if item}
    evidence = [
        "profiler preflight manifest with tool paths, versions, and skipped-tool reasons",
        "profile artifact manifest lists every required profiler artifact or not-applicable note",
    ]
    if "zymtrace" in tool_set:
        evidence.append(
            "Zymtrace launch manifest records CUDA_INJECTION64_PATH or ZYMTRACE_CUDA_INJECTION64_PATH"
        )
    if "hta" in tool_set:
        evidence.append("distributed trace metadata records GPU count, ranks, and topology context")
    if "ncu" in tool_set:
        evidence.append("NCU kernel summary or explicit unsupported-kernel note is attached")
    return evidence


def _build_novelty_instrumentation_plan(
    novelty_budget_plan: dict[str, Any],
    novelty_playbooks: dict[str, Any],
) -> dict[str, Any]:
    """Define profiler preflight and artifact requirements for selected novelty leads."""

    playbook_by_lead = {
        str(playbook.get("lead_id")): playbook
        for playbook in novelty_playbooks.get("playbooks", []) or []
        if playbook.get("lead_id")
    }
    lead_profiles: list[dict[str, Any]] = []
    tool_counts: Counter[str] = Counter()
    for lead in novelty_budget_plan.get("selected", []) or []:
        lead_id = str(lead.get("lead_id") or "")
        if not lead_id:
            continue
        playbook = playbook_by_lead.get(lead_id, {})
        tools = sorted({str(tool) for tool in playbook.get("profiler_tools", []) or ["nsys"]})
        tool_counts.update(tools)
        fallback_tools = [tool for tool in ("nsys", "ncu") if tool in tools]
        if not fallback_tools:
            fallback_tools = ["nsys"]
        launch_environment = []
        if "zymtrace" in tools:
            launch_environment.extend(["CUDA_INJECTION64_PATH", "ZYMTRACE_CUDA_INJECTION64_PATH"])
        if "hta" in tools:
            launch_environment.extend(["NCCL_DEBUG", "CUDA_VISIBLE_DEVICES", "WORLD_SIZE"])
        lead_profiles.append(
            {
                "lead_id": lead_id,
                "lead_type": lead.get("lead_type"),
                "title": lead.get("title"),
                "target": lead.get("target"),
                "playbook_id": playbook.get("playbook_id") or lead.get("playbook_id"),
                "required_profiler_tools": tools,
                "fallback_profiler_tools": fallback_tools,
                "launch_environment": sorted(set(launch_environment)),
                "preflight_checks": _instrumentation_preflight_checks(tools),
                "required_evidence": _instrumentation_required_evidence(tools),
            }
        )
    return {
        "lead_count": len(lead_profiles),
        "tool_counts": dict(tool_counts),
        "policy": (
            "Profiler instrumentation must be proven before interpreting novelty evidence. Missing optional tools require explicit skip notes; missing required launch injection keeps claims blocked."
        ),
        "lead_profiles": lead_profiles,
    }


def _artifact_stage_contract(
    *,
    stage: str,
    job_id: str,
    runbook_files: Iterable[str],
    promotion_files: Iterable[str] = (),
    evidence: Iterable[str] = (),
) -> dict[str, Any]:
    runbook_file_list = list(dict.fromkeys(str(item) for item in runbook_files if item))
    promotion_file_list = list(dict.fromkeys(str(item) for item in promotion_files if item))
    return {
        "stage": stage,
        "job_id": job_id,
        "artifact_root": f"${{AISP_NOVELTY_QUEUE_ROOT}}/{job_id}",
        "runbook_files": runbook_file_list,
        "promotion_files": promotion_file_list,
        "required_files": list(dict.fromkeys([*runbook_file_list, *promotion_file_list])),
        "required_evidence": [str(item) for item in evidence if item],
    }


def _build_novelty_artifact_contract_plan(
    novelty_budget_plan: dict[str, Any],
    novelty_playbooks: dict[str, Any],
    novelty_reproducibility_plan: dict[str, Any],
    novelty_instrumentation_plan: dict[str, Any],
) -> dict[str, Any]:
    """Define the expected artifact package for each selected novelty lead."""

    playbook_by_lead = {
        str(playbook.get("lead_id")): playbook
        for playbook in novelty_playbooks.get("playbooks", []) or []
        if playbook.get("lead_id")
    }
    reproducibility_by_lead = {
        str(item.get("lead_id")): item
        for item in novelty_reproducibility_plan.get("lead_profiles", []) or []
        if item.get("lead_id")
    }
    instrumentation_by_lead = {
        str(item.get("lead_id")): item
        for item in novelty_instrumentation_plan.get("lead_profiles", []) or []
        if item.get("lead_id")
    }
    lead_contracts: list[dict[str, Any]] = []
    required_file_count = 0
    for lead in novelty_budget_plan.get("selected", []) or []:
        lead_id = str(lead.get("lead_id") or "")
        if not lead_id:
            continue
        lead_type = str(lead.get("lead_type") or "")
        ids = _novelty_validation_job_ids(lead_id, lead_type)
        playbook = playbook_by_lead.get(lead_id, {})
        reproducibility = reproducibility_by_lead.get(lead_id, {})
        instrumentation = instrumentation_by_lead.get(lead_id, {})
        profiler_tools = {
            str(tool)
            for tool in (
                instrumentation.get("required_profiler_tools")
                or playbook.get("profiler_tools")
                or []
            )
            if tool
        }

        command_runbook_files = [
            "job.json",
            "artifact_contract.json",
            "command.txt",
            "stdout.log",
            "stderr.log",
            "DONE",
        ]
        profile_files = [
            *command_runbook_files,
            "profiler_preflight_manifest.json",
            "profile_artifacts.json",
        ]
        if "zymtrace" in profiler_tools:
            profile_files.append("zymtrace_launch_manifest.json")
        if "ncu" in profiler_tools:
            profile_files.append("ncu_kernel_summary.json")
        if "hta" in profiler_tools:
            profile_files.append("distributed_trace_metadata.json")
        candidate_promotion_files = [
            "output_verification_record.json",
            "control_candidate_comparison.json",
        ]
        if int(reproducibility.get("repeat_count") or 0) > 1:
            candidate_promotion_files.append("repeat_run_manifest.json")

        stage_contracts = [
            _artifact_stage_contract(
                stage="control",
                job_id=ids["control"],
                runbook_files=command_runbook_files,
                promotion_files=["output_verification_record.json"],
                evidence=[
                    "control command, logs, DONE marker, and output verification are present"
                ],
            ),
            _artifact_stage_contract(
                stage=ids["stage_name"],
                job_id=ids["candidate"],
                runbook_files=command_runbook_files,
                promotion_files=candidate_promotion_files,
                evidence=[
                    "candidate command and logs are tied to the verified control job",
                    "candidate comparison records the same workload contract",
                ],
            ),
            _artifact_stage_contract(
                stage="deep_profile",
                job_id=ids["profile"],
                runbook_files=command_runbook_files,
                promotion_files=profile_files[len(command_runbook_files) :],
                evidence=[
                    "profiler preflight and profile artifact manifests resolve every required tool"
                ],
            ),
            _artifact_stage_contract(
                stage="manual_review",
                job_id=ids["review"],
                runbook_files=[
                    "job.json",
                    "artifact_contract.json",
                    "claim_packet.json",
                    "promotion_review.md",
                    "MANUAL_REVIEW_REQUIRED",
                ],
                promotion_files=[
                    "artifact_contract_manifest.json",
                    "claim_packet.md",
                    "claim_decision.json",
                    "APPROVED",
                ],
                evidence=[
                    "review packet cites every stage artifact before APPROVED is created"
                ],
            ),
        ]
        required_file_count += sum(
            len(stage_contract.get("required_files", []) or [])
            for stage_contract in stage_contracts
        )
        lead_contracts.append(
            {
                "contract_id": _slug(f"artifact-contract-{lead_id}", max_len=96),
                "lead_id": lead_id,
                "lead_type": lead.get("lead_type"),
                "title": lead.get("title"),
                "target": lead.get("target"),
                "package_manifest": "artifact_contract_manifest.json",
                "stage_contracts": stage_contracts,
                "required_evidence": [
                    "artifact contract manifest links each job_id to its artifact_root",
                    "all required files are present or have explicit not-applicable notes",
                    "manual review cites control, candidate, profile, and repeat evidence before approval",
                ],
            }
        )
    return {
        "lead_count": len(lead_contracts),
        "required_file_count": required_file_count,
        "policy": (
            "Every selected novelty lead gets a machine-readable artifact contract. Promotion remains blocked until the review packet links each control, candidate, profile, and approval artifact."
        ),
        "lead_contracts": lead_contracts,
    }


def _novelty_disallowed_claims(lead_type: str) -> list[str]:
    common = [
        "do not claim a speedup unless the candidate beats a verified same-workload control",
        "do not generalize beyond the measured target, shape, precision, hardware, and profiler setup",
        "do not cite profiler interpretation without linking the raw profile artifact or explicit not-applicable note",
    ]
    if lead_type == "frontier_probe":
        return common + [
            "do not turn first clean evidence into an optimization claim",
            "do not claim novelty if equivalent measured evidence already exists in another lane",
        ]
    if lead_type == "source_transfer":
        return common + [
            "do not claim a reusable transfer pattern until the source and recipient controls both pass",
            "do not credit the source pattern if a sham transfer improves equally",
        ]
    if lead_type == "compound_primitive":
        return common + [
            "do not claim a compound effect until the combined stack beats every isolated primitive",
            "do not claim interaction if disabling one primitive leaves the full delta intact",
        ]
    if lead_type == "coverage_gap":
        return common + [
            "do not claim the coverage gap is closed without a recomputed radar result",
            "do not treat taxonomy renaming as new benchmark evidence",
        ]
    if lead_type == "cross_lane_bridge":
        return common + [
            "do not claim a bridge unless both lane metrics are captured in the same evidence bundle",
            "do not hide a regression in one lane behind an improvement in the other",
        ]
    return common


def _claim_packet_sections(
    *,
    falsification: dict[str, Any],
    ablation: dict[str, Any],
    reproducibility: dict[str, Any],
    instrumentation: dict[str, Any],
    artifact_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    repeat_count = reproducibility.get("repeat_count")
    profiler_tools = instrumentation.get("required_profiler_tools") or []
    artifact_contract_id = artifact_contract.get("contract_id")
    return [
        {
            "section": "workload_contract",
            "must_include": "target, command, profile mode, input shape, dtype, seed policy, and hardware context when available",
            "evidence_sources": ["control job.json", "candidate job.json"],
        },
        {
            "section": "measured_delta",
            "must_include": "control metric, candidate metric, guardrail metrics, and the exact comparison method",
            "evidence_sources": ["control stdout/stderr", "candidate stdout/stderr"],
        },
        {
            "section": "falsification_resolution",
            "must_include": "null hypothesis decision and resolved falsification checklist",
            "evidence_sources": list(falsification.get("falsification_checks", []) or [])[:4],
        },
        {
            "section": "ablation_summary",
            "must_include": "negative-control result for each required ablation or a not-applicable note",
            "evidence_sources": [
                str(control.get("control_id"))
                for control in ablation.get("controls", []) or []
                if isinstance(control, dict) and control.get("control_id")
            ][:4],
        },
        {
            "section": "reproducibility_summary",
            "must_include": f"repeat count {repeat_count} with variance and correctness status",
            "evidence_sources": ["repeat_run_manifest.json", "variance summary"],
        },
        {
            "section": "profiler_evidence",
            "must_include": "required profiler tools, preflight status, dominant bottleneck, and raw artifact links",
            "evidence_sources": [str(tool) for tool in profiler_tools],
        },
        {
            "section": "artifact_packet",
            "must_include": "artifact contract manifest plus control, candidate, profile, and review paths",
            "evidence_sources": [str(artifact_contract_id or "artifact_contract_manifest.json")],
        },
    ]


def _build_novelty_claim_packet_plan(
    novelty_budget_plan: dict[str, Any],
    novelty_falsification_plan: dict[str, Any],
    novelty_ablation_plan: dict[str, Any],
    novelty_reproducibility_plan: dict[str, Any],
    novelty_instrumentation_plan: dict[str, Any],
    novelty_artifact_contract_plan: dict[str, Any],
) -> dict[str, Any]:
    """Define bounded claim packets for selected novelty leads."""

    falsification_by_lead = {
        str(item.get("lead_id")): item
        for item in novelty_falsification_plan.get("lead_checks", []) or []
        if item.get("lead_id")
    }
    ablation_by_lead = {
        str(item.get("lead_id")): item
        for item in novelty_ablation_plan.get("lead_controls", []) or []
        if item.get("lead_id")
    }
    reproducibility_by_lead = {
        str(item.get("lead_id")): item
        for item in novelty_reproducibility_plan.get("lead_profiles", []) or []
        if item.get("lead_id")
    }
    instrumentation_by_lead = {
        str(item.get("lead_id")): item
        for item in novelty_instrumentation_plan.get("lead_profiles", []) or []
        if item.get("lead_id")
    }
    artifact_contract_by_lead = {
        str(item.get("lead_id")): item
        for item in novelty_artifact_contract_plan.get("lead_contracts", []) or []
        if item.get("lead_id")
    }
    lead_packets: list[dict[str, Any]] = []
    for lead in novelty_budget_plan.get("selected", []) or []:
        lead_id = str(lead.get("lead_id") or "")
        if not lead_id:
            continue
        lead_type = str(lead.get("lead_type") or "")
        job_ids = _novelty_validation_job_ids(lead_id, lead_type)
        falsification = falsification_by_lead.get(lead_id, {})
        ablation = ablation_by_lead.get(lead_id, {})
        reproducibility = reproducibility_by_lead.get(lead_id, {})
        instrumentation = instrumentation_by_lead.get(lead_id, {})
        artifact_contract = artifact_contract_by_lead.get(lead_id, {})
        sections = _claim_packet_sections(
            falsification=falsification,
            ablation=ablation,
            reproducibility=reproducibility,
            instrumentation=instrumentation,
            artifact_contract=artifact_contract,
        )
        disallowed_claims = _novelty_disallowed_claims(lead_type)
        lead_packets.append(
            {
                "packet_id": _slug(f"claim-packet-{lead_id}", max_len=96),
                "lead_id": lead_id,
                "lead_type": lead.get("lead_type"),
                "title": lead.get("title"),
                "target": lead.get("target"),
                "review_job_id": job_ids["review"],
                "packet_path": f"${{AISP_NOVELTY_QUEUE_ROOT}}/{job_ids['review']}/claim_packet.md",
                "allowed_claim_scope": falsification.get("claim_boundary")
                or _novelty_claim_boundary(lead_type),
                "required_sections": sections,
                "disallowed_claims": disallowed_claims,
                "required_evidence": [
                    "claim packet links every required section to concrete artifacts",
                    "allowed claim text stays within the declared claim boundary",
                    "reviewer confirms each disallowed claim is absent before approval",
                ],
                "approval_rule": (
                    "APPROVED may be created only after claim_packet.md exists, all required sections cite artifacts, and disallowed claims are explicitly rejected."
                ),
            }
        )
    return {
        "lead_count": len(lead_packets),
        "required_section_count": sum(
            len(packet.get("required_sections", []) or []) for packet in lead_packets
        ),
        "disallowed_claim_count": sum(
            len(packet.get("disallowed_claims", []) or []) for packet in lead_packets
        ),
        "policy": (
            "Novelty results need bounded claim packets. Reviewers must connect each claim to evidence and remove overclaims before approval."
        ),
        "lead_packets": lead_packets,
    }


def _novelty_readiness_score(lead: dict[str, Any], playbook: dict[str, Any] | None = None) -> float:
    score = 0.0
    if lead.get("target"):
        score += 2.0
    if lead.get("command"):
        score += 2.0
    if lead.get("evidence_gate"):
        score += 1.0
    if playbook and playbook.get("variant_ladder"):
        score += 1.0
    if playbook and playbook.get("primary_metrics") and playbook.get("guardrail_metrics"):
        score += 1.0
    return score


def _budget_card(
    lead: dict[str, Any], playbook_by_lead: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    playbook = playbook_by_lead.get(str(lead.get("lead_id") or ""))
    cost_units = _novelty_cost_units(lead, playbook)
    risk_flags = _novelty_risk_flags(lead, playbook)
    readiness = _novelty_readiness_score(lead, playbook)
    novelty_score = float(lead.get("novelty_score") or 0.0)
    expected_value = round(
        novelty_score + readiness * 4.0 - cost_units * 1.8 - len(risk_flags) * 2.5, 3
    )
    return {
        "lead_id": lead.get("lead_id"),
        "queue_rank": lead.get("queue_rank"),
        "lead_type": lead.get("lead_type"),
        "title": lead.get("title"),
        "target": lead.get("target"),
        "novelty_score": lead.get("novelty_score"),
        "expected_value_score": expected_value,
        "readiness_score": readiness,
        "cost_units": cost_units,
        "risk_flags": risk_flags,
        "risk_mitigation_steps": _novelty_risk_mitigation_steps(risk_flags),
        "playbook_id": playbook.get("playbook_id") if playbook else None,
        "command": lead.get("command"),
        "evidence_gate": lead.get("evidence_gate"),
    }


def _build_novelty_budget_plan(
    novelty_queue: dict[str, Any],
    novelty_playbooks: dict[str, Any],
    *,
    budget_slots: int = DEFAULT_PORTFOLIO_BUDGET,
    max_cost_units: int | None = None,
) -> dict[str, Any]:
    """Pick a risk-adjusted, target-diverse novelty batch."""

    playbook_by_lead = {
        str(playbook.get("lead_id")): playbook
        for playbook in novelty_playbooks.get("playbooks", []) or []
        if playbook.get("lead_id")
    }
    cards = [
        _budget_card(lead, playbook_by_lead)
        for lead in novelty_queue.get("leads", []) or []
        if lead.get("lead_id")
    ]
    cards.sort(
        key=lambda row: (
            -float(row.get("expected_value_score") or 0.0),
            int(row.get("cost_units") or 0),
            int(row.get("queue_rank") or 9999),
            str(row.get("lead_id") or ""),
        )
    )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_targets: set[str] = set()
    selected_types: set[str] = set()
    total_cost = 0
    cost_cap = max_cost_units if max_cost_units is not None else budget_slots * 7

    def add(
        card: dict[str, Any],
        *,
        selection_phase: str,
        allow_target_repeat: bool = False,
    ) -> None:
        nonlocal total_cost
        lead_id = str(card.get("lead_id") or "")
        target = str(card.get("target") or "")
        if not lead_id or lead_id in selected_ids or len(selected) >= budget_slots:
            return
        if target and target in selected_targets and not allow_target_repeat:
            return
        cost = int(card.get("cost_units") or 0)
        if total_cost + cost > cost_cap and selected:
            return
        selected.append(card)
        selected_ids.add(lead_id)
        if target:
            selected_targets.add(target)
        selected_types.add(str(card.get("lead_type") or "unknown"))
        total_cost += cost
        card["selection_phase"] = selection_phase
        card["selection_reason"] = (
            "Selected by risk-adjusted expected value while preserving novelty portfolio diversity."
        )

    for lead_type in sorted(
        {str(card.get("lead_type") or "unknown") for card in cards},
        key=lambda value: (
            -max(
                float(card.get("expected_value_score") or 0.0)
                for card in cards
                if str(card.get("lead_type") or "unknown") == value
            ),
            value,
        ),
    ):
        best = next(
            (
                card
                for card in cards
                if str(card.get("lead_type") or "unknown") == lead_type
                and str(card.get("target") or "") not in selected_targets
            ),
            None,
        )
        allow_target_repeat = False
        if best is None:
            best = next(
                (
                    card
                    for card in cards
                    if str(card.get("lead_type") or "unknown") == lead_type
                ),
                None,
            )
            allow_target_repeat = True
        if best:
            add(
                best,
                selection_phase=(
                    "lead_type_diversity"
                    if not allow_target_repeat
                    else "lead_type_diversity_target_repeat"
                ),
                allow_target_repeat=allow_target_repeat,
            )
    for card in cards:
        add(card, selection_phase="expected_value_fill")
    for card in cards:
        add(card, selection_phase="target_repeat_fill", allow_target_repeat=True)

    def defer(card: dict[str, Any]) -> dict[str, Any]:
        deferred = dict(card)
        target = str(card.get("target") or "")
        cost = int(card.get("cost_units") or 0)
        if target and target in selected_targets:
            reason = "target_already_selected"
            unlock = (
                "Run or reject the selected lead for this target before spending another slot here."
            )
        elif total_cost + cost > cost_cap:
            reason = "cost_cap_exceeded"
            unlock = "Increase max_cost_units or choose a lower-cost validation slice."
        elif len(selected) >= budget_slots:
            reason = "budget_slots_exhausted"
            unlock = (
                "Increase budget_slots or complete one selected lead before promoting backlog work."
            )
        else:
            reason = "lower_expected_value"
            unlock = "Re-rank after new evidence changes expected value, readiness, cost, or risk."
        deferred["deferral_reason"] = reason
        deferred["next_unlock"] = unlock
        return deferred

    backlog = [defer(card) for card in cards if str(card.get("lead_id") or "") not in selected_ids]
    deferral_counts = Counter(str(card.get("deferral_reason") or "unknown") for card in backlog)
    return {
        "budget_slots": budget_slots,
        "max_cost_units": cost_cap,
        "selected_count": len(selected),
        "backlog_count": len(backlog),
        "selected_cost_units": total_cost,
        "selected_type_count": len(selected_types),
        "selected_target_count": len(selected_targets),
        "deferral_reason_counts": dict(deferral_counts),
        "selection_policy": (
            "Select a risk-adjusted novelty batch by expected value, preserve lead-type and target diversity, and keep deferred leads auditable with unlock conditions."
        ),
        "selected": selected,
        "backlog": backlog,
    }


def _decision_card(
    card: dict[str, Any],
    *,
    lane: str,
    reason: str,
    selected_ids: set[str],
) -> dict[str, Any]:
    lead_id = str(card.get("lead_id") or "")
    return {
        "lead_id": lead_id,
        "lead_type": card.get("lead_type"),
        "title": card.get("title"),
        "target": card.get("target"),
        "decision_lane": lane,
        "decision_reason": reason,
        "selection_state": "selected" if lead_id in selected_ids else "backlog",
        "expected_value_score": card.get("expected_value_score"),
        "novelty_score": card.get("novelty_score"),
        "readiness_score": card.get("readiness_score"),
        "cost_units": card.get("cost_units"),
        "risk_flags": list(card.get("risk_flags", []) or []),
        "next_unlock": card.get("next_unlock"),
    }


def _pareto_frontier_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for card in cards:
        value = float(card.get("expected_value_score") or 0.0)
        readiness = float(card.get("readiness_score") or 0.0)
        cost = int(card.get("cost_units") or 0)
        risk_count = len(card.get("risk_flags", []) or [])
        dominated = False
        for other in cards:
            if other is card:
                continue
            other_value = float(other.get("expected_value_score") or 0.0)
            other_readiness = float(other.get("readiness_score") or 0.0)
            other_cost = int(other.get("cost_units") or 0)
            other_risk_count = len(other.get("risk_flags", []) or [])
            weakly_better = (
                other_value >= value
                and other_readiness >= readiness
                and other_cost <= cost
                and other_risk_count <= risk_count
            )
            strictly_better = (
                other_value > value
                or other_readiness > readiness
                or other_cost < cost
                or other_risk_count < risk_count
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(card)
    frontier.sort(
        key=lambda row: (
            -float(row.get("expected_value_score") or 0.0),
            int(row.get("cost_units") or 0),
            len(row.get("risk_flags", []) or []),
            str(row.get("lead_id") or ""),
        )
    )
    return frontier


def _build_novelty_decision_frontier(
    novelty_budget_plan: dict[str, Any], *, lane_limit: int = 5
) -> dict[str, Any]:
    """Expose portfolio decision lanes beyond the selected budget batch."""

    selected = list(novelty_budget_plan.get("selected", []) or [])
    backlog = list(novelty_budget_plan.get("backlog", []) or [])
    cards = [*selected, *backlog]
    selected_ids = {str(card.get("lead_id") or "") for card in selected if card.get("lead_id")}

    def sort_key(card: dict[str, Any]) -> tuple[float, int, int, str]:
        return (
            -float(card.get("expected_value_score") or 0.0),
            int(card.get("cost_units") or 0),
            len(card.get("risk_flags", []) or []),
            str(card.get("lead_id") or ""),
        )

    quick_proofs = sorted(
        (
            card
            for card in cards
            if int(card.get("cost_units") or 0) <= 3 and len(card.get("risk_flags", []) or []) <= 1
        ),
        key=sort_key,
    )[:lane_limit]
    high_upside = sorted(
        cards,
        key=lambda card: (
            -float(card.get("novelty_score") or 0.0),
            -float(card.get("expected_value_score") or 0.0),
            int(card.get("cost_units") or 0),
            str(card.get("lead_id") or ""),
        ),
    )[:lane_limit]
    de_risk_first = sorted(
        (card for card in cards if card.get("risk_flags")),
        key=lambda card: (
            -len(card.get("risk_flags", []) or []),
            -float(card.get("novelty_score") or 0.0),
            int(card.get("cost_units") or 0),
            str(card.get("lead_id") or ""),
        ),
    )[:lane_limit]
    deferred_unlocks = sorted(backlog, key=sort_key)[:lane_limit]
    pareto = _pareto_frontier_cards(cards)[:lane_limit]

    lanes = [
        {
            "lane": "quick_proofs",
            "policy": "Low-cost, low-risk leads that can create first evidence quickly.",
            "leads": [
                _decision_card(
                    card,
                    lane="quick_proofs",
                    reason="Low cost and at most one risk flag.",
                    selected_ids=selected_ids,
                )
                for card in quick_proofs
            ],
        },
        {
            "lane": "high_upside",
            "policy": "Highest novelty-score leads, even when they require more setup.",
            "leads": [
                _decision_card(
                    card,
                    lane="high_upside",
                    reason="Highest novelty score among available leads.",
                    selected_ids=selected_ids,
                )
                for card in high_upside
            ],
        },
        {
            "lane": "de_risk_first",
            "policy": "Promising leads whose risks should be resolved before expensive validation.",
            "leads": [
                _decision_card(
                    card,
                    lane="de_risk_first",
                    reason="Risk flags should be resolved before treating this as claim evidence.",
                    selected_ids=selected_ids,
                )
                for card in de_risk_first
            ],
        },
        {
            "lane": "pareto_frontier",
            "policy": "Non-dominated leads by expected value, readiness, cost, and risk count.",
            "leads": [
                _decision_card(
                    card,
                    lane="pareto_frontier",
                    reason="No other available lead is simultaneously higher value, more ready, cheaper, and lower risk.",
                    selected_ids=selected_ids,
                )
                for card in pareto
            ],
        },
        {
            "lane": "deferred_unlocks",
            "policy": "Best deferred leads and the condition that would make them runnable later.",
            "leads": [
                _decision_card(
                    card,
                    lane="deferred_unlocks",
                    reason=str(card.get("deferral_reason") or "deferred"),
                    selected_ids=selected_ids,
                )
                for card in deferred_unlocks
            ],
        },
    ]
    return {
        "lane_count": len(lanes),
        "lead_count": len(cards),
        "selected_lead_count": len(selected),
        "backlog_lead_count": len(backlog),
        "policy": (
            "Use decision lanes to choose the next wave: quick proofs for momentum, high-upside for novelty, de-risk-first for infrastructure blockers, and deferred unlocks for backlog planning."
        ),
        "lanes": lanes,
    }


def _novelty_required_evidence(lead: dict[str, Any]) -> list[str]:
    lead_type = str(lead.get("lead_type") or "")
    base = [
        "verified minimal-profile control run",
        "isolated artifact directory for the novelty lead",
        "output correctness or numerical-tolerance evidence",
        "same workload contract for control and candidate",
    ]
    if lead_type == "frontier_probe":
        return [
            "runnable target exists in the benchmark catalog",
            "minimal-profile smoke succeeds with output verification",
            "deep-dive follow-up identifies a dominant bottleneck",
            "no optimization claim is made from first evidence alone",
        ]
    if lead_type == "coverage_gap":
        return base + [
            "gap-specific source term or primitive is introduced or explicitly disproven",
            "coverage state is recomputed after the candidate artifact is captured",
        ]
    if lead_type == "compound_primitive":
        return base + [
            "each primitive is isolated before the combined variant runs",
            "combined variant beats the best isolated primitive without guardrail regression",
        ]
    if lead_type == "cross_lane_bridge":
        return base + [
            "each bridged signal has a captured metric in the same artifact bundle",
            "bridge variant improves one primary metric without regressing the other lane",
        ]
    if lead_type == "source_transfer":
        return base + [
            "source and recipient targets both have clean control evidence",
            "transferred pattern improves the recipient before it is promoted as reusable",
        ]
    return base


def _build_novelty_validation_plan(
    novelty_queue: dict[str, Any],
    novelty_playbooks: dict[str, Any] | None = None,
    novelty_budget_plan: dict[str, Any] | None = None,
    novelty_falsification_plan: dict[str, Any] | None = None,
    novelty_ablation_plan: dict[str, Any] | None = None,
    novelty_reproducibility_plan: dict[str, Any] | None = None,
    novelty_instrumentation_plan: dict[str, Any] | None = None,
    novelty_artifact_contract_plan: dict[str, Any] | None = None,
    novelty_claim_packet_plan: dict[str, Any] | None = None,
    *,
    budget_slots: int = DEFAULT_PORTFOLIO_BUDGET,
) -> dict[str, Any]:
    """Build a dependency-aware validation plan for the top novelty leads."""

    leads_by_id = {
        str(lead.get("lead_id")): lead
        for lead in novelty_queue.get("leads", []) or []
        if lead.get("lead_id")
    }
    budget_selected_ids = [
        str(item.get("lead_id"))
        for item in (novelty_budget_plan or {}).get("selected", []) or []
        if item.get("lead_id")
    ]
    selected = [leads_by_id[lead_id] for lead_id in budget_selected_ids if lead_id in leads_by_id]
    if not selected:
        selected = list(novelty_queue.get("leads") or [])[:budget_slots]
    else:
        selected = selected[:budget_slots]
    playbook_by_lead = {
        str(playbook.get("lead_id")): playbook
        for playbook in (novelty_playbooks or {}).get("playbooks", []) or []
        if playbook.get("lead_id")
    }
    falsification_by_lead = {
        str(item.get("lead_id")): item
        for item in (novelty_falsification_plan or {}).get("lead_checks", []) or []
        if item.get("lead_id")
    }
    ablation_by_lead = {
        str(item.get("lead_id")): item
        for item in (novelty_ablation_plan or {}).get("lead_controls", []) or []
        if item.get("lead_id")
    }
    reproducibility_by_lead = {
        str(item.get("lead_id")): item
        for item in (novelty_reproducibility_plan or {}).get("lead_profiles", []) or []
        if item.get("lead_id")
    }
    instrumentation_by_lead = {
        str(item.get("lead_id")): item
        for item in (novelty_instrumentation_plan or {}).get("lead_profiles", []) or []
        if item.get("lead_id")
    }
    artifact_contract_by_lead = {
        str(item.get("lead_id")): item
        for item in (novelty_artifact_contract_plan or {}).get("lead_contracts", []) or []
        if item.get("lead_id")
    }
    claim_packet_by_lead = {
        str(item.get("lead_id")): item
        for item in (novelty_claim_packet_plan or {}).get("lead_packets", []) or []
        if item.get("lead_id")
    }
    jobs: list[dict[str, Any]] = []
    for lead in selected:
        lead_id = str(lead.get("lead_id") or _slug(str(lead.get("title") or "novelty")))
        playbook = playbook_by_lead.get(lead_id, {})
        playbook_id = playbook.get("playbook_id")
        risk_flags = _novelty_risk_flags(lead, playbook)
        risk_mitigation_steps = _novelty_risk_mitigation_steps(risk_flags)
        falsification = falsification_by_lead.get(
            lead_id,
            {
                "falsification_checks": _novelty_falsification_checks(lead),
                "required_counterevidence": [],
                "claim_boundary": _novelty_claim_boundary(str(lead.get("lead_type") or "")),
            },
        )
        ablation = ablation_by_lead.get(
            lead_id,
            {
                "controls": _novelty_ablation_controls(lead),
                "required_evidence": [],
            },
        )
        reproducibility = reproducibility_by_lead.get(
            lead_id,
            _novelty_reproducibility_profile(lead) | {"required_evidence": []},
        )
        instrumentation = instrumentation_by_lead.get(
            lead_id,
            {
                "required_profiler_tools": list(playbook.get("profiler_tools", []) or ["nsys"]),
                "fallback_profiler_tools": ["nsys"],
                "launch_environment": [],
                "preflight_checks": _instrumentation_preflight_checks(
                    playbook.get("profiler_tools", []) or ["nsys"]
                ),
                "required_evidence": _instrumentation_required_evidence(
                    playbook.get("profiler_tools", []) or ["nsys"]
                ),
            },
        )
        artifact_contract = artifact_contract_by_lead.get(lead_id, {})
        artifact_stage_by_stage = {
            str(item.get("stage")): item
            for item in artifact_contract.get("stage_contracts", []) or []
            if item.get("stage")
        }
        claim_packet = claim_packet_by_lead.get(lead_id, {})
        target = str(lead.get("target") or "")
        job_ids = _novelty_validation_job_ids(lead_id, str(lead.get("lead_type") or ""))
        stage_name = job_ids["stage_name"]
        control_id = job_ids["control"]
        candidate_id = job_ids["candidate"]
        profile_id = job_ids["profile"]
        review_id = job_ids["review"]
        control_command = _profile_command(target, "minimal") if target else lead.get("command")
        candidate_command = lead.get("command") or control_command
        jobs.append(
            {
                "id": control_id,
                "lead_id": lead_id,
                "lead_type": lead.get("lead_type"),
                "target": target,
                "stage": "control",
                "command": control_command,
                "depends_on": [],
                "experiment_playbook_id": playbook_id,
                "artifact_label": _slug(f"novelty-control-{target or lead_id}", max_len=48),
                "artifact_contract_id": artifact_contract.get("contract_id"),
                "artifact_contract": artifact_stage_by_stage.get("control"),
                "evidence_gate": "Control must pass before candidate novelty work starts.",
            }
        )
        jobs.append(
            {
                "id": candidate_id,
                "lead_id": lead_id,
                "lead_type": lead.get("lead_type"),
                "target": target,
                "stage": stage_name,
                "command": candidate_command,
                "depends_on": [control_id],
                "experiment_playbook_id": playbook_id,
                "experiment_variables": list(playbook.get("variables", []) or []),
                "primary_metrics": list(playbook.get("primary_metrics", []) or []),
                "guardrail_metrics": list(playbook.get("guardrail_metrics", []) or []),
                "risk_flags": risk_flags,
                "risk_mitigation_steps": risk_mitigation_steps,
                "falsification_checks": list(falsification.get("falsification_checks", []) or []),
                "ablation_controls": list(ablation.get("controls", []) or []),
                "repeat_count": reproducibility.get("repeat_count"),
                "stability_metrics": list(reproducibility.get("stability_metrics", []) or []),
                "variance_threshold_pct": reproducibility.get("variance_threshold_pct"),
                "required_profiler_tools": list(
                    instrumentation.get("required_profiler_tools", []) or []
                ),
                "instrumentation_preflight": list(
                    instrumentation.get("preflight_checks", []) or []
                ),
                "artifact_label": _slug(f"novelty-{stage_name}-{target or lead_id}", max_len=48),
                "artifact_contract_id": artifact_contract.get("contract_id"),
                "artifact_contract": artifact_stage_by_stage.get(stage_name),
                "evidence_gate": lead.get("evidence_gate"),
                "why": lead.get("why"),
                "related_targets": list(lead.get("related_targets", []) or []),
            }
        )
        jobs.append(
            {
                "id": profile_id,
                "lead_id": lead_id,
                "lead_type": lead.get("lead_type"),
                "target": target,
                "stage": "deep_profile",
                "command": _profile_command(target, "deep_dive") if target else None,
                "depends_on": [candidate_id],
                "experiment_playbook_id": playbook_id,
                "profiler_tools": list(
                    instrumentation.get("required_profiler_tools", [])
                    or playbook.get("profiler_tools", [])
                    or []
                ),
                "fallback_profiler_tools": list(
                    instrumentation.get("fallback_profiler_tools", []) or []
                ),
                "instrumentation_preflight": list(
                    instrumentation.get("preflight_checks", []) or []
                ),
                "artifact_label": _slug(f"novelty-profile-{target or lead_id}", max_len=48),
                "artifact_contract_id": artifact_contract.get("contract_id"),
                "artifact_contract": artifact_stage_by_stage.get("deep_profile"),
                "evidence_gate": "Deep profile must identify the dominant bottleneck or explain why profiling is not applicable.",
            }
        )
        jobs.append(
            {
                "id": review_id,
                "lead_id": lead_id,
                "lead_type": lead.get("lead_type"),
                "target": target,
                "stage": "manual_review",
                "command": None,
                "depends_on": [profile_id],
                "experiment_playbook_id": playbook_id,
                "stop_conditions": list(playbook.get("stop_conditions", []) or []),
                "artifact_label": _slug(f"novelty-review-{target or lead_id}", max_len=48),
                "risk_flags": risk_flags,
                "risk_mitigation_steps": risk_mitigation_steps,
                "falsification_checks": list(falsification.get("falsification_checks", []) or []),
                "claim_boundary": falsification.get("claim_boundary"),
                "ablation_controls": list(ablation.get("controls", []) or []),
                "reproducibility_gate": reproducibility.get("replication_gate"),
                "repeat_count": reproducibility.get("repeat_count"),
                "stability_metrics": list(reproducibility.get("stability_metrics", []) or []),
                "variance_threshold_pct": reproducibility.get("variance_threshold_pct"),
                "required_profiler_tools": list(
                    instrumentation.get("required_profiler_tools", []) or []
                ),
                "fallback_profiler_tools": list(
                    instrumentation.get("fallback_profiler_tools", []) or []
                ),
                "launch_environment": list(instrumentation.get("launch_environment", []) or []),
                "instrumentation_preflight": list(
                    instrumentation.get("preflight_checks", []) or []
                ),
                "artifact_contract_id": artifact_contract.get("contract_id"),
                "artifact_contract": artifact_contract,
                "claim_packet_id": claim_packet.get("packet_id"),
                "claim_packet": claim_packet,
                "required_evidence": _novelty_required_evidence(lead)
                + _novelty_risk_required_evidence(risk_flags)
                + list(falsification.get("required_counterevidence", []) or [])
                + list(ablation.get("required_evidence", []) or [])
                + list(reproducibility.get("required_evidence", []) or [])
                + list(instrumentation.get("required_evidence", []) or [])
                + list(artifact_contract.get("required_evidence", []) or [])
                + list(claim_packet.get("required_evidence", []) or []),
                "evidence_gate": "Manual review keeps claim_allowed false unless all required evidence is present.",
            }
        )

    _annotate_queue_jobs(jobs)
    dispatch_groups = _build_dispatch_groups(jobs)
    return {
        "selected_lead_count": len(selected),
        "job_count": len(jobs),
        "budget_slots": budget_slots,
        "ready_job_ids": dispatch_groups[0]["job_ids"] if dispatch_groups else [],
        "critical_path_groups": len(dispatch_groups),
        "policy": (
            "Validate novelty leads with control, candidate, deep-profile, and manual-review jobs. "
            "This plan is advisory and does not allow claims until the review evidence is complete."
        ),
        "selected_leads": [
            {
                "queue_rank": lead.get("queue_rank"),
                "lead_id": lead.get("lead_id"),
                "lead_type": lead.get("lead_type"),
                "title": lead.get("title"),
                "target": lead.get("target"),
                "novelty_score": lead.get("novelty_score"),
            }
            for lead in selected
        ],
        "dispatch_groups": dispatch_groups,
        "jobs": jobs,
    }


def _build_frontier_discovery_map(opportunities: list[dict[str, Any]]) -> dict[str, Any]:
    frontier_rows = [
        row for row in opportunities if row.get("opportunity_type") == "novel_frontier_probe"
    ]
    lanes_by_name: dict[str, list[dict[str, Any]]] = {}
    for row in frontier_rows:
        signals = [str(signal) for signal in row.get("frontier_signals", [])]
        lane_name = str(
            row.get("frontier_motif") or (signals[0] if signals else "general_frontier")
        )
        lanes_by_name.setdefault(lane_name, []).append(row)

    lanes: list[dict[str, Any]] = []
    for lane_name, rows in lanes_by_name.items():
        rows.sort(key=lambda row: (-float(row.get("score") or 0.0), str(row.get("target"))))
        preferred_signal = LANE_PREFERRED_SIGNALS.get(lane_name, lane_name)
        top_targets = [
            {
                "rank": row.get("rank"),
                "target": row.get("target"),
                "score": row.get("score"),
                "signals": row.get("frontier_signals", []),
                "command": row.get("next_command"),
                "blueprint_ids": [
                    blueprint["id"]
                    for blueprint in _frontier_blueprints_for_row(
                        row, limit=2, preferred_signal=preferred_signal
                    )
                ],
            }
            for row in rows[:5]
        ]
        signal_counts = Counter(
            signal
            for row in rows
            for signal in row.get("frontier_signals", [])
            if isinstance(signal, str)
        )
        lanes.append(
            {
                "lane": lane_name,
                "candidate_count": len(rows),
                "top_score": rows[0].get("score") if rows else 0.0,
                "top_targets": top_targets,
                "dominant_signals": dict(signal_counts.most_common(4)),
                "experiment_blueprints": (
                    _frontier_blueprints_for_row(rows[0], preferred_signal=preferred_signal)
                    if rows
                    else []
                ),
                "first_command": top_targets[0]["command"] if top_targets else None,
                "selection_rule": "Run the top verified smoke first, then compare its deep-dive bottleneck against the next lane before staying within one motif.",
            }
        )

    lanes.sort(
        key=lambda lane: (
            -float(lane.get("top_score") or 0.0),
            -int(lane.get("candidate_count") or 0),
            str(lane.get("lane")),
        )
    )

    diversity_queue: list[dict[str, Any]] = []
    lane_rows = [lanes_by_name[str(lane["lane"])] for lane in lanes]
    max_lane_length = max((len(rows) for rows in lane_rows), default=0)
    for index in range(max_lane_length):
        for lane, rows in zip(lanes, lane_rows, strict=True):
            if index >= len(rows) or len(diversity_queue) >= 20:
                continue
            row = rows[index]
            diversity_queue.append(
                {
                    "lane": lane["lane"],
                    "rank": row.get("rank"),
                    "target": row.get("target"),
                    "score": row.get("score"),
                    "command": row.get("next_command"),
                    "blueprint_ids": [
                        blueprint["id"]
                        for blueprint in _frontier_blueprints_for_row(
                            row,
                            limit=2,
                            preferred_signal=LANE_PREFERRED_SIGNALS.get(
                                str(lane["lane"]), str(lane["lane"])
                            ),
                        )
                    ],
                }
            )
        if len(diversity_queue) >= 20:
            break

    return {
        "frontier_candidate_count": len(frontier_rows),
        "lane_count": len(lanes),
        "selection_policy": "Prefer the diversity_queue for broad first-evidence discovery; prefer lane top_targets when exploiting one motif.",
        "lanes": lanes,
        "diversity_queue": diversity_queue,
    }


def _cluster_for_target(hypotheses: dict[str, Any], target: str) -> dict[str, Any] | None:
    for cluster in hypotheses.get("clusters", []) or []:
        if target in set(cluster.get("support_targets") or []):
            return cluster
    return None


def _primary_metric(row: dict[str, Any]) -> str:
    opportunity_type = str(row.get("opportunity_type") or "")
    if opportunity_type in {"restore_benchmark_evidence", "novel_frontier_probe"}:
        return "verified_status"
    if opportunity_type == "memory_pressure_probe" or str(row.get("optimization_goal")) == "memory":
        return "memory_savings_pct"
    return "median_wall_time_ms"


def _guardrail_metrics(row: dict[str, Any]) -> list[str]:
    guardrails = ["output_correctness", "status_succeeded", "artifact_paths_present"]
    if str(row.get("optimization_goal")) == "memory":
        guardrails.append("median_wall_time_ms")
    else:
        guardrails.append("memory_savings_pct")
    if str(row.get("opportunity_type")) == "novel_frontier_probe":
        guardrails.append("baseline_optimized_pair_exists")
    return guardrails


def _variant_name(text: str, index: int) -> str:
    slug = _slug(text, max_len=38)
    return f"v{index}_{slug}"


def _experiment_variants(
    row: dict[str, Any], cluster: dict[str, Any] | None
) -> list[dict[str, Any]]:
    target = str(row.get("target"))
    opportunity_type = str(row.get("opportunity_type") or "")
    variants: list[dict[str, Any]] = [
        {
            "name": "control_verified_current",
            "hypothesis": "The current baseline/optimized pair is stable enough to compare against.",
            "validation_command": _profile_command(target, "minimal"),
        }
    ]

    if opportunity_type == "novel_frontier_probe":
        variants.extend(
            [
                {
                    "name": "frontier_minimal_smoke",
                    "hypothesis": "The unmeasured target can produce first clean evidence.",
                    "validation_command": _profile_command(target, "minimal"),
                },
                {
                    "name": "frontier_deep_dive_followup",
                    "hypothesis": "Once first evidence is clean, profiler artifacts reveal the next optimization surface.",
                    "validation_command": _profile_command(target, "deep_dive"),
                    "run_after": "frontier_minimal_smoke",
                },
            ]
        )
        return variants

    if opportunity_type in {"restore_benchmark_evidence", "repair_regression"}:
        variants.append(
            {
                "name": "evidence_repair_minimal",
                "hypothesis": "A minimal verified rerun distinguishes stale artifact failure from a real regression.",
                "validation_command": _profile_command(target, "minimal"),
            }
        )
        return variants

    experiments = (
        list(cluster.get("transfer_experiments", [])[:2])
        if cluster
        else list(row.get("recommended_experiments", [])[:2])
    )
    for index, experiment in enumerate(experiments, start=1):
        variants.append(
            {
                "name": _variant_name(str(experiment), index),
                "hypothesis": str(experiment),
                "validation_command": _profile_command(target, "minimal"),
                "profile_command": _profile_command(target, "deep_dive"),
            }
        )
    return variants


def _build_experiment_matrix(
    opportunities: list[dict[str, Any]], hypotheses: dict[str, Any]
) -> dict[str, Any]:
    cards: list[dict[str, Any]] = []
    for row in opportunities[:10]:
        target = str(row.get("target") or "")
        if not target:
            continue
        cluster = _cluster_for_target(hypotheses, target)
        acceptance_gate = (
            cluster.get("acceptance_gate")
            if cluster
            else "Accept only verified runs that improve the primary metric without violating guardrail metrics."
        )
        experiment_blueprints = row.get("experiment_blueprints", [])
        if row.get("opportunity_type") == "novel_frontier_probe":
            preferred_signal = LANE_PREFERRED_SIGNALS.get(str(row.get("frontier_motif") or ""))
            experiment_blueprints = _frontier_blueprints_for_row(
                row, preferred_signal=preferred_signal
            )
        cards.append(
            {
                "target": target,
                "priority": row.get("priority"),
                "opportunity_type": row.get("opportunity_type"),
                "motif": cluster.get("motif") if cluster else "single_target",
                "primary_metric": _primary_metric(row),
                "guardrail_metrics": _guardrail_metrics(row),
                "control_command": _profile_command(target, "minimal"),
                "profile_command": row.get("next_command"),
                "variants": _experiment_variants(row, cluster),
                "experiment_blueprints": experiment_blueprints,
                "acceptance_gate": acceptance_gate,
                "benchmark_run": row.get("benchmark_run"),
            }
        )
    return {
        "card_count": len(cards),
        "cards": cards,
        "design_rules": [
            "Run the control command before each candidate variant.",
            "Change one variable per variant and keep artifact directories isolated.",
            "Promote an idea only when the primary metric improves and every guardrail metric remains clean.",
        ],
    }


def _portfolio_item(card: dict[str, Any]) -> dict[str, Any]:
    variants = card.get("variants") or []
    first_variant = variants[1] if len(variants) > 1 else variants[0] if variants else {}
    return {
        "target": card.get("target"),
        "priority": card.get("priority"),
        "opportunity_type": card.get("opportunity_type"),
        "motif": card.get("motif"),
        "primary_metric": card.get("primary_metric"),
        "first_variant": first_variant.get("name"),
        "experiment_blueprint_ids": [
            blueprint.get("id") for blueprint in card.get("experiment_blueprints", [])
        ],
        "first_experiment_blueprint": (
            (card.get("experiment_blueprints") or [None])[0]
            if card.get("experiment_blueprints")
            else None
        ),
        "command": first_variant.get("validation_command") or card.get("control_command"),
        "acceptance_gate": card.get("acceptance_gate"),
    }


def _build_portfolio_plan(
    matrix: dict[str, Any], *, budget_slots: int = DEFAULT_PORTFOLIO_BUDGET
) -> dict[str, Any]:
    cards = list(matrix.get("cards") or [])
    selected: list[dict[str, Any]] = []
    selected_targets: set[str] = set()
    seen_motifs: set[str] = set()

    def add(card: dict[str, Any]) -> None:
        target = str(card.get("target") or "")
        if not target or target in selected_targets or len(selected) >= budget_slots:
            return
        selected.append(card)
        selected_targets.add(target)
        seen_motifs.add(str(card.get("motif") or "single_target"))

    # Evidence repair makes later optimization claims possible.
    for card in cards:
        if card.get("opportunity_type") in {"restore_benchmark_evidence", "repair_regression"}:
            add(card)

    # Frontier exploration should cover different motifs before exploiting one lane.
    frontier_cards = [
        card for card in cards if card.get("opportunity_type") == "novel_frontier_probe"
    ]
    for card in frontier_cards:
        motif = str(card.get("motif") or "single_target")
        if motif not in seen_motifs:
            add(card)
    for card in frontier_cards:
        add(card)

    # Then maximize optimization motif diversity before filling remaining slots by rank order.
    for card in cards:
        motif = str(card.get("motif") or "single_target")
        if motif not in seen_motifs:
            add(card)
    for card in cards:
        add(card)

    selected_items = [_portfolio_item(card) for card in selected]
    backlog = [
        _portfolio_item(card)
        for card in cards
        if str(card.get("target") or "") not in selected_targets
    ]
    waves = [
        {
            "name": "evidence_and_frontier",
            "items": [
                item
                for item in selected_items
                if item.get("opportunity_type")
                in {"restore_benchmark_evidence", "repair_regression", "novel_frontier_probe"}
            ],
        },
        {
            "name": "diverse_optimization_batch",
            "items": [
                item
                for item in selected_items
                if item.get("opportunity_type")
                not in {"restore_benchmark_evidence", "repair_regression", "novel_frontier_probe"}
            ],
        },
    ]
    return {
        "budget_slots": budget_slots,
        "selected_count": len(selected_items),
        "selection_policy": "Prioritize evidence/frontier unlocks, then maximize motif diversity before filling by opportunity rank.",
        "selected": selected_items,
        "backlog": backlog,
        "waves": [wave for wave in waves if wave["items"]],
    }


def _promotion_state(item: dict[str, Any]) -> str:
    opportunity_type = str(item.get("opportunity_type") or "")
    if opportunity_type in {"restore_benchmark_evidence", "repair_regression"}:
        return "blocked_until_evidence_repaired"
    if opportunity_type == "novel_frontier_probe":
        return "blocked_until_first_evidence"
    return "blocked_until_variant_validated"


def _required_evidence(item: dict[str, Any]) -> list[str]:
    opportunity_type = str(item.get("opportunity_type") or "")
    base = [
        "verified minimal-profile control run",
        "isolated artifact directory for the candidate variant",
        "output correctness or numerical-tolerance evidence",
        "before/after comparison against the same workload contract",
    ]
    if opportunity_type == "novel_frontier_probe":
        return [
            "baseline/optimized benchmark pair is discoverable",
            "first verified minimal-profile run succeeds",
            "deep-dive profiler follow-up identifies the dominant bottleneck",
            "no claim is published until a candidate variant beats the verified control",
        ]
    if opportunity_type in {"restore_benchmark_evidence", "repair_regression"}:
        return [
            "failed or regressed target reruns cleanly with output verification",
            "failure log or regression cause is attached to the artifact bundle",
            "comparison proves the optimized path is no longer slower than baseline",
            "expectations are not refreshed until the rerun evidence is clean",
        ]
    if item.get("primary_metric") == "memory_savings_pct":
        base.append("memory metric improves without wall-time regression")
    else:
        base.append("median wall time improves without memory or correctness regression")
    if item.get("motif") != "single_target":
        base.append("at least one support target reproduces the motif-level result")
    return base


def _build_promotion_gates(portfolio: dict[str, Any], matrix: dict[str, Any]) -> dict[str, Any]:
    card_by_target = {str(card.get("target")): card for card in matrix.get("cards", [])}
    gates: list[dict[str, Any]] = []
    for item in portfolio.get("selected", []) or []:
        target = str(item.get("target") or "")
        card = card_by_target.get(target, {})
        state = _promotion_state(item)
        gates.append(
            {
                "target": target,
                "promotion_state": state,
                "claim_allowed": False,
                "primary_metric": item.get("primary_metric"),
                "guardrail_metrics": card.get("guardrail_metrics", []),
                "required_evidence": _required_evidence(item),
                "control_command": card.get("control_command"),
                "candidate_command": item.get("command"),
                "benchmark_run": card.get("benchmark_run"),
                "promotion_rule": "Promote only after all required evidence is present, guardrails pass, and the same workload contract is used for control and candidate.",
            }
        )
    return {
        "gate_count": len(gates),
        "claim_allowed_count": sum(1 for gate in gates if gate.get("claim_allowed")),
        "blocked_count": sum(1 for gate in gates if not gate.get("claim_allowed")),
        "global_rule": "Optimization claims remain blocked until selected portfolio gates have required evidence and clean guardrails.",
        "gates": gates,
    }


def _job_id(target: str, stage: str) -> str:
    return _slug(f"{target}-{stage}", max_len=72)


def _job_expected_artifacts(stage: str) -> list[str]:
    if stage == "promotion_review":
        return [
            "promotion gate checklist",
            "artifact bundle references",
            "claim decision note",
        ]
    artifacts = [
        "run metadata JSON",
        "stdout/stderr log",
        "benchmark summary JSON",
        "output verification record",
    ]
    if stage == "profile_followup":
        artifacts.append("deep-dive profiler trace or kernel summary")
    return artifacts


def _job_success_criteria(stage: str, required_evidence: Iterable[str] | None = None) -> list[str]:
    if stage == "control":
        return [
            "command exits successfully",
            "output verification passes",
            "artifact directory is isolated and readable",
        ]
    if stage == "candidate":
        return [
            "command exits successfully",
            "candidate uses the same workload contract as control",
            "primary and guardrail metrics are captured",
        ]
    if stage == "profile_followup":
        return [
            "deep-dive profile completes after first evidence is clean",
            "dominant bottleneck is recorded with artifact references",
        ]
    if stage == "promotion_review":
        evidence = [str(item) for item in required_evidence or []]
        return evidence + [
            "guardrail metrics pass",
            "claim decision remains blocked unless all evidence is present",
        ]
    return ["job completes and writes its declared artifacts"]


def _annotate_queue_jobs(jobs: list[dict[str, Any]]) -> None:
    for job in jobs:
        stage = str(job.get("stage") or "")
        job["expected_artifacts"] = _job_expected_artifacts(stage)
        job["success_criteria"] = _job_success_criteria(stage, job.get("required_evidence"))


def _build_dispatch_groups(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining_ids = {str(job.get("id")) for job in jobs if job.get("id")}
    known_ids = set(remaining_ids)
    completed_ids: set[str] = set()
    groups: list[dict[str, Any]] = []

    while remaining_ids:
        ready: list[dict[str, Any]] = []
        for job in jobs:
            job_id = str(job.get("id") or "")
            if job_id not in remaining_ids:
                continue
            dependencies = {str(dep) for dep in job.get("depends_on", [])}
            if dependencies <= completed_ids:
                ready.append(job)

        if not ready:
            missing_dependencies = sorted(
                {
                    str(dep)
                    for job in jobs
                    if str(job.get("id") or "") in remaining_ids
                    for dep in job.get("depends_on", [])
                    if str(dep) not in known_ids
                }
            )
            groups.append(
                {
                    "index": len(groups) + 1,
                    "job_ids": sorted(remaining_ids),
                    "stage_mix": {"blocked": len(remaining_ids)},
                    "target_count": len(
                        {
                            str(job.get("target") or "")
                            for job in jobs
                            if str(job.get("id") or "") in remaining_ids and job.get("target")
                        }
                    ),
                    "missing_dependencies": missing_dependencies,
                    "parallelism_note": "Queue contains an unresolved dependency; inspect before dispatch.",
                }
            )
            break

        ready_ids = [str(job.get("id")) for job in ready]
        groups.append(
            {
                "index": len(groups) + 1,
                "job_ids": ready_ids,
                "stage_mix": dict(Counter(str(job.get("stage") or "unknown") for job in ready)),
                "target_count": len(
                    {str(job.get("target") or "") for job in ready if job.get("target")}
                ),
                "parallelism_note": (
                    "Parallelize only when ready targets have isolated GPU visibility and artifact directories."
                    if len(ready) > 1
                    else "Single ready job; run after dependencies pass."
                ),
            }
        )
        completed_ids.update(ready_ids)
        remaining_ids.difference_update(ready_ids)

    return groups


def _build_run_queue(
    portfolio: dict[str, Any], matrix: dict[str, Any], promotion_gates: dict[str, Any]
) -> dict[str, Any]:
    card_by_target = {str(card.get("target")): card for card in matrix.get("cards", [])}
    gate_by_target = {str(gate.get("target")): gate for gate in promotion_gates.get("gates", [])}
    jobs: list[dict[str, Any]] = []
    for item in portfolio.get("selected", []) or []:
        target = str(item.get("target") or "")
        if not target:
            continue
        card = card_by_target.get(target, {})
        gate = gate_by_target.get(target, {})
        control_id = _job_id(target, "control")
        candidate_id = _job_id(target, str(item.get("first_variant") or "candidate"))
        jobs.append(
            {
                "id": control_id,
                "target": target,
                "stage": "control",
                "command": card.get("control_command"),
                "depends_on": [],
                "artifact_label": _slug(f"control-{target}", max_len=48),
                "promotion_gate": gate.get("promotion_state"),
            }
        )
        jobs.append(
            {
                "id": candidate_id,
                "target": target,
                "stage": "candidate",
                "variant": item.get("first_variant"),
                "command": item.get("command"),
                "depends_on": [control_id],
                "experiment_blueprint_ids": item.get("experiment_blueprint_ids", []),
                "experiment_blueprint": item.get("first_experiment_blueprint"),
                "artifact_label": _slug(
                    f"candidate-{target}-{item.get('first_variant')}", max_len=48
                ),
                "promotion_gate": gate.get("promotion_state"),
            }
        )
        if item.get("opportunity_type") == "novel_frontier_probe":
            profile_id = _job_id(target, "frontier-deep-dive")
            jobs.append(
                {
                    "id": profile_id,
                    "target": target,
                    "stage": "profile_followup",
                    "variant": "frontier_deep_dive_followup",
                    "command": _profile_command(target, "deep_dive"),
                    "depends_on": [candidate_id],
                    "artifact_label": _slug(f"profile-{target}", max_len=48),
                    "promotion_gate": gate.get("promotion_state"),
                }
            )
            candidate_id = profile_id
        review_id = _job_id(target, "promotion-review")
        jobs.append(
            {
                "id": review_id,
                "target": target,
                "stage": "promotion_review",
                "command": None,
                "depends_on": [candidate_id],
                "artifact_label": _slug(f"review-{target}", max_len=48),
                "promotion_gate": gate.get("promotion_state"),
                "required_evidence": gate.get("required_evidence", []),
            }
        )

    _annotate_queue_jobs(jobs)
    dispatch_groups = _build_dispatch_groups(jobs)
    return {
        "job_count": len(jobs),
        "max_parallelism": 1,
        "parallelism_policy": "Run serially by default; only parallelize jobs with disjoint targets and isolated GPU visibility.",
        "ready_job_ids": dispatch_groups[0]["job_ids"] if dispatch_groups else [],
        "critical_path_groups": len(dispatch_groups),
        "dispatch_groups": dispatch_groups,
        "jobs": jobs,
    }


def rank_opportunities(
    candidates: Iterable[BenchmarkCandidate],
    *,
    top_n: int | None = None,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
    target_speedup: float = DEFAULT_TARGET_SPEEDUP,
    min_memory_savings_pct: float = DEFAULT_MIN_MEMORY_SAVINGS_PCT,
    slow_baseline_ms: float = DEFAULT_SLOW_BASELINE_MS,
) -> dict[str, Any]:
    candidate_list = list(candidates)
    opportunities: list[OptimizationOpportunity] = []
    for candidate in candidate_list:
        if not candidate.target:
            continue
        opportunity_type = _classify(
            candidate,
            min_speedup=min_speedup,
            target_speedup=target_speedup,
            min_memory_savings_pct=min_memory_savings_pct,
        )
        score = _score(
            candidate,
            opportunity_type,
            min_speedup=min_speedup,
            target_speedup=target_speedup,
            slow_baseline_ms=slow_baseline_ms,
        )
        frontier_breakdown = _frontier_score_breakdown(candidate)
        opportunities.append(
            OptimizationOpportunity(
                rank=0,
                target=candidate.target,
                priority=_priority(score),
                opportunity_type=opportunity_type,
                score=score,
                status=candidate.status,
                optimization_goal=candidate.optimization_goal,
                baseline_time_ms=candidate.baseline_time_ms,
                optimized_time_ms=candidate.optimized_time_ms,
                best_speedup=candidate.best_speedup,
                memory_savings_pct=candidate.memory_savings_pct,
                frontier_motif=_frontier_motif(candidate),
                frontier_signals=[str(item["signal"]) for item in frontier_breakdown],
                frontier_score_breakdown=frontier_breakdown,
                source_terms=_candidate_source_terms(candidate),
                source_delta_terms=_candidate_source_delta_terms(candidate),
                source_files=_candidate_source_files(candidate),
                catalog_source=_candidate_catalog_source(candidate),
                optimization_primitives=_candidate_optimization_primitives(candidate),
                experiment_blueprints=_frontier_blueprints_for_row(
                    {
                        "target": candidate.target,
                        "frontier_signals": [str(item["signal"]) for item in frontier_breakdown],
                    }
                )
                if opportunity_type == "novel_frontier_probe"
                else [],
                evidence=_evidence(candidate),
                rationale=_rationale(candidate, opportunity_type),
                recommended_experiments=_recommended_experiments(candidate, opportunity_type),
                next_command=_next_command(candidate.target, opportunity_type),
                benchmark_run=_benchmark_run_payload(candidate, opportunity_type),
            )
        )

    opportunities.sort(key=lambda row: (-row.score, row.target))
    all_ranked = [
        asdict(opportunity) | {"rank": index + 1} for index, opportunity in enumerate(opportunities)
    ]
    ranked = all_ranked[:top_n] if top_n is not None and top_n > 0 else all_ranked

    frontier_map = _build_frontier_discovery_map(all_ranked)
    hypotheses = _build_innovation_hypotheses(ranked)
    transfer_map = _build_source_transfer_map(all_ranked)
    compound_hypotheses = _build_compound_primitive_hypotheses(all_ranked)
    primitive_pair_synthesis_plan = _build_primitive_pair_synthesis_plan(all_ranked)
    coverage_gap_map = _build_coverage_gap_map(all_ranked)
    cross_lane_bridge_map = _build_cross_lane_bridge_map(all_ranked)
    novelty_queue = _build_novelty_queue(
        frontier_map=frontier_map,
        transfer_map=transfer_map,
        compound_hypotheses=compound_hypotheses,
        coverage_gap_map=coverage_gap_map,
        cross_lane_bridge_map=cross_lane_bridge_map,
    )
    novelty_experiment_playbooks = _build_novelty_experiment_playbooks(novelty_queue)
    novelty_mutation_plan = _build_novelty_mutation_plan(
        novelty_queue, novelty_experiment_playbooks
    )
    novelty_budget_plan = _build_novelty_budget_plan(novelty_queue, novelty_experiment_playbooks)
    novelty_mutation_budget_plan = _build_novelty_mutation_budget_plan(
        novelty_mutation_plan, novelty_budget_plan
    )
    novelty_decision_frontier = _build_novelty_decision_frontier(novelty_budget_plan)
    novelty_falsification_plan = _build_novelty_falsification_plan(
        novelty_budget_plan, novelty_experiment_playbooks
    )
    novelty_ablation_plan = _build_novelty_ablation_plan(novelty_budget_plan)
    novelty_reproducibility_plan = _build_novelty_reproducibility_plan(novelty_budget_plan)
    novelty_instrumentation_plan = _build_novelty_instrumentation_plan(
        novelty_budget_plan, novelty_experiment_playbooks
    )
    novelty_artifact_contract_plan = _build_novelty_artifact_contract_plan(
        novelty_budget_plan,
        novelty_experiment_playbooks,
        novelty_reproducibility_plan,
        novelty_instrumentation_plan,
    )
    novelty_claim_packet_plan = _build_novelty_claim_packet_plan(
        novelty_budget_plan,
        novelty_falsification_plan,
        novelty_ablation_plan,
        novelty_reproducibility_plan,
        novelty_instrumentation_plan,
        novelty_artifact_contract_plan,
    )
    novelty_validation_plan = _build_novelty_validation_plan(
        novelty_queue,
        novelty_experiment_playbooks,
        novelty_budget_plan,
        novelty_falsification_plan,
        novelty_ablation_plan,
        novelty_reproducibility_plan,
        novelty_instrumentation_plan,
        novelty_artifact_contract_plan,
        novelty_claim_packet_plan,
    )
    matrix = _build_experiment_matrix(ranked, hypotheses)
    portfolio = _build_portfolio_plan(matrix)
    promotion_gates = _build_promotion_gates(portfolio, matrix)
    return {
        "parameters": {
            "min_speedup": min_speedup,
            "target_speedup": target_speedup,
            "min_memory_savings_pct": min_memory_savings_pct,
            "slow_baseline_ms": slow_baseline_ms,
            "top_n": top_n,
        },
        "summary": {
            "total_candidates": len(candidate_list),
            "frontier_candidates": sum(
                1 for candidate in candidate_list if candidate.status == "frontier"
            ),
            "returned": len(ranked),
            "by_priority": dict(Counter(row["priority"] for row in ranked)),
            "by_opportunity_type": dict(Counter(row["opportunity_type"] for row in ranked)),
            "frontier_lane_count": frontier_map["lane_count"],
        },
        "opportunities": ranked,
        "execution_plan": _build_execution_plan(ranked),
        "frontier_discovery_map": frontier_map,
        "innovation_hypotheses": hypotheses,
        "source_transfer_map": transfer_map,
        "compound_primitive_hypotheses": compound_hypotheses,
        "novelty_primitive_pair_synthesis_plan": primitive_pair_synthesis_plan,
        "coverage_gap_map": coverage_gap_map,
        "cross_lane_bridge_map": cross_lane_bridge_map,
        "novelty_queue": novelty_queue,
        "novelty_experiment_playbooks": novelty_experiment_playbooks,
        "novelty_mutation_plan": novelty_mutation_plan,
        "novelty_mutation_budget_plan": novelty_mutation_budget_plan,
        "novelty_budget_plan": novelty_budget_plan,
        "novelty_decision_frontier": novelty_decision_frontier,
        "novelty_falsification_plan": novelty_falsification_plan,
        "novelty_ablation_plan": novelty_ablation_plan,
        "novelty_reproducibility_plan": novelty_reproducibility_plan,
        "novelty_instrumentation_plan": novelty_instrumentation_plan,
        "novelty_artifact_contract_plan": novelty_artifact_contract_plan,
        "novelty_claim_packet_plan": novelty_claim_packet_plan,
        "novelty_validation_plan": novelty_validation_plan,
        "experiment_matrix": matrix,
        "portfolio_plan": portfolio,
        "promotion_gates": promotion_gates,
        "run_queue": _build_run_queue(portfolio, matrix, promotion_gates),
    }


def analyze_opportunity_file(
    path: Path,
    *,
    catalog_path: Path | None = None,
    target_catalog: Any | None = None,
    top_n: int | None = None,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
    target_speedup: float = DEFAULT_TARGET_SPEEDUP,
    min_memory_savings_pct: float = DEFAULT_MIN_MEMORY_SAVINGS_PCT,
    slow_baseline_ms: float = DEFAULT_SLOW_BASELINE_MS,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object in benchmark evidence file {path}, got {type(payload).__name__}"
        )
    if catalog_path is not None:
        catalog_payload = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        payload = dict(payload)
        payload["target_catalog"] = _catalog_entries_from_value(catalog_payload)
    if target_catalog is not None:
        payload = dict(payload)
        payload["available_targets"] = target_catalog
    candidates = normalize_candidates(payload)
    result = rank_opportunities(
        candidates,
        top_n=top_n,
        min_speedup=min_speedup,
        target_speedup=target_speedup,
        min_memory_savings_pct=min_memory_savings_pct,
        slow_baseline_ms=slow_baseline_ms,
    )
    result["source"] = str(path)
    if catalog_path is not None:
        result["target_catalog_source"] = str(catalog_path)
    if target_catalog is not None:
        result["discovered_target_source"] = "benchmark discovery"
    return result


def render_opportunities_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary", {})
    lines = [
        "# Optimization Opportunity Radar",
        "",
        f"Source: `{result.get('source', 'in-memory')}`",
        f"Candidates: `{summary.get('total_candidates', 0)}`; returned: `{summary.get('returned', 0)}`",
        f"Priority counts: `{summary.get('by_priority', {})}`",
        "",
        "| Rank | Target | Priority | Type | Score | Evidence | Next command |",
        "| ---: | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in result.get("opportunities", []):
        evidence = "; ".join(str(item) for item in row.get("evidence", []))
        lines.append(
            f"| {row.get('rank')} | `{row.get('target')}` | `{row.get('priority')}` | "
            f"`{row.get('opportunity_type')}` | {float(row.get('score', 0.0)):.2f} | "
            f"{evidence} | `{row.get('next_command')}` |"
        )

    lines.append("")
    plan = result.get("execution_plan") or {}
    if plan.get("phases"):
        lines.append("## Execution Plan")
        lines.append("")
        for phase in plan.get("phases", []):
            lines.append(f"### {phase.get('title')}")
            lines.append("")
            lines.append(f"Policy: {phase.get('parallelism_policy')}")
            lines.append("")
            for item in phase.get("items", []):
                command = item.get("command") or item.get("validation_command")
                if command:
                    lines.append(f"- `{item.get('target')}`: `{command}`")
                else:
                    lines.append(f"- `{item.get('target')}`")
                benchmark_run = item.get("benchmark_run") or {}
                render_command = benchmark_run.get("render_command")
                if render_command:
                    lines.append(f"  BenchmarkRun: `{render_command}`")
            lines.append("")
        lines.append(f"Completion gate: {plan.get('completion_gate')}")
        lines.append("")

    hypotheses = result.get("innovation_hypotheses") or {}
    clusters = hypotheses.get("clusters") or []
    if clusters:
        lines.append("## Innovation Hypotheses")
        lines.append("")
        guidance = hypotheses.get("portfolio_guidance")
        if guidance:
            lines.append(str(guidance))
            lines.append("")
        for cluster in clusters:
            lines.append(f"### {cluster.get('title')}")
            lines.append("")
            lines.append(
                f"Prototype: `{cluster.get('prototype_target')}`; support targets: `{cluster.get('support_count')}`; priority: `{cluster.get('priority')}`"
            )
            lines.append("")
            lines.append(str(cluster.get("reason") or ""))
            lines.append("")
            for experiment in cluster.get("transfer_experiments", []):
                lines.append(f"- {experiment}")
            lines.append(f"- Acceptance gate: {cluster.get('acceptance_gate')}")
            lines.append("")

    transfer_map = result.get("source_transfer_map") or {}
    transfer_patterns = transfer_map.get("patterns") or []
    if transfer_patterns:
        lines.append("## Source Transfer Map")
        lines.append("")
        lines.append(str(transfer_map.get("policy") or ""))
        lines.append("")
        for pattern in transfer_patterns[:5]:
            recipients = pattern.get("recipient_targets") or []
            lines.append(
                f"- `{pattern.get('pattern')}` from `{pattern.get('source_target')}` "
                f"to {len(recipients)} recipients"
            )
            blueprints = pattern.get("blueprint_ids") or []
            if blueprints:
                lines.append(f"  Blueprints: `{', '.join(str(item) for item in blueprints)}`")
            command = pattern.get("prototype_command")
            if command:
                lines.append(f"  Prototype: `{command}`")
        lines.append("")

    compound_map = result.get("compound_primitive_hypotheses") or {}
    compound_hypotheses = compound_map.get("hypotheses") or []
    if compound_hypotheses:
        lines.append("## Compound Primitive Hypotheses")
        lines.append("")
        lines.append(str(compound_map.get("policy") or ""))
        lines.append("")
        for hypothesis in compound_hypotheses[:5]:
            missing = ", ".join(str(item) for item in hypothesis.get("missing_primitives", []))
            present = ", ".join(str(item) for item in hypothesis.get("present_primitives", []))
            lines.append(
                f"- `{hypothesis.get('name')}` on `{hypothesis.get('target')}`: "
                f"present `{present}`; add `{missing}`"
            )
            support_targets = hypothesis.get("support_targets") or []
            if support_targets:
                lines.append(f"  Support: `{', '.join(str(item) for item in support_targets[:3])}`")
            command = hypothesis.get("prototype_command")
            if command:
                lines.append(f"  Prototype: `{command}`")
            gate = hypothesis.get("acceptance_gate")
            if gate:
                lines.append(f"  Acceptance gate: {gate}")
        lines.append("")

    synthesis_plan = result.get("novelty_primitive_pair_synthesis_plan") or {}
    syntheses = synthesis_plan.get("syntheses") or []
    if syntheses:
        lines.append("## Novelty Primitive Pair Synthesis Plan")
        lines.append("")
        lines.append(str(synthesis_plan.get("policy") or ""))
        lines.append("")
        lines.append(
            f"Syntheses: `{synthesis_plan.get('synthesis_count')}`; primitive pairs: `{synthesis_plan.get('primitive_pair_count')}`"
        )
        if synthesis_plan.get("available_synthesis_count") != synthesis_plan.get(
            "synthesis_count"
        ):
            lines.append(
                f"Available before limit: `{synthesis_plan.get('available_synthesis_count')}`"
            )
        lines.append("")
        for item in syntheses[:5]:
            pair = " + ".join(str(primitive) for primitive in item.get("pair", []) or [])
            lines.append(
                f"- `{pair}` on `{item.get('target')}`: score `{item.get('synthesis_score')}`"
            )
            hypothesis = item.get("hypothesis")
            if hypothesis:
                lines.append(f"  Hypothesis: {hypothesis}")
            command = item.get("prototype_command")
            if command:
                lines.append(f"  Prototype: `{command}`")
            gate = item.get("acceptance_gate")
            if gate:
                lines.append(f"  Acceptance gate: {gate}")
        lines.append("")

    coverage_gap_map = result.get("coverage_gap_map") or {}
    coverage_gaps = coverage_gap_map.get("gaps") or []
    if coverage_gaps:
        lines.append("## Coverage Gap Map")
        lines.append("")
        lines.append(str(coverage_gap_map.get("policy") or ""))
        lines.append("")
        for gap in coverage_gaps[:5]:
            label = (
                gap.get("name")
                or gap.get("primitive")
                or gap.get("signal")
                or gap.get("compound")
                or gap.get("gap_id")
            )
            lines.append(
                f"- `{label}` ({gap.get('gap_type')}, {gap.get('coverage_state')}): "
                f"{gap.get('recommended_action')}"
            )
            command = gap.get("first_probe_command")
            if command:
                lines.append(f"  First probe: `{command}`")
            sample_targets = [
                str(target.get("target"))
                for target in gap.get("sample_targets", []) or []
                if isinstance(target, dict) and target.get("target")
            ]
            if sample_targets:
                lines.append(f"  Sample targets: `{', '.join(sample_targets[:3])}`")
        lines.append("")

    bridge_map = result.get("cross_lane_bridge_map") or {}
    bridges = bridge_map.get("bridges") or []
    if bridges:
        lines.append("## Cross-Lane Bridge Map")
        lines.append("")
        lines.append(str(bridge_map.get("policy") or ""))
        lines.append("")
        for bridge in bridges[:5]:
            signals = ", ".join(str(item) for item in bridge.get("signals", []) or [])
            primitives = ", ".join(str(item) for item in bridge.get("present_primitives", []) or [])
            lines.append(
                f"- `{bridge.get('name')}` on `{bridge.get('prototype_target')}`: "
                f"signals `{signals}`; primitives `{primitives}`"
            )
            command = bridge.get("prototype_command")
            if command:
                lines.append(f"  Prototype: `{command}`")
            gate = bridge.get("acceptance_gate")
            if gate:
                lines.append(f"  Acceptance gate: {gate}")
        lines.append("")

    novelty_queue = result.get("novelty_queue") or {}
    novelty_leads = novelty_queue.get("leads") or []
    if novelty_leads:
        lines.append("## Novelty Queue")
        lines.append("")
        lines.append(str(novelty_queue.get("policy") or ""))
        lines.append("")
        for lead in novelty_leads[:8]:
            lines.append(
                f"- `{lead.get('title')}` ({lead.get('lead_type')}, score {float(lead.get('novelty_score') or 0.0):.2f})"
            )
            target = lead.get("target")
            if target:
                lines.append(f"  Target: `{target}`")
            command = lead.get("command")
            if command:
                lines.append(f"  Command: `{command}`")
            gate = lead.get("evidence_gate")
            if gate:
                lines.append(f"  Evidence gate: {gate}")
        lines.append("")

    playbook_map = result.get("novelty_experiment_playbooks") or {}
    playbooks = playbook_map.get("playbooks") or []
    if playbooks:
        lines.append("## Novelty Experiment Playbooks")
        lines.append("")
        lines.append(str(playbook_map.get("policy") or ""))
        lines.append("")
        for playbook in playbooks[:5]:
            metrics = ", ".join(str(item) for item in playbook.get("primary_metrics", []) or [])
            guardrails = ", ".join(
                str(item) for item in playbook.get("guardrail_metrics", []) or []
            )
            lines.append(
                f"- `{playbook.get('title')}` on `{playbook.get('target')}`: metrics `{metrics}`; guardrails `{guardrails}`"
            )
            variants = playbook.get("variant_ladder") or []
            if variants:
                variant_names = ", ".join(str(item.get("variant")) for item in variants[:4])
                lines.append(f"  Variant ladder: `{variant_names}`")
            tools = playbook.get("profiler_tools") or []
            if tools:
                lines.append(f"  Profilers: `{', '.join(str(item) for item in tools)}`")
        lines.append("")

    mutation_plan = result.get("novelty_mutation_plan") or {}
    lead_mutations = mutation_plan.get("lead_mutations") or []
    if lead_mutations:
        lines.append("## Novelty Mutation Plan")
        lines.append("")
        lines.append(str(mutation_plan.get("policy") or ""))
        lines.append("")
        lines.append(
            f"Leads: `{mutation_plan.get('lead_count')}`; mutations: `{mutation_plan.get('mutation_count')}`"
        )
        operator_counts = mutation_plan.get("operator_counts") or {}
        if operator_counts:
            preview = ", ".join(
                f"{operator}={count}" for operator, count in sorted(operator_counts.items())[:6]
            )
            lines.append(f"Operators: `{preview}`")
        lines.append("")
        for item in lead_mutations[:5]:
            mutations = item.get("mutations") or []
            lines.append(
                f"- `{item.get('title')}` on `{item.get('target')}`: `{item.get('mutation_count')}` mutations"
            )
            if mutations:
                first = mutations[0]
                lines.append(
                    f"  First mutation: `{first.get('operator')}` changes `{first.get('variable')}`"
                )
                command = first.get("command")
                if command:
                    lines.append(f"  Command: `{command}`")
        lines.append("")

    mutation_budget_plan = result.get("novelty_mutation_budget_plan") or {}
    mutation_budget_selected = mutation_budget_plan.get("selected") or []
    if mutation_budget_selected:
        lines.append("## Novelty Mutation Budget Plan")
        lines.append("")
        lines.append(str(mutation_budget_plan.get("selection_policy") or ""))
        lines.append("")
        lines.append(
            f"Selected: `{mutation_budget_plan.get('selected_count')}` mutations; cost units: `{mutation_budget_plan.get('selected_cost_units')}/{mutation_budget_plan.get('max_cost_units')}`"
        )
        lines.append(
            f"Leads: `{mutation_budget_plan.get('selected_lead_count')}`; operators: `{mutation_budget_plan.get('selected_operator_count')}`"
        )
        if mutation_budget_plan.get("backlog_count"):
            deferral_counts = mutation_budget_plan.get("deferral_reason_counts") or {}
            top_deferral = (
                max(deferral_counts.items(), key=lambda item: item[1])[0]
                if deferral_counts
                else "unknown"
            )
            lines.append(
                f"Deferred mutations: `{mutation_budget_plan.get('backlog_count')}`; top reason: `{top_deferral}`"
            )
        lines.append("")
        for item in mutation_budget_selected[:5]:
            risks = ", ".join(str(risk) for risk in item.get("risk_flags", []) or []) or "none"
            lines.append(
                f"- `{item.get('operator')}` on `{item.get('target')}` changes `{item.get('variable')}`: info `{item.get('information_gain_score')}`; cost `{item.get('cost_units')}`; risks `{risks}`"
            )
            command = item.get("command")
            if command:
                lines.append(f"  Command: `{command}`")
        lines.append("")

    budget_plan = result.get("novelty_budget_plan") or {}
    budget_selected = budget_plan.get("selected") or []
    if budget_selected:
        lines.append("## Novelty Budget Plan")
        lines.append("")
        lines.append(str(budget_plan.get("selection_policy") or ""))
        lines.append("")
        lines.append(
            f"Selected: `{budget_plan.get('selected_count')}` leads; cost units: `{budget_plan.get('selected_cost_units')}/{budget_plan.get('max_cost_units')}`"
        )
        if budget_plan.get("backlog_count"):
            deferral_counts = budget_plan.get("deferral_reason_counts") or {}
            top_deferral = (
                max(deferral_counts.items(), key=lambda item: item[1])[0]
                if deferral_counts
                else "unknown"
            )
            lines.append(
                f"Deferred: `{budget_plan.get('backlog_count')}` leads; top reason: `{top_deferral}`"
            )
        lines.append("")
        for item in budget_selected[:5]:
            risks = ", ".join(str(risk) for risk in item.get("risk_flags", []) or []) or "none"
            lines.append(
                f"- `{item.get('title')}` on `{item.get('target')}`: value `{item.get('expected_value_score')}`; cost `{item.get('cost_units')}`; risks `{risks}`"
            )
            mitigations = item.get("risk_mitigation_steps") or []
            if mitigations:
                lines.append(f"  Mitigate: {mitigations[0]}")
        lines.append("")

    decision_frontier = result.get("novelty_decision_frontier") or {}
    decision_lanes = decision_frontier.get("lanes") or []
    if decision_lanes:
        lines.append("## Novelty Decision Frontier")
        lines.append("")
        lines.append(str(decision_frontier.get("policy") or ""))
        lines.append("")
        for lane in decision_lanes[:5]:
            lane_leads = lane.get("leads") or []
            if not lane_leads:
                continue
            lines.append(f"- `{lane.get('lane')}`: {len(lane_leads)} leads; {lane.get('policy')}")
            first = lane_leads[0]
            lines.append(
                f"  First: `{first.get('title')}` on `{first.get('target')}` ({first.get('selection_state')})"
            )
        lines.append("")

    falsification_plan = result.get("novelty_falsification_plan") or {}
    falsification_checks = falsification_plan.get("lead_checks") or []
    if falsification_checks:
        lines.append("## Novelty Falsification Plan")
        lines.append("")
        lines.append(str(falsification_plan.get("policy") or ""))
        lines.append("")
        lines.append(
            f"Selected leads: `{falsification_plan.get('lead_count')}`; falsification checks: `{falsification_plan.get('check_count')}`"
        )
        lines.append("")
        for item in falsification_checks[:5]:
            checks = item.get("falsification_checks") or []
            lines.append(
                f"- `{item.get('title')}` on `{item.get('target')}`: null `{item.get('null_hypothesis')}`"
            )
            if checks:
                lines.append(f"  Disprove if: {checks[0]}")
            boundary = item.get("claim_boundary")
            if boundary:
                lines.append(f"  Claim boundary: {boundary}")
        lines.append("")

    ablation_plan = result.get("novelty_ablation_plan") or {}
    ablation_leads = ablation_plan.get("lead_controls") or []
    if ablation_leads:
        lines.append("## Novelty Ablation Plan")
        lines.append("")
        lines.append(str(ablation_plan.get("policy") or ""))
        lines.append("")
        lines.append(
            f"Selected leads: `{ablation_plan.get('lead_count')}`; ablation controls: `{ablation_plan.get('control_count')}`"
        )
        lines.append("")
        for item in ablation_leads[:5]:
            controls = item.get("controls") or []
            lines.append(
                f"- `{item.get('title')}` on `{item.get('target')}`: `{item.get('control_count')}` controls"
            )
            if controls:
                control = controls[0]
                lines.append(
                    f"  Control: `{control.get('control_type')}` rejects if {control.get('rejects_claim_if')}"
                )
        lines.append("")

    reproducibility_plan = result.get("novelty_reproducibility_plan") or {}
    reproducibility_leads = reproducibility_plan.get("lead_profiles") or []
    if reproducibility_leads:
        lines.append("## Novelty Reproducibility Plan")
        lines.append("")
        lines.append(str(reproducibility_plan.get("policy") or ""))
        lines.append("")
        lines.append(
            f"Selected leads: `{reproducibility_plan.get('lead_count')}`; total repeats: `{reproducibility_plan.get('repeat_count_total')}`"
        )
        lines.append("")
        for item in reproducibility_leads[:5]:
            metrics = ", ".join(str(metric) for metric in item.get("stability_metrics", []) or [])
            lines.append(
                f"- `{item.get('title')}` on `{item.get('target')}`: repeats `{item.get('repeat_count')}`; variance gate `{item.get('variance_threshold_pct')}%`"
            )
            if metrics:
                lines.append(f"  Stability metrics: `{metrics}`")
        lines.append("")

    instrumentation_plan = result.get("novelty_instrumentation_plan") or {}
    instrumentation_leads = instrumentation_plan.get("lead_profiles") or []
    if instrumentation_leads:
        lines.append("## Novelty Instrumentation Plan")
        lines.append("")
        lines.append(str(instrumentation_plan.get("policy") or ""))
        lines.append("")
        tool_counts = instrumentation_plan.get("tool_counts") or {}
        if tool_counts:
            tools = ", ".join(f"{tool}={count}" for tool, count in sorted(tool_counts.items()))
            lines.append(f"Profiler tools: `{tools}`")
            lines.append("")
        for item in instrumentation_leads[:5]:
            tools = ", ".join(str(tool) for tool in item.get("required_profiler_tools", []) or [])
            checks = item.get("preflight_checks") or []
            lines.append(
                f"- `{item.get('title')}` on `{item.get('target')}`: tools `{tools}`; preflight checks `{len(checks)}`"
            )
            if checks:
                lines.append(
                    f"  First check: `{checks[0].get('check')}` via `{checks[0].get('command')}`"
                )
        lines.append("")

    artifact_contract_plan = result.get("novelty_artifact_contract_plan") or {}
    artifact_contracts = artifact_contract_plan.get("lead_contracts") or []
    if artifact_contracts:
        lines.append("## Novelty Artifact Contract Plan")
        lines.append("")
        lines.append(str(artifact_contract_plan.get("policy") or ""))
        lines.append("")
        lines.append(
            f"Selected leads: `{artifact_contract_plan.get('lead_count')}`; required files: `{artifact_contract_plan.get('required_file_count')}`"
        )
        lines.append("")
        for contract in artifact_contracts[:5]:
            stage_contracts = contract.get("stage_contracts") or []
            lines.append(
                f"- `{contract.get('title')}` on `{contract.get('target')}`: package `{contract.get('package_manifest')}`; stages `{len(stage_contracts)}`"
            )
            if stage_contracts:
                first_stage = stage_contracts[0]
                lines.append(
                    f"  First stage: `{first_stage.get('stage')}` writes `{first_stage.get('job_id')}`"
                )
        lines.append("")

    claim_packet_plan = result.get("novelty_claim_packet_plan") or {}
    claim_packets = claim_packet_plan.get("lead_packets") or []
    if claim_packets:
        lines.append("## Novelty Claim Packet Plan")
        lines.append("")
        lines.append(str(claim_packet_plan.get("policy") or ""))
        lines.append("")
        lines.append(
            f"Selected leads: `{claim_packet_plan.get('lead_count')}`; required sections: `{claim_packet_plan.get('required_section_count')}`; blocked overclaims: `{claim_packet_plan.get('disallowed_claim_count')}`"
        )
        lines.append("")
        for packet in claim_packets[:5]:
            disallowed = packet.get("disallowed_claims") or []
            sections = packet.get("required_sections") or []
            lines.append(
                f"- `{packet.get('title')}` on `{packet.get('target')}`: packet `{packet.get('packet_id')}`"
            )
            lines.append(f"  Allowed scope: {packet.get('allowed_claim_scope')}")
            if sections:
                lines.append(f"  First required section: `{sections[0].get('section')}`")
            if disallowed:
                lines.append(f"  First blocked overclaim: {disallowed[0]}")
        lines.append("")

    evidence_audit_plan = result.get("novelty_evidence_audit_plan") or {}
    evidence_audits = evidence_audit_plan.get("lead_audits") or []
    if evidence_audits:
        lines.append("## Novelty Evidence Audit Plan")
        lines.append("")
        lines.append(str(evidence_audit_plan.get("policy") or ""))
        lines.append("")
        lines.append(
            f"Audited leads: `{evidence_audit_plan.get('lead_count')}`; missing files: `{evidence_audit_plan.get('missing_file_count')}`; present files: `{evidence_audit_plan.get('present_file_count')}`"
        )
        lines.append("")
        for audit in evidence_audits[:5]:
            blockers = audit.get("promotion_blockers") or []
            lines.append(
                f"- `{audit.get('title')}` on `{audit.get('target')}`: `{audit.get('audit_status')}`; missing `{audit.get('missing_file_count')}`"
            )
            if blockers:
                lines.append(f"  First blocker: {blockers[0]}")
        lines.append("")

    recovery_plan = result.get("novelty_recovery_plan") or {}
    recovery_actions = recovery_plan.get("actions") or []
    if recovery_actions:
        lines.append("## Novelty Recovery Plan")
        lines.append("")
        lines.append(str(recovery_plan.get("policy") or ""))
        lines.append("")
        lines.append(
            f"Recovery actions: `{recovery_plan.get('action_count')}`; blocking: `{recovery_plan.get('blocking_action_count')}`"
        )
        lines.append("")
        for action in recovery_actions[:5]:
            lines.append(
                f"- `{action.get('issue_type')}` for `{action.get('job_id')}`: {action.get('unblock_condition')}"
            )
            recovery_command = action.get("recovery_command")
            if recovery_command:
                lines.append(f"  Recovery command: `{recovery_command}`")
            rerun_command = action.get("rerun_after_recovery")
            if rerun_command:
                lines.append(f"  Rerun: `{rerun_command}`")
        lines.append("")

    adaptive_plan = result.get("novelty_adaptive_decision_plan") or {}
    adaptive_decisions = adaptive_plan.get("selected_decisions") or []
    if adaptive_decisions:
        lines.append("## Novelty Adaptive Decision Plan")
        lines.append("")
        lines.append(str(adaptive_plan.get("policy") or ""))
        lines.append("")
        lines.append(
            f"Selected leads: `{adaptive_plan.get('selected_count')}`; blocked: `{adaptive_plan.get('blocked_count')}`; replacements: `{adaptive_plan.get('replacement_candidate_count')}`"
        )
        lines.append("")
        for decision in adaptive_decisions[:5]:
            lines.append(
                f"- `{decision.get('title')}` on `{decision.get('target')}`: `{decision.get('disposition')}` ({decision.get('slot_state')})"
            )
            next_step = decision.get("next_step")
            if next_step:
                lines.append(f"  Next step: {next_step}")
        replacements = adaptive_plan.get("replacement_candidates") or []
        if replacements:
            lines.append("")
            lines.append("Replacement candidates:")
            for item in replacements[:3]:
                lines.append(
                    f"- `{item.get('title')}` replaces `{item.get('replacement_for')}`: `{item.get('command')}`"
                )
        lines.append("")

    learning_plan = result.get("novelty_learning_plan") or {}
    learning_adjustments = learning_plan.get("lead_adjustments") or []
    if learning_adjustments:
        lines.append("## Novelty Learning Plan")
        lines.append("")
        lines.append(str(learning_plan.get("policy") or ""))
        lines.append("")
        guidance = learning_plan.get("portfolio_guidance") or {}
        lines.append(
            f"Lead adjustments: `{learning_plan.get('lead_count')}`; blocked: `{learning_plan.get('blocked_learning_count')}`; backups: `{learning_plan.get('backup_candidate_count')}`"
        )
        if guidance:
            lines.append(
                f"Guidance: continue `{guidance.get('continue_count')}`, recover `{guidance.get('recover_count')}`, review `{guidance.get('review_count')}`, backup `{guidance.get('backup_count')}`"
            )
        lines.append("")
        for item in learning_adjustments[:5]:
            risks = ", ".join(str(risk) for risk in item.get("risk_updates", []) or []) or "none"
            lines.append(
                f"- `{item.get('title')}` on `{item.get('target')}`: `{item.get('rerank_action')}`; delta `{item.get('expected_value_adjustment')}`; risks `{risks}`"
            )
            focus = item.get("next_validation_focus")
            if focus:
                lines.append(f"  Focus: {focus}")
        backups = learning_plan.get("backup_learning") or []
        if backups:
            lines.append("")
            lines.append("Backup learning:")
            for item in backups[:3]:
                lines.append(
                    f"- `{item.get('lead_id')}` for `{item.get('replacement_for')}`: {item.get('rerank_action')}"
                )
        lines.append("")

    next_wave_plan = result.get("novelty_next_wave_plan") or {}
    next_waves = next_wave_plan.get("waves") or []
    if next_waves:
        lines.append("## Novelty Next Wave Plan")
        lines.append("")
        lines.append(str(next_wave_plan.get("policy") or ""))
        lines.append("")
        lines.append(
            f"Waves: `{next_wave_plan.get('wave_count')}`; actions: `{next_wave_plan.get('action_count')}`"
        )
        lines.append("")
        for wave in next_waves[:5]:
            lines.append(
                f"- `{wave.get('name')}`: `{wave.get('item_count')}` items; {wave.get('purpose')}"
            )
            items = wave.get("items") or []
            if items:
                first = items[0]
                label = first.get("action_id") or first.get("job_id") or first.get("lead_id")
                target = first.get("target") or first.get("replacement_for") or "portfolio"
                lines.append(f"  First: `{label}` on `{target}`")
                command = first.get("command") or first.get("recovery_command")
                if command:
                    lines.append(f"  Command: `{command}`")
        lines.append("")

    harvest_plan = result.get("novelty_harvest_plan") or {}
    harvest_patterns = harvest_plan.get("patterns") or []
    blocked_harvests = harvest_plan.get("blocked_harvests") or []
    if harvest_patterns or blocked_harvests:
        lines.append("## Novelty Harvest Plan")
        lines.append("")
        lines.append(str(harvest_plan.get("policy") or ""))
        lines.append("")
        lines.append(
            f"Harvested patterns: `{harvest_plan.get('pattern_count')}`; follow-ups: `{harvest_plan.get('followup_count')}`; blocked: `{harvest_plan.get('blocked_harvest_count')}`"
        )
        lines.append("")
        for pattern in harvest_patterns[:5]:
            lines.append(
                f"- `{pattern.get('pattern_id')}` from `{pattern.get('source_target')}`: {pattern.get('pattern_summary')}"
            )
            followups = [
                item
                for item in harvest_plan.get("followup_experiments", []) or []
                if item.get("source_pattern_id") == pattern.get("pattern_id")
            ]
            if followups:
                lines.append(
                    f"  First follow-up: `{followups[0].get('followup_id')}` on `{followups[0].get('target')}`"
                )
        if blocked_harvests:
            lines.append("")
            lines.append("Blocked harvests:")
            for item in blocked_harvests[:3]:
                lines.append(
                    f"- `{item.get('lead_id')}`: {item.get('blocker')}"
                )
        lines.append("")

    novelty_plan = result.get("novelty_validation_plan") or {}
    novelty_jobs = novelty_plan.get("jobs") or []
    if novelty_jobs:
        lines.append("## Novelty Validation Plan")
        lines.append("")
        lines.append(str(novelty_plan.get("policy") or ""))
        lines.append("")
        lines.append(f"Selected leads: `{novelty_plan.get('selected_lead_count')}`")
        lines.append(f"Critical path groups: `{novelty_plan.get('critical_path_groups')}`")
        novelty_resume = novelty_plan.get("resume_plan") or {}
        if novelty_resume:
            lines.append(
                f"Resume: `{len(novelty_resume.get('ready_job_ids', []) or [])}` ready, `{len(novelty_resume.get('blocked_job_ids', []) or [])}` blocked, `{len(novelty_resume.get('failed_job_ids', []) or [])}` failed, `{len(novelty_resume.get('manual_review_job_ids', []) or [])}` awaiting review"
            )
            next_actions = novelty_resume.get("next_actions") or []
            if next_actions:
                lines.append(f"Next action: {next_actions[0]}")
        lines.append("")
        for group in (novelty_plan.get("dispatch_groups") or [])[:4]:
            stages = ", ".join(
                f"{stage}={count}" for stage, count in group.get("stage_mix", {}).items()
            )
            lines.append(
                f"- Dispatch group {group.get('index')}: {len(group.get('job_ids', []))} jobs ({stages})"
            )
        lines.append("")
        for job in novelty_jobs[:8]:
            command = job.get("command") or "manual novelty review"
            depends = ", ".join(str(dep) for dep in job.get("depends_on", [])) or "none"
            lines.append(
                f"- `{job.get('id')}` ({job.get('stage')}, depends: {depends}): `{command}`"
            )
        lines.append("")

    frontier_map = result.get("frontier_discovery_map") or {}
    frontier_lanes = frontier_map.get("lanes") or []
    if frontier_lanes:
        lines.append("## Frontier Discovery Map")
        lines.append("")
        lines.append(str(frontier_map.get("selection_policy") or ""))
        lines.append("")
        for lane in frontier_lanes[:5]:
            top_target = (lane.get("top_targets") or [{}])[0]
            lines.append(
                f"- `{lane.get('lane')}`: {lane.get('candidate_count')} candidates; first probe `{top_target.get('target')}`"
            )
            first_command = lane.get("first_command")
            if first_command:
                lines.append(f"  Command: `{first_command}`")
            blueprints = lane.get("experiment_blueprints") or []
            if blueprints:
                first_blueprint = blueprints[0]
                lines.append(
                    f"  Blueprint: `{first_blueprint.get('name')}` - {first_blueprint.get('hypothesis')}"
                )
        lines.append("")

    matrix = result.get("experiment_matrix") or {}
    cards = matrix.get("cards") or []
    if cards:
        lines.append("## Experiment Matrix")
        lines.append("")
        for rule in matrix.get("design_rules", []):
            lines.append(f"- {rule}")
        lines.append("")
        for card in cards[:5]:
            lines.append(f"### `{card.get('target')}`")
            lines.append("")
            lines.append(
                f"Primary metric: `{card.get('primary_metric')}`; motif: `{card.get('motif')}`"
            )
            lines.append(f"Control: `{card.get('control_command')}`")
            lines.append(f"Acceptance gate: {card.get('acceptance_gate')}")
            lines.append("")
            for variant in card.get("variants", []):
                lines.append(f"- `{variant.get('name')}`: `{variant.get('validation_command')}`")
            lines.append("")

    portfolio = result.get("portfolio_plan") or {}
    selected = portfolio.get("selected") or []
    if selected:
        lines.append("## Portfolio Plan")
        lines.append("")
        lines.append(str(portfolio.get("selection_policy") or ""))
        lines.append("")
        for item in selected:
            lines.append(
                f"- `{item.get('target')}` via `{item.get('first_variant')}`: `{item.get('command')}`"
            )
        lines.append("")

    gates = (result.get("promotion_gates") or {}).get("gates") or []
    if gates:
        lines.append("## Promotion Gates")
        lines.append("")
        lines.append(str((result.get("promotion_gates") or {}).get("global_rule") or ""))
        lines.append("")
        for gate in gates[:5]:
            lines.append(
                f"- `{gate.get('target')}`: `{gate.get('promotion_state')}`; claim_allowed=`{gate.get('claim_allowed')}`"
            )
        lines.append("")

    queue = result.get("run_queue") or {}
    jobs = queue.get("jobs") or []
    if jobs:
        lines.append("## Run Queue")
        lines.append("")
        lines.append(str(queue.get("parallelism_policy") or ""))
        lines.append("")
        dispatch_groups = queue.get("dispatch_groups") or []
        if dispatch_groups:
            lines.append(f"Critical path groups: `{queue.get('critical_path_groups')}`")
            for group in dispatch_groups[:5]:
                stages = ", ".join(
                    f"{stage}={count}" for stage, count in group.get("stage_mix", {}).items()
                )
                lines.append(
                    f"- Dispatch group {group.get('index')}: {len(group.get('job_ids', []))} jobs ({stages})"
                )
            lines.append("")
        for job in jobs[:10]:
            depends = ", ".join(str(dep) for dep in job.get("depends_on", [])) or "none"
            command = job.get("command") or "manual promotion review"
            lines.append(
                f"- `{job.get('id')}` ({job.get('stage')}, depends: {depends}): `{command}`"
            )
        lines.append("")

    lines.append("## Recommended Experiments")
    lines.append("")
    for row in result.get("opportunities", []):
        lines.append(f"### {row.get('rank')}. `{row.get('target')}`")
        lines.append("")
        lines.append(row.get("rationale", ""))
        lines.append("")
        for experiment in row.get("recommended_experiments", []):
            lines.append(f"- {experiment}")
        lines.append("")
    return "\n".join(lines)


def _shell_quote(value: Any) -> str:
    return shlex.quote(str(value))


def _claim_packet_markdown_lines(claim_packet: dict[str, Any], target: str) -> list[str]:
    sections = [
        section
        for section in claim_packet.get("required_sections", []) or []
        if isinstance(section, dict)
    ]
    disallowed_claims = [str(item) for item in claim_packet.get("disallowed_claims", []) or []]
    lines = [
        f"# Novelty Claim Packet: {target}",
        "",
        "Allowed claim scope:",
        f"- {claim_packet.get('allowed_claim_scope') or 'Claim remains blocked.'}",
        "",
        "Required sections:",
    ]
    if sections:
        for section in sections:
            lines.append(
                f"- [ ] `{section.get('section')}`: {section.get('must_include')}"
            )
            sources = [str(item) for item in section.get("evidence_sources", []) or []]
            if sources:
                lines.append(f"  Evidence sources: {', '.join(sources[:4])}")
    else:
        lines.append("- [ ] Attach control, candidate, profile, and review evidence.")
    lines.extend(["", "Disallowed claims:"])
    if disallowed_claims:
        lines.extend(f"- [ ] Confirm absent: {item}" for item in disallowed_claims)
    else:
        lines.append("- [ ] Confirm no unsupported claims are present.")
    lines.extend(
        [
            "",
            "Approval rule:",
            f"- {claim_packet.get('approval_rule') or 'Keep APPROVED absent until evidence is complete.'}",
            "",
        ]
    )
    return lines


def _render_job_plan_shell(
    queue: dict[str, Any],
    *,
    root_env_var: str,
    default_root: str,
    root_label: str,
    empty_message: str,
    completion_message: str,
    review_title: str,
) -> str:
    """Render dependency-ordered jobs as an evidence-preserving shell runbook."""

    jobs = queue.get("jobs") or []
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'AISP_RUN_QUEUE_CWD="${AISP_RUN_QUEUE_CWD:-$(pwd)}"',
        f'{root_env_var}="${{{root_env_var}:-{default_root}}}"',
        f'mkdir -p "${root_env_var}"',
        f'echo "{root_label}: ${root_env_var}"',
        "",
        "require_dependency() {",
        f'  local dep_dir="${root_env_var}/$1"',
        '  if [[ ! -f "$dep_dir/DONE" && ! -f "$dep_dir/APPROVED" ]]; then',
        '    echo "Missing completed dependency: $1" >&2',
        "    exit 1",
        "  fi",
        "}",
        "",
    ]

    if not jobs:
        lines.extend([f'echo "{empty_message}"', "exit 0", ""])
        return "\n".join(lines)

    for job in jobs:
        job_id = str(job.get("id") or "job")
        stage = str(job.get("stage") or "unknown")
        target = str(job.get("target") or "")
        command = job.get("command")
        depends_on = [str(dep) for dep in job.get("depends_on", [])]
        job_json = json.dumps(job, indent=2, sort_keys=True)
        artifact_contract = (
            job.get("artifact_contract")
            if isinstance(job.get("artifact_contract"), dict)
            else None
        )
        artifact_contract_json = (
            json.dumps(artifact_contract, indent=2, sort_keys=True)
            if artifact_contract
            else None
        )
        claim_packet = (
            job.get("claim_packet") if isinstance(job.get("claim_packet"), dict) else None
        )
        claim_packet_json = (
            json.dumps(claim_packet, indent=2, sort_keys=True) if claim_packet else None
        )

        lines.extend(
            [
                f"echo '==> {job_id} [{stage}] {target}'",
                f'job_dir="${root_env_var}/{job_id}"',
                'mkdir -p "$job_dir"',
            ]
        )
        for dependency in depends_on:
            lines.append(f"require_dependency {_shell_quote(dependency)}")

        lines.extend(
            [
                "cat > \"$job_dir/job.json\" <<'JSON'",
                job_json,
                "JSON",
            ]
        )
        if artifact_contract_json:
            lines.extend(
                [
                    "cat > \"$job_dir/artifact_contract.json\" <<'JSON'",
                    artifact_contract_json,
                    "JSON",
                ]
            )
        if claim_packet_json:
            lines.extend(
                [
                    "cat > \"$job_dir/claim_packet.json\" <<'JSON'",
                    claim_packet_json,
                    "JSON",
                ]
            )

        if command:
            lines.extend(
                [
                    'if [[ -f "$job_dir/DONE" ]]; then',
                    '  echo "  skip: DONE already exists"',
                    "else",
                    f"  printf '%s\\n' {_shell_quote(command)} > \"$job_dir/command.txt\"",
                    f'  (cd "$AISP_RUN_QUEUE_CWD" && bash -lc {_shell_quote(command)}) >"$job_dir/stdout.log" 2>"$job_dir/stderr.log"',
                    '  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$job_dir/DONE"',
                    '  echo "  logs: $job_dir/stdout.log $job_dir/stderr.log"',
                    "fi",
                    "",
                ]
            )
            continue

        evidence = [str(item) for item in job.get("required_evidence", [])]
        claim_boundary = str(job.get("claim_boundary") or "")
        falsification_checks = [str(item) for item in job.get("falsification_checks", []) or []]
        risk_mitigation_steps = [str(item) for item in job.get("risk_mitigation_steps", []) or []]
        ablation_controls = [
            control
            for control in job.get("ablation_controls", []) or []
            if isinstance(control, dict)
        ]
        repeat_count = job.get("repeat_count")
        variance_threshold = job.get("variance_threshold_pct")
        stability_metrics = [str(item) for item in job.get("stability_metrics", []) or []]
        reproducibility_gate = str(job.get("reproducibility_gate") or "")
        instrumentation_preflight = [
            check
            for check in job.get("instrumentation_preflight", []) or []
            if isinstance(check, dict)
        ]
        required_profiler_tools = [
            str(item) for item in job.get("required_profiler_tools", []) or []
        ]
        launch_environment = [str(item) for item in job.get("launch_environment", []) or []]
        artifact_contract = (
            job.get("artifact_contract")
            if isinstance(job.get("artifact_contract"), dict)
            else {}
        )
        claim_packet = (
            job.get("claim_packet") if isinstance(job.get("claim_packet"), dict) else {}
        )
        lines.extend(
            [
                'if [[ -f "$job_dir/APPROVED" ]]; then',
                '  echo "  approved: $job_dir/APPROVED"',
                'elif [[ -f "$job_dir/MANUAL_REVIEW_REQUIRED" ]]; then',
                '  echo "  manual review pending: $job_dir/promotion_review.md"',
                "else",
            ]
        )
        lines.extend(
            [
                "  cat > \"$job_dir/promotion_review.md\" <<'REVIEW'",
                f"# {review_title}: {target}",
                "",
                f"Stage: `{stage}`",
                f"Gate: `{job.get('promotion_gate') or job.get('evidence_gate')}`",
                "",
            ]
        )
        if claim_boundary:
            lines.extend(["Claim boundary:", f"- {claim_boundary}", ""])
        if falsification_checks:
            lines.append("Falsification checks:")
            lines.extend(f"- {item}" for item in falsification_checks)
            lines.append("")
        if risk_mitigation_steps:
            lines.append("Risk mitigations:")
            lines.extend(f"- {item}" for item in risk_mitigation_steps)
            lines.append("")
        if ablation_controls:
            lines.append("Ablation controls:")
            for control in ablation_controls:
                lines.append(
                    f"- `{control.get('control_type')}` rejects the claim if {control.get('rejects_claim_if')}"
                )
            lines.append("")
        if repeat_count or stability_metrics or reproducibility_gate:
            lines.append("Reproducibility gate:")
            if repeat_count:
                lines.append(f"- Repeat count: `{repeat_count}`")
            if variance_threshold is not None:
                lines.append(f"- Variance threshold: `{variance_threshold}%`")
            if stability_metrics:
                lines.append(f"- Stability metrics: `{', '.join(stability_metrics)}`")
            if reproducibility_gate:
                lines.append(f"- {reproducibility_gate}")
            lines.append("")
        if instrumentation_preflight or required_profiler_tools:
            lines.append("Instrumentation preflight:")
            if required_profiler_tools:
                lines.append(f"- Required profiler tools: `{', '.join(required_profiler_tools)}`")
            if launch_environment:
                lines.append(f"- Launch environment: `{', '.join(launch_environment)}`")
            for check in instrumentation_preflight:
                lines.append(
                    f"- `{check.get('check')}`: `{check.get('command')}` ({'required' if check.get('required') else 'optional'})"
                )
            lines.append("")
        if claim_packet:
            sections = claim_packet.get("required_sections") or []
            disallowed_claims = claim_packet.get("disallowed_claims") or []
            lines.append("Claim packet:")
            lines.append(f"- Packet: `{claim_packet.get('packet_id')}`")
            lines.append(f"- Path: `{claim_packet.get('packet_path')}`")
            lines.append(f"- Allowed scope: {claim_packet.get('allowed_claim_scope')}")
            lines.append(f"- Required sections: `{len(sections)}`")
            lines.append(f"- Disallowed claims: `{len(disallowed_claims)}`")
            lines.append("")
        if artifact_contract:
            stage_contracts = artifact_contract.get("stage_contracts") or [artifact_contract]
            lines.append("Artifact contract:")
            contract_id = artifact_contract.get("contract_id")
            if contract_id:
                lines.append(f"- Contract: `{contract_id}`")
            package_manifest = artifact_contract.get("package_manifest")
            if package_manifest:
                lines.append(f"- Package manifest: `{package_manifest}`")
            for stage_contract in stage_contracts:
                if not isinstance(stage_contract, dict):
                    continue
                required_files = [
                    str(item) for item in stage_contract.get("required_files", []) or []
                ]
                files_preview = ", ".join(required_files[:6])
                if len(required_files) > 6:
                    files_preview = f"{files_preview}, ..."
                lines.append(
                    f"- `{stage_contract.get('stage')}` `{stage_contract.get('job_id')}` requires `{files_preview}`"
                )
            lines.append("")
        lines.append("Required evidence:")
        if evidence:
            lines.extend(f"- {item}" for item in evidence)
        else:
            lines.append("- Confirm all upstream job artifacts and guardrails before promotion.")
        lines.extend(
            [
                "",
                "After review, create an `APPROVED` file in this directory only when the evidence is complete and the claim remains within the gate.",
                "REVIEW",
            ]
        )
        if claim_packet:
            lines.extend(
                [
                    "  cat > \"$job_dir/claim_packet.md\" <<'CLAIM'",
                    *_claim_packet_markdown_lines(claim_packet, target),
                    "CLAIM",
                ]
            )
        lines.extend(
            [
                '  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$job_dir/MANUAL_REVIEW_REQUIRED"',
                '  echo "  manual review: $job_dir/promotion_review.md"',
                "fi",
                "",
            ]
        )

    lines.extend([f'echo "{completion_message}"', ""])
    return "\n".join(lines)


def render_run_queue_shell(result: dict[str, Any]) -> str:
    """Render the selected run queue as a local evidence-preserving shell runbook."""

    return _render_job_plan_shell(
        result.get("run_queue") or {},
        root_env_var="AISP_RUN_QUEUE_ROOT",
        default_root="artifacts/opportunity_run_queue/$(date -u +%Y%m%d_%H%M%S)",
        root_label="Run queue root",
        empty_message="No run queue jobs found.",
        completion_message="Run queue script complete. Promotion reviews remain manual gates.",
        review_title="Promotion Review",
    )


def render_novelty_validation_shell(result: dict[str, Any]) -> str:
    """Render the novelty validation plan as an evidence-preserving shell runbook."""

    return _render_job_plan_shell(
        result.get("novelty_validation_plan") or {},
        root_env_var="AISP_NOVELTY_QUEUE_ROOT",
        default_root="artifacts/novelty_validation_queue/$(date -u +%Y%m%d_%H%M%S)",
        root_label="Novelty validation root",
        empty_message="No novelty validation jobs found.",
        completion_message="Novelty validation script complete. Manual reviews remain evidence gates.",
        review_title="Novelty Review",
    )


def _next_wave_action_markdown_lines(
    wave: dict[str, Any], item: dict[str, Any]
) -> list[str]:
    title = item.get("title") or item.get("lead_id") or item.get("item_id") or "next-wave action"
    lines = [
        f"# Novelty Next Wave Action: {title}",
        "",
        f"Wave: `{wave.get('name')}`",
        f"Source: `{item.get('source')}`",
    ]
    target = item.get("target")
    if target:
        lines.append(f"Target: `{target}`")
    action_id = item.get("action_id")
    if action_id:
        lines.append(f"Recovery action: `{action_id}`")
    job_id = item.get("job_id")
    if job_id:
        lines.append(f"Job: `{job_id}`")
    mutation_id = item.get("mutation_id")
    if mutation_id:
        lines.append(f"Mutation: `{mutation_id}`")
    operator = item.get("operator")
    variable = item.get("variable")
    if operator or variable:
        lines.append(f"Operator: `{operator}`; variable: `{variable}`")
    replacement_for = item.get("replacement_for")
    if replacement_for:
        lines.append(f"Replacement for: `{replacement_for}`")
    lines.append("")
    reason = item.get("reason")
    if reason:
        lines.extend(["Reason:", f"- {reason}", ""])
    success_condition = item.get("success_condition")
    if success_condition:
        lines.extend(["Success condition:", f"- {success_condition}", ""])
    isolation_rule = item.get("isolation_rule")
    if isolation_rule:
        lines.extend(["Isolation rule:", f"- {isolation_rule}", ""])
    guardrail = item.get("guardrail")
    if guardrail:
        lines.extend(["Guardrail:", f"- {guardrail}", ""])
    risk_updates = [str(risk) for risk in item.get("risk_updates", []) or []]
    if risk_updates:
        lines.extend(["Risk updates:", *(f"- `{risk}`" for risk in risk_updates), ""])
    required_evidence = [str(entry) for entry in item.get("required_evidence", []) or []]
    if required_evidence:
        lines.extend(["Required evidence:", *(f"- {entry}" for entry in required_evidence), ""])
    lines.append("Leave `DONE` absent until this action has been executed or reviewed.")
    return lines


def render_novelty_next_wave_shell(result: dict[str, Any]) -> str:
    """Render the next novelty campaign wave as an evidence-preserving shell checklist."""

    plan = result.get("novelty_next_wave_plan") or {}
    waves = plan.get("waves") or []
    plan_json = json.dumps(plan, indent=2, sort_keys=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'AISP_NEXT_WAVE_CWD="${AISP_NEXT_WAVE_CWD:-$(pwd)}"',
        'AISP_NOVELTY_NEXT_WAVE_ROOT="${AISP_NOVELTY_NEXT_WAVE_ROOT:-artifacts/novelty_next_wave/$(date -u +%Y%m%d_%H%M%S)}"',
        'mkdir -p "$AISP_NOVELTY_NEXT_WAVE_ROOT"',
        'echo "Novelty next-wave root: $AISP_NOVELTY_NEXT_WAVE_ROOT"',
        "cat > \"$AISP_NOVELTY_NEXT_WAVE_ROOT/novelty_next_wave_plan.json\" <<'JSON'",
        plan_json,
        "JSON",
        "",
    ]
    if not waves:
        lines.extend(['echo "No novelty next-wave actions found."', "exit 0", ""])
        return "\n".join(lines)

    for wave in waves:
        wave_name = str(wave.get("name") or "wave")
        wave_order = int(wave.get("order") or 0)
        wave_dir_name = f"{wave_order:02d}-{_slug(wave_name, max_len=48)}"
        wave_json = json.dumps(wave, indent=2, sort_keys=True)
        lines.extend(
            [
                f"echo '==> Wave {wave_order}: {wave_name}'",
                f'wave_dir="$AISP_NOVELTY_NEXT_WAVE_ROOT/{wave_dir_name}"',
                'mkdir -p "$wave_dir"',
                "cat > \"$wave_dir/wave.json\" <<'JSON'",
                wave_json,
                "JSON",
                "",
            ]
        )
        for index, item in enumerate(wave.get("items", []) or [], start=1):
            item_id = str(item.get("item_id") or item.get("action_id") or item.get("job_id") or index)
            item_dir_name = f"{index:02d}-{_slug(item_id, max_len=64)}"
            item_json = json.dumps(item, indent=2, sort_keys=True)
            action_lines = _next_wave_action_markdown_lines(wave, item)
            recovery_command = item.get("recovery_command")
            command = item.get("command")
            lines.extend(
                [
                    f"echo '  -> {item_id}'",
                    f'item_dir="$wave_dir/{item_dir_name}"',
                    'mkdir -p "$item_dir"',
                    "cat > \"$item_dir/item.json\" <<'JSON'",
                    item_json,
                    "JSON",
                    "cat > \"$item_dir/action.md\" <<'ACTION'",
                    *action_lines,
                    "ACTION",
                    'if [[ -f "$item_dir/DONE" ]]; then',
                    '  echo "     skip: DONE already exists"',
                    "else",
                ]
            )
            if recovery_command:
                lines.extend(
                    [
                        f"  printf '%s\\n' {_shell_quote(recovery_command)} > \"$item_dir/recovery_command.txt\"",
                        f'  (cd "$AISP_NEXT_WAVE_CWD" && bash -lc {_shell_quote(recovery_command)}) >"$item_dir/recovery_stdout.log" 2>"$item_dir/recovery_stderr.log"',
                    ]
                )
            if command and command != recovery_command:
                lines.extend(
                    [
                        f"  printf '%s\\n' {_shell_quote(command)} > \"$item_dir/command.txt\"",
                        f'  (cd "$AISP_NEXT_WAVE_CWD" && bash -lc {_shell_quote(command)}) >"$item_dir/stdout.log" 2>"$item_dir/stderr.log"',
                    ]
                )
            if recovery_command or command:
                lines.append('  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$item_dir/DONE"')
            else:
                lines.append(
                    '  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$item_dir/MANUAL_ACTION_REQUIRED"'
                )
            lines.extend(["fi", ""])

    lines.extend(['echo "Novelty next-wave script complete."', ""])
    return "\n".join(lines)


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _tail_text(path: Path, *, max_chars: int = 1200) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _diagnostic_signals(stdout_tail: str, stderr_tail: str) -> list[dict[str, str]]:
    text = f"{stdout_tail}\n{stderr_tail}".lower()
    checks = [
        (
            "zymtrace_injection_missing",
            ("cuda_injection64_path", "zymtrace", "injection"),
            "Launch with the Zymtrace CUDA injection library or rerun without the zymtrace tool.",
        ),
        (
            "cuda_out_of_memory",
            ("cuda out of memory", "cublas_status_alloc_failed", "outofmemoryerror"),
            "Reduce batch/sequence size or isolate GPU memory before rerunning the queue job.",
        ),
        (
            "python_import_error",
            ("modulenotfounderror", "importerror", "no module named"),
            "Check the active Python environment and repo-local dependencies before rerunning.",
        ),
        (
            "missing_file",
            ("no such file or directory", "filenotfounderror"),
            "Verify cwd, artifact paths, and generated benchmark files before rerunning.",
        ),
        (
            "correctness_assertion",
            ("assertionerror", "output verification failed", "numerical mismatch"),
            "Treat the run as a correctness failure; do not promote speedup evidence.",
        ),
        (
            "timeout",
            ("timed out", "timeout", "deadline exceeded"),
            "Rerun with an explicit timeout budget or profile the long-running phase first.",
        ),
        (
            "python_exception",
            ("traceback (most recent call last)",),
            "Inspect the Python traceback and rerun only after the exception cause is fixed.",
        ),
    ]
    signals: list[dict[str, str]] = []
    for signal, needles, hint in checks:
        if any(needle in text for needle in needles):
            signals.append({"signal": signal, "hint": hint})
    return signals


def _run_queue_next_actions(jobs: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    failed = [job for job in jobs if job.get("status") == "failed_or_incomplete"]
    if failed:
        signals = failed[0].get("diagnostic_signals") or []
        if signals:
            actions.append(
                f"Fix `{signals[0].get('signal')}` for `{failed[0]['id']}`: {signals[0].get('hint')}"
            )
        else:
            actions.append(f"Inspect failed or incomplete job logs for `{failed[0]['id']}`.")
    blocked = [job for job in jobs if job.get("status") == "blocked_by_dependency"]
    if blocked:
        actions.append(
            f"Complete dependencies for `{blocked[0]['id']}`: {', '.join(blocked[0].get('missing_dependencies', []))}."
        )
    manual = [job for job in jobs if job.get("status") == "manual_review_required"]
    if manual:
        actions.append(
            f"Review evidence in `{manual[0].get('promotion_review')}` and create APPROVED only if all gates pass."
        )
    pending = [job for job in jobs if job.get("status") == "pending"]
    if pending:
        actions.append(f"Run or resume `{pending[0]['id']}`.")
    if not actions and jobs:
        actions.append(
            "All discovered queue jobs are complete or approved; rerun benchmark_opportunities on the new evidence."
        )
    if not jobs:
        actions.append("No job directories with job.json were found under the run queue root.")
    return actions


def summarize_run_queue_root(root: Path) -> dict[str, Any]:
    """Summarize evidence state from a runbook artifact root."""

    root = Path(root)
    job_dirs = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    jobs: list[dict[str, Any]] = []
    completed_ids: set[str] = set()

    for job_dir in job_dirs:
        job = _read_json_file(job_dir / "job.json")
        job_id = str(job.get("id") or job_dir.name)
        if (job_dir / "DONE").exists() or (job_dir / "APPROVED").exists():
            completed_ids.add(job_id)
        stdout_tail = _tail_text(job_dir / "stdout.log")
        stderr_tail = _tail_text(job_dir / "stderr.log")
        jobs.append(
            {
                "id": job_id,
                "lead_id": job.get("lead_id"),
                "lead_type": job.get("lead_type"),
                "target": job.get("target"),
                "stage": job.get("stage"),
                "job_dir": str(job_dir),
                "command": job.get("command"),
                "depends_on": [str(dep) for dep in job.get("depends_on", [])],
                "experiment_playbook_id": job.get("experiment_playbook_id"),
                "promotion_gate": job.get("promotion_gate"),
                "required_evidence": job.get("required_evidence", []),
                "done": (job_dir / "DONE").exists(),
                "approved": (job_dir / "APPROVED").exists(),
                "manual_review_required": (job_dir / "MANUAL_REVIEW_REQUIRED").exists(),
                "stdout_log": str(job_dir / "stdout.log")
                if (job_dir / "stdout.log").exists()
                else None,
                "stderr_log": str(job_dir / "stderr.log")
                if (job_dir / "stderr.log").exists()
                else None,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "diagnostic_signals": _diagnostic_signals(stdout_tail, stderr_tail),
                "promotion_review": (
                    str(job_dir / "promotion_review.md")
                    if (job_dir / "promotion_review.md").exists()
                    else None
                ),
            }
        )

    for job in jobs:
        missing_dependencies = [
            dependency for dependency in job["depends_on"] if dependency not in completed_ids
        ]
        job["missing_dependencies"] = missing_dependencies
        if job["approved"]:
            status = "approved"
        elif job["manual_review_required"]:
            status = "manual_review_required"
        elif job["done"]:
            status = "completed"
        elif missing_dependencies:
            status = "blocked_by_dependency"
        elif job["command"] and (job["stderr_log"] or job["stdout_log"]):
            status = "failed_or_incomplete"
        else:
            status = "pending"
        job["status"] = status

    status_counts = Counter(str(job["status"]) for job in jobs)
    promotion_jobs = [job for job in jobs if job.get("stage") == "promotion_review"]
    approved_promotions = [job for job in promotion_jobs if job["status"] == "approved"]
    manual_promotions = [job for job in promotion_jobs if job["status"] == "manual_review_required"]
    review_jobs = [job for job in jobs if job.get("stage") in {"promotion_review", "manual_review"}]
    approved_reviews = [job for job in review_jobs if job["status"] == "approved"]
    manual_reviews = [job for job in review_jobs if job["status"] == "manual_review_required"]
    failed_jobs = [job for job in jobs if job["status"] == "failed_or_incomplete"]

    return {
        "root": str(root),
        "exists": root.exists(),
        "job_count": len(jobs),
        "status_counts": dict(status_counts),
        "completed_count": status_counts.get("completed", 0),
        "approved_promotion_count": len(approved_promotions),
        "approved_review_count": len(approved_reviews),
        "manual_review_count": len(manual_reviews),
        "failed_or_incomplete_count": len(failed_jobs),
        "jobs": jobs,
        "promotion_summary": {
            "promotion_job_count": len(promotion_jobs),
            "approved": [job["id"] for job in approved_promotions],
            "manual_review_required": [job["id"] for job in manual_promotions],
            "claim_allowed_count": len(approved_promotions),
            "global_rule": "Claims remain blocked unless the corresponding promotion_review job has an APPROVED marker created after manual evidence review.",
        },
        "review_summary": {
            "review_job_count": len(review_jobs),
            "approved": [job["id"] for job in approved_reviews],
            "manual_review_required": [job["id"] for job in manual_reviews],
            "claim_allowed_count": len(approved_reviews),
            "global_rule": "Claims remain blocked unless the corresponding review job has an APPROVED marker created after manual evidence review.",
        },
        "next_actions": _run_queue_next_actions(jobs),
    }


def _job_complete_for_resume(status: str) -> bool:
    return status in {"completed", "approved"}


def _resume_action_for_status(status: str, *, has_missing_dependencies: bool) -> str:
    if status == "approved":
        return "skip_approved_promotion"
    if status == "completed":
        return "skip_completed"
    if status == "manual_review_required":
        return "review_promotion_evidence"
    if status == "failed_or_incomplete":
        return "inspect_logs_before_rerun"
    if has_missing_dependencies:
        return "wait_for_dependencies"
    return "run_next"


def _target_feedback_from_jobs(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_target: dict[str, dict[str, Any]] = {}
    for job in jobs:
        target = str(job.get("target") or "")
        if not target:
            continue
        entry = by_target.setdefault(
            target,
            {
                "target": target,
                "job_count": 0,
                "status_counts": {},
                "completed_job_ids": [],
                "failed_job_ids": [],
                "manual_review_job_ids": [],
                "approved_review_job_ids": [],
                "promotion_status": "not_started",
            },
        )
        entry["job_count"] += 1
        status = str(job.get("evidence_status") or job.get("status") or "unknown")
        entry["status_counts"][status] = int(entry["status_counts"].get(status, 0)) + 1
        if _job_complete_for_resume(status):
            entry["completed_job_ids"].append(job.get("id"))
        if status == "failed_or_incomplete":
            entry["failed_job_ids"].append(job.get("id"))
        if job.get("stage") == "promotion_review":
            if status == "approved":
                entry["approved_review_job_ids"].append(job.get("id"))
                entry["promotion_status"] = "approved"
            elif status == "manual_review_required" and entry["promotion_status"] != "approved":
                entry["manual_review_job_ids"].append(job.get("id"))
                entry["promotion_status"] = "manual_review_required"
    return by_target


def _lead_feedback_from_jobs(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_lead: dict[str, dict[str, Any]] = {}
    for job in jobs:
        lead_id = str(job.get("lead_id") or "")
        if not lead_id:
            continue
        entry = by_lead.setdefault(
            lead_id,
            {
                "lead_id": lead_id,
                "lead_type": job.get("lead_type"),
                "target": job.get("target"),
                "job_count": 0,
                "status_counts": {},
                "completed_job_ids": [],
                "failed_job_ids": [],
                "manual_review_job_ids": [],
                "approved_review_job_ids": [],
                "validation_status": "not_started",
            },
        )
        entry["job_count"] += 1
        status = str(job.get("evidence_status") or job.get("status") or "unknown")
        entry["status_counts"][status] = int(entry["status_counts"].get(status, 0)) + 1
        if _job_complete_for_resume(status):
            entry["completed_job_ids"].append(job.get("id"))
        if status == "failed_or_incomplete":
            entry["failed_job_ids"].append(job.get("id"))
        if job.get("stage") == "manual_review":
            if status == "approved":
                entry["approved_review_job_ids"].append(job.get("id"))
            elif status == "manual_review_required":
                entry["manual_review_job_ids"].append(job.get("id"))

    for entry in by_lead.values():
        status_counts = entry["status_counts"]
        if entry["approved_review_job_ids"]:
            validation_status = "approved_after_manual_review"
        elif entry["manual_review_job_ids"]:
            validation_status = "manual_review_required"
        elif entry["failed_job_ids"]:
            validation_status = "failed_or_incomplete"
        elif entry["job_count"] and entry["job_count"] == sum(
            status_counts.get(status, 0) for status in ("completed", "approved")
        ):
            validation_status = "evidence_complete"
        elif entry["completed_job_ids"]:
            validation_status = "in_progress"
        else:
            validation_status = "not_started"
        entry["validation_status"] = validation_status
    return by_lead


def _audit_required_files(
    stage_contract: dict[str, Any], evidence_job: dict[str, Any] | None
) -> dict[str, Any]:
    required_files = [str(item) for item in stage_contract.get("required_files", []) or []]
    job_dir_value = str((evidence_job or {}).get("job_dir") or "")
    present_files: list[str] = []
    missing_files: list[str] = []
    if job_dir_value:
        job_dir = Path(job_dir_value)
        for filename in required_files:
            if (job_dir / filename).exists():
                present_files.append(filename)
            else:
                missing_files.append(filename)
    else:
        missing_files = list(required_files)
    if not evidence_job:
        status = "missing_job"
    elif missing_files:
        status = "incomplete"
    else:
        status = "complete"
    return {
        "stage": stage_contract.get("stage"),
        "job_id": stage_contract.get("job_id"),
        "job_status": (evidence_job or {}).get("status"),
        "job_dir": job_dir_value or None,
        "required_file_count": len(required_files),
        "present_file_count": len(present_files),
        "missing_file_count": len(missing_files),
        "present_files": present_files,
        "missing_files": missing_files,
        "audit_status": status,
    }


def _build_novelty_evidence_audit_plan(
    novelty_artifact_contract_plan: dict[str, Any],
    novelty_claim_packet_plan: dict[str, Any],
    novelty_queue_summary: dict[str, Any],
) -> dict[str, Any]:
    """Audit actual novelty queue artifacts against the declared contract files."""

    evidence_by_id = {
        str(job.get("id")): job
        for job in novelty_queue_summary.get("jobs", []) or []
        if job.get("id")
    }
    claim_packet_by_lead = {
        str(packet.get("lead_id")): packet
        for packet in novelty_claim_packet_plan.get("lead_packets", []) or []
        if packet.get("lead_id")
    }
    lead_audits: list[dict[str, Any]] = []
    for contract in novelty_artifact_contract_plan.get("lead_contracts", []) or []:
        lead_id = str(contract.get("lead_id") or "")
        if not lead_id:
            continue
        stage_audits = [
            _audit_required_files(stage_contract, evidence_by_id.get(str(stage_contract.get("job_id"))))
            for stage_contract in contract.get("stage_contracts", []) or []
            if isinstance(stage_contract, dict)
        ]
        required_count = sum(int(item.get("required_file_count") or 0) for item in stage_audits)
        present_count = sum(int(item.get("present_file_count") or 0) for item in stage_audits)
        missing_count = sum(int(item.get("missing_file_count") or 0) for item in stage_audits)
        missing_stage_count = sum(
            1 for item in stage_audits if item.get("audit_status") != "complete"
        )
        review_stage = next(
            (item for item in stage_audits if item.get("stage") == "manual_review"),
            {},
        )
        claim_packet = claim_packet_by_lead.get(lead_id, {})
        promotion_blockers: list[str] = []
        for stage_audit in stage_audits:
            missing_files = stage_audit.get("missing_files") or []
            if missing_files:
                stage = stage_audit.get("stage")
                preview = ", ".join(str(item) for item in missing_files[:4])
                if len(missing_files) > 4:
                    preview = f"{preview}, ..."
                promotion_blockers.append(f"{stage} missing required files: {preview}")
        if claim_packet and review_stage.get("audit_status") != "complete":
            promotion_blockers.append("claim packet remains blocked until review artifacts are complete")
        approved_jobs = [
            item
            for item in stage_audits
            if str(item.get("job_status") or "") == "approved"
        ]
        if missing_count == 0 and approved_jobs:
            audit_status = "approved_and_audited"
        elif missing_count == 0 and stage_audits:
            audit_status = "evidence_packet_complete"
        elif present_count:
            audit_status = "evidence_packet_incomplete"
        else:
            audit_status = "not_started"
        if approved_jobs and missing_count:
            audit_status = "approved_with_missing_artifacts"
        lead_audits.append(
            {
                "lead_id": lead_id,
                "lead_type": contract.get("lead_type"),
                "title": contract.get("title"),
                "target": contract.get("target"),
                "contract_id": contract.get("contract_id"),
                "claim_packet_id": claim_packet.get("packet_id"),
                "audit_status": audit_status,
                "required_file_count": required_count,
                "present_file_count": present_count,
                "missing_file_count": missing_count,
                "missing_stage_count": missing_stage_count,
                "stage_audits": stage_audits,
                "promotion_blockers": promotion_blockers,
            }
        )
    status_counts = Counter(str(item.get("audit_status") or "unknown") for item in lead_audits)
    return {
        "lead_count": len(lead_audits),
        "status_counts": dict(status_counts),
        "required_file_count": sum(
            int(item.get("required_file_count") or 0) for item in lead_audits
        ),
        "present_file_count": sum(int(item.get("present_file_count") or 0) for item in lead_audits),
        "missing_file_count": sum(int(item.get("missing_file_count") or 0) for item in lead_audits),
        "policy": (
            "Audit novelty validation queue artifacts against the declared artifact contract. DONE and APPROVED markers are not sufficient when required files are missing."
        ),
        "lead_audits": lead_audits,
    }


def _recovery_action_for_signal(job: dict[str, Any], signal: dict[str, str]) -> dict[str, Any]:
    job_id = str(job.get("id") or "job")
    signal_name = str(signal.get("signal") or "unknown_failure")
    base = {
        "action_id": _slug(f"recover-{job_id}-{signal_name}", max_len=96),
        "lead_id": job.get("lead_id"),
        "lead_type": job.get("lead_type"),
        "target": job.get("target"),
        "job_id": job_id,
        "stage": job.get("stage"),
        "issue_type": signal_name,
        "diagnostic_hint": signal.get("hint"),
        "job_dir": job.get("existing_job_dir") or job.get("job_dir"),
        "stdout_log": job.get("stdout_log"),
        "stderr_log": job.get("stderr_log"),
        "original_command": job.get("command"),
        "blocks_promotion": True,
    }
    if signal_name == "zymtrace_injection_missing":
        base.update(
            {
                "recovery_command": 'test -n "${CUDA_INJECTION64_PATH:-${ZYMTRACE_CUDA_INJECTION64_PATH:-}}"',
                "rerun_after_recovery": job.get("command"),
                "required_evidence": [
                    "profiler preflight manifest records CUDA injection path",
                    "zymtrace_launch_manifest.json is present or zymtrace is explicitly skipped",
                    "stderr no longer reports missing CUDA injection",
                ],
                "unblock_condition": "CUDA injection resolves before rerunning the failed validation job.",
            }
        )
    elif signal_name == "cuda_out_of_memory":
        base.update(
            {
                "recovery_command": "nvidia-smi",
                "rerun_after_recovery": job.get("command"),
                "required_evidence": [
                    "GPU memory snapshot before rerun",
                    "smaller batch, sequence, or isolated GPU note",
                    "rerun logs show no CUDA OOM",
                ],
                "unblock_condition": "Rerun under a memory envelope that leaves headroom for profiling.",
            }
        )
    elif signal_name == "python_import_error":
        base.update(
            {
                "recovery_command": "python -m pip check",
                "rerun_after_recovery": job.get("command"),
                "required_evidence": [
                    "active Python executable and PYTHONPATH are recorded",
                    "missing dependency or import path is fixed",
                    "rerun logs show import succeeds",
                ],
                "unblock_condition": "The repo-local environment can import the benchmark target.",
            }
        )
    elif signal_name == "missing_file":
        base.update(
            {
                "recovery_command": "pwd && ls -la",
                "rerun_after_recovery": job.get("command"),
                "required_evidence": [
                    "cwd and expected benchmark/artifact paths are recorded",
                    "missing file path exists or command is corrected",
                    "rerun writes the declared artifact contract files",
                ],
                "unblock_condition": "The job command resolves every declared path from the runbook cwd.",
            }
        )
    elif signal_name == "correctness_assertion":
        base.update(
            {
                "recovery_command": None,
                "rerun_after_recovery": None,
                "required_evidence": [
                    "correctness failure is explained before any performance claim",
                    "candidate is narrowed or reverted",
                    "fresh control/candidate output verification passes",
                ],
                "unblock_condition": "Correctness passes before timing or profiler evidence is considered.",
            }
        )
    elif signal_name == "timeout":
        base.update(
            {
                "recovery_command": "python -m cli.aisp bench list --json",
                "rerun_after_recovery": job.get("command"),
                "required_evidence": [
                    "timeout budget and long-running phase are recorded",
                    "rerun uses a bounded profile or smaller validation slice",
                    "logs prove the job completed or was intentionally split",
                ],
                "unblock_condition": "The validation slice completes within an explicit timeout budget.",
            }
        )
    else:
        base.update(
            {
                "recovery_command": None,
                "rerun_after_recovery": job.get("command"),
                "required_evidence": [
                    "stdout/stderr traceback is reviewed",
                    "root cause note is attached to the job directory",
                    "rerun succeeds before downstream dependencies continue",
                ],
                "unblock_condition": "The failed job is rerun successfully or the lead is returned to backlog.",
            }
        )
    return base


def _build_novelty_recovery_plan(
    jobs: list[dict[str, Any]],
    evidence_audit_plan: dict[str, Any],
) -> dict[str, Any]:
    """Turn failed jobs and missing contract artifacts into concrete recovery actions."""

    actions: list[dict[str, Any]] = []
    for job in jobs:
        if str(job.get("evidence_status") or job.get("status") or "") != "failed_or_incomplete":
            continue
        signals = [
            signal
            for signal in job.get("diagnostic_signals", []) or []
            if isinstance(signal, dict)
        ]
        if not signals:
            signals = [
                {
                    "signal": "unknown_failure",
                    "hint": "Inspect stdout/stderr and rerun only after the failure is explained.",
                }
            ]
        for signal in signals:
            actions.append(_recovery_action_for_signal(job, signal))

    for lead_audit in evidence_audit_plan.get("lead_audits", []) or []:
        missing_count = int(lead_audit.get("missing_file_count") or 0)
        if missing_count <= 0:
            continue
        stage_audits = [
            stage
            for stage in lead_audit.get("stage_audits", []) or []
            if isinstance(stage, dict) and int(stage.get("missing_file_count") or 0) > 0
        ]
        for stage in stage_audits[:2]:
            missing_files = [str(item) for item in stage.get("missing_files", []) or []]
            actions.append(
                {
                    "action_id": _slug(
                        f"recover-{lead_audit.get('lead_id')}-{stage.get('stage')}-missing-artifacts",
                        max_len=96,
                    ),
                    "lead_id": lead_audit.get("lead_id"),
                    "lead_type": lead_audit.get("lead_type"),
                    "target": lead_audit.get("target"),
                    "job_id": stage.get("job_id"),
                    "stage": stage.get("stage"),
                    "issue_type": "missing_artifact_contract_files",
                    "job_dir": stage.get("job_dir"),
                    "missing_files": missing_files,
                    "recovery_command": None,
                    "rerun_after_recovery": None,
                    "required_evidence": [
                        "required files are written under the declared job directory",
                        "not-applicable notes are explicit for skipped optional artifacts",
                        "opportunity-run-summary no longer reports missing contract files",
                    ],
                    "unblock_condition": "All required artifact contract files are present or explicitly waived before manual approval.",
                    "blocks_promotion": True,
                }
            )

    actions_by_lead: dict[str, list[str]] = {}
    for action in actions:
        lead_id = str(action.get("lead_id") or "unknown")
        actions_by_lead.setdefault(lead_id, []).append(str(action.get("action_id")))
    return {
        "action_count": len(actions),
        "blocking_action_count": sum(1 for action in actions if action.get("blocks_promotion")),
        "lead_count": len(actions_by_lead),
        "actions_by_lead": actions_by_lead,
        "next_action": actions[0] if actions else None,
        "policy": (
            "Recover novelty validation by fixing failed jobs and missing contract artifacts before rerunning downstream work or approving claims."
        ),
        "actions": actions,
    }


def _build_novelty_adaptive_decision_plan(
    novelty_budget_plan: dict[str, Any],
    novelty_validation_plan: dict[str, Any],
    lead_feedback: dict[str, dict[str, Any]],
    recovery_plan: dict[str, Any],
) -> dict[str, Any]:
    """Decide whether each selected novelty lead should run, recover, review, or yield a slot."""

    ready_job_ids = {
        str(job_id) for job_id in (novelty_validation_plan.get("resume_plan") or {}).get(
            "ready_job_ids", []
        )
    }
    jobs_by_lead: dict[str, list[dict[str, Any]]] = {}
    for job in novelty_validation_plan.get("jobs", []) or []:
        lead_id = str(job.get("lead_id") or "")
        if lead_id:
            jobs_by_lead.setdefault(lead_id, []).append(job)

    recovery_actions_by_lead: dict[str, list[dict[str, Any]]] = {}
    for action in recovery_plan.get("actions", []) or []:
        lead_id = str(action.get("lead_id") or "")
        if lead_id:
            recovery_actions_by_lead.setdefault(lead_id, []).append(action)

    selected_decisions: list[dict[str, Any]] = []
    blocked_decisions: list[dict[str, Any]] = []
    for lead in novelty_budget_plan.get("selected", []) or []:
        lead_id = str(lead.get("lead_id") or "")
        feedback = lead_feedback.get(lead_id, {})
        lead_jobs = jobs_by_lead.get(lead_id, [])
        ready_jobs = [job for job in lead_jobs if str(job.get("id") or "") in ready_job_ids]
        actions = recovery_actions_by_lead.get(lead_id, [])
        failure_actions = [
            action
            for action in actions
            if action.get("issue_type") != "missing_artifact_contract_files"
            or feedback.get("failed_job_ids")
        ]
        artifact_actions = [
            action for action in actions if action.get("issue_type") == "missing_artifact_contract_files"
        ]
        validation_status = str(feedback.get("validation_status") or "not_started")
        audit_status = str(feedback.get("evidence_audit_status") or "not_started")

        if validation_status == "approved_after_manual_review":
            disposition = "claim_ready"
            slot_state = "complete"
            next_step = "Use the approved claim packet; do not rerun unless new evidence invalidates it."
        elif validation_status == "manual_review_required":
            disposition = "finish_manual_review"
            slot_state = "review"
            next_step = "Complete the manual review checklist and claim packet before APPROVED."
        elif failure_actions:
            disposition = "recover_failed_evidence"
            slot_state = "blocked"
            next_step = str(failure_actions[0].get("unblock_condition") or "")
        elif ready_jobs:
            disposition = "run_next_validation_job"
            slot_state = "active"
            next_step = str(ready_jobs[0].get("command") or "Run the ready manual step.")
        elif validation_status == "evidence_complete" and audit_status in {
            "evidence_packet_complete",
            "approved_and_audited",
        }:
            disposition = "prepare_manual_review"
            slot_state = "review"
            next_step = "Open the manual review job and verify the claim packet."
        elif artifact_actions and validation_status != "not_started":
            disposition = "repair_artifact_packet"
            slot_state = "blocked"
            next_step = str(artifact_actions[0].get("unblock_condition") or "")
        elif validation_status == "in_progress":
            disposition = "wait_for_dependencies"
            slot_state = "active"
            next_step = "Complete upstream dependencies before the next novelty stage can run."
        else:
            disposition = "start_validation"
            slot_state = "open"
            next_step = "Run the first control job for this selected novelty lead."

        decision = {
            "lead_id": lead_id,
            "lead_type": lead.get("lead_type"),
            "title": lead.get("title"),
            "target": lead.get("target"),
            "slot_state": slot_state,
            "disposition": disposition,
            "validation_status": validation_status,
            "evidence_audit_status": audit_status,
            "ready_job_ids": [job.get("id") for job in ready_jobs],
            "recovery_action_ids": [action.get("action_id") for action in actions],
            "next_step": next_step,
        }
        selected_decisions.append(decision)
        if slot_state == "blocked":
            blocked_decisions.append(decision)

    replacement_candidates: list[dict[str, Any]] = []
    for index, card in enumerate(novelty_budget_plan.get("backlog", []) or []):
        if index >= len(blocked_decisions):
            break
        blocked = blocked_decisions[index]
        replacement_candidates.append(
            {
                "lead_id": card.get("lead_id"),
                "lead_type": card.get("lead_type"),
                "title": card.get("title"),
                "target": card.get("target"),
                "expected_value_score": card.get("expected_value_score"),
                "risk_flags": list(card.get("risk_flags", []) or []),
                "replacement_for": blocked.get("lead_id"),
                "replacement_reason": (
                    "Use as a parallel backup while the selected lead is blocked on recovery."
                ),
                "command": card.get("command"),
            }
        )

    state_counts = Counter(str(item.get("slot_state") or "unknown") for item in selected_decisions)
    return {
        "selected_count": len(selected_decisions),
        "blocked_count": len(blocked_decisions),
        "replacement_candidate_count": len(replacement_candidates),
        "slot_state_counts": dict(state_counts),
        "policy": (
            "Adapt the novelty portfolio after validation feedback: run ready jobs, repair failed evidence, finish reviews, and use backlog leads only as isolated backups while blocked work recovers."
        ),
        "selected_decisions": selected_decisions,
        "replacement_candidates": replacement_candidates,
    }


def _learning_risk_updates(actions: Iterable[dict[str, Any]]) -> list[str]:
    updates: list[str] = []
    for action in actions:
        issue_type = str(action.get("issue_type") or "")
        if issue_type == "zymtrace_injection_missing":
            updates.extend(["zymtrace_injection_required", "profiler_preflight_required"])
        elif issue_type == "cuda_out_of_memory":
            updates.extend(["memory_envelope_required", "smaller_validation_slice"])
        elif issue_type == "python_import_error":
            updates.extend(["environment_reproducibility_required"])
        elif issue_type == "missing_file":
            updates.extend(["cwd_and_path_preflight_required"])
        elif issue_type == "correctness_assertion":
            updates.extend(["correctness_blocker"])
        elif issue_type == "missing_artifact_contract_files":
            updates.extend(["artifact_contract_completion_required"])
        elif issue_type:
            updates.extend(["failure_root_cause_required"])
    return list(dict.fromkeys(updates))


def _build_novelty_learning_plan(
    novelty_budget_plan: dict[str, Any],
    adaptive_decision_plan: dict[str, Any],
    recovery_plan: dict[str, Any],
    evidence_audit_plan: dict[str, Any],
) -> dict[str, Any]:
    """Translate validation feedback into explicit rerank and risk-learning guidance."""

    actions_by_lead: dict[str, list[dict[str, Any]]] = {}
    for action in recovery_plan.get("actions", []) or []:
        lead_id = str(action.get("lead_id") or "")
        if lead_id:
            actions_by_lead.setdefault(lead_id, []).append(action)
    audit_by_lead = {
        str(item.get("lead_id")): item
        for item in evidence_audit_plan.get("lead_audits", []) or []
        if item.get("lead_id")
    }

    lead_adjustments: list[dict[str, Any]] = []
    for decision in adaptive_decision_plan.get("selected_decisions", []) or []:
        lead_id = str(decision.get("lead_id") or "")
        disposition = str(decision.get("disposition") or "")
        actions = actions_by_lead.get(lead_id, [])
        audit = audit_by_lead.get(lead_id, {})
        if disposition == "recover_failed_evidence":
            learning_state = "infrastructure_or_validation_blocked"
            expected_value_adjustment = -3.0
            rerank_action = "hold_selected_slot_until_recovery"
            next_validation_focus = "Resolve failed job diagnostics before spending downstream profile/review work."
        elif disposition == "repair_artifact_packet":
            learning_state = "artifact_hygiene_blocked"
            expected_value_adjustment = -1.5
            rerank_action = "repair_artifacts_before_review"
            next_validation_focus = "Complete artifact contract files before manual review."
        elif disposition in {"finish_manual_review", "prepare_manual_review"}:
            learning_state = "review_ready"
            expected_value_adjustment = 1.0
            rerank_action = "prioritize_manual_review"
            next_validation_focus = "Resolve claim packet and reviewer evidence before more runs."
        elif disposition == "claim_ready":
            learning_state = "validated_claim_ready"
            expected_value_adjustment = 4.0
            rerank_action = "extract_reusable_pattern"
            next_validation_focus = "Use the approved claim packet to seed transfer or compound hypotheses."
        elif disposition == "run_next_validation_job":
            learning_state = "evidence_in_progress"
            expected_value_adjustment = 0.5
            rerank_action = "continue_selected_validation"
            next_validation_focus = "Run the ready job and preserve artifact contract files."
        else:
            learning_state = "not_started"
            expected_value_adjustment = 0.0
            rerank_action = "start_or_keep_in_queue"
            next_validation_focus = "Capture first control evidence."
        risk_updates = _learning_risk_updates(actions)
        if audit and int(audit.get("missing_file_count") or 0) > 0:
            risk_updates = list(
                dict.fromkeys([*risk_updates, "artifact_contract_completion_required"])
            )
        lead_adjustments.append(
            {
                "lead_id": lead_id,
                "lead_type": decision.get("lead_type"),
                "title": decision.get("title"),
                "target": decision.get("target"),
                "learning_state": learning_state,
                "source_disposition": disposition,
                "expected_value_adjustment": expected_value_adjustment,
                "risk_updates": risk_updates,
                "rerank_action": rerank_action,
                "next_validation_focus": next_validation_focus,
                "recovery_action_ids": list(decision.get("recovery_action_ids", []) or []),
                "audit_status": audit.get("audit_status"),
                "missing_file_count": audit.get("missing_file_count", 0),
            }
        )

    replacement_candidates = adaptive_decision_plan.get("replacement_candidates", []) or []
    backup_learning = [
        {
            "lead_id": item.get("lead_id"),
            "replacement_for": item.get("replacement_for"),
            "learning_state": "backup_candidate",
            "rerank_action": "promote_as_parallel_backup_only",
            "reason": item.get("replacement_reason"),
            "command": item.get("command"),
        }
        for item in replacement_candidates
    ]
    adjustment_counts = Counter(str(item.get("learning_state") or "unknown") for item in lead_adjustments)
    blocked_count = sum(
        1
        for item in lead_adjustments
        if item.get("learning_state")
        in {"infrastructure_or_validation_blocked", "artifact_hygiene_blocked"}
    )
    return {
        "lead_count": len(lead_adjustments),
        "blocked_learning_count": blocked_count,
        "backup_candidate_count": len(backup_learning),
        "learning_state_counts": dict(adjustment_counts),
        "policy": (
            "Use validation feedback as learning input for the next novelty batch. Apply score and risk adjustments explicitly; do not auto-promote claims from advisory feedback."
        ),
        "lead_adjustments": lead_adjustments,
        "backup_learning": backup_learning,
        "portfolio_guidance": {
            "continue_count": sum(
                1
                for item in lead_adjustments
                if item.get("rerank_action") == "continue_selected_validation"
            ),
            "recover_count": sum(
                1
                for item in lead_adjustments
                if item.get("rerank_action") == "hold_selected_slot_until_recovery"
            ),
            "review_count": sum(
                1
                for item in lead_adjustments
                if item.get("rerank_action") == "prioritize_manual_review"
            ),
            "backup_count": len(backup_learning),
        },
    }


def _wave_record(
    name: str,
    purpose: str,
    items: list[dict[str, Any]],
    *,
    order: int,
    depends_on_wave: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "order": order,
        "purpose": purpose,
        "depends_on_wave": depends_on_wave,
        "item_count": len(items),
        "items": items,
    }


def _build_novelty_next_wave_plan(
    novelty_validation_plan: dict[str, Any],
    adaptive_decision_plan: dict[str, Any],
    recovery_plan: dict[str, Any],
    learning_plan: dict[str, Any],
    harvest_plan: dict[str, Any] | None = None,
    novelty_mutation_budget_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert feedback plans into the next concrete novelty campaign wave."""

    jobs_by_id = {
        str(job.get("id")): job
        for job in novelty_validation_plan.get("jobs", []) or []
        if job.get("id")
    }
    jobs_by_lead: dict[str, list[dict[str, Any]]] = {}
    for job in novelty_validation_plan.get("jobs", []) or []:
        lead_id = str(job.get("lead_id") or "")
        if lead_id:
            jobs_by_lead.setdefault(lead_id, []).append(job)

    actions_by_id = {
        str(action.get("action_id")): action
        for action in recovery_plan.get("actions", []) or []
        if action.get("action_id")
    }

    recover_items: list[dict[str, Any]] = []
    continue_items: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    blocked_mutation_lead_ids: set[str] = set()

    for decision in adaptive_decision_plan.get("selected_decisions", []) or []:
        lead_id = str(decision.get("lead_id") or "")
        disposition = str(decision.get("disposition") or "")
        lead_jobs = jobs_by_lead.get(lead_id, [])

        if disposition in {"recover_failed_evidence", "repair_artifact_packet"}:
            blocked_mutation_lead_ids.add(lead_id)
            for action_id in decision.get("recovery_action_ids", []) or []:
                action = actions_by_id.get(str(action_id))
                if not action:
                    continue
                recover_items.append(
                    {
                        "item_id": _slug(f"wave-recover-{action_id}", max_len=96),
                        "lead_id": lead_id,
                        "lead_type": decision.get("lead_type"),
                        "title": decision.get("title"),
                        "target": decision.get("target") or action.get("target"),
                        "source": "novelty_recovery_plan",
                        "disposition": disposition,
                        "action_id": action.get("action_id"),
                        "job_id": action.get("job_id"),
                        "stage": action.get("stage"),
                        "issue_type": action.get("issue_type"),
                        "recovery_command": action.get("recovery_command"),
                        "command": action.get("rerun_after_recovery")
                        or action.get("recovery_command"),
                        "reason": action.get("diagnostic_hint") or decision.get("next_step"),
                        "success_condition": action.get("unblock_condition"),
                        "required_evidence": list(action.get("required_evidence", []) or []),
                    }
                )
            continue

        if disposition in {"run_next_validation_job", "start_validation"}:
            ready_job_ids = [str(job_id) for job_id in decision.get("ready_job_ids", [])]
            if not ready_job_ids and disposition == "start_validation":
                ready_job_ids = [
                    str(job.get("id"))
                    for job in lead_jobs
                    if not job.get("depends_on") and str(job.get("id") or "")
                ][:1]
            for job_id in ready_job_ids[:1]:
                job = jobs_by_id.get(job_id, {})
                continue_items.append(
                    {
                        "item_id": _slug(f"wave-continue-{job_id}", max_len=96),
                        "lead_id": lead_id,
                        "lead_type": decision.get("lead_type"),
                        "title": decision.get("title"),
                        "target": decision.get("target") or job.get("target"),
                        "source": "novelty_validation_plan",
                        "disposition": disposition,
                        "job_id": job_id,
                        "stage": job.get("stage"),
                        "command": job.get("command"),
                        "reason": decision.get("next_step"),
                        "success_condition": job.get("evidence_gate")
                        or "Write the declared artifact contract files before downstream work.",
                    }
                )
            continue

        if disposition in {"finish_manual_review", "prepare_manual_review", "claim_ready"}:
            review_job = next(
                (job for job in lead_jobs if job.get("stage") == "manual_review"),
                {},
            )
            review_items.append(
                {
                    "item_id": _slug(f"wave-review-{lead_id}", max_len=96),
                    "lead_id": lead_id,
                    "lead_type": decision.get("lead_type"),
                    "title": decision.get("title"),
                    "target": decision.get("target") or review_job.get("target"),
                    "source": "novelty_validation_plan",
                    "disposition": disposition,
                    "job_id": review_job.get("id"),
                    "stage": "manual_review",
                    "command": review_job.get("command"),
                    "reason": decision.get("next_step"),
                    "success_condition": (
                        "Manual review records APPROVED with complete claim packet evidence."
                    ),
                }
            )

    backup_items = [
        {
            "item_id": _slug(f"wave-backup-{item.get('lead_id')}", max_len=96),
            "lead_id": item.get("lead_id"),
            "lead_type": item.get("lead_type"),
            "title": item.get("title"),
            "target": item.get("target"),
            "source": "novelty_budget_plan.backlog",
            "replacement_for": item.get("replacement_for"),
            "command": item.get("command"),
            "reason": item.get("replacement_reason"),
            "success_condition": (
                "Run as an isolated backup only; do not merge evidence into the blocked lead."
            ),
        }
        for item in adaptive_decision_plan.get("replacement_candidates", []) or []
    ]

    learning_items = []
    for item in learning_plan.get("lead_adjustments", []) or []:
        learning_items.append(
            {
                "item_id": _slug(f"wave-learn-{item.get('lead_id')}", max_len=96),
                "lead_id": item.get("lead_id"),
                "lead_type": item.get("lead_type"),
                "title": item.get("title"),
                "target": item.get("target"),
                "source": "novelty_learning_plan",
                "learning_state": item.get("learning_state"),
                "rerank_action": item.get("rerank_action"),
                "expected_value_adjustment": item.get("expected_value_adjustment"),
                "risk_updates": list(item.get("risk_updates", []) or []),
                "reason": item.get("next_validation_focus"),
                "success_condition": (
                    "Next rerank records the score/risk adjustment before selecting more leads."
                ),
            }
        )
    for item in learning_plan.get("backup_learning", []) or []:
        learning_items.append(
            {
                "item_id": _slug(f"wave-learn-backup-{item.get('lead_id')}", max_len=96),
                "lead_id": item.get("lead_id"),
                "source": "novelty_learning_plan.backup_learning",
                "replacement_for": item.get("replacement_for"),
                "learning_state": item.get("learning_state"),
                "rerank_action": item.get("rerank_action"),
                "command": item.get("command"),
                "reason": item.get("reason"),
                "success_condition": (
                    "Backup remains isolated unless the blocked selected lead is abandoned."
                ),
            }
        )

    harvest_followup_items = [
        {
            "item_id": _slug(f"wave-harvest-{item.get('followup_id')}", max_len=96),
            "lead_id": item.get("source_lead_id"),
            "target": item.get("target"),
            "source": "novelty_harvest_plan.followup_experiments",
            "source_pattern_id": item.get("source_pattern_id"),
            "followup_id": item.get("followup_id"),
            "followup_type": item.get("followup_type"),
            "command": item.get("command"),
            "reason": item.get("reason"),
            "success_condition": item.get("success_condition"),
            "claim_constraints": list(item.get("claim_constraints", []) or []),
        }
        for item in (harvest_plan or {}).get("followup_experiments", []) or []
    ]

    mutation_items: list[dict[str, Any]] = []
    for item in (novelty_mutation_budget_plan or {}).get("selected", []) or []:
        lead_id = str(item.get("lead_id") or "")
        if lead_id in blocked_mutation_lead_ids:
            continue
        mutation_items.append(
            {
                "item_id": _slug(f"wave-mutation-{item.get('mutation_id')}", max_len=96),
                "lead_id": item.get("lead_id"),
                "lead_type": item.get("lead_type"),
                "title": item.get("title"),
                "target": item.get("target"),
                "source": "novelty_mutation_budget_plan.selected",
                "mutation_id": item.get("mutation_id"),
                "operator": item.get("operator"),
                "variable": item.get("variable"),
                "command": item.get("command"),
                "reason": item.get("selection_reason"),
                "success_condition": (
                    "Mutation writes isolated evidence for the expected signal without changing undeclared variables."
                ),
                "isolation_rule": item.get("isolation_rule"),
                "guardrail": item.get("guardrail"),
                "required_evidence": list(item.get("required_evidence", []) or []),
                "risk_updates": list(item.get("risk_flags", []) or []),
            }
        )

    mutation_depends_on = None
    if review_items:
        mutation_depends_on = "finish_manual_reviews"
    elif continue_items:
        mutation_depends_on = "continue_active_validation"
    elif recover_items:
        mutation_depends_on = "recover_blocked_leads"

    wave_specs = [
        (
            "recover_blocked_leads",
            "Unblock failed jobs and missing artifact packets before spending downstream work.",
            recover_items,
            None,
        ),
        (
            "continue_active_validation",
            "Run the next ready validation job for selected leads with clean dependencies.",
            continue_items,
            None,
        ),
        (
            "finish_manual_reviews",
            "Complete review and claim-packet work for evidence that has reached review state.",
            review_items,
            "continue_active_validation" if continue_items else None,
        ),
        (
            "run_selected_mutations",
            "Run budgeted one-variable mutations as isolated expansion probes after parent-lead dependencies are clean.",
            mutation_items,
            mutation_depends_on,
        ),
        (
            "activate_backups",
            "Use backlog leads as parallel backups while blocked selected leads recover.",
            backup_items,
            "recover_blocked_leads" if recover_items else None,
        ),
        (
            "run_harvest_followups",
            "Run fresh evidence jobs spawned from approved-and-audited novelty patterns.",
            harvest_followup_items,
            "finish_manual_reviews" if review_items else None,
        ),
        (
            "apply_learning_before_rerank",
            "Apply rerank and risk-learning updates before the next novelty selection pass.",
            learning_items,
            None,
        ),
    ]
    waves: list[dict[str, Any]] = []
    for name, purpose, items, depends_on in wave_specs:
        if not items:
            continue
        waves.append(
            _wave_record(
                name,
                purpose,
                items,
                order=len(waves) + 1,
                depends_on_wave=depends_on,
            )
        )
    action_count = sum(int(wave.get("item_count") or 0) for wave in waves)
    next_action = waves[0]["items"][0] if waves and waves[0].get("items") else None
    return {
        "wave_count": len(waves),
        "action_count": action_count,
        "wave_item_counts": {
            str(wave.get("name")): int(wave.get("item_count") or 0) for wave in waves
        },
        "first_wave": waves[0]["name"] if waves else None,
        "next_action": next_action,
        "policy": (
            "Execute novelty work in explicit waves after feedback: recover blockers, continue clean validation jobs, finish reviews, run selected one-variable mutations, run isolated backups, run harvested follow-ups, then apply learning before reranking."
        ),
        "waves": waves,
    }


def _harvest_followup_type(lead_type: str) -> str:
    if lead_type == "frontier_probe":
        return "deep_dive_after_first_evidence"
    if lead_type == "source_transfer":
        return "recipient_transfer_validation"
    if lead_type == "compound_primitive":
        return "support_target_compound_check"
    if lead_type == "coverage_gap":
        return "coverage_gap_replication"
    if lead_type == "cross_lane_bridge":
        return "bridge_replication"
    return "adjacent_target_validation"


def _build_novelty_harvest_plan(
    novelty_queue: dict[str, Any],
    novelty_validation_plan: dict[str, Any],
    novelty_claim_packet_plan: dict[str, Any],
    adaptive_decision_plan: dict[str, Any],
    learning_plan: dict[str, Any],
    lead_feedback: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Harvest approved novelty evidence into reusable patterns and follow-up leads."""

    lead_by_id = {
        str(lead.get("lead_id")): lead
        for lead in novelty_queue.get("leads", []) or []
        if lead.get("lead_id")
    }
    claim_packet_by_lead = {
        str(packet.get("lead_id")): packet
        for packet in novelty_claim_packet_plan.get("lead_packets", []) or []
        if packet.get("lead_id")
    }
    learning_by_lead = {
        str(item.get("lead_id")): item
        for item in learning_plan.get("lead_adjustments", []) or []
        if item.get("lead_id")
    }
    jobs_by_lead: dict[str, list[dict[str, Any]]] = {}
    for job in novelty_validation_plan.get("jobs", []) or []:
        lead_id = str(job.get("lead_id") or "")
        if lead_id:
            jobs_by_lead.setdefault(lead_id, []).append(job)

    patterns: list[dict[str, Any]] = []
    followups: list[dict[str, Any]] = []
    blocked_harvests: list[dict[str, Any]] = []

    for decision in adaptive_decision_plan.get("selected_decisions", []) or []:
        lead_id = str(decision.get("lead_id") or "")
        if not lead_id or decision.get("disposition") != "claim_ready":
            continue
        audit_status = str(decision.get("evidence_audit_status") or "")
        if audit_status != "approved_and_audited":
            blocked_harvests.append(
                {
                    "lead_id": lead_id,
                    "title": decision.get("title"),
                    "target": decision.get("target"),
                    "audit_status": audit_status,
                    "blocker": (
                        "Approved review exists, but the artifact audit is not complete enough to harvest as a reusable pattern."
                    ),
                }
            )
            continue

        lead = lead_by_id.get(lead_id, {})
        feedback = lead_feedback.get(lead_id, {})
        learning = learning_by_lead.get(lead_id, {})
        claim_packet = claim_packet_by_lead.get(lead_id, {})
        source_target = str(decision.get("target") or lead.get("target") or "")
        lead_type = str(decision.get("lead_type") or lead.get("lead_type") or "unknown")
        pattern_id = _slug(f"harvest-{lead_id}", max_len=96)
        evidence_job_ids = list(
            dict.fromkeys(
                [
                    *[str(job_id) for job_id in feedback.get("completed_job_ids", []) or []],
                    *[
                        str(job_id)
                        for job_id in feedback.get("approved_review_job_ids", []) or []
                    ],
                ]
            )
        )
        pattern = {
            "pattern_id": pattern_id,
            "source_lead_id": lead_id,
            "source_lead_type": lead_type,
            "source_target": source_target,
            "title": decision.get("title") or lead.get("title"),
            "pattern_summary": lead.get("why")
            or f"Approved novelty claim from {decision.get('title') or lead_id}.",
            "claim_packet_id": claim_packet.get("packet_id"),
            "allowed_claim_scope": claim_packet.get("allowed_claim_scope"),
            "disallowed_claims": list(claim_packet.get("disallowed_claims", []) or []),
            "evidence_job_ids": evidence_job_ids,
            "learning_state": learning.get("learning_state"),
            "reuse_rule": (
                "Treat this as a reusable hypothesis only after each recipient reruns control, candidate, profile, and review evidence under the same claim boundary."
            ),
        }
        patterns.append(pattern)

        related_targets = [
            str(target)
            for target in lead.get("related_targets", []) or []
            if target and str(target) != source_target
        ]
        if lead_type == "frontier_probe" and source_target:
            related_targets = [source_target, *related_targets]
        related_targets = list(dict.fromkeys(related_targets))
        followup_type = _harvest_followup_type(lead_type)
        if not related_targets:
            followups.append(
                {
                    "followup_id": _slug(f"{pattern_id}-mine-related-targets", max_len=96),
                    "source_pattern_id": pattern_id,
                    "source_lead_id": lead_id,
                    "followup_type": "mine_related_targets",
                    "target": None,
                    "command": None,
                    "reason": "No related targets were attached to the approved lead; mine the target catalog before attempting transfer.",
                    "success_condition": (
                        "At least one recipient target is selected with the same workload and guardrail contract."
                    ),
                    "claim_constraints": [
                        "Do not reuse the approved claim text for a new target without fresh evidence.",
                        "Keep the original disallowed claims blocked.",
                    ],
                }
            )
            continue

        for index, target in enumerate(related_targets[:5], start=1):
            command = (
                _profile_command(target, "deep_dive")
                if followup_type == "deep_dive_after_first_evidence"
                else _profile_command(target, "minimal")
            )
            followups.append(
                {
                    "followup_id": _slug(
                        f"{pattern_id}-{followup_type}-{target}-{index}",
                        max_len=96,
                    ),
                    "source_pattern_id": pattern_id,
                    "source_lead_id": lead_id,
                    "followup_type": followup_type,
                    "target": target,
                    "command": command,
                    "reason": (
                        "Use the approved pattern as a seed, but require fresh recipient evidence before promotion."
                    ),
                    "success_condition": (
                        "Recipient run passes correctness, artifact contract, profiler, and manual review gates before any transfer claim."
                    ),
                    "claim_constraints": [
                        str(claim_packet.get("allowed_claim_scope") or "Keep claim scoped."),
                        "Cite source evidence and recipient evidence separately.",
                        "Keep original disallowed claims blocked unless new evidence explicitly disproves them.",
                    ],
                }
            )

    return {
        "harvest_count": len(patterns),
        "pattern_count": len(patterns),
        "followup_count": len(followups),
        "blocked_harvest_count": len(blocked_harvests),
        "next_followup": followups[0] if followups else None,
        "policy": (
            "Harvest only approved-and-audited novelty claims into reusable patterns. Every spawned follow-up is a new hypothesis and must rerun its own evidence gates before promotion."
        ),
        "patterns": patterns,
        "followup_experiments": followups,
        "blocked_harvests": blocked_harvests,
    }


def apply_run_queue_feedback(
    result: dict[str, Any], run_queue_summary: dict[str, Any]
) -> dict[str, Any]:
    """Overlay completed run-queue artifacts onto a freshly ranked opportunity result."""

    merged = copy.deepcopy(result)
    evidence_jobs = list(run_queue_summary.get("jobs") or [])
    evidence_by_id = {str(job.get("id")): job for job in evidence_jobs if job.get("id")}
    completed_ids = {
        str(job.get("id"))
        for job in evidence_jobs
        if _job_complete_for_resume(str(job.get("status") or ""))
    }
    target_feedback = _target_feedback_from_jobs(evidence_jobs)

    run_queue = merged.get("run_queue") or {}
    current_jobs = run_queue.get("jobs") or []
    ready_job_ids: list[str] = []
    blocked_job_ids: list[str] = []
    failed_job_ids: list[str] = []
    manual_review_job_ids: list[str] = []
    completed_job_ids: list[str] = []

    for job in current_jobs:
        job_id = str(job.get("id") or "")
        evidence = evidence_by_id.get(job_id)
        declared_dependencies = [str(dep) for dep in job.get("depends_on", [])]
        missing_dependencies = [dep for dep in declared_dependencies if dep not in completed_ids]
        if evidence:
            status = str(evidence.get("status") or "pending")
            if status == "blocked_by_dependency":
                missing_dependencies = [
                    str(dep) for dep in evidence.get("missing_dependencies", [])
                ] or missing_dependencies
            job["existing_job_dir"] = evidence.get("job_dir")
            job["stdout_log"] = evidence.get("stdout_log")
            job["stderr_log"] = evidence.get("stderr_log")
            job["diagnostic_signals"] = list(evidence.get("diagnostic_signals", []) or [])
            job["promotion_review"] = evidence.get("promotion_review")
        else:
            status = "blocked_by_dependency" if missing_dependencies else "pending"

        action = _resume_action_for_status(
            status, has_missing_dependencies=bool(missing_dependencies)
        )
        job["evidence_status"] = status
        job["resume_action"] = action
        job["missing_dependencies"] = missing_dependencies

        if _job_complete_for_resume(status):
            completed_job_ids.append(job_id)
        elif status == "manual_review_required":
            manual_review_job_ids.append(job_id)
        elif status == "failed_or_incomplete":
            failed_job_ids.append(job_id)
        elif missing_dependencies:
            blocked_job_ids.append(job_id)
        elif status == "pending":
            ready_job_ids.append(job_id)

    next_commands = [
        job.get("command")
        for job in current_jobs
        if job.get("id") in set(ready_job_ids) and job.get("command")
    ][:10]
    resume_plan = {
        "source_root": run_queue_summary.get("root"),
        "source_exists": run_queue_summary.get("exists", False),
        "evidence_job_count": run_queue_summary.get("job_count", 0),
        "status_counts": run_queue_summary.get("status_counts", {}),
        "ready_job_ids": ready_job_ids,
        "blocked_job_ids": blocked_job_ids,
        "failed_job_ids": failed_job_ids,
        "manual_review_job_ids": manual_review_job_ids,
        "completed_job_ids": completed_job_ids,
        "next_commands": next_commands,
        "next_actions": list(run_queue_summary.get("next_actions") or []),
        "policy": "Resume pending jobs only after dependencies are complete; publish claims only for promotion_review jobs with APPROVED evidence.",
    }
    if not next_commands and not failed_job_ids and not manual_review_job_ids and current_jobs:
        resume_plan["next_actions"].append(
            "No runnable queue jobs remain; rerank with fresh benchmark evidence or approve completed promotion reviews."
        )
    run_queue["resume_plan"] = resume_plan
    run_queue["resume_ready_job_ids"] = ready_job_ids
    merged["run_queue"] = run_queue

    for row in merged.get("opportunities", []) or []:
        feedback = target_feedback.get(str(row.get("target") or ""))
        if feedback:
            row["run_queue_feedback"] = feedback

    gates = (merged.get("promotion_gates") or {}).get("gates") or []
    for gate in gates:
        target = str(gate.get("target") or "")
        feedback = target_feedback.get(target, {})
        promotion_status = str(feedback.get("promotion_status") or "not_started")
        gate["run_queue_promotion_status"] = promotion_status
        if promotion_status == "approved":
            gate["claim_allowed"] = True
            gate["promotion_state"] = "approved_after_manual_review"
            gate["claim_evidence_job_ids"] = feedback.get("approved_review_job_ids", [])
        elif promotion_status == "manual_review_required":
            gate["claim_allowed"] = False
            gate["claim_evidence_job_ids"] = feedback.get("manual_review_job_ids", [])

    if merged.get("promotion_gates"):
        merged["promotion_gates"]["claim_allowed_count"] = sum(
            1 for gate in gates if gate.get("claim_allowed")
        )
        merged["promotion_gates"]["blocked_count"] = sum(
            1 for gate in gates if not gate.get("claim_allowed")
        )
    merged["run_queue_feedback"] = {
        "target_count": len(target_feedback),
        "targets": target_feedback,
        "promotion_summary": run_queue_summary.get("promotion_summary", {}),
        "resume_plan": resume_plan,
    }
    return merged


def apply_novelty_validation_feedback(
    result: dict[str, Any], novelty_queue_summary: dict[str, Any]
) -> dict[str, Any]:
    """Overlay novelty-validation artifacts onto a freshly ranked opportunity result."""

    merged = copy.deepcopy(result)
    evidence_jobs = list(novelty_queue_summary.get("jobs") or [])
    evidence_by_id = {str(job.get("id")): job for job in evidence_jobs if job.get("id")}
    completed_ids = {
        str(job.get("id"))
        for job in evidence_jobs
        if _job_complete_for_resume(str(job.get("status") or ""))
    }

    novelty_plan = merged.get("novelty_validation_plan") or {}
    current_jobs = novelty_plan.get("jobs") or []
    ready_job_ids: list[str] = []
    blocked_job_ids: list[str] = []
    failed_job_ids: list[str] = []
    manual_review_job_ids: list[str] = []
    completed_job_ids: list[str] = []

    for job in current_jobs:
        job_id = str(job.get("id") or "")
        evidence = evidence_by_id.get(job_id)
        declared_dependencies = [str(dep) for dep in job.get("depends_on", [])]
        missing_dependencies = [dep for dep in declared_dependencies if dep not in completed_ids]
        if evidence:
            status = str(evidence.get("status") or "pending")
            if status == "blocked_by_dependency":
                missing_dependencies = [
                    str(dep) for dep in evidence.get("missing_dependencies", [])
                ] or missing_dependencies
            job["existing_job_dir"] = evidence.get("job_dir")
            job["stdout_log"] = evidence.get("stdout_log")
            job["stderr_log"] = evidence.get("stderr_log")
            job["diagnostic_signals"] = list(evidence.get("diagnostic_signals", []) or [])
            job["promotion_review"] = evidence.get("promotion_review")
        else:
            status = "blocked_by_dependency" if missing_dependencies else "pending"

        action = _resume_action_for_status(
            status, has_missing_dependencies=bool(missing_dependencies)
        )
        job["evidence_status"] = status
        job["resume_action"] = action
        job["missing_dependencies"] = missing_dependencies

        if _job_complete_for_resume(status):
            completed_job_ids.append(job_id)
        elif status == "manual_review_required":
            manual_review_job_ids.append(job_id)
        elif status == "failed_or_incomplete":
            failed_job_ids.append(job_id)
        elif missing_dependencies:
            blocked_job_ids.append(job_id)
        elif status == "pending":
            ready_job_ids.append(job_id)

    evidence_audit_plan = _build_novelty_evidence_audit_plan(
        merged.get("novelty_artifact_contract_plan") or {},
        merged.get("novelty_claim_packet_plan") or {},
        novelty_queue_summary,
    )
    recovery_plan = _build_novelty_recovery_plan(current_jobs, evidence_audit_plan)
    audit_by_lead = {
        str(item.get("lead_id")): item
        for item in evidence_audit_plan.get("lead_audits", []) or []
        if item.get("lead_id")
    }
    lead_feedback = _lead_feedback_from_jobs(current_jobs if current_jobs else evidence_jobs)
    for lead_id, audit in audit_by_lead.items():
        feedback = lead_feedback.setdefault(
            lead_id,
            {
                "lead_id": lead_id,
                "lead_type": audit.get("lead_type"),
                "target": audit.get("target"),
                "job_count": 0,
                "status_counts": {},
                "completed_job_ids": [],
                "failed_job_ids": [],
                "manual_review_job_ids": [],
                "approved_review_job_ids": [],
                "validation_status": "not_started",
            },
        )
        feedback["evidence_audit_status"] = audit.get("audit_status")
        feedback["missing_required_file_count"] = audit.get("missing_file_count")
        feedback["promotion_blockers"] = list(audit.get("promotion_blockers", []) or [])
        lead_actions = recovery_plan.get("actions_by_lead", {}).get(lead_id, [])
        if lead_actions:
            feedback["recovery_action_ids"] = lead_actions
    next_commands = [
        job.get("command")
        for job in current_jobs
        if job.get("id") in set(ready_job_ids) and job.get("command")
    ][:10]
    resume_plan = {
        "source_root": novelty_queue_summary.get("root"),
        "source_exists": novelty_queue_summary.get("exists", False),
        "evidence_job_count": novelty_queue_summary.get("job_count", 0),
        "status_counts": novelty_queue_summary.get("status_counts", {}),
        "ready_job_ids": ready_job_ids,
        "blocked_job_ids": blocked_job_ids,
        "failed_job_ids": failed_job_ids,
        "manual_review_job_ids": manual_review_job_ids,
        "completed_job_ids": completed_job_ids,
        "next_commands": next_commands,
        "next_actions": list(novelty_queue_summary.get("next_actions") or []),
        "policy": "Resume novelty validation jobs only after dependencies are complete; publish novelty claims only for manual_review jobs with APPROVED evidence.",
    }
    if not next_commands and not failed_job_ids and not manual_review_job_ids and current_jobs:
        resume_plan["next_actions"].append(
            "No runnable novelty validation jobs remain; rerank with fresh benchmark evidence or approve completed manual reviews."
        )
    novelty_plan["resume_plan"] = resume_plan
    novelty_plan["resume_ready_job_ids"] = ready_job_ids

    for lead in novelty_plan.get("selected_leads", []) or []:
        feedback = lead_feedback.get(str(lead.get("lead_id") or ""))
        if feedback:
            lead["validation_feedback"] = feedback
    adaptive_decision_plan = _build_novelty_adaptive_decision_plan(
        merged.get("novelty_budget_plan") or {},
        novelty_plan,
        lead_feedback,
        recovery_plan,
    )
    learning_plan = _build_novelty_learning_plan(
        merged.get("novelty_budget_plan") or {},
        adaptive_decision_plan,
        recovery_plan,
        evidence_audit_plan,
    )
    harvest_plan = _build_novelty_harvest_plan(
        merged.get("novelty_queue") or {},
        novelty_plan,
        merged.get("novelty_claim_packet_plan") or {},
        adaptive_decision_plan,
        learning_plan,
        lead_feedback,
    )
    next_wave_plan = _build_novelty_next_wave_plan(
        novelty_plan,
        adaptive_decision_plan,
        recovery_plan,
        learning_plan,
        harvest_plan,
        merged.get("novelty_mutation_budget_plan") or {},
    )
    decision_by_lead = {
        str(item.get("lead_id")): item
        for item in adaptive_decision_plan.get("selected_decisions", []) or []
        if item.get("lead_id")
    }
    learning_by_lead = {
        str(item.get("lead_id")): item
        for item in learning_plan.get("lead_adjustments", []) or []
        if item.get("lead_id")
    }
    for lead in novelty_plan.get("selected_leads", []) or []:
        decision = decision_by_lead.get(str(lead.get("lead_id") or ""))
        if decision:
            lead["adaptive_decision"] = decision
        learning = learning_by_lead.get(str(lead.get("lead_id") or ""))
        if learning:
            lead["learning_feedback"] = learning
    merged["novelty_validation_plan"] = novelty_plan

    for collection_name in ("novelty_budget_plan", "novelty_queue", "novelty_mutation_budget_plan"):
        collection = merged.get(collection_name) or {}
        for key in ("selected", "backlog", "leads"):
            for item in collection.get(key, []) or []:
                lead_id = str(item.get("lead_id") or "")
                feedback = lead_feedback.get(lead_id)
                if feedback:
                    item["validation_feedback"] = feedback
                decision = decision_by_lead.get(lead_id)
                if decision:
                    item["adaptive_decision"] = decision
                learning = learning_by_lead.get(lead_id)
                if learning:
                    item["learning_feedback"] = learning
        if collection:
            merged[collection_name] = collection

    for collection_name, item_key in (
        ("novelty_artifact_contract_plan", "lead_contracts"),
        ("novelty_claim_packet_plan", "lead_packets"),
    ):
        collection = merged.get(collection_name) or {}
        for item in collection.get(item_key, []) or []:
            audit = audit_by_lead.get(str(item.get("lead_id") or ""))
            if audit:
                item["evidence_audit"] = audit
        if collection:
            merged[collection_name] = collection

    merged["novelty_evidence_audit_plan"] = evidence_audit_plan
    merged["novelty_recovery_plan"] = recovery_plan
    merged["novelty_adaptive_decision_plan"] = adaptive_decision_plan
    merged["novelty_learning_plan"] = learning_plan
    merged["novelty_next_wave_plan"] = next_wave_plan
    merged["novelty_harvest_plan"] = harvest_plan

    merged["novelty_validation_feedback"] = {
        "lead_count": len(lead_feedback),
        "leads": lead_feedback,
        "review_summary": novelty_queue_summary.get("review_summary", {}),
        "resume_plan": resume_plan,
        "evidence_audit_summary": {
            "status_counts": evidence_audit_plan.get("status_counts", {}),
            "missing_file_count": evidence_audit_plan.get("missing_file_count", 0),
            "present_file_count": evidence_audit_plan.get("present_file_count", 0),
        },
        "recovery_summary": {
            "action_count": recovery_plan.get("action_count", 0),
            "blocking_action_count": recovery_plan.get("blocking_action_count", 0),
            "next_action": recovery_plan.get("next_action"),
        },
        "adaptive_decision_summary": {
            "slot_state_counts": adaptive_decision_plan.get("slot_state_counts", {}),
            "blocked_count": adaptive_decision_plan.get("blocked_count", 0),
            "replacement_candidate_count": adaptive_decision_plan.get(
                "replacement_candidate_count", 0
            ),
        },
        "learning_summary": {
            "learning_state_counts": learning_plan.get("learning_state_counts", {}),
            "blocked_learning_count": learning_plan.get("blocked_learning_count", 0),
            "backup_candidate_count": learning_plan.get("backup_candidate_count", 0),
        },
        "next_wave_summary": {
            "wave_count": next_wave_plan.get("wave_count", 0),
            "action_count": next_wave_plan.get("action_count", 0),
            "first_wave": next_wave_plan.get("first_wave"),
            "wave_item_counts": next_wave_plan.get("wave_item_counts", {}),
            "next_action": next_wave_plan.get("next_action"),
        },
        "mutation_budget_summary": {
            "selected_count": (merged.get("novelty_mutation_budget_plan") or {}).get(
                "selected_count", 0
            ),
            "selected_cost_units": (merged.get("novelty_mutation_budget_plan") or {}).get(
                "selected_cost_units", 0
            ),
            "next_mutation": (merged.get("novelty_mutation_budget_plan") or {}).get(
                "next_mutation"
            ),
        },
        "harvest_summary": {
            "harvest_count": harvest_plan.get("harvest_count", 0),
            "followup_count": harvest_plan.get("followup_count", 0),
            "blocked_harvest_count": harvest_plan.get("blocked_harvest_count", 0),
            "next_followup": harvest_plan.get("next_followup"),
        },
    }
    return merged
