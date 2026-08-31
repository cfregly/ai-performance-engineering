"""Scenario builder for the MoE parallelism lab."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

BYTES_PER_GB = 1024 ** 3


def _even_split(total: int, parts: int) -> List[int]:
    base = total // parts
    remainder = total % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def _bytes_to_gb(value: float) -> float:
    return value / BYTES_PER_GB


def _bytes_to_ms(value_bytes: float, bandwidth_GBps: float) -> float:
    if bandwidth_GBps <= 0:
        return math.inf
    seconds = value_bytes / (bandwidth_GBps * (1024 ** 3))
    return seconds * 1e3


@dataclass(frozen=True)
class ClusterSpec:
    """Hardware cluster description."""

    name: str
    nodes: int
    gpus_per_node: int
    nics_per_node: int
    nic_bandwidth_gbps: float
    nvlink_bandwidth_tbps: float
    hbm_gb: float
    interconnect: str = "InfiniBand"

    @property
    def gpus_total(self) -> int:
        return self.nodes * self.gpus_per_node

    @property
    def nic_bandwidth_GBps(self) -> float:
        return (self.nic_bandwidth_gbps / 8.0) * self.nics_per_node

    @property
    def nvlink_bandwidth_GBps(self) -> float:
        return self.nvlink_bandwidth_tbps * 1000


@dataclass(frozen=True)
class ModelSpec:
    """Model metadata used for sizing calculations."""

    name: str
    params_billion: float
    hidden_size: int
    sequence_length: int
    layers: int
    moe_layers: int
    experts_total: int
    router_topk: int = 2
    ffn_expansion: int = 4
    dtype_bytes: int = 2


DEFAULT_SPEC_PRESET = "gpt_oss_120b_gb200_ib"


def _build_spec_presets() -> Dict[str, Tuple[ClusterSpec, ModelSpec]]:
    gpt_cluster_ib = ClusterSpec(
        name="GB200 NVL72 (8 racks, dual 800G CX8)",
        nodes=72,
        gpus_per_node=8,
        nics_per_node=2,
        nic_bandwidth_gbps=800.0,
        nvlink_bandwidth_tbps=1.8,
        hbm_gb=192.0,
        interconnect="InfiniBand",
    )
    gpt_cluster_eth = ClusterSpec(
        name="GB200 NVL72 (8 racks, dual 400G Ethernet)",
        nodes=72,
        gpus_per_node=8,
        nics_per_node=2,
        nic_bandwidth_gbps=400.0,
        nvlink_bandwidth_tbps=1.8,
        hbm_gb=192.0,
        interconnect="Ethernet",
    )
    gpt_model = ModelSpec(
        name="GPT-OSS-120B MoE",
        params_billion=120.0,
        hidden_size=10240,
        sequence_length=4096,
        layers=96,
        moe_layers=32,
        experts_total=256,
        router_topk=2,
        ffn_expansion=4,
        dtype_bytes=2,
    )
    deepseek_model = ModelSpec(
        name="DeepSeek-R1-678B MoE",
        params_billion=678.0,
        hidden_size=14336,
        sequence_length=4096,
        layers=128,
        moe_layers=48,
        experts_total=256,
        router_topk=2,
        ffn_expansion=4,
        dtype_bytes=2,
    )
    dgx_cluster = ClusterSpec(
        name="DGX A100 (16 nodes, dual HDR100)",
        nodes=16,
        gpus_per_node=8,
        nics_per_node=2,
        nic_bandwidth_gbps=100.0,
        nvlink_bandwidth_tbps=0.6,
        hbm_gb=80.0,
    )
    dgx_model = ModelSpec(
        name="175B MoE Transformer",
        params_billion=175.0,
        hidden_size=12288,
        sequence_length=2048,
        layers=96,
        moe_layers=24,
        experts_total=128,
        router_topk=2,
        ffn_expansion=4,
        dtype_bytes=2,
    )
    return {
        "gpt_oss_120b_gb200_ib": (gpt_cluster_ib, gpt_model),
        "gpt_oss_120b_gb200_ethernet": (gpt_cluster_eth, gpt_model),
        "deepseek_r1_678b_gb200_ib": (gpt_cluster_ib, deepseek_model),
        "dgx_a100_175b": (dgx_cluster, dgx_model),
    }


SPEC_PRESETS: Dict[str, Tuple[ClusterSpec, ModelSpec]] = _build_spec_presets()
AVAILABLE_SPEC_PRESETS: Tuple[str, ...] = tuple(sorted(SPEC_PRESETS.keys()))
_ACTIVE_SPEC_PRESET = DEFAULT_SPEC_PRESET
DEFAULT_CLUSTER: ClusterSpec
DEFAULT_MODEL: ModelSpec


def set_active_spec_preset(name: str) -> None:
    """Update the global default cluster/model preset."""
    global _ACTIVE_SPEC_PRESET, DEFAULT_CLUSTER, DEFAULT_MODEL
    if name not in SPEC_PRESETS:
        raise ValueError(
            f"Unknown spec preset '{name}'. Available: {', '.join(AVAILABLE_SPEC_PRESETS)}"
        )
    _ACTIVE_SPEC_PRESET = name
    cluster, model = SPEC_PRESETS[name]
    DEFAULT_CLUSTER = cluster
    DEFAULT_MODEL = model


def get_active_spec_preset() -> str:
    return _ACTIVE_SPEC_PRESET


def get_default_cluster_spec() -> ClusterSpec:
    return DEFAULT_CLUSTER


def get_default_model_spec() -> ModelSpec:
    return DEFAULT_MODEL


def resolve_specs() -> Tuple[ClusterSpec, ModelSpec]:
    return DEFAULT_CLUSTER, DEFAULT_MODEL


# Initialize defaults
set_active_spec_preset(DEFAULT_SPEC_PRESET)


@dataclass
class ParallelismPlan:
    """Parallelism and scheduling choices for the lab."""

    name: str
    dp: int
    pp: int
    tp: int
    ep: int
    microbatch_sequences: int
    microbatches: int
    experts_per_gpu: int
    capacity_factor: float
    dense_checkpoint_fraction: float = 1.0
    moe_checkpoint_fraction: float = 1.0
    stage_layers: Optional[List[int]] = None
    cross_node_ep: bool = False
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.dense_checkpoint_fraction <= 0 or self.dense_checkpoint_fraction > 1:
            raise ValueError("dense_checkpoint_fraction must be in (0, 1]")
        if self.moe_checkpoint_fraction <= 0 or self.moe_checkpoint_fraction > 1:
            raise ValueError("moe_checkpoint_fraction must be in (0, 1]")
        if self.microbatches <= 0:
            raise ValueError("microbatches must be positive")
        if self.microbatch_sequences <= 0:
            raise ValueError("microbatch_sequences must be positive")
        if self.experts_per_gpu <= 0:
            raise ValueError("experts_per_gpu must be positive")
        if self.capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")


@dataclass
class PlanReport:
    """Detailed analysis of a parallelism plan."""

    plan_name: str
    world_size: int
    world_size_matches: bool
    dp: int
    pp: int
    tp: int
    ep: int
    nodes_per_replica: float
    nodes_per_stage: float
    gpus_per_stage: float
    tp_ep_product: int
    stage_layers: List[int]
    dense_layers_per_stage: List[int]
    moe_layers_per_stage: List[int]
    bubble_fraction: float
    compute_ms: float
    dp_time_ms: float
    pipeline_time_ms: float
    ep_time_ms: float
    estimated_step_ms: float
    throughput_tokens_per_s: float
    params_gb: float
    optimizer_gb: float
    grad_gb: float
    activation_gb: float
    total_memory_gb: float
    memory_margin_gb: float
    ep_cross_node: bool
    tokens_per_gpu: float
    tokens_per_expert: float
    capacity_limit_tokens: float
    load_balance_ok: bool
    hotspots: List[str]
    affinity: List[str]
    notes: List[str]

    def as_dict(self) -> Dict[str, float]:
        return {
            "world_size": float(self.world_size),
            "bubble_fraction": self.bubble_fraction,
            "step_time_ms": self.estimated_step_ms,
            "throughput_tokens_s": self.throughput_tokens_per_s,
            "memory_margin_gb": self.memory_margin_gb,
            "tokens_per_expert": self.tokens_per_expert,
        }


class PlanEvaluator:
    """Compute derived metrics for a parallelism plan."""

    ACTIVATION_MULTIPLIER = 2.0
    MOE_ACTIVATION_MULTIPLIER = 1.5
    OPTIMIZER_MULTIPLIER = 2.5  # AdamW moments/shards
    COMPUTE_COEFF = 2.5e-4
    SAFETY_BUFFER_GB = 4.0

    def __init__(self, cluster: ClusterSpec, model: ModelSpec):
        self.cluster = cluster
        self.model = model

    def analyze(self, plan: ParallelismPlan) -> PlanReport:
        cluster = self.cluster
        model = self.model
        world_size = plan.dp * plan.pp * plan.tp * plan.ep
        world_size_matches = world_size == cluster.gpus_total

        stage_layers = plan.stage_layers or _even_split(model.layers, plan.pp)
        if len(stage_layers) != plan.pp:
            raise ValueError("stage_layers must have length equal to pipeline stages")
        if sum(stage_layers) != model.layers:
            raise ValueError("stage_layers must sum to total layers")

        moe_layers_per_stage = _even_split(model.moe_layers, plan.pp)
        dense_layers_per_stage = [
            max(stage_layers[i] - moe_layers_per_stage[i], 0)
            for i in range(plan.pp)
        ]

        nodes_per_replica = cluster.nodes / plan.dp
        nodes_per_stage = nodes_per_replica / plan.pp if plan.pp else 0
        gpus_per_stage = nodes_per_stage * cluster.gpus_per_node
        tp_ep_product = plan.tp * plan.ep

        per_gpu_sequences = plan.microbatch_sequences / max(plan.ep, 1)
        hidden_per_gpu = model.hidden_size / max(plan.tp, 1)
        dense_block_bytes = (
            hidden_per_gpu
            * model.sequence_length
            * per_gpu_sequences
            * model.dtype_bytes
            * self.ACTIVATION_MULTIPLIER
        )
        tokens_total = model.sequence_length * plan.microbatch_sequences
        moe_tokens_per_gpu = tokens_total * model.router_topk / max(plan.ep, 1)
        moe_block_bytes = (
            hidden_per_gpu
            * moe_tokens_per_gpu
            * model.dtype_bytes
            * self.MOE_ACTIVATION_MULTIPLIER
        )

        activation_per_stage = []
        for dense_layers, moe_layers in zip(dense_layers_per_stage, moe_layers_per_stage):
            dense_bytes = dense_layers * dense_block_bytes * plan.dense_checkpoint_fraction
            moe_bytes = moe_layers * moe_block_bytes * plan.moe_checkpoint_fraction
            activation_per_stage.append(_bytes_to_gb(dense_bytes + moe_bytes))
        activation_gb = max(activation_per_stage) + self.SAFETY_BUFFER_GB

        params_total_bytes = model.params_billion * 1e9 * model.dtype_bytes
        stage_layer_fractions = [layers / model.layers for layers in stage_layers]
        max_stage_fraction = max(stage_layer_fractions)
        params_stage_bytes = params_total_bytes * max_stage_fraction
        params_per_gpu_bytes = params_stage_bytes / max(tp_ep_product, 1)
        params_gb = _bytes_to_gb(params_per_gpu_bytes)
        optimizer_gb = params_gb * self.OPTIMIZER_MULTIPLIER
        grad_gb = params_gb

        total_memory_gb = params_gb + optimizer_gb + grad_gb + activation_gb
        memory_margin_gb = cluster.hbm_gb - total_memory_gb

        bubble_fraction = 0.0
        if plan.pp > 1:
            bubble_fraction = (plan.pp - 1) / (plan.microbatches + plan.pp - 1)

        compute_ms = (
            model.layers
            * model.sequence_length
            * plan.microbatch_sequences
            / max(tp_ep_product, 1)
        ) * self.COMPUTE_COEFF

        dp_bytes = params_total_bytes / max(plan.dp, 1)
        dp_time_ms = _bytes_to_ms(dp_bytes, cluster.nic_bandwidth_GBps)

        pipeline_bytes = (
            model.hidden_size
            * model.sequence_length
            * plan.microbatch_sequences
            * model.dtype_bytes
            * plan.microbatches
            * max(plan.pp - 1, 0)
        )
        pipeline_time_ms = _bytes_to_ms(pipeline_bytes, cluster.nic_bandwidth_GBps)

        ep_bytes = (
            moe_tokens_per_gpu
            * hidden_per_gpu
            * model.dtype_bytes
            * plan.microbatches
            * max(plan.ep - 1, 1)
        )
        ep_bandwidth = (
            cluster.nic_bandwidth_GBps if plan.cross_node_ep else cluster.nvlink_bandwidth_GBps
        )
        ep_time_ms = _bytes_to_ms(ep_bytes, ep_bandwidth)

        estimated_step_ms = compute_ms + dp_time_ms + pipeline_time_ms + ep_time_ms
        global_batch_tokens = (
            model.sequence_length * plan.microbatch_sequences * plan.microbatches * plan.dp
        )
        throughput_tokens_per_s = 0.0
        if estimated_step_ms > 0:
            throughput_tokens_per_s = global_batch_tokens / (estimated_step_ms / 1e3)

        tokens_per_gpu = tokens_total / max(plan.ep, 1)
        tokens_per_expert = (
            tokens_per_gpu * model.router_topk / max(plan.experts_per_gpu, 1)
        )
        capacity_limit_tokens = (
            tokens_per_gpu
            / max(plan.experts_per_gpu, 1)
            * (plan.capacity_factor + model.router_topk - 1.0)
        )
        load_balance_ok = tokens_per_expert <= capacity_limit_tokens + 1e-6

        hotspots: List[str] = []
        affinity: List[str] = []

        if not world_size_matches:
            hotspots.append(
                f"World size {world_size} does not match available GPUs {cluster.gpus_total}"
            )
        if nodes_per_stage < 1:
            hotspots.append("Pipeline stage has <1 node; configuration invalid")
        if abs(gpus_per_stage - tp_ep_product) > 1e-3:
            hotspots.append(
                f"Stage GPU count {gpus_per_stage:.1f} does not match TP×EP grid {tp_ep_product}"
            )
        if bubble_fraction > 0.2:
            hotspots.append(
                f"Pipeline bubble {bubble_fraction:.2f} wastes >20% of a pass"
            )
        if memory_margin_gb < 0:
            hotspots.append(
                f"Memory overcommitted by {abs(memory_margin_gb):.1f} GB (activation tuning needed)"
            )
        if plan.cross_node_ep:
            hotspots.append("Expert groups span nodes, forcing HDR100 all-to-all traffic")
        if not load_balance_ok:
            hotspots.append(
                f"Expert load ratio {tokens_per_expert / max(capacity_limit_tokens, 1e-6):.2f} exceeds capacity"
            )
        if dp_time_ms > compute_ms * 0.6:
            hotspots.append("DP gradient all-reduce dominates step time")
        if pipeline_time_ms > compute_ms * 0.4:
            hotspots.append("Pipeline activation transfers saturate HDR100")

        affinity.append(
            f"Nodes per replica: {nodes_per_replica:.1f}, nodes per stage: {nodes_per_stage:.1f}"
        )
        affinity.append(
            f"Per-stage TP×EP grid: {plan.tp}×{plan.ep} -> {tp_ep_product} GPUs (available {gpus_per_stage:.1f})"
        )
        affinity.append(
            "EP communication: "
            + ("cross-node over HDR100" if plan.cross_node_ep else "kept on NVSwitch per node")
        )

        notes = list(plan.notes)
        if memory_margin_gb < 8:
            notes.append("Tight HBM headroom; consider more checkpointing or smaller micro-batch")
        if bubble_fraction < 0.1 and plan.microbatches >= plan.pp * 2:
            notes.append("Pipeline filled with ≥2× stages worth of micro-batches")

        return PlanReport(
            plan_name=plan.name,
            world_size=world_size,
            world_size_matches=world_size_matches,
            dp=plan.dp,
            pp=plan.pp,
            tp=plan.tp,
            ep=plan.ep,
            nodes_per_replica=nodes_per_replica,
            nodes_per_stage=nodes_per_stage,
            gpus_per_stage=gpus_per_stage,
            tp_ep_product=tp_ep_product,
            stage_layers=stage_layers,
            dense_layers_per_stage=dense_layers_per_stage,
            moe_layers_per_stage=moe_layers_per_stage,
            bubble_fraction=bubble_fraction,
            compute_ms=compute_ms,
            dp_time_ms=dp_time_ms,
            pipeline_time_ms=pipeline_time_ms,
            ep_time_ms=ep_time_ms,
            estimated_step_ms=estimated_step_ms,
            throughput_tokens_per_s=throughput_tokens_per_s,
            params_gb=params_gb,
            optimizer_gb=optimizer_gb,
            grad_gb=grad_gb,
            activation_gb=activation_gb,
            total_memory_gb=total_memory_gb,
            memory_margin_gb=memory_margin_gb,
            ep_cross_node=plan.cross_node_ep,
            tokens_per_gpu=tokens_per_gpu,
            tokens_per_expert=tokens_per_expert,
            capacity_limit_tokens=capacity_limit_tokens,
            load_balance_ok=load_balance_ok,
            hotspots=hotspots,
            affinity=affinity,
            notes=notes,
        )


def format_report(report: PlanReport) -> str:
    """Return a human-readable string for the analysis."""

    lines = [
        f"Plan: {report.plan_name}",
        f"  Parallelism (DP×PP×TP×EP): {report.dp}×{report.pp}×{report.tp}×{report.ep}"
        f" = {report.world_size} GPUs ({'OK' if report.world_size_matches else 'MISMATCH'})",
        f"  Nodes/stage: {report.nodes_per_stage:.1f} | GPUs/stage: {report.gpus_per_stage:.1f}",
        f"  Stage layers: {report.stage_layers} (dense {report.dense_layers_per_stage}, MoE {report.moe_layers_per_stage})",
        f"  Pipeline bubble: {report.bubble_fraction:.2f}",
        f"  Step time breakdown (ms) -> compute {report.compute_ms:.1f}, DP {report.dp_time_ms:.1f},"
        f" pipeline {report.pipeline_time_ms:.1f}, EP {report.ep_time_ms:.1f}, total {report.estimated_step_ms:.1f}",
        f"  Throughput: {report.throughput_tokens_per_s:,.0f} tokens/s",
        f"  Memory (GB) -> params {report.params_gb:.1f}, optimizer {report.optimizer_gb:.1f},"
        f" grads {report.grad_gb:.1f}, activations {report.activation_gb:.1f}; margin {report.memory_margin_gb:.1f}",
        f"  Tokens/gpu {report.tokens_per_gpu:,.0f}, per-expert {report.tokens_per_expert:,.0f} (capacity {report.capacity_limit_tokens:,.0f})",
    ]

    if report.hotspots:
        lines.append("  Hotspots:")
        for item in report.hotspots:
            lines.append(f"    - {item}")
    if report.affinity:
        lines.append("  Affinity:")
        for item in report.affinity:
            lines.append(f"    - {item}")
    if report.notes:
        lines.append("  Notes:")
        for item in report.notes:
            lines.append(f"    - {item}")
    return "\n".join(lines)
