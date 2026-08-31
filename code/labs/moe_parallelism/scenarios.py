"""Scenario definitions for the MoE parallelism planner (tool)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from labs.moe_parallelism.plan import ClusterSpec, ModelSpec, ParallelismPlan, SPEC_PRESETS


@dataclass(frozen=True)
class ScenarioPair:
    name: str
    cluster: ClusterSpec
    model: ModelSpec
    baseline: ParallelismPlan
    optimized: ParallelismPlan


def _gpt_cluster_model() -> Tuple[ClusterSpec, ModelSpec]:
    return SPEC_PRESETS["gpt_oss_120b_gb200_ib"]


def _deepseek_cluster_model() -> Tuple[ClusterSpec, ModelSpec]:
    return SPEC_PRESETS["deepseek_r1_678b_gb200_ib"]


def get_scenario_pairs() -> Mapping[str, ScenarioPair]:
    """Return named baseline/optimized scenario pairs.

    These are planning scenarios: baseline and optimized are intentionally
    different designs, so they are exposed as a tool rather than a benchmark.
    The first five retain their original 128-GPU DGX A100 sizing preset;
    the two explicitly named GB200 scenarios use their separate 576-GPU preset.
    """
    dgx_cluster, dgx_model = SPEC_PRESETS["dgx_a100_175b"]
    gpt_cluster, gpt_model = _gpt_cluster_model()
    deepseek_cluster, deepseek_model = _deepseek_cluster_model()

    scenarios: Dict[str, ScenarioPair] = {}

    scenarios["memory_budget"] = ScenarioPair(
        name="memory_budget",
        cluster=dgx_cluster,
        model=dgx_model,
        baseline=ParallelismPlan(
            name="Baseline memory budget (overcommitted HBM)",
            dp=4,
            pp=4,
            tp=1,
            ep=8,
            microbatch_sequences=64,
            microbatches=24,
            experts_per_gpu=4,
            capacity_factor=1.1,
            dense_checkpoint_fraction=1.0,
            moe_checkpoint_fraction=1.0,
            stage_layers=[24, 24, 24, 24],
            cross_node_ep=False,
            notes=[
                "Full hidden state per GPU plus 64-seq micro-batch blows past 80 GB",
                "No activation checkpointing, so dense blocks stash every tensor",
                "24 in-flight micro-batches also inflate optimizer/shard residency",
            ],
        ),
        optimized=ParallelismPlan(
            name="Optimized memory budget (balanced activations)",
            dp=4,
            pp=4,
            tp=2,
            ep=4,
            microbatch_sequences=32,
            microbatches=16,
            experts_per_gpu=4,
            capacity_factor=1.25,
            dense_checkpoint_fraction=0.5,
            moe_checkpoint_fraction=0.85,
            stage_layers=[24, 24, 24, 24],
            cross_node_ep=False,
            notes=[
                "Tensor parallelism halves the hidden slice per GPU, freeing activation headroom",
                "Checkpoint every other dense block + 85% of MoE blocks to keep margin ≥15 GB",
                "Micro-batch capped at 32 sequences so we can still hold 16 chunks in flight",
            ],
        ),
    )

    scenarios["moe_grouping"] = ScenarioPair(
        name="moe_grouping",
        cluster=dgx_cluster,
        model=dgx_model,
        baseline=ParallelismPlan(
            name="Baseline expert grouping (cross-node EP=8)",
            dp=2,
            pp=4,
            tp=2,
            ep=8,
            microbatch_sequences=24,
            microbatches=12,
            experts_per_gpu=2,
            capacity_factor=0.9,
            dense_checkpoint_fraction=0.75,
            moe_checkpoint_fraction=1.0,
            stage_layers=[24, 24, 24, 24],
            cross_node_ep=True,
            notes=[
                "Each pipeline stage spans two nodes so EP groups straddle HDR100 links",
                "Capacity factor 1.0 gives no safety margin for hot experts, so drops spike",
                "Per-GPU tokens stay high because micro-batch partitioning ignores EP fan-out",
            ],
        ),
        optimized=ParallelismPlan(
            name="Optimized expert grouping (EP kept on-node)",
            dp=4,
            pp=4,
            tp=2,
            ep=4,
            microbatch_sequences=32,
            microbatches=18,
            experts_per_gpu=4,
            capacity_factor=1.25,
            dense_checkpoint_fraction=0.5,
            moe_checkpoint_fraction=1.0,
            stage_layers=[24, 24, 24, 24],
            cross_node_ep=False,
            notes=[
                "EP groups fit inside a single node so MoE all-to-all stays on NVSwitch",
                "Each GPU hosts four experts (32 per stage) which aligns with the 128-expert target",
                "Router slack (capacity 1.25) + top-2 gating keeps load balance auxiliary loss effective",
            ],
        ),
    )

    scenarios["network_affinity"] = ScenarioPair(
        name="network_affinity",
        cluster=dgx_cluster,
        model=dgx_model,
        baseline=ParallelismPlan(
            name="Baseline network affinity (DP-dominated)",
            dp=8,
            pp=2,
            tp=1,
            ep=8,
            microbatch_sequences=16,
            microbatches=10,
            experts_per_gpu=4,
            capacity_factor=1.1,
            dense_checkpoint_fraction=0.75,
            moe_checkpoint_fraction=1.0,
            stage_layers=[48, 48],
            cross_node_ep=False,
            notes=[
                "Eight DP replicas cause full-parameter all-reduce over HDR100 every step",
                "Only two pipeline stages so activation transfers are massive and cross-node",
                "No affinity guidance for NIC binding or NVSwitch locality",
            ],
        ),
        optimized=ParallelismPlan(
            name="Optimized network affinity (hierarchical collectives)",
            dp=4,
            pp=4,
            tp=2,
            ep=4,
            microbatch_sequences=32,
            microbatches=18,
            experts_per_gpu=4,
            capacity_factor=1.25,
            dense_checkpoint_fraction=0.5,
            moe_checkpoint_fraction=0.95,
            stage_layers=[24, 24, 24, 24],
            cross_node_ep=False,
            notes=[
                "DP=4 leaves each replica on four nodes so NCCL trees stay shallow",
                "Stage neighbors share InfiniBand pairs -> easy NIC pinning for pipeline sends",
                "EP dispatch + TP all-reduce never leave NVSwitch so HDR100 handles only DP/PP traffic",
            ],
        ),
    )

    scenarios["parallelism_breakdown"] = ScenarioPair(
        name="parallelism_breakdown",
        cluster=dgx_cluster,
        model=dgx_model,
        baseline=ParallelismPlan(
            name="Baseline parallelism factorization (underspecified)",
            dp=2,
            pp=16,
            tp=1,
            ep=2,
            microbatch_sequences=16,
            microbatches=8,
            experts_per_gpu=8,
            capacity_factor=1.0,
            dense_checkpoint_fraction=1.0,
            moe_checkpoint_fraction=1.0,
            stage_layers=[6] * 16,
            cross_node_ep=True,
            notes=[
                "Intentionally mismatched world size (only 64 ranks assigned to a 128 GPU cluster)",
                "Expert groups bleed across nodes so token exchanges pound HDR100 instead of NVSwitch",
                "TP degree of 1 leaves each GPU with the full hidden size, inflating activation memory",
            ],
        ),
        optimized=ParallelismPlan(
            name="Optimized parallelism factorization (4×4×2×4)",
            dp=4,
            pp=4,
            tp=2,
            ep=4,
            microbatch_sequences=32,
            microbatches=16,
            experts_per_gpu=4,
            capacity_factor=1.25,
            dense_checkpoint_fraction=0.5,
            moe_checkpoint_fraction=1.0,
            stage_layers=[24, 24, 24, 24],
            cross_node_ep=False,
            notes=[
                "TP×EP grid (2×4) fits within a single DGX node so NVSwitch handles the heavy traffic",
                "4 DP replicas fully tile the 16 nodes, keeping optimizer sharding localized",
                "Micro-batch of 32 sequences with 16 in-flight chunks fills the 4-stage pipeline",
            ],
        ),
    )

    scenarios["pipeline_schedule"] = ScenarioPair(
        name="pipeline_schedule",
        cluster=dgx_cluster,
        model=dgx_model,
        baseline=ParallelismPlan(
            name="Baseline pipeline schedule (64-stage mindset)",
            dp=4,
            pp=8,
            tp=1,
            ep=4,
            microbatch_sequences=8,
            microbatches=6,
            experts_per_gpu=4,
            capacity_factor=1.0,
            dense_checkpoint_fraction=1.0,
            moe_checkpoint_fraction=1.0,
            stage_layers=[36, 12, 12, 12, 12, 6, 3, 3],
            cross_node_ep=False,
            notes=[
                "8 pipeline stages squeezed into 4 nodes -> half-node stages and tiny micro-batches",
                "Micro-batches (6) barely exceed the stage count (8), so pipeline bubbles dominate",
                "Stage splits ignore embedding/output heft, leaving stage0 as a straggler",
            ],
        ),
        optimized=ParallelismPlan(
            name="Optimized pipeline (4 stages / 20 micro-batches)",
            dp=4,
            pp=4,
            tp=2,
            ep=4,
            microbatch_sequences=32,
            microbatches=20,
            experts_per_gpu=4,
            capacity_factor=1.25,
            dense_checkpoint_fraction=0.5,
            moe_checkpoint_fraction=0.9,
            stage_layers=[26, 22, 24, 24],
            cross_node_ep=False,
            notes=[
                "Stage 0 soaks the embedding + first 24 blocks; remaining stages split evenly",
                "20 micro-batches keep ≥2×PP chunks in flight so pipeline bubbles stay under 10%",
                "Checkpoint dense layers every other block to buy the micro-batch headroom",
            ],
        ),
    )

    scenarios["gpt_gb200"] = ScenarioPair(
        name="gpt_gb200",
        cluster=gpt_cluster,
        model=gpt_model,
        baseline=ParallelismPlan(
            name="Baseline GPT-OSS-120B (DP9×PP8×TP2×EP4)",
            dp=9,
            pp=8,
            tp=2,
            ep=4,
            microbatch_sequences=48,
            microbatches=24,
            experts_per_gpu=4,
            capacity_factor=1.2,
            dense_checkpoint_fraction=0.6,
            moe_checkpoint_fraction=0.9,
            stage_layers=[12] * 8,
            cross_node_ep=False,
            notes=[
                "Matches the original GB200 NVL72 layout with PP=8 (bubble ~23%)",
                "Stage splits are even; no special handling for embedding/head",
            ],
        ),
        optimized=ParallelismPlan(
            name="Optimized GPT-OSS-120B (DP8×PP6×TP3×EP4)",
            dp=8,
            pp=6,
            tp=3,
            ep=4,
            microbatch_sequences=64,
            microbatches=32,
            experts_per_gpu=4,
            capacity_factor=1.2,
            dense_checkpoint_fraction=0.5,
            moe_checkpoint_fraction=0.85,
            stage_layers=[18, 16, 16, 16, 15, 15],
            cross_node_ep=False,
            notes=[
                "PP reduced to 6 with deeper micro-batching (32) to cut bubble",
                "Stage0 takes extra layers for embeddings/head; later stages balanced",
                "TP raised to 3 to trim per-rank params/activations without leaving NVLink",
            ],
        ),
    )

    scenarios["deepseek_gb200"] = ScenarioPair(
        name="deepseek_gb200",
        cluster=deepseek_cluster,
        model=deepseek_model,
        baseline=ParallelismPlan(
            name="Baseline DeepSeek-R1-678B (DP9×PP8×TP2×EP4)",
            dp=9,
            pp=8,
            tp=2,
            ep=4,
            microbatch_sequences=32,
            microbatches=24,
            experts_per_gpu=4,
            capacity_factor=1.2,
            dense_checkpoint_fraction=0.5,
            moe_checkpoint_fraction=0.9,
            stage_layers=[16] * 8,
            cross_node_ep=False,
            notes=[
                "Even 8-stage pipeline leaves ~23% bubble and high PP traffic",
                "Uses conservative micro-batch to respect 678B footprint",
            ],
        ),
        optimized=ParallelismPlan(
            name="Optimized DeepSeek-R1-678B (DP8×PP6×TP3×EP4)",
            dp=8,
            pp=6,
            tp=3,
            ep=4,
            microbatch_sequences=32,
            microbatches=36,
            experts_per_gpu=4,
            capacity_factor=1.2,
            dense_checkpoint_fraction=0.5,
            moe_checkpoint_fraction=0.9,
            stage_layers=[22, 21, 21, 21, 21, 22],
            cross_node_ep=False,
            notes=[
                "PP=6 with 36 micro-batches reduces bubble vs the 8-stage baseline",
                "TP=3 lowers per-rank params/activations to ease HBM pressure",
                "Stages rebalanced to keep early/late blocks heavier for embeds/head",
            ],
        ),
    )

    return scenarios

