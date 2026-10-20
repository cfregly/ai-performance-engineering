#!/usr/bin/env python3
"""
Parallelism Strategy Optimizer

Recommends optimal parallelism strategies (DP, TP, PP, CP, EP, SP) based on:
- Hardware topology (NVLink mesh, bandwidth, memory)
- Model architecture (size, MoE, attention type)
- Workload requirements (batch size, sequence length, latency)

Key Principles:
- TP: Prefer within NVSwitch/NVLink domains (high bandwidth needed)
- PP: Can span slower interconnects, helps with memory
- DP: Works everywhere, scales throughput
- CP: For long sequences, needs ring bandwidth
- EP: For MoE models, benefits from high all-to-all bandwidth
- SP: Sequence parallelism to reduce activation memory
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

from .topology_detector import TopologyInfo
from .model_analyzer import ModelArchitecture, ModelType


class OptimizationGoal(Enum):
    """Primary optimization goal."""
    THROUGHPUT = "throughput"  # Maximize tokens/second
    LATENCY = "latency"        # Minimize time-to-first-token
    MEMORY = "memory"          # Fit in available memory
    EFFICIENCY = "efficiency"  # Balance throughput vs cost


@dataclass
class ParallelismStrategy:
    """A specific parallelism configuration."""
    
    # Core parallelism dimensions
    tp: int = 1   # Tensor Parallel
    pp: int = 1   # Pipeline Parallel
    dp: int = 1   # Data Parallel
    cp: int = 1   # Context Parallel
    ep: int = 1   # Expert Parallel (for MoE)
    sp: int = 1   # Sequence Parallel
    
    # Additional configuration
    micro_batch_size: int = 1
    num_micro_batches: int = 1
    gradient_accumulation: int = 1
    activation_checkpointing: bool = False
    
    @property
    def world_size(self) -> int:
        """Total number of GPUs needed."""
        return self.tp * self.pp * self.dp * max(self.cp, 1) * max(self.ep, 1)
    
    @property
    def effective_batch_size(self) -> int:
        """Effective batch size per step."""
        return self.micro_batch_size * self.num_micro_batches * self.dp * self.gradient_accumulation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "cp": self.cp,
            "ep": self.ep,
            "sp": self.sp,
            "world_size": self.world_size,
            "micro_batch_size": self.micro_batch_size,
            "num_micro_batches": self.num_micro_batches,
            "gradient_accumulation": self.gradient_accumulation,
            "activation_checkpointing": self.activation_checkpointing,
            "effective_batch_size": self.effective_batch_size,
        }


@dataclass
class StrategyAnalysis:
    """Analysis of a parallelism strategy's performance."""
    
    # Memory estimates
    memory_per_gpu_gb: float
    peak_memory_gb: float
    memory_headroom_gb: float
    fits_in_memory: bool
    
    # Performance estimates
    estimated_throughput_tps: float  # Tokens per second
    estimated_latency_ms: float      # Per-token latency
    compute_efficiency: float         # 0-1, how much compute is utilized
    communication_overhead: float     # 0-1, fraction spent communicating
    
    # Pipeline metrics
    pipeline_bubble_fraction: float
    
    # Bottleneck analysis
    bottleneck: str  # "compute", "memory", "tp_comm", "pp_comm", "dp_comm"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_per_gpu_gb": self.memory_per_gpu_gb,
            "peak_memory_gb": self.peak_memory_gb,
            "memory_headroom_gb": self.memory_headroom_gb,
            "fits_in_memory": self.fits_in_memory,
            "estimated_throughput_tps": self.estimated_throughput_tps,
            "estimated_latency_ms": self.estimated_latency_ms,
            "compute_efficiency": self.compute_efficiency,
            "communication_overhead": self.communication_overhead,
            "pipeline_bubble_fraction": self.pipeline_bubble_fraction,
            "bottleneck": self.bottleneck,
        }


@dataclass
class StrategyRecommendation:
    """A recommended parallelism strategy with rationale."""
    
    strategy: ParallelismStrategy
    analysis: StrategyAnalysis
    score: float  # 0-100 overall score
    rationale: List[str]  # Human-readable explanations
    warnings: List[str]   # Potential issues
    optimizations: List[str]  # Suggested optimizations
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.to_dict(),
            "analysis": self.analysis.to_dict(),
            "score": self.score,
            "rationale": self.rationale,
            "warnings": self.warnings,
            "optimizations": self.optimizations,
        }


class StrategyOptimizer:
    """Optimizes parallelism strategies based on topology and model."""
    
    # Bandwidth thresholds (GB/s) for different parallelism types
    TP_MIN_BANDWIDTH = 200   # TP needs very high bandwidth
    PP_MIN_BANDWIDTH = 50    # PP can work with moderate bandwidth
    EP_MIN_BANDWIDTH = 100   # EP all-to-all needs good bandwidth
    
    # Memory overhead factors
    ACTIVATION_CHECKPOINT_SAVINGS = 0.7  # 70% activation memory reduction
    KV_CACHE_FP8_SAVINGS = 0.5          # 50% KV cache with FP8
    
    def __init__(self):
        pass
    
    def recommend(
        self,
        topology: TopologyInfo,
        model: ModelArchitecture,
        goal: OptimizationGoal = OptimizationGoal.THROUGHPUT,
        batch_size: int = 1,
        seq_length: int = 2048,
        is_training: bool = False,
        max_strategies: int = 5,
    ) -> List[StrategyRecommendation]:
        """Generate recommended parallelism strategies.
        
        Args:
            topology: Hardware topology information
            model: Model architecture
            goal: Primary optimization goal
            batch_size: Target batch size
            seq_length: Target sequence length
            is_training: Whether this is for training (vs inference)
            max_strategies: Maximum number of strategies to return
            
        Returns:
            List of recommended strategies, sorted by score
        """
        candidates = self._generate_candidates(topology, model, is_training)
        
        recommendations = []
        for strategy in candidates:
            analysis = self._analyze_strategy(
                strategy, topology, model, batch_size, seq_length, is_training
            )
            
            if not analysis.fits_in_memory:
                continue
            
            score, rationale, warnings, optimizations = self._score_strategy(
                strategy, analysis, topology, model, goal, is_training
            )
            
            recommendations.append(StrategyRecommendation(
                strategy=strategy,
                analysis=analysis,
                score=score,
                rationale=rationale,
                warnings=warnings,
                optimizations=optimizations,
            ))
        
        # Sort by score descending
        recommendations.sort(key=lambda r: r.score, reverse=True)
        
        return recommendations[:max_strategies]
    
    def _generate_candidates(
        self,
        topology: TopologyInfo,
        model: ModelArchitecture,
        is_training: bool,
    ) -> List[ParallelismStrategy]:
        """Generate candidate parallelism strategies."""
        candidates = []
        num_gpus = topology.num_gpus
        
        # Get optimal TP sizes based on topology
        optimal_tp_sizes = topology.get_optimal_tp_sizes()
        
        # Generate TP/PP/DP combinations
        for tp in optimal_tp_sizes:
            if tp > num_gpus:
                continue
            
            remaining = num_gpus // tp
            
            # PP options: 1, 2, 4, 8 (limited by layers)
            max_pp = min(remaining, model.num_layers // 2, 8)
            pp_options = [p for p in [1, 2, 4, 8] if p <= max_pp]
            
            for pp in pp_options:
                dp = remaining // pp
                if dp < 1:
                    continue
                if tp * pp * dp != num_gpus:
                    continue
                
                # Basic strategy
                candidates.append(ParallelismStrategy(
                    tp=tp, pp=pp, dp=dp,
                    micro_batch_size=1,
                    num_micro_batches=max(pp * 2, 4) if pp > 1 else 1,
                ))
                
                # With activation checkpointing
                candidates.append(ParallelismStrategy(
                    tp=tp, pp=pp, dp=dp,
                    activation_checkpointing=True,
                    micro_batch_size=1,
                    num_micro_batches=max(pp * 2, 4) if pp > 1 else 1,
                ))
        
        # Add context parallel options for long sequences
        if model.max_position_embeddings > 32768:
            for tp in [tp for tp in optimal_tp_sizes if tp >= 2]:
                cp_options = [2, 4, 8]
                for cp in cp_options:
                    if tp * cp > num_gpus:
                        continue
                    dp = num_gpus // (tp * cp)
                    if dp >= 1:
                        candidates.append(ParallelismStrategy(
                            tp=tp, pp=1, dp=dp, cp=cp,
                        ))
        
        # Add expert parallel options for MoE models
        if model.model_type in (ModelType.MOE, ModelType.HYBRID_MOE):
            for tp in optimal_tp_sizes[:3]:  # Limit TP options for MoE
                # EP should divide num_experts evenly
                ep_options = [
                    e for e in [1, 2, 4, 8, 16]
                    if model.num_experts % e == 0 and e <= num_gpus // tp
                ]
                for ep in ep_options:
                    remaining = num_gpus // (tp * ep)
                    for pp in [1, 2, 4]:
                        dp = remaining // pp
                        if dp >= 1 and tp * pp * dp * ep == num_gpus:
                            candidates.append(ParallelismStrategy(
                                tp=tp, pp=pp, dp=dp, ep=ep,
                            ))
        
        return candidates
    
    def _analyze_strategy(
        self,
        strategy: ParallelismStrategy,
        topology: TopologyInfo,
        model: ModelArchitecture,
        batch_size: int,
        seq_length: int,
        is_training: bool,
    ) -> StrategyAnalysis:
        """Analyze a strategy's performance characteristics."""
        
        # Memory estimation
        mem = model.estimate_memory_gb(
            batch_size=strategy.micro_batch_size,
            seq_length=seq_length,
            include_optimizer=is_training,
        )
        
        # Adjust for parallelism
        # TP splits weights and activations
        weights_per_gpu = mem["weights_gb"] / strategy.tp
        # PP splits layers
        if strategy.pp > 1:
            weights_per_gpu = weights_per_gpu / strategy.pp
        # EP splits experts (MoE models)
        if strategy.ep > 1 and model.num_experts > 1:
            # With EP, expert params are split across EP GPUs
            # Non-expert params (attention, embeddings) remain replicated
            moe_weight_fraction = 0.0
            if model.num_moe_layers > 0:
                # Rough estimate: MoE FFN is ~3x hidden*intermediate per expert
                ffn_params_per_expert = 3 * model.hidden_size * model.intermediate_size
                total_expert_params = ffn_params_per_expert * model.num_experts * model.num_moe_layers
                moe_weight_fraction = min(0.95, total_expert_params / (model.total_params_billion * 1e9))
            # Non-MoE weights stay, MoE weights split by EP
            weights_per_gpu = weights_per_gpu * ((1 - moe_weight_fraction) + moe_weight_fraction / strategy.ep)
        
        kv_cache_per_gpu = mem["kv_cache_gb"]
        # CP splits sequence dimension
        if strategy.cp > 1:
            kv_cache_per_gpu /= strategy.cp
        
        activations_per_gpu = mem["activations_gb"] / strategy.tp
        if strategy.activation_checkpointing:
            activations_per_gpu *= (1 - self.ACTIVATION_CHECKPOINT_SAVINGS)
        
        optimizer_per_gpu = mem["optimizer_gb"] / strategy.tp if is_training else 0
        if strategy.pp > 1:
            optimizer_per_gpu /= strategy.pp
        
        memory_per_gpu = (
            weights_per_gpu + kv_cache_per_gpu + activations_per_gpu + optimizer_per_gpu
        )
        
        gpu_memory = topology.gpus[0].memory_gb if topology.gpus else 80
        headroom = gpu_memory - memory_per_gpu
        fits = headroom > 2.0  # At least 2GB headroom
        
        # Communication overhead estimation
        # TP: 2 all-reduces per layer
        tp_overhead = 0.0
        if strategy.tp > 1:
            # Volume: 2 * batch * seq * hidden per layer
            volume_per_layer = 2 * batch_size * seq_length * model.hidden_size * model.dtype_bytes
            total_volume = volume_per_layer * model.num_layers / (1024**3)  # GB
            avg_bandwidth = topology.max_nvlink_bandwidth_gbps if topology.has_nvlink else 32
            tp_overhead = total_volume / avg_bandwidth * 1000  # ms
        
        # PP: Point-to-point per micro-batch
        pp_overhead = 0.0
        if strategy.pp > 1:
            volume = batch_size * seq_length * model.hidden_size * model.dtype_bytes / (1024**3)
            pp_overhead = volume * strategy.num_micro_batches / 50 * 1000  # Assume 50 GB/s
        
        # DP: All-reduce gradients
        dp_overhead = 0.0
        if strategy.dp > 1 and is_training:
            volume = model.total_params_billion * 1e9 * model.dtype_bytes / (1024**3)
            dp_overhead = volume / topology.max_nvlink_bandwidth_gbps * 1000
        
        # Pipeline bubble
        bubble_fraction = 0.0
        if strategy.pp > 1:
            bubble_fraction = (strategy.pp - 1) / (strategy.num_micro_batches + strategy.pp - 1)
        
        # Compute time estimation (very rough)
        # Based on ~2 TFLOPs per parameter per token for transformer forward pass
        flops_per_token = model.active_params_billion * 1e9 * 2
        # B200: ~2.5 PFLOPS FP16, per GPU. Only the exact "blackwell" label (SM 10.0
        # and 10.3) gets this estimate. SM 12.x GPUs are labeled "blackwell_sm12x".
        gpu_tflops = 2500 if topology.gpus[0].architecture == "blackwell" else 1000
        
        tokens_per_gpu = batch_size * seq_length / max(strategy.cp, 1)
        compute_ms = (flops_per_token * tokens_per_gpu) / (gpu_tflops * 1e12) * 1000
        if is_training:
            compute_ms *= 3  # Forward + backward
        
        total_overhead = tp_overhead + pp_overhead + dp_overhead
        comm_fraction = total_overhead / (compute_ms + total_overhead) if compute_ms > 0 else 0
        
        # Throughput estimate
        total_time_ms = compute_ms + total_overhead
        if strategy.pp > 1:
            total_time_ms *= (1 + bubble_fraction)
        
        throughput = batch_size * seq_length * strategy.dp / (total_time_ms / 1000) if total_time_ms > 0 else 0
        
        # Determine bottleneck
        bottleneck = "compute"
        if not fits:
            bottleneck = "memory"
        elif tp_overhead > compute_ms * 0.3:
            bottleneck = "tp_comm"
        elif pp_overhead > compute_ms * 0.3:
            bottleneck = "pp_comm"
        elif dp_overhead > compute_ms * 0.3:
            bottleneck = "dp_comm"
        
        return StrategyAnalysis(
            memory_per_gpu_gb=memory_per_gpu,
            peak_memory_gb=memory_per_gpu * 1.1,  # Add 10% for fragmentation
            memory_headroom_gb=headroom,
            fits_in_memory=fits,
            estimated_throughput_tps=throughput,
            estimated_latency_ms=total_time_ms,
            compute_efficiency=1 - comm_fraction - bubble_fraction,
            communication_overhead=comm_fraction,
            pipeline_bubble_fraction=bubble_fraction,
            bottleneck=bottleneck,
        )
    
    def _score_strategy(
        self,
        strategy: ParallelismStrategy,
        analysis: StrategyAnalysis,
        topology: TopologyInfo,
        model: ModelArchitecture,
        goal: OptimizationGoal,
        is_training: bool,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """Score a strategy and generate rationale."""
        score = 50.0  # Base score
        rationale = []
        warnings = []
        optimizations = []
        
        # Memory efficiency score (0-25 points)
        if analysis.fits_in_memory:
            mem_score = min(25, analysis.memory_headroom_gb * 2)
            score += mem_score
            if analysis.memory_headroom_gb > 20:
                rationale.append(f"Excellent memory headroom ({analysis.memory_headroom_gb:.1f} GB)")
            elif analysis.memory_headroom_gb > 10:
                rationale.append(f"Good memory headroom ({analysis.memory_headroom_gb:.1f} GB)")
            else:
                warnings.append(f"Tight memory ({analysis.memory_headroom_gb:.1f} GB headroom)")
                if not strategy.activation_checkpointing:
                    optimizations.append("Enable activation checkpointing to reduce memory")
        else:
            score -= 50
            warnings.append("Does not fit in GPU memory")
            optimizations.append("Increase PP or enable activation checkpointing")
        
        # Throughput score (0-25 points)
        if goal == OptimizationGoal.THROUGHPUT:
            # Higher throughput = higher score
            throughput_score = min(25, analysis.estimated_throughput_tps / 1000)
            score += throughput_score
            if analysis.estimated_throughput_tps > 10000:
                rationale.append(f"High throughput ({analysis.estimated_throughput_tps:.0f} tokens/s)")
        
        # Latency score (for inference, 0-25 points)
        if goal == OptimizationGoal.LATENCY:
            if analysis.estimated_latency_ms < 50:
                score += 25
                rationale.append(f"Low latency ({analysis.estimated_latency_ms:.1f} ms)")
            elif analysis.estimated_latency_ms < 100:
                score += 15
            else:
                warnings.append(f"High latency ({analysis.estimated_latency_ms:.1f} ms)")
        
        # Communication efficiency (0-15 points)
        if analysis.communication_overhead < 0.1:
            score += 15
            rationale.append("Minimal communication overhead")
        elif analysis.communication_overhead < 0.2:
            score += 10
        elif analysis.communication_overhead > 0.3:
            warnings.append(f"High communication overhead ({analysis.communication_overhead:.1%})")
            if strategy.tp > 2 and not topology.has_nvswitch:
                optimizations.append("Reduce TP size - limited NVLink bandwidth")
        
        # Pipeline bubble (0-10 points)
        if strategy.pp > 1:
            if analysis.pipeline_bubble_fraction < 0.1:
                score += 10
                rationale.append("Low pipeline bubble")
            elif analysis.pipeline_bubble_fraction > 0.2:
                warnings.append(f"Pipeline bubble: {analysis.pipeline_bubble_fraction:.1%}")
                optimizations.append(f"Increase micro-batches to {strategy.pp * 4} to reduce bubble")
        
        # Topology alignment bonus
        if topology.has_nvswitch:
            if strategy.tp in [2, 4, 8]:
                score += 5
                rationale.append(f"TP={strategy.tp} optimal with NVSwitch")
        elif topology.has_nvlink:
            if strategy.tp == 2:
                score += 5
                rationale.append("TP=2 optimal for NVLink pairs")
            elif strategy.tp > 4:
                warnings.append("TP>4 may be limited by NVLink topology")
        else:
            if strategy.tp > 2:
                warnings.append("PCIe-only: large TP will be slow")
                optimizations.append("Prefer PP over TP for PCIe systems")
        
        # MoE-specific scoring
        if model.model_type in (ModelType.MOE, ModelType.HYBRID_MOE):
            if strategy.ep > 1:
                if model.num_experts % strategy.ep == 0:
                    score += 5
                    rationale.append(f"EP={strategy.ep} evenly divides {model.num_experts} experts")
                else:
                    warnings.append(f"EP={strategy.ep} doesn't evenly divide experts")
            else:
                if model.num_experts > 8:
                    optimizations.append("Consider EP for better expert distribution")
        
        # Context parallel for long sequences
        if model.max_position_embeddings > 65536:
            if strategy.cp > 1:
                score += 5
                rationale.append(f"CP={strategy.cp} enables long sequences")
            else:
                optimizations.append("Consider CP for long context support")
        
        # Grace-Blackwell specific optimizations
        if topology.is_grace_cpu and topology.has_nvlink_c2c:
            if strategy.dp >= 2:
                score += 3
                rationale.append("Can leverage NVLink-C2C for DP communication")
        
        # Clamp score
        score = max(0, min(100, score))
        
        return score, rationale, warnings, optimizations
    
    def format_recommendations(
        self, recommendations: List[StrategyRecommendation]
    ) -> str:
        """Format recommendations as a human-readable report."""
        if not recommendations:
            return "No valid strategies found for the given constraints."
        
        lines = [
            "=" * 80,
            "PARALLELISM STRATEGY RECOMMENDATIONS",
            "=" * 80,
        ]
        
        for i, rec in enumerate(recommendations, 1):
            s = rec.strategy
            a = rec.analysis
            
            lines.extend([
                "",
                f"{'─' * 80}",
                f"OPTION {i}: TP={s.tp} × PP={s.pp} × DP={s.dp}"
                + (f" × CP={s.cp}" if s.cp > 1 else "")
                + (f" × EP={s.ep}" if s.ep > 1 else "")
                + f"  [Score: {rec.score:.0f}/100]",
                f"{'─' * 80}",
                "",
                f"  World Size: {s.world_size} GPUs",
                f"  Memory/GPU: {a.memory_per_gpu_gb:.1f} GB (headroom: {a.memory_headroom_gb:.1f} GB)",
                f"  Est. Throughput: {a.estimated_throughput_tps:,.0f} tokens/s",
                f"  Compute Efficiency: {a.compute_efficiency:.1%}",
            ])
            
            if s.pp > 1:
                lines.append(f"  Pipeline Bubble: {a.pipeline_bubble_fraction:.1%}")
            
            lines.append(f"  Bottleneck: {a.bottleneck}")
            
            if rec.rationale:
                lines.append("")
                lines.append("  ✓ Rationale:")
                for r in rec.rationale:
                    lines.append(f"    • {r}")
            
            if rec.warnings:
                lines.append("")
                lines.append("  ⚠ Warnings:")
                for w in rec.warnings:
                    lines.append(f"    • {w}")
            
            if rec.optimizations:
                lines.append("")
                lines.append("  💡 Optimizations:")
                for o in rec.optimizations:
                    lines.append(f"    • {o}")
        
        lines.extend(["", "=" * 80])
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo with mock data
    from .advisor import create_mock_topology_b200_multigpu
    from .model_analyzer import ModelAnalyzer

    topology = create_mock_topology_b200_multigpu(num_gpus=4).materialize()
    
    analyzer = ModelAnalyzer()
    model = analyzer.analyze("llama-3.1-70b")
    
    optimizer = StrategyOptimizer()
    recommendations = optimizer.recommend(
        topology, model,
        goal=OptimizationGoal.THROUGHPUT,
        batch_size=8,
        seq_length=4096,
        is_training=True,
    )
    
    print(optimizer.format_recommendations(recommendations))
