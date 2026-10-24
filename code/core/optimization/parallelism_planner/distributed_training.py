#!/usr/bin/env python3
"""
Distributed Training Optimization Module

Comprehensive support for large-scale distributed training:
- NCCL tuning advisor
- Multi-node network diagnostics
- Communication overlap analysis
- RLHF/DPO memory calculator
- MoE parallelism optimizer
- Long context optimization (Ring Attention, Context Parallel)
- vLLM configuration generator
- Gradient accumulation optimization
- Distributed checkpointing strategies
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class NCCLBackend(Enum):
    """NCCL transport backends."""
    NVLINK = "nvlink"
    PCIE = "pcie"
    INFINIBAND = "infiniband"
    ROCE = "roce"
    TCP = "tcp"


class RLHFAlgorithm(Enum):
    """RLHF training algorithms."""
    PPO = "ppo"
    DPO = "dpo"
    ORPO = "orpo"
    KTO = "kto"
    REINFORCE = "reinforce"
    GRPO = "grpo"


class MoEStrategy(Enum):
    """MoE parallelism strategies."""
    EXPERT_PARALLEL = "ep"
    TENSOR_EXPERT = "tp_ep"
    DATA_EXPERT = "dp_ep"
    FULL_REPLICATE = "replicate"


@dataclass
class NCCLConfig:
    """NCCL configuration recommendation."""
    env_vars: Dict[str, str]
    description: str
    expected_bandwidth_gb_s: float
    warnings: List[str] = field(default_factory=list)
    optimizations: List[str] = field(default_factory=list)


@dataclass
class RLHFMemoryEstimate:
    """Memory estimate for RLHF training."""
    algorithm: RLHFAlgorithm
    actor_memory_gb: float
    critic_memory_gb: float
    reference_memory_gb: float
    reward_memory_gb: float
    optimizer_memory_gb: float
    activation_memory_gb: float
    total_memory_gb: float
    fits_single_gpu: bool
    recommended_tp: int
    recommended_offload: bool
    optimizations: List[str] = field(default_factory=list)


@dataclass
class MoEConfig:
    """MoE parallelism configuration."""
    num_experts: int
    experts_per_rank: int
    expert_parallel_size: int
    tensor_parallel_size: int
    data_parallel_size: int
    capacity_factor: float
    load_balancing_loss_weight: float
    memory_per_gpu_gb: float
    communication_volume_gb: float
    expected_mfu: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class LongContextConfig:
    """Long context optimization configuration."""
    method: str  # "ring_attention", "context_parallel", "flash_attention_v3"
    sequence_length: int
    context_parallel_size: int
    ring_attention_heads: int
    memory_savings_pct: float
    communication_overhead_pct: float
    expected_throughput_tokens_s: float
    launch_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VLLMConfig:
    """vLLM server configuration."""
    model: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    quantization: Optional[str]
    kv_cache_dtype: str
    enforce_eager: bool
    enable_chunked_prefill: bool
    enable_prefix_caching: bool
    speculative_model: Optional[str]
    speculative_num_draft_tokens: int
    launch_command: str
    estimated_throughput_tokens_s: float
    estimated_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize planner output for CLI/MCP/API surfaces."""
        return {
            "model": self.model,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "quantization": self.quantization,
            "kv_cache_dtype": self.kv_cache_dtype,
            "enforce_eager": self.enforce_eager,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "enable_prefix_caching": self.enable_prefix_caching,
            "speculative_model": self.speculative_model,
            "speculative_num_draft_tokens": self.speculative_num_draft_tokens,
            "launch_command": self.launch_command,
            "estimated_throughput_tokens_s": self.estimated_throughput_tokens_s,
            "estimated_latency_ms": self.estimated_latency_ms,
        }


class NCCLTuningAdvisor:
    """Recommends optimal NCCL configuration for distributed training."""
    
    def __init__(self):
        self.detected_backend = self._detect_backend()
    
    def _detect_backend(self) -> NCCLBackend:
        """Detect available NCCL backend."""
        # Check for NVLink
        try:
            result = subprocess.run(
                ["nvidia-smi", "nvlink", "-s"],
                capture_output=True, text=True, timeout=5
            )
            if "NVLink" in result.stdout:
                return NCCLBackend.NVLINK
        except Exception:
            pass
        
        # Check for InfiniBand
        try:
            result = subprocess.run(
                ["ibstat"],
                capture_output=True, text=True, timeout=5
            )
            if "Active" in result.stdout:
                return NCCLBackend.INFINIBAND
        except Exception:
            pass
        
        return NCCLBackend.TCP
    
    def get_optimal_config(
        self,
        num_nodes: int = 1,
        gpus_per_node: int = 8,
        model_size_b: float = 70,
        tp_size: int = 1,
        pp_size: int = 1,
    ) -> NCCLConfig:
        """Get optimal NCCL configuration."""
        
        env_vars = {
            # Core NCCL settings
            "NCCL_DEBUG": "WARN",
            "NCCL_TIMEOUT": "1800",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            
            # Memory settings
            "NCCL_BUFFSIZE": "8388608",  # 8MB
            "NCCL_NTHREADS": "512",
            
            # P2P settings
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
        
        optimizations = []
        warnings = []
        bandwidth = 25.0  # Default PCIe Gen4
        
        if self.detected_backend == NCCLBackend.NVLINK:
            env_vars.update({
                "NCCL_P2P_LEVEL": "NVL",
                "NCCL_P2P_DISABLE": "0",
                "NCCL_NVLS_ENABLE": "1",  # NVLink SHARP
            })
            optimizations.append("NVLink detected - enabling NVL P2P and NVLS")
            bandwidth = 450.0 if gpus_per_node == 8 else 300.0
            
        elif self.detected_backend == NCCLBackend.INFINIBAND:
            env_vars.update({
                "NCCL_IB_DISABLE": "0",
                "NCCL_IB_HCA": "^mlx5",
                "NCCL_IB_TIMEOUT": "23",
                "NCCL_IB_RETRY_CNT": "7",
                "NCCL_IB_GID_INDEX": "3",  # RoCE v2
                "NCCL_NET_GDR_LEVEL": "5",  # GPUDirect RDMA
                "NCCL_NET_GDR_READ": "1",
            })
            optimizations.append("InfiniBand detected - enabling IB with GPUDirect RDMA")
            bandwidth = 50.0  # HDR 200Gbps
            
        else:
            env_vars.update({
                "NCCL_SOCKET_IFNAME": "eth0",
                "NCCL_SOCKET_NTHREADS": "8",
                "NCCL_NSOCKS_PERTHREAD": "4",
            })
            warnings.append("No high-speed interconnect detected - falling back to TCP")
            bandwidth = 12.0
        
        # Multi-node optimizations
        if num_nodes > 1:
            env_vars.update({
                "NCCL_MIN_NCHANNELS": "4",
                "NCCL_MAX_NCHANNELS": "32",
                "NCCL_CROSS_NIC": "1",
            })
            optimizations.append(f"Multi-node ({num_nodes} nodes) - enabling cross-NIC")
        
        # Large model optimizations
        if model_size_b > 100:
            env_vars["NCCL_BUFFSIZE"] = "16777216"  # 16MB for large models
            optimizations.append("Large model (>100B) - increased buffer size")
        
        # TP optimizations
        if tp_size > 1:
            env_vars["NCCL_ALGO"] = "Ring,Tree"
            optimizations.append(f"TP={tp_size} - using Ring+Tree algorithms")
        
        description = f"""
NCCL Configuration for {num_nodes}x{gpus_per_node} GPUs ({model_size_b}B model)
Backend: {self.detected_backend.value}
TP={tp_size}, PP={pp_size}, DP={num_nodes * gpus_per_node // (tp_size * pp_size)}
Expected collective bandwidth: {bandwidth:.1f} GB/s
"""
        
        return NCCLConfig(
            env_vars=env_vars,
            description=description.strip(),
            expected_bandwidth_gb_s=bandwidth,
            warnings=warnings,
            optimizations=optimizations,
        )
    
    def diagnose_issues(self) -> Dict[str, Any]:
        """Diagnose common NCCL issues."""
        issues = []
        recommendations = []
        
        # Check NCCL version
        try:
            import torch.cuda.nccl as nccl
            version = nccl.version()
            if version < (2, 19, 0):
                issues.append(f"NCCL version {version} is outdated")
                recommendations.append("Upgrade to NCCL 2.19+ for better performance")
        except Exception:
            pass
        
        # Check GPU topology
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True, text=True, timeout=10
            )
            if "PIX" in result.stdout or "PXB" in result.stdout:
                issues.append("GPUs connected via PCIe switch (not NVLink)")
                recommendations.append("Consider servers with NVLink for TP")
        except Exception:
            pass
        
        # Check InfiniBand status
        try:
            result = subprocess.run(
                ["ibstat"],
                capture_output=True, text=True, timeout=5
            )
            if "Down" in result.stdout:
                issues.append("InfiniBand port(s) down")
                recommendations.append("Check IB cable connections and subnet manager")
        except Exception:
            pass
        
        return {
            "issues": issues,
            "recommendations": recommendations,
            "backend": self.detected_backend.value,
        }


class RLHFMemoryCalculator:
    """Calculate memory requirements for RLHF training."""
    
    # Memory overhead factors
    ACTOR_OVERHEAD = 1.0
    CRITIC_OVERHEAD = 1.0
    REFERENCE_OVERHEAD = 0.5  # Can be FP16/offloaded
    REWARD_OVERHEAD = 0.5
    OPTIMIZER_STATES = 12  # AdamW: 12 bytes per param for FP32
    
    def calculate(
        self,
        model_params_b: float,
        seq_length: int = 2048,
        batch_size: int = 4,
        algorithm: RLHFAlgorithm = RLHFAlgorithm.PPO,
        gpu_memory_gb: float = 80,
        precision: str = "bf16",
    ) -> RLHFMemoryEstimate:
        """Calculate memory requirements for RLHF."""
        
        bytes_per_param = 2 if precision in ["fp16", "bf16"] else 4
        
        # Base model memory
        base_model_gb = model_params_b * bytes_per_param
        
        # Algorithm-specific memory
        if algorithm == RLHFAlgorithm.PPO:
            # PPO: actor + critic + reference + reward
            actor_mem = base_model_gb * self.ACTOR_OVERHEAD
            critic_mem = base_model_gb * self.CRITIC_OVERHEAD
            reference_mem = base_model_gb * self.REFERENCE_OVERHEAD
            reward_mem = base_model_gb * self.REWARD_OVERHEAD
            
        elif algorithm == RLHFAlgorithm.DPO:
            # DPO: actor + reference only
            actor_mem = base_model_gb * self.ACTOR_OVERHEAD
            critic_mem = 0
            reference_mem = base_model_gb * self.REFERENCE_OVERHEAD
            reward_mem = 0
            
        elif algorithm == RLHFAlgorithm.ORPO:
            # ORPO: actor only (no reference needed)
            actor_mem = base_model_gb * self.ACTOR_OVERHEAD
            critic_mem = 0
            reference_mem = 0
            reward_mem = 0
            
        elif algorithm == RLHFAlgorithm.KTO:
            # KTO: actor + reference
            actor_mem = base_model_gb * self.ACTOR_OVERHEAD
            critic_mem = 0
            reference_mem = base_model_gb * self.REFERENCE_OVERHEAD
            reward_mem = 0
        
        else:
            # Default to PPO-like
            actor_mem = base_model_gb
            critic_mem = base_model_gb * 0.5
            reference_mem = base_model_gb * 0.5
            reward_mem = 0
        
        # Optimizer states (for trainable models)
        optimizer_mem = (model_params_b * self.OPTIMIZER_STATES) / 1e9
        if algorithm in [RLHFAlgorithm.PPO]:
            optimizer_mem *= 2  # Actor + Critic
        
        # Activation memory (rough estimate)
        # ~2 bytes per param per layer per batch element
        hidden_size = int((model_params_b * 1e9 / 100) ** 0.5 * 128)  # Rough estimate
        num_layers = int(model_params_b * 1e9 / (hidden_size ** 2 * 12))
        activation_mem = (batch_size * seq_length * hidden_size * num_layers * 2) / 1e9
        
        total_mem = actor_mem + critic_mem + reference_mem + reward_mem + optimizer_mem + activation_mem
        
        # Recommendations
        fits_single = total_mem < gpu_memory_gb * 0.9
        recommended_tp = 1
        while total_mem / recommended_tp > gpu_memory_gb * 0.8:
            recommended_tp *= 2
            if recommended_tp > 8:
                break
        
        optimizations = []
        if not fits_single:
            optimizations.append(f"Use TP={recommended_tp} to fit in memory")
            
        if reference_mem > 0:
            optimizations.append("Offload reference model to CPU (save ~{:.1f}GB)".format(reference_mem))
            
        if algorithm == RLHFAlgorithm.PPO and model_params_b > 30:
            optimizations.append("Consider DPO instead of PPO (50% less memory)")
            optimizations.append("Use gradient checkpointing for activations")
            
        if batch_size > 1:
            optimizations.append(f"Use gradient accumulation: batch_size=1, grad_accum={batch_size}")
        
        return RLHFMemoryEstimate(
            algorithm=algorithm,
            actor_memory_gb=round(actor_mem, 2),
            critic_memory_gb=round(critic_mem, 2),
            reference_memory_gb=round(reference_mem, 2),
            reward_memory_gb=round(reward_mem, 2),
            optimizer_memory_gb=round(optimizer_mem, 2),
            activation_memory_gb=round(activation_mem, 2),
            total_memory_gb=round(total_mem, 2),
            fits_single_gpu=fits_single,
            recommended_tp=recommended_tp,
            recommended_offload=not fits_single,
            optimizations=optimizations,
        )
    
    def get_optimal_config(
        self,
        model_params_b: float,
        num_gpus: int = 8,
        gpu_memory_gb: float = 80,
    ) -> Dict[str, Any]:
        """Get optimal RLHF configuration."""
        
        results = {}
        for algo in [RLHFAlgorithm.PPO, RLHFAlgorithm.DPO, RLHFAlgorithm.ORPO]:
            estimate = self.calculate(model_params_b, algorithm=algo, gpu_memory_gb=gpu_memory_gb)
            results[algo.value] = {
                "total_memory_gb": estimate.total_memory_gb,
                "fits_single_gpu": estimate.fits_single_gpu,
                "recommended_tp": estimate.recommended_tp,
                "efficiency": "high" if estimate.fits_single_gpu else "medium",
            }
        
        # Recommend best algorithm
        best = min(results.items(), key=lambda x: x[1]["total_memory_gb"])
        
        return {
            "model_params_b": model_params_b,
            "num_gpus": num_gpus,
            "algorithms": results,
            "recommended": best[0],
            "recommendation_reason": f"{best[0].upper()} uses least memory ({best[1]['total_memory_gb']:.1f}GB)",
        }


class MoEOptimizer:
    """Optimize Mixture of Experts parallelism."""
    
    def optimize(
        self,
        model_params_b: float,
        num_experts: int = 8,
        num_gpus: int = 8,
        gpu_memory_gb: float = 80,
        batch_size: int = 8,
        seq_length: int = 4096,
    ) -> MoEConfig:
        """Optimize MoE parallelism configuration."""
        
        # Expert parameters (rough estimate - experts are ~2/3 of model in typical MoE)
        expert_params_b = model_params_b * 0.67 / num_experts
        shared_params_b = model_params_b * 0.33
        
        # Memory per expert
        expert_mem_gb = expert_params_b * 2  # FP16
        shared_mem_gb = shared_params_b * 2
        
        # Find optimal parallelism
        # Rule: EP should divide num_experts evenly
        valid_ep_sizes = [e for e in [1, 2, 4, 8] if num_experts % e == 0 and e <= num_gpus]
        
        best_config = None
        best_mfu = 0
        
        for ep in valid_ep_sizes:
            experts_per_rank = num_experts // ep
            remaining_gpus = num_gpus // ep
            
            # Try different TP sizes
            for tp in [1, 2, 4, 8]:
                if tp > remaining_gpus:
                    continue
                    
                dp = remaining_gpus // tp
                if dp < 1:
                    continue
                
                # Memory calculation
                mem_per_gpu = shared_mem_gb / tp + expert_mem_gb * experts_per_rank
                
                if mem_per_gpu > gpu_memory_gb * 0.8:
                    continue
                
                # Communication volume
                # All-to-all for expert dispatch: batch_size * seq_length * hidden_size * 2
                hidden_size = int((model_params_b * 1e9 / 100) ** 0.5 * 128)
                all_to_all_volume = batch_size * seq_length * hidden_size * 2 / 1e9  # GB
                
                # Estimate MFU (rough)
                compute_efficiency = 0.4 if ep > 1 else 0.5
                comm_penalty = 0.1 * (ep - 1)
                mfu = compute_efficiency - comm_penalty
                
                if mfu > best_mfu:
                    best_mfu = mfu
                    best_config = MoEConfig(
                        num_experts=num_experts,
                        experts_per_rank=experts_per_rank,
                        expert_parallel_size=ep,
                        tensor_parallel_size=tp,
                        data_parallel_size=dp,
                        capacity_factor=1.25,  # Standard default
                        load_balancing_loss_weight=0.01,
                        memory_per_gpu_gb=round(mem_per_gpu, 2),
                        communication_volume_gb=round(all_to_all_volume, 2),
                        expected_mfu=round(mfu, 3),
                        warnings=[],
                    )
        
        if best_config is None:
            # Fallback config
            best_config = MoEConfig(
                num_experts=num_experts,
                experts_per_rank=num_experts,
                expert_parallel_size=1,
                tensor_parallel_size=min(8, num_gpus),
                data_parallel_size=max(1, num_gpus // 8),
                capacity_factor=1.25,
                load_balancing_loss_weight=0.01,
                memory_per_gpu_gb=model_params_b * 2 / num_gpus,
                communication_volume_gb=0,
                expected_mfu=0.3,
                warnings=["Model too large for efficient MoE - consider more GPUs"],
            )
        
        return best_config


class LongContextOptimizer:
    """Optimize for long context training/inference."""
    
    def optimize(
        self,
        model_params_b: float,
        target_seq_length: int = 128000,
        num_gpus: int = 8,
        gpu_memory_gb: float = 80,
        method: str = "auto",
    ) -> LongContextConfig:
        """Get optimal long context configuration."""
        
        # Estimate attention memory
        hidden_size = int((model_params_b * 1e9 / 100) ** 0.5 * 128)
        num_heads = hidden_size // 128
        head_dim = 128
        
        # QKV memory for full attention: 3 * batch * seq * hidden * 2 bytes
        # Attention scores: batch * heads * seq * seq * 2 bytes
        batch_size = 1  # Start with batch=1 for long context
        qkv_mem = 3 * batch_size * target_seq_length * hidden_size * 2 / 1e9
        attn_mem = batch_size * num_heads * target_seq_length * target_seq_length * 2 / 1e9
        
        # Determine best method
        if method == "auto":
            if attn_mem > gpu_memory_gb * 0.5:
                if num_gpus >= 4:
                    method = "context_parallel"
                else:
                    method = "ring_attention"
            else:
                method = "flash_attention_v3"
        
        # Configure based on method
        if method == "context_parallel":
            # Split sequence across GPUs
            cp_size = min(num_gpus, target_seq_length // 4096)  # Min 4K per GPU
            seq_per_gpu = target_seq_length // cp_size
            
            # Memory with CP
            actual_attn_mem = batch_size * num_heads * seq_per_gpu * seq_per_gpu * 2 / 1e9
            memory_savings = 1 - (actual_attn_mem / attn_mem)
            comm_overhead = 0.15 * cp_size  # Ring communication overhead
            
            config = LongContextConfig(
                method="context_parallel",
                sequence_length=target_seq_length,
                context_parallel_size=cp_size,
                ring_attention_heads=0,
                memory_savings_pct=round(memory_savings * 100, 1),
                communication_overhead_pct=round(comm_overhead * 100, 1),
                expected_throughput_tokens_s=target_seq_length * batch_size / (1 + comm_overhead),
                launch_args={
                    "context_parallel_size": cp_size,
                    "sequence_parallel": True,
                    "cp_comm_type": "ring",
                },
            )
            
        elif method == "ring_attention":
            # Ring attention - process chunks in ring pattern
            chunk_size = 8192
            num_chunks = target_seq_length // chunk_size
            
            config = LongContextConfig(
                method="ring_attention",
                sequence_length=target_seq_length,
                context_parallel_size=1,
                ring_attention_heads=num_heads,
                memory_savings_pct=round((1 - 1/num_chunks) * 100, 1),
                communication_overhead_pct=10.0,  # P2P ring comm
                expected_throughput_tokens_s=target_seq_length * batch_size * 0.8,
                launch_args={
                    "ring_attention": True,
                    "ring_chunk_size": chunk_size,
                    "ring_head_stride": 1,
                },
            )
            
        else:  # flash_attention_v3
            config = LongContextConfig(
                method="flash_attention_v3",
                sequence_length=target_seq_length,
                context_parallel_size=1,
                ring_attention_heads=0,
                memory_savings_pct=80.0,  # FA is memory efficient
                communication_overhead_pct=0.0,
                expected_throughput_tokens_s=target_seq_length * batch_size,
                launch_args={
                    "use_flash_attention": True,
                    "flash_attention_version": 3,
                    "flash_attention_causal": True,
                },
            )
        
        return config


class VLLMConfigGenerator:
    """Generate optimal vLLM configurations."""
    
    def generate(
        self,
        model: str,
        model_params_b: float,
        num_gpus: int = 1,
        gpu_memory_gb: float = 80,
        target: str = "throughput",  # "throughput", "latency", "memory"
        max_seq_length: int = 8192,
        quantization: Optional[str] = None,  # "awq", "gptq", "fp8", "int8"
    ) -> VLLMConfig:
        """Generate optimal vLLM configuration."""
        
        # Calculate model memory
        bytes_per_param = {
            None: 2,  # FP16/BF16
            "awq": 0.5,
            "gptq": 0.5,
            "fp8": 1,
            "int8": 1,
        }.get(quantization, 2)
        
        model_mem_gb = model_params_b * bytes_per_param
        
        # Determine TP
        tp = 1
        while model_mem_gb / tp > gpu_memory_gb * 0.7:
            tp *= 2
        tp = min(tp, num_gpus)
        
        # PP (use if TP alone isn't enough)
        pp = 1
        if tp == num_gpus and model_mem_gb / tp > gpu_memory_gb * 0.8:
            pp = 2
            tp = num_gpus // 2
        
        # GPU memory utilization
        if target == "memory":
            gpu_util = 0.95
        elif target == "throughput":
            gpu_util = 0.9
        else:  # latency
            gpu_util = 0.85
        
        # KV cache memory and max sequences
        kv_cache_mem_gb = gpu_memory_gb * gpu_util - model_mem_gb / tp
        
        # Rough KV cache size estimate. Keep this heuristic simple and non-zero so
        # the config generator remains usable for smaller models and dry-run sizing.
        bytes_per_element = 1 if quantization == "fp8" else 2
        num_layers = max(1, int(model_params_b * 1.2))
        kv_per_token_bytes = max(1, 2 * num_layers * 8 * 128 * bytes_per_element)
        
        max_kv_tokens = int(kv_cache_mem_gb * 1e9 / kv_per_token_bytes)
        
        # Max sequences based on target
        if target == "throughput":
            max_num_seqs = min(256, max_kv_tokens // max_seq_length)
            max_batched_tokens = min(65536, max_kv_tokens)
        elif target == "latency":
            max_num_seqs = min(32, max_kv_tokens // max_seq_length)
            max_batched_tokens = min(8192, max_kv_tokens)
        else:
            max_num_seqs = min(128, max_kv_tokens // max_seq_length)
            max_batched_tokens = min(32768, max_kv_tokens)
        
        # Advanced features
        enable_chunked_prefill = target == "latency"
        enable_prefix_caching = target == "throughput"
        enforce_eager = target == "latency"
        
        # KV cache dtype
        kv_dtype = "auto"
        if quantization == "fp8":
            kv_dtype = "fp8"
        
        # Speculative decoding for latency
        speculative_model = None
        speculative_tokens = 0
        if target == "latency" and model_params_b > 30:
            speculative_tokens = 5
            # Use smaller draft model
            if "llama" in model.lower():
                speculative_model = "meta-llama/Llama-3.2-1B"
        
        # Build launch command
        cmd_parts = [
            "python -m vllm.entrypoints.openai.api_server",
            f"--model {model}",
            f"--tensor-parallel-size {tp}",
        ]
        
        if pp > 1:
            cmd_parts.append(f"--pipeline-parallel-size {pp}")
        
        cmd_parts.extend([
            f"--gpu-memory-utilization {gpu_util}",
            f"--max-model-len {max_seq_length}",
            f"--max-num-seqs {max_num_seqs}",
            f"--max-num-batched-tokens {max_batched_tokens}",
        ])
        
        if quantization:
            cmd_parts.append(f"--quantization {quantization}")
        
        if kv_dtype != "auto":
            cmd_parts.append(f"--kv-cache-dtype {kv_dtype}")
        
        if enable_chunked_prefill:
            cmd_parts.append("--enable-chunked-prefill")
        
        if enable_prefix_caching:
            cmd_parts.append("--enable-prefix-caching")
        
        if enforce_eager:
            cmd_parts.append("--enforce-eager")
        
        if speculative_model:
            cmd_parts.append(f"--speculative-model {speculative_model}")
            cmd_parts.append(f"--num-speculative-tokens {speculative_tokens}")
        
        launch_cmd = " \\\n    ".join(cmd_parts)
        
        # Estimate performance
        if target == "throughput":
            throughput = model_params_b * 0.5 * tp * 1000  # Very rough estimate
            latency = 50 + model_params_b * 0.5
        else:
            throughput = model_params_b * 0.2 * tp * 1000
            latency = 20 + model_params_b * 0.2
        
        return VLLMConfig(
            model=model,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            gpu_memory_utilization=gpu_util,
            max_model_len=max_seq_length,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_batched_tokens,
            quantization=quantization,
            kv_cache_dtype=kv_dtype,
            enforce_eager=enforce_eager,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
            speculative_model=speculative_model,
            speculative_num_draft_tokens=speculative_tokens,
            launch_command=launch_cmd,
            estimated_throughput_tokens_s=int(throughput),
            estimated_latency_ms=round(latency, 1),
        )
    
    def compare_engines(
        self,
        model: str,
        model_params_b: float,
        num_gpus: int = 1,
    ) -> Dict[str, Any]:
        """Compare different inference engines."""
        
        engines = {
            "vllm": {
                "strengths": ["Continuous batching", "PagedAttention", "Easy deployment"],
                "weaknesses": ["Not fastest for single request"],
                "best_for": "High-throughput serving",
                "setup": "Build the pinned vLLM source wheel in the isolated runtime. See docs/environment.md.",
            },
            "tensorrt_llm": {
                "strengths": ["Fastest inference", "INT8/FP8 optimization"],
                "weaknesses": ["Complex setup", "Model conversion required"],
                "best_for": "Maximum throughput with fixed models",
                "setup": "Use a separately pinned TensorRT-LLM environment. See docs/environment.md.",
            },
            "text_generation_inference": {
                "strengths": ["Flash Attention", "Easy HF integration"],
                "weaknesses": ["Less flexible than vLLM"],
                "best_for": "HuggingFace model serving",
                "setup": "docker pull ghcr.io/huggingface/text-generation-inference",
            },
            "llama_cpp": {
                "strengths": ["CPU inference", "Low memory", "GGUF format"],
                "weaknesses": ["Slower than GPU options"],
                "best_for": "Edge/CPU deployment",
                "setup": "pip install llama-cpp-python",
            },
        }
        
        # Recommendation based on model size and GPUs
        if model_params_b > 70:
            recommended = "vllm"
            reason = "Best for large models with continuous batching"
        elif num_gpus >= 4:
            recommended = "tensorrt_llm"
            reason = "Fastest for multi-GPU inference"
        else:
            recommended = "vllm"
            reason = "Best balance of speed and ease of use"
        
        return {
            "engines": engines,
            "recommended": recommended,
            "reason": reason,
        }


class CommunicationOverlapAnalyzer:
    """Analyze and optimize communication/computation overlap."""
    
    def analyze(
        self,
        model_params_b: float,
        tp_size: int,
        pp_size: int,
        dp_size: int,
        batch_size: int,
        seq_length: int,
    ) -> Dict[str, Any]:
        """Analyze communication overlap opportunities."""
        
        hidden_size = int((model_params_b * 1e9 / 100) ** 0.5 * 128)
        num_layers = int(model_params_b * 1e9 / (hidden_size ** 2 * 12))
        
        opportunities = []
        recommendations = []
        
        # AllReduce overlap (TP)
        if tp_size > 1:
            allreduce_bytes = batch_size * seq_length * hidden_size * 2
            opportunities.append({
                "type": "TP AllReduce",
                "volume_mb": allreduce_bytes / 1e6,
                "overlap_potential": "High - can overlap with FFN computation",
            })
            recommendations.append("Enable async allreduce for TP communication")
        
        # AllGather/ReduceScatter overlap (ZeRO/FSDP)
        if dp_size > 1:
            param_bytes = model_params_b * 1e9 * 2 / num_layers
            opportunities.append({
                "type": "FSDP AllGather",
                "volume_mb": param_bytes / 1e6,
                "overlap_potential": "Medium - prefetch next layer during compute",
            })
            recommendations.append("Use FSDP with backward_prefetch=BACKWARD_PRE")
        
        # P2P overlap (PP)
        if pp_size > 1:
            micro_batch_bytes = batch_size * seq_length * hidden_size * 2 / pp_size
            opportunities.append({
                "type": "PP P2P",
                "volume_mb": micro_batch_bytes / 1e6,
                "overlap_potential": "High with 1F1B schedule",
            })
            recommendations.append("Use interleaved 1F1B pipeline schedule")
        
        # Gradient accumulation
        if batch_size > 8:
            recommendations.append(f"Use gradient accumulation ({batch_size//8} steps) to reduce comm frequency")
        
        return {
            "opportunities": opportunities,
            "recommendations": recommendations,
            "overlap_configs": {
                "NCCL_LAUNCH_MODE": "GROUP",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_ASYNC_ERROR_HANDLING": "1",
            },
        }


# Export all classes
__all__ = [
    "NCCLBackend",
    "RLHFAlgorithm", 
    "MoEStrategy",
    "NCCLConfig",
    "RLHFMemoryEstimate",
    "MoEConfig",
    "LongContextConfig",
    "VLLMConfig",
    "NCCLTuningAdvisor",
    "RLHFMemoryCalculator",
    "MoEOptimizer",
    "LongContextOptimizer",
    "VLLMConfigGenerator",
    "CommunicationOverlapAnalyzer",
]

