"""
LLM-powered performance optimization advisor.

This module provides dynamic, context-aware recommendations using LLMs
instead of hard-coded suggestions. It uses profiling data and hardware
capabilities to generate intelligent, actionable optimization guidance.

Supports:
- OpenAI (GPT-4, GPT-4o)
- Anthropic (Claude Sonnet 4.6 by default; model overrides are supported)
- Local models via Ollama
- Fallback to rule-based recommendations when LLM unavailable
"""

import json
import os
import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from core.llm import DEFAULT_ANTHROPIC_MODEL
from core.utils.dotenv import find_repo_root, load_repo_dotenv

logger = logging.getLogger(__name__)

# =============================================================================
# KNOWLEDGE BASE - Optimization Techniques & Context
# =============================================================================

OPTIMIZATION_KNOWLEDGE = """
## GPU Performance Optimization Knowledge Base

### Memory Optimization Techniques
1. **Pinned Memory**: Use `torch.cuda.pin_memory()` for faster H2D transfers
2. **Memory Pools**: Use CUDA memory pools to reduce allocation overhead
3. **TMA (Tensor Memory Accelerator)**: Blackwell/Hopper async memory operations
4. **Unified Memory**: For memory-constrained scenarios with automatic migration

### Compute Optimization Techniques
1. **torch.compile**: Dynamic shapes, mode="reduce-overhead" for inference
2. **CUDA Graphs**: Capture and replay kernel sequences, reduces CPU overhead
3. **Kernel Fusion**: Combine elementwise ops, use custom Triton kernels
4. **FlashAttention**: Memory-efficient attention, O(N) instead of O(N²)
5. **FlexAttention**: Dynamic attention patterns with torch.compile

### Precision & Quantization
1. **FP8 (E4M3/E5M2)**: Blackwell/Hopper native support, 2x throughput
2. **FP4 (NF4)**: 4-bit quantization for extreme compression
3. **INT8**: For inference with calibration
4. **Mixed Precision**: FP16/BF16 with FP32 master weights

### Distributed Training
1. **DDP**: Data parallel, all-reduce gradients
2. **FSDP/FSDP2**: Fully sharded data parallel, memory efficient
3. **Tensor Parallelism (TP)**: Split model weights across GPUs
4. **Pipeline Parallelism (PP)**: Split layers across GPUs (GPipe, 1F1B, DualPipe)
5. **Expert Parallelism (EP)**: For MoE models
6. **ZeRO-1/2/3**: Optimizer, gradient, and parameter sharding
7. **Symmetric Memory**: Hopper+ feature for direct GPU-to-GPU access

### NCCL Tuning for Multi-GPU
1. **NCCL_ALGO**: ring, tree, collnetdirect, collnetchain
2. **NCCL_PROTO**: LL, LL128, SIMPLE
3. **NCCL_IB_HCA**: InfiniBand HCA selection
4. **NCCL_NET_GDR_LEVEL**: GPUDirect RDMA level
5. **NCCL_CROSS_NIC**: Cross-NIC communication

### Inference Optimization
1. **Continuous Batching**: vLLM/TRT-LLM style dynamic batching
2. **PagedAttention**: Memory-efficient KV cache management
3. **Speculative Decoding**: Draft model + verification
4. **KV Cache Compression**: Reduce memory for long contexts
5. **Prefix Caching**: SGLang RadixAttention approach
6. **Static Batching with Bucketing**: CUDA Graph compatible

### vLLM Specific
1. **PagedAttention v1/v2**: Block-based KV cache
2. **Chunked Prefill**: Overlap prefill with decode
3. **Prefix Caching**: Reuse KV for common prefixes
4. **Tensor Parallelism**: Model sharding
5. **AWQ/GPTQ/Marlin**: Quantization backends

### TensorRT-LLM Specific
1. **In-flight Batching**: Continuous batching
2. **KV Cache Reuse**: Multi-query attention optimization
3. **Custom Plugins**: For custom operations
4. **Weight Streaming**: For large models

### RL/RLHF Optimization
1. **PPO Optimization**: Vectorized environments, efficient rollouts
2. **Reference Model Sharing**: KV cache sharing between policy and reference
3. **Reward Model Batching**: Efficient reward computation
4. **Gradient Accumulation**: For large effective batch sizes
5. **DeepSpeed-Chat**: End-to-end RLHF training

### Blackwell/GB200 Specific
1. **FP4 Tensor Cores**: 4-bit native support
2. **5th Gen NVLink**: 1.8TB/s bidirectional
3. **NVLink-C2C**: 900GB/s GPU-to-CPU
4. **Decompression Engine**: Hardware decompression
5. **TMA Multicast**: Efficient broadcast to SM clusters
6. **WGMMA**: Warpgroup matrix multiply-accumulate

### Multi-Node Cluster Optimization
1. **SLURM Integration**: Job scheduling and resource allocation
2. **InfiniBand Tuning**: UCX, IB verbs optimization
3. **GPUDirect Storage**: Direct GPU-to-storage transfers
4. **Elastic Training**: Handle node failures gracefully
5. **Gradient Compression**: Reduce communication bandwidth
"""

SYSTEM_PROMPT = f"""You are an expert GPU performance optimization advisor specializing in:
- PyTorch, CUDA, Triton kernel optimization
- Distributed training (DDP, FSDP, TP, PP, EP)
- Inference optimization (vLLM, TensorRT-LLM, SGLang)
- RLHF and reinforcement learning optimization
- Multi-node cluster performance
- Blackwell/Hopper/Ampere architecture specifics

Your role is to analyze profiling data and provide specific, actionable recommendations.
Focus on HIGH-IMPACT optimizations with concrete implementation guidance.

{OPTIMIZATION_KNOWLEDGE}

When providing recommendations:
1. Prioritize by expected speedup impact
2. Consider the specific hardware capabilities provided
3. Suggest compound optimization stacks when applicable
4. Provide code snippets or configuration examples
5. Warn about potential pitfalls or compatibility issues
"""


def _load_env_file():
    """Load .env file from project root."""
    repo_root = find_repo_root(Path(__file__))
    loaded = load_repo_dotenv(repo_root)
    if loaded:
        logger.debug("Loaded dotenv files from %s", repo_root)
        return True
    return False

# Load .env on module import
_load_env_file()


@dataclass
class LLMConfig:
    """Configuration for LLM advisor."""
    provider: str = "auto"  # "openai", "anthropic", "ollama", "auto"
    model: str = ""  # Empty = use default for provider
    temperature: float = 0.3
    max_tokens: int = 16000  # High default for reasoning models (o1, gpt-5.1)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    def __post_init__(self):
        # Load .env if not already loaded
        _load_env_file()
        
        if self.provider == "auto":
            # Check LLM_PROVIDER env var first
            env_provider = os.environ.get("LLM_PROVIDER", "").lower()
            if env_provider in ("openai", "anthropic", "ollama"):
                self.provider = env_provider
            elif os.environ.get("ANTHROPIC_API_KEY"):
                self.provider = "anthropic"
            elif os.environ.get("OPENAI_API_KEY"):
                self.provider = "openai"
            else:
                raise ValueError(
                    "No LLM provider configured! Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env"
                )
        
        # Get API key from environment
        if not self.api_key:
            if self.provider == "anthropic":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not self.api_key:
                    raise ValueError("ANTHROPIC_API_KEY not found in environment")
            elif self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")
                if not self.api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment")
        
        # Get model from environment or use defaults
        if not self.model:
            if self.provider == "anthropic":
                self.model = os.environ.get("ANTHROPIC_MODEL") or os.environ.get("PERF_LLM_MODEL") or DEFAULT_ANTHROPIC_MODEL
            elif self.provider == "openai":
                self.model = os.environ.get("OPENAI_MODEL", "gpt-4o")
            elif self.provider == "ollama":
                self.model = os.environ.get("OLLAMA_MODEL", "llama3.1:70b")


@dataclass
class OptimizationContext:
    """Context for generating optimization recommendations."""
    # Hardware info
    gpu_name: str = ""
    gpu_memory_gb: float = 0
    compute_capability: tuple = (0, 0)
    num_gpus: int = 1
    nvlink_available: bool = False
    
    # Profiling data
    kernel_times: dict = field(default_factory=dict)
    memory_usage: dict = field(default_factory=dict)
    bottleneck_categories: list = field(default_factory=list)
    
    # Workload info
    model_type: str = ""  # "transformer", "moe", "cnn", etc.
    batch_size: int = 0
    sequence_length: int = 0
    is_training: bool = True
    is_distributed: bool = False
    
    # Current optimizations already applied
    current_optimizations: list = field(default_factory=list)
    
    def to_prompt_context(self) -> str:
        """Convert to a string for LLM prompting."""
        return f"""
## Hardware Configuration
- GPU: {self.gpu_name}
- Memory: {self.gpu_memory_gb:.1f} GB
- Compute Capability: {self.compute_capability}
- Number of GPUs: {self.num_gpus}
- NVLink: {'Available' if self.nvlink_available else 'Not available'}

## Workload
- Model Type: {self.model_type or 'Unknown'}
- Batch Size: {self.batch_size}
- Sequence Length: {self.sequence_length}
- Mode: {'Training' if self.is_training else 'Inference'}
- Distributed: {'Yes' if self.is_distributed else 'No'}

## Current Bottlenecks
{json.dumps(self.bottleneck_categories, indent=2) if self.bottleneck_categories else 'No bottleneck analysis available'}

## Kernel Profile Summary
{json.dumps(self.kernel_times, indent=2) if self.kernel_times else 'No kernel timing data available'}

## Memory Usage
{json.dumps(self.memory_usage, indent=2) if self.memory_usage else 'No memory data available'}

## Already Applied Optimizations
{', '.join(self.current_optimizations) if self.current_optimizations else 'None detected'}
"""


class LLMAdvisor:
    """LLM-powered optimization advisor."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = None
        self._initialized = False
    
    def _init_client(self):
        """Lazily initialize the LLM client."""
        if self._initialized:
            return
        
        self._initialized = True
        
        if self.config.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.config.api_key)
                logger.info(f"✅ Anthropic client initialized (model: {self.config.model})")
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        elif self.config.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
                logger.info(f"✅ OpenAI client initialized (model: {self.config.model})")
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        
        elif self.config.provider == "ollama":
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key="ollama",
                    base_url=self.config.base_url or "http://localhost:11434/v1"
                )
                logger.info(f"✅ Ollama client initialized (model: {self.config.model})")
            except ImportError:
                raise ImportError("openai package not installed for Ollama. Run: pip install openai")
    
    def _call_llm(self, user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """Make an LLM API call."""
        self._init_client()
        
        if self._client is None:
            raise RuntimeError("LLM client not initialized")
        
        try:
            if self.config.provider == "anthropic":
                response = self._client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=self.config.temperature,
                )
                return response.content[0].text
            
            else:  # OpenAI or Ollama
                # Use max_completion_tokens for newer OpenAI models (o1, gpt-4.1, gpt-5.1, etc.)
                # Fall back to max_tokens for older models and Ollama
                try:
                    response = self._client.chat.completions.create(
                        model=self.config.model,
                        max_completion_tokens=self.config.max_tokens,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.config.temperature,
                    )
                except Exception as e:
                    if "max_completion_tokens" in str(e) or "unsupported_parameter" in str(e):
                        # Fallback for older models that don't support max_completion_tokens
                        response = self._client.chat.completions.create(
                            model=self.config.model,
                            max_tokens=self.config.max_tokens,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=self.config.temperature,
                        )
                    else:
                        raise
                return response.choices[0].message.content
        
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"LLM API call failed ({self.config.provider}/{self.config.model}): {e}")
    
    def analyze_bottlenecks(
        self,
        context: OptimizationContext,
        focus_area: Optional[str] = None
    ) -> dict:
        """Analyze bottlenecks and generate recommendations."""
        
        prompt = f"""Analyze the following GPU workload and provide specific optimization recommendations.

{context.to_prompt_context()}

{f'Focus specifically on: {focus_area}' if focus_area else ''}

Provide your analysis in the following JSON format:
{{
    "summary": "Brief summary of key findings",
    "primary_bottleneck": "The main performance bottleneck",
    "recommendations": [
        {{
            "title": "Short title",
            "priority": "high|medium|low",
            "expected_speedup": "2x" or "30%",
            "description": "What to do and why",
            "implementation": "Code snippet or configuration",
            "considerations": "Any caveats or requirements"
        }}
    ],
    "compound_stack": {{
        "name": "Recommended optimization stack name",
        "techniques": ["technique1", "technique2"],
        "combined_speedup": "Expected combined improvement"
    }}
}}
"""
        
        llm_response = self._call_llm(prompt)
        
        if not llm_response:
            raise RuntimeError("LLM returned empty response")
        
        # Try to parse as JSON
        try:
            # Find JSON in response
            import re
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                result = json.loads(json_match.group())
                result["llm_powered"] = True
                result["provider"] = self.config.provider
                result["model"] = self.config.model
                return result
        except json.JSONDecodeError:
            pass
        
        # Return as freeform if JSON parsing fails
        return {
            "summary": "LLM analysis completed",
            "recommendations": [{"title": "See analysis", "description": llm_response, "priority": "high"}],
            "raw_response": llm_response,
            "llm_powered": True,
            "provider": self.config.provider,
            "model": self.config.model
        }
    
    def _rule_based_analysis(
        self,
        context: OptimizationContext,
        focus_area: Optional[str] = None
    ) -> dict:
        """Fallback rule-based analysis when LLM is unavailable."""
        
        recommendations = []
        
        # GPU-specific recommendations
        gpu_lower = context.gpu_name.lower()
        
        if "blackwell" in gpu_lower or "b200" in gpu_lower or "gb200" in gpu_lower:
            recommendations.append({
                "title": "Enable FP4 Tensor Cores",
                "priority": "high",
                "expected_speedup": "2-3x",
                "description": "Blackwell supports native FP4 computation with hardware decompression",
                "implementation": "model = model.to(torch.float4_e2m1)  # or use transformer_engine",
                "considerations": "Requires calibration for accuracy"
            })
            recommendations.append({
                "title": "Use TMA Multicast",
                "priority": "medium",
                "expected_speedup": "20-40%",
                "description": "Tensor Memory Accelerator with multicast for efficient SM cluster broadcast",
                "implementation": "See ch10/tma_multicast_cluster.cu + ch10/tma_multicast_tool.py for the CUDA multicast demo",
                "considerations": "Best for compute-bound kernels"
            })
        
        elif "h100" in gpu_lower or "hopper" in gpu_lower:
            recommendations.append({
                "title": "Enable FP8 Training/Inference",
                "priority": "high",
                "expected_speedup": "1.5-2x",
                "description": "Hopper's native FP8 tensor cores provide 2x throughput over FP16",
                "implementation": "from transformer_engine.pytorch import fp8_autocast\nwith fp8_autocast(): ...",
                "considerations": "Use E4M3 for forward, E5M2 for gradients"
            })
        
        # Bottleneck-specific recommendations
        for bottleneck in context.bottleneck_categories:
            bn_type = bottleneck.get("type", "")
            bn_pct = bottleneck.get("percentage", 0)
            
            if bn_type == "memory_transfer" and bn_pct > 15:
                recommendations.append({
                    "title": "Optimize Memory Transfers",
                    "priority": "high" if bn_pct > 25 else "medium",
                    "expected_speedup": f"{bn_pct * 0.5:.0f}%",
                    "description": f"Memory transfers consuming {bn_pct:.1f}% of runtime",
                    "implementation": "# Use async transfers with streams\nstream = torch.cuda.Stream()\nwith torch.cuda.stream(stream):\n    data = data.to('cuda', non_blocking=True)",
                    "considerations": "Ensure synchronization before using transferred data"
                })
            
            if bn_type == "cpu_overhead" and bn_pct > 10:
                recommendations.append({
                    "title": "Enable CUDA Graphs",
                    "priority": "high",
                    "expected_speedup": f"{bn_pct * 0.7:.0f}%",
                    "description": f"CPU overhead is {bn_pct:.1f}% - CUDA Graphs eliminate launch overhead",
                    "implementation": "# Capture graph\ng = torch.cuda.CUDAGraph()\nwith torch.cuda.graph(g):\n    output = model(static_input)\n# Replay\ng.replay()",
                    "considerations": "Requires static shapes; use bucketing for variable lengths"
                })
        
        # Distributed recommendations
        if context.num_gpus > 1:
            if context.num_gpus <= 8 and context.nvlink_available:
                recommendations.append({
                    "title": "Use Tensor Parallelism",
                    "priority": "high" if context.num_gpus >= 4 else "medium",
                    "expected_speedup": f"{context.num_gpus * 0.85:.1f}x",
                    "description": "With NVLink, TP is efficient for single-node multi-GPU",
                    "implementation": "# Using Megatron-style TP\nfrom megatron.core import tensor_parallel\ntensor_parallel.model_parallel_cuda_manual_seed(seed)",
                    "considerations": "Ensure model layers are properly partitioned"
                })
            
            if context.gpu_memory_gb < 40 and context.model_type == "transformer":
                recommendations.append({
                    "title": "Enable FSDP2 with FP8",
                    "priority": "high",
                    "expected_speedup": "Memory: 4x reduction",
                    "description": "FSDP2 shards parameters; FP8 halves communication",
                    "implementation": "from torch.distributed.fsdp import FullyShardedDataParallel\nmodel = FullyShardedDataParallel(model, ...)",
                    "considerations": "Use activation checkpointing for additional memory savings"
                })
        
        # Inference-specific
        if not context.is_training:
            recommendations.append({
                "title": "Use vLLM/TensorRT-LLM",
                "priority": "high",
                "expected_speedup": "2-4x",
                "description": "Production inference engines with PagedAttention and continuous batching",
                "implementation": "# vLLM\nfrom vllm import LLM, SamplingParams\nllm = LLM(model='meta-llama/Llama-2-7b-hf', tensor_parallel_size=num_gpus)",
                "considerations": "vLLM for flexibility, TRT-LLM for maximum performance"
            })
            
            if context.sequence_length > 4096:
                recommendations.append({
                    "title": "Enable KV Cache Compression",
                    "priority": "high",
                    "expected_speedup": "2-4x memory",
                    "description": f"Sequence length {context.sequence_length} benefits from KV compression",
                    "implementation": "# In vLLM config\nkv_cache_dtype='fp8'  # or use H2O/StreamingLLM for eviction",
                    "considerations": "Some accuracy trade-off with aggressive compression"
                })
        
        # Build compound stack
        compound_stack = self._build_compound_stack(context, recommendations)
        
        return {
            "summary": f"Rule-based analysis for {context.gpu_name} with {context.num_gpus} GPU(s)",
            "primary_bottleneck": context.bottleneck_categories[0].get("type") if context.bottleneck_categories else "unknown",
            "recommendations": recommendations[:5],  # Top 5
            "compound_stack": compound_stack,
            "llm_available": False
        }
    
    def _build_compound_stack(self, context: OptimizationContext, recommendations: list) -> dict:
        """Build a recommended compound optimization stack."""
        
        if not context.is_training:
            # Inference stack
            return {
                "name": "Maximum Inference Throughput",
                "techniques": [
                    "torch.compile(mode='max-autotune')",
                    "CUDA Graphs with bucketing",
                    "FP8/FP4 quantization",
                    "FlashAttention-2/3",
                    "PagedAttention",
                    "Speculative decoding"
                ],
                "combined_speedup": "4-10x over naive PyTorch"
            }
        
        elif context.num_gpus > 1:
            # Distributed training stack
            return {
                "name": "Distributed Training Optimization",
                "techniques": [
                    "FSDP2 with FP8 all-gather",
                    "Activation checkpointing",
                    "torch.compile",
                    "FlashAttention-2",
                    "Gradient accumulation",
                    "NCCL tuning (ring/tree selection)"
                ],
                "combined_speedup": "2-3x over vanilla DDP"
            }
        
        else:
            # Single GPU training stack
            return {
                "name": "Single GPU Maximum Efficiency",
                "techniques": [
                    "torch.compile(mode='reduce-overhead')",
                    "Mixed precision (BF16)",
                    "FlashAttention-2",
                    "Gradient checkpointing",
                    "Efficient data loading"
                ],
                "combined_speedup": "1.5-2x over eager mode"
            }
    
    def get_distributed_recommendations(
        self,
        num_nodes: int,
        gpus_per_node: int,
        model_params_b: float,
        interconnect: str = "infiniband"
    ) -> dict:
        """Get recommendations for distributed training setup."""
        
        total_gpus = num_nodes * gpus_per_node
        
        prompt = f"""Provide distributed training recommendations for:
- Nodes: {num_nodes}
- GPUs per node: {gpus_per_node}
- Total GPUs: {total_gpus}
- Model parameters: {model_params_b}B
- Interconnect: {interconnect}

Focus on:
1. Optimal parallelism strategy (TP, PP, DP, EP configuration)
2. NCCL tuning recommendations
3. Gradient communication optimization
4. Fault tolerance considerations
5. Expected scaling efficiency

Provide specific configuration values and environment variables.
"""
        
        llm_response = self._call_llm(prompt)
        
        if not llm_response:
            raise RuntimeError("LLM returned empty response")
        
        return {
            "analysis": llm_response,
            "llm_powered": True,
            "provider": self.config.provider,
            "model": self.config.model,
            "config": {
                "num_nodes": num_nodes,
                "gpus_per_node": gpus_per_node,
                "total_gpus": total_gpus,
                "model_params_b": model_params_b,
                "interconnect": interconnect
            }
        }
    
    def _calculate_optimal_parallelism(
        self,
        num_nodes: int,
        gpus_per_node: int,
        model_params_b: float
    ) -> dict:
        """Calculate optimal parallelism configuration."""
        
        total_gpus = num_nodes * gpus_per_node
        
        # Heuristics for parallelism
        if model_params_b < 7:
            # Small model - DDP or FSDP
            return {
                "strategy": "FSDP" if model_params_b > 3 else "DDP",
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "data_parallel": total_gpus,
                "expert_parallel": 1
            }
        
        elif model_params_b < 70:
            # Medium model - TP + FSDP
            tp = min(8, gpus_per_node)
            return {
                "strategy": "TP + FSDP",
                "tensor_parallel": tp,
                "pipeline_parallel": 1,
                "data_parallel": total_gpus // tp,
                "expert_parallel": 1
            }
        
        else:
            # Large model - Full 3D parallelism
            tp = min(8, gpus_per_node)
            pp = min(4, num_nodes)
            dp = total_gpus // (tp * pp)
            return {
                "strategy": "3D Parallelism (TP + PP + DP)",
                "tensor_parallel": tp,
                "pipeline_parallel": pp,
                "data_parallel": dp,
                "expert_parallel": 1,
                "recommendation": "Consider using Megatron-LM or DeepSpeed for this scale"
            }
    
    def get_inference_recommendations(
        self,
        model_name: str,
        target_latency_ms: Optional[float] = None,
        target_throughput: Optional[float] = None,
        max_batch_size: int = 32,
        max_sequence_length: int = 4096
    ) -> dict:
        """Get inference optimization recommendations."""
        
        prompt = f"""Provide inference optimization recommendations for:
- Model: {model_name}
- Target latency: {target_latency_ms}ms (if specified)
- Target throughput: {target_throughput} tokens/sec (if specified)
- Max batch size: {max_batch_size}
- Max sequence length: {max_sequence_length}

Compare vLLM, TensorRT-LLM, and SGLang options.
Include quantization recommendations (FP8, INT8, AWQ, GPTQ).
Provide specific configuration for each framework.
"""
        
        llm_response = self._call_llm(prompt)
        
        if not llm_response:
            raise RuntimeError("LLM returned empty response")
        
        return {
            "analysis": llm_response,
            "llm_powered": True,
            "provider": self.config.provider,
            "model": self.config.model,
            "config": {
                "model_name": model_name,
                "target_latency_ms": target_latency_ms,
                "target_throughput": target_throughput,
                "max_batch_size": max_batch_size,
                "max_sequence_length": max_sequence_length
            }
        }
    
    def get_rlhf_recommendations(
        self,
        policy_model_size_b: float,
        reward_model_size_b: float,
        num_gpus: int
    ) -> dict:
        """Get RLHF training optimization recommendations."""
        
        prompt = f"""Provide RLHF training optimization recommendations for:
- Policy model: {policy_model_size_b}B parameters
- Reward model: {reward_model_size_b}B parameters
- Available GPUs: {num_gpus}

Focus on:
1. Memory optimization (reference model sharing, KV cache reuse)
2. Efficient rollout generation
3. PPO optimization tricks
4. Gradient checkpointing strategy
5. Framework recommendations (DeepSpeed-Chat, TRL, OpenRLHF)
"""
        
        llm_response = self._call_llm(prompt)
        
        if not llm_response:
            raise RuntimeError("LLM returned empty response")
        
        return {
            "analysis": llm_response,
            "llm_powered": True,
            "provider": self.config.provider,
            "model": self.config.model,
            "config": {
                "policy_model_size_b": policy_model_size_b,
                "reward_model_size_b": reward_model_size_b,
                "num_gpus": num_gpus
            }
        }
    
    def is_llm_available(self) -> bool:
        """Check if LLM is available for analysis."""
        self._init_client()
        return self._client is not None
    
    def get_provider_info(self) -> dict:
        """Get information about the current LLM provider."""
        self._init_client()
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "available": self.is_llm_available(),
            "fallback": "rule_based" if not self.is_llm_available() else None
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_advisor: Optional[LLMAdvisor] = None


def get_advisor() -> LLMAdvisor:
    """Get or create the default LLM advisor."""
    global _default_advisor
    if _default_advisor is None:
        _default_advisor = LLMAdvisor()
    return _default_advisor


def analyze_profile(
    kernel_data: dict,
    hardware_info: dict,
    focus_area: Optional[str] = None
) -> dict:
    """Convenience function to analyze profiling data."""
    
    advisor = get_advisor()
    
    # Build context from profiling data
    context = OptimizationContext(
        gpu_name=hardware_info.get("gpu_name", "Unknown"),
        gpu_memory_gb=hardware_info.get("memory_gb", 0),
        compute_capability=tuple(hardware_info.get("compute_capability", [0, 0])),
        num_gpus=hardware_info.get("num_gpus", 1),
        nvlink_available=hardware_info.get("nvlink", False),
        kernel_times=kernel_data.get("kernel_summary", {}),
        bottleneck_categories=kernel_data.get("bottlenecks", []),
        model_type=kernel_data.get("model_type", ""),
        is_training=kernel_data.get("is_training", True)
    )
    
    return advisor.analyze_bottlenecks(context, focus_area)
