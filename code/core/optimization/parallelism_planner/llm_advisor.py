#!/usr/bin/env python3
"""
LLM-Powered Optimization Advisor

Uses LLM intelligence combined with our comprehensive system context to provide:
- Context-aware optimization recommendations
- Hardware-specific tuning suggestions
- Workload-specific optimizations
- Compound optimization strategies
- Large-scale distributed training guidance

The LLM receives rich system context (hardware, software, topology, benchmarks)
and generates tailored recommendations rather than hard-coded rules.

Anthropic defaults to Claude Sonnet 4.6. ANTHROPIC_MODEL, then PERF_LLM_MODEL,
can explicitly select another model.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from core.llm import DEFAULT_ANTHROPIC_MODEL

class OptimizationGoal(Enum):
    """Optimization goals for LLM advisor."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY = "memory"
    EFFICIENCY = "efficiency"
    COST = "cost"


@dataclass
class SystemContext:
    """Complete system context for LLM analysis."""
    # Hardware
    gpu_name: str = "Unknown"
    gpu_architecture: str = "Unknown"
    gpu_memory_gb: float = 80
    gpu_count: int = 8
    compute_capability: str = "9.0"
    tensor_core_gen: int = 5
    
    # Interconnect
    has_nvlink: bool = True
    nvlink_version: int = 5
    nvlink_bandwidth_gb_s: float = 900
    has_nvswitch: bool = False
    pcie_gen: int = 5
    pcie_bandwidth_gb_s: float = 64
    
    # Multi-node
    num_nodes: int = 1
    network_type: str = "infiniband"  # infiniband, roce, ethernet
    inter_node_bandwidth_gbps: float = 400
    has_rdma: bool = True
    
    # Software
    pytorch_version: str = "2.5"
    cuda_version: str = "12.8"
    flash_attention_version: str = "3.0"
    transformer_engine_version: str = "1.11"
    
    # Model
    model_name: str = ""
    model_params_b: float = 70
    model_architecture: str = "transformer"
    num_layers: int = 80
    hidden_size: int = 8192
    num_attention_heads: int = 64
    num_kv_heads: int = 8
    is_moe: bool = False
    num_experts: int = 1
    
    # Workload
    batch_size: int = 8
    sequence_length: int = 4096
    is_training: bool = True
    precision: str = "bf16"
    
    # Current configuration
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 8
    context_parallel: int = 1
    expert_parallel: int = 1
    sharding_strategy: str = "none"  # none, zero1, zero2, zero3, fsdp
    
    # Performance metrics (if available)
    current_mfu: Optional[float] = None
    current_throughput_tokens_s: Optional[float] = None
    current_memory_utilization: Optional[float] = None
    
    def to_prompt_context(self) -> str:
        """Convert to rich context string for LLM."""
        return f"""
## System Hardware

**GPU Configuration:**
- GPU: {self.gpu_name} ({self.gpu_architecture})
- Memory: {self.gpu_memory_gb} GB HBM per GPU
- GPUs: {self.gpu_count} total across {self.num_nodes} node(s)
- Compute Capability: {self.compute_capability}
- Tensor Core Generation: {self.tensor_core_gen}

**Interconnect:**
- Intra-node: {'NVLink v' + str(self.nvlink_version) + f' ({self.nvlink_bandwidth_gb_s} GB/s)' if self.has_nvlink else f'PCIe Gen{self.pcie_gen}'}
- NVSwitch: {'Yes' if self.has_nvswitch else 'No'}
- Inter-node: {self.network_type.upper()} ({self.inter_node_bandwidth_gbps} Gbps)
- RDMA: {'Enabled' if self.has_rdma else 'Disabled'}

## Software Stack

- PyTorch: {self.pytorch_version}
- CUDA: {self.cuda_version}
- Flash Attention: {self.flash_attention_version}
- Transformer Engine: {self.transformer_engine_version}

## Model Configuration

- Model: {self.model_name} ({self.model_params_b:.1f}B parameters)
- Architecture: {self.model_architecture}
- Layers: {self.num_layers}
- Hidden Size: {self.hidden_size}
- Attention Heads: {self.num_attention_heads} (KV Heads: {self.num_kv_heads})
- MoE: {'Yes (' + str(self.num_experts) + ' experts)' if self.is_moe else 'No'}

## Current Workload

- Mode: {'Training' if self.is_training else 'Inference'}
- Batch Size: {self.batch_size}
- Sequence Length: {self.sequence_length:,}
- Precision: {self.precision.upper()}

## Current Parallelism Configuration

- Tensor Parallel (TP): {self.tensor_parallel}
- Pipeline Parallel (PP): {self.pipeline_parallel}
- Data Parallel (DP): {self.data_parallel}
- Context Parallel (CP): {self.context_parallel}
- Expert Parallel (EP): {self.expert_parallel}
- Sharding: {self.sharding_strategy.upper()}

## Performance Metrics (if available)

- MFU: {f'{self.current_mfu:.1%}' if self.current_mfu else 'Not measured'}
- Throughput: {f'{self.current_throughput_tokens_s:,.0f} tokens/s' if self.current_throughput_tokens_s else 'Not measured'}
- Memory Utilization: {f'{self.current_memory_utilization:.1%}' if self.current_memory_utilization else 'Not measured'}
"""


@dataclass
class OptimizationRequest:
    """Request for LLM optimization advice."""
    context: SystemContext
    goal: OptimizationGoal
    constraints: List[str] = field(default_factory=list)
    specific_questions: List[str] = field(default_factory=list)
    include_code_examples: bool = True
    include_launch_commands: bool = True


@dataclass
class OptimizationAdvice:
    """Structured optimization advice from LLM."""
    summary: str
    priority_recommendations: List[Dict[str, Any]]
    parallelism_changes: Dict[str, Any]
    memory_optimizations: List[Dict[str, Any]]
    kernel_optimizations: List[Dict[str, Any]]
    communication_optimizations: List[Dict[str, Any]]
    compound_strategies: List[Dict[str, Any]]
    launch_command: Optional[str]
    environment_variables: Dict[str, str]
    expected_improvements: Dict[str, str]
    warnings: List[str]
    raw_response: str


class LLMOptimizationAdvisor:
    """
    LLM-powered optimization advisor.
    
    Uses rich system context to generate intelligent, tailored optimization
    recommendations rather than hard-coded rules.
    """
    
    SYSTEM_PROMPT = """You are an expert AI systems performance engineer specializing in:
- Large-scale distributed training (1000s of GPUs)
- CUDA kernel optimization and GPU programming
- PyTorch, DeepSpeed, Megatron-LM, FSDP
- Tensor/Pipeline/Data/Context/Expert parallelism
- Memory optimization (activation checkpointing, offloading, quantization)
- Communication optimization (NCCL, NVLink, InfiniBand)
- Inference optimization (vLLM, TensorRT-LLM, speculative decoding)

You have deep knowledge of:
- GPU microarchitecture (NVIDIA Blackwell, Hopper, Ampere)
- Hardware limits (SM count, registers, shared memory, cache)
- Roofline analysis and arithmetic intensity
- Flash Attention, CUDA Graphs, kernel fusion
- MoE routing and load balancing
- Long context methods (Ring Attention, Context Parallel)
- RLHF training (PPO, DPO, ORPO)

When providing recommendations:
1. Be specific and actionable, not generic
2. Explain WHY each optimization helps for THIS specific configuration
3. Consider hardware-software co-design
4. Suggest compound optimizations that work well together
5. Provide concrete numbers (expected speedup, memory savings)
6. Include code examples and launch commands when helpful
7. Warn about potential pitfalls or trade-offs
"""

    def __init__(self, llm_provider: str = "anthropic"):
        """Initialize with LLM provider."""
        self.llm_provider = llm_provider
        self._client = None
    
    def _get_client(self):
        """Get or create LLM client."""
        if self._client is not None:
            return self._client
        
        if self.llm_provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic()
                return self._client
            except ImportError:
                raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
        elif self.llm_provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI()
                return self._client
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
    
    def _build_optimization_prompt(self, request: OptimizationRequest) -> str:
        """Build the optimization request prompt."""
        goal_descriptions = {
            OptimizationGoal.THROUGHPUT: "Maximize training/inference throughput (tokens/second)",
            OptimizationGoal.LATENCY: "Minimize latency (time to first token, inter-token latency)",
            OptimizationGoal.MEMORY: "Minimize GPU memory usage to enable larger batches/models",
            OptimizationGoal.EFFICIENCY: "Maximize MFU (Model FLOPS Utilization)",
            OptimizationGoal.COST: "Minimize cost ($/token or $/training run)",
        }
        
        prompt = f"""
# Optimization Request

## Goal
{goal_descriptions[request.goal]}

{request.context.to_prompt_context()}

## Constraints
{chr(10).join('- ' + c for c in request.constraints) if request.constraints else 'No specific constraints'}

## Specific Questions
{chr(10).join('- ' + q for q in request.specific_questions) if request.specific_questions else 'Provide comprehensive optimization recommendations'}

## Required Output Format

Provide your recommendations in the following JSON structure:

```json
{{
    "summary": "Brief summary of key recommendations",
    "priority_recommendations": [
        {{
            "rank": 1,
            "title": "Optimization name",
            "description": "What it does and why it helps for this specific configuration",
            "expected_impact": "e.g., 1.3x throughput, 20% memory reduction",
            "effort": "low|medium|high",
            "implementation": "How to implement"
        }}
    ],
    "parallelism_changes": {{
        "tensor_parallel": 4,
        "pipeline_parallel": 2,
        "data_parallel": 1,
        "context_parallel": 1,
        "expert_parallel": 1,
        "rationale": "Why this configuration is optimal"
    }},
    "memory_optimizations": [
        {{
            "technique": "Name",
            "savings_gb": 10.5,
            "trade_off": "Any performance trade-off",
            "how_to_enable": "Code or flag to enable"
        }}
    ],
    "kernel_optimizations": [
        {{
            "kernel": "Attention/FFN/etc",
            "optimization": "What to optimize",
            "expected_speedup": "1.2x",
            "implementation": "How to implement"
        }}
    ],
    "communication_optimizations": [
        {{
            "collective": "AllReduce/AllGather/etc",
            "optimization": "What to optimize",
            "expected_speedup": "1.1x",
            "nccl_settings": {{"VAR": "VALUE"}}
        }}
    ],
    "compound_strategies": [
        {{
            "name": "Strategy name",
            "techniques": ["List of techniques that work well together"],
            "combined_impact": "Overall expected improvement",
            "implementation_order": ["Which to enable first", "Then second", "etc"]
        }}
    ],
    "launch_command": "Complete launch command with all optimizations",
    "environment_variables": {{
        "VAR_NAME": "VALUE"
    }},
    "expected_improvements": {{
        "throughput": "+30%",
        "memory": "-25%",
        "mfu": "from 35% to 45%"
    }},
    "warnings": [
        "Any potential issues or trade-offs to be aware of"
    ]
}}
```

Provide ONLY the JSON output, no additional text.
"""
        return prompt
    
    def _get_prompt_only_advice(
        self,
        request: OptimizationRequest,
        prompt: str,
        error: str,
    ) -> OptimizationAdvice:
        """Return advice with prompts when LLM is unavailable."""
        
        # Generate basic engine-based recommendations
        from .advisor import ParallelismAdvisor, create_mock_topology_h100_multigpu
        from .model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        arch = analyzer.analyze(request.context.model_name)
        
        advisor = ParallelismAdvisor(auto_detect_topology=False)
        advisor.set_topology(create_mock_topology_h100_multigpu())
        
        result = advisor.recommend(
            model=request.context.model_name,
            is_training=request.context.is_training,
            batch_size=request.context.batch_size,
            seq_length=request.context.sequence_length,
        )
        rec = result.best if result else None
        
        # Build recommendations based on engine output
        if rec:
            priority_recs = [
                {
                    "title": f"Use TP={rec.strategy.tp}, PP={rec.strategy.pp}, DP={rec.strategy.dp}",
                    "description": "Engine-calculated optimal parallelism for your hardware",
                    "impact": "high",
                    "effort": "low",
                },
                {
                    "title": "Use the LLM prompt below for deeper insights",
                    "description": "Copy the prompt from raw_response and use with Claude, GPT-4, or any capable LLM",
                    "impact": "high",
                    "effort": "low",
                },
            ]
            parallelism = {
                "tensor_parallel": {"current": 1, "recommended": rec.strategy.tp},
                "pipeline_parallel": {"current": 1, "recommended": rec.strategy.pp},
                "data_parallel": {"current": 1, "recommended": rec.strategy.dp},
                "rationale": rec.rationale[:3] if rec.rationale else [],
            }
            mem = f"{rec.analysis.memory_per_gpu_gb:.1f}GB" if rec.analysis and rec.analysis.memory_per_gpu_gb else "N/A"
        else:
            priority_recs = [
                {
                    "title": "Use the LLM prompt below for recommendations",
                    "description": "Copy the prompt from raw_response and use with Claude, GPT-4, or any capable LLM",
                    "impact": "high",
                    "effort": "low",
                },
            ]
            parallelism = {}
            mem = "N/A"
        
        return OptimizationAdvice(
            summary=f"LLM unavailable ({error}). Showing engine-based recommendations with prompts you can use with any LLM.",
            priority_recommendations=priority_recs,
            parallelism_changes=parallelism,
            memory_optimizations=[],
            kernel_optimizations=[],
            communication_optimizations=[],
            compound_strategies=[],
            launch_command=None,
            environment_variables={},
            expected_improvements={"memory_per_gpu": mem},
            warnings=[
                "LLM API not available. To get AI-powered recommendations:",
                "  1. Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable",
                "  2. Or copy the prompt below and use with your preferred LLM",
            ],
            raw_response=f"=== PROMPT FOR LLM ===\n\n{self.SYSTEM_PROMPT}\n\n---\n\n{prompt}",
        )
    
    def get_advice(self, request: OptimizationRequest) -> OptimizationAdvice:
        """Get optimization advice from LLM."""
        prompt = self._build_optimization_prompt(request)
        
        try:
            client = self._get_client()
        except Exception as e:
            # Return prompt-only response when no API key is set
            return self._get_prompt_only_advice(request, prompt, str(e))
        
        try:
            if self.llm_provider == "anthropic":
                response = client.messages.create(
                    model=os.environ.get("ANTHROPIC_MODEL") or os.environ.get("PERF_LLM_MODEL") or DEFAULT_ANTHROPIC_MODEL,
                    max_tokens=8192,
                    system=self.SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}]
                )
                raw_response = response.content[0].text
            else:  # openai
                response = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=8192,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                )
                raw_response = response.choices[0].message.content
        except Exception as e:
            # API call failed - return prompt-only advice
            return self._get_prompt_only_advice(request, prompt, str(e))
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = raw_response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str)
            
            return OptimizationAdvice(
                summary=data.get("summary", ""),
                priority_recommendations=data.get("priority_recommendations", []),
                parallelism_changes=data.get("parallelism_changes", {}),
                memory_optimizations=data.get("memory_optimizations", []),
                kernel_optimizations=data.get("kernel_optimizations", []),
                communication_optimizations=data.get("communication_optimizations", []),
                compound_strategies=data.get("compound_strategies", []),
                launch_command=data.get("launch_command"),
                environment_variables=data.get("environment_variables", {}),
                expected_improvements=data.get("expected_improvements", {}),
                warnings=data.get("warnings", []),
                raw_response=raw_response,
            )
        except json.JSONDecodeError:
            # Return raw response if parsing fails
            return OptimizationAdvice(
                summary="Failed to parse structured response",
                priority_recommendations=[],
                parallelism_changes={},
                memory_optimizations=[],
                kernel_optimizations=[],
                communication_optimizations=[],
                compound_strategies=[],
                launch_command=None,
                environment_variables={},
                expected_improvements={},
                warnings=["Response was not in expected JSON format"],
                raw_response=raw_response,
            )
    
    def get_quick_advice(
        self,
        model: str,
        goal: str = "throughput",
        num_gpus: int = 8,
        is_training: bool = True,
    ) -> str:
        """Get quick optimization advice with minimal input."""
        from .model_analyzer import ModelAnalyzer
        from .topology_detector import TopologyDetector
        
        # Analyze model
        analyzer = ModelAnalyzer()
        arch = analyzer.analyze(model)
        
        # Detect topology (or use mock)
        try:
            detector = TopologyDetector()
            topology = detector.detect()
            gpu_name = topology.gpus[0].name if topology.gpus else "H100"
            gpu_mem = topology.gpus[0].memory_gb if topology.gpus else 80
            has_nvlink = topology.has_nvlink
        except Exception:
            gpu_name = "H100 SXM"
            gpu_mem = 80
            has_nvlink = True
        
        # Build context
        context = SystemContext(
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_mem,
            gpu_count=num_gpus,
            has_nvlink=has_nvlink,
            model_name=model,
            model_params_b=arch.total_params_billion,
            num_layers=arch.num_layers,
            hidden_size=arch.hidden_size,
            is_training=is_training,
        )
        
        goal_map = {
            "throughput": OptimizationGoal.THROUGHPUT,
            "latency": OptimizationGoal.LATENCY,
            "memory": OptimizationGoal.MEMORY,
            "efficiency": OptimizationGoal.EFFICIENCY,
            "cost": OptimizationGoal.COST,
        }
        
        request = OptimizationRequest(
            context=context,
            goal=goal_map.get(goal, OptimizationGoal.THROUGHPUT),
        )
        
        advice = self.get_advice(request)
        return self._format_advice(advice)
    
    def _format_advice(self, advice: OptimizationAdvice) -> str:
        """Format advice for display."""
        lines = [
            "=" * 70,
            "LLM OPTIMIZATION ADVISOR",
            "=" * 70,
            "",
            "📋 SUMMARY",
            advice.summary,
            "",
        ]
        
        if advice.priority_recommendations:
            lines.extend([
                "🎯 PRIORITY RECOMMENDATIONS",
                "-" * 40,
            ])
            for rec in advice.priority_recommendations[:5]:
                lines.extend([
                    f"\n{rec.get('rank', '?')}. {rec.get('title', 'Unknown')}",
                    f"   Impact: {rec.get('expected_impact', 'Unknown')}",
                    f"   Effort: {rec.get('effort', 'Unknown')}",
                    f"   {rec.get('description', '')}",
                ])
        
        if advice.parallelism_changes:
            lines.extend([
                "",
                "⚡ RECOMMENDED PARALLELISM",
                "-" * 40,
                f"   TP={advice.parallelism_changes.get('tensor_parallel', '?')} "
                f"PP={advice.parallelism_changes.get('pipeline_parallel', '?')} "
                f"DP={advice.parallelism_changes.get('data_parallel', '?')} "
                f"CP={advice.parallelism_changes.get('context_parallel', '?')}",
                f"   Rationale: {advice.parallelism_changes.get('rationale', '')}",
            ])
        
        if advice.compound_strategies:
            lines.extend([
                "",
                "🔗 COMPOUND OPTIMIZATION STRATEGIES",
                "-" * 40,
            ])
            for strat in advice.compound_strategies[:3]:
                lines.extend([
                    f"\n   {strat.get('name', 'Unknown')}",
                    f"   Techniques: {', '.join(strat.get('techniques', []))}",
                    f"   Combined Impact: {strat.get('combined_impact', 'Unknown')}",
                ])
        
        if advice.expected_improvements:
            lines.extend([
                "",
                "📈 EXPECTED IMPROVEMENTS",
                "-" * 40,
            ])
            for metric, improvement in advice.expected_improvements.items():
                lines.append(f"   {metric}: {improvement}")
        
        if advice.launch_command:
            lines.extend([
                "",
                "🚀 LAUNCH COMMAND",
                "-" * 40,
                advice.launch_command,
            ])
        
        if advice.warnings:
            lines.extend([
                "",
                "⚠️  WARNINGS",
                "-" * 40,
            ])
            for warn in advice.warnings:
                lines.append(f"   • {warn}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


def collect_system_context() -> SystemContext:
    """Collect complete system context for LLM analysis."""
    context = SystemContext()
    
    # Try to detect GPU info
    try:
        import torch
        if torch.cuda.is_available():
            context.gpu_count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            context.gpu_name = props.name
            context.gpu_memory_gb = props.total_memory / (1024**3)
            context.compute_capability = f"{props.major}.{props.minor}"
            
            # Detect architecture from compute capability
            if props.major >= 10:
                context.gpu_architecture = "Blackwell"
                context.tensor_core_gen = 6
            elif props.major == 9:
                context.gpu_architecture = "Hopper"
                context.tensor_core_gen = 5
            elif props.major == 8:
                context.gpu_architecture = "Ampere"
                context.tensor_core_gen = 3
    except Exception:
        pass
    
    # Try to detect software versions
    try:
        import torch
        context.pytorch_version = torch.__version__
        context.cuda_version = torch.version.cuda or "Unknown"
    except Exception:
        pass
    
    try:
        import flash_attn
        context.flash_attention_version = flash_attn.__version__
    except Exception:
        pass
    
    try:
        import transformer_engine
        context.transformer_engine_version = transformer_engine.__version__
    except Exception:
        pass
    
    return context


# CLI integration
def cmd_llm_advisor(args):
    """Handle llm-advisor subcommand."""
    advisor = LLMOptimizationAdvisor(llm_provider=args.provider)
    
    result = advisor.get_quick_advice(
        model=args.model,
        goal=args.goal,
        num_gpus=args.gpus,
        is_training=not args.inference,
    )
    
    if args.json:
        # Re-get structured advice
        from .model_analyzer import ModelAnalyzer
        analyzer = ModelAnalyzer()
        arch = analyzer.analyze(args.model)
        
        context = SystemContext(
            model_name=args.model,
            model_params_b=arch.total_params_billion,
            gpu_count=args.gpus,
            is_training=not args.inference,
        )
        
        goal_map = {
            "throughput": OptimizationGoal.THROUGHPUT,
            "latency": OptimizationGoal.LATENCY,
            "memory": OptimizationGoal.MEMORY,
            "efficiency": OptimizationGoal.EFFICIENCY,
            "cost": OptimizationGoal.COST,
        }
        
        request = OptimizationRequest(
            context=context,
            goal=goal_map.get(args.goal, OptimizationGoal.THROUGHPUT),
        )
        
        advice = advisor.get_advice(request)
        print(json.dumps({
            "summary": advice.summary,
            "recommendations": advice.priority_recommendations,
            "parallelism": advice.parallelism_changes,
            "compound_strategies": advice.compound_strategies,
            "launch_command": advice.launch_command,
            "expected_improvements": advice.expected_improvements,
        }, indent=2))
    else:
        print(result)


__all__ = [
    "LLMOptimizationAdvisor",
    "SystemContext",
    "OptimizationRequest",
    "OptimizationAdvice",
    "OptimizationGoal",
    "collect_system_context",
    "cmd_llm_advisor",
]
