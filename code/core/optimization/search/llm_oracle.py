#!/usr/bin/env python3
"""
🔮 LLM-Guided Optimization Oracle

Provides dynamic, context-aware optimization suggestions using LLM reasoning.
NOT hard-coded - the LLM receives real system metrics and profiling data
and generates specific recommendations.

Features:
- Learns from historical optimization results
- Incorporates real-time profiling data
- Suggests novel compound optimizations
- Explains trade-offs and risks
- Provides implementation guidance

The Oracle:
1. Collects system context (hardware, software, profiling)
2. Queries knowledge base for similar past optimizations
3. Constructs rich prompt with all context
4. Gets LLM recommendations
5. Validates and structures the output
6. Learns from feedback

Usage:
    from core.optimization.search import LLMOracle
    
    oracle = LLMOracle()
    
    # Get recommendations
    suggestions = oracle.suggest_optimizations(
        profile_data={"bottlenecks": ["memory_bound", ...]},
        constraints={"max_memory_gb": 70, "min_speedup": 1.5}
    )
    
    # Ask specific questions
    answer = oracle.ask(
        "Why is my attention kernel only achieving 40% MFU?",
        context={"kernel_metrics": {...}}
    )
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OptimizationSuggestion:
    """A single optimization suggestion from the oracle."""
    title: str
    description: str
    expected_speedup: str  # e.g., "1.3x-1.5x"
    expected_memory_impact: str  # e.g., "-20GB" or "+5GB"
    difficulty: str  # "easy", "medium", "hard"
    category: str  # "parallelism", "precision", "kernels", etc.
    implementation_steps: List[str]
    code_snippet: Optional[str] = None
    risks: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    confidence: float = 0.8  # Oracle's confidence in this suggestion
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "expected_speedup": self.expected_speedup,
            "expected_memory_impact": self.expected_memory_impact,
            "difficulty": self.difficulty,
            "category": self.category,
            "implementation_steps": self.implementation_steps,
            "code_snippet": self.code_snippet,
            "risks": self.risks,
            "prerequisites": self.prerequisites,
            "confidence": self.confidence,
        }


@dataclass
class OracleQuery:
    """Record of a query to the oracle."""
    query_id: str
    query_type: str  # "suggest", "ask", "validate"
    context: Dict[str, Any]
    response: Any
    timestamp: float
    latency_ms: float
    feedback_score: Optional[float] = None  # 0-1 score after user feedback


# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

class OracleKnowledgeBase:
    """
    Persistent knowledge base for the oracle.
    Stores past queries, responses, feedback, and learned patterns.
    """
    
    def __init__(self, path: Optional[Path] = None):
        self.path = path or Path.home() / ".cache" / "perf_oracle" / "knowledge.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()
    
    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                with open(self.path) as f:
                    return json.load(f)
            except:
                pass
        return {
            "queries": [],
            "successful_patterns": {},  # hash -> list of successful optimizations
            "failed_patterns": {},  # hash -> list of failed optimizations
            "optimization_scores": {},  # optimization_name -> avg success score
            "hardware_profiles": {},  # hardware_hash -> known good configs
        }
    
    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        except:
            pass
    
    def record_query(self, query: OracleQuery):
        """Record a query for learning."""
        self.data["queries"].append({
            "id": query.query_id,
            "type": query.query_type,
            "context_hash": self._hash_context(query.context),
            "timestamp": query.timestamp,
            "latency_ms": query.latency_ms,
        })
        # Keep only last 1000 queries
        self.data["queries"] = self.data["queries"][-1000:]
    
    def record_feedback(self, query_id: str, score: float, applied_optimizations: List[str]):
        """Record feedback on a query result."""
        for opt in applied_optimizations:
            if opt not in self.data["optimization_scores"]:
                self.data["optimization_scores"][opt] = {"total": 0, "count": 0}
            self.data["optimization_scores"][opt]["total"] += score
            self.data["optimization_scores"][opt]["count"] += 1
        self.save()
    
    def get_similar_contexts(self, context: Dict[str, Any], limit: int = 5) -> List[Dict]:
        """Find similar past contexts."""
        context_hash = self._hash_context(context)
        return self.data.get("successful_patterns", {}).get(context_hash[:8], [])[:limit]
    
    def get_optimization_score(self, optimization_name: str) -> float:
        """Get average success score for an optimization."""
        data = self.data["optimization_scores"].get(optimization_name, {})
        if data.get("count", 0) > 0:
            return data["total"] / data["count"]
        return 0.5  # Default neutral score
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Hash context for similarity lookup."""
        # Extract key features for hashing
        features = {
            "gpu_arch": context.get("hardware", {}).get("gpu_arch", ""),
            "model_size_bucket": self._bucket_model_size(
                context.get("model", {}).get("parameters_billions", 0)
            ),
            "bottleneck_type": context.get("profile", {}).get("primary_bottleneck", ""),
        }
        return hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()
    
    def _bucket_model_size(self, params_b: float) -> str:
        if params_b < 3:
            return "small"
        elif params_b < 13:
            return "medium"
        elif params_b < 70:
            return "large"
        else:
            return "xlarge"


# =============================================================================
# CONTEXT COLLECTORS
# =============================================================================

class ContextCollector:
    """Collects real-time context for the oracle."""
    
    @staticmethod
    def get_hardware_context() -> Dict[str, Any]:
        """Get current hardware context."""
        import subprocess
        
        context = {
            "gpu_count": 0,
            "gpu_arch": "unknown",
            "gpu_memory_gb": 0,
            "gpu_name": "unknown",
            "has_nvlink": False,
            "cuda_version": "unknown",
            "driver_version": "unknown",
        }
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                context["gpu_count"] = len(lines)
                if lines:
                    parts = lines[0].split(',')
                    context["gpu_name"] = parts[0].strip()
                    context["gpu_memory_gb"] = int(parts[1].strip()) // 1024
                    context["driver_version"] = parts[2].strip()
                    
                    # Determine architecture
                    name = context["gpu_name"].lower()
                    if "b100" in name or "b200" in name or "gb200" in name:
                        context["gpu_arch"] = "blackwell"
                    elif "h100" in name or "h200" in name:
                        context["gpu_arch"] = "hopper"
                    elif "a100" in name:
                        context["gpu_arch"] = "ampere"
                    elif "4090" in name or "4080" in name:
                        context["gpu_arch"] = "ada"
                    
                    # Check NVLink
                    context["has_nvlink"] = context["gpu_count"] > 1 and (
                        "a100" in name or "h100" in name or "b100" in name or
                        "h200" in name or "b200" in name or "gb200" in name
                    )
        except:
            pass
        
        try:
            import torch
            if torch.cuda.is_available():
                context["cuda_version"] = torch.version.cuda or "unknown"
        except:
            pass
        
        return context
    
    @staticmethod
    def get_software_context() -> Dict[str, Any]:
        """Get software environment context."""
        context = {
            "torch_version": "unknown",
            "cuda_version": "unknown",
            "has_flash_attn": False,
            "has_triton": False,
            "has_transformer_engine": False,
            "has_bitsandbytes": False,
        }
        
        try:
            import torch
            context["torch_version"] = torch.__version__
            context["cuda_version"] = torch.version.cuda or "unknown"
        except:
            pass
        
        for pkg, key in [
            ("flash_attn", "has_flash_attn"),
            ("triton", "has_triton"),
            ("transformer_engine", "has_transformer_engine"),
            ("bitsandbytes", "has_bitsandbytes"),
        ]:
            try:
                __import__(pkg)
                context[key] = True
            except:
                pass
        
        return context
    
    @staticmethod
    def analyze_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze profile data to extract key insights."""
        analysis = {
            "primary_bottleneck": "unknown",
            "secondary_bottlenecks": [],
            "memory_pressure": "low",
            "compute_utilization": "unknown",
            "recommendations_hint": [],
        }
        
        if not profile_data:
            return analysis
        
        # Analyze kernel times
        kernel_times = profile_data.get("kernel_times", {})
        if kernel_times:
            # Find slowest kernels
            sorted_kernels = sorted(kernel_times.items(), key=lambda x: x[1], reverse=True)
            if sorted_kernels:
                top_kernel = sorted_kernels[0][0].lower()
                if "attention" in top_kernel or "sdpa" in top_kernel:
                    analysis["primary_bottleneck"] = "attention"
                    analysis["recommendations_hint"].append("flash_attention")
                elif "gemm" in top_kernel or "matmul" in top_kernel:
                    analysis["primary_bottleneck"] = "gemm"
                    analysis["recommendations_hint"].append("tensor_cores")
                elif "allreduce" in top_kernel or "nccl" in top_kernel:
                    analysis["primary_bottleneck"] = "communication"
                    analysis["recommendations_hint"].append("overlap_communication")
        
        # Analyze memory
        memory_peak = profile_data.get("memory_peak_mb", 0)
        memory_total = profile_data.get("memory_total_mb", 80000)
        if memory_peak > 0 and memory_total > 0:
            utilization = memory_peak / memory_total
            if utilization > 0.9:
                analysis["memory_pressure"] = "critical"
                analysis["recommendations_hint"].append("checkpointing")
                analysis["recommendations_hint"].append("memory_efficient_optimizer")
            elif utilization > 0.7:
                analysis["memory_pressure"] = "high"
            elif utilization > 0.4:
                analysis["memory_pressure"] = "medium"
        
        # Analyze compute utilization
        if "gpu_utilization" in profile_data:
            util = profile_data["gpu_utilization"]
            if util > 90:
                analysis["compute_utilization"] = "high"
            elif util > 60:
                analysis["compute_utilization"] = "medium"
            else:
                analysis["compute_utilization"] = "low"
                analysis["recommendations_hint"].append("batch_size_increase")
        
        return analysis


# =============================================================================
# LLM ORACLE
# =============================================================================

class LLMOracle:
    """
    LLM-powered optimization oracle that provides dynamic,
    context-aware optimization suggestions.
    """
    
    def __init__(
        self,
        llm_provider: str = "auto",
        model: Optional[str] = None,
        knowledge_base_path: Optional[Path] = None,
    ):
        self.llm_provider = llm_provider
        self.model = model
        self.knowledge_base = OracleKnowledgeBase(knowledge_base_path)
        self.collector = ContextCollector()
        
        # Initialize LLM backend
        self._llm_backend = None
    
    def _get_llm_backend(self):
        """Lazy initialization of LLM backend."""
        if self._llm_backend is None:
            try:
                from core.llm import llm_call, is_available, PERF_EXPERT_SYSTEM
                if is_available():
                    self._llm_backend = lambda prompt: llm_call(prompt, system=PERF_EXPERT_SYSTEM)
            except Exception as e:
                print(f"Warning: Could not initialize LLM backend: {e}")
                self._llm_backend = None
        return self._llm_backend
    
    def suggest_optimizations(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        profile_data: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        current_config: Optional[Dict[str, Any]] = None,
        num_suggestions: int = 5,
    ) -> List[OptimizationSuggestion]:
        """
        Get optimization suggestions based on context.
        
        Args:
            model_config: Model configuration (size, architecture, etc.)
            profile_data: Profiling data (kernel times, memory, etc.)
            constraints: Constraints (max memory, min speedup, etc.)
            current_config: Current optimization configuration
            num_suggestions: Number of suggestions to return
        
        Returns:
            List of optimization suggestions
        """
        start_time = time.time()
        
        # Collect context
        hardware_context = self.collector.get_hardware_context()
        software_context = self.collector.get_software_context()
        profile_analysis = self.collector.analyze_profile(profile_data or {})
        
        # Check knowledge base for similar contexts
        context = {
            "hardware": hardware_context,
            "software": software_context,
            "model": model_config or {},
            "profile": profile_analysis,
            "constraints": constraints or {},
            "current_config": current_config or {},
        }
        similar_contexts = self.knowledge_base.get_similar_contexts(context)
        
        # Build prompt
        prompt = self._build_suggestion_prompt(
            hardware_context,
            software_context,
            model_config or {},
            profile_analysis,
            constraints or {},
            current_config or {},
            similar_contexts,
            num_suggestions,
        )
        
        # Query LLM
        suggestions = []
        llm = self._get_llm_backend()
        
        if llm:
            try:
                if callable(llm):
                    response = llm(prompt)
                else:
                    response = llm.backend.generate(prompt, self._get_system_prompt())
                suggestions = self._parse_suggestions(response)
            except Exception as e:
                print(f"LLM query failed: {e}")
        
        # If LLM failed or not available, use heuristics
        if not suggestions:
            suggestions = self._heuristic_suggestions(context, num_suggestions)
        
        # Record query
        query = OracleQuery(
            query_id=hashlib.md5(str(time.time()).encode()).hexdigest()[:12],
            query_type="suggest",
            context=context,
            response=[s.to_dict() for s in suggestions],
            timestamp=time.time(),
            latency_ms=(time.time() - start_time) * 1000,
        )
        self.knowledge_base.record_query(query)
        self.knowledge_base.save()
        
        return suggestions
    
    def ask(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Ask a specific question about optimization.
        
        Args:
            question: The question to ask
            context: Additional context (profile data, config, etc.)
        
        Returns:
            Answer from the oracle
        """
        # Collect context
        hardware_context = self.collector.get_hardware_context()
        
        prompt = f"""## Question
{question}

## Hardware Context
{json.dumps(hardware_context, indent=2)}

## Additional Context
{json.dumps(context or {}, indent=2)}

Provide a specific, actionable answer based on the context provided.
Include code examples where helpful.
Explain trade-offs and potential risks."""

        llm = self._get_llm_backend()
        if llm:
            try:
                if callable(llm):
                    return llm(prompt)
                return llm.backend.generate(prompt, self._get_system_prompt())
            except Exception as e:
                return f"Error querying LLM: {e}"
        
        return "LLM backend not available. Please configure OPENAI_API_KEY or start Ollama."
    
    def validate_config(
        self,
        config: Dict[str, Any],
        model_config: Dict[str, Any],
        hardware_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate an optimization configuration.
        
        Returns:
            Validation result with issues and recommendations
        """
        hardware = hardware_config or self.collector.get_hardware_context()
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check parallelism validity
        tp = config.get("tensor_parallel", 1)
        pp = config.get("pipeline_parallel", 1)
        dp = config.get("data_parallel", 1)
        total = tp * pp * dp
        
        if total > hardware.get("gpu_count", 8):
            issues.append(f"Total parallelism ({total}) exceeds available GPUs ({hardware.get('gpu_count', 8)})")
        
        # Check memory fit
        model_params_b = model_config.get("parameters_billions", 7)
        gpu_memory_gb = hardware.get("gpu_memory_gb", 80)
        
        estimated_memory = model_params_b * 2 / (tp * pp)  # BF16
        if config.get("precision") == "fp32":
            estimated_memory *= 2
        if config.get("gradient_checkpointing"):
            estimated_memory *= 0.5
        
        if estimated_memory > gpu_memory_gb * 0.9:
            issues.append(f"Estimated memory ({estimated_memory:.1f}GB) exceeds GPU capacity")
            recommendations.append("Enable gradient checkpointing or increase parallelism")
        
        # Check FP8 availability
        if config.get("precision") == "fp8":
            if hardware.get("gpu_arch") not in ["hopper", "blackwell"]:
                issues.append("FP8 requires Hopper or Blackwell GPU")
        
        # Check PP schedule
        if pp > 1 and config.get("pipeline_schedule") == "zero_bubble":
            if pp < 4:
                warnings.append("Zero-bubble scheduling works best with PP >= 4")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
        }
    
    def _build_suggestion_prompt(
        self,
        hardware: Dict,
        software: Dict,
        model: Dict,
        profile: Dict,
        constraints: Dict,
        current_config: Dict,
        similar_contexts: List,
        num_suggestions: int,
    ) -> str:
        """Build the prompt for optimization suggestions."""
        
        similar_section = ""
        if similar_contexts:
            similar_section = f"""
## Similar Past Optimizations (for reference)
{json.dumps(similar_contexts[:3], indent=2)}
"""
        
        return f"""Analyze this system and provide {num_suggestions} optimization suggestions.

## Hardware
{json.dumps(hardware, indent=2)}

## Software Environment
{json.dumps(software, indent=2)}

## Model Configuration
{json.dumps(model, indent=2)}

## Profile Analysis
{json.dumps(profile, indent=2)}

## Constraints
{json.dumps(constraints, indent=2)}

## Current Configuration
{json.dumps(current_config, indent=2)}
{similar_section}

## Task
Provide {num_suggestions} specific, actionable optimization suggestions.
For each suggestion, provide:
1. Title (concise name)
2. Description (what it does and why it helps)
3. Expected speedup range (e.g., "1.3x-1.5x")
4. Memory impact (e.g., "-20GB" or "+5GB")
5. Difficulty (easy/medium/hard)
6. Category (parallelism/precision/kernels/memory/communication)
7. Implementation steps (numbered list)
8. Code snippet (if applicable)
9. Risks (what could go wrong)
10. Prerequisites (what's needed first)

Output as JSON array:
```json
[
  {{
    "title": "...",
    "description": "...",
    "expected_speedup": "1.Xx-1.Yx",
    "expected_memory_impact": "...",
    "difficulty": "easy|medium|hard",
    "category": "...",
    "implementation_steps": ["1. ...", "2. ..."],
    "code_snippet": "...",
    "risks": ["..."],
    "prerequisites": ["..."]
  }}
]
```

Focus on the PRIMARY BOTTLENECK: {profile.get('primary_bottleneck', 'unknown')}
Memory pressure is: {profile.get('memory_pressure', 'unknown')}
"""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the oracle."""
        return """You are an expert AI systems performance engineer. You analyze real hardware 
configurations, profiling data, and constraints to provide specific, actionable optimization 
recommendations.

Key expertise areas:
- GPU optimization (CUDA, cuDNN, CUTLASS, Triton, Tensor Cores)
- Distributed training (FSDP, DeepSpeed, Megatron-LM, 3D parallelism)
- Inference optimization (vLLM, TensorRT-LLM, speculative decoding)
- Quantization (FP8, INT8, FP4, mixed precision)
- Memory optimization (checkpointing, offloading, efficient optimizers)
- Attention mechanisms (Flash Attention, Ring Attention)
- RLHF and reinforcement learning optimization

Guidelines:
1. Base recommendations on the ACTUAL data provided - don't make assumptions
2. Prioritize by impact * feasibility
3. Consider compound effects (optimizations that work well together)
4. Warn about incompatibilities and risks
5. Provide specific code examples when helpful
6. Consider the hardware architecture capabilities"""
    
    def _parse_suggestions(self, response: str) -> List[OptimizationSuggestion]:
        """Parse LLM response into structured suggestions."""
        import re
        
        suggestions = []
        
        # Try to extract JSON
        json_match = re.search(r'\[[\s\S]*?\]', response)
        if json_match:
            try:
                items = json.loads(json_match.group())
                for item in items:
                    suggestions.append(OptimizationSuggestion(
                        title=item.get("title", "Unknown"),
                        description=item.get("description", ""),
                        expected_speedup=item.get("expected_speedup", "unknown"),
                        expected_memory_impact=item.get("expected_memory_impact", "unknown"),
                        difficulty=item.get("difficulty", "medium"),
                        category=item.get("category", "other"),
                        implementation_steps=item.get("implementation_steps", []),
                        code_snippet=item.get("code_snippet"),
                        risks=item.get("risks", []),
                        prerequisites=item.get("prerequisites", []),
                    ))
            except json.JSONDecodeError:
                pass
        
        return suggestions
    
    def _heuristic_suggestions(
        self,
        context: Dict[str, Any],
        num_suggestions: int
    ) -> List[OptimizationSuggestion]:
        """Generate suggestions using heuristics when LLM is unavailable."""
        suggestions = []
        
        hardware = context.get("hardware", {})
        profile = context.get("profile", {})
        current = context.get("current_config", {})
        
        # Flash Attention
        if (hardware.get("gpu_arch") in ["ampere", "hopper", "blackwell"] and
            not current.get("flash_attention") and
            profile.get("primary_bottleneck") == "attention"):
            suggestions.append(OptimizationSuggestion(
                title="Enable Flash Attention",
                description="Memory-efficient attention that's 2-4x faster",
                expected_speedup="1.5x-2.5x",
                expected_memory_impact="-30%",
                difficulty="easy",
                category="kernels",
                implementation_steps=[
                    "Build the pinned FlashAttention-2 wheel in a separate training runtime. See docs/environment.md.",
                    "Load the model with attn_implementation='flash_attention_2'",
                    "Validate complete outputs and end-to-end serving latency",
                ],
                code_snippet="model = AutoModelForCausalLM.from_pretrained(name, attn_implementation='flash_attention_2')",
            ))
        
        # FP8 for Hopper/Blackwell
        if (hardware.get("gpu_arch") in ["hopper", "blackwell"] and
            current.get("precision") != "fp8"):
            suggestions.append(OptimizationSuggestion(
                title="Enable FP8 Training",
                description="Use FP8 precision with Transformer Engine for 2x throughput",
                expected_speedup="1.5x-2.0x",
                expected_memory_impact="-50%",
                difficulty="medium",
                category="precision",
                implementation_steps=[
                    "Build the Transformer Engine 2.20.2 binding against the locked runtime. See docs/environment.md.",
                    "Enable FP8 training in config",
                    "Use TE layers for linear/attention",
                ],
                risks=["May need loss scaling tuning"],
            ))
        
        # Gradient checkpointing
        if (profile.get("memory_pressure") in ["high", "critical"] and
            not current.get("gradient_checkpointing")):
            suggestions.append(OptimizationSuggestion(
                title="Enable Gradient Checkpointing",
                description="Trade compute for memory - recompute activations during backward",
                expected_speedup="0.7x (slower)",
                expected_memory_impact="-60%",
                difficulty="easy",
                category="memory",
                implementation_steps=[
                    "model.gradient_checkpointing_enable()",
                    "Or set gradient_checkpointing: true in config",
                ],
            ))
        
        # torch.compile
        if not current.get("torch_compile"):
            suggestions.append(OptimizationSuggestion(
                title="Enable torch.compile",
                description="JIT compile model for automatic kernel fusion",
                expected_speedup="1.15x-1.4x",
                expected_memory_impact="0",
                difficulty="easy",
                category="kernels",
                implementation_steps=[
                    "model = torch.compile(model, mode='reduce-overhead')",
                ],
                code_snippet="model = torch.compile(model, mode='reduce-overhead', fullgraph=True)",
            ))
        
        return suggestions[:num_suggestions]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_suggestions(
    model_config: Optional[Dict] = None,
    profile_data: Optional[Dict] = None,
    constraints: Optional[Dict] = None,
    num_suggestions: int = 5,
) -> List[Dict[str, Any]]:
    """Get optimization suggestions."""
    oracle = LLMOracle()
    suggestions = oracle.suggest_optimizations(
        model_config=model_config,
        profile_data=profile_data,
        constraints=constraints,
        num_suggestions=num_suggestions,
    )
    return [s.to_dict() for s in suggestions]


def ask_oracle(question: str, context: Optional[Dict] = None) -> str:
    """Ask the oracle a question."""
    oracle = LLMOracle()
    return oracle.ask(question, context)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Optimization Oracle")
    subparsers = parser.add_subparsers(dest="command")
    
    # Suggest command
    suggest_parser = subparsers.add_parser("suggest", help="Get optimization suggestions")
    suggest_parser.add_argument("--model-size", type=float, help="Model size in billions")
    suggest_parser.add_argument("--num", type=int, default=5, help="Number of suggestions")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", nargs="+", help="Your question")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("config", help="Path to config JSON")
    
    args = parser.parse_args()
    
    oracle = LLMOracle()
    
    if args.command == "suggest":
        model_config = {"parameters_billions": args.model_size} if args.model_size else None
        suggestions = oracle.suggest_optimizations(
            model_config=model_config,
            num_suggestions=args.num,
        )
        
        print("\n🔮 Optimization Suggestions")
        print("=" * 60)
        for i, s in enumerate(suggestions, 1):
            print(f"\n{i}. {s.title}")
            print(f"   {s.description}")
            print(f"   Expected: {s.expected_speedup} speedup, {s.expected_memory_impact} memory")
            print(f"   Difficulty: {s.difficulty} | Category: {s.category}")
            if s.code_snippet:
                print(f"   Code: {s.code_snippet[:100]}...")
    
    elif args.command == "ask":
        question = " ".join(args.question)
        print(f"\n🔮 Question: {question}\n")
        answer = oracle.ask(question)
        print(answer)
    
    elif args.command == "validate":
        with open(args.config) as f:
            config = json.load(f)
        result = oracle.validate_config(
            config=config.get("optimization", {}),
            model_config=config.get("model", {}),
        )
        print("\n🔮 Configuration Validation")
        print("=" * 60)
        print(f"Valid: {'✅' if result['valid'] else '❌'}")
        if result["issues"]:
            print("\nIssues:")
            for issue in result["issues"]:
                print(f"  ❌ {issue}")
        if result["warnings"]:
            print("\nWarnings:")
            for warning in result["warnings"]:
                print(f"  ⚠️ {warning}")
        if result["recommendations"]:
            print("\nRecommendations:")
            for rec in result["recommendations"]:
                print(f"  💡 {rec}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
