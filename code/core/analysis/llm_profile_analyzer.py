#!/usr/bin/env python3
"""
LLM-Enhanced Profile Analyzer - Uses Claude or GPT to provide deep performance insights.

This module takes structured profiling data and uses LLMs to:
1. Explain WHY the optimized version is faster (root cause analysis)
2. Suggest specific code changes for further improvement
3. Identify missed optimization opportunities
4. Provide architecture-specific guidance (Blackwell, Hopper, etc.)

Environment variables (via .env or .env.local):
    ANTHROPIC_API_KEY: API key for Claude
    ANTHROPIC_MODEL: Model name (default: claude-sonnet-4-6)
    PERF_LLM_MODEL: Shared model override when ANTHROPIC_MODEL is unset
    OPENAI_API_KEY: API key for OpenAI
    OPENAI_MODEL: Model name (default: gpt-4o)
    LLM_PROVIDER: Default provider (anthropic, openai)
    LLM_ANALYSIS_ENABLED: Enable/disable LLM analysis (default: true)

Usage:
    from core.analysis.llm_profile_analyzer import LLMProfileAnalyzer
    
    analyzer = LLMProfileAnalyzer(provider="anthropic")
    analysis = analyzer.analyze_differential(
        differential_report,
        baseline_code="...",
        optimized_code="...",
    )
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.llm import DEFAULT_ANTHROPIC_MODEL
from core.utils.dotenv import load_repo_dotenv

load_repo_dotenv(Path(__file__).resolve().parents[2])

from core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnvironmentContext:
    """Complete environment context for LLM analysis."""
    # System
    python_version: str = ""
    os_info: str = ""
    
    # CUDA
    cuda_version: str = ""
    cuda_driver_version: str = ""
    cudnn_version: str = ""
    nccl_version: str = ""
    cublas_version: str = ""
    cusparse_version: str = ""
    cufft_version: str = ""
    
    # PyTorch
    pytorch_version: str = ""
    torchvision_version: str = ""
    torchaudio_version: str = ""
    torch_cuda_version: str = ""
    
    # GPU - Static info
    gpu_name: str = ""
    gpu_arch: str = ""
    compute_capability: str = ""
    gpu_memory_gb: float = 0.0
    num_gpus: int = 1
    
    # GPU - Live metrics
    gpu_temperature_c: Optional[float] = None
    gpu_power_draw_w: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    memory_utilization_pct: Optional[float] = None
    gpu_clock_mhz: Optional[int] = None
    memory_clock_mhz: Optional[int] = None
    memory_used_gb: Optional[float] = None
    memory_free_gb: Optional[float] = None
    
    # NVLink
    nvlink_connected: bool = False
    nvlink_bandwidth_gbps: Optional[float] = None
    
    # Hardware specs (for roofline context)
    peak_fp32_tflops: Optional[float] = None
    peak_fp16_tflops: Optional[float] = None
    peak_bf16_tflops: Optional[float] = None
    peak_fp8_tflops: Optional[float] = None
    hbm_bandwidth_tbs: Optional[float] = None
    l2_cache_mb: Optional[float] = None
    num_sms: Optional[int] = None
    
    # Libraries
    triton_version: str = ""
    transformer_engine_version: str = ""
    flash_attn_version: str = ""
    xformers_version: str = ""
    apex_version: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v}
    
    def to_markdown(self) -> str:
        lines = ["## Environment Context", ""]
        lines.append("### System")
        lines.append(f"- Python: {self.python_version}")
        lines.append(f"- OS: {self.os_info}")
        lines.append("")
        
        lines.append("### CUDA Stack")
        lines.append(f"- CUDA Version: {self.cuda_version}")
        lines.append(f"- CUDA Driver: {self.cuda_driver_version}")
        if self.cudnn_version:
            lines.append(f"- cuDNN: {self.cudnn_version}")
        if self.nccl_version:
            lines.append(f"- NCCL: {self.nccl_version}")
        lines.append("")
        
        lines.append("### PyTorch")
        lines.append(f"- PyTorch: {self.pytorch_version}")
        lines.append(f"- Torch CUDA: {self.torch_cuda_version}")
        lines.append("")
        
        lines.append("### GPU")
        lines.append(f"- Model: {self.gpu_name}")
        lines.append(f"- Architecture: {self.gpu_arch}")
        lines.append(f"- Compute Capability: {self.compute_capability}")
        lines.append(f"- Memory: {self.gpu_memory_gb:.1f} GB")
        if self.num_gpus > 1:
            lines.append(f"- GPU Count: {self.num_gpus}")
        lines.append("")
        
        lines.append("### GPU Live Status")
        if self.gpu_temperature_c:
            lines.append(f"- Temperature: {self.gpu_temperature_c}°C")
        if self.gpu_power_draw_w:
            lines.append(f"- Power Draw: {self.gpu_power_draw_w}W")
        if self.gpu_utilization_pct is not None:
            lines.append(f"- GPU Utilization: {self.gpu_utilization_pct}%")
        if self.memory_utilization_pct is not None:
            lines.append(f"- Memory Utilization: {self.memory_utilization_pct}%")
        if self.gpu_clock_mhz:
            lines.append(f"- GPU Clock: {self.gpu_clock_mhz} MHz")
        if self.memory_clock_mhz:
            lines.append(f"- Memory Clock: {self.memory_clock_mhz} MHz")
        if self.memory_used_gb is not None:
            lines.append(f"- Memory Used: {self.memory_used_gb:.1f} GB")
        if self.memory_free_gb is not None:
            lines.append(f"- Memory Free: {self.memory_free_gb:.1f} GB")
        if self.nvlink_connected:
            lines.append(f"- NVLink: Connected")
            if self.nvlink_bandwidth_gbps:
                lines.append(f"- NVLink Bandwidth: {self.nvlink_bandwidth_gbps} GB/s")
        lines.append("")
        
        if self.peak_bf16_tflops:
            lines.append("### Hardware Limits (Theoretical Peak)")
            if self.peak_fp32_tflops:
                lines.append(f"- FP32: {self.peak_fp32_tflops} TFLOPS")
            if self.peak_bf16_tflops:
                lines.append(f"- BF16: {self.peak_bf16_tflops} TFLOPS")
            if self.peak_fp8_tflops:
                lines.append(f"- FP8: {self.peak_fp8_tflops} TFLOPS")
            if self.hbm_bandwidth_tbs:
                lines.append(f"- HBM Bandwidth: {self.hbm_bandwidth_tbs} TB/s")
            if self.l2_cache_mb:
                lines.append(f"- L2 Cache: {self.l2_cache_mb} MB")
            if self.num_sms:
                lines.append(f"- SMs: {self.num_sms}")
            lines.append("")
        
        return "\n".join(lines)


@dataclass
class LLMAnalysisResult:
    """Result from LLM analysis."""
    provider: str
    model: str
    
    # Main analysis sections
    why_faster: str = ""
    root_cause_analysis: str = ""
    improvement_suggestions: List[str] = field(default_factory=list)
    code_suggestions: str = ""
    missed_opportunities: List[str] = field(default_factory=list)
    architecture_specific_tips: str = ""
    
    # Metadata
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    # Raw response for debugging
    raw_response: str = ""
    
    def to_markdown(self) -> str:
        lines = []
        lines.append("# LLM Performance Analysis")
        lines.append("")
        lines.append(f"*Analyzed by {self.model} ({self.provider})*")
        lines.append("")
        
        if self.why_faster:
            lines.append("## Why Is It Faster?")
            lines.append("")
            lines.append(self.why_faster)
            lines.append("")
        
        if self.root_cause_analysis:
            lines.append("## Root Cause Analysis")
            lines.append("")
            lines.append(self.root_cause_analysis)
            lines.append("")
        
        if self.improvement_suggestions:
            lines.append("## How to Improve Further")
            lines.append("")
            for i, suggestion in enumerate(self.improvement_suggestions, 1):
                lines.append(f"### {i}. {suggestion.split(chr(10))[0]}")
                if "\n" in suggestion:
                    lines.append("")
                    lines.append("\n".join(suggestion.split("\n")[1:]))
                lines.append("")
        
        if self.code_suggestions:
            lines.append("## Suggested Code Changes")
            lines.append("")
            lines.append(self.code_suggestions)
            lines.append("")
        
        if self.missed_opportunities:
            lines.append("## Missed Optimization Opportunities")
            lines.append("")
            for opp in self.missed_opportunities:
                lines.append(f"- {opp}")
            lines.append("")
        
        if self.architecture_specific_tips:
            lines.append("## Architecture-Specific Tips")
            lines.append("")
            lines.append(self.architecture_specific_tips)
            lines.append("")

        if self.warnings:
            lines.append("## Analysis Warnings")
            lines.append("")
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")
        
        return "\n".join(lines)


def _read_optional_source_code(path: Path, *, label: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        return path.read_text(encoding="utf-8"), None
    except Exception as exc:
        return None, f"Failed to read {label} source code from {path}: {exc}"


def collect_environment_context() -> EnvironmentContext:
    """Collect comprehensive environment information."""
    ctx = EnvironmentContext()
    
    # Python version
    ctx.python_version = sys.version.split()[0]
    
    # OS info
    import platform
    ctx.os_info = f"{platform.system()} {platform.release()}"
    
    # PyTorch and CUDA
    try:
        import torch
        ctx.pytorch_version = torch.__version__
        ctx.torch_cuda_version = torch.version.cuda or "N/A"
        
        if torch.cuda.is_available():
            ctx.cuda_version = torch.version.cuda or ""
            
            # GPU info
            props = torch.cuda.get_device_properties(0)
            ctx.gpu_name = props.name
            ctx.compute_capability = f"{props.major}.{props.minor}"
            ctx.gpu_memory_gb = props.total_memory / (1024**3)
            ctx.num_sms = props.multi_processor_count
            ctx.num_gpus = torch.cuda.device_count()
            
            # Determine architecture
            if props.major == 10:
                ctx.gpu_arch = "Blackwell"
                # B200 specs
                ctx.peak_fp32_tflops = 250.0
                ctx.peak_bf16_tflops = 5040.0
                ctx.peak_fp8_tflops = 10080.0
                ctx.hbm_bandwidth_tbs = 8.0
                ctx.l2_cache_mb = 96.0
            elif props.major == 9:
                ctx.gpu_arch = "Hopper"
                ctx.peak_fp32_tflops = 67.0
                ctx.peak_bf16_tflops = 1979.0
                ctx.peak_fp8_tflops = 3958.0
                ctx.hbm_bandwidth_tbs = 3.35
                ctx.l2_cache_mb = 50.0
            elif props.major == 8:
                ctx.gpu_arch = "Ampere"
                ctx.peak_fp32_tflops = 19.5
                ctx.peak_bf16_tflops = 312.0
                ctx.hbm_bandwidth_tbs = 2.0
            else:
                ctx.gpu_arch = f"SM {props.major}.{props.minor}"
    except ImportError:
        pass
    
    # Comprehensive GPU info from nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", 
             "--query-gpu=driver_version,temperature.gpu,power.draw,utilization.gpu,utilization.memory,clocks.gr,clocks.mem,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(", ")]
            if len(parts) >= 1:
                ctx.cuda_driver_version = parts[0]
            if len(parts) >= 2:
                try:
                    ctx.gpu_temperature_c = float(parts[1])
                except ValueError:
                    pass
            if len(parts) >= 3:
                try:
                    ctx.gpu_power_draw_w = float(parts[2])
                except ValueError:
                    pass
            if len(parts) >= 4:
                try:
                    ctx.gpu_utilization_pct = float(parts[3])
                except ValueError:
                    pass
            if len(parts) >= 5:
                try:
                    ctx.memory_utilization_pct = float(parts[4])
                except ValueError:
                    pass
            if len(parts) >= 6:
                try:
                    ctx.gpu_clock_mhz = int(parts[5])
                except ValueError:
                    pass
            if len(parts) >= 7:
                try:
                    ctx.memory_clock_mhz = int(parts[6])
                except ValueError:
                    pass
            if len(parts) >= 8:
                try:
                    ctx.memory_used_gb = float(parts[7]) / 1024  # MiB to GB
                except ValueError:
                    pass
            if len(parts) >= 9:
                try:
                    ctx.memory_free_gb = float(parts[8]) / 1024  # MiB to GB
                except ValueError:
                    pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # NVLink info
    try:
        result = subprocess.run(
            ["nvidia-smi", "nvlink", "--status"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and "Link" in result.stdout:
            ctx.nvlink_connected = True
            # Try to get NVLink bandwidth info
            if "NVLink" in result.stdout:
                # B200 NVLink-C2C has ~1.8 TB/s bidirectional
                if "B200" in ctx.gpu_name or ctx.gpu_arch == "Blackwell":
                    ctx.nvlink_bandwidth_gbps = 1800.0
                elif "H100" in ctx.gpu_name or ctx.gpu_arch == "Hopper":
                    ctx.nvlink_bandwidth_gbps = 900.0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # cuDNN version
        import torch.backends.cudnn as cudnn
        if cudnn.is_available():
            ctx.cudnn_version = str(cudnn.version())
    
    # NCCL version
        import torch.distributed as dist
        if hasattr(dist, 'get_nccl_version'):
            ctx.nccl_version = str(dist.get_nccl_version())
    
    # Triton
    try:
        import triton
        ctx.triton_version = triton.__version__
    except ImportError:
        pass
    
    # Transformer Engine
    try:
        import transformer_engine
        ctx.transformer_engine_version = transformer_engine.__version__
    except ImportError:
        pass
    
    # Flash Attention
    try:
        import flash_attn
        ctx.flash_attn_version = flash_attn.__version__
    except ImportError:
        pass
    
    # xFormers
    try:
        import xformers
        ctx.xformers_version = xformers.__version__
    except ImportError:
        pass
    
    # NVIDIA Apex
    try:
        import apex
        ctx.apex_version = apex.__version__
    except (ImportError, AttributeError):
        pass
    
    # CUDA library versions via torch
    try:
        import torch
        # cuBLAS version
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
            # cuBLAS is used internally, version tied to CUDA
            ctx.cublas_version = ctx.cuda_version
    except ImportError:
        pass  # torch not available
    
    return ctx


class LLMProfileAnalyzer:
    """
    LLM-enhanced profile analyzer using Claude or GPT.
    
    Provides deep analysis of profiling data by combining structured
    metrics with LLM reasoning capabilities.
    """
    
    SUPPORTED_PROVIDERS = {"anthropic", "openai"}
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the LLM analyzer.
        
        Args:
            provider: "anthropic" or "openai" (default from LLM_PROVIDER env)
            model: Model name (default from env or provider default)
            api_key: API key (default from env)
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
        
        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {self.provider}. Use: {self.SUPPORTED_PROVIDERS}")
        
        # Set model defaults
        if model:
            self.model = model
        elif self.provider == "anthropic":
            self.model = os.getenv("ANTHROPIC_MODEL") or os.getenv("PERF_LLM_MODEL") or DEFAULT_ANTHROPIC_MODEL
        else:  # openai
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        # Get API key
        if api_key:
            self.api_key = api_key
        elif self.provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                f"No API key found for {self.provider}. "
                f"Set {'ANTHROPIC_API_KEY' if self.provider == 'anthropic' else 'OPENAI_API_KEY'} "
                "in environment or .env file."
            )
        
        self._client = None
    
    @property
    def client(self):
        """Lazy-load the API client."""
        if self._client is None:
            if self.provider == "anthropic":
                try:
                    import anthropic
                    self._client = anthropic.Anthropic(api_key=self.api_key)
                except ImportError:
                    raise ImportError("Install anthropic: pip install anthropic")
            else:
                try:
                    import openai
                    self._client = openai.OpenAI(api_key=self.api_key)
                except ImportError:
                    raise ImportError("Install openai: pip install openai")
        return self._client
    
    def analyze_differential(
        self,
        differential_report: Dict[str, Any],
        baseline_code: Optional[str] = None,
        optimized_code: Optional[str] = None,
        environment: Optional[EnvironmentContext] = None,
        additional_context: Optional[str] = None,
    ) -> LLMAnalysisResult:
        """
        Analyze a differential profiling report using LLM.
        
        Args:
            differential_report: Output from differential_profile_analyzer (as dict)
            baseline_code: Source code of baseline implementation
            optimized_code: Source code of optimized implementation
            environment: Environment context (auto-collected if None)
            additional_context: Any additional context to include
        
        Returns:
            LLMAnalysisResult with detailed analysis
        """
        import time
        start_time = time.time()
        
        # Collect environment if not provided
        if environment is None:
            environment = collect_environment_context()
        
        # Build the prompt
        prompt = self._build_analysis_prompt(
            differential_report,
            baseline_code,
            optimized_code,
            environment,
            additional_context,
        )
        
        # Call the LLM
        response, tokens = self._call_llm(prompt)
        
        # Parse the response
        result = self._parse_response(response)
        result.provider = self.provider
        result.model = self.model
        result.latency_seconds = time.time() - start_time
        result.prompt_tokens = tokens.get("prompt", 0)
        result.completion_tokens = tokens.get("completion", 0)
        result.raw_response = response
        
        return result
    
    def _extract_class_structure(self, code: str) -> str:
        """Extract class structure showing available attributes and methods."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return "(Could not parse code structure)"
        
        structure_parts = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                attributes = []
                methods = []
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        # Get method signature
                        args = []
                        for arg in item.args.args:
                            args.append(arg.arg)
                        methods.append(f"{item.name}({', '.join(args)})")
                        
                        # Extract self.xxx assignments in __init__
                        if item.name == '__init__':
                            for stmt in ast.walk(item):
                                if isinstance(stmt, ast.Assign):
                                    for target in stmt.targets:
                                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                                            if target.value.id == 'self':
                                                attributes.append(f"self.{target.attr}")
                
                if attributes or methods:
                    structure_parts.append(f"### class {class_name}")
                    if attributes:
                        structure_parts.append("**Attributes (from __init__):**")
                        for attr in sorted(set(attributes)):
                            structure_parts.append(f"  - `{attr}`")
                    if methods:
                        structure_parts.append("**Methods:**")
                        for method in methods:
                            structure_parts.append(f"  - `{method}`")
                    structure_parts.append("")
        
        return "\n".join(structure_parts) if structure_parts else "(No classes found)"
    
    def _build_analysis_prompt(
        self,
        differential_report: Dict[str, Any],
        baseline_code: Optional[str],
        optimized_code: Optional[str],
        environment: EnvironmentContext,
        additional_context: Optional[str],
    ) -> str:
        """Build the analysis prompt."""
        
        speedup = differential_report.get("overall_speedup", 1.0)
        baseline_time = differential_report.get("total_baseline_time_ms", 0)
        optimized_time = differential_report.get("total_optimized_time_ms", 0)
        
        prompt_parts = []
        
        # System context
        prompt_parts.append(f"""You are an expert GPU performance engineer analyzing profiling data from a {environment.gpu_arch} GPU ({environment.gpu_name}).

Your task is to:
1. Explain WHY the optimized version is faster (root cause, not just symptoms)
2. Suggest SPECIFIC code changes for further improvement
3. Identify any missed optimization opportunities
4. Provide architecture-specific tips for {environment.gpu_arch}

Be concrete and specific. Reference actual metrics and suggest actual code patterns.""")
        
    # Environment context
        prompt_parts.append("\n" + environment.to_markdown())
        
        # Performance summary
        prompt_parts.append(f"""
## Performance Results

- **Baseline Time:** {baseline_time:.3f} ms
- **Optimized Time:** {optimized_time:.3f} ms  
- **Speedup:** {speedup:.2f}x
- **Time Saved:** {baseline_time - optimized_time:.3f} ms
""")
        
    # Binding analysis
        binding_shift = differential_report.get("binding_shift")
        if binding_shift:
            prompt_parts.append(f"- **Binding Shift:** {binding_shift}")
        
        prompt_parts.append(f"- **Baseline Dominant Binding:** {differential_report.get('baseline_dominant_binding', 'unknown')}")
        prompt_parts.append(f"- **Optimized Dominant Binding:** {differential_report.get('optimized_dominant_binding', 'unknown')}")
        
        # Kernel-level data
        kernel_diffs = differential_report.get("kernel_diffs", [])
        if kernel_diffs:
            prompt_parts.append("\n## Kernel-Level Analysis\n")
            prompt_parts.append("| Kernel | Baseline (ms) | Optimized (ms) | Speedup | Binding Change | Primary Improvement |")
            prompt_parts.append("|--------|---------------|----------------|---------|----------------|---------------------|")
            
            for kd in kernel_diffs[:10]:
                b_time = f"{kd['baseline_time_ms']:.3f}" if kd.get('baseline_time_ms') else "N/A"
                o_time = f"{kd['optimized_time_ms']:.3f}" if kd.get('optimized_time_ms') else "N/A"
                speedup_str = f"{kd.get('speedup', 1.0):.2f}x"
                binding = f"{kd.get('baseline_binding', '?')} → {kd.get('optimized_binding', '?')}" if kd.get('binding_changed') else kd.get('optimized_binding', 'unknown')
                improvement = kd.get('primary_improvement', 'unknown')
                prompt_parts.append(f"| {kd.get('name', 'unknown')[:30]} | {b_time} | {o_time} | {speedup_str} | {binding} | {improvement} |")
        
        # Metric deltas
        prompt_parts.append("\n## Key Metric Changes (Optimized - Baseline)\n")
        for kd in kernel_diffs[:5]:
            name = kd.get('name', 'unknown')
            deltas = []
            if kd.get('sm_util_delta') is not None:
                deltas.append(f"SM Util: {kd['sm_util_delta']:+.1f}%")
            if kd.get('dram_util_delta') is not None:
                deltas.append(f"DRAM Util: {kd['dram_util_delta']:+.1f}%")
            if kd.get('occupancy_delta') is not None:
                deltas.append(f"Occupancy: {kd['occupancy_delta']:+.1f}%")
            if kd.get('l2_hit_delta') is not None:
                deltas.append(f"L2 Hit: {kd['l2_hit_delta']:+.1f}%")
            if kd.get('arithmetic_intensity_delta') is not None:
                deltas.append(f"AI: {kd['arithmetic_intensity_delta']:+.2f}")
            
            if deltas:
                prompt_parts.append(f"- **{name}:** {', '.join(deltas)}")
        
        # Remaining bottlenecks
        bottlenecks = differential_report.get("remaining_bottlenecks", [])
        if bottlenecks:
            prompt_parts.append("\n## Remaining Bottlenecks\n")
            for bn in bottlenecks:
                prompt_parts.append(f"- {bn}")
        
        # Source code if available
        # Detect language from code content
        def detect_lang(code: str) -> str:
            """Detect if code is CUDA C++ or Python."""
            cuda_indicators = ['__global__', '__device__', '__shared__', 'cudaMalloc', 
                              '#include <cuda', 'blockIdx', 'threadIdx', '<<<', '>>>']
            if any(ind in code for ind in cuda_indicators):
                return 'cuda'
            return 'python'
        
        # Include FULL source code - no truncation for context completeness
        # Claude has 200K context, GPT-4 has 128K - we can afford generous limits
        max_chars = 50000  # 50K chars per file
        
        if baseline_code:
            lang = detect_lang(baseline_code)
            prompt_parts.append("\n## Baseline Code (FULL FILE)\n")
            prompt_parts.append(f"```{lang}")
            prompt_parts.append(baseline_code[:max_chars])
            if len(baseline_code) > max_chars:
                prompt_parts.append(f"\n// ... ({len(baseline_code) - max_chars} more characters truncated)")
            prompt_parts.append("```")
        
        if optimized_code:
            lang = detect_lang(optimized_code)
            prompt_parts.append("\n## Optimized Code (FULL FILE - TARGET FOR IMPROVEMENT)\n")
            prompt_parts.append(f"```{lang}")
            prompt_parts.append(optimized_code[:max_chars])
            if len(optimized_code) > max_chars:
                prompt_parts.append(f"\n// ... ({len(optimized_code) - max_chars} more characters truncated)")
            prompt_parts.append("```")
            
            # Extract and list existing class attributes and methods
            prompt_parts.append("\n## Existing Class Structure (from optimized code)\n")
            prompt_parts.append(self._extract_class_structure(optimized_code))
        
        # Additional context
        if additional_context:
            prompt_parts.append(f"\n## Additional Context\n\n{additional_context}")
        
        # Request format - comprehensive JSON structured output
        gpu_arch = environment.gpu_arch
        prompt_parts.append(f"""
## Your Task

Analyze the performance data and provide optimization patches. Output valid JSON only.

**CRITICAL CONSTRAINTS:**
1. Your code patches MUST be completely self-contained
2. You can ONLY use attributes that ALREADY EXIST in the class (listed above) OR that you ADD in init_changes
3. If you reference a helper method like `self._my_helper()`, you MUST provide it in method_replacements
4. Every method you call must either exist in the original code or be provided by you
5. Provide 1-2 HIGH-QUALITY patches rather than many broken ones

## JSON Output Format

```json
{{
  "why_faster": "Concise explanation of the current optimization...",
  "root_cause_analysis": "Hardware-level analysis of what changed...",
  "improvement_suggestions": [
    {{
      "title": "Optimization name",
      "description": "What it does and why it helps on {gpu_arch}",
      "expected_speedup": "1.Xx-1.Yx",
      "difficulty": "easy|medium|hard"
    }}
  ],
  "code_patches": [
    {{
      "variant_name": "descriptive_snake_case_name",
      "description": "What this optimization does",
      "expected_speedup": "1.Xx",
      "new_imports": ["import statement if needed"],
      "init_changes": {{
        "add_attributes": ["self.new_attr = value"]
      }},
      "method_replacements": [
        {{
          "class_name": "ExactClassName",
          "method_name": "method_name",
          "complete_code": "def method_name(self, ...args):\\n    # COMPLETE implementation\\n    # No placeholders or TODOs\\n    pass"
        }}
      ],
      "new_methods": [
        {{
          "class_name": "ExactClassName",
          "method_name": "_helper_method",
          "complete_code": "def _helper_method(self, x):\\n    # Helper that other methods call\\n    return x"
        }}
      ]
    }}
  ],
  "missed_opportunities": ["Optimizations not yet applied"],
  "architecture_tips": ["Tips specific to {gpu_arch}"]
}}
```

## Self-Containment Checklist (mentally verify before outputting):
- [ ] Every `self.xxx` I use is either in the original class OR in my init_changes
- [ ] Every `self._method()` I call is either in the original class OR in my method_replacements/new_methods
- [ ] My setup() initializes everything that benchmark_fn() uses
- [ ] My code would pass `python -c "import ast; ast.parse(code)"`
- [ ] I'm not assuming any methods exist that I haven't seen in the source code

## Response Guidelines:
- Output ONLY the JSON object, no markdown code fences around it
- Escape newlines as \\n in code strings
- Keep the JSON valid - use proper escaping for quotes
- Limit to 2 patches maximum to ensure quality
- Each patch should be independently testable
""")
        
        return "\n".join(prompt_parts)
    
    def _try_parse_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Try to parse the response as JSON."""
        import re
        
        # Try to extract JSON from the response
        # The LLM might wrap it in ```json ... ```, output raw JSON, 
        # or mix markdown headers with JSON content
        
        # First, try to find JSON in a code block
        json_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', response)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find a complete JSON object anywhere in the response
        # Look for content between first { and last matching }
        try:
            start = response.find('{')
            if start != -1:
                # Find the matching closing brace
                depth = 0
                end = start
                in_string = False
                escape_next = False
                
                for i, c in enumerate(response[start:], start):
                    if escape_next:
                        escape_next = False
                        continue
                    if c == '\\':
                        escape_next = True
                        continue
                    if c == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                
                if end > start:
                    json_str = response[start:end]
                    return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Try the whole response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _call_llm(self, prompt: str) -> Tuple[str, Dict[str, int]]:
        """Call the LLM API."""
        tokens = {"prompt": 0, "completion": 0}
        
        # Use 16K tokens to ensure complete JSON responses with full method bodies
        max_output_tokens = 16384
        
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_output_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            tokens["prompt"] = response.usage.input_tokens
            tokens["completion"] = response.usage.output_tokens
            return text, tokens
        
        else:  # openai
            # Use max_completion_tokens for newer OpenAI models (gpt-4o, o1, etc.)
            # Fall back to max_tokens for older models
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=max_output_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as e:
                if "max_completion_tokens" in str(e) or "max_tokens" in str(e):
                    # Fallback for older models
                    response = self.client.chat.completions.create(
                        model=self.model,
                        max_tokens=max_output_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                else:
                    raise
            text = response.choices[0].message.content
            if response.usage:
                tokens["prompt"] = response.usage.prompt_tokens
                tokens["completion"] = response.usage.completion_tokens
            return text, tokens
    
    def refine_patches(
        self,
        original_analysis: str,
        error_message: str,
        original_code: str,
        max_attempts: int = 2,
    ) -> Optional[str]:
        """
        Request LLM to fix patches that failed.
        
        Args:
            original_analysis: The original LLM response with broken patches
            error_message: The error that occurred when applying/running patches
            original_code: The source code being patched
            max_attempts: Maximum refinement attempts
            
        Returns:
            Refined analysis JSON string, or None if refinement failed
        """
        refinement_prompt = f"""
Your previous optimization patches failed with this error:

```
{error_message}
```

Original code being patched:
```python
{original_code[:20000]}
```

Your previous response (relevant parts):
```
{original_analysis[:10000]}
```

Please provide FIXED patches that address the error. Common issues:
1. Referencing methods like `self._helper()` without providing them in new_methods
2. Using attributes like `self.xxx` without adding them to init_changes
3. Assuming methods exist that don't

Provide ONLY the corrected JSON output with fixed code_patches.
Include ALL necessary helper methods in new_methods if you reference them.

Output ONLY valid JSON with this structure:
{{
  "code_patches": [
    {{
      "variant_name": "fixed_variant_name",
      "description": "What was fixed",
      "expected_speedup": "1.Xx",
      "new_imports": [],
      "init_changes": {{"add_attributes": []}},
      "method_replacements": [
        {{"class_name": "ClassName", "method_name": "method", "complete_code": "def method(self):\\n    pass"}}
      ],
      "new_methods": [
        {{"class_name": "ClassName", "method_name": "_helper", "complete_code": "def _helper(self):\\n    pass"}}
      ]
    }}
  ]
}}
"""
        
        try:
            response, tokens = self._call_llm(refinement_prompt)
            logger.info(f"Refinement used {tokens.get('completion', 0)} completion tokens")
            return response
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return None
    
    def _parse_response(self, response: str) -> LLMAnalysisResult:
        """Parse the LLM response - tries JSON first, falls back to markdown."""
        result = LLMAnalysisResult(provider=self.provider, model=self.model)
        
        import re
        
        # Try to parse as JSON first
        json_parsed = self._try_parse_json(response)
        if json_parsed:
            result.why_faster = json_parsed.get("why_faster", "")
            result.root_cause_analysis = json_parsed.get("root_cause_analysis", "")
            result.architecture_specific_tips = "\n".join(json_parsed.get("architecture_tips", []))
            
            # Parse improvement suggestions
            suggestions = json_parsed.get("improvement_suggestions", [])
            result.improvement_suggestions = [
                f"{s.get('title', '')}: {s.get('description', '')} (Expected: {s.get('expected_speedup', 'unknown')})"
                for s in suggestions
            ]
            
            # Parse missed opportunities
            result.missed_opportunities = json_parsed.get("missed_opportunities", [])
            
            # Store code patches as JSON in code_suggestions for the patch applier
            code_patches = json_parsed.get("code_patches", [])
            if code_patches:
                result.code_suggestions = json.dumps({"patches": code_patches}, indent=2)
            
            logger.info(f"Parsed JSON response with {len(code_patches)} code patches")
            return result
        
        # Fallback: Parse as markdown
        logger.info("JSON parsing failed, falling back to markdown parsing")
        sections = {}
        current_section = None
        current_content = []
        
        for line in response.split("\n"):
            line_lower = line.lower().strip()
            # Check for section headers (## or ###, case-insensitive)
            if re.match(r'^#{2,3}\s*why\s*(is\s+it\s+)?faster', line_lower):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "why_faster"
                current_content = []
            elif re.match(r'^#{2,3}\s*root\s*cause', line_lower):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "root_cause"
                current_content = []
            elif re.match(r'^#{2,3}\s*(how\s+to\s+)?improve', line_lower):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "improvements"
                current_content = []
            elif re.match(r'^#{2,3}\s*(code\s+)?suggest|^#{2,3}\s*function\s+replace', line_lower):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "code"
                current_content = []
            elif re.match(r'^#{2,3}\s*missed', line_lower):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "missed"
                current_content = []
            elif re.match(r'^#{2,3}\s*architecture', line_lower):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "arch"
                current_content = []
            else:
                current_content.append(line)
        
        # Don't forget the last section
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()
        
        # Map to result
        result.why_faster = sections.get("why_faster", "")
        result.root_cause_analysis = sections.get("root_cause", "")
        result.code_suggestions = sections.get("code", "")
        result.architecture_specific_tips = sections.get("arch", "")
        
        # Parse improvements as list
        improvements_text = sections.get("improvements", "")
        if improvements_text:
            # Split by numbered items or bullet points
            items = re.split(r'\n(?=\d+\.|\*|-)', improvements_text)
            result.improvement_suggestions = [item.strip() for item in items if item.strip()]
        
        # Parse missed opportunities as list
        missed_text = sections.get("missed", "")
        if missed_text:
            lines = [line.strip() for line in missed_text.split("\n") if line.strip()]
            result.missed_opportunities = [line.lstrip("*- ") for line in lines if line]
        
        # If parsing failed, store raw response
        if not result.why_faster and not result.improvement_suggestions:
            result.why_faster = response
        
        return result


def analyze_with_llm(
    differential_json_path: Path,
    baseline_code_path: Optional[Path] = None,
    optimized_code_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    provider: Optional[str] = None,
) -> Optional[LLMAnalysisResult]:
    """
    Convenience function to run LLM analysis on a differential report.
    
    Args:
        differential_json_path: Path to differential_*.json
        baseline_code_path: Optional path to baseline source code
        optimized_code_path: Optional path to optimized source code
        output_path: Optional path to save markdown output
        provider: LLM provider (anthropic or openai)
    
    Returns:
        LLMAnalysisResult or None if analysis fails
    """
    # Check if LLM analysis is enabled
    if os.getenv("LLM_ANALYSIS_ENABLED", "true").lower() == "false":
        logger.info("LLM analysis disabled via LLM_ANALYSIS_ENABLED=false")
        return None
    
    # Load differential report
    try:
        with differential_json_path.open() as f:
            diff_report = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load differential report: {e}")
        return None
    
    # Load source code if available
    baseline_code = None
    optimized_code = None
    source_warnings: List[str] = []
    
    if baseline_code_path and baseline_code_path.exists():
        baseline_code, warning = _read_optional_source_code(baseline_code_path, label="baseline")
        if warning is not None:
            source_warnings.append(warning)
            logger.warning(warning)
    if optimized_code_path and optimized_code_path.exists():
        optimized_code, warning = _read_optional_source_code(optimized_code_path, label="optimized")
        if warning is not None:
            source_warnings.append(warning)
            logger.warning(warning)
    
    # Run analysis
    try:
        analyzer = LLMProfileAnalyzer(provider=provider)
        result = analyzer.analyze_differential(
            diff_report,
            baseline_code=baseline_code,
            optimized_code=optimized_code,
        )
        result.warnings.extend(source_warnings)
        
        # Save output if requested
        if output_path:
            output_path.write_text(result.to_markdown())
            logger.info(f"LLM analysis saved to: {output_path}")
        
        return result
    
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        return None


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-Enhanced Profile Analyzer")
    parser.add_argument(
        "--differential",
        type=Path,
        required=True,
        help="Path to differential analysis JSON",
    )
    parser.add_argument(
        "--baseline-code",
        type=Path,
        help="Path to baseline source code",
    )
    parser.add_argument(
        "--optimized-code",
        type=Path,
        help="Path to optimized source code",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for markdown report",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        help="LLM provider (default: from LLM_PROVIDER env or anthropic)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print analysis to stdout",
    )
    
    args = parser.parse_args()
    
    result = analyze_with_llm(
        args.differential,
        baseline_code_path=args.baseline_code,
        optimized_code_path=args.optimized_code,
        output_path=args.output,
        provider=args.provider,
    )
    
    if result and args.print:
        print(result.to_markdown())
    
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
