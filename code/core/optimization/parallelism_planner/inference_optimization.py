"""
Inference Optimization Module for Parallelism Planner

Comprehensive inference optimization recommendations:
- Quantization strategies (AWQ, GPTQ, GGUF, FP8, INT8)
- KV cache optimization (paged attention, FP8 KV)
- Speculative decoding recommendations
- Continuous batching settings
- vLLM/TensorRT-LLM configurations
- Inference engine recommendations
"""

import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class QuantizationType(Enum):
    """Quantization methods for inference."""
    NONE = "none"
    FP8 = "fp8"           # Weight-only FP8
    INT8 = "int8"         # Weight-only INT8
    INT4 = "int4"         # Weight-only INT4
    AWQ = "awq"           # Activation-aware Weight Quantization
    GPTQ = "gptq"         # GPTQ quantization
    GGUF = "gguf"         # GGML Universal Format
    SQUEEZELLM = "squeezellm"  # SqueezeLLM
    SMOOTH_QUANT = "smooth_quant"  # SmoothQuant (W8A8)


class InferenceEngine(Enum):
    """Inference engine options."""
    VLLM = "vllm"
    TENSORRT_LLM = "tensorrt_llm"
    HUGGINGFACE = "huggingface"
    LLAMA_CPP = "llama_cpp"
    DEEPSPEED_INFERENCE = "deepspeed_inference"
    TEXT_GEN_INFERENCE = "text_generation_inference"


@dataclass
class QuantizationRecommendation:
    """Quantization strategy recommendation."""
    method: QuantizationType
    bits: int
    memory_reduction_pct: float
    accuracy_impact: str  # "none", "minimal", "moderate", "significant"
    throughput_change_pct: float
    latency_change_pct: float
    
    # Configuration
    config: Dict[str, Any]
    
    # Requirements
    requirements: List[str]
    
    # Rationale
    rationale: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "bits": self.bits,
            "memory_reduction_pct": self.memory_reduction_pct,
            "accuracy_impact": self.accuracy_impact,
            "throughput_change_pct": self.throughput_change_pct,
            "latency_change_pct": self.latency_change_pct,
            "config": self.config,
            "requirements": self.requirements,
            "rationale": self.rationale,
        }


@dataclass
class KVCacheOptimization:
    """KV cache optimization recommendation."""
    paged_attention: bool
    kv_cache_dtype: str  # "fp16", "fp8", "int8"
    block_size: int
    max_num_seqs: int
    max_model_len: int
    
    # Memory savings
    memory_per_token_bytes: float
    total_kv_cache_gb: float
    
    # Configuration
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paged_attention": self.paged_attention,
            "kv_cache_dtype": self.kv_cache_dtype,
            "block_size": self.block_size,
            "max_num_seqs": self.max_num_seqs,
            "max_model_len": self.max_model_len,
            "memory": {
                "per_token_bytes": self.memory_per_token_bytes,
                "total_kv_cache_gb": self.total_kv_cache_gb,
            },
            "config": self.config,
        }


@dataclass
class SpeculativeDecodingRecommendation:
    """Speculative decoding configuration."""
    enabled: bool
    method: str  # "draft_model", "medusa", "eagle", "lookahead"
    
    # Draft model (if applicable)
    draft_model: Optional[str]
    draft_model_params_b: Optional[float]
    
    # Expected speedup
    expected_speedup: float
    acceptance_rate: float
    
    # Configuration
    config: Dict[str, Any]
    
    # Trade-offs
    trade_offs: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "draft_model": self.draft_model,
            "draft_model_params_b": self.draft_model_params_b,
            "expected_speedup": self.expected_speedup,
            "acceptance_rate": self.acceptance_rate,
            "config": self.config,
            "trade_offs": self.trade_offs,
        }


@dataclass
class InferenceEngineRecommendation:
    """Inference engine recommendation."""
    engine: InferenceEngine
    version: str
    
    # Performance expectations
    expected_throughput_tps: float
    expected_latency_ms: float
    expected_memory_gb: float
    
    # Configuration
    launch_command: str
    config: Dict[str, Any]
    
    # Features
    features: List[str]
    limitations: List[str]

    # A directly executable argument vector is present only for single-command plans.
    launch_argv: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine.value,
            "version": self.version,
            "performance": {
                "throughput_tps": self.expected_throughput_tps,
                "latency_ms": self.expected_latency_ms,
                "memory_gb": self.expected_memory_gb,
            },
            "launch_command": self.launch_command,
            "launch_argv": self.launch_argv,
            "config": self.config,
            "features": self.features,
            "limitations": self.limitations,
        }


@dataclass
class InferenceOptimizationReport:
    """Complete inference optimization report."""
    model_name: str
    model_params_b: float
    
    # Recommendations
    quantization: QuantizationRecommendation
    kv_cache: KVCacheOptimization
    speculative_decoding: Optional[SpeculativeDecodingRecommendation]
    engine: InferenceEngineRecommendation
    
    # Summary metrics
    original_memory_gb: float
    optimized_memory_gb: float
    memory_reduction_pct: float
    expected_speedup: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": {
                "name": self.model_name,
                "parameters_billions": self.model_params_b,
            },
            "quantization": self.quantization.to_dict(),
            "kv_cache": self.kv_cache.to_dict(),
            "speculative_decoding": self.speculative_decoding.to_dict() if self.speculative_decoding else None,
            "engine": self.engine.to_dict(),
            "summary": {
                "original_memory_gb": self.original_memory_gb,
                "optimized_memory_gb": self.optimized_memory_gb,
                "memory_reduction_pct": self.memory_reduction_pct,
                "expected_speedup": self.expected_speedup,
            },
        }


class QuantizationOptimizer:
    """Recommends optimal quantization strategy."""
    
    QUANTIZATION_SPECS = {
        QuantizationType.NONE: {
            "bits": 16,
            "memory_reduction": 0,
            "accuracy_impact": "none",
            "throughput_change": 0,
            "latency_change": 0,
        },
        QuantizationType.FP8: {
            "bits": 8,
            "memory_reduction": 50,
            "accuracy_impact": "minimal",
            "throughput_change": 80,  # Faster on Hopper+
            "latency_change": -20,
        },
        QuantizationType.INT8: {
            "bits": 8,
            "memory_reduction": 50,
            "accuracy_impact": "minimal",
            "throughput_change": 30,
            "latency_change": -10,
        },
        QuantizationType.AWQ: {
            "bits": 4,
            "memory_reduction": 75,
            "accuracy_impact": "minimal",
            "throughput_change": 50,
            "latency_change": -30,
        },
        QuantizationType.GPTQ: {
            "bits": 4,
            "memory_reduction": 75,
            "accuracy_impact": "moderate",
            "throughput_change": 40,
            "latency_change": -20,
        },
        QuantizationType.GGUF: {
            "bits": 4,
            "memory_reduction": 75,
            "accuracy_impact": "minimal",
            "throughput_change": 30,
            "latency_change": -10,
        },
    }
    
    def __init__(self, gpu_arch: str = "hopper"):
        self.gpu_arch = gpu_arch.lower()
        self.supports_fp8 = gpu_arch in ["hopper", "blackwell", "h100", "h200", "b100", "b200"]
    
    def recommend(
        self,
        model_params_b: float,
        gpu_memory_gb: float,
        accuracy_priority: str = "balanced",  # "high", "balanced", "speed"
        use_case: str = "server",  # "server", "edge", "research"
    ) -> QuantizationRecommendation:
        """
        Recommend optimal quantization strategy.
        """
        
        # Calculate base memory requirement
        base_memory_gb = model_params_b * 2  # BF16
        
        # If model fits comfortably, minimal quantization
        if base_memory_gb < gpu_memory_gb * 0.6:
            if self.supports_fp8 and accuracy_priority != "high":
                return self._create_recommendation(
                    QuantizationType.FP8,
                    model_params_b,
                    "FP8 for throughput boost on Hopper+ with minimal accuracy loss"
                )
            else:
                return self._create_recommendation(
                    QuantizationType.NONE,
                    model_params_b,
                    "Model fits comfortably - no quantization needed"
                )
        
        # If model barely fits, use INT8/FP8
        if base_memory_gb < gpu_memory_gb:
            if self.supports_fp8:
                return self._create_recommendation(
                    QuantizationType.FP8,
                    model_params_b,
                    "FP8 reduces memory by 50% with minimal accuracy loss on Hopper+"
                )
            else:
                return self._create_recommendation(
                    QuantizationType.INT8,
                    model_params_b,
                    "INT8 reduces memory by 50% with minimal accuracy loss"
                )
        
        # If model doesn't fit, need aggressive quantization
        if accuracy_priority == "high":
            return self._create_recommendation(
                QuantizationType.AWQ,
                model_params_b,
                "AWQ 4-bit with best accuracy preservation"
            )
        else:
            return self._create_recommendation(
                QuantizationType.AWQ,
                model_params_b,
                "AWQ 4-bit for maximum memory reduction with good accuracy"
            )
    
    def _create_recommendation(
        self,
        method: QuantizationType,
        model_params_b: float,
        rationale: str,
    ) -> QuantizationRecommendation:
        """Create a quantization recommendation."""
        
        spec = self.QUANTIZATION_SPECS[method]
        
        # Generate config
        config = self._get_quantization_config(method, model_params_b)
        
        # Requirements
        requirements = self._get_requirements(method)
        
        return QuantizationRecommendation(
            method=method,
            bits=spec["bits"],
            memory_reduction_pct=spec["memory_reduction"],
            accuracy_impact=spec["accuracy_impact"],
            throughput_change_pct=spec["throughput_change"],
            latency_change_pct=spec["latency_change"],
            config=config,
            requirements=requirements,
            rationale=rationale,
        )
    
    def _get_quantization_config(
        self,
        method: QuantizationType,
        model_params_b: float,
    ) -> Dict[str, Any]:
        """Get configuration for quantization method."""
        
        if method == QuantizationType.FP8:
            return {
                "quantization": "fp8",
                "kv_cache_dtype": "fp8",
                "dtype": "float16",
            }
        elif method == QuantizationType.INT8:
            return {
                "quantization": "int8",
                "load_in_8bit": True,
            }
        elif method == QuantizationType.AWQ:
            return {
                "quantization": "awq",
                "quant_config": {
                    "zero_point": True,
                    "q_group_size": 128,
                    "w_bit": 4,
                    "version": "GEMM",
                }
            }
        elif method == QuantizationType.GPTQ:
            return {
                "quantization": "gptq",
                "bits": 4,
                "group_size": 128,
                "desc_act": True,
            }
        elif method == QuantizationType.GGUF:
            return {
                "quantization": "q4_k_m",
                "format": "gguf",
            }
        else:
            return {}
    
    def _get_requirements(self, method: QuantizationType) -> List[str]:
        """Get requirements for quantization method."""
        
        requirements = {
            QuantizationType.NONE: ["PyTorch 2.0+"],
            QuantizationType.FP8: ["Hopper/Blackwell GPU", "vLLM or TensorRT-LLM", "CUDA 12.0+"],
            QuantizationType.INT8: ["bitsandbytes", "PyTorch 2.0+"],
            QuantizationType.AWQ: ["AutoAWQ", "vLLM with AWQ support"],
            QuantizationType.GPTQ: ["AutoGPTQ", "vLLM with GPTQ support"],
            QuantizationType.GGUF: ["llama.cpp", "GGUF model weights"],
        }
        
        return requirements.get(method, [])


class KVCacheOptimizer:
    """Optimizes KV cache configuration."""
    
    def __init__(self, gpu_arch: str = "hopper", gpu_memory_gb: float = 80):
        self.gpu_arch = gpu_arch.lower()
        self.gpu_memory_gb = gpu_memory_gb
        self.supports_fp8_kv = gpu_arch in ["hopper", "blackwell", "h100", "h200", "b100", "b200"]
    
    def optimize(
        self,
        model_params_b: float,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_length: int = 4096,
        max_batch_size: int = 256,
    ) -> KVCacheOptimization:
        """
        Optimize KV cache configuration.
        """
        
        # Calculate KV cache size per token
        # KV cache = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
        
        if self.supports_fp8_kv:
            kv_dtype = "fp8"
            bytes_per_element = 1
        else:
            kv_dtype = "fp16"
            bytes_per_element = 2
        
        bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
        
        # Total KV cache for max settings
        total_kv_tokens = max_seq_length * max_batch_size
        total_kv_cache_gb = (total_kv_tokens * bytes_per_token) / 1e9
        
        # Model memory (approximate)
        model_memory_gb = model_params_b * 2  # BF16
        
        # Available for KV cache
        available_for_kv = self.gpu_memory_gb - model_memory_gb - 2  # 2GB headroom
        
        # Adjust max sequences if needed
        if total_kv_cache_gb > available_for_kv:
            max_tokens = int(available_for_kv * 1e9 / bytes_per_token)
            adjusted_max_seqs = max(1, max_tokens // max_seq_length)
        else:
            adjusted_max_seqs = max_batch_size
        
        # Block size for paged attention
        block_size = 16 if max_seq_length <= 8192 else 32
        
        return KVCacheOptimization(
            paged_attention=True,
            kv_cache_dtype=kv_dtype,
            block_size=block_size,
            max_num_seqs=adjusted_max_seqs,
            max_model_len=max_seq_length,
            memory_per_token_bytes=bytes_per_token,
            total_kv_cache_gb=min(total_kv_cache_gb, available_for_kv),
            config={
                "enable_prefix_caching": True,
                "block_size": block_size,
                "swap_space": 4,  # GB
                "max_num_batched_tokens": adjusted_max_seqs * max_seq_length,
            },
        )


class SpeculativeDecodingOptimizer:
    """Recommends speculative decoding configuration."""
    
    # Draft model mappings
    DRAFT_MODELS = {
        "llama-70b": ("llama-7b", 7),
        "llama-3.1-70b": ("llama-3.1-8b", 8),
        "llama-3.1-405b": ("llama-3.1-70b", 70),
        "mixtral-8x7b": ("mistral-7b", 7),
        "deepseek-v3": ("deepseek-7b", 7),
    }
    
    def recommend(
        self,
        model_name: str,
        model_params_b: float,
        gpu_memory_gb: float,
        latency_priority: bool = True,
    ) -> Optional[SpeculativeDecodingRecommendation]:
        """
        Recommend speculative decoding configuration.
        """
        
        # Only recommend for larger models where it helps
        if model_params_b < 20:
            return None
        
        # Check if we have a suitable draft model
        draft_info = self.DRAFT_MODELS.get(model_name.lower().replace("-", "_").replace(".", "_"))
        
        if draft_info:
            draft_model, draft_params = draft_info
            
            # Check if both models fit
            total_memory = (model_params_b + draft_params) * 2  # BF16
            
            if total_memory < gpu_memory_gb * 0.8:
                return SpeculativeDecodingRecommendation(
                    enabled=True,
                    method="draft_model",
                    draft_model=draft_model,
                    draft_model_params_b=draft_params,
                    expected_speedup=2.0,  # Typical 2x speedup
                    acceptance_rate=0.7,
                    config={
                        "speculative_model": draft_model,
                        "num_speculative_tokens": 5,
                        "speculative_draft_tensor_parallel_size": 1,
                    },
                    trade_offs=[
                        "Requires additional GPU memory for draft model",
                        "Effectiveness depends on draft/target model similarity",
                        "Best for greedy/low-temperature sampling",
                    ],
                )
        
        # Recommend Medusa heads as alternative
        return SpeculativeDecodingRecommendation(
            enabled=True,
            method="medusa",
            draft_model=None,
            draft_model_params_b=None,
            expected_speedup=1.5,
            acceptance_rate=0.6,
            config={
                "speculative_method": "medusa",
                "num_medusa_heads": 4,
                "medusa_num_layers": 1,
            },
            trade_offs=[
                "Requires Medusa-trained model or fine-tuning",
                "Lower speedup than draft model approach",
                "No additional model memory needed",
            ],
        )


class InferenceEngineRecommender:
    """Recommends optimal inference engine."""
    
    def recommend(
        self,
        model_name: str,
        model_params_b: float,
        num_gpus: int = 1,
        gpu_memory_gb: float = 80,
        quantization: QuantizationType = QuantizationType.NONE,
        use_case: str = "server",  # "server", "edge", "research"
        max_throughput: bool = True,
    ) -> InferenceEngineRecommendation:
        """
        Recommend optimal inference engine.
        """
        
        # vLLM for most production cases
        if use_case == "server" and max_throughput:
            return self._create_vllm_recommendation(
                model_name, model_params_b, num_gpus, gpu_memory_gb, quantization
            )
        
        # TensorRT-LLM for maximum performance
        if use_case == "server" and model_params_b < 100:
            return self._create_tensorrt_recommendation(
                model_name, model_params_b, num_gpus, gpu_memory_gb, quantization
            )
        
        # llama.cpp for edge/CPU
        if use_case == "edge" or quantization == QuantizationType.GGUF:
            return self._create_llama_cpp_recommendation(
                model_name, model_params_b
            )
        
        # Default to vLLM
        return self._create_vllm_recommendation(
            model_name, model_params_b, num_gpus, gpu_memory_gb, quantization
        )
    
    def _create_vllm_recommendation(
        self,
        model_name: str,
        model_params_b: float,
        num_gpus: int,
        gpu_memory_gb: float,
        quantization: QuantizationType,
    ) -> InferenceEngineRecommendation:
        """Create vLLM recommendation."""
        
        # Estimate TP needed
        model_memory = model_params_b * 2
        if quantization in [QuantizationType.AWQ, QuantizationType.GPTQ]:
            model_memory = model_params_b * 0.5
        
        tp = 1
        while model_memory / tp > gpu_memory_gb * 0.7:
            tp *= 2
        tp = min(tp, num_gpus)
        
        # Estimate performance
        throughput = self._estimate_throughput(model_params_b, tp, quantization)
        latency = self._estimate_latency(model_params_b, tp)
        
        # Build one executable argument vector and derive the display command from it.
        launch_argv = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_name,
            "--tensor-parallel-size",
            str(tp),
            "--gpu-memory-utilization",
            "0.9",
            "--max-model-len",
            "8192",
        ]
        if quantization == QuantizationType.AWQ:
            launch_argv.extend(["--quantization", "awq"])
        elif quantization == QuantizationType.FP8:
            launch_argv.extend(["--quantization", "fp8", "--kv-cache-dtype", "fp8"])
        
        launch_cmd = shlex.join(launch_argv)
        
        return InferenceEngineRecommendation(
            engine=InferenceEngine.VLLM,
            version="0.6.0+",
            expected_throughput_tps=throughput,
            expected_latency_ms=latency,
            expected_memory_gb=model_memory / tp,
            launch_command=launch_cmd,
            launch_argv=launch_argv,
            config={
                "tensor_parallel_size": tp,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 8192,
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
            },
            features=[
                "PagedAttention for efficient memory",
                "Continuous batching",
                "OpenAI-compatible API",
                "Prefix caching",
                "AWQ/GPTQ/FP8 quantization support",
            ],
            limitations=[
                "Requires CUDA GPUs",
                "Best with single-node deployments",
            ],
        )
    
    def _create_tensorrt_recommendation(
        self,
        model_name: str,
        model_params_b: float,
        num_gpus: int,
        gpu_memory_gb: float,
        quantization: QuantizationType,
    ) -> InferenceEngineRecommendation:
        """Create TensorRT-LLM recommendation."""
        
        tp = min(num_gpus, 8)
        
        throughput = self._estimate_throughput(model_params_b, tp, quantization) * 1.3  # TRT is faster
        latency = self._estimate_latency(model_params_b, tp) * 0.7
        
        build_argv = [
            "python",
            "build.py",
            "--model_dir",
            model_name,
            "--tp_size",
            str(tp),
            "--dtype",
            "bfloat16",
            "--use_gpt_attention_plugin",
            "bfloat16",
            "--use_gemm_plugin",
            "bfloat16",
            "--max_batch_size",
            "256",
            "--max_input_len",
            "4096",
            "--max_output_len",
            "4096",
        ]
        server_argv = [
            "mpirun",
            "-n",
            str(tp),
            "python",
            "run.py",
            "--engine_dir",
            "./engine",
        ]
        launch_cmd = (
            f"# Build TensorRT engine\n{shlex.join(build_argv)}\n\n"
            f"# Run inference server\n{shlex.join(server_argv)}"
        )
        
        return InferenceEngineRecommendation(
            engine=InferenceEngine.TENSORRT_LLM,
            version="0.12.0+",
            expected_throughput_tps=throughput,
            expected_latency_ms=latency,
            expected_memory_gb=model_params_b * 2 / tp,
            launch_command=launch_cmd,
            config={
                "tp_size": tp,
                "dtype": "bfloat16",
                "max_batch_size": 256,
                "use_inflight_batching": True,
            },
            features=[
                "Highest throughput with TensorRT optimization",
                "FP8 support on Hopper+",
                "In-flight batching",
                "KV cache reuse",
            ],
            limitations=[
                "Requires engine rebuild for different configs",
                "More complex setup",
                "NVIDIA GPUs only",
            ],
        )
    
    def _create_llama_cpp_recommendation(
        self,
        model_name: str,
        model_params_b: float,
    ) -> InferenceEngineRecommendation:
        """Create llama.cpp recommendation."""
        
        convert_argv = [
            "python",
            "convert-hf-to-gguf.py",
            model_name,
            "--outtype",
            "q4_k_m",
        ]
        server_argv = [
            "./server",
            "-m",
            "model-q4_k_m.gguf",
            "-ngl",
            "99",
            "-c",
            "4096",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
        ]
        launch_cmd = (
            f"# Convert to GGUF format (if needed)\n{shlex.join(convert_argv)}\n\n"
            f"# Run server\n{shlex.join(server_argv)}"
        )
        
        return InferenceEngineRecommendation(
            engine=InferenceEngine.LLAMA_CPP,
            version="latest",
            expected_throughput_tps=50,  # Conservative estimate
            expected_latency_ms=100,
            expected_memory_gb=model_params_b * 0.5,  # 4-bit quantized
            launch_command=launch_cmd,
            config={
                "quantization": "q4_k_m",
                "n_gpu_layers": 99,
                "context_size": 4096,
            },
            features=[
                "CPU + GPU inference",
                "Low memory with quantization",
                "Easy deployment",
                "GGUF format support",
            ],
            limitations=[
                "Lower throughput than vLLM/TensorRT",
                "Limited batching support",
            ],
        )
    
    def _estimate_throughput(
        self,
        model_params_b: float,
        tp: int,
        quantization: QuantizationType,
    ) -> float:
        """Estimate throughput in tokens/second."""
        
        # Base estimate for H100
        base = 1000 / model_params_b * tp
        
        # Quantization bonus
        if quantization == QuantizationType.FP8:
            base *= 1.8
        elif quantization in [QuantizationType.AWQ, QuantizationType.GPTQ]:
            base *= 1.5
        
        return base * 100  # Scale to reasonable tokens/s
    
    def _estimate_latency(
        self,
        model_params_b: float,
        tp: int,
    ) -> float:
        """Estimate latency in ms."""
        
        # Rough estimate
        return model_params_b / tp * 0.5


def get_inference_optimization_report(
    model_config: Dict[str, Any],
    hardware_config: Dict[str, Any],
    optimization_goal: str = "throughput",  # "throughput", "latency", "memory"
) -> Dict[str, Any]:
    """
    Generate comprehensive inference optimization report.
    """
    
    model_name = model_config.get("name", "model")
    model_params_b = model_config.get("parameters_billions", 70)
    num_layers = model_config.get("num_layers", 80)
    hidden_size = model_config.get("hidden_size", 8192)
    num_kv_heads = model_config.get("num_kv_heads", 8)
    head_dim = hidden_size // model_config.get("num_attention_heads", 64)
    max_seq_length = model_config.get("max_sequence_length", 4096)
    
    gpu_arch = hardware_config.get("gpu_arch", "hopper")
    gpu_memory_gb = hardware_config.get("gpu_memory_gb", 80)
    num_gpus = hardware_config.get("num_gpus", 1)
    
    # Quantization
    quant_optimizer = QuantizationOptimizer(gpu_arch)
    accuracy_priority = "high" if optimization_goal == "quality" else "balanced"
    quantization = quant_optimizer.recommend(
        model_params_b, gpu_memory_gb, accuracy_priority
    )
    
    # KV Cache
    kv_optimizer = KVCacheOptimizer(gpu_arch, gpu_memory_gb)
    kv_cache = kv_optimizer.optimize(
        model_params_b, num_layers, num_kv_heads, head_dim, max_seq_length
    )
    
    # Speculative Decoding
    spec_optimizer = SpeculativeDecodingOptimizer()
    speculative = spec_optimizer.recommend(
        model_name, model_params_b, gpu_memory_gb,
        latency_priority=(optimization_goal == "latency")
    )
    
    # Engine
    engine_recommender = InferenceEngineRecommender()
    engine = engine_recommender.recommend(
        model_name, model_params_b, num_gpus, gpu_memory_gb,
        quantization.method,
        max_throughput=(optimization_goal == "throughput")
    )
    
    # Summary
    original_memory = model_params_b * 2
    optimized_memory = original_memory * (1 - quantization.memory_reduction_pct / 100)
    
    report = InferenceOptimizationReport(
        model_name=model_name,
        model_params_b=model_params_b,
        quantization=quantization,
        kv_cache=kv_cache,
        speculative_decoding=speculative,
        engine=engine,
        original_memory_gb=original_memory,
        optimized_memory_gb=optimized_memory,
        memory_reduction_pct=quantization.memory_reduction_pct,
        expected_speedup=1 + quantization.throughput_change_pct / 100,
    )
    
    return report.to_dict()
