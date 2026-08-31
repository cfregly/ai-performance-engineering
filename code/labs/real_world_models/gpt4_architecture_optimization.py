#!/usr/bin/env python3
"""Real-world case study: GPT-4 architecture optimization for Blackwell.

Demonstrates optimization strategies for GPT-4 scale models:
- Expert parallelism for MoE layers
- Context parallelism for 128K context
- FP8 quantization
- Disaggregated prefill/decode
"""

import math
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkConfig, BenchmarkMode
from core.utils.logger import get_logger

logger = get_logger(__name__)


class GPT4ArchitectureOptimization:
    """GPT-4 architecture optimization benchmark."""
    
    # GPT-4 approximate specifications (publicly available estimates)
    HIDDEN_SIZE = 12288
    NUM_HEADS = 96
    NUM_LAYERS = 120
    NUM_EXPERTS_PER_LAYER = 16  # Estimated MoE configuration
    VOCAB_SIZE = 100277
    
    def __init__(
        self,
        batch_size: int = 1,
        seq_length: int = 8192,
        use_moe: bool = True,
        use_fp8: bool = True,
        use_context_parallel: bool = False,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        # The compact executable proxy below is a dense BF16 layer stack. Keep
        # the requested architecture assumptions for the memory estimate, but
        # report only optimizations that the timed workload actually executes.
        self.requested_use_moe = bool(use_moe)
        self.requested_use_fp8 = bool(use_fp8)
        self.requested_use_context_parallel = bool(use_context_parallel)
        self.use_moe = False
        self.use_fp8 = False
        self.use_context_parallel = False
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output: Optional[torch.Tensor] = None
        self._timing_events: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._last_elapsed_ms = 0.0
        self._last_tokens_per_sec = 0.0
        self._throughput_logged = False
        self._timing_pending = False
        
        logger.info("GPT-4 Architecture Optimization")
        logger.info("  Requested MoE architecture estimate: %s", self.requested_use_moe)
        logger.info("  Requested FP8 architecture estimate: %s", self.requested_use_fp8)
        logger.info(
            "  Requested context-parallel architecture estimate: %s",
            self.requested_use_context_parallel,
        )
        logger.info("  Executed proxy precision: BF16 dense")
        
        # Estimate memory requirements
        self._estimate_memory()
    
    def _estimate_memory(self):
        """Estimate memory requirements."""
        # Model parameters (approximate). MoE routing activates only a subset
        # of experts per token, but every expert's weights must remain stored.
        attention_params_per_layer = self.HIDDEN_SIZE * self.HIDDEN_SIZE * 4
        ffn_params_per_expert = self.HIDDEN_SIZE * self.HIDDEN_SIZE * 4 * 3
        if self.requested_use_moe:
            params_per_layer = (
                attention_params_per_layer
                + ffn_params_per_expert * self.NUM_EXPERTS_PER_LAYER
            )
        else:
            params_per_layer = attention_params_per_layer + ffn_params_per_expert

        total_params = params_per_layer * self.NUM_LAYERS

        # Memory in GB (FP16)
        param_memory_gb = (total_params * 2) / (1024**3)
        
        # KV cache
        kv_memory_gb = (
            self.batch_size * self.seq_length *
            self.NUM_LAYERS * 2 * self.NUM_HEADS *
            (self.HIDDEN_SIZE // self.NUM_HEADS) * 2
        ) / (1024**3)
        
        if self.requested_use_fp8:
            kv_memory_gb /= 2

        total_memory_gb = param_memory_gb + kv_memory_gb

        self.estimated_parameter_count = total_params
        self.estimated_parameter_memory_gb = param_memory_gb
        self.estimated_kv_memory_gb = kv_memory_gb
        self.estimated_total_memory_gb = total_memory_gb
        self.estimated_min_b200_gpus = max(1, math.ceil(total_memory_gb / 192))

        logger.info(f"Estimated memory: {total_memory_gb:.2f} GB")
        logger.info(f"  Parameters: {param_memory_gb:.2f} GB")
        logger.info(f"  KV cache: {kv_memory_gb:.2f} GB")

        if total_memory_gb > 192:  # B200 capacity
            logger.warning(f"Memory exceeds single B200 (192GB)")
            logger.info("Requires %d GPUs minimum", self.estimated_min_b200_gpus)
    
    def setup(self):
        """Initialize simplified GPT-4 model (for benchmarking)."""
        # Use smaller model for actual execution
        class SimplifiedGPT4Layer(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
                self.norm = nn.LayerNorm(hidden_size)
            
            def forward(self, x):
                return x + self.linear(self.norm(x))
        
        # Create just a few layers for benchmarking
        test_hidden = 4096  # Smaller for testing
        self.layers = nn.ModuleList([
            SimplifiedGPT4Layer(test_hidden)
            for _ in range(4)  # Test with 4 layers
        ]).to(self.device).to(torch.bfloat16).eval()
        
        # Create input
        self.input = torch.randn(
            self.batch_size,
            self.seq_length,
            test_hidden,
            device=self.device,
            dtype=torch.bfloat16
        )
        if self.device.type == "cuda":
            self._timing_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        self._last_elapsed_ms = 0.0
        self._throughput_logged = False
        self._timing_pending = False
        
        logger.info("Simplified GPT-4 model initialized")
    
    @torch.inference_mode()
    def run(self) -> float:
        """Execute forward pass."""
        if self.device.type == "cuda":
            if self._timing_events is None:
                raise RuntimeError("CUDA timing events are not initialized")
            start_event, end_event = self._timing_events
            current_stream = torch.cuda.current_stream(self.device)
            start_event.record(current_stream)
            x = self.input
            for layer in self.layers:
                x = layer(x)
            end_event.record(current_stream)
            self._timing_pending = True
            elapsed_ms = self._last_elapsed_ms
        else:
            start = time.perf_counter()
            x = self.input
            for layer in self.layers:
                x = layer(x)
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._last_elapsed_ms = elapsed_ms
        self.output = x

        return elapsed_ms

    def finalize_timing(self) -> float:
        """Resolve the last event pair and report throughput after timing."""
        if self.device.type == "cuda" and self._timing_pending:
            if self._timing_events is None:
                raise RuntimeError("CUDA timing events are not initialized")
            start_event, end_event = self._timing_events
            end_event.synchronize()
            self._last_elapsed_ms = start_event.elapsed_time(end_event)
            self._timing_pending = False
        elapsed_ms = self._last_elapsed_ms
        if elapsed_ms <= 0.0:
            raise RuntimeError("No completed timing sample is available")
        tokens_per_sec = (self.batch_size * self.seq_length) / (elapsed_ms / 1000)
        self._last_tokens_per_sec = tokens_per_sec

        if not self._throughput_logged:
            logger.info("Throughput: %.2f tokens/sec", tokens_per_sec)
            self._throughput_logged = True

        return elapsed_ms
    
    def cleanup(self):
        """Clean up."""
        del self.layers, self.input
        self.output = None
        self._timing_events = None
        self._timing_pending = False
        torch.cuda.empty_cache()


class GPT4ArchitectureOptimizationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Harness-friendly wrapper around the GPT-4 architecture demo."""

    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self.model_wrapper: Optional[GPT4ArchitectureOptimization] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count = 0
        self._last_metrics: Dict[str, float] = {
            "gpt4_architecture.mean_time_ms": 0.0,
            "gpt4_architecture.use_moe": 0.0,
            "gpt4_architecture.use_fp8": 0.0,
            "gpt4_architecture.use_context_parallel": 0.0,
        }
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("SKIPPED: GPT-4 architecture benchmark requires CUDA")
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.model_wrapper = GPT4ArchitectureOptimization()
        self.model_wrapper.setup()
        self.parameter_count = sum(p.numel() for p in self.model_wrapper.layers.parameters())
        self._verify_output_buffer = torch.empty(
            (1, min(4, self.model_wrapper.seq_length), min(8, self.model_wrapper.input.shape[2])),
            device=self.device,
            dtype=torch.float32,
        )

    def benchmark_fn(self) -> None:
        if self.model_wrapper is None:
            raise RuntimeError("Model wrapper not initialized")
        elapsed_ms = self.model_wrapper.run()
        self.output = self.model_wrapper.output
        metrics = self._last_metrics
        metrics["gpt4_architecture.mean_time_ms"] = float(elapsed_ms)
        metrics["gpt4_architecture.use_moe"] = 1.0 if self.model_wrapper.use_moe else 0.0
        metrics["gpt4_architecture.use_fp8"] = 1.0 if self.model_wrapper.use_fp8 else 0.0
        metrics["gpt4_architecture.use_context_parallel"] = (
            1.0 if self.model_wrapper.use_context_parallel else 0.0
        )
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.model_wrapper is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        elapsed_ms = self.model_wrapper.finalize_timing()
        metrics = self._last_metrics
        metrics["gpt4_architecture.mean_time_ms"] = float(elapsed_ms)
        metrics["gpt4_architecture.use_moe"] = 1.0 if self.model_wrapper.use_moe else 0.0
        metrics["gpt4_architecture.use_fp8"] = 1.0 if self.model_wrapper.use_fp8 else 0.0
        metrics["gpt4_architecture.use_context_parallel"] = (
            1.0 if self.model_wrapper.use_context_parallel else 0.0
        )
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
            : self._verify_output_buffer.shape[2],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"input": self.model_wrapper.input.detach()},
            output=self._verify_output_buffer,
            batch_size=self.model_wrapper.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={
                "bf16": True,
                "fp16": False,
                "fp8": bool(self.model_wrapper.use_fp8),
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        if self.model_wrapper is not None:
            self.model_wrapper.cleanup()
        self.model_wrapper = None
        self.output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=3, warmup=5)

    def get_custom_metrics(self) -> dict:
        return self._last_metrics


def run_benchmark(
    batch_size: int = 1,
    seq_length: int = 8192,
    use_moe: bool = True,
    use_fp8: bool = True,
    use_context_parallel: bool = False,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run GPT-4 architecture benchmark."""
    
    benchmark = GPT4ArchitectureOptimization(
        batch_size=batch_size,
        seq_length=seq_length,
        use_moe=use_moe,
        use_fp8=use_fp8,
        use_context_parallel=use_context_parallel,
    )
    benchmark.setup()
    
    config = BenchmarkConfig(iterations=3, warmup=5, profile_mode=profile)
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    
    result = harness.benchmark(benchmark.run, name="gpt4_architecture")
    benchmark.finalize_timing()
    benchmark.cleanup()
    
    return {
        "mean_time_ms": result.timing.mean_ms,
        "optimizations": {
            "moe": benchmark.use_moe,
            "fp8": benchmark.use_fp8,
            "context_parallel": benchmark.use_context_parallel,
        },
        "requested_architecture_assumptions": {
            "moe": benchmark.requested_use_moe,
            "fp8": benchmark.requested_use_fp8,
            "context_parallel": benchmark.requested_use_context_parallel,
        },
    }


def get_benchmark() -> BaseBenchmark:
    return GPT4ArchitectureOptimizationBenchmark()
