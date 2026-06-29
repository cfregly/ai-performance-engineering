#!/usr/bin/env python3
"""Base benchmark class for MoE Optimization Journey.

Each level inherits from this and sets its LEVEL constant.
Optimizations are applied cumulatively based on level.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from core.benchmark.verification import PrecisionFlags, simple_signature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.utils.compile_utils import get_optimal_compile_mode
from labs.moe_optimization_journey.moe_model import (
    MoEOptimizations,
    create_model,
)


# Optimization descriptions for each level
LEVEL_DESCRIPTIONS = {
    0: ("Naive", "Python loops over experts"),
    1: ("+ Batched", "Batched GEMMs parallelize all tokens"),
    2: ("+ Fused", "Triton kernel fuses SiLU*up"),
    3: ("+ MemEfficient", "Reuse buffers, reduce allocations"),
    4: ("+ Grouped", "Sort tokens + per-expert GEMM"),
    5: ("+ BMM Fusion", "Vectorized scatter + single BMM"),
    6: ("+ CUDAGraphs", "Capture and replay the fused level-5 kernel sequence"),
    7: ("+ Compiled", "torch.compile on top of the graph-friendly path"),
}


class MoEJourneyBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Base benchmark for MoE optimization journey.
    
    Subclasses just need to set LEVEL class variable.
    """
    
    LEVEL: int = 0  # Override in subclasses
    
    # Model configuration - Llama-7B like dimensions for realistic GPU utilization!
    VOCAB_SIZE = 32000
    HIDDEN_SIZE = 512        # aggressively trimmed for single-GPU demos
    INTERMEDIATE_SIZE = 2048  # scale down to avoid OOM
    NUM_LAYERS = 1           # Just 1 layer for benchmarking MoE
    NUM_HEADS = 32
    NUM_EXPERTS = 64
    NUM_EXPERTS_PER_TOK = 2
    BATCH_SIZE = 4    # lighter workload to keep footprint manageable
    SEQ_LEN = 128
    
    WARMUP = 3
    ITERATIONS = 10
    
    def __init__(self):
        super().__init__()
        self.model: Optional[Any] = None
        self.compiled_model: Optional[Any] = None
        self.input_ids: Optional[torch.Tensor] = None
        self.opts: Optional[MoEOptimizations] = None
        self.last_latency_ms: float = 0.0
        self.last_tokens_per_sec: float = 0.0
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        
        total_tokens = self.BATCH_SIZE * self.SEQ_LEN
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.BATCH_SIZE),
            tokens_per_iteration=float(total_tokens),
        )
    
    def setup(self) -> None:
        import gc
        
        # Clean up CUDA graph state from previous benchmarks
        # to prevent "Offset increment outside graph capture" errors
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            try:
                if hasattr(torch.cuda, 'graph_pool_trim'):
                    torch.cuda.graph_pool_trim()
            except Exception:
                pass
            
            # Reset CUDA RNG state to prevent graph capture errors
            try:
                device_idx = torch.cuda.current_device()
                gen = torch.cuda.default_generators[device_idx]
                gen.set_offset(0)
                gen.manual_seed(42)
            except Exception:
                pass
            
            try:
                torch._dynamo.reset()
            except Exception:
                pass
            
            try:
                torch._inductor.cudagraph_trees.reset_cudagraph_trees()
            except Exception:
                pass
        
        level = self.LEVEL
        name, desc = LEVEL_DESCRIPTIONS.get(level, (f"Level {level}", ""))
        
        print("=" * 60)
        print(f"LEVEL {level}: {name}")
        print("=" * 60)
        print(f"  {desc}")
        print()
        
        # Show cumulative optimizations
        print("  Optimizations enabled:")
        for l in range(level + 1):
            _, opt_desc = LEVEL_DESCRIPTIONS.get(l, ("", ""))
            if l == 0:
                print(f"    Level 0: {opt_desc}")
            else:
                print(f"    Level {l}: {opt_desc}")
        print()
        
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        # Create model with optimizations up to this level
        self.model, self.opts = create_model(
            level=level,
            vocab_size=self.VOCAB_SIZE,
            hidden_size=self.HIDDEN_SIZE,
            intermediate_size=self.INTERMEDIATE_SIZE,
            num_layers=self.NUM_LAYERS,
            num_heads=self.NUM_HEADS,
            num_experts=self.NUM_EXPERTS,
            num_experts_per_tok=self.NUM_EXPERTS_PER_TOK,
        )
        self.model = self.model.to(self.device).to(torch.bfloat16)
        self.model.eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {self.parameter_count / 1e6:.1f}M")
        print(f"  Batch: {self.BATCH_SIZE} x {self.SEQ_LEN} = {self.BATCH_SIZE * self.SEQ_LEN} tokens")
        
        # Apply torch.compile if enabled (Level 7). get_optimal_compile_mode keeps
        # max-autotune on the pinned toolchain but falls back to "default" on
        # sm_103 + Triton >= 3.6, where max-autotune emits an unloadable tcgen05
        # kernel (LLVM ERROR: Cannot select intrinsic %llvm.nvvm.tcgen05.wait.st).
        if self.opts.use_compile:
            _compile_mode = get_optimal_compile_mode("max-autotune")
            print(f"\n  Compiling with mode='{_compile_mode}'...")
            self.compiled_model = torch.compile(self.model, mode=_compile_mode)
        else:
            self.compiled_model = self.model
        
        # Create input using CPU random + to(device) to avoid CUDA RNG graph capture issues
        self.input_ids = torch.randint(
            0, self.VOCAB_SIZE,
            (self.BATCH_SIZE, self.SEQ_LEN),
        ).to(self.device)
        self._verify_output_buffer = torch.empty(
            (self.BATCH_SIZE, 1, min(8, self.VOCAB_SIZE)),
            device=self.device,
            dtype=torch.float32,
        )
        
        # Warmup
        print(f"\n  Warmup ({self.WARMUP + 2} iterations)...")
        for i in range(self.WARMUP + 2):
            with torch.inference_mode():
                _ = self.compiled_model(self.input_ids)
            if i == 0 and self.opts.use_compile:
                print("    First run (compile): done")
        torch.cuda.synchronize()
        print("  Ready")

    def benchmark_fn(self) -> None:
        with self._nvtx_range(f"level{self.LEVEL}"):
            with torch.inference_mode():
                logits = self.compiled_model(self.input_ids)
        self.output = logits
        if self.input_ids is None or self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.input_ids is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
            : self._verify_output_buffer.shape[2],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids.detach()},
            output=self._verify_output_buffer,
            batch_size=self.BATCH_SIZE,
            parameter_count=self.parameter_count,
            precision_flags={"bf16": True, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        del self.compiled_model
        del self.model
        self.compiled_model = None
        self.model = None
        self.input_ids = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=self.ITERATIONS,
            warmup=self.WARMUP,
            # This lab can hit intermittent subprocess exit -11 on some hosts
            # during process teardown; run in-process for stability.
            use_subprocess=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def validate_result(self) -> Optional[str]:
        return None if self.compiled_model is not None else "Model not initialized"

    def get_input_signature(self) -> dict:
        tf32_enabled = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
        return simple_signature(
            batch_size=self.BATCH_SIZE,
            dtype="int64",
            seq_len=self.SEQ_LEN,
            vocab_size=self.VOCAB_SIZE,
            hidden_size=self.HIDDEN_SIZE,
            intermediate_size=self.INTERMEDIATE_SIZE,
            num_layers=self.NUM_LAYERS,
            num_heads=self.NUM_HEADS,
            num_experts=self.NUM_EXPERTS,
            num_experts_per_tok=self.NUM_EXPERTS_PER_TOK,
            precision_flags=PrecisionFlags(bf16=True, tf32=tf32_enabled),
        ).to_dict()

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        metrics = {
            "level": float(self.LEVEL),
            "use_batched": float(self.opts.use_batched if self.opts else 0),
            "use_fused": float(self.opts.use_fused if self.opts else 0),
            "use_mem_efficient": float(self.opts.use_mem_efficient if self.opts else 0),
            "use_grouped": float(self.opts.use_grouped if self.opts else 0),
            "use_cuda_graphs": float(self.opts.use_cuda_graphs if self.opts else 0),
            "use_compile": float(self.opts.use_compile if self.opts else 0),
        }
        if self.model is not None and hasattr(self.model, "get_cuda_graph_metrics"):
            metrics.update(self.model.get_cuda_graph_metrics())
        return metrics
    
    


def run_level(level: int) -> None:
    """Run a specific level benchmark."""
    class LevelBenchmark(MoEJourneyBenchmark):
        LEVEL = level
    
    benchmark = LevelBenchmark()
    benchmark.setup()
    
    total_ms = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    for i in range(5):
        start_event.record(current_stream)
        benchmark.benchmark_fn()
        end_event.record(current_stream)
        end_event.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        total_ms += elapsed_ms
        total_tokens = benchmark.BATCH_SIZE * benchmark.SEQ_LEN
        tok_s = total_tokens / (elapsed_ms / 1000)
        print(f"  Run {i+1}: {elapsed_ms:.1f} ms ({tok_s:,.0f} tok/s)")
    
    avg = total_ms / 5
    print(f"\nMean: {avg:.1f} ms")
    benchmark.teardown()
    return avg


if __name__ == "__main__":
    import sys
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_level(level)
