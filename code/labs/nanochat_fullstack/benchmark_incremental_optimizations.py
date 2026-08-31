#!/usr/bin/env python3
"""
Incremental Optimization Benchmark for NanoChat

Measures the performance impact of enabling each optimization one by one.
Tests inference (prefill + decode). Engine features execute through Engine;
unavailable requested backends fail instead of producing mislabeled timings.

Usage:
    python -m labs.nanochat_fullstack.benchmark_incremental_optimizations
"""

import argparse
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import torch

from labs.nanochat_fullstack.nanochat.gpt import GPT, GPTConfig
from labs.nanochat_fullstack.nanochat.engine import Engine, KVCache


@dataclass
class BenchmarkConfig:
    name: str
    flags: Dict[str, Any]
    description: str
    category: str


@dataclass
class BenchmarkResult:
    name: str
    prefill_tok_s: float
    decode_tok_s: float
    total_time: float
    category: str
    description: str
    decode_execution: str = "not_run"
    decode_attention: str = "not_run"
    improvement_vs_baseline: float = 0.0
    cumulative_improvement: float = 0.0


class IncrementalBenchmark:
    def __init__(self, device=None, warmup=5, iterations=5):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.warmup = warmup
        self.iterations = iterations
        if warmup < 1 or iterations < 1:
            raise ValueError("warmup and iterations must both be positive")
        self.last_decode_execution = "not_run"
        self.last_decode_attention = "not_run"
        self.batch_size = 4
        self.prompt_len = 256
        self.decode_len = 64
        self.vocab_size = 10000
        
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Prompt length: {self.prompt_len}")
        print(f"Decode length: {self.decode_len}")
        print(f"Warmup iterations: {self.warmup}")
        print(f"Timed iterations: {self.iterations}")
        print()
    
    def create_model(self, config_overrides: Dict[str, Any]) -> GPT:
        """Create a small GPT model for benchmarking."""
        # Identical model weights and token streams across feature arms.
        torch.manual_seed(718)
        base_config = {
            'sequence_len': 1024,
            'vocab_size': self.vocab_size,
            'n_layer': 4,  # Small model for fast benchmarking
            'n_head': 8,
            'n_kv_head': 8,
            'n_embd': 512,
        }
        base_config.update(config_overrides)
        
        config = GPTConfig(**base_config)
        
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=self.device)
        model.init_weights()
        # init_weights intentionally zeros output projections for training;
        # nonzero projections are necessary for meaningful inference parity.
        model.apply(model._init_weights)
        
        if self.device.type == "cuda":
            model = model.to(dtype=torch.bfloat16)
        
        model.eval()
        return model

    def _time_region_seconds(self, fn: Callable[[], None]) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - t0
    
    def benchmark_inference(self, config_overrides: Dict[str, Any]) -> tuple:
        """Benchmark prefill + decode performance."""
        model = self.create_model(config_overrides)
        cfg = model.config
        engine = Engine(model, tokenizer=None, enable_batch_decode=False)
        if cfg.use_cuda_graphs or cfg.enable_persistent_decode:
            if self.device.type != "cuda":
                raise RuntimeError("Requested CUDA graph/side-stream benchmark requires a CUDA device")
        if cfg.use_flash3 and any(not b.attn.use_flash3 or b.attn.flash3_fn is None for b in model.transformer.h):
            raise RuntimeError("Requested FA3 benchmark backend is unavailable")
        if cfg.use_cta_clustering and any(
            not b.attn._flash3_accepts_clusters or self.prompt_len < b.attn.cta_cluster_seq_threshold
            for b in model.transformer.h
        ):
            raise RuntimeError("Requested CTA clustering benchmark requires an active cluster-capable FA3 backend")
        
        # Prepare inputs
        prompt = torch.randint(0, self.vocab_size, (self.batch_size, self.prompt_len), 
                              device=self.device, dtype=torch.long)
        decode_tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.decode_len),
                                     device=self.device, dtype=torch.long)
        decode_token_steps = tuple(
            decode_tokens[:, t:t + 1]
            for t in range(self.decode_len)
        )
        
        # Setup KV cache
        head_dim = cfg.n_embd // cfg.n_head
        kv_cache = KVCache(
            batch_size=self.batch_size,
            num_heads=cfg.n_kv_head,
            seq_len=self.prompt_len + self.decode_len + 16,
            head_dim=head_dim,
            num_layers=cfg.n_layer,
            block_size=getattr(cfg, "kv_block_size", None),
            page_size=getattr(cfg, "kv_page_size", None),
        )
        
        # Warmup
        for _ in range(self.warmup):
            kv_cache.reset()
            with torch.inference_mode():
                _ = model(prompt, kv_cache=kv_cache)
                for step_ids in decode_token_steps[: min(8, self.decode_len)]:
                    _ = engine._execute_decode(step_ids, kv_cache)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark prefill
        prefill_time_total = 0.0

        def _run_prefill() -> None:
            with torch.inference_mode():
                _ = model(prompt, kv_cache=kv_cache)

        for _ in range(self.iterations):
            kv_cache.reset()
            prefill_time_total += self._time_region_seconds(_run_prefill)
        
        prefill_time = prefill_time_total / self.iterations
        prefill_tok_s = (self.batch_size * self.prompt_len) / prefill_time
        
        # Benchmark decode
        decode_time_total = 0.0

        def _run_decode() -> None:
            with torch.inference_mode():
                for step_ids in decode_token_steps:
                    _ = engine._execute_decode(step_ids, kv_cache)

        for _ in range(self.iterations):
            kv_cache.reset()
            # Prefill first
            with torch.inference_mode():
                _ = model(prompt, kv_cache=kv_cache)

                # Capture/warmup is setup, outside steady-state replay timing.
                # Capture does not consume a logical token or alter the prefix.
                if cfg.use_cuda_graphs:
                    engine._capture_decode_graph(decode_token_steps[0], kv_cache)

            decode_time_total += self._time_region_seconds(_run_decode)
        
        decode_time = decode_time_total / self.iterations
        decode_tok_s = (self.batch_size * self.decode_len) / decode_time
        self.last_decode_execution = engine.decode_execution_mode
        self.last_decode_attention = "sdpa_full_capacity_mask" if cfg.use_cuda_graphs else "model_backend_preferences"
        expected_execution = "cuda_graph" if cfg.use_cuda_graphs else ("side_stream" if cfg.enable_persistent_decode else "eager")
        if self.last_decode_execution != expected_execution:
            raise RuntimeError(f"Requested {expected_execution}, observed {self.last_decode_execution}")
        
        total_time = prefill_time + decode_time
        
        # Cleanup
        model = None
        kv_cache = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return prefill_tok_s, decode_tok_s, total_time
    
    def run_incremental_benchmark(self) -> List[BenchmarkResult]:
        """Run benchmark with optimizations enabled incrementally."""
        
        # Define optimization configurations in order of enablement
        configs = [
            BenchmarkConfig(
                name="Baseline",
                flags={
                    'use_flash_sdp': False,
                    'use_flash3': False,
                    'use_padded_attention': False,
                    'use_cta_clustering': False,
                    'kv_block_size': None,
                    'kv_page_size': None,
                    'enable_persistent_decode': False,
                    'use_cuda_graphs': False,
                    'use_te_weight_only': False,
                    'use_fp32_logits': True,
                },
                description="All optimizations OFF",
                category="Baseline"
            ),
            BenchmarkConfig(
                name="+ use_flash_sdp",
                flags={
                    'use_flash_sdp': True,
                    'use_flash3': False,
                    'use_padded_attention': False,
                    'use_cta_clustering': False,
                    'kv_block_size': None,
                    'kv_page_size': None,
                    'enable_persistent_decode': False,
                    'use_cuda_graphs': False,
                    'use_te_weight_only': False,
                    'use_fp32_logits': True,
                },
                description="Enable Flash/efficient SDP kernels",
                category="Attention"
            ),
            BenchmarkConfig(
                name="+ use_flash3",
                flags={
                    'use_flash_sdp': True,
                    'use_flash3': True,
                    'use_padded_attention': False,
                    'use_cta_clustering': False,
                    'kv_block_size': None,
                    'kv_page_size': None,
                    'enable_persistent_decode': False,
                    'use_cuda_graphs': False,
                    'use_te_weight_only': False,
                    'use_fp32_logits': True,
                },
                description="Enable FlashAttention-3 varlen kernels",
                category="Attention"
            ),
            BenchmarkConfig(
                name="+ kv_block_size",
                flags={
                    'use_flash_sdp': True,
                    'use_flash3': True,
                    'use_padded_attention': False,
                    'use_cta_clustering': False,
                    'kv_block_size': 32,
                    'kv_page_size': 1024,
                    'enable_persistent_decode': False,
                    'use_cuda_graphs': False,
                    'use_te_weight_only': False,
                    'use_fp32_logits': True,
                },
                description="Enable KV cache blocking (TMA hints)",
                category="KV Cache"
            ),
            BenchmarkConfig(
                name="+ use_cta_clustering",
                flags={
                    'use_flash_sdp': True,
                    'use_flash3': True,
                    'use_padded_attention': False,
                    'use_cta_clustering': True,
                    'cta_cluster_size': 2,
                    'cta_cluster_seq_threshold': 128,  # Low threshold for testing
                    'kv_block_size': 32,
                    'kv_page_size': 1024,
                    'enable_persistent_decode': False,
                    'use_cuda_graphs': False,
                    'use_te_weight_only': False,
                    'use_fp32_logits': True,
                },
                description="Enable CTA clustering for attention",
                category="Attention"
            ),
            BenchmarkConfig(
                name="+ enable_persistent_decode",
                flags={
                    'use_flash_sdp': True,
                    'use_flash3': True,
                    'use_padded_attention': False,
                    'use_cta_clustering': True,
                    'cta_cluster_size': 2,
                    'cta_cluster_seq_threshold': 128,
                    'kv_block_size': 32,
                    'kv_page_size': 1024,
                    'enable_persistent_decode': True,
                    'use_cuda_graphs': False,
                    'use_te_weight_only': False,
                    'use_fp32_logits': True,
                },
                description="Enable persistent decode (streams + buffers)",
                category="Inference"
            ),
            BenchmarkConfig(
                name="+ use_cuda_graphs",
                flags={
                    'use_flash_sdp': True,
                    'use_flash3': True,
                    'use_padded_attention': False,
                    'use_cta_clustering': True,
                    'cta_cluster_size': 2,
                    'cta_cluster_seq_threshold': 128,
                    'kv_block_size': 32,
                    'kv_page_size': 1024,
                    'enable_persistent_decode': True,
                    'use_cuda_graphs': True,
                    'use_te_weight_only': False,
                    'use_fp32_logits': True,
                },
                description="Enable CUDA graph capture for decode",
                category="Inference"
            ),
            BenchmarkConfig(
                name="+ use_fp32_logits=False",
                flags={
                    'use_flash_sdp': True,
                    'use_flash3': True,
                    'use_padded_attention': False,
                    'use_cta_clustering': True,
                    'cta_cluster_size': 2,
                    'cta_cluster_seq_threshold': 128,
                    'kv_block_size': 32,
                    'kv_page_size': 1024,
                    'enable_persistent_decode': True,
                    'use_cuda_graphs': True,
                    'use_te_weight_only': False,
                    'use_fp32_logits': False,  # BF16 logits
                },
                description="Use BF16 logits (faster, slightly less precise)",
                category="Precision"
            ),
        ]
        
        results = []
        baseline_total = None
        
        print("=" * 80)
        print("INCREMENTAL OPTIMIZATION BENCHMARK")
        print("=" * 80)
        print()
        
        for i, config in enumerate(configs):
            print(f"[{i+1}/{len(configs)}] Testing: {config.name}")
            print(f"    {config.description}")
            
            try:
                prefill_tok_s, decode_tok_s, total_time = self.benchmark_inference(config.flags)
                
                # Calculate improvements
                if baseline_total is None:
                    baseline_total = total_time
                    improvement_vs_baseline = 0.0
                    cumulative_improvement = 0.0
                else:
                    improvement_vs_baseline = ((baseline_total - total_time) / baseline_total) * 100
                    cumulative_improvement = improvement_vs_baseline
                
                result = BenchmarkResult(
                    name=config.name,
                    prefill_tok_s=prefill_tok_s,
                    decode_tok_s=decode_tok_s,
                    total_time=total_time,
                    category=config.category,
                    description=config.description,
                    decode_execution=self.last_decode_execution,
                    decode_attention=self.last_decode_attention,
                    improvement_vs_baseline=improvement_vs_baseline,
                    cumulative_improvement=cumulative_improvement,
                )
                results.append(result)
                
                print(f"    Prefill: {prefill_tok_s:>8.1f} tok/s")
                print(f"    Decode:  {decode_tok_s:>8.1f} tok/s")
                print(f"    Execution: {self.last_decode_execution}; synchronized wall timing includes host submission")
                print(f"    Decode attention: {self.last_decode_attention}")
                print(f"    Total:   {total_time:>8.3f}s")
                if baseline_total is not None and i > 0:
                    print(f"    Speedup: {cumulative_improvement:>7.1f}% vs baseline")
                print()
                
            except Exception as e:
                raise RuntimeError(f"Benchmark arm {config.name!r} failed; no timing result recorded") from e
        
        return results


def print_summary_table(results: List[BenchmarkResult]):
    """Print a formatted summary table of results."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print()
    print(f"{'Optimization':<30} {'Category':<15} {'Prefill':<12} {'Decode':<12} {'Total':<10} {'Speedup':>10}")
    print(f"{'':30} {'':15} {'(tok/s)':<12} {'(tok/s)':<12} {'(sec)':<10} {'vs Base':>10}")
    print("-" * 100)
    
    for result in results:
        speedup_str = f"{result.cumulative_improvement:>+6.1f}%" if result.cumulative_improvement != 0 else "baseline"
        print(f"{result.name:<30} {result.category:<15} {result.prefill_tok_s:>10.1f}  "
              f"{result.decode_tok_s:>10.1f}  {result.total_time:>8.3f}  {speedup_str:>10}")
    
    print("-" * 100)
    
    if len(results) > 1:
        baseline = results[0]
        final = results[-1]
        total_speedup = ((baseline.total_time - final.total_time) / baseline.total_time) * 100
        prefill_speedup = ((final.prefill_tok_s - baseline.prefill_tok_s) / baseline.prefill_tok_s) * 100
        decode_speedup = ((final.decode_tok_s - baseline.decode_tok_s) / baseline.decode_tok_s) * 100
        
        print(f"\n{'TOTAL IMPROVEMENT:':<30} Prefill: +{prefill_speedup:.1f}%  |  "
              f"Decode: +{decode_speedup:.1f}%  |  Overall: {total_speedup:+.1f}%")
    
    print("=" * 100)
    print()


def print_category_breakdown(results: List[BenchmarkResult]):
    """Print performance impact by optimization category."""
    if len(results) < 2:
        return
    
    print("\n" + "=" * 80)
    print("CATEGORY BREAKDOWN")
    print("=" * 80)
    print()
    
    categories = {}
    for i, result in enumerate(results[1:], 1):  # Skip baseline
        cat = result.category
        if cat not in categories:
            categories[cat] = []
        
        prev_result = results[i-1]
        improvement = ((prev_result.total_time - result.total_time) / prev_result.total_time) * 100
        categories[cat].append((result.name, improvement))
    
    for category, improvements in categories.items():
        total_improvement = sum(imp for _, imp in improvements)
        print(f"{category}:")
        for name, improvement in improvements:
            print(f"  {name:<35} {improvement:>+7.2f}%")
        print(f"  {'Category Total:':<35} {total_improvement:>+7.2f}%")
        print()
    
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description="Incremental optimization benchmark for NanoChat")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=5, help="Number of timed iterations")
    parser.add_argument("--output", type=str, help="Output file for results (markdown format)")
    args = parser.parse_args()
    
    benchmark = IncrementalBenchmark(warmup=args.warmup, iterations=args.iterations)
    results = benchmark.run_incremental_benchmark()
    
    print_summary_table(results)
    print_category_breakdown(results)
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("# NanoChat Incremental Optimization Benchmark Results\n\n")
            f.write(f"**Device**: {benchmark.device}\n")
            f.write(f"**Batch Size**: {benchmark.batch_size}\n")
            f.write(f"**Prompt Length**: {benchmark.prompt_len}\n")
            f.write(f"**Decode Length**: {benchmark.decode_len}\n")
            f.write(f"**Iterations**: {benchmark.iterations}\n\n")
            
            f.write("## Summary Table\n\n")
            f.write("Timing: synchronized wall latency including host submission; graph capture is excluded from steady-state decode.\n\n")
            f.write("| Optimization | Category | Execution | Decode attention | Prefill (tok/s) | Decode (tok/s) | Total (sec) | Speedup vs Baseline |\n")
            f.write("|--------------|----------|-----------|------------------|-----------------|----------------|-------------|---------------------|\n")
            
            for result in results:
                speedup = f"{result.cumulative_improvement:+.1f}%" if result.cumulative_improvement != 0 else "baseline"
                f.write(f"| {result.name} | {result.category} | {result.decode_execution} | {result.decode_attention} | {result.prefill_tok_s:.1f} | "
                       f"{result.decode_tok_s:.1f} | {result.total_time:.3f} | {speedup} |\n")
            
            if len(results) > 1:
                baseline = results[0]
                final = results[-1]
                total_speedup = ((baseline.total_time - final.total_time) / baseline.total_time) * 100
                f.write(f"\n**Total Improvement**: {total_speedup:+.1f}%\n")
        
        print(f"\n✅ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
