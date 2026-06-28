#!/usr/bin/env python3
"""Optimized: context-parallel attention with ring-exchange KV streaming."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from core.benchmark.gpu_requirements import require_min_gpus
from core.benchmark.verification import PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, LaunchVia, TorchrunLaunchSpec

from ch13.context_parallel_benchmark_common import (
    ContextParallelConfig,
    align_seq_len,
    build_layers,
    dtype_from_name,
    ring_attention,
    run_context_parallel,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized context-parallel attention (ring).")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="bf16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument(
        "--non-causal",
        action="store_false",
        dest="causal",
        help="Disable causal masking (default is causal).",
    )
    parser.set_defaults(causal=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    require_min_gpus(2, script_name="optimized_context_parallel_multigpu.py")
    config = ContextParallelConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dtype=dtype_from_name(args.dtype),
        causal=args.causal,
    )
    run_context_parallel(config=config, iters=args.iters, warmup=args.warmup, attention_fn=ring_attention)


class OptimizedContextParallelMultigpuBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Harness entry that launches this module via torchrun."""

    multi_gpu_required = True
    story_metadata = {
        "pair_role": "canonical",
        "chapter_alignment": "native",
        "chapter_native_exemplar": True,
        "timed_launch_mode": "torchrun_multi_gpu",
        "verification_mode": "single_process_surrogate",
        "optimization_mechanism": "ring_attention",
    }

    def __init__(self) -> None:
        super().__init__()
        self._cp_config = ContextParallelConfig()
        self._world_size = torch.cuda.device_count()
        require_min_gpus(2, script_name="optimized_context_parallel_multigpu.py")
        self._seq_len = align_seq_len(self._cp_config.seq_len, self._world_size)
        tokens = self._cp_config.batch_size * self._seq_len
        self.register_workload_metadata(
            requests_per_iteration=float(self._cp_config.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self._layers: Optional[torch.nn.ModuleList] = None
        self._input: Optional[torch.Tensor] = None
        self._output: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._layers = build_layers(self._cp_config, self.device)
        self._payload_parameter_count = sum(
            p.numel()
            for layer in self._layers
            for p in layer.parameters()
        )
        self._input = torch.randn(
            self._cp_config.batch_size,
            self._seq_len,
            self._cp_config.hidden_size,
            device=self.device,
            dtype=self._cp_config.dtype,
        )

    def benchmark_fn(self) -> None:
        if self._layers is None or self._input is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        x = self._input
        seq_shard = x.shape[1]
        for layer in self._layers:
            q, k, v = layer.split_qkv(x)
            attn_out = ring_attention(
                q,
                k,
                v,
                rank=0,
                world_size=1,
                process_group=None,
                causal=self._cp_config.causal,
                seq_shard=seq_shard,
                scale=layer.scale,
            )
            x = layer.proj(layer.merge_heads(attn_out))
        self._output = x

    def capture_verification_payload(self) -> None:
        if self._output is None or self._input is None or self._layers is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input": self._input},
            output=self._output,
            batch_size=self._cp_config.batch_size,
            parameter_count=self._payload_parameter_count,
            precision_flags=PrecisionFlags(
                fp16=self._cp_config.dtype == torch.float16,
                bf16=self._cp_config.dtype == torch.bfloat16,
                tf32=False,
            ),
            output_tolerance=(0.5, 5.0),
            signature_overrides={"world_size": self._world_size},
        )

    def _prepare_verification_payload(self) -> None:
        if hasattr(self, "_subprocess_verify_output"):
            return
        self.setup()
        try:
            self.benchmark_fn()
            self.capture_verification_payload()
            self._subprocess_verify_output = self.get_verify_output()
            self._subprocess_output_tolerance = self.get_output_tolerance()
            self._subprocess_input_signature = self.get_input_signature()
        finally:
            self.teardown()

    def validate_result(self) -> Optional[str]:
        if self._output is None:
            return "No output captured"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=self._world_size,
            iterations=5,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=900,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        self._prepare_verification_payload()
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve(),
            script_args=[],
            multi_gpu_required=True,
            name="optimized_context_parallel_multigpu",
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedContextParallelMultigpuBenchmark()
