#!/usr/bin/env python3
"""Optimized TP+SP benchmark that keeps activations sequence-sharded across layers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from core.benchmark.gpu_requirements import require_min_gpus
from core.benchmark.verification import PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, LaunchVia, TorchrunLaunchSpec

from ch13.sequence_parallel_benchmark_common import (
    SequenceParallelConfig,
    align_seq_len,
    build_layers,
    dtype_from_name,
    run_sequence_parallel,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized TP+SP hybrid benchmark.")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ffn-hidden-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="bf16", choices=("fp16", "bf16", "fp32"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    require_min_gpus(2, script_name="optimized_sequence_parallel_multigpu.py")
    config = SequenceParallelConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_layers=args.num_layers,
        dtype=dtype_from_name(args.dtype),
    )
    run_sequence_parallel(config=config, iters=args.iters, warmup=args.warmup, sequence_parallel=True)


class OptimizedSequenceParallelMultigpuBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Harness entry that launches the TP+SP hybrid path via torchrun."""

    multi_gpu_required = True
    story_metadata = {
        "pair_role": "canonical",
        "chapter_alignment": "native",
        "chapter_native_exemplar": True,
        "timed_launch_mode": "torchrun_multi_gpu",
        "verification_mode": "single_process_surrogate",
        "optimization_mechanism": "keep_tp_activations_sequence_sharded_between_layers",
    }

    def __init__(self) -> None:
        super().__init__()
        self._sp_config = SequenceParallelConfig()
        self._world_size = torch.cuda.device_count()
        require_min_gpus(2, script_name="optimized_sequence_parallel_multigpu.py")
        self._seq_len = align_seq_len(self._sp_config.seq_len, self._world_size)
        tokens = self._sp_config.batch_size * self._seq_len
        self.register_workload_metadata(
            requests_per_iteration=float(self._sp_config.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self._up_proj = None
        self._down_proj = None
        self._norms = None
        self._input = None
        self._output = None
        self._layer_triples = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._up_proj, self._down_proj, self._norms = build_layers(self._sp_config, self._world_size, self.device)
        self._layer_triples = tuple(zip(self._up_proj, self._down_proj, self._norms, strict=True))
        self._input = torch.randn(
            self._sp_config.batch_size,
            self._seq_len // self._world_size,
            self._sp_config.hidden_size,
            device=self.device,
            dtype=self._sp_config.dtype,
        )

    def benchmark_fn(self) -> None:
        if (
            self._input is None
            or self._up_proj is None
            or self._down_proj is None
            or self._norms is None
            or self._layer_triples is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")
        x = self._input
        for up_proj, down_proj, norm in self._layer_triples:
            hidden_local = torch.nn.functional.gelu(up_proj(x), approximate="tanh")
            out_partial = down_proj(hidden_local)
            x = norm(out_partial)
        self._output = x

    def capture_verification_payload(self) -> None:
        if self._output is None or self._input is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        param_count = sum(
            p.numel()
            for module in (self._up_proj, self._down_proj, self._norms)
            for p in module.parameters()
        )
        self._set_verification_payload(
            inputs={"input": self._input},
            output=self._output,
            batch_size=self._sp_config.batch_size,
            parameter_count=int(param_count),
            precision_flags=PrecisionFlags(
                fp16=self._sp_config.dtype == torch.float16,
                bf16=self._sp_config.dtype == torch.bfloat16,
                tf32=False,
            ),
            output_tolerance=(0.0, 0.0),
            signature_overrides={"world_size": self._world_size, "collective_type": "all_reduce"},
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
            name="optimized_sequence_parallel_multigpu",
            config_arg_map={"iterations": "--iters", "warmup": "--warmup"},
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedSequenceParallelMultigpuBenchmark()
