#!/usr/bin/env python3
"""Baseline: expert-parallel all-to-all using reusable list buffers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.gpu_requirements import require_min_gpus
from core.benchmark.verification import PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, LaunchVia, TorchrunLaunchSpec

from ch13.expert_parallel_common import ExpertParallelConfig, dtype_from_name, run_expert_parallel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline expert-parallel all-to-all (list).")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bf16", choices=("fp16", "bf16", "fp32"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    require_min_gpus(2, script_name="baseline_expert_parallel_multigpu.py")
    config = ExpertParallelConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        dtype=dtype_from_name(args.dtype),
    )
    run_expert_parallel(config=config, iters=args.iters, warmup=args.warmup, impl="list")


class BaselineExpertParallelMultigpuBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Harness entry that launches this module via torchrun."""

    multi_gpu_required = True
    story_metadata = {
        "pair_role": "canonical",
        "chapter_alignment": "native",
        "chapter_native_exemplar": True,
        "timed_launch_mode": "torchrun_multi_gpu",
        "verification_mode": "single_process_surrogate",
        "optimization_mechanism": "reused_list_all_to_all_buffers",
    }

    def __init__(self) -> None:
        super().__init__()
        self._ep_config = ExpertParallelConfig()
        self._world_size = torch.cuda.device_count()
        require_min_gpus(2, script_name="baseline_expert_parallel_multigpu.py")
        tokens = self._ep_config.batch_size * self._ep_config.seq_len * self._world_size
        self.register_workload_metadata(
            requests_per_iteration=float(self._ep_config.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self._input: Optional[torch.Tensor] = None
        self._output: Optional[torch.Tensor] = None
        self._expert_proj: Optional[nn.Linear] = None
        self._payload_parameter_count = 0

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        tokens_per_rank = self._ep_config.batch_size * self._ep_config.seq_len
        self._input = torch.randn(
            tokens_per_rank,
            self._ep_config.hidden_size,
            device=self.device,
            dtype=self._ep_config.dtype,
        )
        self._expert_proj = nn.Linear(
            self._ep_config.hidden_size,
            self._ep_config.hidden_size,
            bias=False,
            dtype=self._ep_config.dtype,
        ).to(self.device)
        self._payload_parameter_count = sum(p.numel() for p in self._expert_proj.parameters())

    def benchmark_fn(self) -> None:
        if self._input is None or self._expert_proj is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        self._output = self._expert_proj(self._input)

    def capture_verification_payload(self) -> None:
        if self._output is None or self._input is None or self._expert_proj is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"tokens": self._input},
            output=self._output,
            batch_size=self._ep_config.batch_size,
            parameter_count=self._payload_parameter_count,
            precision_flags=PrecisionFlags(
                fp16=self._ep_config.dtype == torch.float16,
                bf16=self._ep_config.dtype == torch.bfloat16,
                tf32=False,
            ),
            output_tolerance=(0.2, 2.0),
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
            name="baseline_expert_parallel_multigpu",
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )


def get_benchmark() -> BaseBenchmark:
    return BaselineExpertParallelMultigpuBenchmark()
