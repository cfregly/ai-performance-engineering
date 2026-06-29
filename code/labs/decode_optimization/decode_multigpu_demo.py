"""Multi-GPU decode demo to stress NVLink-C2C throughput on B200 nodes.

This script is a multi-GPU demo/tool and is intentionally *not* a harness-comparable
baseline/optimized benchmark pair.
"""

from __future__ import annotations

import argparse

from core.common.device_utils import resolve_local_rank
import time
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig


def _resolve_world_size() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for multi-GPU decode demo")
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("Multi-GPU decode demo requires >=2 GPUs")
    return world_size


class MultiGPUDecodeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Torchrun-only entry that launches this script across available GPUs."""
    multi_gpu_required = True

    def __init__(self) -> None:
        super().__init__()
        self.metrics = torch.zeros((1, 1), dtype=torch.float32)
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)
        self.verification_not_applicable_reason = "SKIPPED: requires >=2 GPUs (torchrun only)"

    def benchmark_fn(self) -> None:  # pragma: no cover - torchrun path only
        raise RuntimeError("SKIPPED: requires >=2 GPUs (torchrun path only)")

    def get_config(self) -> BenchmarkConfig:
        world_size = _resolve_world_size()
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=world_size,
            iterations=1,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=600,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "decode_multigpu.batch_size": float(getattr(self, 'batch_size', 0)),
            "decode_multigpu.seq_len": float(getattr(self, 'seq_len', 0)),
            "decode_multigpu.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
        }

    def get_torchrun_spec(self, config: BenchmarkConfig | None = None) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(
            module_name="labs.decode_optimization.decode_multigpu_demo",
            script_args=[],
            env={
                "NCCL_DEBUG": "WARN",
                "NCCL_NET_GDR_LEVEL": "PHB",
                "NCCL_P2P_LEVEL": "NVL",
                "NCCL_P2P_DISABLE": "0",
            },
            parse_rank0_only=True,
            multi_gpu_required=True,
            name="optimized_decode_multigpu",
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )


def _run_worker(iters: int, warmup: int) -> None:
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = resolve_local_rank()
    torch.cuda.set_device(local_rank)

    cfg = DecodeConfig(
        batch_size=8,
        prompt_tokens=1024,
        decode_tokens=256,
        hidden_size=2048,
        use_fp8=True,
        use_pinned_host=True,
        use_copy_stream=True,
        use_compute_stream=True,
        use_cuda_graphs=True,
        graph_full_iteration=True,
        use_torch_compile=True,
        label="optimized_decode_multigpu",
    )

    bench = DecodeBenchmark(cfg)
    bench.setup()

    # Warmup
    for _ in range(max(warmup, 0)):
        bench.benchmark_fn()
    torch.cuda.synchronize()

    # Timed iterations
    sample_count = max(iters, 1)
    start = time.perf_counter()
    for _ in range(sample_count):
        bench.benchmark_fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_iter = cfg.batch_size * (cfg.prompt_tokens + cfg.decode_tokens)
    local_tokens_per_s = tokens_per_iter * (sample_count / elapsed)
    global_tokens_per_s = local_tokens_per_s * dist.get_world_size()

    if rank == 0:
        print(f"rank0 tokens/s: {global_tokens_per_s:.2f} tokens/s (local {local_tokens_per_s:.2f})")
        print(f"rank0 time_per_iter_ms: {(elapsed / sample_count) * 1000.0:.3f}")

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    _run_worker(args.iters, args.warmup)


def get_benchmark() -> BaseBenchmark:
    return MultiGPUDecodeBenchmark()
