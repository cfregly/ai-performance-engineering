"""Optimized disaggregated inference benchmark (multi-GPU torchrun pipeline).

Chapter 15: Disaggregated Inference (Optimized)

Optimizations:
- Overlap prefill and decode by pipelining KV transfers per request.
- Use non-blocking device transfers in the local verification path.
"""

from __future__ import annotations

import argparse

from ch15.baseline_disaggregated_inference_multigpu import (  # noqa: E402
    OPTIMIZED_PROFILE_NVTX_RANGE,
    DisaggConfig,
    _DisaggregatedInferenceMultiGPUBenchmark,
    _run_torchrun_worker,
)
from core.harness.benchmark_harness import BaseBenchmark  # noqa: E402


class OptimizedDisaggregatedInferenceMultiGPUBenchmark(_DisaggregatedInferenceMultiGPUBenchmark):
    """Pipelined prefill/decode overlap across multi-GPU ranks."""

    multi_gpu_required = True
    profile_nvtx_range = OPTIMIZED_PROFILE_NVTX_RANGE
    story_metadata = {
        "pair_role": "canonical",
        "chapter_alignment": "native",
        "chapter_native_exemplar": True,
        "timed_launch_mode": "torchrun_multi_gpu",
        "verification_mode": "local_multi_device_surrogate",
        "shared_harness_layout": "baseline_owned_shared_base",
        "shared_harness_owner": "ch15/baseline_disaggregated_inference_multigpu.py",
        "execution_pattern": "overlapped_prefill_decode_pipeline",
    }

    def __init__(self) -> None:
        super().__init__(
            overlap=True,
            label="optimized_disaggregated_inference_multigpu",
            profile_nvtx_range=OPTIMIZED_PROFILE_NVTX_RANGE,
            worker_module="ch15.optimized_disaggregated_inference_multigpu",
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedDisaggregatedInferenceMultiGPUBenchmark()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _run_torchrun_worker(
        DisaggConfig(),
        overlap=True,
        label="optimized_disaggregated_inference_multigpu",
        iters=int(args.iters),
        warmup=int(args.warmup),
        profile_nvtx_range=OPTIMIZED_PROFILE_NVTX_RANGE,
    )
