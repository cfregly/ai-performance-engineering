"""Baseline ZeRO-2 comparison: standard DDP all-reduce without sharding."""

from __future__ import annotations

from pathlib import Path

from labs.train_distributed.training_utils.zero2_torchrun_benchmark import Zero2TorchrunBenchmark


def parse_args():
    from labs.train_distributed.zero2_common import parse_args as common_args
    return common_args()


def _build_model(hidden_size: int, device):
    from labs.train_distributed.zero2_common import build_model
    return build_model(hidden_size, device)


def main():
    from labs.train_distributed.zero2_common import run_training
    run_training(parse_args(), optimized=False, multi_gpu=True)


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return Zero2TorchrunBenchmark(
        mode="baseline",
        variant="multigpu",
        script_path=Path(__file__).parent / "zero2.py",
        base_args=[
            "--mode",
            "baseline",
            "--variant",
            "multigpu",
            "--batch-size",
            "16",
            "--hidden-size",
            "10000",
            "--grad-accum",
            "1",
            "--extra-grad-mb",
            "12288",
        ],
        config_arg_map={"iterations": "--steps"},
        multi_gpu_required=True,
        target_label="labs/train_distributed:zero2_multigpu",
        default_nproc_per_node=None,
        name="baseline_zero2_multigpu",
    )
