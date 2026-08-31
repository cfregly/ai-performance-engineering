"""ZeRO comparison using a supported RS/AG hook and explicit sharded AdamW.

The single-rank path has no sharding benefit. Multi-rank RS/AG restores full DDP
gradients; optimizer computation does not overlap backward. This is not persistent
gradient sharding, and no previous overlap speedup is qualified by this repair.
"""

from __future__ import annotations

from pathlib import Path

from labs.train_distributed.training_utils.zero2_torchrun_benchmark import Zero2TorchrunBenchmark


def parse_args():
    from labs.train_distributed.zero2_common import parse_args as common_args
    return common_args()


def _build_model(hidden_size: int, device):
    from labs.train_distributed.zero2_common import build_model
    return build_model(hidden_size, device)


def _build_training_components(model, learning_rate, device_ids=None):
    from labs.train_distributed.zero2_common import build_training_components
    return build_training_components(model, learning_rate, optimized=True, device_ids=device_ids)


def main():
    from labs.train_distributed.zero2_common import run_training
    run_training(parse_args(), optimized=True, multi_gpu=False)


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return Zero2TorchrunBenchmark(
        mode="optimized",
        variant="single",
        script_path=Path(__file__).parent / "zero2.py",
        base_args=[
            "--mode",
            "optimized",
            "--variant",
            "single",
            "--batch-size",
            "16",
            "--hidden-size",
            "10000",
            "--grad-accum",
            "1",
        ],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:zero2",
        default_nproc_per_node=1,
        multi_gpu_required=False,
        name="optimized_zero2",
    )
