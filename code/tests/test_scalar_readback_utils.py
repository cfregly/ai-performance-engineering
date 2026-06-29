from __future__ import annotations

import re
from pathlib import Path

import torch

from core.benchmark.utils import scalar_tensor_to_float


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_scalar_tensor_to_float_materializes_scalar_without_item() -> None:
    value = torch.tensor(1.25)

    assert scalar_tensor_to_float(value) == 1.25


def test_validation_paths_use_shared_scalar_readback_helper() -> None:
    files = (
        "ch08/ai_optimization_benchmark_base.py",
        "ch08/hbm_benchmark_base.py",
        "ch08/loop_unrolling_benchmark_base.py",
        "ch08/nccl_benchmark_base.py",
        "ch08/threshold_benchmark_base.py",
        "ch08/tiling_benchmark_base.py",
        "labs/blackwell_gemm_optimizations/blackwell_grouped_gemm_common.py",
        "labs/blackwell_matmul/blackwell_benchmarks.py",
        "labs/flashattention4/flashattention4_common.py",
        "labs/flashattention4/tflops_microbench.py",
        "labs/fullstack_cluster/capstone_benchmarks.py",
        "labs/fullstack_cluster/run_lab_fullstack_cluster.py",
        "labs/nccl_nixl_nvshmem/run_lab_nccl_nixl_nvshmem.py",
        "labs/recsys_sequence_ranking/compare_sequence_ranking.py",
        "labs/software_pipelining/software_pipelining_common.py",
        "labs/training_hotpath/compare.py",
    )

    for relative_path in files:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "scalar_tensor_to_float" in source
        assert ".abs().max().item()" not in source
        assert "diff.max().item()" not in source
        assert re.search(r"torch\.max\(torch\.abs\([^\n]+\)\)\.item\(\)", source) is None
