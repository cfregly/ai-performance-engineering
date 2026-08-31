from __future__ import annotations

import importlib

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required - benchmark_peak imports Transformer Engine and CUDA libraries",
)


def test_benchmark_peak_imports_with_serving_stack_runtime() -> None:
    module = importlib.import_module("core.benchmark.benchmark_peak")
    module._load_transformer_engine()
    assert module.TE_AVAILABLE is True
    assert isinstance(module._SERVING_STACK_LIB_DIRS, list)
