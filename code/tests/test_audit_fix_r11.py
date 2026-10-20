"""Regression tests for audit work unit r11.

W1-127: the symmetric-memory demo tests were named ``*_runs_on_gloo`` but their
two-GPU gate means ``ch16.symmetric_memory_inference.init_distributed`` always
picks NCCL. The workers now refuse any backend other than NCCL, so the renamed
``*_runs_on_nccl`` tests cannot pass on a backend their names do not state.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest
import torch
from torch import distributed as dist

from tests import test_symmetric_memory_inference as demo_tests


@pytest.mark.skipif(not dist.is_available(), reason="SKIPPED: torch.distributed is not available")
@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="SKIPPED: init_distributed picks NCCL on CUDA hosts, so the gloo rejection path cannot run here",
)
@pytest.mark.parametrize(
    ("worker", "workload_arg"),
    [
        (demo_tests._kv_worker, 16),
        (demo_tests._multi_model_worker, 2),
        (demo_tests._speculative_worker, 2),
    ],
    ids=["kv_cache", "multi_model", "speculative"],
)
def test_symmetric_memory_demo_workers_reject_gloo(worker, workload_arg) -> None:
    with mock.patch.dict(os.environ):
        try:
            with pytest.raises(RuntimeError, match="must run on NCCL, got backend 'gloo'"):
                worker(0, 1, workload_arg, demo_tests._find_free_port())
            assert not dist.is_initialized()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def test_symmetric_memory_demo_tests_name_the_backend_they_run() -> None:
    test_names = {name for name in vars(demo_tests) if name.startswith("test_")}

    assert not {name for name in test_names if "gloo" in name}
    assert {
        "test_kv_cache_demo_runs_on_nccl",
        "test_multi_model_demo_runs_on_nccl",
        "test_speculative_demo_runs_on_nccl",
    } <= test_names
