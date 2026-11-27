from pathlib import Path

import torch

from ch01.performance_common import restore_tf32_state, set_tf32_state

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text()


def test_optimized_performance_disables_and_restores_tf32() -> None:
    source = _read("ch01/optimized_performance.py")
    assert "self._tf32_state = set_tf32_state(False)" in source
    assert "restore_tf32_state(self._tf32_state)" in source


def test_optimized_performance_fp16_disables_and_restores_tf32() -> None:
    source = _read("ch01/optimized_performance_fp16.py")
    assert "self._tf32_state = set_tf32_state(False)" in source
    assert "restore_tf32_state(self._tf32_state)" in source


def test_strict_math_sets_and_restores_float32_matmul_precision() -> None:
    previous = set_tf32_state(True)
    try:
        state = set_tf32_state(False)
        try:
            assert torch.get_float32_matmul_precision() == "highest"
        finally:
            restore_tf32_state(state)
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        restore_tf32_state(previous)
