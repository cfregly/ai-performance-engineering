from __future__ import annotations

from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
NVSHMEM_EDUCATIONAL_SOURCES = (
    "nvshmem_multigpu_examples.cu",
    "nvshmem_advanced_multigpu.cu",
    "nvshmem_multinode_example.cu",
    "nvshmem_pipeline_patterns.cu",
    "nvshmem_tensor_parallel.cu",
)


@pytest.mark.parametrize("source_name", NVSHMEM_EDUCATIONAL_SOURCES)
def test_nvtx_helper_is_available_without_nvshmem(source_name: str) -> None:
    source = (CODE_ROOT / "ch04" / source_name).read_text(encoding="utf-8")

    nvtx_include = '#include "../core/common/nvtx_utils.cuh"'
    assert source.count(nvtx_include) == 1
    assert source.index(nvtx_include) < source.index("#ifdef USE_NVSHMEM")
    assert "NVTX_RANGE(" in source
