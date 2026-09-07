from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ch02.baseline_memory_transfer_multigpu import (
    BaselineMemoryTransferMultigpuBenchmark,
)
from ch02.memory_transfer_multigpu_common import (
    MemoryTransferMultiGPUBenchmark,
    load_complete_transfer_output,
    validate_transfer_pattern,
)
from ch02.optimized_memory_transfer_multigpu import (
    OptimizedMemoryTransferMultigpuBenchmark,
)

CODE_ROOT = Path(__file__).resolve().parents[1]


def _write_pattern(path: Path, elements: int) -> torch.Tensor:
    values = torch.arange(elements, dtype=torch.float32)
    values.remainder_(4093).sub_(2046).mul_(1.0 / 256.0)
    path.write_bytes(values.numpy().tobytes())
    return values


def test_complete_output_loader_rejects_corrupt_final_destination_element(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "destination.f32"
    expected = _write_pattern(output_path, 4100)
    output = load_complete_transfer_output(output_path, expected_elements=4100)

    assert torch.equal(output, expected)
    validate_transfer_pattern(output)

    corrupted = expected.clone()
    corrupted[-1] += 1.0
    output_path.write_bytes(corrupted.numpy().tobytes())
    output = load_complete_transfer_output(output_path, expected_elements=4100)

    with pytest.raises(RuntimeError, match=r"element 4099"):
        validate_transfer_pattern(output)


def test_complete_output_loader_rejects_truncated_destination(tmp_path: Path) -> None:
    output_path = tmp_path / "destination.f32"
    _write_pattern(output_path, 36)

    with pytest.raises(RuntimeError, match=r"144 bytes; expected 148"):
        load_complete_transfer_output(output_path, expected_elements=37)


@pytest.mark.parametrize(
    ("filename", "benchmark_type"),
    (
        (
            "baseline_memory_transfer_multigpu.cu",
            BaselineMemoryTransferMultigpuBenchmark,
        ),
        (
            "optimized_memory_transfer_multigpu.cu",
            OptimizedMemoryTransferMultigpuBenchmark,
        ),
    ),
)
def test_transfer_pair_validates_full_timed_destination_after_timing(
    filename: str,
    benchmark_type: type,
) -> None:
    source = (CODE_ROOT / "ch02" / filename).read_text(encoding="utf-8")

    assert issubclass(benchmark_type, MemoryTransferMultiGPUBenchmark)
    assert "constexpr size_t kElementCount = 100 * 1024 * 1024;" in source
    assert "constexpr int kIterations = 100;" in source
    assert "cudaMemset(d_src, 1, bytes)" not in source
    validation = source.split("static void validate_and_dump_destination(", maxsplit=1)[1].split(
        "int main(", maxsplit=1
    )[0]
    assert "output.data(), d_dst" in validation
    assert "for (size_t i = 0; i < count; ++i)" in validation
    assert source.index('std::printf("TIME_MS:') < source.index(
        "validate_and_dump_destination(d_dst"
    )
