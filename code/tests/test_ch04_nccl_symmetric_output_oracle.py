"""Independent full-output oracle checks for the NCCL symmetric-memory lab."""

from pathlib import Path

import numpy as np

from ch04 import run_nccl_symmetric_memory as runner


def test_vectorized_output_oracle_matches_scalar_reference() -> None:
    for ranks in (2, 3, 8):
        start = 37
        count = 257
        vectorized = runner._expected_output_chunk(ranks, start, count)
        scalar = np.asarray(
            [runner._expected_value(ranks, index) for index in range(start, start + count)],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(vectorized, scalar)


def test_full_output_validation_does_not_use_the_scalar_loop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workload = runner.NativeWorkload(
        arm="cuda_malloc",
        ranks=2,
        elements=4097,
        iterations=1,
        warmup=0,
    )
    output = tmp_path / "output.f32"
    output.write_bytes(
        runner._expected_output_chunk(workload.ranks, 0, workload.elements).tobytes()
    )

    def reject_scalar_loop(_ranks: int, _index: int) -> float:
        raise AssertionError("The full-output validator regressed to the scalar oracle")

    monkeypatch.setattr(runner, "_expected_value", reject_scalar_loop)

    assert runner.validate_full_output(output, workload) == {
        "checked_elements": workload.elements,
        "max_abs_error": 0.0,
    }
