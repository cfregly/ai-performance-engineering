from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from ch04.optimized_pipeline_parallel_1f1b import _run_contiguous_1f1b_iterations
from ch04.pipeline_parallel_common import (
    PipelineIterationCapture,
    verify_and_concatenate_pipeline_capture,
)

_WORLD_SIZE = 2
_MICROBATCHES = 4
_ITERATIONS = 3
_SHAPE = (_MICROBATCHES, 1, 2)


def _continuous_gloo_worker(rank: int, rendezvous: str, result_dir: str) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    try:
        rank0_input = (
            torch.arange(torch.tensor(_SHAPE).prod().item(), dtype=torch.float32)
            .reshape(_SHAPE)
            .to(torch.bfloat16)
        )
        forward_calls = 0
        backward_calls = 0

        def forward_step(value: torch.Tensor) -> torch.Tensor:
            nonlocal forward_calls
            forward_calls += 1
            return value + float(rank + 1)

        def backward_step(
            _activation: torch.Tensor,
            value: torch.Tensor,
        ) -> torch.Tensor:
            nonlocal backward_calls
            backward_calls += 1
            return value + float((rank + 1) * 10)

        recv_forward = (
            [torch.empty((1, 1, 2), dtype=torch.bfloat16) for _ in range(_MICROBATCHES)]
            if rank > 0
            else []
        )
        recv_backward = (
            [torch.empty((1, 1, 2), dtype=torch.bfloat16) for _ in range(_MICROBATCHES)]
            if rank < _WORLD_SIZE - 1
            else []
        )
        scheduled_microbatches = _ITERATIONS * _MICROBATCHES
        capture = PipelineIterationCapture.create(
            _MICROBATCHES,
            first_microbatch_index=scheduled_microbatches - _MICROBATCHES,
        )

        _run_contiguous_1f1b_iterations(
            rank=rank,
            world_size=_WORLD_SIZE,
            micro_batches_per_iteration=_MICROBATCHES,
            iteration_count=_ITERATIONS,
            get_rank0_microbatch=lambda index: rank0_input[index : index + 1],
            recv_forward_buffers=recv_forward,
            recv_backward_buffers=recv_backward,
            forward_step=forward_step,
            backward_step=backward_step,
            activation_slots=[None] * max(_WORLD_SIZE - rank - 1, 1),
            capture=capture,
        )

        forward_stages = [
            lambda value, increment=float(stage + 1): value + increment
            for stage in range(_WORLD_SIZE)
        ]
        backward_stages = [
            lambda value, increment=float((stage + 1) * 10): value + increment
            for stage in range(_WORLD_SIZE)
        ]
        verify_input, verify_output = verify_and_concatenate_pipeline_capture(
            rank=rank,
            capture=capture,
            forward_stages=forward_stages,
            backward_stages=backward_stages,
            tolerance=(0.0, 0.0),
        )
        torch.save(
            {
                "forward_calls": forward_calls,
                "backward_calls": backward_calls,
                "verify_input": verify_input,
                "verify_output": verify_output,
                "unique_forward_buffers": len({tensor.data_ptr() for tensor in recv_forward}),
                "unique_backward_buffers": len({tensor.data_ptr() for tensor in recv_backward}),
            },
            Path(result_dir) / f"rank-{rank}.pt",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_contiguous_repeats_preserve_all_work_and_last_iteration_output(
    tmp_path: Path,
) -> None:
    rendezvous_path = tmp_path / "gloo-rendezvous"
    torch.multiprocessing.spawn(
        _continuous_gloo_worker,
        args=(f"file://{rendezvous_path}", str(tmp_path)),
        nprocs=_WORLD_SIZE,
        join=True,
        daemon=False,
    )

    expected_calls = _ITERATIONS * _MICROBATCHES
    rank0 = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    for result in (rank0, rank1):
        assert result["forward_calls"] == expected_calls
        assert result["backward_calls"] == expected_calls
        assert result["verify_input"].shape == _SHAPE
        assert result["verify_output"].shape == _SHAPE

    original = (
        torch.arange(torch.tensor(_SHAPE).prod().item(), dtype=torch.float32)
        .reshape(_SHAPE)
        .to(torch.bfloat16)
    )
    torch.testing.assert_close(rank0["verify_input"], original, rtol=0.0, atol=0.0)
    torch.testing.assert_close(rank0["verify_output"], original + 33.0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(rank1["verify_input"], original + 1.0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(rank1["verify_output"], original + 23.0, rtol=0.0, atol=0.0)

    assert rank0["unique_backward_buffers"] == _MICROBATCHES
    assert rank1["unique_forward_buffers"] == _MICROBATCHES


def test_capture_window_rejects_negative_start_and_requires_complete_window() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        PipelineIterationCapture.create(1, first_microbatch_index=-1)

    capture = PipelineIterationCapture.create(1, first_microbatch_index=4)
    with pytest.raises(RuntimeError, match="capture is incomplete"):
        capture.concatenate()
