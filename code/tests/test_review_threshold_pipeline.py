"""Static regressions for the threshold async-pipeline ownership contract."""

from __future__ import annotations

from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "ch08" / "threshold_tma_kernel.cuh"


def test_threshold_pipeline_has_thread_local_stage_lifetime() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "auto pipe = cuda::make_pipeline();" in text
    assert "pipeline_shared_state" not in text
    assert "cuda::make_pipeline(block" not in text
    assert "block.sync()" not in text
    assert "threadIdx.x * values_per_thread" in text
    assert "stage_ptr(stage), inputs + offset, copy_bytes, pipe" in text

    acquire = text.index("pipe.producer_acquire()")
    commit = text.index("pipe.producer_commit()")
    wait = text.index("pipe.consumer_wait()")
    release = text.index("pipe.consumer_release()")
    assert acquire < commit < wait < release


@pytest.mark.parametrize("values_per_thread", [4, 6, 8])
@pytest.mark.parametrize("count", [1, 3, 4, 7, 2047, 2048, 2049, 8193, 12293])
def test_threshold_thread_striding_covers_each_element_once(
    values_per_thread: int,
    count: int,
) -> None:
    threads = 512
    blocks = 2
    tile_span = threads * values_per_thread
    stride = blocks * tile_span
    covered: list[int] = []

    for block in range(blocks):
        for thread in range(threads):
            thread_start = block * tile_span + thread * values_per_thread
            while thread_start < count:
                covered.extend(range(thread_start, min(thread_start + values_per_thread, count)))
                thread_start += stride

    assert sorted(covered) == list(range(count))
    assert len(covered) == count
