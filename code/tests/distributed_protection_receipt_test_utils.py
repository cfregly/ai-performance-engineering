"""Shared real contract controls for retained distributed anti-cheat test IDs."""

from __future__ import annotations

from dataclasses import replace

from core.benchmark.distributed_work_contract import (
    BARRIER_BEFORE_TIMED_CLOSE,
    DECLARED_ALGORITHM_EVIDENCE,
    WAIT_FOR_ASYNC_BEFORE_TIMED_CLOSE,
    DistributedRankWorkReceipt,
    validate_distributed_work_receipts,
)
from core.benchmark.verification import DistributedTopology, compare_topologies


def _topology(**overrides: object) -> DistributedTopology:
    values = {
        "world_size": 2,
        "ranks": [0, 1],
        "shards": 2,
        "per_rank_batch_size": 2,
        "collective_type": "all_reduce",
        "collective_algorithm": "ring",
        "gradient_bucket_bytes": 4096,
        "barrier_policy": BARRIER_BEFORE_TIMED_CLOSE,
        "async_completion_policy": WAIT_FOR_ASYNC_BEFORE_TIMED_CLOSE,
    }
    values.update(overrides)
    return DistributedTopology(**values)


def _receipt(rank: int, **overrides: object) -> DistributedRankWorkReceipt:
    values = {
        "rank": rank,
        "world_size": 2,
        "backend": "gloo",
        "collective_type": "all_reduce",
        "declared_collective_algorithm": "ring",
        "gradient_bucket_bytes": 4096,
        "barrier_policy": BARRIER_BEFORE_TIMED_CLOSE,
        "async_completion_policy": WAIT_FOR_ASYNC_BEFORE_TIMED_CLOSE,
        "timed_region_start_ns": 100,
        "collective_launch_ns": (110,),
        "collective_completion_ns": (130,),
        "barrier_entry_ns": 140,
        "barrier_completion_ns": 150,
        "timed_region_close_ns": 160,
    }
    values.update(overrides)
    return DistributedRankWorkReceipt(**values)


def _assert_clean_receipts() -> tuple[DistributedRankWorkReceipt, ...]:
    receipts = (_receipt(0), _receipt(1))
    validation = validate_distributed_work_receipts(
        _topology(),
        receipts,
        expected_backend="gloo",
    )
    assert validation.passed, validation.errors
    assert validation.collective_algorithm_evidence == DECLARED_ALGORITHM_EVIDENCE
    return receipts


def assert_collective_algorithm_declaration_controls() -> None:
    """Check declared ring/tree parity without claiming runtime algorithm inspection."""

    baseline = _topology()
    assert compare_topologies(baseline, replace(baseline)) == (True, None)
    passed, differences = compare_topologies(
        baseline,
        replace(baseline, collective_algorithm="tree"),
    )
    assert not passed
    assert differences == "Collective algorithm mismatch: ring vs tree"

    receipts = list(_assert_clean_receipts())
    receipts[1] = replace(receipts[1], declared_collective_algorithm="tree")
    validation = validate_distributed_work_receipts(_topology(), receipts)
    assert not validation.passed
    assert any("declared_collective_algorithm mismatch" in error for error in validation.errors)


def assert_gradient_bucket_declaration_controls() -> None:
    """Check bucket-byte parity in topology declarations and registered receipts."""

    baseline = _topology()
    assert compare_topologies(baseline, replace(baseline)) == (True, None)
    passed, differences = compare_topologies(
        baseline,
        replace(baseline, gradient_bucket_bytes=8192),
    )
    assert not passed
    assert differences == "Gradient bucket bytes mismatch: 4096 vs 8192"

    receipts = list(_assert_clean_receipts())
    receipts[1] = replace(receipts[1], gradient_bucket_bytes=8192)
    validation = validate_distributed_work_receipts(_topology(), receipts)
    assert not validation.passed
    assert any("gradient_bucket_bytes mismatch" in error for error in validation.errors)


def assert_barrier_completion_receipt_controls() -> None:
    """Check that a registered final barrier completes before timed-region close."""

    receipts = list(_assert_clean_receipts())
    receipts[1] = replace(receipts[1], barrier_completion_ns=170)
    validation = validate_distributed_work_receipts(_topology(), receipts)
    assert not validation.passed
    assert any(
        "barrier must enter and complete before timed-region close" in error
        for error in validation.errors
    )


def assert_async_completion_receipt_controls() -> None:
    """Check that every registered async collective completes before the barrier."""

    receipts = list(_assert_clean_receipts())
    receipts[1] = replace(receipts[1], collective_completion_ns=())
    validation = validate_distributed_work_receipts(_topology(), receipts)
    assert not validation.passed
    assert any("asynchronous collectives incomplete" in error for error in validation.errors)


__all__ = [
    "assert_async_completion_receipt_controls",
    "assert_barrier_completion_receipt_controls",
    "assert_collective_algorithm_declaration_controls",
    "assert_gradient_bucket_declaration_controls",
]
