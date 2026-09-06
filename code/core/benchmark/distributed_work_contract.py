"""Observed completion receipts for distributed measured work.

The topology fields in :mod:`core.benchmark.verification` establish declared
baseline/optimized parity.  They do not prove which collective algorithm a
runtime selected.  This module covers a narrower runtime fact: every declared
rank observed its registered asynchronous collectives and final barrier finish
before that rank closed its timed region.

Runtime algorithm selection (for example NCCL ring versus tree) still requires
profiler-backed evidence.  A valid receipt from this module must not be treated
as that proof.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from core.benchmark.verification import DistributedTopology

BARRIER_BEFORE_TIMED_CLOSE = "barrier_before_timed_close"
WAIT_FOR_ASYNC_BEFORE_TIMED_CLOSE = "wait_for_async_before_timed_close"
DECLARED_ALGORITHM_EVIDENCE = "declared_only"


@dataclass(frozen=True)
class DistributedRankWorkReceipt:
    """Per-rank ordering evidence for one measured distributed region."""

    rank: int
    world_size: int
    backend: str
    collective_type: str
    declared_collective_algorithm: str
    gradient_bucket_bytes: int
    barrier_policy: str
    async_completion_policy: str
    timed_region_start_ns: int
    collective_launch_ns: tuple[int, ...]
    collective_completion_ns: tuple[int, ...]
    barrier_entry_ns: int | None
    barrier_completion_ns: int | None
    timed_region_close_ns: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize a receipt without filling missing evidence."""

        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "backend": self.backend,
            "collective_type": self.collective_type,
            "declared_collective_algorithm": self.declared_collective_algorithm,
            "gradient_bucket_bytes": self.gradient_bucket_bytes,
            "barrier_policy": self.barrier_policy,
            "async_completion_policy": self.async_completion_policy,
            "timed_region_start_ns": self.timed_region_start_ns,
            "collective_launch_ns": list(self.collective_launch_ns),
            "collective_completion_ns": list(self.collective_completion_ns),
            "barrier_entry_ns": self.barrier_entry_ns,
            "barrier_completion_ns": self.barrier_completion_ns,
            "timed_region_close_ns": self.timed_region_close_ns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DistributedRankWorkReceipt:
        """Deserialize a receipt, rejecting absent evidence fields."""

        required = {
            "rank",
            "world_size",
            "backend",
            "collective_type",
            "declared_collective_algorithm",
            "gradient_bucket_bytes",
            "barrier_policy",
            "async_completion_policy",
            "timed_region_start_ns",
            "collective_launch_ns",
            "collective_completion_ns",
            "barrier_entry_ns",
            "barrier_completion_ns",
            "timed_region_close_ns",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(
                "Distributed rank receipt missing required fields: " + ", ".join(missing)
            )
        launches = data["collective_launch_ns"]
        completions = data["collective_completion_ns"]
        if not isinstance(launches, list | tuple):
            raise TypeError("collective_launch_ns must be a list or tuple")
        if not isinstance(completions, list | tuple):
            raise TypeError("collective_completion_ns must be a list or tuple")
        return cls(
            rank=data["rank"],
            world_size=data["world_size"],
            backend=data["backend"],
            collective_type=data["collective_type"],
            declared_collective_algorithm=data["declared_collective_algorithm"],
            gradient_bucket_bytes=data["gradient_bucket_bytes"],
            barrier_policy=data["barrier_policy"],
            async_completion_policy=data["async_completion_policy"],
            timed_region_start_ns=data["timed_region_start_ns"],
            collective_launch_ns=tuple(launches),
            collective_completion_ns=tuple(completions),
            barrier_entry_ns=data["barrier_entry_ns"],
            barrier_completion_ns=data["barrier_completion_ns"],
            timed_region_close_ns=data["timed_region_close_ns"],
        )


@dataclass(frozen=True)
class DistributedWorkValidation:
    """Validation result with an explicit algorithm-evidence boundary."""

    passed: bool
    errors: tuple[str, ...]
    backend: str | None
    collective_algorithm_evidence: str = DECLARED_ALGORITHM_EVIDENCE

    def raise_for_failure(self) -> None:
        """Raise one bounded diagnostic when any receipt is invalid."""

        if not self.passed:
            raise RuntimeError("DISTRIBUTED WORK RECEIPT INVALID: " + " | ".join(self.errors))


def _is_timestamp(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _declared_contract_errors(topology: DistributedTopology) -> list[str]:
    errors: list[str] = []
    if topology.world_size <= 1:
        errors.append("world_size must be greater than one")
    if len(topology.ranks) != topology.world_size:
        errors.append(
            f"ranks must contain world_size entries: {len(topology.ranks)} vs {topology.world_size}"
        )
    if len(set(topology.ranks)) != len(topology.ranks):
        errors.append("ranks must be unique")
    if not isinstance(topology.collective_type, str) or not topology.collective_type.strip():
        errors.append("collective_type is required")
    if (
        not isinstance(topology.collective_algorithm, str)
        or not topology.collective_algorithm.strip()
    ):
        errors.append("collective_algorithm is required")
    if (
        isinstance(topology.gradient_bucket_bytes, bool)
        or not isinstance(topology.gradient_bucket_bytes, int)
        or topology.gradient_bucket_bytes <= 0
    ):
        errors.append("gradient_bucket_bytes must be a positive integer")
    if topology.barrier_policy != BARRIER_BEFORE_TIMED_CLOSE:
        errors.append(f"barrier_policy must be {BARRIER_BEFORE_TIMED_CLOSE!r}")
    if topology.async_completion_policy != WAIT_FOR_ASYNC_BEFORE_TIMED_CLOSE:
        errors.append(f"async_completion_policy must be {WAIT_FOR_ASYNC_BEFORE_TIMED_CLOSE!r}")
    return errors


def validate_distributed_work_receipts(
    topology: DistributedTopology,
    receipts: Sequence[DistributedRankWorkReceipt],
    *,
    expected_backend: str | None = None,
) -> DistributedWorkValidation:
    """Validate rank coverage and local event ordering for a measured region.

    Monotonic timestamps are compared only within a rank.  They are not assumed
    to be synchronized across hosts.
    """

    errors = _declared_contract_errors(topology)
    by_rank: dict[int, DistributedRankWorkReceipt] = {}
    for receipt in receipts:
        if not isinstance(receipt, DistributedRankWorkReceipt):
            errors.append("receipt entries must be DistributedRankWorkReceipt instances")
            continue
        if receipt.rank in by_rank:
            errors.append(f"duplicate receipt for rank {receipt.rank}")
            continue
        by_rank[receipt.rank] = receipt

    expected_ranks = set(topology.ranks)
    observed_ranks = set(by_rank)
    missing = sorted(expected_ranks - observed_ranks)
    extra = sorted(observed_ranks - expected_ranks)
    if missing:
        errors.append(f"missing receipts from ranks {missing}")
    if extra:
        errors.append(f"unexpected receipts from ranks {extra}")

    observed_backends = {
        receipt.backend
        for receipt in by_rank.values()
        if isinstance(receipt.backend, str) and receipt.backend.strip()
    }
    backend: str | None = None
    if len(observed_backends) == 1:
        backend = next(iter(observed_backends))
    elif observed_backends:
        errors.append(f"backend mismatch across ranks: {sorted(observed_backends)}")
    if expected_backend is not None and observed_backends != {expected_backend}:
        errors.append(
            f"expected backend {expected_backend!r}, observed {sorted(observed_backends)}"
        )

    for rank in sorted(expected_ranks & observed_ranks):
        receipt = by_rank[rank]
        prefix = f"rank {rank}: "
        if not isinstance(receipt.backend, str) or not receipt.backend.strip():
            errors.append(f"{prefix}backend must be a non-empty string")
        metadata_pairs = (
            ("world_size", receipt.world_size, topology.world_size),
            ("collective_type", receipt.collective_type, topology.collective_type),
            (
                "declared_collective_algorithm",
                receipt.declared_collective_algorithm,
                topology.collective_algorithm,
            ),
            (
                "gradient_bucket_bytes",
                receipt.gradient_bucket_bytes,
                topology.gradient_bucket_bytes,
            ),
            ("barrier_policy", receipt.barrier_policy, topology.barrier_policy),
            (
                "async_completion_policy",
                receipt.async_completion_policy,
                topology.async_completion_policy,
            ),
        )
        for field_name, observed, declared in metadata_pairs:
            if observed != declared:
                errors.append(f"{prefix}{field_name} mismatch: {observed!r} vs {declared!r}")

        timestamps = (
            ("timed_region_start_ns", receipt.timed_region_start_ns),
            ("timed_region_close_ns", receipt.timed_region_close_ns),
            ("barrier_entry_ns", receipt.barrier_entry_ns),
            ("barrier_completion_ns", receipt.barrier_completion_ns),
        )
        for field_name, value in timestamps:
            if not _is_timestamp(value):
                errors.append(f"{prefix}{field_name} must be a non-negative integer")

        if not receipt.collective_launch_ns:
            errors.append(f"{prefix}no asynchronous collective launch was recorded")
        if len(receipt.collective_completion_ns) != len(receipt.collective_launch_ns):
            errors.append(
                f"{prefix}asynchronous collectives incomplete: "
                f"{len(receipt.collective_completion_ns)} of "
                f"{len(receipt.collective_launch_ns)} completed"
            )
        for field_name, values in (
            ("collective_launch_ns", receipt.collective_launch_ns),
            ("collective_completion_ns", receipt.collective_completion_ns),
        ):
            if any(not _is_timestamp(value) for value in values):
                errors.append(f"{prefix}{field_name} entries must be non-negative integers")

        if all(
            _is_timestamp(value)
            for value in (
                receipt.timed_region_start_ns,
                receipt.timed_region_close_ns,
                receipt.barrier_entry_ns,
                receipt.barrier_completion_ns,
            )
        ):
            start = receipt.timed_region_start_ns
            close = receipt.timed_region_close_ns
            barrier_entry = receipt.barrier_entry_ns
            barrier_complete = receipt.barrier_completion_ns
            if start > close:
                errors.append(f"{prefix}timed region closes before it starts")
            if not (start <= barrier_entry <= barrier_complete <= close):
                errors.append(f"{prefix}barrier must enter and complete before timed-region close")
            for index, (launch, complete) in enumerate(
                zip(
                    receipt.collective_launch_ns,
                    receipt.collective_completion_ns,
                    strict=False,
                )
            ):
                if (
                    _is_timestamp(launch)
                    and _is_timestamp(complete)
                    and not (start <= launch <= complete <= barrier_entry)
                ):
                    errors.append(
                        f"{prefix}collective {index} was not observed complete "
                        "before the final barrier"
                    )

    return DistributedWorkValidation(
        passed=not errors,
        errors=tuple(errors),
        backend=backend,
    )


class DistributedWorkRecorder:
    """Record actual ``Work.wait`` and barrier returns inside a timed region."""

    def __init__(
        self,
        topology: DistributedTopology,
        *,
        rank: int,
        backend: str,
    ) -> None:
        contract_errors = _declared_contract_errors(topology)
        if contract_errors:
            raise ValueError(
                "Invalid declared distributed work contract: " + " | ".join(contract_errors)
            )
        if rank not in topology.ranks:
            raise ValueError(f"rank {rank} is not declared in topology ranks {topology.ranks}")
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError("backend must be a non-empty string")

        self._topology = topology
        self._rank = rank
        self._backend = backend
        self._timed_region_start_ns: int | None = None
        self._timed_region_close_ns: int | None = None
        self._work_handles: list[Any] = []
        self._collective_launch_ns: list[int] = []
        self._collective_completion_ns: list[int] = []
        self._barrier_entry_ns: int | None = None
        self._barrier_completion_ns: int | None = None

    def begin_timed_region(self) -> None:
        if self._timed_region_start_ns is not None:
            raise RuntimeError("timed region has already started")
        self._timed_region_start_ns = time.monotonic_ns()

    def record_async_collective(self, work: Any) -> None:
        if self._timed_region_start_ns is None:
            raise RuntimeError("begin_timed_region() must be called first")
        if self._timed_region_close_ns is not None:
            raise RuntimeError("cannot record a collective after timed-region close")
        if self._barrier_entry_ns is not None:
            raise RuntimeError("cannot record a collective after the final barrier")
        if not callable(getattr(work, "wait", None)):
            raise TypeError("async collective work must expose a callable wait()")
        self._work_handles.append(work)
        self._collective_launch_ns.append(time.monotonic_ns())

    def wait_for_async_collectives(self) -> None:
        if not self._work_handles:
            raise RuntimeError("no asynchronous collective work was recorded")
        for work in self._work_handles[len(self._collective_completion_ns) :]:
            work.wait()
            self._collective_completion_ns.append(time.monotonic_ns())

    def run_final_barrier(self, barrier: Callable[[], Any]) -> None:
        if self._timed_region_start_ns is None:
            raise RuntimeError("begin_timed_region() must be called first")
        if self._timed_region_close_ns is not None:
            raise RuntimeError("cannot run final barrier after timed-region close")
        if not self._work_handles:
            raise RuntimeError("no asynchronous collective work was recorded")
        if not callable(barrier):
            raise TypeError("barrier must be callable")
        if len(self._collective_completion_ns) != len(self._work_handles):
            raise RuntimeError(
                "all registered asynchronous collectives must complete before the final barrier"
            )
        if self._barrier_entry_ns is not None:
            raise RuntimeError("final barrier has already run")
        self._barrier_entry_ns = time.monotonic_ns()
        barrier_result = barrier()
        if callable(getattr(barrier_result, "wait", None)):
            raise RuntimeError(
                "final barrier returned asynchronous work; pass a synchronous barrier callable"
            )
        self._barrier_completion_ns = time.monotonic_ns()

    def close_timed_region(self) -> DistributedRankWorkReceipt:
        if self._timed_region_start_ns is None:
            raise RuntimeError("begin_timed_region() must be called first")
        if self._timed_region_close_ns is not None:
            raise RuntimeError("timed region has already closed")
        if not self._work_handles:
            raise RuntimeError("no asynchronous collective work was recorded")
        if len(self._collective_completion_ns) != len(self._work_handles):
            raise RuntimeError(
                "cannot close timed region before all asynchronous collectives complete"
            )
        if self._barrier_completion_ns is None:
            raise RuntimeError("cannot close timed region before the final barrier completes")
        self._timed_region_close_ns = time.monotonic_ns()
        return DistributedRankWorkReceipt(
            rank=self._rank,
            world_size=self._topology.world_size,
            backend=self._backend,
            collective_type=self._topology.collective_type or "",
            declared_collective_algorithm=self._topology.collective_algorithm or "",
            gradient_bucket_bytes=self._topology.gradient_bucket_bytes or 0,
            barrier_policy=self._topology.barrier_policy or "",
            async_completion_policy=self._topology.async_completion_policy or "",
            timed_region_start_ns=self._timed_region_start_ns,
            collective_launch_ns=tuple(self._collective_launch_ns),
            collective_completion_ns=tuple(self._collective_completion_ns),
            barrier_entry_ns=self._barrier_entry_ns,
            barrier_completion_ns=self._barrier_completion_ns,
            timed_region_close_ns=self._timed_region_close_ns,
        )


__all__ = [
    "BARRIER_BEFORE_TIMED_CLOSE",
    "DECLARED_ALGORITHM_EVIDENCE",
    "DistributedRankWorkReceipt",
    "DistributedWorkRecorder",
    "DistributedWorkValidation",
    "WAIT_FOR_ASYNC_BEFORE_TIMED_CLOSE",
    "validate_distributed_work_receipts",
]
