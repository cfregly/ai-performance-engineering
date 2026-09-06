"""Distributed work-contract controls with an actual CPU/Gloo execution path."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from core.benchmark.distributed_work_contract import (
    BARRIER_BEFORE_TIMED_CLOSE,
    DECLARED_ALGORITHM_EVIDENCE,
    WAIT_FOR_ASYNC_BEFORE_TIMED_CLOSE,
    DistributedRankWorkReceipt,
    DistributedWorkRecorder,
    validate_distributed_work_receipts,
)
from core.benchmark.quarantine import QuarantineManager
from core.benchmark.verification import (
    DistributedTopology,
    InputSignature,
    PrecisionFlags,
    QuarantineReason,
    compare_topologies,
    extract_distributed_topology,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.verify_runner import VerifyConfig, VerifyRunner


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


def _signature(**overrides: object) -> InputSignature:
    values = {
        "shapes": {"input": (4,)},
        "dtypes": {"input": "float32"},
        "batch_size": 4,
        "parameter_count": 0,
        "precision_flags": PrecisionFlags(tf32=False),
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
    return InputSignature(**values)


def _gloo_topology() -> DistributedTopology:
    # Gloo does not expose ring/tree execution proof through this test. The
    # declared value intentionally says only that the backend selects it.
    return _topology(collective_algorithm="backend_selected")


@dataclass
class _WorkloadMetadata:
    requests_per_iteration: float = 1.0
    tokens_per_iteration: float | None = None
    samples_per_iteration: float | None = 4.0
    bytes_per_iteration: float | None = 16.0
    custom_units_per_iteration: float | None = None


class _CpuPairBenchmark:
    """Real tensor work used to exercise VerifyRunner's signature cache path."""

    _is_deterministic = True

    def __init__(self, signature: InputSignature) -> None:
        self._signature = signature
        self._input = torch.arange(4, dtype=torch.float32)
        self._output = self._input * 2
        self._workload = _WorkloadMetadata()

    def setup(self) -> None:
        self._input = torch.arange(4, dtype=torch.float32)

    def benchmark_fn(self) -> None:
        self._output = self._input * 2

    def teardown(self) -> None:
        pass

    def get_input_signature(self) -> InputSignature:
        return self._signature

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        return {"input": self._input}

    def get_verify_output(self) -> torch.Tensor:
        return self._output

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def get_workload_metadata(self) -> _WorkloadMetadata:
        return self._workload

    def validate_result(self) -> None:
        return None


class _MixinCpuPairBenchmark(VerificationPayloadMixin):
    """Actual CPU tensor work declared through the preferred payload mixin."""

    _is_deterministic = True

    def __init__(self, *, algorithm: str, bucket_bytes: object) -> None:
        self._algorithm = algorithm
        self._bucket_bytes = bucket_bytes
        self._input = torch.arange(4, dtype=torch.float32)
        self._output = self._input * 2
        self._workload = _WorkloadMetadata()

    def setup(self) -> None:
        self._input = torch.arange(4, dtype=torch.float32)

    def benchmark_fn(self) -> None:
        self._output = self._input * 2

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self._input},
            output=self._output,
            batch_size=4,
            precision_flags=PrecisionFlags(tf32=False),
            output_tolerance=(0.0, 0.0),
            signature_overrides={
                "world_size": 2,
                "ranks": [0, 1],
                "shards": 2,
                "per_rank_batch_size": 2,
                "collective_type": "all_reduce",
                "collective_algorithm": self._algorithm,
                "gradient_bucket_bytes": self._bucket_bytes,
                "barrier_policy": BARRIER_BEFORE_TIMED_CLOSE,
                "async_completion_policy": WAIT_FOR_ASYNC_BEFORE_TIMED_CLOSE,
            },
        )

    def teardown(self) -> None:
        pass

    def get_workload_metadata(self) -> _WorkloadMetadata:
        return self._workload

    def validate_result(self) -> None:
        return None


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


def test_distributed_signature_round_trip_and_undeclared_compatibility() -> None:
    declared = _signature()
    restored = InputSignature.from_dict(declared.to_dict())
    assert restored == declared
    assert extract_distributed_topology(restored) == _topology()
    assert declared.validate(strict=True) == []

    undeclared = _signature(
        collective_algorithm=None,
        gradient_bucket_bytes=None,
        barrier_policy=None,
        async_completion_policy=None,
    )
    payload = undeclared.to_dict()
    assert "collective_algorithm" not in payload
    assert "gradient_bucket_bytes" not in payload
    assert "barrier_policy" not in payload
    assert "async_completion_policy" not in payload
    assert InputSignature.from_dict(payload) == undeclared


@pytest.mark.parametrize(
    ("field_name", "optimized_value", "diagnostic"),
    [
        ("ranks", [1, 0], "Ranks mismatch"),
        ("per_rank_batch_size", 1, "Per-rank batch size mismatch"),
        ("collective_type", "all_gather", "Collective type mismatch"),
        ("collective_algorithm", "tree", "Collective algorithm mismatch"),
        ("gradient_bucket_bytes", 8192, "Gradient bucket bytes mismatch"),
        ("barrier_policy", None, "Barrier policy mismatch"),
        ("async_completion_policy", None, "Async completion policy mismatch"),
    ],
)
def test_compare_topologies_checks_every_distributed_work_field(
    field_name: str,
    optimized_value: object,
    diagnostic: str,
) -> None:
    match, reason = compare_topologies(
        _topology(),
        _topology(**{field_name: optimized_value}),
    )
    assert not match
    assert reason is not None and diagnostic in reason


@pytest.mark.parametrize(
    ("optimized_overrides", "mismatch_path"),
    [
        ({"collective_algorithm": "tree"}, "collective_algorithm"),
        ({"gradient_bucket_bytes": 8192}, "gradient_bucket_bytes"),
    ],
)
def test_verify_runner_rejects_distributed_contract_mismatch(
    tmp_path: Path,
    optimized_overrides: dict[str, object],
    mismatch_path: str,
) -> None:
    runner = VerifyRunner(
        cache_dir=tmp_path / "cache",
        quarantine_manager=QuarantineManager(tmp_path / "quarantine.json"),
    )
    result = runner.verify_pair(
        _CpuPairBenchmark(_signature()),
        _CpuPairBenchmark(_signature(**optimized_overrides)),
        VerifyConfig(skip_jitter_check=True, skip_fresh_input_check=True),
    )
    assert not result.passed
    assert result.reason == QuarantineReason.SIGNATURE_MISMATCH.value
    assert mismatch_path in result.details["signature_mismatches"]


def test_verify_runner_accepts_matching_declared_distributed_contract(
    tmp_path: Path,
) -> None:
    runner = VerifyRunner(
        cache_dir=tmp_path / "cache",
        quarantine_manager=QuarantineManager(tmp_path / "quarantine.json"),
    )
    result = runner.verify_pair(
        _CpuPairBenchmark(_signature()),
        _CpuPairBenchmark(_signature()),
        VerifyConfig(skip_jitter_check=True, skip_fresh_input_check=True),
    )
    assert result.passed


@pytest.mark.parametrize(
    ("optimized_algorithm", "optimized_bucket", "mismatch_path"),
    [
        ("tree", "4096", "collective_algorithm"),
        ("ring", 8192, "gradient_bucket_bytes"),
    ],
)
def test_payload_mixin_reaches_real_signature_comparison_path(
    tmp_path: Path,
    optimized_algorithm: str,
    optimized_bucket: object,
    mismatch_path: str,
) -> None:
    runner = VerifyRunner(
        cache_dir=tmp_path / "cache",
        quarantine_manager=QuarantineManager(tmp_path / "quarantine.json"),
    )
    result = runner.verify_pair(
        _MixinCpuPairBenchmark(algorithm="ring", bucket_bytes="4096"),
        _MixinCpuPairBenchmark(
            algorithm=optimized_algorithm,
            bucket_bytes=optimized_bucket,
        ),
        VerifyConfig(skip_jitter_check=True, skip_fresh_input_check=True),
    )
    assert not result.passed
    assert result.reason == QuarantineReason.SIGNATURE_MISMATCH.value
    assert mismatch_path in result.details["signature_mismatches"]


def test_payload_mixin_rejects_boolean_bucket_size() -> None:
    benchmark = _MixinCpuPairBenchmark(algorithm="ring", bucket_bytes=True)
    benchmark.setup()
    benchmark.benchmark_fn()
    with pytest.raises(TypeError, match="gradient_bucket_bytes.*not bool"):
        benchmark.capture_verification_payload()


@pytest.mark.parametrize(
    ("overrides", "diagnostic"),
    [
        ({"collective_algorithm": ""}, "collective_algorithm"),
        ({"collective_algorithm": "ring", "collective_type": None}, "collective_type"),
        ({"gradient_bucket_bytes": 0}, "gradient_bucket_bytes"),
        ({"barrier_policy": "outside_timing"}, "barrier_policy"),
        ({"async_completion_policy": "fire_and_forget"}, "async_completion_policy"),
        ({"world_size": 1}, "world_size > 1"),
    ],
)
def test_input_signature_rejects_invalid_declared_distributed_contract(
    overrides: dict[str, object],
    diagnostic: str,
) -> None:
    errors = _signature(**overrides).validate(strict=True)
    assert any(diagnostic in error for error in errors)


def test_receipt_round_trip_requires_every_evidence_field() -> None:
    receipt = _receipt(0)
    assert DistributedRankWorkReceipt.from_dict(receipt.to_dict()) == receipt
    incomplete = receipt.to_dict()
    del incomplete["barrier_completion_ns"]
    with pytest.raises(ValueError, match="barrier_completion_ns"):
        DistributedRankWorkReceipt.from_dict(incomplete)


def test_receipt_validator_clean_control_and_algorithm_boundary() -> None:
    validation = validate_distributed_work_receipts(
        _topology(),
        [_receipt(0), _receipt(1)],
        expected_backend="gloo",
    )
    assert validation.passed
    assert validation.errors == ()
    assert validation.backend == "gloo"
    assert validation.collective_algorithm_evidence == DECLARED_ALGORITHM_EVIDENCE


@pytest.mark.parametrize(
    ("receipts", "diagnostic"),
    [
        ([_receipt(0)], "missing receipts from ranks [1]"),
        (
            [_receipt(0), _receipt(1, collective_completion_ns=())],
            "asynchronous collectives incomplete",
        ),
        (
            [_receipt(0), _receipt(1, barrier_completion_ns=170)],
            "barrier must enter and complete before timed-region close",
        ),
        (
            [_receipt(0), _receipt(1, collective_completion_ns=(145,))],
            "was not observed complete before the final barrier",
        ),
        (
            [_receipt(0), _receipt(1, declared_collective_algorithm="tree")],
            "declared_collective_algorithm mismatch",
        ),
    ],
)
def test_receipt_validator_rejects_incomplete_or_late_evidence(
    receipts: list[DistributedRankWorkReceipt],
    diagnostic: str,
) -> None:
    validation = validate_distributed_work_receipts(
        _topology(),
        receipts,
        expected_backend="gloo",
    )
    assert not validation.passed
    assert any(diagnostic in error for error in validation.errors)
    with pytest.raises(RuntimeError, match="DISTRIBUTED WORK RECEIPT INVALID"):
        validation.raise_for_failure()


def test_recorder_refuses_to_close_before_async_completion_or_barrier() -> None:
    recorder = DistributedWorkRecorder(_topology(), rank=0, backend="gloo")
    recorder.begin_timed_region()
    with pytest.raises(RuntimeError, match="no asynchronous collective work"):
        recorder.close_timed_region()


def _gloo_receipt_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    output_dir: str,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=15),
    )
    try:
        topology = _gloo_topology()
        recorder = DistributedWorkRecorder(topology, rank=rank, backend="gloo")
        value = torch.full((1024,), float(rank + 1), dtype=torch.float32)
        assert value.numel() * value.element_size() == topology.gradient_bucket_bytes
        recorder.begin_timed_region()
        work = dist.all_reduce(value, async_op=True)
        recorder.record_async_collective(work)
        recorder.wait_for_async_collectives()
        torch.testing.assert_close(
            value,
            torch.full_like(value, 3.0),
            rtol=0,
            atol=0,
        )
        recorder.run_final_barrier(dist.barrier)
        receipt = recorder.close_timed_region()
        Path(output_dir, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "receipt": receipt.to_dict(),
                    "output_elements_verified": value.numel(),
                },
                sort_keys=True,
            )
            + "\n"
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Actual Gloo backend required",
)
def test_actual_two_rank_gloo_async_completion_and_barrier_receipts(
    tmp_path: Path,
) -> None:
    context = mp.spawn(
        _gloo_receipt_worker,
        args=(2, f"file://{tmp_path / 'gloo-store'}", str(tmp_path)),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 30
    try:
        while not context.join(timeout=2):
            if time.monotonic() >= deadline:
                pytest.fail("Actual two-rank Gloo receipt control exceeded 30 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)

    payloads = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert [payload["output_elements_verified"] for payload in payloads] == [1024, 1024]
    receipts = [DistributedRankWorkReceipt.from_dict(payload["receipt"]) for payload in payloads]
    validation = validate_distributed_work_receipts(
        _gloo_topology(),
        receipts,
        expected_backend="gloo",
    )
    assert validation.passed, validation.errors
    assert validation.collective_algorithm_evidence == DECLARED_ALGORITHM_EVIDENCE
