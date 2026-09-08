"""Real CPU and capability-gated CUDA controls for explicit execution audits."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from core.harness.execution_audit import (
    DestinationWriteCoverageGuard,
    HostTensorAllowance,
    audit_callable_once,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="real CUDA execution-audit integration requires a device",
)


def test_cpu_operation_audit_retains_positive_and_negative_extent_evidence() -> None:
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    clean = audit_callable_once(lambda: value.square(), expected_device="cpu")
    assert clean.passed
    assert clean.placement.operations_seen == 1
    assert clean.placement.expected_device_operations_seen == 1
    assert clean.placement.execution_observed
    assert clean.placement.failure_reasons == ()
    clean_record = clean.placement.operation_evidence[0]
    assert clean_record.operator == "aten.pow.Tensor_Scalar"
    assert clean_record.mismatched_paths == ()
    assert {(item.device, item.shape, item.numel) for item in clean_record.tensors} == {
        ("cpu", (3, 4), 12)
    }

    violation = audit_callable_once(lambda: value.square(), expected_device="cuda")
    assert not violation.passed
    assert violation.placement.expected_device_operations_seen == 0
    assert not violation.placement.execution_observed
    assert violation.placement.violations_seen == 1
    violation_record = violation.placement.violation_evidence[0]
    assert violation_record.operator == "aten.pow.Tensor_Scalar"
    assert violation_record.mismatched_paths == ("args[0]", "output")
    assert {(item.device, item.shape, item.numel) for item in violation_record.tensors} == {
        ("cpu", (3, 4), 12)
    }
    with pytest.raises(RuntimeError, match="EXECUTION AUDIT FAILED"):
        violation.raise_for_failure()


def test_host_tensor_allowance_requires_exact_identity_and_operator_scope() -> None:
    declared_scalar = torch.tensor(7.0)
    other_scalar = torch.tensor(8.0)
    allowance = HostTensorAllowance(
        label="declared_scalar",
        tensor=declared_scalar,
        operations=("aten._local_scalar_dense.default",),
    )

    allowed_host_only = audit_callable_once(
        declared_scalar.item,
        expected_device="cuda",
        allowed_host_tensors=(allowance,),
    )
    assert not allowed_host_only.passed
    assert allowed_host_only.placement.operations_seen == 1
    assert allowed_host_only.placement.violations_seen == 0
    assert allowed_host_only.placement.expected_device_operations_seen == 0
    assert allowed_host_only.placement.failure_reasons == (
        "no dispatcher-visible tensor operation touched the expected device cuda",
    )
    tensor_evidence = allowed_host_only.placement.operation_evidence[0].tensors
    assert len(tensor_evidence) == 1
    assert tensor_evidence[0].allowed_host_tensor

    wrong_identity = audit_callable_once(
        other_scalar.item,
        expected_device="cuda",
        allowed_host_tensors=(allowance,),
    )
    assert not wrong_identity.passed
    assert not wrong_identity.placement.violation_evidence[0].tensors[0].allowed_host_tensor

    wrong_operator = audit_callable_once(
        declared_scalar.neg,
        expected_device="cuda",
        allowed_host_tensors=(allowance,),
    )
    assert not wrong_operator.passed
    assert wrong_operator.placement.violation_evidence[0].operator == "aten.neg.default"


def test_noop_audit_fails_closed_without_scanning_bounded_evidence() -> None:
    result = audit_callable_once(lambda: None, expected_device="cuda:0", evidence_limit=1)

    assert not result.passed
    assert result.placement.operations_seen == 0
    assert result.placement.expected_device_operations_seen == 0
    assert not result.placement.execution_observed
    assert result.placement.failure_reasons == (
        "no dispatcher-visible tensor operations were observed",
    )
    with pytest.raises(RuntimeError, match="no dispatcher-visible tensor operations"):
        result.raise_for_failure()


def test_declared_destination_write_coverage_accepts_full_write_and_rejects_partial_write() -> None:
    source = torch.arange(8, dtype=torch.float32)
    full_destination = torch.empty_like(source)
    full = audit_callable_once(
        lambda: full_destination.copy_(source),
        expected_device="cpu",
        destinations={"output": full_destination},
    )
    assert full.passed
    assert len(full.destinations) == 1
    full_evidence = full.destinations[0]
    assert full_evidence.shape == (8,)
    assert full_evidence.numel == 8
    assert full_evidence.nbytes == 32
    assert full_evidence.unwritten_elements == 0
    torch.testing.assert_close(full_destination, source, rtol=0, atol=0)

    partial_destination = torch.empty_like(source)
    partial = audit_callable_once(
        lambda: partial_destination[:3].copy_(source[:3]),
        expected_device="cpu",
        destinations={"output": partial_destination},
    )
    assert partial.placement.passed
    assert not partial.passed
    partial_evidence = partial.destinations[0]
    assert partial_evidence.unwritten_elements == 5
    assert partial_evidence.first_unwritten_flat_indices == (3, 4, 5, 6, 7)


@pytest.mark.parametrize(
    ("tensor", "exception", "diagnostic"),
    [
        (torch.empty(8, dtype=torch.int64), TypeError, "floating-point or complex"),
        (torch.empty(2, 4).T, ValueError, "contiguous"),
        (torch.empty(0), ValueError, "empty"),
    ],
)
def test_destination_write_coverage_refuses_unprovable_tensor_contracts(
    tensor: torch.Tensor,
    exception: type[Exception],
    diagnostic: str,
) -> None:
    with pytest.raises(exception, match=diagnostic):
        DestinationWriteCoverageGuard({"output": tensor})


def _write_cli_benchmark(path: Path, *, mode: str) -> None:
    write_statement = {
        "noop": "pass",
        "full": "torch.mul(self.input, 2, out=self.output)",
        "partial": "self.output[:3].copy_(self.input[:3] * 2)",
        "reassigned": (
            "torch.mul(self.input, 2, out=self.output)\n"
            "                    self.output = self.input * 3"
        ),
    }[mode]
    path.write_text(
        textwrap.dedent(
            f"""
            import torch

            class AuditedCpuBenchmark:
                def setup(self):
                    self.input = torch.arange(6, dtype=torch.float32)
                    self.output = torch.empty_like(self.input)

                def benchmark_fn(self):
                    {write_statement}

                def teardown(self):
                    pass

            def get_benchmark():
                return AuditedCpuBenchmark()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("mode", "expected_returncode", "unwritten_elements", "identity_errors"),
    [
        ("full", 0, 0, []),
        ("partial", 2, 3, []),
        (
            "reassigned",
            2,
            0,
            ["destination attribute 'output' no longer references the declared tensor"],
        ),
    ],
)
def test_fresh_benchmark_cli_runs_real_out_of_timing_audit(
    tmp_path: Path,
    mode: str,
    expected_returncode: int,
    unwritten_elements: int,
    identity_errors: list[str],
) -> None:
    benchmark_path = tmp_path / "audited_cpu_benchmark.py"
    _write_cli_benchmark(benchmark_path, mode=mode)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.harness.execution_audit",
            str(benchmark_path),
            "--expected-device",
            "cpu",
            "--destination",
            "output",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == expected_returncode, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["schema"] == "aisp.execution-audit.v1"
    assert payload["fresh_instance"] is True
    assert payload["normal_timing_lifecycle_modified"] is False
    assert payload["placement"]["operations_seen"] > 0
    assert payload["destinations"][0]["numel"] == 6
    assert payload["destinations"][0]["unwritten_elements"] == unwritten_elements
    assert payload["destination_identity_errors"] == identity_errors
    assert payload["passed"] is (mode == "full")


def test_fresh_benchmark_cli_rejects_noop_without_declared_destination(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "noop_cpu_benchmark.py"
    _write_cli_benchmark(benchmark_path, mode="noop")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.harness.execution_audit",
            str(benchmark_path),
            "--expected-device",
            "cuda:0",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 2, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    assert payload["placement"]["operations_seen"] == 0
    assert payload["placement"]["expected_device_operations_seen"] == 0
    assert payload["placement"]["execution_observed"] is False
    assert payload["placement"]["failure_reasons"] == [
        "no dispatcher-visible tensor operations were observed"
    ]


@pytest.mark.parametrize("primary_failure", [False, True])
def test_fresh_benchmark_cli_preserves_primary_failure_when_teardown_also_fails(
    tmp_path: Path,
    primary_failure: bool,
) -> None:
    benchmark_path = tmp_path / "failing_lifecycle_cpu_benchmark.py"
    primary_statement = (
        'raise ValueError("primary execution failure")' if primary_failure else "pass"
    )
    benchmark_path.write_text(
        textwrap.dedent(
            f"""
            import torch

            class FailingLifecycleCpuBenchmark:
                def setup(self):
                    self.input = torch.arange(6, dtype=torch.float32)

                def benchmark_fn(self):
                    self.input.square()
                    {primary_statement}

                def teardown(self):
                    raise RuntimeError("secondary teardown failure")

            def get_benchmark():
                return FailingLifecycleCpuBenchmark()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.harness.execution_audit",
            str(benchmark_path),
            "--expected-device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    if primary_failure:
        assert payload["error"] == "ValueError: primary execution failure"
        assert payload["error_notes"] == [
            "fresh benchmark teardown failed: RuntimeError: secondary teardown failure"
        ]
    else:
        assert payload["error"] == (
            "RuntimeError: fresh benchmark teardown failed: RuntimeError: "
            "secondary teardown failure"
        )
        assert "error_notes" not in payload


@requires_cuda
def test_cuda_operation_audit_detects_real_cpu_spillover() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    cuda_value = torch.arange(16, dtype=torch.float32, device=device)
    cpu_value = torch.arange(5, dtype=torch.float32)

    clean = audit_callable_once(lambda: cuda_value.square(), expected_device=device)
    assert clean.passed
    assert clean.placement.operations_seen == 1
    assert clean.placement.expected_device_operations_seen == 1
    assert {item.device for item in clean.placement.operation_evidence[0].tensors} == {str(device)}

    def spill_to_cpu() -> None:
        cuda_value.square()
        cpu_value.square()

    violation = audit_callable_once(spill_to_cpu, expected_device=device)
    assert not violation.passed
    assert violation.placement.operations_seen == 2
    assert violation.placement.expected_device_operations_seen == 1
    assert violation.placement.violations_seen == 1
    record = violation.placement.violation_evidence[0]
    assert record.operator == "aten.pow.Tensor_Scalar"
    assert {(item.device, item.shape, item.numel) for item in record.tensors} == {("cpu", (5,), 5)}


@requires_cuda
def test_cuda_destination_write_coverage_detects_real_partial_write() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    source = torch.arange(16, dtype=torch.float32, device=device)

    full_destination = torch.empty_like(source)
    full = audit_callable_once(
        lambda: full_destination.copy_(source),
        expected_device=device,
        destinations={"output": full_destination},
    )
    assert full.passed
    assert full.destinations[0].unwritten_elements == 0

    partial_destination = torch.empty_like(source)
    partial = audit_callable_once(
        lambda: partial_destination[:7].copy_(source[:7]),
        expected_device=device,
        destinations={"output": partial_destination},
    )
    assert partial.placement.passed
    assert not partial.passed
    assert partial.destinations[0].shape == (16,)
    assert partial.destinations[0].unwritten_elements == 9
    assert partial.destinations[0].first_unwritten_flat_indices == tuple(range(7, 15))
