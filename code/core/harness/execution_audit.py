"""Explicit, out-of-timing execution audits for benchmark callbacks.

This module intentionally does not participate in normal benchmark timing.  A
dispatcher mode observes tensor operands and results for one audited invocation,
and an optional destination guard poisons declared floating-point output buffers
before that invocation to prove that every logical element was overwritten.

The placement audit covers PyTorch dispatcher operations executed on the current
thread.  It cannot see arbitrary work inside custom extensions, background
threads, host callbacks, or external processes.  Destination poisoning proves
write coverage only for the exact buffers supplied by the caller; it is not a
general uninitialized-memory or allocation-provenance detector.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

PLACEMENT_SCOPE = (
    "PyTorch dispatcher-visible tensor operations on the audited invocation's current thread"
)
WRITE_COVERAGE_SCOPE = (
    "exact declared contiguous floating-point or complex destinations poisoned before the "
    "audited invocation"
)


@dataclass(frozen=True, eq=False)
class HostTensorAllowance:
    """Allow one exact CPU tensor only for explicitly named dispatcher operations."""

    label: str
    tensor: torch.Tensor
    operations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("host tensor allowance label must be non-empty")
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("host tensor allowance must reference a torch.Tensor")
        if self.tensor.device.type != "cpu":
            raise ValueError("host tensor allowances apply only to exact CPU tensor identities")
        if not self.operations or any(not operation.strip() for operation in self.operations):
            raise ValueError("host tensor allowance operations must be non-empty strings")


@dataclass(frozen=True)
class TensorExtentEvidence:
    """Observed tensor identity-independent extent and placement evidence."""

    path: str
    device: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    numel: int
    nbytes: int
    matches_expected_device: bool
    allowed_host_tensor: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "device": self.device,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "numel": self.numel,
            "nbytes": self.nbytes,
            "matches_expected_device": self.matches_expected_device,
            "allowed_host_tensor": self.allowed_host_tensor,
        }


@dataclass(frozen=True)
class OperationEvidence:
    """Bounded evidence for one dispatcher operation."""

    operator: str
    tensors: tuple[TensorExtentEvidence, ...]
    mismatched_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "mismatched_paths": list(self.mismatched_paths),
        }


@dataclass(frozen=True)
class OperationPlacementResult:
    """Result of checking every dispatcher-visible tensor in one invocation."""

    expected_device: str
    operations_seen: int
    expected_device_operations_seen: int
    operator_counts: tuple[tuple[str, int], ...]
    operation_evidence: tuple[OperationEvidence, ...]
    operation_evidence_truncated: bool
    violations_seen: int
    violation_evidence: tuple[OperationEvidence, ...]
    violation_evidence_truncated: bool

    @property
    def passed(self) -> bool:
        return self.violations_seen == 0 and self.expected_device_operations_seen > 0

    @property
    def execution_observed(self) -> bool:
        """Whether at least one operation touched a tensor on the expected device."""

        return self.expected_device_operations_seen > 0

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.execution_observed:
            if self.operations_seen == 0:
                reasons.append("no dispatcher-visible tensor operations were observed")
            else:
                reasons.append(
                    "no dispatcher-visible tensor operation touched the expected device "
                    f"{self.expected_device}"
                )
        if self.violations_seen:
            reasons.append(f"{self.violations_seen} operation placement violation(s)")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "scope": PLACEMENT_SCOPE,
            "expected_device": self.expected_device,
            "operations_seen": self.operations_seen,
            "expected_device_operations_seen": self.expected_device_operations_seen,
            "execution_observed": self.execution_observed,
            "failure_reasons": list(self.failure_reasons),
            "operator_counts": dict(self.operator_counts),
            "operation_evidence": [item.to_dict() for item in self.operation_evidence],
            "operation_evidence_truncated": self.operation_evidence_truncated,
            "violations_seen": self.violations_seen,
            "violation_evidence": [item.to_dict() for item in self.violation_evidence],
            "violation_evidence_truncated": self.violation_evidence_truncated,
        }


def _iter_tensor_paths(value: Any, path: str):
    if isinstance(value, torch.Tensor):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_tensor_paths(item, f"{path}[{key!r}]")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _iter_tensor_paths(item, f"{path}[{index}]")


class TensorOperationPlacementAudit(TorchDispatchMode):
    """Observe tensor devices for PyTorch operations on the current thread."""

    def __init__(
        self,
        expected_device: str | torch.device,
        *,
        allowed_host_tensors: Sequence[HostTensorAllowance] = (),
        evidence_limit: int = 64,
    ) -> None:
        super().__init__()
        if isinstance(evidence_limit, bool) or evidence_limit <= 0:
            raise ValueError("evidence_limit must be a positive integer")
        self.expected_device = torch.device(expected_device)
        self.evidence_limit = evidence_limit
        self._allowances = tuple(allowed_host_tensors)
        self._allowed_operations_by_identity: dict[int, set[str]] = {}
        for allowance in self._allowances:
            if not isinstance(allowance, HostTensorAllowance):
                raise TypeError("allowed_host_tensors must contain HostTensorAllowance instances")
            self._allowed_operations_by_identity.setdefault(id(allowance.tensor), set()).update(
                allowance.operations
            )
        self._operations_seen = 0
        self._expected_device_operations_seen = 0
        self._operator_counts: Counter[str] = Counter()
        self._operation_evidence: list[OperationEvidence] = []
        self._violations_seen = 0
        self._violation_evidence: list[OperationEvidence] = []

    def _matches_expected_device(self, actual: torch.device) -> bool:
        if actual.type != self.expected_device.type:
            return False
        if self.expected_device.index is None:
            return True
        return actual.index == self.expected_device.index

    def _is_allowed_host_tensor(self, tensor: torch.Tensor, operator: str) -> bool:
        if tensor.device.type != "cpu":
            return False
        return operator in self._allowed_operations_by_identity.get(id(tensor), set())

    def _tensor_evidence(
        self,
        *,
        path: str,
        tensor: torch.Tensor,
        operator: str,
    ) -> TensorExtentEvidence:
        matches = self._matches_expected_device(tensor.device)
        allowed = not matches and self._is_allowed_host_tensor(tensor, operator)
        return TensorExtentEvidence(
            path=path,
            device=str(tensor.device),
            dtype=str(tensor.dtype).removeprefix("torch."),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            numel=tensor.numel(),
            nbytes=tensor.numel() * tensor.element_size(),
            matches_expected_device=matches,
            allowed_host_tensor=allowed,
        )

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        actual_kwargs = kwargs or {}
        result = func(*args, **actual_kwargs)
        operator = str(func)
        self._operations_seen += 1
        self._operator_counts[operator] += 1

        observed = [
            *self._collect_tensor_evidence(args, "args", operator),
            *self._collect_tensor_evidence(actual_kwargs, "kwargs", operator),
            *self._collect_tensor_evidence(result, "output", operator),
        ]
        if any(item.matches_expected_device for item in observed):
            self._expected_device_operations_seen += 1
        mismatched_paths = tuple(
            item.path
            for item in observed
            if not item.matches_expected_device and not item.allowed_host_tensor
        )
        evidence = OperationEvidence(
            operator=operator,
            tensors=tuple(observed),
            mismatched_paths=mismatched_paths,
        )
        if len(self._operation_evidence) < self.evidence_limit:
            self._operation_evidence.append(evidence)
        if mismatched_paths:
            self._violations_seen += 1
            if len(self._violation_evidence) < self.evidence_limit:
                self._violation_evidence.append(evidence)
        return result

    def _collect_tensor_evidence(
        self,
        value: Any,
        path: str,
        operator: str,
    ) -> list[TensorExtentEvidence]:
        return [
            self._tensor_evidence(path=item_path, tensor=tensor, operator=operator)
            for item_path, tensor in _iter_tensor_paths(value, path)
        ]

    def result(self) -> OperationPlacementResult:
        """Return immutable bounded evidence collected so far."""

        return OperationPlacementResult(
            expected_device=str(self.expected_device),
            operations_seen=self._operations_seen,
            expected_device_operations_seen=self._expected_device_operations_seen,
            operator_counts=tuple(sorted(self._operator_counts.items())),
            operation_evidence=tuple(self._operation_evidence),
            operation_evidence_truncated=self._operations_seen > len(self._operation_evidence),
            violations_seen=self._violations_seen,
            violation_evidence=tuple(self._violation_evidence),
            violation_evidence_truncated=self._violations_seen > len(self._violation_evidence),
        )


@dataclass(frozen=True)
class DestinationWriteCoverageEvidence:
    """Write-coverage result for one exact declared destination tensor."""

    name: str
    device: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    numel: int
    nbytes: int
    unwritten_elements: int
    first_unwritten_flat_indices: tuple[int, ...]

    @property
    def passed(self) -> bool:
        return self.unwritten_elements == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "scope": WRITE_COVERAGE_SCOPE,
            "name": self.name,
            "device": self.device,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "numel": self.numel,
            "nbytes": self.nbytes,
            "unwritten_elements": self.unwritten_elements,
            "first_unwritten_flat_indices": list(self.first_unwritten_flat_indices),
        }


class DestinationWriteCoverageGuard:
    """Poison exact declared output buffers and check that all elements change."""

    def __init__(self, destinations: Mapping[str, torch.Tensor]) -> None:
        self._destinations = dict(destinations)
        if not self._destinations:
            raise ValueError("at least one destination tensor is required")
        seen_storage: dict[tuple[str, int], str] = {}
        for name, tensor in self._destinations.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("destination names must be non-empty strings")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"destination {name!r} must be a torch.Tensor")
            if tensor.device.type == "meta":
                raise ValueError(f"destination {name!r} cannot be a meta tensor")
            if tensor.numel() == 0:
                raise ValueError(f"destination {name!r} is empty; write coverage is unobservable")
            if not (tensor.is_floating_point() or tensor.is_complex()):
                raise TypeError(
                    f"destination {name!r} must be floating-point or complex for NaN poisoning"
                )
            if not tensor.is_contiguous():
                raise ValueError(
                    f"destination {name!r} must be contiguous for exact logical write coverage"
                )
            storage_key = (str(tensor.device), tensor.untyped_storage().data_ptr())
            previous_name = seen_storage.get(storage_key)
            if previous_name is not None:
                raise ValueError(
                    f"destinations {previous_name!r} and {name!r} share storage; "
                    "audit them separately"
                )
            seen_storage[storage_key] = name
        self._poisoned = False

    def _synchronize_cuda_destinations(self) -> None:
        devices = {
            tensor.device for tensor in self._destinations.values() if tensor.device.type == "cuda"
        }
        for device in sorted(devices, key=str):
            torch.cuda.synchronize(device)

    def poison(self) -> None:
        """Fill destinations with NaNs before the audited invocation."""

        if self._poisoned:
            raise RuntimeError("destination write-coverage guard has already been poisoned")
        with torch.no_grad():
            for tensor in self._destinations.values():
                tensor.fill_(float("nan"))
        self._synchronize_cuda_destinations()
        self._poisoned = True

    def inspect(self) -> tuple[DestinationWriteCoverageEvidence, ...]:
        """Synchronize CUDA and report any poison that survived the invocation."""

        if not self._poisoned:
            raise RuntimeError("poison() must be called before inspect()")
        self._synchronize_cuda_destinations()
        results: list[DestinationWriteCoverageEvidence] = []
        for name, tensor in self._destinations.items():
            unwritten_mask = torch.isnan(tensor)
            unwritten_elements = int(unwritten_mask.count_nonzero().item())
            first_indices: tuple[int, ...] = ()
            if unwritten_elements:
                first_indices = tuple(
                    int(index)
                    for index in unwritten_mask.flatten().nonzero()[:8].flatten().tolist()
                )
            results.append(
                DestinationWriteCoverageEvidence(
                    name=name,
                    device=str(tensor.device),
                    dtype=str(tensor.dtype).removeprefix("torch."),
                    shape=tuple(tensor.shape),
                    stride=tuple(tensor.stride()),
                    storage_offset=int(tensor.storage_offset()),
                    numel=tensor.numel(),
                    nbytes=tensor.numel() * tensor.element_size(),
                    unwritten_elements=unwritten_elements,
                    first_unwritten_flat_indices=first_indices,
                )
            )
        return tuple(results)


@dataclass(frozen=True)
class ExecutionAuditResult:
    """Combined placement and declared-destination evidence."""

    placement: OperationPlacementResult
    destinations: tuple[DestinationWriteCoverageEvidence, ...]
    destination_identity_errors: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            self.placement.passed
            and all(destination.passed for destination in self.destinations)
            and not self.destination_identity_errors
        )

    def raise_for_failure(self) -> None:
        if self.passed:
            return
        diagnostics: list[str] = []
        if not self.placement.execution_observed:
            diagnostics.append(self.placement.failure_reasons[0])
        if self.placement.violation_evidence:
            first = self.placement.violation_evidence[0]
            diagnostics.append(
                f"{self.placement.violations_seen} operation placement violation(s); "
                f"first={first.operator} paths={list(first.mismatched_paths)}"
            )
        for destination in self.destinations:
            if not destination.passed:
                diagnostics.append(
                    f"destination {destination.name!r} retained poison in "
                    f"{destination.unwritten_elements}/{destination.numel} elements"
                )
        diagnostics.extend(self.destination_identity_errors)
        raise RuntimeError("EXECUTION AUDIT FAILED: " + " | ".join(diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aisp.execution-audit.v1",
            "passed": self.passed,
            "placement": self.placement.to_dict(),
            "destinations": [destination.to_dict() for destination in self.destinations],
            "destination_identity_errors": list(self.destination_identity_errors),
            "limits": [
                "Placement covers dispatcher-visible tensor operands/results on the current thread.",
                "Custom extension internals, background work, host callbacks, and external processes "
                "are outside this audit.",
                "Destination checks prove overwrite coverage only for exact declared buffers.",
                "Destination checks do not establish general allocation or uninitialized-memory "
                "provenance.",
            ],
        }


def audit_callable_once(
    callback: Callable[[], Any],
    *,
    expected_device: str | torch.device,
    destinations: Mapping[str, torch.Tensor] | None = None,
    allowed_host_tensors: Sequence[HostTensorAllowance] = (),
    evidence_limit: int = 64,
) -> ExecutionAuditResult:
    """Audit one callback invocation outside benchmark timing."""

    if not callable(callback):
        raise TypeError("callback must be callable")
    write_guard = DestinationWriteCoverageGuard(destinations) if destinations else None
    if write_guard is not None:
        write_guard.poison()
    placement_audit = TensorOperationPlacementAudit(
        expected_device,
        allowed_host_tensors=allowed_host_tensors,
        evidence_limit=evidence_limit,
    )
    with placement_audit:
        callback()
    destination_results = write_guard.inspect() if write_guard is not None else ()
    return ExecutionAuditResult(
        placement=placement_audit.result(),
        destinations=destination_results,
    )


def _resolve_attribute(instance: Any, attribute_path: str) -> Any:
    if not attribute_path or any(part == "" for part in attribute_path.split(".")):
        raise ValueError("attribute paths must be non-empty dotted names")
    value = instance
    for part in attribute_path.split("."):
        if part.startswith("__"):
            raise ValueError("dunder attribute paths are not supported")
        value = getattr(value, part)
    return value


def _load_target_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_aisp_execution_audit_target", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one fresh benchmark invocation under explicit execution audits."
    )
    parser.add_argument("benchmark_path", type=Path)
    parser.add_argument(
        "--factory",
        default="get_benchmark",
        help="No-argument factory function or class name in the benchmark module.",
    )
    parser.add_argument("--expected-device", required=True)
    parser.add_argument(
        "--destination",
        action="append",
        default=[],
        metavar="ATTRIBUTE",
        help="Exact preallocated tensor attribute to poison and verify; may be repeated.",
    )
    parser.add_argument(
        "--allow-host-tensor",
        action="append",
        default=[],
        metavar="ATTRIBUTE=OPERATOR",
        help=(
            "Allow one exact CPU tensor attribute for one exact dispatcher operator; "
            "repeat for additional operators."
        ),
    )
    parser.add_argument(
        "--target-arg",
        action="append",
        default=[],
        help="Argument exposed to the target module as sys.argv; may be repeated.",
    )
    parser.add_argument("--evidence-limit", type=int, default=64)
    return parser


def _host_allowances(instance: Any, specs: Sequence[str]) -> tuple[HostTensorAllowance, ...]:
    operations_by_attribute: dict[str, list[str]] = {}
    for spec in specs:
        attribute, separator, operator = spec.partition("=")
        if not separator or not attribute.strip() or not operator.strip():
            raise ValueError("--allow-host-tensor requires ATTRIBUTE=OPERATOR")
        operations_by_attribute.setdefault(attribute, []).append(operator)
    return tuple(
        HostTensorAllowance(
            label=attribute,
            tensor=_resolve_attribute(instance, attribute),
            operations=tuple(operations),
        )
        for attribute, operations in operations_by_attribute.items()
    )


def _audit_fresh_benchmark(args: argparse.Namespace) -> tuple[ExecutionAuditResult, dict[str, Any]]:
    benchmark_path = args.benchmark_path.expanduser().resolve()
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"benchmark module does not exist: {benchmark_path}")

    original_argv = sys.argv
    benchmark = None
    primary_error: Exception | None = None
    teardown_error: Exception | None = None
    try:
        sys.argv = [str(benchmark_path), *args.target_arg]
        module = _load_target_module(benchmark_path)
        factory = getattr(module, args.factory)
        if not callable(factory):
            raise TypeError(f"target symbol {args.factory!r} is not callable")
        benchmark = factory()
        for method_name in ("setup", "benchmark_fn", "teardown"):
            if not callable(getattr(benchmark, method_name, None)):
                raise TypeError(f"fresh benchmark must implement callable {method_name}()")
        benchmark.setup()

        destinations: dict[str, torch.Tensor] = {}
        for attribute in args.destination:
            if attribute in destinations:
                raise ValueError(f"duplicate destination attribute: {attribute}")
            destinations[attribute] = _resolve_attribute(benchmark, attribute)
        original_destination_identities = {
            attribute: id(tensor) for attribute, tensor in destinations.items()
        }
        result = audit_callable_once(
            benchmark.benchmark_fn,
            expected_device=args.expected_device,
            destinations=destinations,
            allowed_host_tensors=_host_allowances(benchmark, args.allow_host_tensor),
            evidence_limit=args.evidence_limit,
        )
        identity_errors = tuple(
            f"destination attribute {attribute!r} no longer references the declared tensor"
            for attribute, original_identity in original_destination_identities.items()
            if id(_resolve_attribute(benchmark, attribute)) != original_identity
        )
        result = replace(result, destination_identity_errors=identity_errors)
        metadata = {
            "benchmark_path": str(benchmark_path),
            "factory": args.factory,
            "process_id": os.getpid(),
            "fresh_instance": True,
            "normal_timing_lifecycle_modified": False,
        }
        return result, metadata
    except Exception as error:
        primary_error = error
        raise
    finally:
        if benchmark is not None and callable(getattr(benchmark, "teardown", None)):
            try:
                benchmark.teardown()
            except Exception as error:  # preserve teardown evidence after the primary audit
                teardown_error = error
        sys.argv = original_argv
        if teardown_error is not None:
            detail = (
                f"fresh benchmark teardown failed: {type(teardown_error).__name__}: "
                f"{teardown_error}"
            )
            if primary_error is not None:
                primary_error.add_note(detail)
            else:
                raise RuntimeError(detail) from teardown_error


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for a fresh, standalone audited invocation."""

    args = _build_parser().parse_args(argv)
    try:
        result, metadata = _audit_fresh_benchmark(args)
    except Exception as error:
        payload = {
            "schema": "aisp.execution-audit.v1",
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }
        error_notes = [str(note) for note in getattr(error, "__notes__", ())]
        if error_notes:
            payload["error_notes"] = error_notes
        print(
            json.dumps(payload, sort_keys=True)
        )
        return 1
    payload = result.to_dict()
    payload.update(metadata)
    print(json.dumps(payload, sort_keys=True))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DestinationWriteCoverageEvidence",
    "DestinationWriteCoverageGuard",
    "ExecutionAuditResult",
    "HostTensorAllowance",
    "OperationEvidence",
    "OperationPlacementResult",
    "TensorExtentEvidence",
    "TensorOperationPlacementAudit",
    "audit_callable_once",
    "main",
]
