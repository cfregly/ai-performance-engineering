"""Fail-closed child-result transport for torchrun training scripts.

This module transports correctness evidence; it does not manufacture it.  An
opted-in child must publish the tensors it consumed, the complete outputs under
comparison, outputs from an independently implemented reference, and a second
input/output pair proving that the result responds to its inputs.  The parent
accepts a bundle only after every expected rank publishes a fresh result and the
declared number of training iterations completed.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from core.benchmark.verification import InputSignature, PrecisionFlags

SCHEMA_VERSION = "aisp.training.child-result.v1"
RESULT_CALLBACK = "consume_training_child_results"
RESULT_DIR_ENV = "AISP_TRAINING_RESULT_DIR"
RUN_ID_ENV = "AISP_TRAINING_RESULT_RUN_ID"
CONTRACT_ENV = "AISP_TRAINING_RESULT_CONTRACT"
WORLD_SIZE_ENV = "AISP_TRAINING_RESULT_WORLD_SIZE"
ITERATIONS_ENV = "AISP_TRAINING_RESULT_ITERATIONS"
LAUNCH_WALL_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS"
LAUNCH_MONOTONIC_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_MONOTONIC_NS"

_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_PAYLOAD_KEYS = {
    "schema_version",
    "run_id",
    "contract",
    "rank",
    "world_size",
    "completed_iterations",
    "parameter_count",
    "torch_seed",
    "pid",
    "launch_wall_ns",
    "launch_monotonic_ns",
    "created_wall_ns",
    "created_monotonic_ns",
    "inputs",
    "outputs",
    "reference_outputs",
    "sensitivity_inputs",
    "sensitivity_outputs",
    "sensitivity_reference_outputs",
}


@dataclass(frozen=True)
class TorchrunChildResultContract:
    """Static workload facts the parent and every child must agree on."""

    profile: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    per_rank_batch_size: int
    parameter_count: int | None
    precision_flags: PrecisionFlags
    output_tolerance: tuple[float, float]
    independent_reference: str
    collective_type: str | None = None
    collective_algorithm: str | None = None
    max_rank_payload_bytes: int = 64 * 1024 * 1024

    def validate(self) -> None:
        if not isinstance(self.profile, str) or not _PROFILE_RE.fullmatch(self.profile):
            raise ValueError("Training child-result profile must be a stable non-empty identifier")
        for label, names in (
            ("input_names", self.input_names),
            ("output_names", self.output_names),
        ):
            if not isinstance(names, tuple) or not names:
                raise ValueError(f"Training child-result {label} must be a non-empty tuple")
            if any(not isinstance(name, str) or not name for name in names):
                raise ValueError(f"Training child-result {label} must contain non-empty strings")
            if len(set(names)) != len(names):
                raise ValueError(f"Training child-result {label} contains duplicates")
        if "completed_iterations" in self.output_names:
            raise ValueError("Training child-result output name 'completed_iterations' is reserved")
        if (
            isinstance(self.per_rank_batch_size, bool)
            or not isinstance(self.per_rank_batch_size, int)
            or self.per_rank_batch_size <= 0
        ):
            raise ValueError("Training child-result per_rank_batch_size must be positive")
        if self.parameter_count is not None and (
            isinstance(self.parameter_count, bool)
            or not isinstance(self.parameter_count, int)
            or self.parameter_count <= 0
        ):
            raise ValueError("Training child-result parameter_count must be None or positive")
        if not isinstance(self.precision_flags, PrecisionFlags):
            raise TypeError("Training child-result precision_flags must be PrecisionFlags")
        if (
            not isinstance(self.output_tolerance, tuple)
            or len(self.output_tolerance) != 2
            or any(
                not math.isfinite(float(value)) or float(value) < 0
                for value in self.output_tolerance
            )
        ):
            raise ValueError(
                "Training child-result output_tolerance must contain two finite non-negative values"
            )
        if (
            not isinstance(self.independent_reference, str)
            or not self.independent_reference.strip()
        ):
            raise ValueError(
                "Training child-result must identify its independent reference implementation"
            )
        for name in ("collective_type", "collective_algorithm"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Training child-result {name} must be None or a non-empty string")
        if (
            isinstance(self.max_rank_payload_bytes, bool)
            or not isinstance(self.max_rank_payload_bytes, int)
            or self.max_rank_payload_bytes <= 0
            or self.max_rank_payload_bytes > 1024 * 1024 * 1024
        ):
            raise ValueError("Training child-result max_rank_payload_bytes must be in (0, 1 GiB]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "profile": self.profile,
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "per_rank_batch_size": self.per_rank_batch_size,
            "parameter_count": self.parameter_count,
            "precision_flags": self.precision_flags.to_dict(),
            "output_tolerance": list(self.output_tolerance),
            "independent_reference": self.independent_reference,
            "collective_type": self.collective_type,
            "collective_algorithm": self.collective_algorithm,
            "max_rank_payload_bytes": self.max_rank_payload_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TorchrunChildResultContract:
        if not isinstance(value, Mapping):
            raise TypeError("Training child-result contract must be a mapping")
        expected = {
            "profile",
            "input_names",
            "output_names",
            "per_rank_batch_size",
            "parameter_count",
            "precision_flags",
            "output_tolerance",
            "independent_reference",
            "collective_type",
            "collective_algorithm",
            "max_rank_payload_bytes",
        }
        if set(value) != expected:
            raise ValueError("Training child-result contract fields do not match the schema")
        tolerance = value["output_tolerance"]
        if not isinstance(tolerance, list | tuple) or len(tolerance) != 2:
            raise ValueError("Training child-result output_tolerance must have two values")
        contract = cls(
            profile=value["profile"],
            input_names=tuple(value["input_names"]),
            output_names=tuple(value["output_names"]),
            per_rank_batch_size=value["per_rank_batch_size"],
            parameter_count=value["parameter_count"],
            precision_flags=PrecisionFlags.from_dict(value["precision_flags"]),
            output_tolerance=(float(tolerance[0]), float(tolerance[1])),
            independent_reference=value["independent_reference"],
            collective_type=value["collective_type"],
            collective_algorithm=value["collective_algorithm"],
            max_rank_payload_bytes=value["max_rank_payload_bytes"],
        )
        contract.validate()
        return contract


def child_result_requested() -> bool:
    """Return whether a parent explicitly requested the shared result protocol."""

    return bool(os.environ.get(RESULT_DIR_ENV))


def _required_environment() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        RESULT_DIR_ENV,
        RUN_ID_ENV,
        CONTRACT_ENV,
        WORLD_SIZE_ENV,
        ITERATIONS_ENV,
        LAUNCH_WALL_NS_ENV,
        LAUNCH_MONOTONIC_NS_ENV,
    ):
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"Training child-result protocol requires {name}")
        result[name] = value
    return result


def _copy_tensor_mapping(
    value: Mapping[str, torch.Tensor],
    *,
    names: tuple[str, ...],
    label: str,
    floating: bool,
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError(f"Training child-result {label} names must equal {list(names)}")
    result: dict[str, torch.Tensor] = {}
    for name in names:
        tensor = value[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Training child-result {label}[{name!r}] must be a tensor")
        copied = tensor.detach().cpu().contiguous().clone()
        if floating and not (copied.is_floating_point() or copied.is_complex()):
            raise TypeError(f"Training child-result {label}[{name!r}] must be floating point")
        if (copied.is_floating_point() or copied.is_complex()) and not bool(
            torch.isfinite(copied).all()
        ):
            raise ValueError(f"Training child-result {label}[{name!r}] contains non-finite values")
        result[name] = copied
    return result


def _assert_same_layout(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    if set(left) != set(right):
        raise ValueError(f"Training child-result {label} tensor names differ")
    for name in left:
        if left[name].shape != right[name].shape or left[name].dtype != right[name].dtype:
            raise ValueError(f"Training child-result {label}[{name!r}] shape or dtype differs")


def _assert_reference(
    actual: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    *,
    tolerance: tuple[float, float],
    label: str,
) -> None:
    _assert_same_layout(actual, reference, label=label)
    for name in actual:
        try:
            torch.testing.assert_close(
                actual[name],
                reference[name],
                rtol=tolerance[0],
                atol=tolerance[1],
            )
        except AssertionError as exc:
            raise RuntimeError(
                f"Training child-result full {label}[{name!r}] differs from its independent "
                f"reference: {exc}"
            ) from exc


def _assert_sensitivity(
    inputs: Mapping[str, torch.Tensor],
    sensitivity_inputs: Mapping[str, torch.Tensor],
    outputs: Mapping[str, torch.Tensor],
    sensitivity_outputs: Mapping[str, torch.Tensor],
) -> None:
    _assert_same_layout(inputs, sensitivity_inputs, label="sensitivity inputs")
    _assert_same_layout(outputs, sensitivity_outputs, label="sensitivity outputs")
    if all(torch.equal(inputs[name], sensitivity_inputs[name]) for name in inputs):
        raise RuntimeError("Training child-result sensitivity inputs are unchanged")
    if all(torch.equal(outputs[name], sensitivity_outputs[name]) for name in outputs):
        raise RuntimeError(
            "Training child-result outputs did not respond to the changed verification input"
        )


def _share_identical_reference_storage(
    actual: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Losslessly pack already-validated CPU references for torch.save.

    The independently computed references have already passed the full checks.
    Only byte-identical tensors share storage; tolerance-close values (including
    differently signed zeros) retain their own bytes. torch.save/load preserves
    all named full tensors, and the parent still validates every reference.
    """

    return {
        name: (
            actual[name]
            if torch.equal(
                actual[name].reshape(-1).view(torch.uint8),
                tensor.reshape(-1).view(torch.uint8),
            )
            else tensor
        )
        for name, tensor in reference.items()
    }


def write_training_child_result(
    *,
    inputs: Mapping[str, torch.Tensor],
    outputs: Mapping[str, torch.Tensor],
    reference_outputs: Mapping[str, torch.Tensor],
    sensitivity_inputs: Mapping[str, torch.Tensor],
    sensitivity_outputs: Mapping[str, torch.Tensor],
    sensitivity_reference_outputs: Mapping[str, torch.Tensor],
    completed_iterations: int,
    parameter_count: int | None = None,
) -> Path:
    """Atomically publish one rank's real tensors under the requested contract."""

    environment = _required_environment()
    try:
        contract = TorchrunChildResultContract.from_dict(json.loads(environment[CONTRACT_ENV]))
        expected_world_size = int(environment[WORLD_SIZE_ENV])
        requested_iterations = int(environment[ITERATIONS_ENV])
        launch_wall_ns = int(environment[LAUNCH_WALL_NS_ENV])
        launch_monotonic_ns = int(environment[LAUNCH_MONOTONIC_NS_ENV])
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Training child-result environment is malformed") from exc
    if world_size != expected_world_size or rank < 0 or rank >= world_size:
        raise RuntimeError("Training child-result rank topology does not match the parent contract")
    if (
        isinstance(completed_iterations, bool)
        or not isinstance(completed_iterations, int)
        or completed_iterations <= 0
        or completed_iterations > requested_iterations
    ):
        raise RuntimeError(
            "Training child-result completed iteration count must be within the requested maximum: "
            f"completed={completed_iterations!r}, requested_max={requested_iterations}"
        )
    observed_parameter_count = (
        contract.parameter_count if parameter_count is None else parameter_count
    )
    if (
        isinstance(observed_parameter_count, bool)
        or not isinstance(observed_parameter_count, int)
        or observed_parameter_count <= 0
    ):
        raise RuntimeError("Training child-result parameter_count must be a positive integer")
    if (
        contract.parameter_count is not None
        and observed_parameter_count != contract.parameter_count
    ):
        raise RuntimeError("Training child-result parameter_count differs from its contract")

    primary_inputs = _copy_tensor_mapping(
        inputs, names=contract.input_names, label="inputs", floating=False
    )
    changed_inputs = _copy_tensor_mapping(
        sensitivity_inputs,
        names=contract.input_names,
        label="sensitivity_inputs",
        floating=False,
    )
    primary_outputs = _copy_tensor_mapping(
        outputs, names=contract.output_names, label="outputs", floating=True
    )
    primary_reference = _copy_tensor_mapping(
        reference_outputs,
        names=contract.output_names,
        label="reference_outputs",
        floating=True,
    )
    changed_outputs = _copy_tensor_mapping(
        sensitivity_outputs,
        names=contract.output_names,
        label="sensitivity_outputs",
        floating=True,
    )
    changed_reference = _copy_tensor_mapping(
        sensitivity_reference_outputs,
        names=contract.output_names,
        label="sensitivity_reference_outputs",
        floating=True,
    )
    _assert_reference(
        primary_outputs,
        primary_reference,
        tolerance=contract.output_tolerance,
        label="outputs",
    )
    _assert_reference(
        changed_outputs,
        changed_reference,
        tolerance=contract.output_tolerance,
        label="sensitivity_outputs",
    )
    _assert_sensitivity(primary_inputs, changed_inputs, primary_outputs, changed_outputs)
    primary_reference = _share_identical_reference_storage(primary_outputs, primary_reference)
    changed_reference = _share_identical_reference_storage(changed_outputs, changed_reference)

    result_dir = Path(environment[RESULT_DIR_ENV])
    if not result_dir.is_dir() or result_dir.is_symlink():
        raise RuntimeError("Training child-result directory must be an existing regular directory")
    destination = result_dir / f"rank-{rank}.pt"
    temporary = result_dir / f"rank-{rank}.pt.tmp-{os.getpid()}"
    if destination.exists() or temporary.exists():
        raise RuntimeError(f"Refusing to overwrite training child result for rank {rank}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": environment[RUN_ID_ENV],
        "contract": contract.to_dict(),
        "rank": rank,
        "world_size": world_size,
        "completed_iterations": completed_iterations,
        "parameter_count": observed_parameter_count,
        "torch_seed": int(torch.initial_seed()),
        "pid": os.getpid(),
        "launch_wall_ns": launch_wall_ns,
        "launch_monotonic_ns": launch_monotonic_ns,
        "created_wall_ns": time.time_ns(),
        "created_monotonic_ns": time.monotonic_ns(),
        "inputs": primary_inputs,
        "outputs": primary_outputs,
        "reference_outputs": primary_reference,
        "sensitivity_inputs": changed_inputs,
        "sensitivity_outputs": changed_outputs,
        "sensitivity_reference_outputs": changed_reference,
    }
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    payload_bytes = temporary.stat().st_size
    if payload_bytes > contract.max_rank_payload_bytes:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Training child-result payload exceeds its declared size limit: "
            f"{payload_bytes} bytes > {contract.max_rank_payload_bytes} bytes"
        )
    os.replace(temporary, destination)
    return destination


def validate_training_child_result_bundle(
    result_dir: Path,
    *,
    contract: TorchrunChildResultContract,
    run_id: str,
    world_size: int,
    requested_iterations: int,
    launch_wall_ns: int,
    launch_monotonic_ns: int,
    finish_wall_ns: int,
    finish_monotonic_ns: int,
) -> dict[str, Any]:
    """Validate and combine a fresh complete rank quorum."""

    contract.validate()
    for label, value in (
        ("world_size", world_size),
        ("requested_iterations", requested_iterations),
        ("launch_wall_ns", launch_wall_ns),
        ("launch_monotonic_ns", launch_monotonic_ns),
        ("finish_wall_ns", finish_wall_ns),
        ("finish_monotonic_ns", finish_monotonic_ns),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Training child-result {label} must be a positive integer")
    if finish_wall_ns < launch_wall_ns or finish_monotonic_ns < launch_monotonic_ns:
        raise ValueError("Training child-result finish time precedes launch time")
    entries = sorted(result_dir.iterdir())
    paths = [path for path in entries if re.fullmatch(r"rank-[0-9]+\.pt", path.name)]
    if len(entries) != len(paths):
        raise RuntimeError("Training child-result directory contains an unexpected artifact")
    if len(paths) != world_size:
        raise RuntimeError(
            "Training child-result rank quorum is incomplete: "
            f"expected {world_size}, found {len(paths)}"
        )
    payloads: dict[int, dict[str, Any]] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Training child result must be a regular file: {path}")
        if path.stat().st_size > contract.max_rank_payload_bytes:
            raise RuntimeError(f"Training child result exceeds its size limit: {path}")
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict) or set(loaded) != _PAYLOAD_KEYS:
            raise RuntimeError(f"Training child-result payload has invalid fields: {path}")
        if loaded["schema_version"] != SCHEMA_VERSION or loaded["run_id"] != run_id:
            raise RuntimeError(f"Training child-result identity mismatch: {path}")
        observed_contract = TorchrunChildResultContract.from_dict(loaded["contract"])
        if observed_contract != contract:
            raise RuntimeError(f"Training child-result contract mismatch: {path}")
        rank = loaded["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 or rank >= world_size:
            raise RuntimeError(f"Training child-result rank is invalid: {path}")
        if rank in payloads or path.name != f"rank-{rank}.pt":
            raise RuntimeError(f"Training child-result rank is duplicate or misnamed: {path}")
        if (
            isinstance(loaded["world_size"], bool)
            or not isinstance(loaded["world_size"], int)
            or loaded["world_size"] != world_size
        ):
            raise RuntimeError(f"Training child-result world size mismatch: {path}")
        if (
            isinstance(loaded["completed_iterations"], bool)
            or not isinstance(loaded["completed_iterations"], int)
            or loaded["completed_iterations"] <= 0
            or loaded["completed_iterations"] > requested_iterations
        ):
            raise RuntimeError(f"Training child-result iteration count mismatch: {path}")
        if (
            isinstance(loaded["parameter_count"], bool)
            or not isinstance(loaded["parameter_count"], int)
            or loaded["parameter_count"] <= 0
            or (
                contract.parameter_count is not None
                and loaded["parameter_count"] != contract.parameter_count
            )
        ):
            raise RuntimeError(f"Training child-result parameter count mismatch: {path}")
        if (
            isinstance(loaded["torch_seed"], bool)
            or not isinstance(loaded["torch_seed"], int)
            or loaded["torch_seed"] < 0
        ):
            raise RuntimeError(f"Training child-result torch seed is invalid: {path}")
        if not isinstance(loaded["pid"], int) or loaded["pid"] <= 0:
            raise RuntimeError(f"Training child-result pid is invalid: {path}")
        if (
            loaded["launch_wall_ns"] != launch_wall_ns
            or loaded["launch_monotonic_ns"] != launch_monotonic_ns
        ):
            raise RuntimeError(f"Training child-result launch identity mismatch: {path}")
        for time_name in (
            "launch_wall_ns",
            "launch_monotonic_ns",
            "created_wall_ns",
            "created_monotonic_ns",
        ):
            observed_time = loaded[time_name]
            if (
                isinstance(observed_time, bool)
                or not isinstance(observed_time, int)
                or observed_time <= 0
            ):
                raise RuntimeError(f"Training child-result {time_name} is invalid: {path}")
        if not launch_wall_ns <= loaded["created_wall_ns"] <= finish_wall_ns:
            raise RuntimeError(f"Training child-result wall-clock freshness failed: {path}")
        if not launch_monotonic_ns <= loaded["created_monotonic_ns"] <= finish_monotonic_ns:
            raise RuntimeError(f"Training child-result monotonic freshness failed: {path}")

        primary_inputs = _copy_tensor_mapping(
            loaded["inputs"], names=contract.input_names, label="inputs", floating=False
        )
        for name, tensor in primary_inputs.items():
            if tensor.ndim < 1 or tensor.shape[0] != contract.per_rank_batch_size:
                raise RuntimeError(
                    "Training child-result input batch size differs from its contract: "
                    f"inputs[{name!r}] at {path}"
                )
        changed_inputs = _copy_tensor_mapping(
            loaded["sensitivity_inputs"],
            names=contract.input_names,
            label="sensitivity_inputs",
            floating=False,
        )
        primary_outputs = _copy_tensor_mapping(
            loaded["outputs"], names=contract.output_names, label="outputs", floating=True
        )
        primary_reference = _copy_tensor_mapping(
            loaded["reference_outputs"],
            names=contract.output_names,
            label="reference_outputs",
            floating=True,
        )
        changed_outputs = _copy_tensor_mapping(
            loaded["sensitivity_outputs"],
            names=contract.output_names,
            label="sensitivity_outputs",
            floating=True,
        )
        changed_reference = _copy_tensor_mapping(
            loaded["sensitivity_reference_outputs"],
            names=contract.output_names,
            label="sensitivity_reference_outputs",
            floating=True,
        )
        _assert_reference(
            primary_outputs,
            primary_reference,
            tolerance=contract.output_tolerance,
            label="outputs",
        )
        _assert_reference(
            changed_outputs,
            changed_reference,
            tolerance=contract.output_tolerance,
            label="sensitivity_outputs",
        )
        _assert_sensitivity(primary_inputs, changed_inputs, primary_outputs, changed_outputs)
        loaded["inputs"] = primary_inputs
        loaded["outputs"] = primary_outputs
        payloads[rank] = loaded

    ordered = [payloads[rank] for rank in range(world_size)]
    completed_iterations = {payload["completed_iterations"] for payload in ordered}
    if len(completed_iterations) != 1:
        raise RuntimeError("Training child-result completed iteration counts differ across ranks")
    observed_parameter_counts = {payload["parameter_count"] for payload in ordered}
    if len(observed_parameter_counts) != 1:
        raise RuntimeError("Training child-result parameter counts differ across ranks")
    completed_iteration_count = completed_iterations.pop()
    observed_parameter_count = observed_parameter_counts.pop()
    torch_seeds = {payload["torch_seed"] for payload in ordered}
    if len(torch_seeds) != 1:
        raise RuntimeError("Training child-result torch seeds differ across ranks")
    observed_torch_seed = torch_seeds.pop()

    # Keep each rank's complete tensors under a stable key.  Independently
    # padded data-loader batches can have different sequence lengths and must
    # not be truncated or padded merely to make parent-side stacking possible.
    verify_inputs = {
        f"rank-{rank}:{name}": payload["inputs"][name]
        for rank, payload in enumerate(ordered)
        for name in contract.input_names
    }
    verify_output = {
        f"rank-{rank}:{name}": payload["outputs"][name]
        for rank, payload in enumerate(ordered)
        for name in contract.output_names
    }
    # Encode the count in tensor shape so workload comparison requires exact
    # equality even when the numerical outputs use a nonzero BF16 tolerance.
    verify_output["completed_iterations"] = torch.ones(
        completed_iteration_count, dtype=torch.float32
    )
    # InputSignature names are matched directly against get_verify_inputs().
    # Output tensors and the completed-step proof belong to get_verify_output()
    # and are compared by the output verifier, not declared as inputs.
    shapes = {name: tuple(tensor.shape) for name, tensor in verify_inputs.items()}
    dtypes = {name: str(tensor.dtype) for name, tensor in verify_inputs.items()}
    signature = InputSignature(
        shapes=shapes,
        dtypes=dtypes,
        batch_size=contract.per_rank_batch_size * world_size,
        parameter_count=observed_parameter_count,
        precision_flags=PrecisionFlags.from_dict(contract.precision_flags.to_dict()),
        world_size=world_size,
        ranks=list(range(world_size)),
        per_rank_batch_size=contract.per_rank_batch_size,
        collective_type=contract.collective_type,
        collective_algorithm=contract.collective_algorithm,
    )
    return {
        "contract": contract,
        "payloads": ordered,
        "verify_inputs": verify_inputs,
        "verify_output": verify_output,
        "input_signature": signature,
        "output_tolerance": contract.output_tolerance,
        "completed_iterations": completed_iteration_count,
        "torch_seed": observed_torch_seed,
    }


__all__ = [
    "CONTRACT_ENV",
    "ITERATIONS_ENV",
    "LAUNCH_MONOTONIC_NS_ENV",
    "LAUNCH_WALL_NS_ENV",
    "RESULT_CALLBACK",
    "RESULT_DIR_ENV",
    "RUN_ID_ENV",
    "SCHEMA_VERSION",
    "TorchrunChildResultContract",
    "WORLD_SIZE_ENV",
    "child_result_requested",
    "validate_training_child_result_bundle",
    "write_training_child_result",
]
