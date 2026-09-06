"""Fresh full-output transport for chapter 4 NVSHMEM torchrun workers."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from core.benchmark.verification import (
    InputSignature,
    PrecisionFlags,
    coerce_input_signature,
)

NVSHMEM_CHILD_RESULT_CALLBACK = "consume_nvshmem_child_results"
NVSHMEM_CHILD_RESULT_DIR_ENV = "AISP_NVSHMEM_CHILD_RESULT_DIR"
NVSHMEM_CHILD_RESULT_TOKEN_ENV = "AISP_NVSHMEM_CHILD_RESULT_TOKEN"
NVSHMEM_CHILD_RESULT_VARIANT_ENV = "AISP_NVSHMEM_CHILD_RESULT_VARIANT"
NVSHMEM_CHILD_RESULT_WORKLOAD_ENV = "AISP_NVSHMEM_CHILD_RESULT_WORKLOAD"
NVSHMEM_CHILD_RESULT_LAUNCH_WALL_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS"
NVSHMEM_CHILD_RESULT_SCHEMA = "aisp.nvshmem.child-result.v1"

_SUPPORTED_VARIANTS = frozenset({"baseline", "optimized"})
_SUPPORTED_WORKLOADS = frozenset(
    {"training-example", "training-patterns", "collective", "symmetric-ring"}
)
_ConfigurationValue = bool | int | str


def _normalize_configuration(
    configuration: Mapping[str, _ConfigurationValue],
) -> dict[str, _ConfigurationValue]:
    normalized: dict[str, _ConfigurationValue] = {}
    for key, value in sorted(configuration.items()):
        if not isinstance(key, str) or not key:
            raise TypeError("NVSHMEM child-result configuration keys must be non-empty strings")
        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            normalized[key] = int(value)
        elif isinstance(value, str):
            normalized[key] = value
        else:
            raise TypeError(
                "NVSHMEM child-result configuration values must be bool, int, or str"
            )
    if not normalized:
        raise ValueError("NVSHMEM child-result configuration cannot be empty")
    return normalized


def _canonical_dtype(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _precision_flags(dtype: torch.dtype) -> PrecisionFlags:
    return PrecisionFlags(
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        fp8=str(dtype).startswith("torch.float8"),
        tf32=(
            dtype == torch.float32
            and torch.cuda.is_available()
            and bool(torch.backends.cuda.matmul.allow_tf32)
        ),
    )


def _require_finite_tensor(name: str, tensor: Any) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        raise RuntimeError(f"NVSHMEM child-result {name} must be a non-empty tensor")
    if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
        torch.isfinite(tensor).all()
    ):
        raise RuntimeError(f"NVSHMEM child-result {name} contains non-finite values")
    return tensor


@dataclass(frozen=True)
class NVSHMEMWorkloadResult:
    """Actual post-timing worker tensors and their independently computed oracle."""

    workload: str
    rank: int
    world_size: int
    iterations: int
    time_per_iter_ms: float
    configuration: Mapping[str, _ConfigurationValue]
    verify_inputs: Mapping[str, torch.Tensor]
    verify_output: torch.Tensor
    reference_output: torch.Tensor
    batch_size: int
    parameter_count: int
    collective_type: str
    output_tolerance: tuple[float, float]

    def validate(self) -> None:
        if self.workload not in _SUPPORTED_WORKLOADS:
            raise ValueError(f"Unsupported NVSHMEM workload: {self.workload!r}")
        if self.world_size < 2 or self.rank not in range(self.world_size):
            raise ValueError("NVSHMEM worker rank/world_size identity is invalid")
        if self.iterations <= 0:
            raise ValueError("NVSHMEM worker iterations must be positive")
        if not math.isfinite(self.time_per_iter_ms) or self.time_per_iter_ms <= 0:
            raise ValueError("NVSHMEM worker iteration timing must be finite and positive")
        _normalize_configuration(self.configuration)
        if not self.verify_inputs:
            raise ValueError("NVSHMEM worker verification inputs cannot be empty")
        for name, tensor in self.verify_inputs.items():
            if not isinstance(name, str) or not name:
                raise TypeError("NVSHMEM worker verification input names must be non-empty")
            _require_finite_tensor(f"input {name!r}", tensor)
        output = _require_finite_tensor("output", self.verify_output)
        reference = _require_finite_tensor("reference", self.reference_output)
        if output.shape != reference.shape or output.dtype != reference.dtype:
            raise RuntimeError("NVSHMEM worker output/reference shape or dtype mismatch")
        if self.batch_size <= 0 or self.parameter_count < 0:
            raise ValueError("NVSHMEM worker batch and parameter counts are invalid")
        if not self.collective_type:
            raise ValueError("NVSHMEM worker collective type must be declared")
        rtol, atol = self.output_tolerance
        if not all(math.isfinite(float(value)) and float(value) >= 0 for value in (rtol, atol)):
            raise ValueError("NVSHMEM worker output tolerance must be finite and nonnegative")
        try:
            torch.testing.assert_close(output, reference, rtol=float(rtol), atol=float(atol))
        except AssertionError as exc:
            raise RuntimeError(
                f"NVSHMEM {self.workload} full measured output differs from its oracle"
            ) from exc

    def input_signature(self) -> InputSignature:
        self.validate()
        shapes = {name: tuple(tensor.shape) for name, tensor in self.verify_inputs.items()}
        dtypes = {name: _canonical_dtype(tensor) for name, tensor in self.verify_inputs.items()}
        shapes["output"] = tuple(self.verify_output.shape)
        dtypes["output"] = _canonical_dtype(self.verify_output)
        return coerce_input_signature(
            InputSignature(
                shapes=shapes,
                dtypes=dtypes,
                batch_size=int(self.batch_size),
                parameter_count=int(self.parameter_count),
                precision_flags=_precision_flags(self.verify_output.dtype),
                world_size=int(self.world_size),
                ranks=list(range(self.world_size)),
                per_rank_batch_size=int(self.batch_size),
                collective_type=self.collective_type,
            )
        )


class NVSHMEMChildResultMixin:
    """Require a fresh, complete rank quorum from the measured torchrun child."""

    _nvshmem_child_result_context: dict[str, Any] | None = None
    _nvshmem_child_result_bundle: dict[str, Any] | None = None

    def prepare_nvshmem_child_result(
        self,
        *,
        variant: str,
        workload: str,
        world_size: int,
        iterations: int,
        configuration: Mapping[str, _ConfigurationValue],
    ) -> dict[str, str]:
        if variant not in _SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported NVSHMEM variant: {variant!r}")
        if workload not in _SUPPORTED_WORKLOADS:
            raise ValueError(f"Unsupported NVSHMEM workload: {workload!r}")
        if world_size < 2:
            raise RuntimeError("SKIPPED: NVSHMEM child results require world_size >= 2")
        if iterations <= 0:
            raise ValueError("NVSHMEM child-result iteration count must be positive")

        result_dir = Path(tempfile.mkdtemp(prefix="aisp-nvshmem-result-"))
        token = uuid.uuid4().hex
        self._nvshmem_child_result_context = {
            "result_dir": result_dir,
            "token": token,
            "variant": variant,
            "workload": workload,
            "world_size": int(world_size),
            "iterations": int(iterations),
            "configuration": _normalize_configuration(configuration),
            "retention": "pending-child-result",
        }
        self._nvshmem_child_result_bundle = None
        for attribute in (
            "_subprocess_verify_output",
            "_subprocess_output_tolerance",
            "_subprocess_input_signature",
            "_subprocess_verify_inputs",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return {
            NVSHMEM_CHILD_RESULT_DIR_ENV: str(result_dir),
            NVSHMEM_CHILD_RESULT_TOKEN_ENV: token,
            NVSHMEM_CHILD_RESULT_VARIANT_ENV: variant,
            NVSHMEM_CHILD_RESULT_WORKLOAD_ENV: workload,
        }

    def consume_nvshmem_child_results(
        self,
        *,
        launch_wall_ns: int,
        finish_wall_ns: int,
        returncode: int,
        **_: Any,
    ) -> None:
        context = self._nvshmem_child_result_context
        if context is None:
            raise RuntimeError("NVSHMEM child-result callback has no launch context")
        result_dir = cast(Path, context["result_dir"])
        if returncode != 0:
            context["retention"] = "retained-child-failure"
            raise RuntimeError(
                "NVSHMEM child-result callback requires a clean child exit; "
                f"artifacts retained at {result_dir}"
            )

        expected_world_size = int(context["world_size"])
        paths = sorted(result_dir.glob("rank-*.pt"))
        if len(paths) != expected_world_size:
            context["retention"] = "retained-incomplete-rank-quorum"
            raise RuntimeError(
                "NVSHMEM child-result rank quorum is incomplete: "
                f"expected {expected_world_size}, found {len(paths)}; "
                f"artifacts retained at {result_dir}"
            )

        payloads: list[dict[str, Any]] = []
        seen_ranks: set[int] = set()
        signature: InputSignature | None = None
        tolerance: tuple[float, float] | None = None
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"NVSHMEM child result must be a regular file: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Invalid NVSHMEM child-result payload at {path}")
            for key in ("schema", "token", "variant", "workload"):
                expected = (
                    NVSHMEM_CHILD_RESULT_SCHEMA if key == "schema" else context[key]
                )
                if payload.get(key) != expected:
                    raise RuntimeError(f"NVSHMEM child-result {key} mismatch at {path}")
            if int(payload.get("world_size", -1)) != expected_world_size:
                raise RuntimeError(f"NVSHMEM child-result world-size mismatch at {path}")
            if int(payload.get("iterations", -1)) != int(context["iterations"]):
                raise RuntimeError(f"NVSHMEM child-result iteration mismatch at {path}")
            if payload.get("configuration") != context["configuration"]:
                raise RuntimeError(f"NVSHMEM child-result configuration mismatch at {path}")

            rank = int(payload.get("rank", -1))
            if rank not in range(expected_world_size) or rank in seen_ranks:
                raise RuntimeError(f"Invalid or duplicate NVSHMEM child-result rank at {path}")
            if path.name != f"rank-{rank}.pt":
                raise RuntimeError(f"NVSHMEM child-result filename/rank mismatch at {path}")
            seen_ranks.add(rank)
            created_wall_ns = int(payload.get("created_wall_ns", 0))
            if not launch_wall_ns <= created_wall_ns <= finish_wall_ns:
                raise RuntimeError(f"Stale NVSHMEM child-result payload at {path}")
            if int(payload.get("launch_wall_ns", 0)) != int(launch_wall_ns):
                raise RuntimeError(f"NVSHMEM child-result launch identity mismatch at {path}")

            verify_inputs = payload.get("verify_inputs")
            if not isinstance(verify_inputs, dict) or not verify_inputs:
                raise RuntimeError(f"NVSHMEM child-result inputs are missing at {path}")
            for name, tensor in verify_inputs.items():
                _require_finite_tensor(f"rank {rank} input {name!r}", tensor)
            output = _require_finite_tensor(f"rank {rank} output", payload.get("verify_output"))
            reference = _require_finite_tensor(
                f"rank {rank} reference", payload.get("reference_output")
            )
            if output.shape != reference.shape or output.dtype != reference.dtype:
                raise RuntimeError(f"NVSHMEM output/reference mismatch at rank {rank}")

            raw_tolerance = payload.get("output_tolerance")
            if not isinstance(raw_tolerance, tuple | list) or len(raw_tolerance) != 2:
                raise RuntimeError(f"Invalid NVSHMEM output tolerance at {path}")
            rank_tolerance = (float(raw_tolerance[0]), float(raw_tolerance[1]))
            if not all(math.isfinite(value) and value >= 0 for value in rank_tolerance):
                raise RuntimeError(f"Invalid NVSHMEM output tolerance at {path}")
            if tolerance is None:
                tolerance = rank_tolerance
            elif tolerance != rank_tolerance:
                raise RuntimeError("NVSHMEM child-result tolerances differ across ranks")
            try:
                torch.testing.assert_close(
                    output,
                    reference,
                    rtol=rank_tolerance[0],
                    atol=rank_tolerance[1],
                )
            except AssertionError as exc:
                raise RuntimeError(
                    f"NVSHMEM full measured output differs from its oracle at rank {rank}"
                ) from exc

            rank_signature = coerce_input_signature(payload.get("input_signature"))
            signature_errors = rank_signature.validate(strict=True)
            if signature_errors:
                raise RuntimeError(
                    f"Invalid NVSHMEM input signature at rank {rank}: {signature_errors[0]}"
                )
            expected_shapes = {
                **{name: tuple(tensor.shape) for name, tensor in verify_inputs.items()},
                "output": tuple(output.shape),
            }
            expected_dtypes = {
                **{name: _canonical_dtype(tensor) for name, tensor in verify_inputs.items()},
                "output": _canonical_dtype(output),
            }
            if rank_signature.shapes != expected_shapes or rank_signature.dtypes != expected_dtypes:
                raise RuntimeError(f"NVSHMEM signature tensor metadata mismatch at rank {rank}")
            if signature is None:
                signature = rank_signature
            elif not signature.matches(rank_signature):
                raise RuntimeError("NVSHMEM child-result input signatures differ across ranks")
            payloads.append(payload)

        if signature is None or tolerance is None:
            raise RuntimeError("NVSHMEM child-result signature or tolerance is missing")
        payloads.sort(key=lambda payload: int(payload["rank"]))
        aggregate_inputs: dict[str, torch.Tensor] = {}
        aggregate_outputs: list[torch.Tensor] = []
        for payload in payloads:
            rank = int(payload["rank"])
            for name, tensor in cast(dict[str, torch.Tensor], payload["verify_inputs"]).items():
                aggregate_inputs[f"rank_{rank}_{name}"] = tensor
            aggregate_outputs.append(cast(torch.Tensor, payload["verify_output"]).reshape(-1))
        aggregate_output = torch.cat(aggregate_outputs)
        aggregate_signature = coerce_input_signature(
            InputSignature(
                shapes={
                    **{name: tuple(tensor.shape) for name, tensor in aggregate_inputs.items()},
                    "output": tuple(aggregate_output.shape),
                },
                dtypes={
                    **{name: _canonical_dtype(tensor) for name, tensor in aggregate_inputs.items()},
                    "output": _canonical_dtype(aggregate_output),
                },
                batch_size=int(signature.batch_size) * expected_world_size,
                parameter_count=int(signature.parameter_count),
                precision_flags=signature.precision_flags,
                world_size=expected_world_size,
                ranks=list(range(expected_world_size)),
                per_rank_batch_size=int(signature.batch_size),
                collective_type=signature.collective_type,
            )
        )

        self._subprocess_verify_inputs = aggregate_inputs
        self._subprocess_verify_output = aggregate_output
        self._subprocess_output_tolerance = tolerance
        self._subprocess_input_signature = aggregate_signature
        self._nvshmem_child_result_bundle = {"ranks": payloads}
        try:
            shutil.rmtree(result_dir)
        except OSError:
            context["retention"] = "retained-cleanup-failure"
        else:
            context["retention"] = "cleaned-after-success"

    def require_nvshmem_child_result(self) -> None:
        if self._nvshmem_child_result_bundle is None:
            context = self._nvshmem_child_result_context
            retained = context.get("result_dir") if context else "unavailable"
            raise RuntimeError(
                "NVSHMEM verification requires a fresh full-rank worker result; "
                f"retained path: {retained}"
            )

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        if self._nvshmem_child_result_bundle is not None:
            return dict(self._subprocess_verify_inputs)
        return super().get_verify_inputs()  # type: ignore[misc]

    def get_verify_output(self) -> torch.Tensor:
        if self._nvshmem_child_result_bundle is not None:
            return self._subprocess_verify_output.detach().clone()
        return super().get_verify_output()  # type: ignore[misc]

    def get_input_signature(self) -> InputSignature:
        if self._nvshmem_child_result_bundle is not None:
            return self._subprocess_input_signature
        return super().get_input_signature()  # type: ignore[misc]

    def get_output_tolerance(self) -> tuple[float, float]:
        if self._nvshmem_child_result_bundle is not None:
            return self._subprocess_output_tolerance
        return super().get_output_tolerance()  # type: ignore[misc]

    def validate_result(self) -> str | None:
        if self._nvshmem_child_result_bundle is None:
            return "Fresh full-rank NVSHMEM worker output is missing"
        return None


def write_nvshmem_child_result(
    result: NVSHMEMWorkloadResult,
    *,
    variant: str,
) -> bool:
    """Atomically write one measured rank result when a callback launch owns it."""
    launch_wall_ns = os.environ.get(NVSHMEM_CHILD_RESULT_LAUNCH_WALL_NS_ENV)
    if not launch_wall_ns:
        return False
    result_dir_value = os.environ.get(NVSHMEM_CHILD_RESULT_DIR_ENV)
    token = os.environ.get(NVSHMEM_CHILD_RESULT_TOKEN_ENV)
    expected_variant = os.environ.get(NVSHMEM_CHILD_RESULT_VARIANT_ENV)
    expected_workload = os.environ.get(NVSHMEM_CHILD_RESULT_WORKLOAD_ENV)
    if not all((result_dir_value, token, expected_variant, expected_workload)):
        raise RuntimeError("NVSHMEM child-result environment is incomplete")
    if variant != expected_variant or result.workload != expected_workload:
        raise RuntimeError("NVSHMEM child-result variant/workload identity mismatch")
    result.validate()

    result_dir = Path(cast(str, result_dir_value))
    if result_dir.is_symlink() or not result_dir.is_dir():
        raise RuntimeError("NVSHMEM child-result directory must be the prepared directory")
    verify_inputs = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in result.verify_inputs.items()
    }
    payload = {
        "schema": NVSHMEM_CHILD_RESULT_SCHEMA,
        "token": token,
        "variant": variant,
        "workload": result.workload,
        "rank": int(result.rank),
        "world_size": int(result.world_size),
        "iterations": int(result.iterations),
        "time_per_iter_ms": float(result.time_per_iter_ms),
        "configuration": _normalize_configuration(result.configuration),
        "launch_wall_ns": int(launch_wall_ns),
        "created_wall_ns": time.time_ns(),
        "verify_inputs": verify_inputs,
        "verify_output": result.verify_output.detach().to(device="cpu").contiguous(),
        "reference_output": result.reference_output.detach().to(device="cpu").contiguous(),
        "input_signature": result.input_signature().to_dict(),
        "output_tolerance": list(result.output_tolerance),
    }
    temporary_path = result_dir / f".rank-{result.rank}-{os.getpid()}.tmp"
    destination = result_dir / f"rank-{result.rank}.pt"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, destination)
    return True


__all__ = [
    "NVSHMEM_CHILD_RESULT_CALLBACK",
    "NVSHMEM_CHILD_RESULT_DIR_ENV",
    "NVSHMEM_CHILD_RESULT_LAUNCH_WALL_NS_ENV",
    "NVSHMEM_CHILD_RESULT_SCHEMA",
    "NVSHMEM_CHILD_RESULT_TOKEN_ENV",
    "NVSHMEM_CHILD_RESULT_VARIANT_ENV",
    "NVSHMEM_CHILD_RESULT_WORKLOAD_ENV",
    "NVSHMEMChildResultMixin",
    "NVSHMEMWorkloadResult",
    "write_nvshmem_child_result",
]
