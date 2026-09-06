"""Fresh full-output results from the NVSHMEM pipeline torchrun worker."""

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

from core.benchmark.verification import InputSignature, PrecisionFlags, coerce_input_signature

NVSHMEM_PIPELINE_RESULT_CALLBACK = "consume_nvshmem_pipeline_child_results"
NVSHMEM_PIPELINE_RESULT_DIR_ENV = "AISP_NVSHMEM_PIPELINE_RESULT_DIR"
NVSHMEM_PIPELINE_RESULT_TOKEN_ENV = "AISP_NVSHMEM_PIPELINE_RESULT_TOKEN"
NVSHMEM_PIPELINE_RESULT_VARIANT_ENV = "AISP_NVSHMEM_PIPELINE_RESULT_VARIANT"
NVSHMEM_PIPELINE_RESULT_LAUNCH_WALL_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS"
NVSHMEM_PIPELINE_RESULT_SCHEMA = "aisp.nvshmem.pipeline-result.v1"

_ConfigurationValue = bool | int | str
_SUPPORTED_VARIANTS = frozenset({"baseline", "optimized"})
_SUPPORTED_TRANSPORTS = frozenset({"nccl", "nvshmem"})


def _normalize_configuration(
    configuration: Mapping[str, _ConfigurationValue],
) -> dict[str, _ConfigurationValue]:
    normalized: dict[str, _ConfigurationValue] = {}
    for key, value in sorted(configuration.items()):
        if not isinstance(key, str) or not key:
            raise TypeError("Pipeline result configuration keys must be non-empty strings")
        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            normalized[key] = int(value)
        elif isinstance(value, str):
            normalized[key] = value
        else:
            raise TypeError("Pipeline result configuration values must be bool, int, or str")
    if not normalized:
        raise ValueError("Pipeline result configuration cannot be empty")
    return normalized


def _canonical_dtype(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _require_finite_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        raise RuntimeError(f"Pipeline result {name} must be a non-empty tensor")
    if (value.is_floating_point() or value.is_complex()) and not bool(
        torch.isfinite(value).all()
    ):
        raise RuntimeError(f"Pipeline result {name} contains non-finite values")
    return value


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


@dataclass(frozen=True)
class NVSHMEMPipelineWorkloadResult:
    """One rank's actual pipeline output and independently evaluated oracle."""

    rank: int
    world_size: int
    iterations: int
    time_per_iter_ms: float
    transport: str
    configuration: Mapping[str, _ConfigurationValue]
    verify_inputs: Mapping[str, torch.Tensor]
    verify_output: torch.Tensor
    reference_output: torch.Tensor
    batch_size: int
    parameter_count: int
    output_tolerance: tuple[float, float]

    def validate(self) -> None:
        if self.world_size < 2 or self.rank not in range(self.world_size):
            raise ValueError("Pipeline result rank/world-size identity is invalid")
        if self.iterations <= 0:
            raise ValueError("Pipeline result iterations must be positive")
        if not math.isfinite(self.time_per_iter_ms) or self.time_per_iter_ms <= 0:
            raise ValueError("Pipeline result timing must be finite and positive")
        if self.transport not in _SUPPORTED_TRANSPORTS:
            raise ValueError(f"Unsupported pipeline transport: {self.transport!r}")
        configuration = _normalize_configuration(self.configuration)
        if configuration.get("transport") != self.transport:
            raise ValueError("Pipeline result transport/configuration mismatch")
        if not self.verify_inputs:
            raise ValueError("Pipeline verification inputs cannot be empty")
        for name, tensor in self.verify_inputs.items():
            if not isinstance(name, str) or not name:
                raise TypeError("Pipeline verification input names must be non-empty strings")
            _require_finite_tensor(f"input {name!r}", tensor)
        output = _require_finite_tensor("output", self.verify_output)
        reference = _require_finite_tensor("reference", self.reference_output)
        if output.shape != reference.shape or output.dtype != reference.dtype:
            raise RuntimeError("Pipeline output/reference shape or dtype differs")
        if self.batch_size <= 0 or self.parameter_count <= 0:
            raise ValueError("Pipeline batch and parameter counts must be positive")
        rtol, atol = (float(value) for value in self.output_tolerance)
        if not all(math.isfinite(value) and value >= 0 for value in (rtol, atol)):
            raise ValueError("Pipeline output tolerance must be finite and nonnegative")
        try:
            torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
        except AssertionError as exc:
            raise RuntimeError("Pipeline full measured output differs from its oracle") from exc

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
                collective_type="pipeline-point-to-point",
                collective_algorithm="1f1b-point-to-point",
                async_completion_policy="wait_for_async_before_timed_close",
                pipeline_stages=int(self.world_size),
                pipeline_stage_boundaries=[(rank, rank) for rank in range(self.world_size)],
            )
        )


class NVSHMEMPipelineChildResultMixin:
    """Require a fresh, matching result from every pipeline worker rank."""

    _nvshmem_pipeline_result_context: dict[str, Any] | None = None
    _nvshmem_pipeline_result_bundle: dict[str, Any] | None = None

    def prepare_nvshmem_pipeline_child_result(
        self,
        *,
        variant: str,
        world_size: int,
        iterations: int,
        configuration: Mapping[str, _ConfigurationValue],
    ) -> dict[str, str]:
        if variant not in _SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported pipeline variant: {variant!r}")
        if world_size < 2:
            raise RuntimeError("SKIPPED: pipeline child results require world_size >= 2")
        if iterations <= 0:
            raise ValueError("Pipeline result iterations must be positive")
        normalized = _normalize_configuration(configuration)
        expected_transport = "nccl" if variant == "baseline" else "nvshmem"
        if normalized.get("transport") != expected_transport:
            raise ValueError("Pipeline variant must use its declared transport")

        result_dir = Path(tempfile.mkdtemp(prefix="aisp-nvshmem-pipeline-result-"))
        context = {
            "result_dir": result_dir,
            "token": uuid.uuid4().hex,
            "variant": variant,
            "world_size": int(world_size),
            "iterations": int(iterations),
            "configuration": normalized,
            "retention": "pending-child-result",
        }
        self._nvshmem_pipeline_result_context = context
        self._nvshmem_pipeline_result_bundle = None
        for attribute in (
            "_subprocess_verify_output",
            "_subprocess_output_tolerance",
            "_subprocess_input_signature",
            "_subprocess_verify_inputs",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return {
            NVSHMEM_PIPELINE_RESULT_DIR_ENV: str(result_dir),
            NVSHMEM_PIPELINE_RESULT_TOKEN_ENV: cast(str, context["token"]),
            NVSHMEM_PIPELINE_RESULT_VARIANT_ENV: variant,
        }

    def consume_nvshmem_pipeline_child_results(
        self,
        *,
        launch_wall_ns: int,
        finish_wall_ns: int,
        returncode: int,
        **_: Any,
    ) -> None:
        context = self._nvshmem_pipeline_result_context
        if context is None:
            raise RuntimeError("Pipeline result callback has no launch context")
        result_dir = cast(Path, context["result_dir"])
        if returncode != 0:
            context["retention"] = "retained-child-failure"
            raise RuntimeError(
                "Pipeline result callback requires a clean child exit; "
                f"artifacts retained at {result_dir}"
            )

        paths = sorted(result_dir.glob("rank-*.pt"))
        expected_world_size = int(context["world_size"])
        if len(paths) != expected_world_size:
            context["retention"] = "retained-incomplete-rank-quorum"
            raise RuntimeError(
                "Pipeline result rank quorum is incomplete: "
                f"expected {expected_world_size}, found {len(paths)}; "
                f"artifacts retained at {result_dir}"
            )

        payloads: list[dict[str, Any]] = []
        seen_ranks: set[int] = set()
        canonical_signature: InputSignature | None = None
        canonical_output: torch.Tensor | None = None
        canonical_reference: torch.Tensor | None = None
        canonical_inputs: dict[str, torch.Tensor] | None = None
        canonical_tolerance: tuple[float, float] | None = None
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Pipeline result must be a regular file: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Invalid pipeline result payload at {path}")
            expected_fields = {
                "schema": NVSHMEM_PIPELINE_RESULT_SCHEMA,
                "token": context["token"],
                "variant": context["variant"],
                "world_size": expected_world_size,
                "iterations": context["iterations"],
                "configuration": context["configuration"],
            }
            for key, expected in expected_fields.items():
                if payload.get(key) != expected:
                    raise RuntimeError(f"Pipeline result {key} mismatch at {path}")
            rank = int(payload.get("rank", -1))
            if rank not in range(expected_world_size) or rank in seen_ranks:
                raise RuntimeError(f"Invalid or duplicate pipeline result rank at {path}")
            if path.name != f"rank-{rank}.pt":
                raise RuntimeError(f"Pipeline result filename/rank mismatch at {path}")
            seen_ranks.add(rank)
            created_wall_ns = int(payload.get("created_wall_ns", 0))
            if not launch_wall_ns <= created_wall_ns <= finish_wall_ns:
                raise RuntimeError(f"Stale pipeline result at {path}")
            if int(payload.get("launch_wall_ns", 0)) != int(launch_wall_ns):
                raise RuntimeError(f"Pipeline result launch identity mismatch at {path}")

            inputs = payload.get("verify_inputs")
            if not isinstance(inputs, dict) or not inputs:
                raise RuntimeError(f"Pipeline verification inputs are missing at {path}")
            typed_inputs = {
                str(name): _require_finite_tensor(f"rank {rank} input {name!r}", tensor)
                for name, tensor in inputs.items()
            }
            output = _require_finite_tensor(f"rank {rank} output", payload.get("verify_output"))
            reference = _require_finite_tensor(
                f"rank {rank} reference", payload.get("reference_output")
            )
            raw_tolerance = payload.get("output_tolerance")
            if not isinstance(raw_tolerance, list | tuple) or len(raw_tolerance) != 2:
                raise RuntimeError(f"Invalid pipeline result tolerance at {path}")
            tolerance = (float(raw_tolerance[0]), float(raw_tolerance[1]))
            if not all(math.isfinite(value) and value >= 0 for value in tolerance):
                raise RuntimeError(f"Invalid pipeline result tolerance at {path}")
            time_per_iter_ms = float(payload.get("time_per_iter_ms", 0.0))
            if not math.isfinite(time_per_iter_ms) or time_per_iter_ms <= 0:
                raise RuntimeError(f"Invalid pipeline result timing at {path}")
            try:
                torch.testing.assert_close(output, reference, rtol=tolerance[0], atol=tolerance[1])
            except AssertionError as exc:
                raise RuntimeError(f"Pipeline output differs from oracle at rank {rank}") from exc
            raw_signature = payload.get("input_signature")
            if not isinstance(raw_signature, dict):
                raise RuntimeError(f"Pipeline input signature is missing at {path}")
            signature = coerce_input_signature(InputSignature.from_dict(raw_signature))
            signature_errors = signature.validate(strict=True)
            if signature_errors:
                raise RuntimeError(
                    f"Invalid pipeline input signature at rank {rank}: {signature_errors[0]}"
                )
            expected_shapes = {
                **{name: tuple(tensor.shape) for name, tensor in typed_inputs.items()},
                "output": tuple(output.shape),
            }
            expected_dtypes = {
                **{name: _canonical_dtype(tensor) for name, tensor in typed_inputs.items()},
                "output": _canonical_dtype(output),
            }
            if signature.shapes != expected_shapes or signature.dtypes != expected_dtypes:
                raise RuntimeError(f"Pipeline signature tensor metadata mismatch at rank {rank}")
            if canonical_signature is None:
                canonical_signature = signature
                canonical_output = output
                canonical_reference = reference
                canonical_inputs = typed_inputs
                canonical_tolerance = tolerance
            else:
                if not canonical_signature.matches(signature):
                    raise RuntimeError("Pipeline input signatures differ across ranks")
                if canonical_tolerance != tolerance:
                    raise RuntimeError("Pipeline tolerances differ across ranks")
                try:
                    torch.testing.assert_close(output, canonical_output, rtol=0, atol=0)
                    torch.testing.assert_close(reference, canonical_reference, rtol=0, atol=0)
                    for name, tensor in typed_inputs.items():
                        torch.testing.assert_close(tensor, canonical_inputs[name], rtol=0, atol=0)
                except (AssertionError, KeyError) as exc:
                    raise RuntimeError("Pipeline full tensors differ across worker ranks") from exc
            payloads.append(payload)

        if any(
            value is None
            for value in (
                canonical_signature,
                canonical_output,
                canonical_reference,
                canonical_inputs,
                canonical_tolerance,
            )
        ):
            raise RuntimeError("Pipeline result quorum contains no usable tensors")
        self._subprocess_verify_inputs = canonical_inputs
        self._subprocess_verify_output = canonical_output
        self._subprocess_input_signature = canonical_signature
        self._subprocess_output_tolerance = canonical_tolerance
        self._nvshmem_pipeline_result_bundle = {"ranks": sorted(payloads, key=lambda item: item["rank"])}
        try:
            shutil.rmtree(result_dir)
        except OSError:
            context["retention"] = "retained-cleanup-failure"
        else:
            context["retention"] = "cleaned-after-success"

    def require_nvshmem_pipeline_child_result(self) -> None:
        if self._nvshmem_pipeline_result_bundle is None:
            context = self._nvshmem_pipeline_result_context
            retained = context.get("result_dir") if context else "unavailable"
            raise RuntimeError(
                "Pipeline verification requires a fresh full-rank worker result; "
                f"retained path: {retained}"
            )

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        if self._nvshmem_pipeline_result_bundle is not None:
            return dict(self._subprocess_verify_inputs)
        return super().get_verify_inputs()  # type: ignore[misc]

    def get_verify_output(self) -> torch.Tensor:
        if self._nvshmem_pipeline_result_bundle is not None:
            return self._subprocess_verify_output.detach().clone()
        return super().get_verify_output()  # type: ignore[misc]

    def get_input_signature(self) -> InputSignature:
        if self._nvshmem_pipeline_result_bundle is not None:
            return self._subprocess_input_signature
        return super().get_input_signature()  # type: ignore[misc]

    def get_output_tolerance(self) -> tuple[float, float]:
        if self._nvshmem_pipeline_result_bundle is not None:
            return self._subprocess_output_tolerance
        return super().get_output_tolerance()  # type: ignore[misc]

    def validate_result(self) -> str | None:
        if self._nvshmem_pipeline_result_bundle is None:
            return "Fresh full-rank pipeline worker output is missing"
        return None


def write_nvshmem_pipeline_child_result(
    result: NVSHMEMPipelineWorkloadResult,
    *,
    variant: str,
) -> bool:
    """Atomically write one rank result when the parent prepared a callback launch."""
    launch_wall_ns = os.environ.get(NVSHMEM_PIPELINE_RESULT_LAUNCH_WALL_NS_ENV)
    if not launch_wall_ns:
        return False
    result_dir_value = os.environ.get(NVSHMEM_PIPELINE_RESULT_DIR_ENV)
    token = os.environ.get(NVSHMEM_PIPELINE_RESULT_TOKEN_ENV)
    expected_variant = os.environ.get(NVSHMEM_PIPELINE_RESULT_VARIANT_ENV)
    if not all((result_dir_value, token, expected_variant)):
        raise RuntimeError("Pipeline result environment is incomplete")
    if variant != expected_variant:
        raise RuntimeError("Pipeline result variant identity mismatch")
    expected_transport = "nccl" if variant == "baseline" else "nvshmem"
    if result.transport != expected_transport:
        raise RuntimeError("Pipeline result variant/transport mismatch")
    result.validate()

    result_dir = Path(cast(str, result_dir_value))
    if result_dir.is_symlink() or not result_dir.is_dir():
        raise RuntimeError("Pipeline result directory must be the prepared directory")
    verify_inputs = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in result.verify_inputs.items()
    }
    payload = {
        "schema": NVSHMEM_PIPELINE_RESULT_SCHEMA,
        "token": token,
        "variant": variant,
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
    "NVSHMEM_PIPELINE_RESULT_CALLBACK",
    "NVSHMEM_PIPELINE_RESULT_DIR_ENV",
    "NVSHMEM_PIPELINE_RESULT_LAUNCH_WALL_NS_ENV",
    "NVSHMEM_PIPELINE_RESULT_SCHEMA",
    "NVSHMEM_PIPELINE_RESULT_TOKEN_ENV",
    "NVSHMEM_PIPELINE_RESULT_VARIANT_ENV",
    "NVSHMEMPipelineChildResultMixin",
    "NVSHMEMPipelineWorkloadResult",
    "write_nvshmem_pipeline_child_result",
]
