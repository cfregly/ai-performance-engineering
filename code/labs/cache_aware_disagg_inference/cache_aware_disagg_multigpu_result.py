"""Fresh child-result transport for cache-aware disaggregated inference."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, cast

import torch

from core.benchmark.verification import InputSignature, PrecisionFlags

CACHE_AWARE_RESULT_CALLBACK = "consume_cache_aware_child_results"
CACHE_AWARE_RESULT_DIR_ENV = "AISP_CACHE_AWARE_DISAGG_RESULT_DIR"
CACHE_AWARE_RESULT_TOKEN_ENV = "AISP_CACHE_AWARE_DISAGG_RESULT_TOKEN"
CACHE_AWARE_VARIANT_ENV = "AISP_CACHE_AWARE_DISAGG_VARIANT"
CACHE_AWARE_METRICS_PATH_ENV = "AISP_CACHE_AWARE_DISAGG_METRICS_PATH"
CACHE_AWARE_LAUNCH_WALL_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS"
CACHE_AWARE_RESULT_SCHEMA = "aisp.cache-aware-disagg.child-result.v1"
CACHE_AWARE_OUTPUT_TOLERANCE = (1e-2, 1e-2)


def make_cache_aware_result_config(
    *,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    requests_per_rank: int,
    context_window: int,
    chunk_size: int,
    decode_tokens: int,
    warm_request_ratio: float,
    warm_prefix_ratio: float,
    prefill_ranks: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Build the primitive workload identity stored in every rank receipt."""
    integer_values = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "requests_per_rank": requests_per_rank,
        "context_window": context_window,
        "chunk_size": chunk_size,
        "decode_tokens": decode_tokens,
        "prefill_ranks": prefill_ranks,
    }
    if any(isinstance(value, bool) or int(value) <= 0 for value in integer_values.values()):
        raise ValueError("Cache-aware result workload dimensions must be positive integers")
    if not 0.0 <= float(warm_request_ratio) <= 1.0:
        raise ValueError("warm_request_ratio must be between zero and one")
    if not 0.0 <= float(warm_prefix_ratio) <= 1.0:
        raise ValueError("warm_prefix_ratio must be between zero and one")
    return {
        **{key: int(value) for key, value in integer_values.items()},
        "warm_request_ratio": float(warm_request_ratio),
        "warm_prefix_ratio": float(warm_prefix_ratio),
        "dtype": str(dtype),
    }


def _cache_aware_signature(
    *,
    config: dict[str, Any],
    world_size: int,
    tf32: bool,
) -> InputSignature:
    prefill_ranks = int(config["prefill_ranks"])
    total_requests = prefill_ranks * int(config["requests_per_rank"])
    batch_size = int(config["batch_size"])
    hidden_size = int(config["hidden_size"])
    dtype = str(config["dtype"])
    parameter_count = world_size * (int(config["num_layers"]) + 1) * hidden_size * hidden_size
    return InputSignature(
        shapes={
            "prompt": (batch_size, int(config["context_window"]), hidden_size),
            "output": (total_requests, batch_size, hidden_size),
        },
        dtypes={"prompt": dtype, "output": dtype},
        batch_size=batch_size,
        parameter_count=parameter_count,
        precision_flags=PrecisionFlags(
            bf16=dtype == str(torch.bfloat16),
            fp16=dtype == str(torch.float16),
            tf32=bool(tf32),
        ),
        world_size=world_size,
        pipeline_stages=2,
        pipeline_stage_boundaries=[
            (0, prefill_ranks - 1),
            (prefill_ranks, world_size - 1),
        ],
        per_rank_batch_size=batch_size,
        collective_type="send_recv",
    )


def _validate_tensor_mapping(
    value: Any,
    *,
    field: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Cache-aware child {field} must be a request-indexed mapping")
    normalized: dict[int, torch.Tensor] = {}
    for request_id, tensor in value.items():
        if type(request_id) is not int or request_id < 0:
            raise RuntimeError(f"Cache-aware child {field} has an invalid request ID")
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"Cache-aware child {field}[{request_id}] is not a tensor")
        if tensor.shape != shape or tensor.dtype != dtype:
            raise RuntimeError(
                f"Cache-aware child {field}[{request_id}] has an unexpected shape or dtype"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"Cache-aware child {field}[{request_id}] contains non-finite values")
        normalized[request_id] = tensor.contiguous()
    return normalized


class CacheAwareDisaggChildResultMixin:
    """Validate measured rank outputs before exposing pair verification data."""

    _cache_aware_result_context: dict[str, Any] | None = None
    _cache_aware_result_bundle: dict[str, Any] | None = None
    _cache_aware_result_metrics: dict[str, float] | None = None

    def prepare_cache_aware_child_result(
        self,
        *,
        variant: str,
        label: str,
        world_size: int,
        iterations: int,
        config: dict[str, Any],
        output_rank_by_request: dict[int, int],
        reference_rank_by_request: dict[int, int],
    ) -> dict[str, str]:
        if variant not in {"baseline", "optimized"}:
            raise ValueError(f"Unsupported cache-aware variant: {variant!r}")
        if world_size < 2:
            raise RuntimeError("SKIPPED: cache-aware child results require world_size >= 2")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("Cache-aware child-result iterations must be positive")
        prefill_ranks = int(config["prefill_ranks"])
        if prefill_ranks >= world_size:
            raise ValueError("Cache-aware result config leaves no decode rank")
        expected_requests = set(
            range(prefill_ranks * int(config["requests_per_rank"]))
        )
        if set(output_rank_by_request) != expected_requests:
            raise ValueError("Cache-aware output-rank map does not cover every request")
        if set(reference_rank_by_request) != expected_requests:
            raise ValueError("Cache-aware reference-rank map does not cover every request")
        if any(
            rank < prefill_ranks or rank >= world_size
            for rank in output_rank_by_request.values()
        ):
            raise ValueError("Cache-aware output-rank map names a non-decode rank")
        if any(
            rank < 0 or rank >= prefill_ranks
            for rank in reference_rank_by_request.values()
        ):
            raise ValueError("Cache-aware reference-rank map names a non-prefill rank")

        result_dir = Path(tempfile.mkdtemp(prefix="aisp-cache-aware-disagg-result-"))
        token = uuid.uuid4().hex
        self._cache_aware_result_context = {
            "result_dir": result_dir,
            "token": token,
            "variant": variant,
            "label": label,
            "world_size": int(world_size),
            "iterations": iterations,
            "config": dict(config),
            "output_rank_by_request": dict(output_rank_by_request),
            "reference_rank_by_request": dict(reference_rank_by_request),
            "retention": "pending-child-result",
        }
        self._cache_aware_result_bundle = None
        self._cache_aware_result_metrics = None
        for attribute in (
            "_subprocess_verify_output",
            "_subprocess_output_tolerance",
            "_subprocess_input_signature",
            "_subprocess_verify_inputs",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return {
            CACHE_AWARE_RESULT_DIR_ENV: str(result_dir),
            CACHE_AWARE_RESULT_TOKEN_ENV: token,
            CACHE_AWARE_VARIANT_ENV: variant,
            CACHE_AWARE_METRICS_PATH_ENV: str(result_dir / "metrics.json"),
        }

    def consume_cache_aware_child_results(
        self,
        *,
        launch_wall_ns: int,
        finish_wall_ns: int,
        returncode: int,
        **_: Any,
    ) -> None:
        context = self._cache_aware_result_context
        if context is None:
            raise RuntimeError("Cache-aware child-result callback has no launch context")
        result_dir = cast(Path, context["result_dir"])
        context["retention"] = "retained-invalid-child-result"
        if returncode != 0:
            context["retention"] = "retained-child-failure"
            raise RuntimeError(
                "Cache-aware child-result callback requires a clean child exit; "
                f"artifacts retained at {result_dir}"
            )

        expected_world_size = int(context["world_size"])
        expected_config = cast(dict[str, Any], context["config"])
        expected_iterations = int(context["iterations"])
        paths = sorted(result_dir.glob("rank-*.pt"))
        if len(paths) != expected_world_size:
            context["retention"] = "retained-incomplete-rank-quorum"
            raise RuntimeError(
                "Cache-aware child-result rank quorum is incomplete: "
                f"expected {expected_world_size}, found {len(paths)}; "
                f"artifacts retained at {result_dir}"
            )

        dtype_name = str(expected_config["dtype"])
        dtype_by_name = {
            str(torch.bfloat16): torch.bfloat16,
            str(torch.float16): torch.float16,
            str(torch.float32): torch.float32,
        }
        if dtype_name not in dtype_by_name:
            raise RuntimeError(f"Unsupported cache-aware result dtype: {dtype_name}")
        dtype = dtype_by_name[dtype_name]
        output_shape = (
            int(expected_config["batch_size"]),
            int(expected_config["hidden_size"]),
        )
        prompt_shape = (
            int(expected_config["batch_size"]),
            int(expected_config["context_window"]),
            int(expected_config["hidden_size"]),
        )
        payloads: dict[int, dict[str, Any]] = {}
        actual_outputs: dict[int, torch.Tensor] = {}
        reference_outputs: dict[int, torch.Tensor] = {}
        actual_owners: dict[int, int] = {}
        reference_owners: dict[int, int] = {}
        common_signature: InputSignature | None = None
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Cache-aware child result must be a regular file: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Invalid cache-aware child-result payload at {path}")
            if payload.get("schema") != CACHE_AWARE_RESULT_SCHEMA:
                raise RuntimeError(f"Invalid cache-aware child-result schema at {path}")
            if payload.get("token") != context["token"]:
                raise RuntimeError(f"Cache-aware child-result token mismatch at {path}")
            if payload.get("variant") != context["variant"]:
                raise RuntimeError(f"Cache-aware child-result variant mismatch at {path}")
            if payload.get("label") != context["label"]:
                raise RuntimeError(f"Cache-aware child-result label mismatch at {path}")
            if payload.get("config") != expected_config:
                raise RuntimeError(f"Cache-aware child-result config mismatch at {path}")
            if int(payload.get("world_size", -1)) != expected_world_size:
                raise RuntimeError(f"Cache-aware child-result world-size mismatch at {path}")
            if int(payload.get("iterations_completed", -1)) != expected_iterations:
                raise RuntimeError(f"Cache-aware child-result iteration mismatch at {path}")
            rank = int(payload.get("rank", -1))
            if rank < 0 or rank >= expected_world_size or rank in payloads:
                raise RuntimeError(f"Invalid or duplicate cache-aware child rank at {path}")
            if path.name != f"rank-{rank}.pt":
                raise RuntimeError(f"Cache-aware child filename/rank mismatch at {path}")
            created_wall_ns = int(payload.get("created_wall_ns", 0))
            if not launch_wall_ns <= created_wall_ns <= finish_wall_ns:
                raise RuntimeError(f"Stale cache-aware child-result payload at {path}")
            if int(payload.get("launch_wall_ns", 0)) != int(launch_wall_ns):
                raise RuntimeError(f"Cache-aware child launch identity mismatch at {path}")

            signature_payload = payload.get("input_signature")
            if not isinstance(signature_payload, dict):
                raise RuntimeError(f"Cache-aware child input signature is missing at {path}")
            signature = InputSignature.from_dict(signature_payload)
            signature_errors = signature.validate(strict=True)
            if signature_errors:
                raise RuntimeError(
                    f"Invalid cache-aware child input signature at {path}: {signature_errors[0]}"
                )
            expected_signature = _cache_aware_signature(
                config=expected_config,
                world_size=expected_world_size,
                tf32=signature.precision_flags.tf32,
            )
            if signature.to_dict() != expected_signature.to_dict():
                raise RuntimeError(f"Cache-aware child input signature mismatch at {path}")
            if common_signature is None:
                common_signature = signature
            elif signature.to_dict() != common_signature.to_dict():
                raise RuntimeError("Cache-aware child input signatures differ across ranks")
            if payload.get("output_tolerance") != list(CACHE_AWARE_OUTPUT_TOLERANCE):
                raise RuntimeError(f"Cache-aware child output tolerance mismatch at {path}")

            rank_actual = _validate_tensor_mapping(
                payload.get("actual_outputs"),
                field="actual_outputs",
                shape=output_shape,
                dtype=dtype,
            )
            rank_reference = _validate_tensor_mapping(
                payload.get("reference_outputs"),
                field="reference_outputs",
                shape=output_shape,
                dtype=dtype,
            )
            for request_id, tensor in rank_actual.items():
                if request_id in actual_outputs:
                    raise RuntimeError(f"Duplicate cache-aware actual output for request {request_id}")
                actual_outputs[request_id] = tensor
                actual_owners[request_id] = rank
            for request_id, tensor in rank_reference.items():
                if request_id in reference_outputs:
                    raise RuntimeError(f"Duplicate cache-aware reference for request {request_id}")
                reference_outputs[request_id] = tensor
                reference_owners[request_id] = rank

            prompt = payload.get("verification_prompt")
            if rank == 0:
                if not isinstance(prompt, torch.Tensor):
                    raise RuntimeError("Cache-aware rank 0 verification prompt is missing")
                if prompt.shape != prompt_shape or prompt.dtype != dtype:
                    raise RuntimeError("Cache-aware rank 0 verification prompt shape/dtype mismatch")
                if not bool(torch.isfinite(prompt).all()):
                    raise RuntimeError("Cache-aware rank 0 verification prompt is non-finite")
            elif prompt is not None:
                raise RuntimeError(f"Unexpected cache-aware verification prompt from rank {rank}")
            payloads[rank] = payload

        if set(payloads) != set(range(expected_world_size)):
            raise RuntimeError("Cache-aware child rank quorum is not contiguous")
        expected_output_owners = cast(dict[int, int], context["output_rank_by_request"])
        expected_reference_owners = cast(dict[int, int], context["reference_rank_by_request"])
        if actual_owners != expected_output_owners:
            raise RuntimeError("Cache-aware actual output request ownership is incomplete or incorrect")
        if reference_owners != expected_reference_owners:
            raise RuntimeError("Cache-aware reference request ownership is incomplete or incorrect")

        ordered_request_ids = sorted(expected_output_owners)
        for request_id in ordered_request_ids:
            try:
                torch.testing.assert_close(
                    actual_outputs[request_id],
                    reference_outputs[request_id],
                    rtol=CACHE_AWARE_OUTPUT_TOLERANCE[0],
                    atol=CACHE_AWARE_OUTPUT_TOLERANCE[1],
                )
            except AssertionError as exc:
                raise RuntimeError(
                    f"Cache-aware full timed output differs from its reference for request {request_id}"
                ) from exc

        metrics_payload = payloads[0].get("custom_metrics")
        if not isinstance(metrics_payload, dict) or not metrics_payload:
            raise RuntimeError("Cache-aware rank 0 custom metrics are missing")
        metrics: dict[str, float] = {}
        for key, value in metrics_payload.items():
            if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int | float):
                raise RuntimeError("Cache-aware rank 0 custom metrics are malformed")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise RuntimeError(f"Cache-aware metric {key!r} is non-finite")
            metrics[key] = numeric
        for rank in range(1, expected_world_size):
            if payloads[rank].get("custom_metrics") not in ({}, None):
                raise RuntimeError(f"Unexpected cache-aware custom metrics from rank {rank}")
        if common_signature is None:
            raise RuntimeError("Cache-aware child input signature is missing")

        prompt = cast(torch.Tensor, payloads[0]["verification_prompt"]).contiguous()
        output = torch.stack([actual_outputs[request_id] for request_id in ordered_request_ids])
        self._subprocess_verify_inputs = {"prompt": prompt}
        self._subprocess_verify_output = output
        self._subprocess_output_tolerance = CACHE_AWARE_OUTPUT_TOLERANCE
        self._subprocess_input_signature = common_signature
        self._cache_aware_result_metrics = metrics
        self._cache_aware_result_bundle = {
            "prompt": prompt,
            "output": output,
            "custom_metrics": metrics,
            "world_size": expected_world_size,
            "request_count": len(ordered_request_ids),
        }
        try:
            shutil.rmtree(result_dir)
        except OSError:
            context["retention"] = "retained-cleanup-failure"
        else:
            context["retention"] = "cleaned-after-success"

    def require_cache_aware_child_result(self) -> None:
        if self._cache_aware_result_bundle is None:
            context = self._cache_aware_result_context
            retained = context.get("result_dir") if context else "unavailable"
            raise RuntimeError(
                "Cache-aware verification requires a fresh measured child result; "
                f"retained path: {retained}"
            )

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        if self._cache_aware_result_bundle is not None:
            return dict(self._subprocess_verify_inputs)
        return super().get_verify_inputs()  # type: ignore[misc]

    def get_verify_output(self) -> torch.Tensor:
        if self._cache_aware_result_bundle is not None:
            return self._subprocess_verify_output.detach().clone()
        return super().get_verify_output()  # type: ignore[misc]

    def get_input_signature(self) -> InputSignature:
        if self._cache_aware_result_bundle is not None:
            return self._subprocess_input_signature
        return super().get_input_signature()  # type: ignore[misc]

    def get_output_tolerance(self) -> tuple[float, float]:
        if self._cache_aware_result_bundle is not None:
            return self._subprocess_output_tolerance
        return super().get_output_tolerance()  # type: ignore[misc]


def write_cache_aware_child_result(
    *,
    variant: str,
    label: str,
    rank: int,
    world_size: int,
    iterations_completed: int,
    config: dict[str, Any],
    actual_outputs: dict[int, torch.Tensor],
    reference_outputs: dict[int, torch.Tensor],
    verification_prompt: torch.Tensor | None,
    custom_metrics: dict[str, float] | None,
) -> bool:
    """Atomically write one rank's measured output when a harness launch requests it."""
    launch_wall_ns = os.environ.get(CACHE_AWARE_LAUNCH_WALL_NS_ENV)
    if not launch_wall_ns:
        return False
    result_dir_value = os.environ.get(CACHE_AWARE_RESULT_DIR_ENV)
    token = os.environ.get(CACHE_AWARE_RESULT_TOKEN_ENV)
    expected_variant = os.environ.get(CACHE_AWARE_VARIANT_ENV)
    if not all((result_dir_value, token, expected_variant)):
        raise RuntimeError("Cache-aware child-result environment is incomplete")
    if expected_variant != variant:
        raise RuntimeError(
            f"Cache-aware child variant mismatch: {expected_variant!r} != {variant!r}"
        )
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Cache-aware child rank {rank} is outside world size {world_size}")

    signature = _cache_aware_signature(
        config=config,
        world_size=world_size,
        tf32=torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
    )
    payload = {
        "schema": CACHE_AWARE_RESULT_SCHEMA,
        "token": token,
        "variant": variant,
        "label": label,
        "rank": int(rank),
        "world_size": int(world_size),
        "iterations_completed": int(iterations_completed),
        "launch_wall_ns": int(launch_wall_ns),
        "created_wall_ns": time.time_ns(),
        "config": dict(config),
        "actual_outputs": {
            int(request_id): tensor.detach().to(device="cpu").contiguous()
            for request_id, tensor in actual_outputs.items()
        },
        "reference_outputs": {
            int(request_id): tensor.detach().to(device="cpu").contiguous()
            for request_id, tensor in reference_outputs.items()
        },
        "verification_prompt": (
            verification_prompt.detach().to(device="cpu").contiguous()
            if verification_prompt is not None
            else None
        ),
        "custom_metrics": dict(custom_metrics or {}),
        "input_signature": signature.to_dict(),
        "output_tolerance": list(CACHE_AWARE_OUTPUT_TOLERANCE),
    }
    result_dir = Path(cast(str, result_dir_value))
    if result_dir.is_symlink() or not result_dir.is_dir():
        raise RuntimeError("Cache-aware child-result directory must be the prepared directory")
    temporary_path = result_dir / f".rank-{rank}-{os.getpid()}.tmp"
    destination = result_dir / f"rank-{rank}.pt"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, destination)
    return True


__all__ = [
    "CACHE_AWARE_METRICS_PATH_ENV",
    "CACHE_AWARE_OUTPUT_TOLERANCE",
    "CACHE_AWARE_RESULT_CALLBACK",
    "CacheAwareDisaggChildResultMixin",
    "make_cache_aware_result_config",
    "write_cache_aware_child_result",
]
