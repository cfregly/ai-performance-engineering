"""Fresh full-output transport for the gradient-fusion torchrun pair."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from core.benchmark.verification import InputSignature, PrecisionFlags

if TYPE_CHECKING:
    from core.harness.benchmark_harness import BenchmarkConfig, TorchrunLaunchSpec

GRADIENT_FUSION_RESULT_CALLBACK = "consume_gradient_fusion_child_results"
GRADIENT_FUSION_RESULT_DIR_ENV = "AISP_GRADIENT_FUSION_RESULT_DIR"
GRADIENT_FUSION_RESULT_TOKEN_ENV = "AISP_GRADIENT_FUSION_RESULT_TOKEN"
GRADIENT_FUSION_VARIANT_ENV = "AISP_GRADIENT_FUSION_VARIANT"
GRADIENT_FUSION_LAUNCH_WALL_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS"
GRADIENT_FUSION_RESULT_SCHEMA = "aisp.gradient-fusion.child-result.v1"
GRADIENT_FUSION_OUTPUT_TOLERANCE = (1e-3, 1e-5)


def _gradient_fusion_signature(
    *,
    num_tensors: int,
    numel_per_tensor: int,
    world_size: int,
) -> InputSignature:
    total_numel = num_tensors * numel_per_tensor
    shape = (total_numel,)
    input_shapes = {
        f"rank_{rank}_initial_gradients": shape for rank in range(world_size)
    }
    input_dtypes = {
        f"rank_{rank}_initial_gradients": str(torch.float16)
        for rank in range(world_size)
    }
    return InputSignature(
        shapes={
            **input_shapes,
            "reference_average": shape,
            "output": shape,
        },
        dtypes={
            **input_dtypes,
            "reference_average": str(torch.float16),
            "output": str(torch.float16),
        },
        batch_size=num_tensors,
        parameter_count=0,
        precision_flags=PrecisionFlags(fp16=True),
        world_size=world_size,
        ranks=list(range(world_size)),
        collective_type="all_reduce",
        collective_algorithm="premultiplied_sum_average",
    )


class GradientFusionChildResultMixin:
    """Validate measured rank outputs before exposing them to pair verification."""

    def get_profile_torchrun_spec(
        self,
        *,
        profiler: str,
        config: BenchmarkConfig | None = None,
        output_path: Path | None = None,
    ) -> TorchrunLaunchSpec | None:
        if profiler != "ncu":
            return None
        spec = self.get_torchrun_spec(config)
        return replace(spec, script_args=[*(spec.script_args or []), "--profile-rank", "0"])

    _gradient_fusion_result_context: dict[str, Any] | None = None
    _gradient_fusion_result_bundle: dict[str, Any] | None = None

    def prepare_gradient_fusion_child_result(
        self,
        *,
        variant: str,
        world_size: int,
        num_tensors: int,
        tensor_kb: int,
        iterations: int,
    ) -> dict[str, str]:
        if variant not in {"baseline", "optimized"}:
            raise ValueError(f"Unsupported gradient-fusion variant: {variant!r}")
        if world_size < 2:
            raise RuntimeError("SKIPPED: gradient-fusion child results require world_size >= 2")
        if num_tensors <= 0 or tensor_kb <= 0 or iterations <= 0:
            raise ValueError("Gradient-fusion workload dimensions must be positive")

        result_dir = Path(tempfile.mkdtemp(prefix="aisp-gradient-fusion-result-"))
        token = uuid.uuid4().hex
        self._gradient_fusion_result_context = {
            "result_dir": result_dir,
            "token": token,
            "variant": variant,
            "world_size": int(world_size),
            "num_tensors": int(num_tensors),
            "tensor_kb": int(tensor_kb),
            "iterations": int(iterations),
            "retention": "pending-child-result",
        }
        self._gradient_fusion_result_bundle = None
        for attribute in (
            "_subprocess_verify_output",
            "_subprocess_output_tolerance",
            "_subprocess_input_signature",
            "_subprocess_verify_inputs",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return {
            GRADIENT_FUSION_RESULT_DIR_ENV: str(result_dir),
            GRADIENT_FUSION_RESULT_TOKEN_ENV: token,
            GRADIENT_FUSION_VARIANT_ENV: variant,
        }

    def consume_gradient_fusion_child_results(
        self,
        *,
        launch_wall_ns: int,
        finish_wall_ns: int,
        returncode: int,
        **_: Any,
    ) -> None:
        context = self._gradient_fusion_result_context
        if context is None:
            raise RuntimeError("Gradient-fusion child-result callback has no launch context")
        result_dir = cast(Path, context["result_dir"])
        if returncode != 0:
            context["retention"] = "retained-child-failure"
            raise RuntimeError(
                "Gradient-fusion child-result callback requires a clean child exit; "
                f"artifacts retained at {result_dir}"
            )

        expected_world_size = int(context["world_size"])
        paths = sorted(result_dir.glob("rank-*.pt"))
        if len(paths) != expected_world_size:
            context["retention"] = "retained-incomplete-rank-quorum"
            raise RuntimeError(
                "Gradient-fusion child-result rank quorum is incomplete: "
                f"expected {expected_world_size}, found {len(paths)}; artifacts retained at {result_dir}"
            )

        expected_numel_per_tensor = (int(context["tensor_kb"]) * 1024) // (
            torch.finfo(torch.float16).bits // 8
        )
        expected_total_numel = int(context["num_tensors"]) * expected_numel_per_tensor
        ranked_payloads: list[tuple[int, dict[str, Any], InputSignature]] = []
        seen_ranks: set[int] = set()
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Gradient-fusion child result must be a regular file: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Invalid gradient-fusion child-result payload at {path}")
            if payload.get("schema") != GRADIENT_FUSION_RESULT_SCHEMA:
                raise RuntimeError(f"Invalid gradient-fusion child-result schema at {path}")
            if payload.get("token") != context["token"]:
                raise RuntimeError(f"Gradient-fusion child-result token mismatch at {path}")
            if payload.get("variant") != context["variant"]:
                raise RuntimeError(f"Gradient-fusion child-result variant mismatch at {path}")
            if int(payload.get("world_size", -1)) != expected_world_size:
                raise RuntimeError(f"Gradient-fusion child-result world-size mismatch at {path}")
            for key in ("num_tensors", "tensor_kb", "iterations"):
                if int(payload.get(key, -1)) != int(context[key]):
                    raise RuntimeError(f"Gradient-fusion child-result {key} mismatch at {path}")

            rank = int(payload.get("rank", -1))
            if rank < 0 or rank >= expected_world_size or rank in seen_ranks:
                raise RuntimeError(f"Invalid or duplicate gradient-fusion child-result rank at {path}")
            if path.name != f"rank-{rank}.pt":
                raise RuntimeError(f"Gradient-fusion child-result filename/rank mismatch at {path}")
            seen_ranks.add(rank)
            created_wall_ns = int(payload.get("created_wall_ns", 0))
            if not launch_wall_ns <= created_wall_ns <= finish_wall_ns:
                raise RuntimeError(f"Stale gradient-fusion child-result payload at {path}")
            if int(payload.get("launch_wall_ns", 0)) != int(launch_wall_ns):
                raise RuntimeError(f"Gradient-fusion child-result launch identity mismatch at {path}")

            tensors = {
                name: payload.get(name)
                for name in ("initial_gradients", "reference_average", "verify_output")
            }
            for name, tensor in tensors.items():
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError(f"Gradient-fusion {name} is missing at {path}")
                if tensor.dtype != torch.float16 or tensor.shape != (expected_total_numel,):
                    raise RuntimeError(f"Gradient-fusion {name} shape/dtype mismatch at {path}")
                if not bool(torch.isfinite(tensor).all()):
                    raise RuntimeError(f"Gradient-fusion {name} contains non-finite values at {path}")

            try:
                torch.testing.assert_close(
                    tensors["verify_output"],
                    tensors["reference_average"],
                    rtol=GRADIENT_FUSION_OUTPUT_TOLERANCE[0],
                    atol=GRADIENT_FUSION_OUTPUT_TOLERANCE[1],
                )
            except AssertionError as exc:
                raise RuntimeError(
                    f"Gradient-fusion full timed output differs from the one-average oracle at {path}"
                ) from exc

            signature_payload = payload.get("input_signature")
            if not isinstance(signature_payload, dict):
                raise RuntimeError(f"Gradient-fusion input signature is missing at {path}")
            signature = InputSignature.from_dict(signature_payload)
            signature_errors = signature.validate(strict=True)
            if signature_errors:
                raise RuntimeError(
                    f"Invalid gradient-fusion input signature at {path}: {signature_errors[0]}"
                )
            expected_signature = _gradient_fusion_signature(
                num_tensors=int(context["num_tensors"]),
                numel_per_tensor=expected_numel_per_tensor,
                world_size=expected_world_size,
            )
            if signature.to_dict() != expected_signature.to_dict():
                raise RuntimeError(f"Gradient-fusion input signature mismatch at {path}")
            if payload.get("output_tolerance") != list(GRADIENT_FUSION_OUTPUT_TOLERANCE):
                raise RuntimeError(f"Gradient-fusion output tolerance mismatch at {path}")
            ranked_payloads.append((rank, payload, signature))

        ranked_payloads.sort(key=lambda item: item[0])
        rank0 = ranked_payloads[0][1]
        rank0_signature = ranked_payloads[0][2]
        for _, payload, signature in ranked_payloads[1:]:
            if signature.to_dict() != rank0_signature.to_dict():
                raise RuntimeError("Gradient-fusion input signatures differ across ranks")
            if not torch.equal(payload["reference_average"], rank0["reference_average"]):
                raise RuntimeError("Gradient-fusion one-average oracle differs across ranks")
            if not torch.equal(payload["verify_output"], rank0["verify_output"]):
                raise RuntimeError("Gradient-fusion timed output differs across ranks")

        computed_average = torch.zeros(expected_total_numel, dtype=torch.float32)
        for _, payload, _ in ranked_payloads:
            computed_average.add_(payload["initial_gradients"].float())
        computed_average.div_(expected_world_size)
        computed_average_fp16 = computed_average.to(dtype=torch.float16)
        try:
            torch.testing.assert_close(
                rank0["reference_average"],
                computed_average_fp16,
                rtol=GRADIENT_FUSION_OUTPUT_TOLERANCE[0],
                atol=GRADIENT_FUSION_OUTPUT_TOLERANCE[1],
            )
        except AssertionError as exc:
            raise RuntimeError(
                "Gradient-fusion one-average oracle does not match the full-rank inputs"
            ) from exc

        self._subprocess_verify_inputs = {
            **{
                f"rank_{rank}_initial_gradients": payload["initial_gradients"]
                for rank, payload, _ in ranked_payloads
            },
            "reference_average": rank0["reference_average"],
        }
        self._subprocess_verify_output = rank0["verify_output"]
        self._subprocess_output_tolerance = GRADIENT_FUSION_OUTPUT_TOLERANCE
        self._subprocess_input_signature = rank0_signature
        self._gradient_fusion_result_bundle = rank0
        try:
            shutil.rmtree(result_dir)
        except OSError:
            context["retention"] = "retained-cleanup-failure"
        else:
            context["retention"] = "cleaned-after-success"

    def require_gradient_fusion_child_result(self) -> None:
        if self._gradient_fusion_result_bundle is None:
            context = self._gradient_fusion_result_context
            retained = context.get("result_dir") if context else "unavailable"
            raise RuntimeError(
                "Gradient-fusion verification requires a fresh measured child result; "
                f"retained path: {retained}"
            )

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        if self._gradient_fusion_result_bundle is not None:
            return dict(self._subprocess_verify_inputs)
        return super().get_verify_inputs()  # type: ignore[misc]

    def get_verify_output(self) -> torch.Tensor:
        if self._gradient_fusion_result_bundle is not None:
            return self._subprocess_verify_output.detach().clone()
        return super().get_verify_output()  # type: ignore[misc]

    def get_input_signature(self) -> InputSignature:
        if self._gradient_fusion_result_bundle is not None:
            return self._subprocess_input_signature
        return super().get_input_signature()  # type: ignore[misc]

    def get_output_tolerance(self) -> tuple[float, float]:
        if self._gradient_fusion_result_bundle is not None:
            return self._subprocess_output_tolerance
        return super().get_output_tolerance()  # type: ignore[misc]

    def validate_result(self) -> str | None:
        if self._gradient_fusion_result_bundle is None:
            return "Fresh full-rank gradient-fusion worker output is missing"
        return None


def write_gradient_fusion_child_result(
    *,
    variant: str,
    rank: int,
    world_size: int,
    num_tensors: int,
    tensor_kb: int,
    iterations: int,
    initial_gradients: torch.Tensor,
    reference_average: torch.Tensor,
    verify_output: torch.Tensor,
) -> bool:
    """Atomically write one rank's measured result when launched by the harness."""
    result_dir_value = os.environ.get(GRADIENT_FUSION_RESULT_DIR_ENV)
    token = os.environ.get(GRADIENT_FUSION_RESULT_TOKEN_ENV)
    expected_variant = os.environ.get(GRADIENT_FUSION_VARIANT_ENV)
    launch_wall_ns = os.environ.get(GRADIENT_FUSION_LAUNCH_WALL_NS_ENV)
    # Profiler launches reuse the torchrun spec but have no parent result callback.
    # Only a harness callback launch carries the wall-clock identity, so do not
    # create success-shaped output that no parent can freshness-check.
    if not launch_wall_ns:
        return False
    if not all((result_dir_value, token, expected_variant)):
        raise RuntimeError("Gradient-fusion child-result environment is incomplete")
    if expected_variant != variant:
        raise RuntimeError(
            f"Gradient-fusion child variant mismatch: {expected_variant!r} != {variant!r}"
        )

    expected_numel_per_tensor = (tensor_kb * 1024) // (
        torch.finfo(torch.float16).bits // 8
    )
    expected_total_numel = num_tensors * expected_numel_per_tensor
    for name, tensor in (
        ("initial_gradients", initial_gradients),
        ("reference_average", reference_average),
        ("verify_output", verify_output),
    ):
        if tensor.dtype != torch.float16 or tensor.shape != (expected_total_numel,):
            raise RuntimeError(
                f"Gradient-fusion {name} has unexpected shape/dtype for the declared workload"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"Gradient-fusion {name} contains non-finite values")
    try:
        torch.testing.assert_close(
            verify_output,
            reference_average,
            rtol=GRADIENT_FUSION_OUTPUT_TOLERANCE[0],
            atol=GRADIENT_FUSION_OUTPUT_TOLERANCE[1],
        )
    except AssertionError as exc:
        raise RuntimeError(
            "Gradient-fusion full timed output differs from the one-average oracle"
        ) from exc

    signature = _gradient_fusion_signature(
        num_tensors=num_tensors,
        numel_per_tensor=expected_numel_per_tensor,
        world_size=world_size,
    )
    payload = {
        "schema": GRADIENT_FUSION_RESULT_SCHEMA,
        "token": token,
        "variant": variant,
        "rank": int(rank),
        "world_size": int(world_size),
        "num_tensors": int(num_tensors),
        "tensor_kb": int(tensor_kb),
        "iterations": int(iterations),
        "launch_wall_ns": int(launch_wall_ns),
        "created_wall_ns": time.time_ns(),
        "initial_gradients": initial_gradients.detach().to(device="cpu").contiguous(),
        "reference_average": reference_average.detach().to(device="cpu").contiguous(),
        "verify_output": verify_output.detach().to(device="cpu").contiguous(),
        "input_signature": signature.to_dict(),
        "output_tolerance": list(GRADIENT_FUSION_OUTPUT_TOLERANCE),
    }
    result_dir = Path(cast(str, result_dir_value))
    if result_dir.is_symlink() or not result_dir.is_dir():
        raise RuntimeError(
            "Gradient-fusion child-result directory must be the prepared regular directory"
        )
    temporary_path = result_dir / f".rank-{rank}-{os.getpid()}.tmp"
    destination = result_dir / f"rank-{rank}.pt"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, destination)
    return True


__all__ = [
    "GRADIENT_FUSION_OUTPUT_TOLERANCE",
    "GRADIENT_FUSION_LAUNCH_WALL_NS_ENV",
    "GRADIENT_FUSION_RESULT_CALLBACK",
    "GRADIENT_FUSION_RESULT_DIR_ENV",
    "GRADIENT_FUSION_RESULT_SCHEMA",
    "GRADIENT_FUSION_RESULT_TOKEN_ENV",
    "GRADIENT_FUSION_VARIANT_ENV",
    "GradientFusionChildResultMixin",
    "write_gradient_fusion_child_result",
]
