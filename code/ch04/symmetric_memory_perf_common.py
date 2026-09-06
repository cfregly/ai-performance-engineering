"""Shared helpers for the distributed symmetric-memory performance pair."""

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

from core.benchmark.verification import InputSignature

SYMMETRIC_MEMORY_PERF_BASELINE_NVTX_RANGE = (
    "transfer_sync:symmetric_memory_perf_nccl_p2p"
)
SYMMETRIC_MEMORY_PERF_OPTIMIZED_NVTX_RANGE = (
    "transfer_async:symmetric_memory_perf_peer_copy"
)
SYMMETRIC_MEMORY_PERF_RESULT_CALLBACK = (
    "consume_symmetric_memory_perf_child_results"
)
SYMMETRIC_MEMORY_PERF_RESULT_DIR_ENV = "AISP_SYMMETRIC_MEMORY_PERF_RESULT_DIR"
SYMMETRIC_MEMORY_PERF_RESULT_TOKEN_ENV = "AISP_SYMMETRIC_MEMORY_PERF_RESULT_TOKEN"
SYMMETRIC_MEMORY_PERF_VARIANT_ENV = "AISP_SYMMETRIC_MEMORY_PERF_VARIANT"
SYMMETRIC_MEMORY_PERF_LAUNCH_WALL_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS"
SYMMETRIC_MEMORY_PERF_RESULT_SCHEMA = "aisp.symmetric-memory-perf.child-result.v1"


def make_rank_distinct_input(numel: int, device: torch.device, rank: int) -> torch.Tensor:
    """Keep peer transfers distinguishable even with identical per-rank seeds."""
    return torch.randn(numel, device=device, dtype=torch.float32).add_(float(rank))


def build_square_verification_probe(
    tensor: torch.Tensor,
    *,
    max_elements: int = 256 * 256,
) -> tuple[torch.Tensor, int]:
    """Return the largest square probe view that fits within the tensor."""
    available = int(tensor.numel())
    if available <= 0:
        raise ValueError("Verification probe requires a non-empty tensor")

    probe_numel = min(available, max_elements)
    side = math.isqrt(probe_numel)
    if side <= 0:
        raise ValueError("Verification probe side length must be positive")
    probe_numel = side * side
    return tensor[:probe_numel].view(side, side).detach(), probe_numel


class SymmetricMemoryPerfChildResultMixin:
    """Transport fresh, measured worker outputs back to a torchrun parent."""

    _symmetric_memory_perf_result_context: dict[str, Any] | None = None
    _symmetric_memory_perf_result_bundle: dict[str, Any] | None = None

    def prepare_symmetric_memory_perf_child_result(
        self,
        *,
        variant: str,
        world_size: int,
    ) -> dict[str, str]:
        if variant not in {"baseline", "optimized"}:
            raise ValueError(f"Unsupported symmetric-memory perf variant: {variant!r}")
        if world_size < 2:
            raise RuntimeError(
                "SKIPPED: symmetric_memory_perf child results require world_size >= 2"
            )

        result_dir = Path(tempfile.mkdtemp(prefix="aisp-symmetric-memory-perf-result-"))
        token = uuid.uuid4().hex
        self._symmetric_memory_perf_result_context = {
            "result_dir": result_dir,
            "token": token,
            "variant": variant,
            "world_size": int(world_size),
            "retention": "pending-child-result",
        }
        self._symmetric_memory_perf_result_bundle = None
        for attribute in (
            "_subprocess_verify_output",
            "_subprocess_output_tolerance",
            "_subprocess_input_signature",
            "_subprocess_verify_inputs",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return {
            SYMMETRIC_MEMORY_PERF_RESULT_DIR_ENV: str(result_dir),
            SYMMETRIC_MEMORY_PERF_RESULT_TOKEN_ENV: token,
            SYMMETRIC_MEMORY_PERF_VARIANT_ENV: variant,
        }

    def consume_symmetric_memory_perf_child_results(
        self,
        *,
        launch_wall_ns: int,
        finish_wall_ns: int,
        returncode: int,
        **_: Any,
    ) -> None:
        context = self._symmetric_memory_perf_result_context
        if context is None:
            raise RuntimeError(
                "Symmetric-memory perf child-result callback has no launch context"
            )
        result_dir = cast(Path, context["result_dir"])
        if returncode != 0:
            context["retention"] = "retained-child-failure"
            raise RuntimeError(
                "Symmetric-memory perf child-result callback requires a clean child "
                f"exit; artifacts retained at {result_dir}"
            )

        expected_world_size = int(context["world_size"])
        paths = sorted(result_dir.glob("rank-*.pt"))
        if len(paths) != expected_world_size:
            context["retention"] = "retained-incomplete-rank-quorum"
            raise RuntimeError(
                "Symmetric-memory perf child-result rank quorum is incomplete: "
                f"expected {expected_world_size}, found {len(paths)}; "
                f"artifacts retained at {result_dir}"
            )

        payloads: list[dict[str, Any]] = []
        seen_ranks: set[int] = set()
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Invalid child-result payload at {path}")
            if payload.get("schema") != SYMMETRIC_MEMORY_PERF_RESULT_SCHEMA:
                raise RuntimeError(f"Invalid child-result schema at {path}")
            if payload.get("token") != context["token"]:
                raise RuntimeError(f"Child-result token mismatch at {path}")
            if payload.get("variant") != context["variant"]:
                raise RuntimeError(f"Child-result variant mismatch at {path}")
            if int(payload.get("world_size", -1)) != expected_world_size:
                raise RuntimeError(f"Child-result world-size mismatch at {path}")
            rank = int(payload.get("rank", -1))
            if rank < 0 or rank >= expected_world_size or rank in seen_ranks:
                raise RuntimeError(f"Invalid or duplicate child-result rank at {path}")
            seen_ranks.add(rank)
            created_wall_ns = int(payload.get("created_wall_ns", 0))
            if not launch_wall_ns <= created_wall_ns <= finish_wall_ns:
                raise RuntimeError(f"Stale child-result payload at {path}")
            if int(payload.get("launch_wall_ns", 0)) != int(launch_wall_ns):
                raise RuntimeError(f"Child-result launch identity mismatch at {path}")
            verify_inputs = payload.get("verify_inputs")
            verify_output = payload.get("verify_output")
            if not isinstance(verify_inputs, dict) or set(verify_inputs) != {"tensor"}:
                raise RuntimeError(f"Child-result inputs are missing at {path}")
            if not all(isinstance(value, torch.Tensor) for value in verify_inputs.values()):
                raise RuntimeError(f"Child-result inputs must be tensors at {path}")
            if not isinstance(verify_output, torch.Tensor) or verify_output.numel() == 0:
                raise RuntimeError(f"Child-result output is missing at {path}")
            verify_input = verify_inputs["tensor"]
            if verify_output.shape != verify_input.shape:
                raise RuntimeError(f"Child-result output shape mismatch at {path}")
            if verify_output.dtype != verify_input.dtype:
                raise RuntimeError(f"Child-result output dtype mismatch at {path}")
            payloads.append(payload)

        payloads.sort(key=lambda payload: int(payload["rank"]))
        rank0 = payloads[0]
        signature: InputSignature | None = None
        for rank, payload in enumerate(payloads):
            path = result_dir / f"rank-{rank}.pt"
            rank_signature = InputSignature.from_dict(payload["input_signature"])
            signature_errors = rank_signature.validate(strict=True)
            if signature_errors:
                raise RuntimeError(
                    f"Invalid symmetric-memory perf input signature at rank {rank}: "
                    f"{signature_errors[0]}"
                )
            if rank_signature.world_size != expected_world_size:
                raise RuntimeError(
                    f"Child-result signature world-size mismatch at rank {rank}"
                )
            verify_input = payload["verify_inputs"]["tensor"]
            verify_output = payload["verify_output"]
            if rank_signature.shapes != {
                "tensor": tuple(verify_input.shape),
                "output": tuple(verify_output.shape),
            }:
                raise RuntimeError(f"Child-result signature shape mismatch at rank {rank}")
            if rank_signature.dtypes != {
                "tensor": str(verify_input.dtype),
                "output": str(verify_output.dtype),
            }:
                raise RuntimeError(f"Child-result signature dtype mismatch at rank {rank}")
            if signature is None:
                signature = rank_signature
            elif not signature.matches(rank_signature):
                raise RuntimeError(
                    f"Child-result input signatures differ across ranks at rank {rank}"
                )

            tolerance = payload.get("output_tolerance")
            if not isinstance(tolerance, tuple | list) or len(tolerance) != 2:
                raise RuntimeError(f"Invalid child-result output tolerance at {path}")
            rtol, atol = float(tolerance[0]), float(tolerance[1])
            if rtol < 0 or atol < 0 or not math.isfinite(rtol + atol):
                raise RuntimeError(f"Invalid child-result output tolerance at {path}")
            sender_rank = (rank - 1) % expected_world_size
            sender_input = payloads[sender_rank]["verify_inputs"]["tensor"]
            if not torch.allclose(
                verify_output,
                sender_input,
                rtol=rtol,
                atol=atol,
            ):
                raise RuntimeError(
                    "Child-result output does not match its measured sender input "
                    f"at rank {rank}"
                )
        if signature is None:
            raise RuntimeError("Symmetric-memory perf input signature is missing")
        tolerance = rank0["output_tolerance"]

        self._subprocess_verify_inputs = cast(
            dict[str, torch.Tensor], rank0["verify_inputs"]
        )
        self._subprocess_verify_output = cast(torch.Tensor, rank0["verify_output"])
        self._subprocess_output_tolerance = (float(tolerance[0]), float(tolerance[1]))
        self._subprocess_input_signature = signature
        self._symmetric_memory_perf_result_bundle = rank0
        try:
            shutil.rmtree(result_dir)
        except OSError:
            context["retention"] = "retained-cleanup-failure"
        else:
            context["retention"] = "cleaned-after-success"

    def require_symmetric_memory_perf_child_result(self) -> None:
        if self._symmetric_memory_perf_result_bundle is None:
            context = self._symmetric_memory_perf_result_context
            retained = context.get("result_dir") if context else "unavailable"
            raise RuntimeError(
                "Symmetric-memory perf verification requires a fresh measured child "
                f"result; retained path: {retained}"
            )

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        if self._symmetric_memory_perf_result_bundle is not None:
            return dict(self._subprocess_verify_inputs)
        return super().get_verify_inputs()  # type: ignore[misc]

    def get_verify_output(self) -> torch.Tensor:
        if self._symmetric_memory_perf_result_bundle is not None:
            return self._subprocess_verify_output.detach().clone()
        return super().get_verify_output()  # type: ignore[misc]

    def get_input_signature(self) -> InputSignature:
        if self._symmetric_memory_perf_result_bundle is not None:
            return self._subprocess_input_signature
        return super().get_input_signature()  # type: ignore[misc]

    def get_output_tolerance(self) -> tuple[float, float]:
        if self._symmetric_memory_perf_result_bundle is not None:
            return self._subprocess_output_tolerance
        return super().get_output_tolerance()  # type: ignore[misc]


def write_symmetric_memory_perf_child_result(
    benchmark: Any,
    *,
    variant: str,
    rank: int,
    world_size: int,
) -> None:
    result_dir_value = os.environ.get(SYMMETRIC_MEMORY_PERF_RESULT_DIR_ENV)
    token = os.environ.get(SYMMETRIC_MEMORY_PERF_RESULT_TOKEN_ENV)
    expected_variant = os.environ.get(SYMMETRIC_MEMORY_PERF_VARIANT_ENV)
    launch_wall_ns = os.environ.get(SYMMETRIC_MEMORY_PERF_LAUNCH_WALL_NS_ENV)
    if not result_dir_value or not token or not expected_variant or not launch_wall_ns:
        raise RuntimeError("Symmetric-memory perf child-result environment is incomplete")
    if expected_variant != variant:
        raise RuntimeError(
            f"Symmetric-memory perf child variant mismatch: {expected_variant!r} != {variant!r}"
        )

    verify_inputs = {
        name: tensor.detach().cpu()
        for name, tensor in benchmark.get_verify_inputs().items()
    }
    verify_output = benchmark.get_verify_output().detach().cpu()
    result_dir = Path(result_dir_value)
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SYMMETRIC_MEMORY_PERF_RESULT_SCHEMA,
        "token": token,
        "variant": variant,
        "rank": int(rank),
        "world_size": int(world_size),
        "launch_wall_ns": int(launch_wall_ns),
        "created_wall_ns": time.time_ns(),
        "verify_inputs": verify_inputs,
        "verify_output": verify_output,
        "input_signature": benchmark.get_input_signature().to_dict(),
        "output_tolerance": list(benchmark.get_output_tolerance()),
    }
    temporary_path = result_dir / f".rank-{rank}-{os.getpid()}.tmp"
    destination = result_dir / f"rank-{rank}.pt"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, destination)
