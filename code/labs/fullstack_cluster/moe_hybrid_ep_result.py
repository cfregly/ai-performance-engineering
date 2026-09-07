"""Fresh full-output transport for the hybrid expert-parallel benchmark pair."""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from core.benchmark.verification import (
    InputSignature,
    PrecisionFlags,
    get_tolerance_for_dtype,
)

MOE_HYBRID_EP_RESULT_CALLBACK = "consume_moe_hybrid_ep_child_results"
MOE_HYBRID_EP_RESULT_DIR_ENV = "AISP_MOE_HYBRID_EP_RESULT_DIR"
MOE_HYBRID_EP_RESULT_TOKEN_ENV = "AISP_MOE_HYBRID_EP_RESULT_TOKEN"
MOE_HYBRID_EP_RESULT_LABEL_ENV = "AISP_MOE_HYBRID_EP_RESULT_LABEL"
MOE_HYBRID_EP_LAUNCH_WALL_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS"
MOE_HYBRID_EP_RESULT_SCHEMA = "aisp.moe-hybrid-ep.child-result.v1"
MOE_HYBRID_EP_REFERENCE_EXECUTION = "canonical_baseline_replay_outside_timed_range"

_SUPPORTED_DTYPES = {
    str(torch.float32): torch.float32,
    str(torch.float16): torch.float16,
    str(torch.bfloat16): torch.bfloat16,
}


@dataclass(frozen=True)
class MoEHybridEPResultContract:
    """Primitive workload identity shared by the parent and every worker rank."""

    label: str
    variant: str
    world_size: int
    iterations: int
    warmup_steps: int
    tokens_per_rank: int
    hidden_size: int
    num_experts: int
    local_experts: int
    top_k: int
    route_mode: str
    dtype: str
    learning_rate: float
    aux_loss_scale: float
    tf32: bool
    profile_range: str

    def validate(self) -> None:
        if self.variant not in {"baseline", "optimized"}:
            raise ValueError(f"Unsupported hybrid-EP variant: {self.variant!r}")
        if not self.label or not self.profile_range:
            raise ValueError("Hybrid-EP label and profile range must be non-empty")
        integers = {
            "world_size": self.world_size,
            "iterations": self.iterations,
            "tokens_per_rank": self.tokens_per_rank,
            "hidden_size": self.hidden_size,
            "num_experts": self.num_experts,
            "local_experts": self.local_experts,
            "top_k": self.top_k,
        }
        if any(isinstance(value, bool) or int(value) <= 0 for value in integers.values()):
            raise ValueError("Hybrid-EP workload dimensions must be positive integers")
        if isinstance(self.warmup_steps, bool) or self.warmup_steps < 0:
            raise ValueError("Hybrid-EP warmup_steps must be a non-negative integer")
        if self.num_experts != self.local_experts * self.world_size:
            raise ValueError("Hybrid-EP num_experts must equal local_experts * world_size")
        if self.top_k > self.num_experts:
            raise ValueError("Hybrid-EP top_k cannot exceed num_experts")
        if self.route_mode not in {"uniform", "topology_aware"}:
            raise ValueError(f"Unsupported hybrid-EP route mode: {self.route_mode!r}")
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported hybrid-EP result dtype: {self.dtype!r}")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Hybrid-EP learning_rate must be finite and positive")
        if not math.isfinite(self.aux_loss_scale) or self.aux_loss_scale < 0:
            raise ValueError("Hybrid-EP aux_loss_scale must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> MoEHybridEPResultContract:
        if not isinstance(value, dict):
            raise RuntimeError("Hybrid-EP child result is missing its workload contract")
        expected_keys = set(cls.__dataclass_fields__)
        if set(value) != expected_keys:
            raise RuntimeError("Hybrid-EP child-result workload contract keys are invalid")
        contract = cls(**value)
        contract.validate()
        return contract


def _dtype(contract: MoEHybridEPResultContract) -> torch.dtype:
    return _SUPPORTED_DTYPES[contract.dtype]


def _logical_parameter_count(contract: MoEHybridEPResultContract) -> int:
    hidden = contract.hidden_size
    # input/output projections and router are replicated; experts are sharded.
    replicated = (2 * hidden * hidden) + (hidden * contract.num_experts)
    expert = 12 * hidden * hidden
    return replicated + (contract.num_experts * expert)


def _per_rank_parameter_count(contract: MoEHybridEPResultContract) -> int:
    hidden = contract.hidden_size
    replicated = (2 * hidden * hidden) + (hidden * contract.num_experts)
    expert = 12 * hidden * hidden
    return replicated + (contract.local_experts * expert)


def _output_tolerance(contract: MoEHybridEPResultContract) -> tuple[float, float]:
    tolerance = get_tolerance_for_dtype(_dtype(contract))
    return float(tolerance.rtol), float(tolerance.atol)


def _input_signature(contract: MoEHybridEPResultContract) -> InputSignature:
    total_tokens = contract.world_size * contract.tokens_per_rank
    value_shape = (total_tokens, contract.hidden_size)
    assignment_shape = (total_tokens, contract.top_k)
    dtype = contract.dtype
    return InputSignature(
        shapes={
            "inputs": value_shape,
            "targets": value_shape,
            "route_assignments": assignment_shape,
            "reference_output": value_shape,
            "output": value_shape,
        },
        dtypes={
            "inputs": dtype,
            "targets": dtype,
            "route_assignments": str(torch.int64),
            "reference_output": dtype,
            "output": dtype,
        },
        batch_size=total_tokens,
        parameter_count=_logical_parameter_count(contract),
        precision_flags=PrecisionFlags(
            fp16=_dtype(contract) == torch.float16,
            bf16=_dtype(contract) == torch.bfloat16,
            tf32=contract.tf32,
        ),
        world_size=contract.world_size,
        ranks=list(range(contract.world_size)),
        shards=contract.world_size,
        per_rank_batch_size=contract.tokens_per_rank,
        collective_type="all_to_all" if contract.world_size > 1 else None,
        collective_algorithm=(
            "bidirectional_expert_route_exchange" if contract.world_size > 1 else None
        ),
    )


def _validate_tensor(
    tensor: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"Hybrid-EP child {name} is missing")
    if tensor.shape != shape or tensor.dtype != dtype:
        raise RuntimeError(f"Hybrid-EP child {name} has an unexpected shape or dtype")
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"Hybrid-EP child {name} contains non-finite values")
    return tensor.detach().to(device="cpu").contiguous()


def _assert_outputs_match(
    actual: torch.Tensor,
    reference: torch.Tensor,
    contract: MoEHybridEPResultContract,
) -> None:
    rtol, atol = _output_tolerance(contract)
    try:
        torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise RuntimeError(
            "Hybrid-EP full timed output differs from the canonical baseline replay"
        ) from exc


class MoEHybridEPChildResultMixin:
    """Expose verification only after a fresh, complete worker result quorum."""

    _moe_hybrid_ep_result_context: dict[str, Any] | None = None
    _moe_hybrid_ep_result_bundle: dict[str, Any] | None = None
    _moe_hybrid_ep_result_metrics: dict[str, float] | None = None

    def prepare_moe_hybrid_ep_child_result(
        self,
        contract: MoEHybridEPResultContract,
    ) -> dict[str, str]:
        contract.validate()
        result_dir = Path(tempfile.mkdtemp(prefix="aisp-moe-hybrid-ep-result-"))
        token = uuid.uuid4().hex
        self._moe_hybrid_ep_result_context = {
            "result_dir": result_dir,
            "token": token,
            "contract": contract,
            "retention": "pending-child-result",
        }
        self._moe_hybrid_ep_result_bundle = None
        self._moe_hybrid_ep_result_metrics = None
        for attribute in (
            "_subprocess_verify_output",
            "_subprocess_output_tolerance",
            "_subprocess_input_signature",
            "_subprocess_verify_inputs",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return {
            MOE_HYBRID_EP_RESULT_DIR_ENV: str(result_dir),
            MOE_HYBRID_EP_RESULT_TOKEN_ENV: token,
            MOE_HYBRID_EP_RESULT_LABEL_ENV: contract.label,
            "AISP_MOE_HYBRID_EP_METRICS_PATH": str(result_dir / "metrics.json"),
        }

    def consume_moe_hybrid_ep_child_results(
        self,
        *,
        launch_wall_ns: int,
        finish_wall_ns: int,
        returncode: int,
        stdout: str = "",
        **_: Any,
    ) -> None:
        context = self._moe_hybrid_ep_result_context
        if context is None:
            raise RuntimeError("Hybrid-EP child-result callback has no launch context")
        result_dir = cast(Path, context["result_dir"])
        context["retention"] = "retained-invalid-child-result"
        if returncode != 0:
            context["retention"] = "retained-child-failure"
            raise RuntimeError(
                "Hybrid-EP child-result callback requires a clean child exit; "
                f"artifacts retained at {result_dir}"
            )

        expected = cast(MoEHybridEPResultContract, context["contract"])
        paths = sorted(result_dir.glob("rank-*.pt"))
        if len(paths) != expected.world_size:
            context["retention"] = "retained-incomplete-rank-quorum"
            raise RuntimeError(
                "Hybrid-EP child-result rank quorum is incomplete: "
                f"expected {expected.world_size}, found {len(paths)}; "
                f"artifacts retained at {result_dir}"
            )

        expected_signature = _input_signature(expected)
        signature_errors = expected_signature.validate(strict=True)
        if signature_errors:
            raise RuntimeError(f"Invalid expected hybrid-EP signature: {signature_errors[0]}")
        expected_tolerance = _output_tolerance(expected)
        rank_shape = (expected.tokens_per_rank, expected.hidden_size)
        assignment_shape = (expected.tokens_per_rank, expected.top_k)
        # Bound untrusted torch payloads before deserializing them.
        payload_limit = (
            8 * expected.tokens_per_rank * expected.hidden_size * 4
            + 4 * expected.tokens_per_rank * expected.top_k * 8
            + 2_000_000
        )
        payloads: dict[int, dict[str, Any]] = {}
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Hybrid-EP child result must be a regular file: {path}")
            if path.stat().st_size > payload_limit:
                raise RuntimeError(f"Hybrid-EP child result exceeds its size bound: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Invalid hybrid-EP child-result payload at {path}")
            if payload.get("schema") != MOE_HYBRID_EP_RESULT_SCHEMA:
                raise RuntimeError(f"Invalid hybrid-EP child-result schema at {path}")
            if payload.get("token") != context["token"]:
                raise RuntimeError(f"Hybrid-EP child-result token mismatch at {path}")
            observed_contract = MoEHybridEPResultContract.from_dict(payload.get("contract"))
            if observed_contract != expected:
                raise RuntimeError(f"Hybrid-EP child-result workload mismatch at {path}")
            rank = int(payload.get("rank", -1))
            if rank < 0 or rank >= expected.world_size or rank in payloads:
                raise RuntimeError(f"Invalid or duplicate hybrid-EP rank at {path}")
            if path.name != f"rank-{rank}.pt":
                raise RuntimeError(f"Hybrid-EP child-result filename/rank mismatch at {path}")
            created_wall_ns = int(payload.get("created_wall_ns", 0))
            if not launch_wall_ns <= created_wall_ns <= finish_wall_ns:
                raise RuntimeError(f"Stale hybrid-EP child-result payload at {path}")
            if int(payload.get("launch_wall_ns", 0)) != int(launch_wall_ns):
                raise RuntimeError(f"Hybrid-EP child-result launch identity mismatch at {path}")
            if payload.get("reference_execution") != MOE_HYBRID_EP_REFERENCE_EXECUTION:
                raise RuntimeError(
                    f"Hybrid-EP child result lacks canonical replay evidence at {path}"
                )
            if payload.get("timed_range") != expected.profile_range:
                raise RuntimeError(f"Hybrid-EP child timed-range identity mismatch at {path}")
            if int(payload.get("timed_iterations_completed", -1)) != expected.iterations:
                raise RuntimeError(f"Hybrid-EP child timed-iteration count mismatch at {path}")
            if int(payload.get("parameter_count", -1)) != _per_rank_parameter_count(expected):
                raise RuntimeError(f"Hybrid-EP child parameter count mismatch at {path}")

            for name in ("inputs", "targets", "verify_output", "reference_output"):
                payload[name] = _validate_tensor(
                    payload.get(name), name=name, shape=rank_shape, dtype=_dtype(expected)
                )
            for name in ("route_assignments", "reference_route_assignments"):
                payload[name] = _validate_tensor(
                    payload.get(name), name=name, shape=assignment_shape, dtype=torch.int64
                )
            if not torch.equal(
                payload["route_assignments"], payload["reference_route_assignments"]
            ):
                raise RuntimeError(f"Hybrid-EP canonical replay route assignments differ at {path}")
            _assert_outputs_match(payload["verify_output"], payload["reference_output"], expected)

            signature_payload = payload.get("input_signature")
            if not isinstance(signature_payload, dict):
                raise RuntimeError(f"Hybrid-EP input signature is missing at {path}")
            signature = InputSignature.from_dict(signature_payload)
            if signature.to_dict() != expected_signature.to_dict():
                raise RuntimeError(f"Hybrid-EP input signature mismatch at {path}")
            if payload.get("output_tolerance") != list(expected_tolerance):
                raise RuntimeError(f"Hybrid-EP output tolerance mismatch at {path}")
            metrics = payload.get("custom_metrics")
            if not isinstance(metrics, dict) or not metrics:
                raise RuntimeError(f"Hybrid-EP child metrics are missing at {path}")
            if any(
                not isinstance(value, int | float) or not math.isfinite(float(value))
                for value in metrics.values()
            ):
                raise RuntimeError(f"Hybrid-EP child metrics are non-numeric at {path}")
            time_per_iter_ms = float(payload.get("time_per_iter_ms", 0.0))
            if not math.isfinite(time_per_iter_ms) or time_per_iter_ms <= 0:
                raise RuntimeError(f"Hybrid-EP child timing is invalid at {path}")
            payloads[rank] = payload

        ranked = [payloads[rank] for rank in range(expected.world_size)]
        rank0_metrics = {key: float(value) for key, value in ranked[0]["custom_metrics"].items()}
        rank0_time = float(ranked[0]["time_per_iter_ms"])
        timing_values = re.findall(
            r"(?m)^\s*rank0 time_per_iter_ms:\s*(\S+)\s*$",
            stdout,
        )
        if len(timing_values) != 1:
            raise RuntimeError("Hybrid-EP stdout must contain exactly one rank-0 timing sample")
        try:
            reported_time = float(timing_values[0])
        except ValueError as exc:
            raise RuntimeError("Hybrid-EP rank-0 timing sample is not numeric") from exc
        if not math.isclose(reported_time, rank0_time, rel_tol=5e-7, abs_tol=5e-7):
            raise RuntimeError("Hybrid-EP rank-0 timing sample does not match its rank receipt")

        aggregate = {
            name: torch.cat([payload[name] for payload in ranked], dim=0).contiguous()
            for name in (
                "inputs",
                "targets",
                "route_assignments",
                "verify_output",
                "reference_output",
            )
        }
        self._subprocess_verify_inputs = {
            "inputs": aggregate["inputs"],
            "targets": aggregate["targets"],
            "route_assignments": aggregate["route_assignments"],
            "reference_output": aggregate["reference_output"],
        }
        self._subprocess_verify_output = aggregate["verify_output"]
        self._subprocess_output_tolerance = expected_tolerance
        self._subprocess_input_signature = expected_signature
        self._moe_hybrid_ep_result_metrics = rank0_metrics
        self._moe_hybrid_ep_result_bundle = aggregate
        try:
            shutil.rmtree(result_dir)
        except OSError:
            context["retention"] = "retained-cleanup-failure"
        else:
            context["retention"] = "cleaned-after-success"

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        if self._moe_hybrid_ep_result_bundle is not None:
            return dict(self._subprocess_verify_inputs)
        return super().get_verify_inputs()  # type: ignore[misc]

    def get_verify_output(self) -> torch.Tensor:
        if self._moe_hybrid_ep_result_bundle is not None:
            return self._subprocess_verify_output.detach().clone()
        return super().get_verify_output()  # type: ignore[misc]

    def get_input_signature(self) -> InputSignature:
        if self._moe_hybrid_ep_result_bundle is not None:
            return self._subprocess_input_signature
        return super().get_input_signature()  # type: ignore[misc]

    def get_output_tolerance(self) -> tuple[float, float]:
        if self._moe_hybrid_ep_result_bundle is not None:
            return self._subprocess_output_tolerance
        return super().get_output_tolerance()  # type: ignore[misc]

    def validate_result(self) -> str | None:
        if self._moe_hybrid_ep_result_context is not None:
            if self._moe_hybrid_ep_result_bundle is None:
                return "Fresh full-rank hybrid-EP worker output is missing"
            return None
        hook = getattr(self, "_validate_local_result", None)
        if callable(hook):
            return hook()
        return super().validate_result()  # type: ignore[misc]


def moe_hybrid_ep_child_result_requested() -> bool:
    """Return whether this worker belongs to a callback-bound benchmark launch."""
    return bool(os.environ.get(MOE_HYBRID_EP_LAUNCH_WALL_NS_ENV))


def write_moe_hybrid_ep_child_result(
    *,
    contract: MoEHybridEPResultContract,
    rank: int,
    parameter_count: int,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    route_assignments: torch.Tensor,
    verify_output: torch.Tensor,
    reference_route_assignments: torch.Tensor,
    reference_output: torch.Tensor,
    custom_metrics: dict[str, float],
    time_per_iter_ms: float,
) -> bool:
    """Atomically write one rank's full measured output and independent replay."""
    launch_wall_ns = os.environ.get(MOE_HYBRID_EP_LAUNCH_WALL_NS_ENV)
    if not launch_wall_ns:
        return False
    result_dir_value = os.environ.get(MOE_HYBRID_EP_RESULT_DIR_ENV)
    token = os.environ.get(MOE_HYBRID_EP_RESULT_TOKEN_ENV)
    expected_label = os.environ.get(MOE_HYBRID_EP_RESULT_LABEL_ENV)
    if not all((result_dir_value, token, expected_label)):
        raise RuntimeError("Hybrid-EP child-result environment is incomplete")
    contract.validate()
    if expected_label != contract.label:
        raise RuntimeError("Hybrid-EP child-result label does not match its launch")
    if rank < 0 or rank >= contract.world_size:
        raise RuntimeError("Hybrid-EP child-result rank is outside the declared world")
    if parameter_count != _per_rank_parameter_count(contract):
        raise RuntimeError("Hybrid-EP worker parameter count does not match its workload")

    rank_shape = (contract.tokens_per_rank, contract.hidden_size)
    assignment_shape = (contract.tokens_per_rank, contract.top_k)
    tensors = {
        "inputs": _validate_tensor(inputs, name="inputs", shape=rank_shape, dtype=_dtype(contract)),
        "targets": _validate_tensor(
            targets, name="targets", shape=rank_shape, dtype=_dtype(contract)
        ),
        "route_assignments": _validate_tensor(
            route_assignments,
            name="route_assignments",
            shape=assignment_shape,
            dtype=torch.int64,
        ),
        "verify_output": _validate_tensor(
            verify_output, name="verify_output", shape=rank_shape, dtype=_dtype(contract)
        ),
        "reference_route_assignments": _validate_tensor(
            reference_route_assignments,
            name="reference_route_assignments",
            shape=assignment_shape,
            dtype=torch.int64,
        ),
        "reference_output": _validate_tensor(
            reference_output, name="reference_output", shape=rank_shape, dtype=_dtype(contract)
        ),
    }
    if not torch.equal(tensors["route_assignments"], tensors["reference_route_assignments"]):
        raise RuntimeError("Hybrid-EP canonical replay changed final route assignments")
    _assert_outputs_match(tensors["verify_output"], tensors["reference_output"], contract)
    if not custom_metrics or any(
        not isinstance(value, int | float) or not math.isfinite(float(value))
        for value in custom_metrics.values()
    ):
        raise RuntimeError("Hybrid-EP child metrics must be finite numeric values")
    time_per_iter_ms = float(time_per_iter_ms)
    if not math.isfinite(time_per_iter_ms) or time_per_iter_ms <= 0:
        raise RuntimeError("Hybrid-EP measured mean time must be finite and positive")

    signature = _input_signature(contract)
    payload = {
        "schema": MOE_HYBRID_EP_RESULT_SCHEMA,
        "token": token,
        "contract": contract.to_dict(),
        "rank": int(rank),
        "launch_wall_ns": int(launch_wall_ns),
        "created_wall_ns": time.time_ns(),
        "parameter_count": int(parameter_count),
        "reference_execution": MOE_HYBRID_EP_REFERENCE_EXECUTION,
        "timed_range": contract.profile_range,
        "timed_iterations_completed": contract.iterations,
        "time_per_iter_ms": time_per_iter_ms,
        "custom_metrics": {key: float(value) for key, value in custom_metrics.items()},
        "input_signature": signature.to_dict(),
        "output_tolerance": list(_output_tolerance(contract)),
        **tensors,
    }
    result_dir = Path(cast(str, result_dir_value))
    if result_dir.is_symlink() or not result_dir.is_dir():
        raise RuntimeError("Hybrid-EP child-result directory must be the prepared directory")
    temporary_path = result_dir / f".rank-{rank}-{os.getpid()}.tmp"
    destination = result_dir / f"rank-{rank}.pt"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, destination)
    return True


__all__ = [
    "MOE_HYBRID_EP_LAUNCH_WALL_NS_ENV",
    "MOE_HYBRID_EP_REFERENCE_EXECUTION",
    "MOE_HYBRID_EP_RESULT_CALLBACK",
    "MOE_HYBRID_EP_RESULT_DIR_ENV",
    "MOE_HYBRID_EP_RESULT_LABEL_ENV",
    "MOE_HYBRID_EP_RESULT_SCHEMA",
    "MOE_HYBRID_EP_RESULT_TOKEN_ENV",
    "MoEHybridEPChildResultMixin",
    "MoEHybridEPResultContract",
    "moe_hybrid_ep_child_result_requested",
    "write_moe_hybrid_ep_child_result",
]
