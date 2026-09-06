"""Fresh full-rank result transport for the Chapter 4 DDP overlap pair."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn.functional as functional

from core.benchmark.verification import InputSignature, PrecisionFlags

RESULT_CALLBACK = "consume_ddp_overlap_child_results"
RESULT_DIR_ENV = "AISP_DDP_OVERLAP_RESULT_DIR"
RESULT_TOKEN_ENV = "AISP_DDP_OVERLAP_RESULT_TOKEN"
RESULT_VARIANT_ENV = "AISP_DDP_OVERLAP_VARIANT"
LAUNCH_WALL_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS"
RESULT_SCHEMA = "aisp.ch04-ddp-overlap.child-result.v1"
OUTPUT_TOLERANCE = (0.1, 1.0)
SEED = 42
LEARNING_RATE = 0.01

_PARAMETER_NAMES = (
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "fc3.weight",
    "fc3.bias",
)


@dataclass(frozen=True)
class DdpOverlapResultContract:
    variant: str
    world_size: int
    batch_size: int
    hidden_size: int
    iterations: int
    warmup: int
    seed: int
    learning_rate: float
    tf32: bool

    def validate(self) -> None:
        if self.variant not in {"no-overlap", "overlap"}:
            raise ValueError(f"Unsupported DDP overlap variant: {self.variant!r}")
        if isinstance(self.world_size, bool) or self.world_size < 2:
            raise ValueError("DDP overlap child results require world_size >= 2")
        for name in ("batch_size", "hidden_size", "iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"DDP overlap {name} must be a positive integer")
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int) or self.warmup < 0:
            raise ValueError("DDP overlap warmup must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("DDP overlap seed must be a non-negative integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("DDP overlap learning_rate must be finite and positive")
        if not isinstance(self.tf32, bool):
            raise TypeError("DDP overlap tf32 must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DdpOverlapResultContract:
        if not isinstance(payload, dict):
            raise TypeError("DDP overlap result contract must be a dictionary")
        try:
            contract = cls(**payload)
        except TypeError as exc:
            raise ValueError("Invalid DDP overlap result contract fields") from exc
        contract.validate()
        return contract


@dataclass(frozen=True)
class DdpOverlapWorkloadResult:
    contract: DdpOverlapResultContract
    rank: int
    data: torch.Tensor
    target: torch.Tensor
    timed_output: torch.Tensor
    post_update_output: torch.Tensor
    reference_timed_output: torch.Tensor
    reference_post_update_output: torch.Tensor
    time_per_iter_ms: float


def tf32_is_enabled() -> bool:
    if not torch.cuda.is_available():
        return False
    get_precision = getattr(torch, "get_float32_matmul_precision", None)
    if callable(get_precision):
        return str(get_precision()) != "highest"
    return bool(torch.backends.cuda.matmul.allow_tf32)


def make_result_contract(
    *,
    variant: str,
    world_size: int,
    batch_size: int,
    hidden_size: int,
    iterations: int,
    warmup: int,
    seed: int = SEED,
    learning_rate: float = LEARNING_RATE,
    tf32: bool | None = None,
) -> DdpOverlapResultContract:
    contract = DdpOverlapResultContract(
        variant=variant,
        world_size=int(world_size),
        batch_size=int(batch_size),
        hidden_size=int(hidden_size),
        iterations=int(iterations),
        warmup=int(warmup),
        seed=int(seed),
        learning_rate=float(learning_rate),
        tf32=tf32_is_enabled() if tf32 is None else tf32,
    )
    contract.validate()
    return contract


def _parameter_count(contract: DdpOverlapResultContract) -> int:
    return 2 * contract.hidden_size**2 + 3 * contract.hidden_size + 1


def _signature(contract: DdpOverlapResultContract) -> InputSignature:
    return InputSignature(
        shapes={
            "data": (
                contract.world_size,
                contract.batch_size,
                contract.hidden_size,
            ),
            "target": (contract.world_size, contract.batch_size, 1),
            "output": (contract.world_size, contract.batch_size, 1),
        },
        dtypes={
            "data": str(torch.float32),
            "target": str(torch.float32),
            "output": str(torch.float32),
        },
        batch_size=contract.batch_size,
        parameter_count=_parameter_count(contract),
        precision_flags=PrecisionFlags(tf32=contract.tf32),
        world_size=contract.world_size,
        ranks=list(range(contract.world_size)),
        per_rank_batch_size=contract.batch_size,
        collective_type="all_reduce",
        collective_algorithm="sum_then_divide_world_size",
    )


def reference_training_outputs(
    *,
    initial_state: Mapping[str, torch.Tensor],
    data: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    learning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay SGD with functional linear algebra, outside the measured path."""
    if set(initial_state) != set(_PARAMETER_NAMES):
        raise ValueError("DDP overlap reference received an unexpected model state")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("DDP overlap reference steps must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("DDP overlap reference learning_rate must be finite and positive")

    parameters = {
        name: initial_state[name].detach().clone().requires_grad_(True)
        for name in _PARAMETER_NAMES
    }
    ordered = tuple(parameters[name] for name in _PARAMETER_NAMES)

    def forward() -> torch.Tensor:
        hidden = functional.linear(
            data,
            parameters["fc1.weight"],
            parameters["fc1.bias"],
        )
        hidden = functional.relu(hidden, inplace=False)
        hidden = functional.linear(
            hidden,
            parameters["fc2.weight"],
            parameters["fc2.bias"],
        )
        hidden = functional.relu(hidden, inplace=False)
        return functional.linear(
            hidden,
            parameters["fc3.weight"],
            parameters["fc3.bias"],
        )

    timed_reference: torch.Tensor | None = None
    for _ in range(steps):
        output = forward()
        loss = functional.mse_loss(output, target)
        gradients = torch.autograd.grad(loss, ordered)
        timed_reference = output.detach().clone()
        with torch.no_grad():
            for parameter, gradient in zip(ordered, gradients, strict=True):
                parameter.sub_(gradient, alpha=learning_rate)
    if timed_reference is None:  # pragma: no cover - guarded by steps validation
        raise RuntimeError("DDP overlap reference did not execute")
    with torch.no_grad():
        post_update_reference = forward().detach().clone()
    return timed_reference, post_update_reference


def _validate_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"DDP overlap {name} must be a tensor")
    if tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
        raise ValueError(f"DDP overlap {name} has unexpected shape or dtype")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"DDP overlap {name} contains non-finite values")


def validate_workload_result(result: DdpOverlapWorkloadResult) -> None:
    contract = result.contract
    contract.validate()
    if result.rank < 0 or result.rank >= contract.world_size:
        raise ValueError(f"Invalid DDP overlap rank: {result.rank}")
    if not math.isfinite(result.time_per_iter_ms) or result.time_per_iter_ms <= 0:
        raise ValueError("DDP overlap time_per_iter_ms must be finite and positive")
    expected_shapes = {
        "data": (contract.batch_size, contract.hidden_size),
        "target": (contract.batch_size, 1),
        "timed_output": (contract.batch_size, 1),
        "post_update_output": (contract.batch_size, 1),
        "reference_timed_output": (contract.batch_size, 1),
        "reference_post_update_output": (contract.batch_size, 1),
    }
    for name, shape in expected_shapes.items():
        _validate_tensor(name, getattr(result, name), shape=shape)
    for actual_name, reference_name in (
        ("timed_output", "reference_timed_output"),
        ("post_update_output", "reference_post_update_output"),
    ):
        try:
            torch.testing.assert_close(
                getattr(result, actual_name),
                getattr(result, reference_name),
                rtol=OUTPUT_TOLERANCE[0],
                atol=OUTPUT_TOLERANCE[1],
            )
        except AssertionError as exc:
            raise RuntimeError(
                f"DDP overlap full {actual_name} differs from its independent reference"
            ) from exc


def run_ddp_overlap_worker(
    benchmark: Any,
    *,
    variant: str,
    iterations: int,
    warmup: int,
) -> DdpOverlapWorkloadResult:
    """Run the actual benchmark and its independent post-timing oracle."""
    if iterations <= 0:
        raise ValueError("DDP overlap worker iterations must be positive")
    if warmup < 0:
        raise ValueError("DDP overlap worker warmup must be non-negative")
    benchmark.setup()
    try:
        if benchmark.model is None or benchmark.data is None or benchmark.target is None:
            raise RuntimeError("DDP overlap benchmark setup did not produce model inputs")
        if not benchmark.device.type == "cuda":
            raise RuntimeError("DDP overlap production worker requires CUDA")
        base_model = getattr(benchmark.model, "module", benchmark.model)
        initial_state = {
            name: tensor.detach().clone()
            for name, tensor in base_model.state_dict().items()
        }
        data = benchmark.data.detach().clone()
        target = benchmark.target.detach().clone()

        for _ in range(warmup):
            benchmark.benchmark_fn()
        torch.cuda.synchronize(benchmark.device)
        dist.barrier()
        torch.cuda.synchronize(benchmark.device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            benchmark.benchmark_fn()
        end_event.record()
        torch.cuda.synchronize(benchmark.device)
        local_elapsed_ms = float(start_event.elapsed_time(end_event))
        elapsed = torch.tensor(
            [local_elapsed_ms],
            dtype=torch.float32,
            device=benchmark.device,
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        time_per_iter_ms = float(elapsed.item()) / iterations

        if benchmark.output is None:
            raise RuntimeError("DDP overlap timed execution produced no output")
        timed_output = benchmark.output.detach().clone()
        with torch.no_grad():
            post_update_output = base_model(data).detach().clone()
        reference_timed_output, reference_post_update_output = reference_training_outputs(
            initial_state=initial_state,
            data=data,
            target=target,
            steps=warmup + iterations,
            learning_rate=LEARNING_RATE,
        )
        torch.cuda.synchronize(benchmark.device)

        contract = make_result_contract(
            variant=variant,
            world_size=int(benchmark.world_size),
            batch_size=int(benchmark.batch_size),
            hidden_size=int(benchmark.hidden_size),
            iterations=iterations,
            warmup=warmup,
            seed=SEED,
            learning_rate=LEARNING_RATE,
        )
        result = DdpOverlapWorkloadResult(
            contract=contract,
            rank=int(benchmark.rank),
            data=data,
            target=target,
            timed_output=timed_output,
            post_update_output=post_update_output,
            reference_timed_output=reference_timed_output,
            reference_post_update_output=reference_post_update_output,
            time_per_iter_ms=time_per_iter_ms,
        )
        validate_workload_result(result)
        return result
    finally:
        benchmark.teardown()


class DdpOverlapChildResultMixin:
    """Validate a fresh rank quorum and expose the actual measured outputs."""

    _ddp_overlap_result_context: dict[str, Any] | None = None
    _ddp_overlap_result_bundle: dict[str, Any] | None = None

    def prepare_ddp_overlap_child_result(
        self,
        contract: DdpOverlapResultContract,
    ) -> dict[str, str]:
        contract.validate()
        result_dir = Path(tempfile.mkdtemp(prefix="aisp-ch04-ddp-overlap-result-"))
        token = uuid.uuid4().hex
        self._ddp_overlap_result_context = {
            "result_dir": result_dir,
            "token": token,
            "contract": contract,
            "retention": "pending-child-result",
        }
        self._ddp_overlap_result_bundle = None
        for attribute in (
            "_subprocess_verify_inputs",
            "_subprocess_verify_output",
            "_subprocess_input_signature",
            "_subprocess_output_tolerance",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return {
            RESULT_DIR_ENV: str(result_dir),
            RESULT_TOKEN_ENV: token,
            RESULT_VARIANT_ENV: contract.variant,
        }

    def consume_ddp_overlap_child_results(
        self,
        *,
        launch_wall_ns: int,
        finish_wall_ns: int,
        returncode: int,
        **_: Any,
    ) -> None:
        context = self._ddp_overlap_result_context
        if context is None:
            raise RuntimeError("DDP overlap child-result callback has no launch context")
        result_dir = cast(Path, context["result_dir"])
        context["retention"] = "retained-invalid-child-result"
        if returncode != 0:
            context["retention"] = "retained-child-failure"
            raise RuntimeError(
                "DDP overlap child-result callback requires a clean child exit; "
                f"artifacts retained at {result_dir}"
            )

        expected = cast(DdpOverlapResultContract, context["contract"])
        expected_signature = _signature(expected)
        paths = sorted(result_dir.glob("rank-*.pt"))
        if len(paths) != expected.world_size:
            context["retention"] = "retained-incomplete-rank-quorum"
            raise RuntimeError(
                "DDP overlap child-result rank quorum is incomplete: "
                f"expected {expected.world_size}, found {len(paths)}; "
                f"artifacts retained at {result_dir}"
            )

        payloads: dict[int, dict[str, Any]] = {}
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"DDP overlap child result must be a regular file: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("schema") != RESULT_SCHEMA:
                raise RuntimeError(f"Invalid DDP overlap child-result payload at {path}")
            if payload.get("token") != context["token"]:
                raise RuntimeError(f"DDP overlap child-result token mismatch at {path}")
            if payload.get("variant") != expected.variant:
                raise RuntimeError(f"DDP overlap child-result variant mismatch at {path}")
            observed_contract = DdpOverlapResultContract.from_dict(payload.get("contract"))
            if observed_contract != expected:
                raise RuntimeError(f"DDP overlap child-result workload mismatch at {path}")
            rank = int(payload.get("rank", -1))
            if rank < 0 or rank >= expected.world_size or rank in payloads:
                raise RuntimeError(f"Invalid or duplicate DDP overlap rank at {path}")
            if path.name != f"rank-{rank}.pt":
                raise RuntimeError(f"DDP overlap child filename/rank mismatch at {path}")
            created = int(payload.get("created_wall_ns", 0))
            if not launch_wall_ns <= created <= finish_wall_ns:
                raise RuntimeError(f"Stale DDP overlap child result at {path}")
            if int(payload.get("launch_wall_ns", 0)) != launch_wall_ns:
                raise RuntimeError(f"DDP overlap child launch identity mismatch at {path}")
            observed_signature = InputSignature.from_dict(payload.get("input_signature"))
            if observed_signature != expected_signature:
                raise RuntimeError(f"DDP overlap child input signature mismatch at {path}")
            result = DdpOverlapWorkloadResult(
                contract=observed_contract,
                rank=rank,
                data=payload.get("data"),
                target=payload.get("target"),
                timed_output=payload.get("timed_output"),
                post_update_output=payload.get("post_update_output"),
                reference_timed_output=payload.get("reference_timed_output"),
                reference_post_update_output=payload.get("reference_post_update_output"),
                time_per_iter_ms=float(payload.get("time_per_iter_ms", float("nan"))),
            )
            validate_workload_result(result)
            payloads[rank] = payload

        ordered = [payloads[rank] for rank in range(expected.world_size)]
        for tensor_name in (
            "data",
            "target",
            "timed_output",
            "post_update_output",
            "reference_timed_output",
            "reference_post_update_output",
        ):
            first = ordered[0][tensor_name]
            for rank, payload in enumerate(ordered[1:], 1):
                try:
                    torch.testing.assert_close(
                        payload[tensor_name],
                        first,
                        rtol=0,
                        atol=0,
                    )
                except AssertionError as exc:
                    raise RuntimeError(
                        f"DDP overlap seed-{expected.seed} rank parity failed for "
                        f"rank {rank} {tensor_name}"
                    ) from exc

        self._subprocess_verify_inputs = {
            "data": torch.stack([payload["data"] for payload in ordered]),
            "target": torch.stack([payload["target"] for payload in ordered]),
        }
        self._subprocess_verify_output = torch.stack(
            [payload["timed_output"] for payload in ordered]
        )
        self._subprocess_input_signature = expected_signature
        self._subprocess_output_tolerance = OUTPUT_TOLERANCE
        self._ddp_overlap_result_bundle = {
            "contract": expected,
            "rank_outputs": [payload["timed_output"] for payload in ordered],
        }
        try:
            shutil.rmtree(result_dir)
        except OSError:
            context["retention"] = "retained-cleanup-failure"
        else:
            context["retention"] = "cleaned-after-success"

    def require_ddp_overlap_child_result(self) -> None:
        if self._ddp_overlap_result_bundle is None:
            context = self._ddp_overlap_result_context
            retained = context.get("result_dir") if context else "unavailable"
            raise RuntimeError(
                "DDP overlap verification requires a fresh full-rank measured result; "
                f"retained path: {retained}"
            )

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        if self._ddp_overlap_result_bundle is not None:
            return dict(self._subprocess_verify_inputs)
        return super().get_verify_inputs()  # type: ignore[misc]

    def get_verify_output(self) -> torch.Tensor:
        if self._ddp_overlap_result_bundle is not None:
            return self._subprocess_verify_output.detach().clone()
        return super().get_verify_output()  # type: ignore[misc]

    def get_input_signature(self) -> InputSignature:
        if self._ddp_overlap_result_bundle is not None:
            return self._subprocess_input_signature
        return super().get_input_signature()  # type: ignore[misc]

    def get_output_tolerance(self) -> tuple[float, float]:
        if self._ddp_overlap_result_bundle is not None:
            return self._subprocess_output_tolerance
        return super().get_output_tolerance()  # type: ignore[misc]


def write_ddp_overlap_child_result(result: DdpOverlapWorkloadResult) -> bool:
    """Atomically write one rank's validated result for a harness launch."""
    validate_workload_result(result)
    result_dir_value = os.environ.get(RESULT_DIR_ENV)
    token = os.environ.get(RESULT_TOKEN_ENV)
    expected_variant = os.environ.get(RESULT_VARIANT_ENV)
    launch_wall_ns = os.environ.get(LAUNCH_WALL_NS_ENV)
    if not launch_wall_ns:
        return False
    if not all((result_dir_value, token, expected_variant)):
        raise RuntimeError("DDP overlap child-result environment is incomplete")
    if expected_variant != result.contract.variant:
        raise RuntimeError(
            f"DDP overlap child variant mismatch: "
            f"{expected_variant!r} != {result.contract.variant!r}"
        )

    result_dir = Path(cast(str, result_dir_value))
    if result_dir.is_symlink() or not result_dir.is_dir():
        raise RuntimeError(
            "DDP overlap child-result directory must be the prepared regular directory"
        )
    payload = {
        "schema": RESULT_SCHEMA,
        "token": token,
        "variant": result.contract.variant,
        "rank": result.rank,
        "contract": result.contract.to_dict(),
        "launch_wall_ns": int(launch_wall_ns),
        "created_wall_ns": time.time_ns(),
        "input_signature": _signature(result.contract).to_dict(),
        "time_per_iter_ms": result.time_per_iter_ms,
        **{
            name: getattr(result, name).detach().to(device="cpu").contiguous()
            for name in (
                "data",
                "target",
                "timed_output",
                "post_update_output",
                "reference_timed_output",
                "reference_post_update_output",
            )
        },
    }
    temporary = result_dir / f".rank-{result.rank}-{os.getpid()}.tmp"
    destination = result_dir / f"rank-{result.rank}.pt"
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return True


__all__ = [
    "DdpOverlapChildResultMixin",
    "DdpOverlapResultContract",
    "DdpOverlapWorkloadResult",
    "LEARNING_RATE",
    "OUTPUT_TOLERANCE",
    "RESULT_CALLBACK",
    "SEED",
    "make_result_contract",
    "reference_training_outputs",
    "run_ddp_overlap_worker",
    "validate_workload_result",
    "write_ddp_overlap_child_result",
]
