"""Fresh full-output transport for Chapter 4 gradient-compression workers."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from core.benchmark.distributed_work_contract import (
    BARRIER_BEFORE_TIMED_CLOSE,
    DECLARED_ALGORITHM_EVIDENCE,
    WAIT_FOR_ASYNC_BEFORE_TIMED_CLOSE,
    DistributedRankWorkReceipt,
    validate_distributed_work_receipts,
)
from core.benchmark.verification import DistributedTopology, InputSignature, PrecisionFlags

RESULT_CALLBACK = "consume_gradient_compression_child_results"
RESULT_DIR_ENV = "AISP_GRADIENT_COMPRESSION_RESULT_DIR"
RESULT_TOKEN_ENV = "AISP_GRADIENT_COMPRESSION_RESULT_TOKEN"
RESULT_VARIANT_ENV = "AISP_GRADIENT_COMPRESSION_VARIANT"
LAUNCH_WALL_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS"
RESULT_SCHEMA = "aisp.ch04-gradient-compression.child-result.v2"
COLLECTIVE_ALGORITHM = "backend_selected_all_reduce"


@dataclass(frozen=True)
class GradientCompressionResultContract:
    """Exact workload and result contract shared by parent and two-rank worker."""

    variant: str
    pair_compression: str
    comm_only: bool
    world_size: int
    tensor_size_mb: int
    bucket_mb: int
    iterations: int
    warmup: int
    seed: int
    output_rtol: float
    output_atol: float

    @property
    def effective_compression(self) -> str:
        if self.comm_only and self.variant == "baseline":
            return "none"
        return self.pair_compression

    @property
    def process_collective_mode(self) -> str:
        if self.comm_only:
            return "functional_out_of_place_constant_payload"
        return "in_place_preallocated_rewritten_payload"

    @property
    def numel(self) -> int:
        return (self.tensor_size_mb * 1024 * 1024) // torch.float32.itemsize

    @property
    def full_gradient_bytes(self) -> int:
        return self.numel * torch.float32.itemsize

    @property
    def gradient_bucket_bytes(self) -> int:
        if self.bucket_mb > 0:
            return min(self.bucket_mb * 1024 * 1024, self.full_gradient_bytes)
        return self.full_gradient_bytes

    @property
    def expected_collectives_per_rank(self) -> int:
        bucket_count = (
            self.full_gradient_bytes + self.gradient_bucket_bytes - 1
        ) // self.gradient_bucket_bytes
        per_iteration = bucket_count
        if self.effective_compression == "int8" and not self.comm_only:
            per_iteration += 1  # global MAX used to derive a common INT8 scale
        return self.iterations * per_iteration

    @property
    def raw_result_tensor_bytes(self) -> int:
        # Every rank writes its input and full FP32 timed output; rank zero also
        # writes the independent full FP32 reference.
        return (2 * self.world_size + 1) * self.full_gradient_bytes

    def validate(self) -> None:
        if self.variant not in {"baseline", "optimized"}:
            raise ValueError(f"Unsupported gradient-compression variant: {self.variant!r}")
        if self.pair_compression not in {"fp16", "int8"}:
            raise ValueError(
                f"Unsupported gradient-compression pair: {self.pair_compression!r}"
            )
        if not isinstance(self.comm_only, bool):
            raise TypeError("Gradient-compression comm_only must be bool")
        if self.world_size != 2:
            raise ValueError("Gradient-compression worker requires exactly two ranks")
        for name in ("tensor_size_mb", "iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Gradient-compression {name} must be a positive integer")
        for name in ("bucket_mb", "warmup", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Gradient-compression {name} must be a non-negative integer"
                )
        if self.comm_only and self.bucket_mb != 0:
            raise ValueError("Communication-only gradient compression uses one full payload")
        if not self.comm_only:
            if self.variant == "baseline" and self.bucket_mb <= 0:
                raise ValueError("Baseline compression requires its declared small bucket")
            if self.variant == "optimized" and self.bucket_mb != 0:
                raise ValueError("Optimized compression requires one full-gradient bucket")
        for name in ("output_rtol", "output_atol"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Gradient-compression {name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Any) -> GradientCompressionResultContract:
        if not isinstance(payload, dict):
            raise TypeError("Gradient-compression result contract must be a dictionary")
        try:
            contract = cls(**payload)
        except TypeError as exc:
            raise ValueError("Invalid gradient-compression result contract fields") from exc
        contract.validate()
        return contract


def make_result_contract(
    *,
    variant: str,
    pair_compression: str,
    comm_only: bool,
    world_size: int,
    tensor_size_mb: int,
    bucket_mb: int,
    iterations: int,
    warmup: int,
    seed: int,
    output_tolerance: tuple[float, float],
) -> GradientCompressionResultContract:
    contract = GradientCompressionResultContract(
        variant=variant,
        pair_compression=pair_compression,
        comm_only=comm_only,
        world_size=int(world_size),
        tensor_size_mb=int(tensor_size_mb),
        bucket_mb=int(bucket_mb),
        iterations=int(iterations),
        warmup=int(warmup),
        seed=int(seed),
        output_rtol=float(output_tolerance[0]),
        output_atol=float(output_tolerance[1]),
    )
    contract.validate()
    return contract


def distributed_topology(contract: GradientCompressionResultContract) -> DistributedTopology:
    """Return the measured transport contract, including the actual bucket size."""

    contract.validate()
    return DistributedTopology(
        world_size=contract.world_size,
        ranks=list(range(contract.world_size)),
        per_rank_batch_size=contract.numel,
        collective_type="all_reduce",
        collective_algorithm=COLLECTIVE_ALGORITHM,
        gradient_bucket_bytes=contract.gradient_bucket_bytes,
        barrier_policy=BARRIER_BEFORE_TIMED_CLOSE,
        async_completion_policy=WAIT_FOR_ASYNC_BEFORE_TIMED_CLOSE,
    )


def input_signature(contract: GradientCompressionResultContract) -> InputSignature:
    """Return pair-comparison fields shared by control and candidate.

    The physical bucket size is deliberately excluded here because changing the
    number of collective buckets is the optimization under test. It remains
    mandatory in every measured rank receipt and is validated against the exact
    variant contract by the parent callback.
    """

    shape = (contract.numel,)
    rank_shapes = {f"rank_{rank}_input": shape for rank in range(contract.world_size)}
    rank_dtypes = {
        f"rank_{rank}_input": str(torch.float32)
        for rank in range(contract.world_size)
    }
    return InputSignature(
        shapes={**rank_shapes, "reference": shape, "output": shape},
        dtypes={
            **rank_dtypes,
            "reference": str(torch.float32),
            "output": str(torch.float32),
        },
        batch_size=contract.numel,
        parameter_count=0,
        precision_flags=PrecisionFlags(
            fp16=contract.effective_compression == "fp16"
        ),
        world_size=contract.world_size,
        ranks=list(range(contract.world_size)),
        per_rank_batch_size=contract.numel,
        collective_type="all_reduce",
        collective_algorithm=COLLECTIVE_ALGORITHM,
        barrier_policy=BARRIER_BEFORE_TIMED_CLOSE,
        async_completion_policy=WAIT_FOR_ASYNC_BEFORE_TIMED_CLOSE,
    )


def _iter_slices(numel: int, chunk_numel: int = 1 << 24):
    for start in range(0, numel, chunk_numel):
        yield slice(start, min(start + chunk_numel, numel))


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    for section in _iter_slices(tensor.numel()):
        if not bool(torch.isfinite(tensor[section]).all()):
            raise RuntimeError(f"Gradient-compression {name} contains non-finite values")


def assert_close_full(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    label: str,
) -> None:
    """Compare every element without allocating another giant full-size tensor."""

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError(f"Gradient-compression {label} shape/dtype mismatch")
    for section in _iter_slices(actual.numel()):
        try:
            torch.testing.assert_close(
                actual[section], expected[section], rtol=rtol, atol=atol
            )
        except AssertionError as exc:
            raise RuntimeError(
                f"Gradient-compression {label} differs in full-output section "
                f"starting at {section.start}"
            ) from exc


def _reference_scale(
    contract: GradientCompressionResultContract,
    rank_inputs: list[torch.Tensor],
    *,
    arithmetic_device: torch.device,
) -> torch.Tensor | None:
    if contract.effective_compression != "int8":
        return None
    maximum = torch.zeros((), dtype=torch.float32, device=arithmetic_device)
    for rank_input in rank_inputs:
        for section in _iter_slices(rank_input.numel()):
            device_section = rank_input[section].to(
                device=arithmetic_device,
                dtype=torch.float32,
            )
            maximum = torch.maximum(maximum, device_section.abs().max())
    limit = max(1, 127 // contract.world_size)
    scale = maximum / float(limit)
    return torch.where(scale == 0, torch.ones_like(scale), scale)


def _reference_arithmetic_device(execution_device_type: str) -> torch.device:
    """Return the device whose FP32 quantizer semantics produced the child output."""

    if execution_device_type == "cpu":
        return torch.device("cpu")
    if execution_device_type != "cuda":
        raise RuntimeError(
            "Gradient-compression execution device type must be 'cpu' or 'cuda'"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Gradient-compression CUDA child evidence requires CUDA for exact-backend "
            "INT8 oracle replay; CPU replay is not equivalent at rounding boundaries"
        )
    return torch.device("cuda", torch.cuda.current_device())


def _validate_execution_device_for_backend(
    *,
    backend: str,
    execution_device_type: str,
    label: str,
) -> None:
    expected_by_backend = {"gloo": "cpu", "nccl": "cuda"}
    expected_device_type = expected_by_backend.get(backend)
    if expected_device_type is None:
        raise RuntimeError(
            f"Gradient-compression execution backend is unsupported at {label}: {backend!r}"
        )
    if execution_device_type != expected_device_type:
        raise RuntimeError(
            "Gradient-compression execution device/backend evidence differs at "
            f"{label}: {execution_device_type!r} vs {backend!r}"
        )


def assert_reference_from_rank_inputs(
    contract: GradientCompressionResultContract,
    rank_inputs: list[torch.Tensor],
    reference: torch.Tensor,
    *,
    execution_device_type: str,
) -> None:
    """Recompute the full oracle, preserving INT8 execution-device arithmetic."""

    if len(rank_inputs) != contract.world_size:
        raise RuntimeError("Gradient-compression oracle requires every rank input")
    arithmetic_device = (
        _reference_arithmetic_device(execution_device_type)
        if contract.effective_compression == "int8"
        else torch.device("cpu")
    )
    scale = _reference_scale(
        contract,
        rank_inputs,
        arithmetic_device=arithmetic_device,
    )
    limit = max(1, 127 // contract.world_size)
    for section in _iter_slices(contract.numel):
        if contract.effective_compression == "none":
            expected = torch.zeros_like(rank_inputs[0][section], dtype=torch.float32)
            for rank_input in rank_inputs:
                expected.add_(rank_input[section])
        elif contract.effective_compression == "fp16":
            expected_half = torch.zeros_like(rank_inputs[0][section], dtype=torch.float16)
            for rank_input in rank_inputs:
                expected_half.add_(rank_input[section].to(torch.float16))
            expected = expected_half.float()
        else:
            assert scale is not None
            # CUDA and CPU FP32 division can land on opposite sides of an exact
            # half-integer for a tiny number of values in a GiB-scale tensor.
            # Replaying bounded sections on the recorded execution device keeps
            # this an independent input-derived oracle without weakening the
            # tolerance or retaining another full-size device tensor.
            expected_int = torch.zeros(
                rank_inputs[0][section].shape,
                dtype=torch.int32,
                device=arithmetic_device,
            )
            for rank_input in rank_inputs:
                quantized = (
                    rank_input[section]
                    .to(device=arithmetic_device, dtype=torch.float32)
                    .div(scale)
                    .round()
                    .clamp(-limit, limit)
                    .to(torch.int32)
                )
                expected_int.add_(quantized)
            expected = expected_int.float().mul_(scale)
            if expected.device != reference.device:
                expected = expected.to(device=reference.device)
        try:
            torch.testing.assert_close(
                reference[section],
                expected,
                rtol=contract.output_rtol,
                atol=contract.output_atol,
            )
        except AssertionError as exc:
            raise RuntimeError(
                "Gradient-compression independent reference does not match all "
                f"rank inputs at section starting {section.start}"
            ) from exc


def _load_tensor(path: Path, *, numel: int, label: str) -> torch.Tensor:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Gradient-compression {label} must be a regular file: {path}")
    tensor = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"Gradient-compression {label} is not a tensor at {path}")
    if tensor.dtype != torch.float32 or tensor.shape != (numel,):
        raise RuntimeError(f"Gradient-compression {label} shape/dtype mismatch at {path}")
    _require_finite(label, tensor)
    return tensor


class GradientCompressionChildResultMixin:
    """Validate a fresh two-rank quorum before exposing measured full outputs."""

    _gradient_compression_result_context: dict[str, Any] | None = None
    _gradient_compression_result_bundle: dict[str, Any] | None = None

    def prepare_gradient_compression_child_result(
        self, contract: GradientCompressionResultContract
    ) -> dict[str, str]:
        contract.validate()
        result_dir = Path(tempfile.mkdtemp(prefix="aisp-gradient-compression-result-"))
        token = uuid.uuid4().hex
        self._gradient_compression_result_context = {
            "result_dir": result_dir,
            "token": token,
            "contract": contract,
            "raw_result_tensor_bytes": contract.raw_result_tensor_bytes,
            "retention": "pending-child-result",
        }
        self._gradient_compression_result_bundle = None
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

    def consume_gradient_compression_child_results(
        self,
        *,
        launch_wall_ns: int,
        finish_wall_ns: int,
        returncode: int,
        **_: Any,
    ) -> None:
        context = self._gradient_compression_result_context
        if context is None:
            raise RuntimeError(
                "Gradient-compression child-result callback has no launch context"
            )
        result_dir = cast(Path, context["result_dir"])
        context["retention"] = "retained-invalid-child-result"
        if returncode != 0:
            context["retention"] = "retained-child-failure"
            raise RuntimeError(
                "Gradient-compression child-result callback requires a clean child exit; "
                f"artifacts retained at {result_dir}"
            )

        expected = cast(GradientCompressionResultContract, context["contract"])
        metadata_paths = sorted(result_dir.glob("rank-*.json"))
        if len(metadata_paths) != expected.world_size:
            context["retention"] = "retained-incomplete-rank-quorum"
            raise RuntimeError(
                "Gradient-compression child-result rank quorum is incomplete: "
                f"expected {expected.world_size}, found {len(metadata_paths)}; "
                f"artifacts retained at {result_dir}"
            )

        payloads: dict[int, dict[str, Any]] = {}
        receipts: list[DistributedRankWorkReceipt] = []
        execution_device_types: set[str] = set()
        raw_bytes = 0
        for path in metadata_paths:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(
                    f"Gradient-compression metadata must be a regular file: {path}"
                )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Invalid gradient-compression child metadata at {path}"
                ) from exc
            if not isinstance(payload, dict) or payload.get("schema") != RESULT_SCHEMA:
                raise RuntimeError(
                    f"Invalid gradient-compression child-result schema at {path}"
                )
            if payload.get("token") != context["token"]:
                raise RuntimeError(f"Gradient-compression token mismatch at {path}")
            observed = GradientCompressionResultContract.from_dict(payload.get("contract"))
            if observed != expected:
                raise RuntimeError(f"Gradient-compression workload mismatch at {path}")
            rank = int(payload.get("rank", -1))
            if rank < 0 or rank >= expected.world_size or rank in payloads:
                raise RuntimeError(f"Invalid or duplicate gradient-compression rank at {path}")
            if path.name != f"rank-{rank}.json":
                raise RuntimeError(
                    f"Gradient-compression child filename/rank mismatch at {path}"
                )
            created = int(payload.get("created_wall_ns", 0))
            if not launch_wall_ns <= created <= finish_wall_ns:
                raise RuntimeError(f"Stale gradient-compression child result at {path}")
            if int(payload.get("launch_wall_ns", 0)) != int(launch_wall_ns):
                raise RuntimeError(
                    f"Gradient-compression launch identity mismatch at {path}"
                )
            if payload.get("algorithm_evidence") != DECLARED_ALGORITHM_EVIDENCE:
                raise RuntimeError(
                    f"Gradient-compression algorithm evidence boundary mismatch at {path}"
                )
            if payload.get("process_collective_mode") != expected.process_collective_mode:
                raise RuntimeError(
                    f"Gradient-compression process collective mode mismatch at {path}"
                )
            receipt = DistributedRankWorkReceipt.from_dict(payload.get("work_receipt"))
            execution_device_type = payload.get("execution_device_type")
            if execution_device_type not in {"cpu", "cuda"}:
                raise RuntimeError(
                    f"Gradient-compression execution device evidence is invalid at {path}"
                )
            _validate_execution_device_for_backend(
                backend=receipt.backend,
                execution_device_type=execution_device_type,
                label=str(path),
            )
            execution_device_types.add(execution_device_type)
            if len(receipt.collective_launch_ns) != expected.expected_collectives_per_rank:
                raise RuntimeError(
                    "Gradient-compression measured collective count mismatch at "
                    f"{path}: {len(receipt.collective_launch_ns)} vs "
                    f"{expected.expected_collectives_per_rank}"
                )
            receipts.append(receipt)

            expected_names = {
                "input_file": f"rank-{rank}-input.pt",
                "output_file": f"rank-{rank}-output.pt",
            }
            if rank == 0:
                expected_names["reference_file"] = "rank-0-reference.pt"
            elif payload.get("reference_file") is not None:
                raise RuntimeError("Only rank zero may publish the shared reference")
            tensors: dict[str, torch.Tensor] = {}
            for key, filename in expected_names.items():
                if payload.get(key) != filename:
                    raise RuntimeError(
                        f"Gradient-compression {key} mismatch at {path}"
                    )
                tensor = _load_tensor(
                    result_dir / filename,
                    numel=expected.numel,
                    label=f"rank {rank} {key}",
                )
                tensors[key.removesuffix("_file")] = tensor
                raw_bytes += tensor.numel() * tensor.element_size()
            payload["tensors"] = tensors
            payloads[rank] = payload

        if raw_bytes != expected.raw_result_tensor_bytes:
            raise RuntimeError(
                "Gradient-compression raw result tensor footprint mismatch: "
                f"{raw_bytes} vs {expected.raw_result_tensor_bytes}"
            )
        validation = validate_distributed_work_receipts(
            distributed_topology(expected), receipts
        )
        validation.raise_for_failure()
        if len(execution_device_types) != 1:
            raise RuntimeError(
                "Gradient-compression execution device type differs across ranks: "
                f"{sorted(execution_device_types)}"
            )
        execution_device_type = next(iter(execution_device_types))

        ordered = [payloads[rank] for rank in range(expected.world_size)]
        reference = ordered[0]["tensors"]["reference"]
        rank_inputs = [item["tensors"]["input"] for item in ordered]
        rank_outputs = [item["tensors"]["output"] for item in ordered]
        assert_reference_from_rank_inputs(
            expected,
            rank_inputs,
            reference,
            execution_device_type=execution_device_type,
        )
        for rank, output in enumerate(rank_outputs):
            assert_close_full(
                output,
                reference,
                rtol=expected.output_rtol,
                atol=expected.output_atol,
                label=f"rank {rank} timed output",
            )
        for rank, output in enumerate(rank_outputs[1:], start=1):
            assert_close_full(
                output,
                rank_outputs[0],
                rtol=0.0,
                atol=0.0,
                label=f"rank {rank}/rank 0 parity",
            )

        signature = input_signature(expected)
        signature_errors = signature.validate(strict=True)
        if signature_errors:
            raise RuntimeError(
                "Invalid gradient-compression input signature: " + signature_errors[0]
            )
        self._subprocess_verify_inputs = {
            **{
                f"rank_{rank}_input": rank_input
                for rank, rank_input in enumerate(rank_inputs)
            },
            "reference": reference,
        }
        self._subprocess_verify_output = rank_outputs[0]
        self._subprocess_input_signature = signature
        self._subprocess_output_tolerance = (
            expected.output_rtol,
            expected.output_atol,
        )
        self._gradient_compression_result_bundle = {
            "contract": expected,
            "rank_outputs": rank_outputs,
            "work_receipts": receipts,
            "collective_algorithm_evidence": validation.collective_algorithm_evidence,
            "process_collective_mode": expected.process_collective_mode,
            "execution_device_type": execution_device_type,
            "raw_result_tensor_bytes": raw_bytes,
        }
        try:
            shutil.rmtree(result_dir)
        except OSError:
            context["retention"] = "retained-cleanup-failure"
        else:
            context["retention"] = "cleaned-after-success"

    def require_gradient_compression_child_result(self) -> None:
        if self._gradient_compression_result_bundle is None:
            context = self._gradient_compression_result_context
            retained = context.get("result_dir") if context else "unavailable"
            raise RuntimeError(
                "Gradient-compression verification requires a fresh full-rank result; "
                f"retained path: {retained}"
            )

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        if self._gradient_compression_result_bundle is not None:
            return dict(self._subprocess_verify_inputs)
        return super().get_verify_inputs()  # type: ignore[misc]

    def get_verify_output(self) -> torch.Tensor:
        if self._gradient_compression_result_bundle is not None:
            # The mapped full tensor is read-only verification state. Avoid a
            # second 1 GiB clone here; the isolated runner snapshots it once.
            return self._subprocess_verify_output.detach()
        return super().get_verify_output()  # type: ignore[misc]

    def get_input_signature(self) -> InputSignature:
        if self._gradient_compression_result_bundle is not None:
            return self._subprocess_input_signature
        return super().get_input_signature()  # type: ignore[misc]

    def get_output_tolerance(self) -> tuple[float, float]:
        if self._gradient_compression_result_bundle is not None:
            return self._subprocess_output_tolerance
        return super().get_output_tolerance()  # type: ignore[misc]


def child_result_requested() -> bool:
    """Return whether this launch has a freshness-checking parent callback."""

    return bool(os.environ.get(LAUNCH_WALL_NS_ENV))


def _save_tensor_atomic(result_dir: Path, filename: str, tensor: torch.Tensor) -> None:
    temporary = result_dir / f".{filename}-{os.getpid()}.tmp"
    destination = result_dir / filename
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    try:
        torch.save(cpu_tensor, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
        del cpu_tensor


def write_gradient_compression_child_result(
    *,
    contract: GradientCompressionResultContract,
    rank: int,
    initial_input: torch.Tensor,
    reference_output: torch.Tensor,
    timed_output: torch.Tensor,
    work_receipt: DistributedRankWorkReceipt,
) -> bool:
    """Write full tensors one at a time, publishing metadata only after success."""

    result_dir_value = os.environ.get(RESULT_DIR_ENV)
    token = os.environ.get(RESULT_TOKEN_ENV)
    expected_variant = os.environ.get(RESULT_VARIANT_ENV)
    launch_wall_ns = os.environ.get(LAUNCH_WALL_NS_ENV)
    if not launch_wall_ns:
        return False
    if not all((result_dir_value, token, expected_variant)):
        raise RuntimeError("Gradient-compression child-result environment is incomplete")
    contract.validate()
    if expected_variant != contract.variant:
        raise RuntimeError("Gradient-compression child variant does not match launch")
    if rank < 0 or rank >= contract.world_size:
        raise RuntimeError(f"Invalid gradient-compression child rank: {rank}")

    for name, tensor in (
        ("initial input", initial_input),
        ("reference output", reference_output),
        ("timed output", timed_output),
    ):
        if tensor.dtype != torch.float32 or tensor.shape != (contract.numel,):
            raise RuntimeError(
                f"Gradient-compression {name} has unexpected shape/dtype"
            )
        _require_finite(name, tensor)
    assert_close_full(
        timed_output,
        reference_output,
        rtol=contract.output_rtol,
        atol=contract.output_atol,
        label=f"rank {rank} timed output/reference",
    )

    result_dir = Path(cast(str, result_dir_value))
    if result_dir.is_symlink() or not result_dir.is_dir():
        raise RuntimeError(
            "Gradient-compression child-result directory must be the prepared directory"
        )
    input_name = f"rank-{rank}-input.pt"
    output_name = f"rank-{rank}-output.pt"
    reference_name = "rank-0-reference.pt" if rank == 0 else None
    _save_tensor_atomic(result_dir, input_name, initial_input)
    _save_tensor_atomic(result_dir, output_name, timed_output)
    if reference_name is not None:
        _save_tensor_atomic(result_dir, reference_name, reference_output)

    metadata = {
        "schema": RESULT_SCHEMA,
        "token": token,
        "variant": contract.variant,
        "rank": int(rank),
        "contract": contract.to_dict(),
        "launch_wall_ns": int(launch_wall_ns),
        "created_wall_ns": time.time_ns(),
        "input_file": input_name,
        "output_file": output_name,
        "reference_file": reference_name,
        "work_receipt": work_receipt.to_dict(),
        "algorithm_evidence": DECLARED_ALGORITHM_EVIDENCE,
        "process_collective_mode": contract.process_collective_mode,
        "execution_device_type": initial_input.device.type,
    }
    temporary_metadata = result_dir / f".rank-{rank}-{os.getpid()}.json.tmp"
    destination = result_dir / f"rank-{rank}.json"
    temporary_metadata.write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_metadata, destination)
    return True


__all__ = [
    "COLLECTIVE_ALGORITHM",
    "GradientCompressionChildResultMixin",
    "GradientCompressionResultContract",
    "RESULT_CALLBACK",
    "assert_close_full",
    "assert_reference_from_rank_inputs",
    "child_result_requested",
    "distributed_topology",
    "input_signature",
    "make_result_contract",
    "write_gradient_compression_child_result",
]
