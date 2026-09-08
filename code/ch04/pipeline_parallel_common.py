"""Shared scheduling and child-result support for the Chapter 4 pipeline pair."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist

from core.benchmark.verification import InputSignature, PrecisionFlags
from labs.train_distributed.training_utils.child_result import (
    CONTRACT_ENV,
    ITERATIONS_ENV,
    LAUNCH_MONOTONIC_NS_ENV,
    LAUNCH_WALL_NS_ENV,
    RESULT_DIR_ENV,
    RUN_ID_ENV,
    WORLD_SIZE_ENV,
    TorchrunChildResultContract,
)

if TYPE_CHECKING:
    from core.harness.benchmark_harness import BenchmarkConfig, TorchrunLaunchSpec


PIPELINE_RESULT_CALLBACK = "consume_pipeline_child_results"
PIPELINE_RESULT_SCHEMA = "aisp.ch04.pipeline.child-result.v1"
PIPELINE_RESULT_RECEIPT_SCHEMA = "aisp.ch04.pipeline.child-result-receipt.v1"
PIPELINE_RESULT_RECEIPT_PREFIX = "AISP_PIPELINE_RESULT_RECEIPT:"
PIPELINE_RESULT_SOURCE_ENV = "AISP_PIPELINE_RESULT_SOURCE"
PIPELINE_RESULT_VARIANT_ENV = "AISP_PIPELINE_RESULT_VARIANT"
PIPELINE_RESULT_SCHEDULE_ENV = "AISP_PIPELINE_RESULT_SCHEDULE"
PIPELINE_RESULT_SHAPE_ENV = "AISP_PIPELINE_RESULT_SHAPE"
PIPELINE_RESULT_LAYERS_ENV = "AISP_PIPELINE_RESULT_LAYERS"
# Both schedules perform the same per-microbatch BF16 operations. Full-shape
# two-B200 repeats across four seeds produced bitwise-equal worker outputs.
PIPELINE_OUTPUT_TOLERANCE = (0.0, 0.0)

_MAX_RAW_RANK_BYTES = 1024 * 1024 * 1024
_SERIALIZATION_OVERHEAD_BYTES = 1024 * 1024
_FINITE_CHUNK_ELEMENTS = 16 * 1024 * 1024


TensorStep = Callable[[torch.Tensor], torch.Tensor]
BackwardStep = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MicrobatchGetter = Callable[[int], torch.Tensor]


@dataclass
class PipelineIterationCapture:
    """References to one measured iteration, indexed by original microbatch."""

    forward_inputs: list[torch.Tensor | None]
    backward_inputs: list[torch.Tensor | None]
    backward_outputs: list[torch.Tensor | None]

    @classmethod
    def create(cls, num_micro_batches: int) -> PipelineIterationCapture:
        if num_micro_batches <= 0:
            raise ValueError("num_micro_batches must be positive")
        return cls(
            forward_inputs=[None] * num_micro_batches,
            backward_inputs=[None] * num_micro_batches,
            backward_outputs=[None] * num_micro_batches,
        )

    def concatenate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concatenate only after the measured interval has closed."""

        ordered: list[list[torch.Tensor]] = []
        for label, values in (
            ("forward_inputs", self.forward_inputs),
            ("backward_inputs", self.backward_inputs),
            ("backward_outputs", self.backward_outputs),
        ):
            if any(value is None for value in values):
                raise RuntimeError(f"Pipeline capture is incomplete: {label}")
            ordered.append([cast(torch.Tensor, value) for value in values])
        return tuple(torch.cat(values, dim=0) for values in ordered)  # type: ignore[return-value]


def _validate_schedule_inputs(
    *,
    rank: int,
    world_size: int,
    num_micro_batches: int,
    recv_forward_buffers: Sequence[torch.Tensor],
    recv_backward_buffers: Sequence[torch.Tensor],
) -> None:
    if world_size < 2 or rank < 0 or rank >= world_size:
        raise ValueError("Pipeline schedule requires a valid multi-rank topology")
    if num_micro_batches < world_size:
        raise ValueError("Pipeline schedule requires at least one microbatch per stage")
    if rank > 0 and len(recv_forward_buffers) != num_micro_batches:
        raise ValueError("Every non-first stage needs one forward receive buffer per microbatch")
    if rank < world_size - 1 and len(recv_backward_buffers) != num_micro_batches:
        raise ValueError("Every non-last stage needs one backward receive buffer per microbatch")


def _record(
    capture: PipelineIterationCapture | None,
    collection: str,
    microbatch_index: int,
    tensor: torch.Tensor,
) -> None:
    if capture is not None:
        getattr(capture, collection)[microbatch_index] = tensor


def _exchange_neighbor(
    *,
    rank: int,
    peer: int,
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
) -> None:
    """Pair opposite-direction transfers and wait before either buffer is reused."""

    if rank < peer:
        operations = [
            dist.P2POp(dist.isend, send_tensor, peer),
            dist.P2POp(dist.irecv, recv_tensor, peer),
        ]
    else:
        operations = [
            dist.P2POp(dist.irecv, recv_tensor, peer),
            dist.P2POp(dist.isend, send_tensor, peer),
        ]
    requests = dist.batch_isend_irecv(operations)
    for request in requests:
        request.wait()


def run_gpipe_iteration(
    *,
    rank: int,
    world_size: int,
    num_micro_batches: int,
    get_rank0_microbatch: MicrobatchGetter,
    recv_forward_buffers: Sequence[torch.Tensor],
    recv_backward_buffers: Sequence[torch.Tensor],
    forward_step: TensorStep,
    backward_step: BackwardStep,
    capture: PipelineIterationCapture | None = None,
) -> None:
    """Run the existing all-forward/all-backward reference schedule."""

    _validate_schedule_inputs(
        rank=rank,
        world_size=world_size,
        num_micro_batches=num_micro_batches,
        recv_forward_buffers=recv_forward_buffers,
        recv_backward_buffers=recv_backward_buffers,
    )
    activations: list[tuple[int, torch.Tensor]] = []
    for microbatch_index in range(num_micro_batches):
        if rank == 0:
            microbatch = get_rank0_microbatch(microbatch_index)
        else:
            microbatch = recv_forward_buffers[microbatch_index]
            dist.recv(microbatch, src=rank - 1)
        _record(capture, "forward_inputs", microbatch_index, microbatch)
        output = forward_step(microbatch)
        activations.append((microbatch_index, output))
        if rank < world_size - 1:
            dist.send(output, dst=rank + 1)

    while activations:
        microbatch_index, activation = activations.pop()
        if rank < world_size - 1:
            backward_input = recv_backward_buffers[microbatch_index]
            dist.recv(backward_input, src=rank + 1)
        else:
            backward_input = activation
        _record(capture, "backward_inputs", microbatch_index, backward_input)
        output = backward_step(activation, backward_input)
        _record(capture, "backward_outputs", microbatch_index, output)
        if rank > 0:
            dist.send(output, dst=rank - 1)


def run_1f1b_iteration(
    *,
    rank: int,
    world_size: int,
    num_micro_batches: int,
    get_rank0_microbatch: MicrobatchGetter,
    recv_forward_buffers: Sequence[torch.Tensor],
    recv_backward_buffers: Sequence[torch.Tensor],
    forward_step: TensorStep,
    backward_step: BackwardStep,
    activation_slots: list[tuple[int, torch.Tensor] | None],
    capture: PipelineIterationCapture | None = None,
) -> None:
    """Run rank-staggered 1F1B with paired bidirectional P2P transitions."""

    _validate_schedule_inputs(
        rank=rank,
        world_size=world_size,
        num_micro_batches=num_micro_batches,
        recv_forward_buffers=recv_forward_buffers,
        recv_backward_buffers=recv_backward_buffers,
    )
    warmup_steps = min(world_size - rank - 1, num_micro_batches)
    required_slots = max(warmup_steps, 1)
    if len(activation_slots) != required_slots:
        raise ValueError(
            f"Pipeline activation slot count mismatch: expected {required_slots}, "
            f"got {len(activation_slots)}"
        )
    for index in range(len(activation_slots)):
        activation_slots[index] = None
    activation_head = 0
    activation_count = 0

    def push(microbatch_index: int, tensor: torch.Tensor) -> None:
        nonlocal activation_count
        if activation_count >= warmup_steps:
            raise RuntimeError("Pipeline activation ring overflow")
        slot = (activation_head + activation_count) % required_slots
        activation_slots[slot] = (microbatch_index, tensor)
        activation_count += 1

    def pop() -> tuple[int, torch.Tensor]:
        nonlocal activation_head, activation_count
        if activation_count <= 0:
            raise RuntimeError("Pipeline activation ring underflow")
        entry = activation_slots[activation_head]
        if entry is None:
            raise RuntimeError("Pipeline activation slot is empty")
        activation_slots[activation_head] = None
        activation_head = (activation_head + 1) % required_slots
        activation_count -= 1
        return entry

    def receive_forward(microbatch_index: int) -> torch.Tensor:
        if rank == 0:
            return get_rank0_microbatch(microbatch_index)
        tensor = recv_forward_buffers[microbatch_index]
        dist.recv(tensor, src=rank - 1)
        return tensor

    for microbatch_index in range(warmup_steps):
        microbatch = receive_forward(microbatch_index)
        _record(capture, "forward_inputs", microbatch_index, microbatch)
        output = forward_step(microbatch)
        push(microbatch_index, output)
        dist.send(output, dst=rank + 1)

    remaining_steps = num_micro_batches - warmup_steps
    current_input = receive_forward(warmup_steps) if remaining_steps else None
    for steady_index in range(remaining_steps):
        current_microbatch = warmup_steps + steady_index
        if current_input is None:
            raise RuntimeError("Pipeline steady state is missing its forward input")
        _record(capture, "forward_inputs", current_microbatch, current_input)
        current_output = forward_step(current_input)

        if rank == world_size - 1:
            backward_microbatch = current_microbatch
            activation = current_output
            backward_input = current_output
        else:
            backward_microbatch, activation = pop()
            backward_input = recv_backward_buffers[backward_microbatch]
            _exchange_neighbor(
                rank=rank,
                peer=rank + 1,
                send_tensor=current_output,
                recv_tensor=backward_input,
            )
        _record(capture, "backward_inputs", backward_microbatch, backward_input)
        backward_output = backward_step(activation, backward_input)
        _record(capture, "backward_outputs", backward_microbatch, backward_output)
        if rank < world_size - 1:
            push(current_microbatch, current_output)

        has_next_forward = steady_index + 1 < remaining_steps
        if rank == 0:
            current_input = (
                get_rank0_microbatch(current_microbatch + 1)
                if has_next_forward
                else None
            )
        elif has_next_forward:
            current_input = recv_forward_buffers[current_microbatch + 1]
            _exchange_neighbor(
                rank=rank,
                peer=rank - 1,
                send_tensor=backward_output,
                recv_tensor=current_input,
            )
        else:
            dist.send(backward_output, dst=rank - 1)
            current_input = None

    while activation_count:
        microbatch_index, activation = pop()
        if rank >= world_size - 1:
            raise RuntimeError("Last pipeline stage retained an unexpected activation")
        backward_input = recv_backward_buffers[microbatch_index]
        dist.recv(backward_input, src=rank + 1)
        _record(capture, "backward_inputs", microbatch_index, backward_input)
        backward_output = backward_step(activation, backward_input)
        _record(capture, "backward_outputs", microbatch_index, backward_output)
        if rank > 0:
            dist.send(backward_output, dst=rank - 1)


def verify_and_concatenate_pipeline_capture(
    *,
    rank: int,
    capture: PipelineIterationCapture,
    forward_stages: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    backward_stages: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    tolerance: tuple[float, float] = PIPELINE_OUTPUT_TOLERANCE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check an independent post-timing reference, then form full tensors."""

    if not forward_stages or len(forward_stages) != len(backward_stages):
        raise ValueError("Pipeline reference requires equal non-empty stage lists")
    if rank < 0 or rank >= len(forward_stages):
        raise ValueError("Pipeline reference rank is outside the stage list")
    for microbatch_index, actual_value in enumerate(capture.backward_outputs):
        actual = cast(torch.Tensor, actual_value)
        if rank == 0:
            reference = cast(torch.Tensor, capture.forward_inputs[microbatch_index])
            for stage in forward_stages:
                reference = stage(reference)
            for stage in reversed(backward_stages):
                reference = stage(reference)
        else:
            reference = backward_stages[rank](
                cast(torch.Tensor, capture.backward_inputs[microbatch_index])
            )
        torch.testing.assert_close(
            actual,
            reference,
            rtol=tolerance[0],
            atol=tolerance[1],
        )
    full_input, _, full_output = capture.concatenate()
    return full_input, full_output


def make_pipeline_child_result_contract(
    *,
    batch_size: int,
    parameter_count: int,
) -> TorchrunChildResultContract:
    contract = TorchrunChildResultContract(
        profile="ch04.pipeline-parallel-toy-v1",
        input_names=("pipeline_input",),
        output_names=("pipeline_output",),
        per_rank_batch_size=batch_size,
        parameter_count=parameter_count,
        precision_flags=PrecisionFlags(bf16=True, tf32=False),
        output_tolerance=PIPELINE_OUTPUT_TOLERANCE,
        independent_reference="post-timing sequential pipeline replay",
        collective_type="send_recv",
        collective_algorithm="ordered-point-to-point-pipeline",
        max_rank_payload_bytes=_MAX_RAW_RANK_BYTES,
    )
    contract.validate()
    return contract


def pipeline_child_result_requested() -> bool:
    """Return whether the harness requested actual timed-worker tensors."""

    return bool(os.environ.get(RESULT_DIR_ENV))


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _require_finite_tensor(
    tensor: Any,
    *,
    label: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"Pipeline {label} is not a tensor")
    if tensor.shape != shape or tensor.dtype != torch.bfloat16:
        raise RuntimeError(
            f"Pipeline {label} shape/dtype mismatch: expected {shape}/torch.bfloat16, "
            f"got {tuple(tensor.shape)}/{tensor.dtype}"
        )
    flat = tensor.reshape(-1)
    for start in range(0, flat.numel(), _FINITE_CHUNK_ELEMENTS):
        if not bool(torch.isfinite(flat[start : start + _FINITE_CHUNK_ELEMENTS]).all()):
            raise RuntimeError(f"Pipeline {label} contains non-finite values")
    return tensor


def _atomic_save_tensor(tensor: torch.Tensor, destination: Path) -> int:
    raw_bytes = _tensor_nbytes(tensor)
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        torch.save(cpu_tensor, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return raw_bytes


def _atomic_write_json(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def write_pipeline_child_result(
    *,
    verify_input: torch.Tensor,
    verify_output: torch.Tensor,
    completed_iterations: int,
    time_per_iter_ms: float,
    reference_verified: bool,
) -> bool:
    """Publish each full actual tensor once; metadata is the atomic commit marker."""

    result_dir_value = os.environ.get(RESULT_DIR_ENV)
    if not result_dir_value:
        return False
    required_names = (
        RUN_ID_ENV,
        CONTRACT_ENV,
        WORLD_SIZE_ENV,
        ITERATIONS_ENV,
        LAUNCH_WALL_NS_ENV,
        LAUNCH_MONOTONIC_NS_ENV,
        PIPELINE_RESULT_SOURCE_ENV,
        PIPELINE_RESULT_VARIANT_ENV,
        PIPELINE_RESULT_SCHEDULE_ENV,
        PIPELINE_RESULT_SHAPE_ENV,
        PIPELINE_RESULT_LAYERS_ENV,
    )
    missing = [name for name in required_names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Pipeline child-result environment is incomplete: {missing}")
    contract = TorchrunChildResultContract.from_dict(
        json.loads(os.environ[CONTRACT_ENV])
    )
    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    expected_world_size = int(os.environ[WORLD_SIZE_ENV])
    requested_iterations = int(os.environ[ITERATIONS_ENV])
    shape = tuple(int(value) for value in json.loads(os.environ[PIPELINE_RESULT_SHAPE_ENV]))
    if rank < 0 or rank >= world_size or world_size != expected_world_size:
        raise RuntimeError("Pipeline child rank topology differs from its contract")
    if completed_iterations != requested_iterations or completed_iterations <= 0:
        raise RuntimeError("Pipeline child completed iteration count differs from its contract")
    if not math.isfinite(time_per_iter_ms) or time_per_iter_ms <= 0:
        raise RuntimeError("Pipeline child timing must be finite and positive")
    if reference_verified is not True:
        raise RuntimeError("Pipeline child result requires its independent post-timing reference")
    verified_input = _require_finite_tensor(verify_input, label="input", shape=shape)
    verified_output = _require_finite_tensor(verify_output, label="output", shape=shape)
    input_bytes = _tensor_nbytes(verified_input)
    output_bytes = _tensor_nbytes(verified_output)
    if input_bytes + output_bytes > contract.max_rank_payload_bytes:
        raise RuntimeError("Pipeline raw rank payload exceeds its declared byte limit")

    result_dir = Path(result_dir_value)
    if result_dir.is_symlink() or not result_dir.is_dir():
        raise RuntimeError("Pipeline result directory must be the prepared regular directory")
    input_path = result_dir / f"rank-{rank}.input.pt"
    output_path = result_dir / f"rank-{rank}.output.pt"
    metadata_path = result_dir / f"rank-{rank}.meta.json"
    if any(path.exists() for path in (input_path, output_path, metadata_path)):
        raise RuntimeError(f"Refusing to overwrite pipeline child result for rank {rank}")
    stored_input_bytes = _atomic_save_tensor(verified_input, input_path)
    stored_output_bytes = _atomic_save_tensor(verified_output, output_path)
    metadata = {
        "schema": PIPELINE_RESULT_SCHEMA,
        "run_id": os.environ[RUN_ID_ENV],
        "contract": contract.to_dict(),
        "source": os.environ[PIPELINE_RESULT_SOURCE_ENV],
        "variant": os.environ[PIPELINE_RESULT_VARIANT_ENV],
        "schedule": os.environ[PIPELINE_RESULT_SCHEDULE_ENV],
        "rank": rank,
        "world_size": world_size,
        "pid": os.getpid(),
        "completed_iterations": completed_iterations,
        "time_per_iter_ms": time_per_iter_ms,
        "num_layers": int(os.environ[PIPELINE_RESULT_LAYERS_ENV]),
        "shape": list(shape),
        "dtype": str(torch.bfloat16),
        "input_bytes": stored_input_bytes,
        "output_bytes": stored_output_bytes,
        "reference_verified": True,
        "launch_wall_ns": int(os.environ[LAUNCH_WALL_NS_ENV]),
        "launch_monotonic_ns": int(os.environ[LAUNCH_MONOTONIC_NS_ENV]),
        "created_wall_ns": time.time_ns(),
        "created_monotonic_ns": time.monotonic_ns(),
    }
    _atomic_write_json(metadata, metadata_path)
    receipt = {
        "schema": PIPELINE_RESULT_RECEIPT_SCHEMA,
        "run_id": metadata["run_id"],
        "source": metadata["source"],
        "variant": metadata["variant"],
        "schedule": metadata["schedule"],
        "rank": rank,
        "world_size": world_size,
        "pid": metadata["pid"],
        "completed_iterations": completed_iterations,
        "time_per_iter_ms": time_per_iter_ms,
    }
    print(
        f"{PIPELINE_RESULT_RECEIPT_PREFIX}"
        f"{json.dumps(receipt, sort_keys=True, separators=(',', ':'))}",
        flush=True,
    )
    return True


def _load_tensor_file(path: Path, *, raw_bytes: int) -> torch.Tensor:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Pipeline tensor artifact must be a regular file: {path}")
    if path.stat().st_size > raw_bytes + _SERIALIZATION_OVERHEAD_BYTES:
        raise RuntimeError(f"Pipeline tensor artifact exceeds its expected byte bound: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, torch.Tensor):
        raise RuntimeError(f"Pipeline tensor artifact did not contain one tensor: {path}")
    return loaded


def _parse_pipeline_result_receipts(
    stdout: str,
    *,
    run_id: str,
    source: str,
    variant: str,
    schedule: str,
    world_size: int,
    requested_iterations: int,
) -> dict[int, dict[str, Any]]:
    if not isinstance(stdout, str):
        raise TypeError("Pipeline child stdout must be text")
    expected_fields = {
        "schema",
        "run_id",
        "source",
        "variant",
        "schedule",
        "rank",
        "world_size",
        "pid",
        "completed_iterations",
        "time_per_iter_ms",
    }
    receipts: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if PIPELINE_RESULT_RECEIPT_PREFIX not in line:
            continue
        if not line.startswith(PIPELINE_RESULT_RECEIPT_PREFIX):
            raise RuntimeError(
                "Pipeline child-result receipt prefix is misplaced on stdout "
                f"line {line_number}"
            )
        try:
            receipt = json.loads(line[len(PIPELINE_RESULT_RECEIPT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Pipeline child-result receipt is malformed on stdout line {line_number}"
            ) from exc
        if not isinstance(receipt, dict) or set(receipt) != expected_fields:
            raise RuntimeError(
                f"Pipeline child-result receipt fields are invalid on stdout line {line_number}"
            )
        rank = receipt["rank"]
        pid = receipt["pid"]
        observed_time = receipt["time_per_iter_ms"]
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank >= world_size
            or rank in receipts
        ):
            raise RuntimeError("Pipeline child-result receipt rank is invalid or duplicated")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise RuntimeError(f"Pipeline child-result receipt PID is invalid for rank {rank}")
        if (
            isinstance(observed_time, bool)
            or not isinstance(observed_time, int | float)
            or not math.isfinite(observed_time)
            or observed_time <= 0
        ):
            raise RuntimeError(f"Pipeline child-result receipt timing is invalid for rank {rank}")
        if (
            receipt["schema"] != PIPELINE_RESULT_RECEIPT_SCHEMA
            or receipt["run_id"] != run_id
            or receipt["source"] != source
            or receipt["variant"] != variant
            or receipt["schedule"] != schedule
            or receipt["world_size"] != world_size
            or receipt["completed_iterations"] != requested_iterations
        ):
            raise RuntimeError(f"Pipeline child-result receipt identity mismatch for rank {rank}")
        receipts[rank] = receipt
    if set(receipts) != set(range(world_size)):
        raise RuntimeError("Pipeline child-result stdout receipt rank quorum is incomplete")

    rank_zero_timings: list[float] = []
    timing_prefix = "rank0 time_per_iter_ms:"
    for line in stdout.splitlines():
        if not line.strip().startswith(timing_prefix):
            continue
        try:
            rank_zero_timings.append(float(line.strip()[len(timing_prefix) :].strip()))
        except ValueError as exc:
            raise RuntimeError("Pipeline rank-zero stdout timing is malformed") from exc
    if len(rank_zero_timings) != 1:
        raise RuntimeError("Pipeline child stdout requires exactly one rank-zero timing")
    if not math.isclose(
        rank_zero_timings[0],
        float(receipts[0]["time_per_iter_ms"]),
        rel_tol=1e-12,
        abs_tol=5.1e-10,
    ):
        raise RuntimeError("Pipeline rank-zero stdout timing differs from its worker receipt")
    return receipts


def validate_pipeline_child_result_bundle(
    result_dir: Path,
    *,
    contract: TorchrunChildResultContract,
    run_id: str,
    source: str,
    variant: str,
    schedule: str,
    world_size: int,
    requested_iterations: int,
    shape: tuple[int, ...],
    num_layers: int,
    launch_wall_ns: int,
    launch_monotonic_ns: int,
    finish_wall_ns: int,
    finish_monotonic_ns: int,
    stdout: str,
) -> dict[str, Any]:
    """Fail closed unless a fresh, exact rank quorum carries the full tensors."""

    contract.validate()
    receipts_by_rank = _parse_pipeline_result_receipts(
        stdout,
        run_id=run_id,
        source=source,
        variant=variant,
        schedule=schedule,
        world_size=world_size,
        requested_iterations=requested_iterations,
    )
    expected_names = {
        f"rank-{rank}.{suffix}"
        for rank in range(world_size)
        for suffix in ("input.pt", "output.pt", "meta.json")
    }
    entries = {path.name for path in result_dir.iterdir()}
    if entries != expected_names:
        raise RuntimeError(
            "Pipeline child-result rank quorum is incomplete or contains unexpected artifacts"
        )
    expected_raw_bytes = math.prod(shape) * torch.empty((), dtype=torch.bfloat16).element_size()
    if expected_raw_bytes * 2 > contract.max_rank_payload_bytes:
        raise RuntimeError("Pipeline expected raw rank payload exceeds its contract")
    verify_inputs: dict[str, torch.Tensor] = {}
    verify_outputs: dict[str, torch.Tensor] = {}
    metadata_by_rank: dict[int, dict[str, Any]] = {}
    seen_pids: set[int] = set()
    for rank in range(world_size):
        metadata_path = result_dir / f"rank-{rank}.meta.json"
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise RuntimeError(f"Pipeline metadata must be a regular file: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_metadata_fields = {
            "schema", "run_id", "contract", "source", "variant", "schedule",
            "rank", "world_size", "pid", "completed_iterations", "time_per_iter_ms",
            "num_layers", "shape", "dtype", "input_bytes", "output_bytes",
            "reference_verified", "launch_wall_ns", "launch_monotonic_ns",
            "created_wall_ns", "created_monotonic_ns",
        }
        if not isinstance(metadata, dict) or set(metadata) != expected_metadata_fields:
            raise RuntimeError(f"Pipeline metadata fields are invalid for rank {rank}")
        if metadata["schema"] != PIPELINE_RESULT_SCHEMA:
            raise RuntimeError(f"Pipeline result schema mismatch for rank {rank}")
        if metadata["run_id"] != run_id:
            raise RuntimeError(f"Pipeline result run mismatch for rank {rank}")
        if metadata["source"] != source or metadata["variant"] != variant:
            raise RuntimeError(f"Pipeline result source/variant mismatch for rank {rank}")
        if metadata["schedule"] != schedule:
            raise RuntimeError(f"Pipeline result schedule mismatch for rank {rank}")
        if TorchrunChildResultContract.from_dict(metadata["contract"]) != contract:
            raise RuntimeError(f"Pipeline result contract mismatch for rank {rank}")
        if metadata["rank"] != rank or metadata["world_size"] != world_size:
            raise RuntimeError(f"Pipeline result rank topology mismatch for rank {rank}")
        pid = metadata["pid"]
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or pid in seen_pids:
            raise RuntimeError(f"Pipeline result PID is invalid or duplicated for rank {rank}")
        seen_pids.add(pid)
        if metadata["completed_iterations"] != requested_iterations:
            raise RuntimeError(f"Pipeline result iteration count mismatch for rank {rank}")
        observed_time = metadata["time_per_iter_ms"]
        if not isinstance(observed_time, int | float) or not math.isfinite(observed_time) or observed_time <= 0:
            raise RuntimeError(f"Pipeline result timing is invalid for rank {rank}")
        receipt = receipts_by_rank[rank]
        if (
            receipt["pid"] != pid
            or receipt["completed_iterations"] != metadata["completed_iterations"]
            or receipt["time_per_iter_ms"] != observed_time
        ):
            raise RuntimeError(
                f"Pipeline result metadata differs from its worker stdout receipt for rank {rank}"
            )
        if metadata["num_layers"] != num_layers:
            raise RuntimeError(f"Pipeline result layer count mismatch for rank {rank}")
        if tuple(metadata["shape"]) != shape or metadata["dtype"] != str(torch.bfloat16):
            raise RuntimeError(f"Pipeline result tensor metadata mismatch for rank {rank}")
        if metadata["input_bytes"] != expected_raw_bytes or metadata["output_bytes"] != expected_raw_bytes:
            raise RuntimeError(f"Pipeline result byte count mismatch for rank {rank}")
        if metadata["reference_verified"] is not True:
            raise RuntimeError(f"Pipeline independent reference is missing for rank {rank}")
        if (
            metadata["launch_wall_ns"] != launch_wall_ns
            or metadata["launch_monotonic_ns"] != launch_monotonic_ns
        ):
            raise RuntimeError(f"Pipeline result launch identity mismatch for rank {rank}")
        if not launch_wall_ns <= metadata["created_wall_ns"] <= finish_wall_ns:
            raise RuntimeError(f"Pipeline result wall-clock freshness failed for rank {rank}")
        if not launch_monotonic_ns <= metadata["created_monotonic_ns"] <= finish_monotonic_ns:
            raise RuntimeError(f"Pipeline result monotonic freshness failed for rank {rank}")

        input_tensor = _require_finite_tensor(
            _load_tensor_file(result_dir / f"rank-{rank}.input.pt", raw_bytes=expected_raw_bytes),
            label=f"rank {rank} input",
            shape=shape,
        )
        output_tensor = _require_finite_tensor(
            _load_tensor_file(result_dir / f"rank-{rank}.output.pt", raw_bytes=expected_raw_bytes),
            label=f"rank {rank} output",
            shape=shape,
        )
        verify_inputs[f"rank-{rank}:pipeline_input"] = input_tensor
        verify_outputs[f"rank-{rank}:pipeline_output"] = output_tensor
        metadata_by_rank[rank] = metadata

    verify_outputs["completed_iterations"] = torch.ones(
        requested_iterations, dtype=torch.float32
    )
    signature = InputSignature(
        shapes={name: tuple(tensor.shape) for name, tensor in verify_inputs.items()},
        dtypes={name: str(tensor.dtype) for name, tensor in verify_inputs.items()},
        batch_size=contract.per_rank_batch_size,
        parameter_count=cast(int, contract.parameter_count),
        precision_flags=PrecisionFlags.from_dict(contract.precision_flags.to_dict()),
        world_size=world_size,
        ranks=list(range(world_size)),
        pipeline_stages=world_size,
        pipeline_stage_boundaries=[
            (stage * (num_layers // world_size), (stage + 1) * (num_layers // world_size) - 1)
            for stage in range(world_size)
        ],
        per_rank_batch_size=contract.per_rank_batch_size,
        collective_type=contract.collective_type,
        collective_algorithm=contract.collective_algorithm,
        async_completion_policy="wait_for_async_before_timed_close",
    )
    signature_errors = signature.validate(strict=True)
    if signature_errors:
        raise RuntimeError(f"Pipeline child-result signature is invalid: {signature_errors[0]}")
    return {
        "contract": contract,
        "verify_inputs": verify_inputs,
        "verify_output": verify_outputs,
        "input_signature": signature,
        "output_tolerance": contract.output_tolerance,
        "metadata_by_rank": metadata_by_rank,
        "completed_iterations": requested_iterations,
    }


class PipelineParallelChildResultMixin:
    """Prepare and consume an exact child-produced pipeline result bundle."""

    _pipeline_result_context: dict[str, Any] | None = None
    _pipeline_result_bundle: dict[str, Any] | None = None

    def get_profile_torchrun_spec(
        self,
        *,
        profiler: str,
        config: BenchmarkConfig | None = None,
        output_path: Path | None = None,
    ) -> TorchrunLaunchSpec | None:
        if profiler not in {"ncu", "nsys", "torch"}:
            return None
        spec = self.get_torchrun_spec(config)
        context = self._pipeline_result_context
        if context is None:
            raise RuntimeError("Pipeline profiling spec did not prepare a result context")
        result_dir = cast(Path, context["result_dir"])
        if any(result_dir.iterdir()):
            raise RuntimeError(
                "Refusing to discard populated pipeline child artifacts for profiling"
            )
        result_dir.rmdir()
        self._pipeline_result_context = None
        self._pipeline_result_bundle = None
        transport_names = {
            RESULT_DIR_ENV,
            RUN_ID_ENV,
            CONTRACT_ENV,
            WORLD_SIZE_ENV,
            ITERATIONS_ENV,
            PIPELINE_RESULT_SOURCE_ENV,
            PIPELINE_RESULT_VARIANT_ENV,
            PIPELINE_RESULT_SCHEDULE_ENV,
            PIPELINE_RESULT_SHAPE_ENV,
            PIPELINE_RESULT_LAYERS_ENV,
        }
        profile_env = {
            name: value
            for name, value in (spec.env or {}).items()
            if name not in transport_names
        }
        if profiler == "torch":
            if output_path is None:
                raise ValueError("Torch profiling requires an output path")
            profile_env["AISP_TORCH_PROFILE_OUTPUT"] = str(output_path)
        return replace(spec, env=profile_env, result_callback=None)

    def prepare_pipeline_child_result(
        self,
        *,
        source: str,
        variant: str,
        schedule: str,
        world_size: int,
        iterations: int,
        batch_size: int,
        seq_length: int,
        hidden: int,
        num_layers: int,
    ) -> dict[str, str]:
        if not source or variant not in {"baseline", "optimized"} or not schedule:
            raise ValueError("Pipeline child-result source, variant and schedule must be explicit")
        if world_size < 2 or num_layers <= 0 or num_layers % world_size:
            raise ValueError("Pipeline child-result topology/layer partition is invalid")
        if iterations <= 0 or min(batch_size, seq_length, hidden) <= 0:
            raise ValueError("Pipeline child-result workload dimensions must be positive")
        previous = self._pipeline_result_context
        if previous is not None and Path(previous["result_dir"]).exists():
            raise RuntimeError("Refusing to replace an unconsumed pipeline child-result context")
        parameter_count = 2 * num_layers * hidden * hidden
        contract = make_pipeline_child_result_contract(
            batch_size=batch_size,
            parameter_count=parameter_count,
        )
        result_dir = Path(tempfile.mkdtemp(prefix="aisp-pipeline-child-result-"))
        run_id = uuid.uuid4().hex
        shape = (batch_size, seq_length, hidden)
        self._pipeline_result_context = {
            "result_dir": result_dir,
            "run_id": run_id,
            "contract": contract,
            "source": source,
            "variant": variant,
            "schedule": schedule,
            "world_size": world_size,
            "requested_iterations": iterations,
            "shape": shape,
            "num_layers": num_layers,
            "retention": "pending-child-result",
        }
        self._pipeline_result_bundle = None
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
            RUN_ID_ENV: run_id,
            CONTRACT_ENV: json.dumps(contract.to_dict(), sort_keys=True, separators=(",", ":")),
            WORLD_SIZE_ENV: str(world_size),
            ITERATIONS_ENV: str(iterations),
            PIPELINE_RESULT_SOURCE_ENV: source,
            PIPELINE_RESULT_VARIANT_ENV: variant,
            PIPELINE_RESULT_SCHEDULE_ENV: schedule,
            PIPELINE_RESULT_SHAPE_ENV: json.dumps(shape, separators=(",", ":")),
            PIPELINE_RESULT_LAYERS_ENV: str(num_layers),
        }

    def consume_pipeline_child_results(
        self,
        *,
        spec: TorchrunLaunchSpec,
        launch_wall_ns: int,
        launch_monotonic_ns: int,
        finish_wall_ns: int,
        finish_monotonic_ns: int,
        returncode: int,
        stdout: str,
        **_: Any,
    ) -> None:
        context = self._pipeline_result_context
        if context is None:
            raise RuntimeError("Pipeline child-result callback has no prepared launch context")
        result_dir = cast(Path, context["result_dir"])
        if returncode != 0:
            context["retention"] = "retained-child-failure"
            raise RuntimeError(
                f"Pipeline child exited unsuccessfully; artifacts retained at {result_dir}"
            )
        try:
            bundle = validate_pipeline_child_result_bundle(
                result_dir,
                contract=context["contract"],
                run_id=context["run_id"],
                source=context["source"],
                variant=context["variant"],
                schedule=context["schedule"],
                world_size=context["world_size"],
                requested_iterations=context["requested_iterations"],
                shape=context["shape"],
                num_layers=context["num_layers"],
                launch_wall_ns=int(launch_wall_ns),
                launch_monotonic_ns=int(launch_monotonic_ns),
                finish_wall_ns=int(finish_wall_ns),
                finish_monotonic_ns=int(finish_monotonic_ns),
                stdout=stdout,
            )
        except Exception as exc:
            context["retention"] = "retained-validation-failure"
            raise RuntimeError(
                f"{exc}; failed pipeline child artifacts retained at {result_dir}"
            ) from exc
        self._pipeline_result_bundle = bundle
        self._subprocess_verify_inputs = bundle["verify_inputs"]
        self._subprocess_verify_output = bundle["verify_output"]
        self._subprocess_input_signature = bundle["input_signature"]
        self._subprocess_output_tolerance = bundle["output_tolerance"]
        spec.timing_iterations_per_sample = bundle["completed_iterations"]
        try:
            shutil.rmtree(result_dir)
        except OSError:
            context["retention"] = "retained-cleanup-failure"
        else:
            context["retention"] = "cleaned-after-success"

    def require_pipeline_child_result(self) -> None:
        if self._pipeline_result_bundle is None:
            context = self._pipeline_result_context
            retained = context["result_dir"] if context else "unavailable"
            raise RuntimeError(
                "Pipeline verification requires a fresh full-rank timed-worker result; "
                f"retained path: {retained}"
            )

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        self.require_pipeline_child_result()
        return dict(self._subprocess_verify_inputs)

    def get_verify_output(self) -> dict[str, torch.Tensor]:
        self.require_pipeline_child_result()
        return dict(self._subprocess_verify_output)

    def get_input_signature(self) -> InputSignature:
        self.require_pipeline_child_result()
        return self._subprocess_input_signature

    def get_output_tolerance(self) -> tuple[float, float]:
        self.require_pipeline_child_result()
        return self._subprocess_output_tolerance

    def validate_result(self) -> str | None:
        if self._pipeline_result_bundle is None:
            return "Fresh full-rank timed pipeline worker output is missing"
        return None

    def retain_failed_pipeline_child_result(self) -> None:
        context = self._pipeline_result_context
        if (
            self._pipeline_result_bundle is None
            and context is not None
            and Path(context["result_dir"]).exists()
        ):
            print(
                f"[pipeline-child-result] retained unsuccessful artifacts at "
                f"{context['result_dir']}",
                flush=True,
            )


__all__ = [
    "PIPELINE_OUTPUT_TOLERANCE",
    "PIPELINE_RESULT_CALLBACK",
    "PipelineIterationCapture",
    "PipelineParallelChildResultMixin",
    "make_pipeline_child_result_contract",
    "pipeline_child_result_requested",
    "run_1f1b_iteration",
    "run_gpipe_iteration",
    "validate_pipeline_child_result_bundle",
    "verify_and_concatenate_pipeline_capture",
    "write_pipeline_child_result",
]
