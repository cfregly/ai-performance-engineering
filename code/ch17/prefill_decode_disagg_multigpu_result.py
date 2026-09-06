"""Fresh per-rank result transport for Chapter 17 prefill/decode workers."""

from __future__ import annotations

import os
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

PREFILL_DECODE_RESULT_CALLBACK = "consume_prefill_decode_child_results"
PREFILL_DECODE_RESULT_DIR_ENV = "AISP_PREFILL_DECODE_RESULT_DIR"
PREFILL_DECODE_RESULT_TOKEN_ENV = "AISP_PREFILL_DECODE_RESULT_TOKEN"
PREFILL_DECODE_RESULT_LABEL_ENV = "AISP_PREFILL_DECODE_RESULT_LABEL"
PREFILL_DECODE_LAUNCH_WALL_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS"
PREFILL_DECODE_RESULT_SCHEMA = "aisp.prefill-decode-disagg.child-result.v1"


def _dtype_from_name(name: str) -> torch.dtype:
    supported = {
        str(torch.float16): torch.float16,
        str(torch.bfloat16): torch.bfloat16,
        str(torch.float32): torch.float32,
    }
    try:
        return supported[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported prefill/decode result dtype: {name!r}") from exc


@dataclass(frozen=True)
class PrefillDecodeResultContract:
    """Serializable identity for one distributed prefill/decode launch."""

    label: str
    handoff_mode: str
    world_size: int
    prefill_ranks: int
    hidden_size: int
    num_layers: int
    batch_size: int
    requests_per_rank: int
    context_window: int
    decode_tokens: int
    transfer_group: int
    sync_per_request: bool
    barrier_per_request: bool
    dtype: str
    iterations: int
    warmup: int

    @property
    def decode_ranks(self) -> int:
        return self.world_size - self.prefill_ranks

    def validate(self) -> None:
        if not self.label or not self.label.startswith(("baseline_", "optimized_")):
            raise ValueError("Prefill/decode label must identify a baseline or optimized variant")
        if self.handoff_mode not in {"serial", "overlap", "batched"}:
            raise ValueError(f"Unsupported prefill/decode handoff mode: {self.handoff_mode!r}")
        if self.world_size < 2:
            raise ValueError("Prefill/decode result contract requires world_size >= 2")
        if not 0 < self.prefill_ranks < self.world_size:
            raise ValueError("Prefill/decode prefill_ranks must partition the world")
        for name in (
            "hidden_size",
            "num_layers",
            "batch_size",
            "requests_per_rank",
            "context_window",
            "decode_tokens",
            "transfer_group",
            "iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Prefill/decode {name} must be a positive integer")
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int) or self.warmup < 0:
            raise ValueError("Prefill/decode warmup must be a non-negative integer")
        if (
            not isinstance(self.sync_per_request, bool)
            or not isinstance(self.barrier_per_request, bool)
        ):
            raise TypeError("Prefill/decode boolean contract fields must be booleans")
        _dtype_from_name(self.dtype)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PrefillDecodeResultContract:
        if not isinstance(payload, dict):
            raise TypeError("Prefill/decode result contract must be a dictionary")
        try:
            contract = cls(**payload)
        except TypeError as exc:
            raise ValueError("Invalid prefill/decode result contract fields") from exc
        contract.validate()
        return contract


def make_prefill_decode_result_contract(
    *,
    label: str,
    handoff_mode: str,
    world_size: int,
    prefill_ranks: int,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    requests_per_rank: int,
    context_window: int,
    decode_tokens: int,
    transfer_group: int,
    sync_per_request: bool,
    barrier_per_request: bool,
    dtype: torch.dtype,
    iterations: int,
    warmup: int,
) -> PrefillDecodeResultContract:
    contract = PrefillDecodeResultContract(
        label=label,
        handoff_mode=handoff_mode,
        world_size=int(world_size),
        prefill_ranks=int(prefill_ranks),
        hidden_size=int(hidden_size),
        num_layers=int(num_layers),
        batch_size=int(batch_size),
        requests_per_rank=int(requests_per_rank),
        context_window=int(context_window),
        decode_tokens=int(decode_tokens),
        transfer_group=int(transfer_group),
        sync_per_request=sync_per_request,
        barrier_per_request=barrier_per_request,
        dtype=str(dtype),
        iterations=int(iterations),
        warmup=int(warmup),
    )
    contract.validate()
    return contract


def _assigned_prefills(contract: PrefillDecodeResultContract, decode_rank: int) -> list[int]:
    decode_index = decode_rank - contract.prefill_ranks
    return [
        rank
        for rank in range(contract.prefill_ranks)
        if rank % contract.decode_ranks == decode_index
    ]


def _output_shape_for_rank(
    contract: PrefillDecodeResultContract,
    rank: int,
) -> tuple[int, int, int]:
    request_count = contract.requests_per_rank
    if rank >= contract.prefill_ranks:
        request_count *= len(_assigned_prefills(contract, rank))
    return (request_count, contract.batch_size, contract.hidden_size)


def _output_tolerance(contract: PrefillDecodeResultContract) -> tuple[float, float]:
    tolerance = get_tolerance_for_dtype(_dtype_from_name(contract.dtype))
    return (float(tolerance.rtol), float(tolerance.atol))


def _input_signature(
    contract: PrefillDecodeResultContract,
    *,
    tf32: bool,
) -> InputSignature:
    global_requests = contract.prefill_ranks * contract.requests_per_rank
    prompt_shape = (
        contract.batch_size,
        contract.context_window,
        contract.hidden_size,
    )
    output_shape = (global_requests, contract.batch_size, contract.hidden_size)
    dtype = _dtype_from_name(contract.dtype)
    flags = PrecisionFlags(
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
    )
    per_model_parameters = (contract.num_layers + 1) * contract.hidden_size**2
    return InputSignature(
        shapes={
            "prompt": prompt_shape,
            "decode_tokens": (contract.decode_tokens,),
            "hidden_size": (contract.hidden_size,),
            "num_layers": (contract.num_layers,),
            "output": output_shape,
        },
        dtypes={
            "prompt": contract.dtype,
            "decode_tokens": str(torch.float32),
            "hidden_size": str(torch.float32),
            "num_layers": str(torch.float32),
            "output": contract.dtype,
        },
        batch_size=global_requests,
        parameter_count=contract.world_size * per_model_parameters,
        precision_flags=PrecisionFlags(
            fp16=flags.fp16,
            bf16=flags.bf16,
            tf32=tf32,
        ),
        world_size=contract.world_size,
        ranks=list(range(contract.world_size)),
        pipeline_stages=2,
        pipeline_stage_boundaries=[
            (0, contract.prefill_ranks - 1),
            (contract.prefill_ranks, contract.world_size - 1),
        ],
        per_rank_batch_size=contract.requests_per_rank,
        collective_type="send_recv",
    )


class PrefillDecodeChildResultMixin:
    """Validate fresh source references against all measured decode outputs."""

    _prefill_decode_result_context: dict[str, Any] | None = None
    _prefill_decode_result_bundle: dict[str, Any] | None = None

    def prepare_prefill_decode_child_result(
        self,
        contract: PrefillDecodeResultContract,
    ) -> dict[str, str]:
        contract.validate()
        result_dir = Path(tempfile.mkdtemp(prefix="aisp-prefill-decode-result-"))
        token = uuid.uuid4().hex
        self._prefill_decode_result_context = {
            "result_dir": result_dir,
            "token": token,
            "contract": contract,
            "retention": "pending-child-result",
        }
        self._prefill_decode_result_bundle = None
        for attribute in (
            "_subprocess_verify_output",
            "_subprocess_output_tolerance",
            "_subprocess_input_signature",
            "_subprocess_verify_inputs",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return {
            PREFILL_DECODE_RESULT_DIR_ENV: str(result_dir),
            PREFILL_DECODE_RESULT_TOKEN_ENV: token,
            PREFILL_DECODE_RESULT_LABEL_ENV: contract.label,
        }

    def consume_prefill_decode_child_results(
        self,
        *,
        launch_wall_ns: int,
        finish_wall_ns: int,
        returncode: int,
        **_: Any,
    ) -> None:
        context = self._prefill_decode_result_context
        if context is None:
            raise RuntimeError("Prefill/decode child-result callback has no launch context")
        result_dir = cast(Path, context["result_dir"])
        context["retention"] = "retained-invalid-child-result"
        if returncode != 0:
            context["retention"] = "retained-child-failure"
            raise RuntimeError(
                "Prefill/decode child-result callback requires a clean child exit; "
                f"artifacts retained at {result_dir}"
            )

        expected = cast(PrefillDecodeResultContract, context["contract"])
        paths = sorted(result_dir.glob("rank-*.pt"))
        if len(paths) != expected.world_size:
            context["retention"] = "retained-incomplete-rank-quorum"
            raise RuntimeError(
                "Prefill/decode child-result rank quorum is incomplete: "
                f"expected {expected.world_size}, found {len(paths)}; "
                f"artifacts retained at {result_dir}"
            )

        expected_tolerance = _output_tolerance(expected)
        payloads: dict[int, dict[str, Any]] = {}
        common_signature: InputSignature | None = None
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Prefill/decode child result must be a regular file: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Invalid prefill/decode child-result payload at {path}")
            if payload.get("schema") != PREFILL_DECODE_RESULT_SCHEMA:
                raise RuntimeError(f"Invalid prefill/decode child-result schema at {path}")
            if payload.get("token") != context["token"]:
                raise RuntimeError(f"Prefill/decode child-result token mismatch at {path}")
            if payload.get("label") != expected.label:
                raise RuntimeError(f"Prefill/decode child-result label mismatch at {path}")
            contract_payload = payload.get("contract")
            if not isinstance(contract_payload, dict):
                raise RuntimeError(f"Prefill/decode child-result contract is missing at {path}")
            observed_contract = PrefillDecodeResultContract.from_dict(contract_payload)
            if observed_contract != expected:
                raise RuntimeError(f"Prefill/decode child-result workload mismatch at {path}")

            rank = int(payload.get("rank", -1))
            if rank < 0 or rank >= expected.world_size or rank in payloads:
                raise RuntimeError(f"Invalid or duplicate prefill/decode rank at {path}")
            if path.name != f"rank-{rank}.pt":
                raise RuntimeError(f"Prefill/decode child-result filename/rank mismatch at {path}")
            created_wall_ns = int(payload.get("created_wall_ns", 0))
            if not launch_wall_ns <= created_wall_ns <= finish_wall_ns:
                raise RuntimeError(f"Stale prefill/decode child-result payload at {path}")
            if int(payload.get("launch_wall_ns", 0)) != int(launch_wall_ns):
                raise RuntimeError(f"Prefill/decode child-result launch identity mismatch at {path}")

            signature_payload = payload.get("input_signature")
            if not isinstance(signature_payload, dict):
                raise RuntimeError(f"Prefill/decode input signature is missing at {path}")
            signature = InputSignature.from_dict(signature_payload)
            signature_errors = signature.validate(strict=True)
            if signature_errors:
                raise RuntimeError(
                    f"Invalid prefill/decode input signature at {path}: {signature_errors[0]}"
                )
            observed_tf32 = payload.get("tf32")
            if not isinstance(observed_tf32, bool):
                raise RuntimeError(f"Prefill/decode TF32 state is missing at {path}")
            expected_signature = _input_signature(expected, tf32=observed_tf32)
            if signature.to_dict() != expected_signature.to_dict():
                raise RuntimeError(f"Prefill/decode input signature mismatch at {path}")
            if common_signature is None:
                common_signature = signature
            elif signature.to_dict() != common_signature.to_dict():
                raise RuntimeError("Prefill/decode input signatures differ across ranks")
            if payload.get("output_tolerance") != list(expected_tolerance):
                raise RuntimeError(f"Prefill/decode output tolerance mismatch at {path}")

            is_prefill = rank < expected.prefill_ranks
            expected_role = "prefill" if is_prefill else "decode"
            if payload.get("role") != expected_role:
                raise RuntimeError(f"Prefill/decode child-result role mismatch at {path}")
            expected_assigned = [] if is_prefill else _assigned_prefills(expected, rank)
            if payload.get("assigned_prefills") != expected_assigned:
                raise RuntimeError(f"Prefill/decode rank assignment mismatch at {path}")
            verification_prompt = payload.get("verification_prompt")
            if rank == 0:
                prompt_shape = (
                    expected.batch_size,
                    expected.context_window,
                    expected.hidden_size,
                )
                if (
                    not isinstance(verification_prompt, torch.Tensor)
                    or verification_prompt.shape != prompt_shape
                    or verification_prompt.dtype != _dtype_from_name(expected.dtype)
                    or not bool(torch.isfinite(verification_prompt).all())
                ):
                    raise RuntimeError(
                        f"Prefill/decode verification prompt is invalid at {path}"
                    )
            elif verification_prompt is not None:
                raise RuntimeError(
                    f"Unexpected prefill/decode verification prompt from rank {rank}"
                )
            tensor_name = "reference_output" if is_prefill else "timed_output"
            absent_name = "timed_output" if is_prefill else "reference_output"
            tensor = payload.get(tensor_name)
            if payload.get(absent_name) is not None:
                raise RuntimeError(f"Prefill/decode {absent_name} must be absent at {path}")
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"Prefill/decode {tensor_name} is missing at {path}")
            expected_shape = _output_shape_for_rank(expected, rank)
            if tensor.dtype != _dtype_from_name(expected.dtype) or tensor.shape != expected_shape:
                raise RuntimeError(
                    f"Prefill/decode {tensor_name} shape/dtype mismatch at {path}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"Prefill/decode {tensor_name} is non-finite at {path}")
            payloads[rank] = payload

        references = {
            rank: cast(torch.Tensor, payloads[rank]["reference_output"])
            for rank in range(expected.prefill_ranks)
        }
        timed_by_prefill: dict[int, torch.Tensor] = {}
        for decode_rank in range(expected.prefill_ranks, expected.world_size):
            assigned = _assigned_prefills(expected, decode_rank)
            timed_output = cast(torch.Tensor, payloads[decode_rank]["timed_output"])
            for index, prefill_rank in enumerate(assigned):
                start = index * expected.requests_per_rank
                stop = start + expected.requests_per_rank
                timed_by_prefill[prefill_rank] = timed_output[start:stop]
        if set(timed_by_prefill) != set(references):
            raise RuntimeError("Prefill/decode measured outputs do not cover every prefill rank")
        if common_signature is None:
            raise RuntimeError("Prefill/decode child input signature is missing")

        global_reference = torch.cat(
            [references[rank] for rank in range(expected.prefill_ranks)], dim=0
        )
        global_timed = torch.cat(
            [timed_by_prefill[rank] for rank in range(expected.prefill_ranks)], dim=0
        )
        try:
            torch.testing.assert_close(
                global_timed,
                global_reference,
                rtol=expected_tolerance[0],
                atol=expected_tolerance[1],
            )
        except AssertionError as exc:
            raise RuntimeError(
                "Prefill/decode full timed output differs from the source-side reference"
            ) from exc

        self._subprocess_verify_inputs = {
            "prompt": cast(torch.Tensor, payloads[0]["verification_prompt"]),
            "decode_tokens": torch.zeros((expected.decode_tokens,), dtype=torch.float32),
            "hidden_size": torch.zeros((expected.hidden_size,), dtype=torch.float32),
            "num_layers": torch.zeros((expected.num_layers,), dtype=torch.float32),
        }
        self._subprocess_verify_output = global_timed
        self._subprocess_output_tolerance = expected_tolerance
        self._subprocess_input_signature = common_signature
        self._prefill_decode_result_bundle = {
            "contract": expected,
            "reference_output": global_reference,
            "timed_output": global_timed,
        }
        try:
            shutil.rmtree(result_dir)
        except OSError:
            context["retention"] = "retained-cleanup-failure"
        else:
            context["retention"] = "cleaned-after-success"

    def require_prefill_decode_child_result(self) -> None:
        if self._prefill_decode_result_bundle is None:
            context = self._prefill_decode_result_context
            retained = context.get("result_dir") if context else "unavailable"
            raise RuntimeError(
                "Prefill/decode verification requires a fresh measured child result; "
                f"retained path: {retained}"
            )

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        self.require_prefill_decode_child_result()
        return dict(self._subprocess_verify_inputs)

    def get_verify_output(self) -> torch.Tensor:
        self.require_prefill_decode_child_result()
        return self._subprocess_verify_output.detach().clone()

    def get_input_signature(self) -> InputSignature:
        self.require_prefill_decode_child_result()
        return self._subprocess_input_signature

    def get_output_tolerance(self) -> tuple[float, float]:
        self.require_prefill_decode_child_result()
        return self._subprocess_output_tolerance


def write_prefill_decode_child_result(
    *,
    contract: PrefillDecodeResultContract,
    rank: int,
    reference_output: torch.Tensor | None,
    timed_output: torch.Tensor | None,
    verification_prompt: torch.Tensor | None,
) -> bool:
    """Write the full role-specific output only for a callback-bound launch."""
    result_dir_value = os.environ.get(PREFILL_DECODE_RESULT_DIR_ENV)
    token = os.environ.get(PREFILL_DECODE_RESULT_TOKEN_ENV)
    expected_label = os.environ.get(PREFILL_DECODE_RESULT_LABEL_ENV)
    launch_wall_ns = os.environ.get(PREFILL_DECODE_LAUNCH_WALL_NS_ENV)
    if not launch_wall_ns:
        return False
    if not all((result_dir_value, token, expected_label)):
        raise RuntimeError("Prefill/decode child-result environment is incomplete")

    contract.validate()
    if expected_label != contract.label:
        raise RuntimeError(
            f"Prefill/decode child label mismatch: {expected_label!r} != {contract.label!r}"
        )
    if rank < 0 or rank >= contract.world_size:
        raise RuntimeError(f"Invalid prefill/decode child rank: {rank}")
    is_prefill = rank < contract.prefill_ranks
    expected_dtype = _dtype_from_name(contract.dtype)
    tf32_enabled = bool(torch.backends.cuda.matmul.allow_tf32)
    if rank == 0:
        prompt_shape = (
            contract.batch_size,
            contract.context_window,
            contract.hidden_size,
        )
        if (
            not isinstance(verification_prompt, torch.Tensor)
            or verification_prompt.shape != prompt_shape
            or verification_prompt.dtype != expected_dtype
            or not bool(torch.isfinite(verification_prompt).all())
        ):
            raise RuntimeError(
                "Prefill/decode rank 0 must provide the declared verification prompt"
            )
    elif verification_prompt is not None:
        raise RuntimeError("Only prefill rank 0 may provide a verification prompt")
    tensor = reference_output if is_prefill else timed_output
    absent = timed_output if is_prefill else reference_output
    if absent is not None or not isinstance(tensor, torch.Tensor):
        raise RuntimeError("Prefill/decode child must provide exactly its role-specific output")
    expected_shape = _output_shape_for_rank(contract, rank)
    if tensor.shape != expected_shape or tensor.dtype != expected_dtype:
        raise RuntimeError("Prefill/decode child output does not match its declared workload")
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError("Prefill/decode child output contains non-finite values")

    result_dir = Path(cast(str, result_dir_value))
    if result_dir.is_symlink() or not result_dir.is_dir():
        raise RuntimeError(
            "Prefill/decode child-result directory must be the prepared regular directory"
        )
    payload = {
        "schema": PREFILL_DECODE_RESULT_SCHEMA,
        "token": token,
        "label": contract.label,
        "rank": int(rank),
        "role": "prefill" if is_prefill else "decode",
        "assigned_prefills": [] if is_prefill else _assigned_prefills(contract, rank),
        "tf32": tf32_enabled,
        "contract": contract.to_dict(),
        "launch_wall_ns": int(launch_wall_ns),
        "created_wall_ns": time.time_ns(),
        "reference_output": (
            tensor.detach().to(device="cpu").contiguous() if is_prefill else None
        ),
        "timed_output": (
            tensor.detach().to(device="cpu").contiguous() if not is_prefill else None
        ),
        "verification_prompt": (
            verification_prompt.detach().to(device="cpu").contiguous()
            if verification_prompt is not None
            else None
        ),
        "input_signature": _input_signature(contract, tf32=tf32_enabled).to_dict(),
        "output_tolerance": list(_output_tolerance(contract)),
    }
    temporary_path = result_dir / f".rank-{rank}-{os.getpid()}.tmp"
    destination = result_dir / f"rank-{rank}.pt"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, destination)
    return True


__all__ = [
    "PREFILL_DECODE_LAUNCH_WALL_NS_ENV",
    "PREFILL_DECODE_RESULT_CALLBACK",
    "PREFILL_DECODE_RESULT_DIR_ENV",
    "PREFILL_DECODE_RESULT_LABEL_ENV",
    "PREFILL_DECODE_RESULT_SCHEMA",
    "PREFILL_DECODE_RESULT_TOKEN_ENV",
    "PrefillDecodeChildResultMixin",
    "PrefillDecodeResultContract",
    "make_prefill_decode_result_contract",
    "write_prefill_decode_child_result",
]
