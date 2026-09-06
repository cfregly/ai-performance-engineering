"""Fresh full-rank result transport for the Chapter 4 disaggregated pair."""

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

from core.benchmark.verification import InputSignature, PrecisionFlags

RESULT_CALLBACK = "consume_disaggregated_child_results"
RESULT_DIR_ENV = "AISP_DISAGGREGATED_RESULT_DIR"
RESULT_TOKEN_ENV = "AISP_DISAGGREGATED_RESULT_TOKEN"
RESULT_VARIANT_ENV = "AISP_DISAGGREGATED_VARIANT"
LAUNCH_WALL_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS"
RESULT_SCHEMA = "aisp.ch04-disaggregated.child-result.v1"
OUTPUT_TOLERANCE = (1e-5, 1e-5)


@dataclass(frozen=True)
class DisaggregatedResultContract:
    variant: str
    world_size: int
    batch_size: int
    prefill_len: int
    hidden_dim: int
    iterations: int
    warmup: int
    seed: int

    def validate(self) -> None:
        if self.variant not in {"baseline", "optimized"}:
            raise ValueError(f"Unsupported disaggregated variant: {self.variant!r}")
        if self.world_size != 2:
            raise ValueError("Chapter 4 disaggregated worker requires exactly two ranks")
        for name in ("batch_size", "prefill_len", "hidden_dim", "iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Disaggregated {name} must be a positive integer")
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int) or self.warmup < 0:
            raise ValueError("Disaggregated warmup must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Disaggregated seed must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DisaggregatedResultContract:
        if not isinstance(payload, dict):
            raise TypeError("Disaggregated result contract must be a dictionary")
        try:
            contract = cls(**payload)
        except TypeError as exc:
            raise ValueError("Invalid disaggregated result contract fields") from exc
        contract.validate()
        return contract


def make_result_contract(
    *,
    variant: str,
    world_size: int,
    batch_size: int,
    prefill_len: int,
    hidden_dim: int,
    iterations: int,
    warmup: int,
    seed: int,
) -> DisaggregatedResultContract:
    contract = DisaggregatedResultContract(
        variant=variant,
        world_size=int(world_size),
        batch_size=int(batch_size),
        prefill_len=int(prefill_len),
        hidden_dim=int(hidden_dim),
        iterations=int(iterations),
        warmup=int(warmup),
        seed=int(seed),
    )
    contract.validate()
    return contract


def _parameter_count(contract: DisaggregatedResultContract) -> int:
    # Two logical phase models. Rank replicas do not change the workload model.
    per_model = 4 * contract.hidden_dim**2 + 3 * contract.hidden_dim
    return 2 * per_model


def _signature(contract: DisaggregatedResultContract) -> InputSignature:
    prefill_shape = (
        contract.world_size,
        contract.batch_size,
        contract.prefill_len,
        contract.hidden_dim,
    )
    decode_shape = (
        contract.world_size,
        contract.batch_size,
        1,
        contract.hidden_dim,
    )
    output_shape = (
        contract.world_size,
        contract.batch_size,
        contract.prefill_len + 1,
        contract.hidden_dim,
    )
    return InputSignature(
        shapes={"prefill": prefill_shape, "decode": decode_shape, "output": output_shape},
        dtypes={
            "prefill": str(torch.float32),
            "decode": str(torch.float32),
            "output": str(torch.float32),
        },
        batch_size=contract.batch_size,
        parameter_count=_parameter_count(contract),
        precision_flags=PrecisionFlags(),
        world_size=contract.world_size,
        ranks=list(range(contract.world_size)),
        per_rank_batch_size=contract.batch_size,
        collective_type="all_reduce",
        collective_algorithm="sum_then_divide_world_size",
    )


class DisaggregatedChildResultMixin:
    """Validate a fresh quorum and expose the actual measured full outputs."""

    _disaggregated_result_context: dict[str, Any] | None = None
    _disaggregated_result_bundle: dict[str, Any] | None = None

    def prepare_disaggregated_child_result(
        self, contract: DisaggregatedResultContract
    ) -> dict[str, str]:
        contract.validate()
        result_dir = Path(tempfile.mkdtemp(prefix="aisp-ch04-disaggregated-result-"))
        token = uuid.uuid4().hex
        self._disaggregated_result_context = {
            "result_dir": result_dir,
            "token": token,
            "contract": contract,
            "retention": "pending-child-result",
        }
        self._disaggregated_result_bundle = None
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

    def consume_disaggregated_child_results(
        self,
        *,
        launch_wall_ns: int,
        finish_wall_ns: int,
        returncode: int,
        **_: Any,
    ) -> None:
        context = self._disaggregated_result_context
        if context is None:
            raise RuntimeError("Disaggregated child-result callback has no launch context")
        result_dir = cast(Path, context["result_dir"])
        context["retention"] = "retained-invalid-child-result"
        if returncode != 0:
            context["retention"] = "retained-child-failure"
            raise RuntimeError(
                "Disaggregated child-result callback requires a clean child exit; "
                f"artifacts retained at {result_dir}"
            )

        expected = cast(DisaggregatedResultContract, context["contract"])
        paths = sorted(result_dir.glob("rank-*.pt"))
        if len(paths) != expected.world_size:
            context["retention"] = "retained-incomplete-rank-quorum"
            raise RuntimeError(
                "Disaggregated child-result rank quorum is incomplete: "
                f"expected {expected.world_size}, found {len(paths)}; "
                f"artifacts retained at {result_dir}"
            )

        payloads: dict[int, dict[str, Any]] = {}
        expected_signature = _signature(expected)
        expected_shapes = {
            "prefill_input": (
                expected.batch_size,
                expected.prefill_len,
                expected.hidden_dim,
            ),
            "decode_input": (expected.batch_size, 1, expected.hidden_dim),
            "reference_output": (
                expected.batch_size,
                expected.prefill_len + 1,
                expected.hidden_dim,
            ),
            "timed_output": (
                expected.batch_size,
                expected.prefill_len + 1,
                expected.hidden_dim,
            ),
        }
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Disaggregated child result must be a regular file: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("schema") != RESULT_SCHEMA:
                raise RuntimeError(f"Invalid disaggregated child-result payload at {path}")
            if payload.get("token") != context["token"]:
                raise RuntimeError(f"Disaggregated child-result token mismatch at {path}")
            if payload.get("variant") != expected.variant:
                raise RuntimeError(f"Disaggregated child-result variant mismatch at {path}")
            observed_contract = DisaggregatedResultContract.from_dict(payload.get("contract"))
            if observed_contract != expected:
                raise RuntimeError(f"Disaggregated child-result workload mismatch at {path}")
            rank = int(payload.get("rank", -1))
            if rank < 0 or rank >= expected.world_size or rank in payloads:
                raise RuntimeError(f"Invalid or duplicate disaggregated rank at {path}")
            if path.name != f"rank-{rank}.pt":
                raise RuntimeError(f"Disaggregated child filename/rank mismatch at {path}")
            created = int(payload.get("created_wall_ns", 0))
            if not launch_wall_ns <= created <= finish_wall_ns:
                raise RuntimeError(f"Stale disaggregated child result at {path}")
            if int(payload.get("launch_wall_ns", 0)) != launch_wall_ns:
                raise RuntimeError(f"Disaggregated child launch identity mismatch at {path}")
            observed_signature = InputSignature.from_dict(payload.get("input_signature"))
            if observed_signature != expected_signature:
                raise RuntimeError(f"Disaggregated child input signature mismatch at {path}")
            for tensor_name, shape in expected_shapes.items():
                tensor = payload.get(tensor_name)
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.dtype != torch.float32
                    or tensor.shape != shape
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise RuntimeError(
                        f"Disaggregated {tensor_name} shape/dtype/value mismatch at {path}"
                    )
            try:
                torch.testing.assert_close(
                    payload["timed_output"],
                    payload["reference_output"],
                    rtol=OUTPUT_TOLERANCE[0],
                    atol=OUTPUT_TOLERANCE[1],
                )
            except AssertionError as exc:
                raise RuntimeError(
                    f"Disaggregated rank {rank} full timed output differs from reference"
                ) from exc
            payloads[rank] = payload

        ordered = [payloads[rank] for rank in range(expected.world_size)]
        for tensor_name in ("prefill_input", "decode_input", "timed_output"):
            first = ordered[0][tensor_name]
            for rank, payload in enumerate(ordered[1:], 1):
                try:
                    torch.testing.assert_close(
                        payload[tensor_name], first, rtol=0, atol=0
                    )
                except AssertionError as exc:
                    raise RuntimeError(
                        f"Disaggregated seed-42 runtime parity failed for rank {rank} "
                        f"{tensor_name}"
                    ) from exc

        self._subprocess_verify_inputs = {
            "prefill": torch.stack([item["prefill_input"] for item in ordered]),
            "decode": torch.stack([item["decode_input"] for item in ordered]),
        }
        self._subprocess_verify_output = torch.stack(
            [item["timed_output"] for item in ordered]
        )
        self._subprocess_input_signature = expected_signature
        self._subprocess_output_tolerance = OUTPUT_TOLERANCE
        self._disaggregated_result_bundle = {
            "contract": expected,
            "rank_outputs": [item["timed_output"] for item in ordered],
        }
        try:
            shutil.rmtree(result_dir)
        except OSError:
            context["retention"] = "retained-cleanup-failure"
        else:
            context["retention"] = "cleaned-after-success"

    def require_disaggregated_child_result(self) -> None:
        if self._disaggregated_result_bundle is None:
            context = self._disaggregated_result_context
            retained = context.get("result_dir") if context else "unavailable"
            raise RuntimeError(
                "Disaggregated verification requires a fresh full-rank measured result; "
                f"retained path: {retained}"
            )

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        if self._disaggregated_result_bundle is not None:
            return dict(self._subprocess_verify_inputs)
        return super().get_verify_inputs()  # type: ignore[misc]

    def get_verify_output(self) -> torch.Tensor:
        if self._disaggregated_result_bundle is not None:
            return self._subprocess_verify_output.detach().clone()
        return super().get_verify_output()  # type: ignore[misc]

    def get_input_signature(self) -> InputSignature:
        if self._disaggregated_result_bundle is not None:
            return self._subprocess_input_signature
        return super().get_input_signature()  # type: ignore[misc]

    def get_output_tolerance(self) -> tuple[float, float]:
        if self._disaggregated_result_bundle is not None:
            return self._subprocess_output_tolerance
        return super().get_output_tolerance()  # type: ignore[misc]


def write_disaggregated_child_result(
    *,
    contract: DisaggregatedResultContract,
    rank: int,
    prefill_input: torch.Tensor,
    decode_input: torch.Tensor,
    reference_output: torch.Tensor,
    timed_output: torch.Tensor,
) -> bool:
    """Atomically write one rank's actual final measured output."""
    result_dir_value = os.environ.get(RESULT_DIR_ENV)
    token = os.environ.get(RESULT_TOKEN_ENV)
    expected_variant = os.environ.get(RESULT_VARIANT_ENV)
    launch_wall_ns = os.environ.get(LAUNCH_WALL_NS_ENV)
    if not launch_wall_ns:
        return False
    if not all((result_dir_value, token, expected_variant)):
        raise RuntimeError("Disaggregated child-result environment is incomplete")
    contract.validate()
    if expected_variant != contract.variant:
        raise RuntimeError("Disaggregated child-result variant does not match launch")
    if rank < 0 or rank >= contract.world_size:
        raise RuntimeError(f"Invalid disaggregated child rank: {rank}")

    shapes = {
        "prefill_input": (contract.batch_size, contract.prefill_len, contract.hidden_dim),
        "decode_input": (contract.batch_size, 1, contract.hidden_dim),
        "reference_output": (
            contract.batch_size,
            contract.prefill_len + 1,
            contract.hidden_dim,
        ),
        "timed_output": (
            contract.batch_size,
            contract.prefill_len + 1,
            contract.hidden_dim,
        ),
    }
    tensors = {
        "prefill_input": prefill_input,
        "decode_input": decode_input,
        "reference_output": reference_output,
        "timed_output": timed_output,
    }
    for name, tensor in tensors.items():
        if (
            tensor.shape != shapes[name]
            or tensor.dtype != torch.float32
            or not bool(torch.isfinite(tensor).all())
        ):
            raise RuntimeError(f"Disaggregated {name} does not match the declared workload")
    try:
        torch.testing.assert_close(
            timed_output,
            reference_output,
            rtol=OUTPUT_TOLERANCE[0],
            atol=OUTPUT_TOLERANCE[1],
        )
    except AssertionError as exc:
        raise RuntimeError("Disaggregated full timed output differs from reference") from exc

    result_dir = Path(cast(str, result_dir_value))
    if result_dir.is_symlink() or not result_dir.is_dir():
        raise RuntimeError("Disaggregated child-result directory must be prepared and regular")
    payload = {
        "schema": RESULT_SCHEMA,
        "token": token,
        "variant": contract.variant,
        "rank": int(rank),
        "contract": contract.to_dict(),
        "launch_wall_ns": int(launch_wall_ns),
        "created_wall_ns": time.time_ns(),
        "input_signature": _signature(contract).to_dict(),
        **{
            name: tensor.detach().to(device="cpu").contiguous()
            for name, tensor in tensors.items()
        },
    }
    temporary = result_dir / f".rank-{rank}-{os.getpid()}.tmp"
    destination = result_dir / f"rank-{rank}.pt"
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return True


__all__ = [
    "DisaggregatedChildResultMixin",
    "DisaggregatedResultContract",
    "OUTPUT_TOLERANCE",
    "RESULT_CALLBACK",
    "make_result_contract",
    "write_disaggregated_child_result",
]
