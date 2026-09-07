"""Fresh full-rank result transport for the communicator-reinit pair."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from core.benchmark.verification import InputSignature, PrecisionFlags

if TYPE_CHECKING:
    from core.harness.benchmark_harness import BenchmarkConfig, TorchrunLaunchSpec


WORLD_SIZE = 2
OUTPUT_TOLERANCE = (1e-6, 1e-6)
RESULT_CALLBACK = "consume_reinit_comm_child_results"
RESULT_DIR_ENV = "AISP_REINIT_COMM_RESULT_DIR"
RESULT_TOKEN_ENV = "AISP_REINIT_COMM_RESULT_TOKEN"
RESULT_VARIANT_ENV = "AISP_REINIT_COMM_VARIANT"
LAUNCH_WALL_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS"
LAUNCH_MONOTONIC_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_MONOTONIC_NS"
RESULT_SCHEMA = "aisp.reinit-comm.child-result.v1"


def _input_signature(world_size: int) -> InputSignature:
    rank_shapes = {f"rank_{rank}_input": (1, 1) for rank in range(world_size)}
    rank_dtypes = {f"rank_{rank}_input": str(torch.float32) for rank in range(world_size)}
    return InputSignature(
        shapes={**rank_shapes, "output": (world_size, 1, 1)},
        dtypes={**rank_dtypes, "output": str(torch.float32)},
        batch_size=1,
        parameter_count=0,
        precision_flags=PrecisionFlags(tf32=False),
        world_size=world_size,
        ranks=list(range(world_size)),
        collective_type="all_reduce",
        collective_algorithm="nccl_sum",
    )


def _require_scalar(name: str, value: object, *, path: Path) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Reinit-comm {name} is missing at {path}")
    if value.dtype != torch.float32 or value.shape != (1, 1):
        raise RuntimeError(f"Reinit-comm {name} shape/dtype mismatch at {path}")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"Reinit-comm {name} contains non-finite values at {path}")
    return value


class ReinitCommChildResultMixin:
    """Expose only fresh outputs produced by the measured two-rank worker."""

    _reinit_comm_variant: str
    _reinit_comm_result_context: dict[str, Any] | None = None
    _reinit_comm_result_bundle: dict[str, Any] | None = None

    def prepare_reinit_comm_child_result(
        self,
        *,
        variant: str,
        world_size: int,
        iterations: int,
        warmup: int,
    ) -> dict[str, str]:
        if variant not in {"baseline", "optimized"}:
            raise ValueError(f"Unsupported reinit-comm variant: {variant!r}")
        if world_size != WORLD_SIZE:
            raise RuntimeError(f"SKIPPED: reinit-comm requires exactly {WORLD_SIZE} ranks")
        if iterations <= 0 or warmup < 0:
            raise ValueError("Reinit-comm iterations must be positive and warmup non-negative")
        previous = self._reinit_comm_result_context
        if previous is not None and Path(previous["result_dir"]).exists():
            raise RuntimeError("Refusing to replace an unconsumed reinit-comm child-result context")

        result_dir = Path(tempfile.mkdtemp(prefix="aisp-reinit-comm-result-"))
        token = uuid.uuid4().hex
        self._reinit_comm_result_context = {
            "result_dir": result_dir,
            "token": token,
            "variant": variant,
            "world_size": world_size,
            "iterations": iterations,
            "warmup": warmup,
            "retention": "pending-child-result",
        }
        self._reinit_comm_result_bundle = None
        for attribute in (
            "_subprocess_verify_inputs",
            "_subprocess_verify_output",
            "_subprocess_output_tolerance",
            "_subprocess_input_signature",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return {
            RESULT_DIR_ENV: str(result_dir),
            RESULT_TOKEN_ENV: token,
            RESULT_VARIANT_ENV: variant,
        }

    def get_torchrun_spec(self, config: BenchmarkConfig | None = None) -> TorchrunLaunchSpec:
        from core.harness.benchmark_harness import TorchrunLaunchSpec

        effective = config or self.get_config()  # type: ignore[attr-defined]
        if int(effective.nnodes or 1) != 1:
            raise RuntimeError("Reinit-comm child results require nnodes == 1")
        world_size = int(effective.nproc_per_node or WORLD_SIZE)
        iterations = int(effective.iterations or 0)
        warmup = int(effective.warmup or 0)
        env = self.prepare_reinit_comm_child_result(
            variant=self._reinit_comm_variant,
            world_size=world_size,
            iterations=iterations,
            warmup=warmup,
        )
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve().with_name("reinit_comm_multigpu_worker.py"),
            script_args=["--variant", self._reinit_comm_variant],
            env={
                "NCCL_DEBUG": "WARN",
                "OMP_NUM_THREADS": "1",
                "MASTER_PORT": os.environ.get("MASTER_PORT", "29524"),
                **env,
            },
            multi_gpu_required=True,
            name=f"{self._reinit_comm_variant}_reinit_comm_multigpu",
            config_arg_map={"iterations": "--iterations", "warmup": "--warmup"},
            result_callback=RESULT_CALLBACK,
            timing_source="rank0_time_per_iter_ms",
            timing_iterations_per_sample=iterations,
        )

    def consume_reinit_comm_child_results(
        self,
        *,
        launch_wall_ns: int,
        launch_monotonic_ns: int,
        finish_wall_ns: int,
        finish_monotonic_ns: int,
        returncode: int,
        **_: Any,
    ) -> None:
        context = self._reinit_comm_result_context
        if context is None:
            raise RuntimeError("Reinit-comm child-result callback has no launch context")
        result_dir = cast(Path, context["result_dir"])

        def fail(status: str, message: str) -> None:
            context["retention"] = status
            raise RuntimeError(f"{message}; artifacts retained at {result_dir}")

        if returncode != 0:
            fail("retained-child-failure", "Reinit-comm worker did not exit cleanly")
        paths = sorted(result_dir.glob("rank-*.pt"))
        if len(paths) != int(context["world_size"]):
            fail(
                "retained-incomplete-rank-quorum",
                "Reinit-comm child-result rank quorum is incomplete",
            )

        payloads: list[tuple[int, dict[str, Any]]] = []
        seen_ranks: set[int] = set()
        expected_signature = _input_signature(int(context["world_size"]))
        for path in paths:
            if path.is_symlink() or not path.is_file():
                fail("retained-invalid-result", f"Result is not a regular file: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("schema") != RESULT_SCHEMA:
                fail("retained-invalid-result", f"Invalid result schema at {path}")
            for key in ("token", "variant"):
                if payload.get(key) != context[key]:
                    fail("retained-invalid-result", f"Result {key} mismatch at {path}")
            for key in ("world_size", "iterations", "warmup"):
                if int(payload.get(key, -1)) != int(context[key]):
                    fail("retained-invalid-result", f"Result {key} mismatch at {path}")
            rank = int(payload.get("rank", -1))
            if rank not in range(int(context["world_size"])) or rank in seen_ranks:
                fail("retained-invalid-result", f"Invalid or duplicate rank at {path}")
            if path.name != f"rank-{rank}.pt":
                fail("retained-invalid-result", f"Filename/rank mismatch at {path}")
            seen_ranks.add(rank)

            created_wall_ns = int(payload.get("created_wall_ns", 0))
            created_monotonic_ns = int(payload.get("created_monotonic_ns", 0))
            if not launch_wall_ns <= created_wall_ns <= finish_wall_ns:
                fail("retained-stale-result", f"Stale wall clock at {path}")
            if not launch_monotonic_ns <= created_monotonic_ns <= finish_monotonic_ns:
                fail("retained-stale-result", f"Stale monotonic clock at {path}")
            if int(payload.get("launch_wall_ns", 0)) != launch_wall_ns:
                fail("retained-stale-result", f"Wall launch identity mismatch at {path}")
            if int(payload.get("launch_monotonic_ns", 0)) != launch_monotonic_ns:
                fail("retained-stale-result", f"Monotonic launch identity mismatch at {path}")

            _require_scalar("input", payload.get("input"), path=path)
            _require_scalar("output", payload.get("output"), path=path)
            signature_payload = payload.get("input_signature")
            if not isinstance(signature_payload, dict):
                fail("retained-invalid-result", f"Input signature is missing at {path}")
            signature = InputSignature.from_dict(signature_payload)
            errors = signature.validate(strict=True)
            if errors or signature.to_dict() != expected_signature.to_dict():
                fail("retained-invalid-result", f"Input signature mismatch at {path}")
            if payload.get("output_tolerance") != list(OUTPUT_TOLERANCE):
                fail("retained-invalid-result", f"Output tolerance mismatch at {path}")
            time_per_iter_ms = float(payload.get("time_per_iter_ms", 0.0))
            if not math.isfinite(time_per_iter_ms) or time_per_iter_ms <= 0:
                fail("retained-invalid-result", f"Invalid host timing at {path}")
            if payload.get("timing_source") != "max_rank_host_perf_counter":
                fail("retained-invalid-result", f"Timing source mismatch at {path}")
            payloads.append((rank, payload))

        payloads.sort(key=lambda item: item[0])
        expected_output = torch.zeros((1, 1), dtype=torch.float32)
        for _, payload in payloads:
            expected_output.add_(payload["input"])
        outputs = []
        for rank, payload in payloads:
            if not torch.equal(payload["output"], expected_output):
                fail(
                    "retained-invalid-output",
                    f"Rank {rank} output does not equal the full-rank input sum",
                )
            outputs.append(payload["output"])
        if any(
            payload["time_per_iter_ms"] != payloads[0][1]["time_per_iter_ms"]
            for _, payload in payloads[1:]
        ):
            fail("retained-invalid-result", "Ranks disagree on max-rank host timing")

        self._subprocess_verify_inputs = {
            f"rank_{rank}_input": payload["input"] for rank, payload in payloads
        }
        self._subprocess_verify_output = torch.stack(outputs)
        self._subprocess_output_tolerance = OUTPUT_TOLERANCE
        self._subprocess_input_signature = expected_signature
        self._reinit_comm_result_bundle = {
            "payloads": payloads,
            "expected_output": expected_output,
        }
        try:
            shutil.rmtree(result_dir)
        except OSError:
            context["retention"] = "retained-cleanup-failure"
        else:
            context["retention"] = "cleaned-after-success"

    def require_reinit_comm_child_result(self) -> None:
        if self._reinit_comm_result_bundle is None:
            context = self._reinit_comm_result_context
            retained = context.get("result_dir") if context else "unavailable"
            raise RuntimeError(
                "Reinit-comm verification requires a fresh measured child result; "
                f"retained path: {retained}"
            )

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        if self._reinit_comm_result_bundle is not None:
            return dict(self._subprocess_verify_inputs)
        return super().get_verify_inputs()  # type: ignore[misc]

    def get_verify_output(self) -> torch.Tensor:
        if self._reinit_comm_result_bundle is not None:
            return self._subprocess_verify_output.detach().clone()
        return super().get_verify_output()  # type: ignore[misc]

    def get_input_signature(self) -> InputSignature:
        if self._reinit_comm_result_bundle is not None:
            return self._subprocess_input_signature
        return super().get_input_signature()  # type: ignore[misc]

    def get_output_tolerance(self) -> tuple[float, float]:
        if self._reinit_comm_result_bundle is not None:
            return self._subprocess_output_tolerance
        return super().get_output_tolerance()  # type: ignore[misc]

    def validate_reinit_comm_child_result(self) -> str | None:
        if self._reinit_comm_result_context is None:
            return None
        if self._reinit_comm_result_bundle is None:
            return "Fresh full-rank reinit-comm worker output is missing"
        return None


def write_reinit_comm_child_result(
    *,
    variant: str,
    rank: int,
    world_size: int,
    iterations: int,
    warmup: int,
    input_tensor: torch.Tensor,
    output: torch.Tensor,
    time_per_iter_ms: float,
) -> bool:
    """Atomically persist one rank's actual final timed output."""
    result_dir_value = os.environ.get(RESULT_DIR_ENV)
    token = os.environ.get(RESULT_TOKEN_ENV)
    expected_variant = os.environ.get(RESULT_VARIANT_ENV)
    launch_wall_ns = os.environ.get(LAUNCH_WALL_NS_ENV)
    launch_monotonic_ns = os.environ.get(LAUNCH_MONOTONIC_NS_ENV)
    if not launch_wall_ns and not launch_monotonic_ns:
        return False
    if not all(
        (
            result_dir_value,
            token,
            expected_variant,
            launch_wall_ns,
            launch_monotonic_ns,
        )
    ):
        raise RuntimeError("Reinit-comm child-result environment is incomplete")
    if expected_variant != variant:
        raise RuntimeError("Reinit-comm worker variant does not match its launch contract")
    if world_size != WORLD_SIZE or rank not in range(world_size):
        raise RuntimeError("Reinit-comm worker rank topology is invalid")
    if not math.isfinite(time_per_iter_ms) or time_per_iter_ms <= 0:
        raise RuntimeError("Reinit-comm worker host timing must be finite and positive")
    input_cpu = _require_scalar("input", input_tensor.detach().cpu(), path=Path("worker"))
    output_cpu = _require_scalar("output", output.detach().cpu(), path=Path("worker"))
    signature = _input_signature(world_size)
    payload = {
        "schema": RESULT_SCHEMA,
        "token": token,
        "variant": variant,
        "rank": rank,
        "world_size": world_size,
        "iterations": iterations,
        "warmup": warmup,
        "launch_wall_ns": int(cast(str, launch_wall_ns)),
        "launch_monotonic_ns": int(cast(str, launch_monotonic_ns)),
        "created_wall_ns": time.time_ns(),
        "created_monotonic_ns": time.monotonic_ns(),
        "input": input_cpu.contiguous(),
        "output": output_cpu.contiguous(),
        "time_per_iter_ms": float(time_per_iter_ms),
        "timing_source": "max_rank_host_perf_counter",
        "input_signature": signature.to_dict(),
        "output_tolerance": list(OUTPUT_TOLERANCE),
    }
    result_dir = Path(cast(str, result_dir_value))
    if result_dir.is_symlink() or not result_dir.is_dir():
        raise RuntimeError("Reinit-comm result directory is not the prepared directory")
    temporary = result_dir / f".rank-{rank}-{os.getpid()}.tmp"
    destination = result_dir / f"rank-{rank}.pt"
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return True


__all__ = [
    "LAUNCH_MONOTONIC_NS_ENV",
    "LAUNCH_WALL_NS_ENV",
    "OUTPUT_TOLERANCE",
    "RESULT_CALLBACK",
    "RESULT_DIR_ENV",
    "RESULT_SCHEMA",
    "RESULT_TOKEN_ENV",
    "RESULT_VARIANT_ENV",
    "ReinitCommChildResultMixin",
    "WORLD_SIZE",
    "write_reinit_comm_child_result",
]
