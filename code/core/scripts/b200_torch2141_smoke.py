#!/usr/bin/env python3
"""Run a bounded B200 compatibility smoke for the PyTorch 2.14.1 stack.

This is a diagnostic runner. It performs tiny PyTorch, Triton, FlashInfer,
NCCL, and DTensor CUDA operations, import and ABI checks for the declared
companion packages, and runs a small manifest of real repository CUDA
entrypoints. Full mode also imports every owned chapter and lab
baseline/optimized module plus the checked compatibility-import manifest. It
does not produce performance evidence.

Run from the ``code`` directory:

    python -m core.scripts.b200_torch2141_smoke

For short B200 validation that prioritizes the required GPU checks and six GPU
entrypoints while excluding the optional vLLM ABI import:

    python -m core.scripts.b200_torch2141_smoke --mode short-gpu

The default deadline is four minutes. A missed deadline, an incomplete
entrypoint run, any attempted check failure, or an incomplete full-mode import
sweep returns a nonzero exit code. Full mode requires every declared companion
package. Short GPU mode retains the declared vLLM pin but does not validate it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import select
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, redirect_stdout, suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import import_module, invalidate_caches, metadata
from pathlib import Path
from typing import Any, BinaryIO

from core.utils.process_groups import ProcessGroupRegistry

EXPECTED_VERSIONS = {
    "cuda-python": "13.1.1",
    "flashinfer-python": "0.7.2",
    "nvidia-nccl-cu13": "2.30.7",
    "torch": "2.14.1+cu130",
    "torch-c-dlpack-ext": "0.1.5+torch2141.nobackend1",
    "torchao": "0.18.0+cu130",
    "torchaudio": "2.11.0+cu130",
    "torchtitan": "0.3.0",
    "torchvision": "0.29.1+cu130",
    "transformer-engine": "2.20.2",
    "transformer-engine-cu13": "2.20.2",
    "transformer-engine-torch": "2.20.2",
    "triton": "3.8.0",
    "vllm": "0.33.0.dev0+cu130.torch2141",
}

DEFAULT_DEADLINE_SECONDS = 240.0
DEFAULT_IMPORT_TIMEOUT_SECONDS = 10.0
FLASHINFER_CHECK_TIMEOUT_SECONDS = 45.0
NCCL_DTENSOR_CHECK_TIMEOUT_SECONDS = 30.0
TRANSFORMER_ENGINE_CHECK_TIMEOUT_SECONDS = 20.0
PROCESS_GROUP_CLEANUP_RESERVE_SECONDS = 2.0
PROCESS_GROUP_TERM_GRACE_SECONDS = 0.25
PROCESS_GROUP_KILL_WAIT_SECONDS = 0.75
SMOKE_MODE_FULL = "full"
SMOKE_MODE_SHORT_GPU = "short-gpu"
SMOKE_MODES = (SMOKE_MODE_FULL, SMOKE_MODE_SHORT_GPU)
COMPATIBILITY_IMPORT_MANIFEST = Path("core/b200_torch2141_runnable_examples.txt")
ENTRYPOINT_MANIFEST = Path("core/b200_torch2141_entrypoints.json")
ENTRYPOINT_MANIFEST_SCHEMA_VERSION = 1
MAX_CAPTURED_OUTPUT_CHARS = 16_384
_IMPORT_WORKER_FLAG = "--internal-import-worker"
_GPU_CHECK_WORKER_FLAG = "--internal-gpu-check-worker"
_GPU_CHECK_REPORT_PREFIX = "B200_GPU_CHECK_REPORT="
_IMPORT_WORKER_POLL_SECONDS = 0.05
_PROCESS_GROUPS = ProcessGroupRegistry()
_ACTIVE_CHILD_PROCESSES = _PROCESS_GROUPS.active

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_NOT_RUN = "not_run"

IMPORT_FAILURE_EXCEPTION = "exception"
IMPORT_FAILURE_TIMEOUT = "timeout"
IMPORT_FAILURE_SIGNAL = "signal"
IMPORT_FAILURE_EXIT = "exit"
IMPORT_FAILURE_LAUNCH = "launch_error"
IMPORT_FAILURE_PROTOCOL = "protocol_error"

POLICY_REQUIRED = "required"
POLICY_NOT_REQUESTED = "not_requested"

IMPORT_COVERAGE_COMPLETE = "complete"
IMPORT_COVERAGE_INCOMPLETE = "incomplete"
IMPORT_COVERAGE_NOT_REQUESTED = "not_requested"

# ``tl`` is populated only inside the Triton check. Keeping the import lazy
# makes this module safe to import in CPU-only unit tests.
tl: Any = None


class SmokeDeadlineExceededError(TimeoutError):
    """A bounded smoke action exceeded its wall-clock allowance."""


@dataclass(frozen=True)
class CheckResult:
    """Result of one required stack check."""

    name: str
    policy: str
    status: str
    duration_ms: float
    details: dict[str, Any] = field(default_factory=dict)
    diagnostic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EntrypointSpec:
    """One bounded public repository demo command."""

    name: str
    demo: str
    args: tuple[str, ...]
    timeout_seconds: float
    required_stdout: tuple[str, ...]
    coverage: str

    @property
    def command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "cli.aisp",
            "demos",
            self.demo,
            "--",
            *self.args,
        ]


@dataclass(frozen=True)
class EntrypointResult:
    """Observed result from one real repository entrypoint process."""

    name: str
    demo: str
    coverage: str
    command: list[str]
    required_stdout: list[str]
    status: str
    duration_ms: float
    timeout_seconds: float
    returncode: int | None
    stdout: str
    stderr: str
    diagnostic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntrypointSweepResult:
    """Aggregate result for the real repository entrypoint manifest."""

    discovered: int
    passed: int
    failed: int
    not_run: int
    duration_ms: float
    deadline_exceeded: bool
    results: list[EntrypointResult]

    @property
    def complete(self) -> bool:
        return self.not_run == 0 and not self.deadline_exceeded

    @property
    def succeeded(self) -> bool:
        return self.complete and self.discovered > 0 and self.failed == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovered": self.discovered,
            "passed": self.passed,
            "failed": self.failed,
            "not_run": self.not_run,
            "duration_ms": self.duration_ms,
            "deadline_exceeded": self.deadline_exceeded,
            "complete": self.complete,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass
class ImportGroupCounts:
    """Exact import counts for one chapter or lab directory."""

    discovered: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    not_run: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ImportFailure:
    """One benchmark module that did not import successfully."""

    module: str
    path: str
    diagnostic: str
    outcome: str = IMPORT_FAILURE_EXCEPTION
    returncode: int | None = None
    signal_number: int | None = None
    timeout_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImportSweepResult:
    """Aggregate repository benchmark import result."""

    discovered: int
    passed: int
    failed: int
    skipped: int
    not_run: int
    duration_ms: float
    deadline_exceeded: bool
    groups: dict[str, ImportGroupCounts]
    failures: list[ImportFailure]
    requested: bool = True
    omission_reason: str | None = None

    @property
    def complete(self) -> bool:
        return self.requested and self.not_run == 0 and not self.deadline_exceeded

    @property
    def coverage_status(self) -> str:
        if not self.requested:
            return IMPORT_COVERAGE_NOT_REQUESTED
        if not self.complete:
            return IMPORT_COVERAGE_INCOMPLETE
        return IMPORT_COVERAGE_COMPLETE

    @property
    def succeeded(self) -> bool:
        return self.complete and self.discovered > 0 and self.failed == 0

    def to_dict(self) -> dict[str, Any]:
        counts_available = self.requested
        return {
            "discovered": self.discovered if counts_available else None,
            "passed": self.passed if counts_available else None,
            "failed": self.failed if counts_available else None,
            "skipped": self.skipped if counts_available else None,
            "not_run": self.not_run if counts_available else None,
            "duration_ms": self.duration_ms,
            "deadline_exceeded": self.deadline_exceeded,
            "complete": self.complete,
            "requested": self.requested,
            "counts_available": counts_available,
            "coverage_status": self.coverage_status,
            "omission_reason": self.omission_reason,
            "groups": {name: counts.to_dict() for name, counts in sorted(self.groups.items())},
            "failures": [failure.to_dict() for failure in self.failures],
        }


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return value


def _bounded_deadline_float(raw: str) -> float:
    value = _positive_float(raw)
    if value > DEFAULT_DEADLINE_SECONDS:
        raise argparse.ArgumentTypeError(
            f"value must not exceed {DEFAULT_DEADLINE_SECONDS:g} seconds"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded PyTorch 2.14.1 compatibility smoke on a B200 GPU."
    )
    parser.add_argument(
        "--mode",
        choices=SMOKE_MODES,
        default=SMOKE_MODE_FULL,
        help=(
            "Validation mode. 'full' runs all stack checks, the GPU entrypoints, "
            "then the repository import sweep. 'short-gpu' skips repository imports "
            "and runs the stack checks except the optional vLLM ABI import, then runs "
            "the GPU entrypoints. Default: full."
        ),
    )
    parser.add_argument(
        "--deadline-seconds",
        type=_bounded_deadline_float,
        default=DEFAULT_DEADLINE_SECONDS,
        help="Overall wall-clock deadline, capped at 240 seconds. Default: 240 seconds.",
    )
    parser.add_argument(
        "--per-import-timeout-seconds",
        type=_positive_float,
        default=DEFAULT_IMPORT_TIMEOUT_SECONDS,
        help="Maximum wall time for one repository module import. Default: 10 seconds.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. Default: text.",
    )
    return parser


def _exception_diagnostic(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


@contextmanager
def _wall_timeout(seconds: float, action: str) -> Iterator[None]:
    """Interrupt Python-level work after ``seconds`` on the main thread."""

    if seconds <= 0:
        raise SmokeDeadlineExceededError(f"deadline reached before {action}")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("bounded smoke checks must run on the main thread")
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise RuntimeError("bounded smoke checks require SIGALRM and setitimer")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise SmokeDeadlineExceededError(f"timed out after {seconds:.3f}s while {action}")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            elapsed = time.monotonic() - started
            remaining = max(1e-6, previous_timer[0] - elapsed)
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def _require_distribution_version(distribution: str) -> str:
    expected = EXPECTED_VERSIONS[distribution]
    actual = metadata.version(distribution)
    if actual != expected:
        raise RuntimeError(
            f"{distribution} version mismatch: expected {expected!r}, got {actual!r}"
        )
    return actual


def _module_version(module: Any) -> str | None:
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else None


def _triton_vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    element_count,
    block_size: tl.constexpr,
):
    """One-block vector add used only by the real Triton smoke check."""

    offsets = tl.arange(0, block_size)
    mask = offsets < element_count
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def _check_torch_cuda() -> dict[str, Any]:
    torch_version = _require_distribution_version("torch")
    nccl_distribution_version = _require_distribution_version("nvidia-nccl-cu13")
    torch = import_module("torch")
    distributed = import_module("torch.distributed")
    module_version = str(torch.__version__)
    if module_version != torch_version:
        raise RuntimeError(
            f"torch module version mismatch: metadata={torch_version!r}, module={module_version!r}"
        )
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"torch CUDA mismatch: expected '13.0', got {torch.version.cuda!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned False")

    device_count = int(torch.cuda.device_count())
    if device_count < 1:
        raise RuntimeError("torch reported no visible CUDA devices")
    device_index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device_index)
    capability = (int(properties.major), int(properties.minor))
    if capability != (10, 0):
        raise RuntimeError(
            f"B200 requires compute capability (10, 0), got {capability} on {properties.name!r}"
        )
    if "B200" not in properties.name.upper():
        raise RuntimeError(f"expected a B200 device name, got {properties.name!r}")
    if not distributed.is_nccl_available():
        raise RuntimeError("torch.distributed reports that NCCL is unavailable")
    nccl_runtime_version = tuple(int(value) for value in torch.cuda.nccl.version())
    if nccl_runtime_version != (2, 30, 7):
        raise RuntimeError(
            f"NCCL runtime mismatch: expected (2, 30, 7), got {nccl_runtime_version!r}"
        )

    lhs = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
    rhs = torch.eye(4, device="cuda", dtype=torch.float32)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    output = torch.relu(lhs @ rhs + 1.0)
    end_event.record()
    torch.cuda.synchronize()
    expected = lhs + 1.0
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)

    return {
        "torch": torch_version,
        "torch_cuda": torch.version.cuda,
        "nccl_distribution": nccl_distribution_version,
        "nccl_runtime": ".".join(str(value) for value in nccl_runtime_version),
        "cudnn": int(torch.backends.cudnn.version())
        if torch.backends.cudnn.is_available()
        else None,
        "device_count": device_count,
        "selected_device": device_index,
        "gpu_name": properties.name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_memory_bytes": int(properties.total_memory),
        "torch_arch_list": list(torch.cuda.get_arch_list()),
        "operation": "4x4 CUDA float32 identity matmul plus ReLU",
        "correctness": "torch.testing.assert_close with rtol=0 and atol=0",
        "operation_gpu_event_ms": float(start_event.elapsed_time(end_event)),
    }


def _check_triton() -> dict[str, Any]:
    global tl

    triton_version = _require_distribution_version("triton")
    triton = import_module("triton")
    module_version = _module_version(triton)
    if module_version != triton_version:
        raise RuntimeError(
            f"triton module version mismatch: metadata={triton_version!r}, module={module_version!r}"
        )
    tl = import_module("triton.language")
    torch = import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("Triton execution requires an available CUDA device")

    element_count = 128
    block_size = 128
    x = torch.arange(element_count, device="cuda", dtype=torch.float32)
    y = torch.full((element_count,), 2.0, device="cuda", dtype=torch.float32)
    output = torch.empty_like(x)
    kernel = triton.jit(_triton_vector_add_kernel)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    kernel[(1,)](x, y, output, element_count, block_size=block_size, num_warps=1)
    end_event.record()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, x + y, rtol=0.0, atol=0.0)

    return {
        "triton": triton_version,
        "operation": "single 128-element CUDA vector-add kernel",
        "correctness": "torch.testing.assert_close with rtol=0 and atol=0",
        "operation_gpu_event_ms": float(start_event.elapsed_time(end_event)),
        "grid": [1],
        "block_size": block_size,
    }


def _check_flashinfer_block_sparse() -> dict[str, Any]:
    """Plan and run one tiny real FlashInfer block-sparse attention workload."""

    flashinfer_version = _require_distribution_version("flashinfer-python")
    flashinfer = import_module("flashinfer")
    torch = import_module("torch")
    functional = import_module("torch.nn.functional")
    module_version = _module_version(flashinfer)
    if module_version is not None and module_version != flashinfer_version:
        raise RuntimeError(
            "flashinfer module version mismatch: "
            f"metadata={flashinfer_version!r}, module={module_version!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("FlashInfer execution requires an available CUDA device")

    sequence_length = 32
    block_size = 16
    heads = 2
    head_dimension = 64
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    q = torch.randn(
        sequence_length,
        heads,
        head_dimension,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    k = torch.randn(
        sequence_length,
        heads,
        head_dimension,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    v = torch.randn(
        sequence_length,
        heads,
        head_dimension,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    dense_mask = torch.zeros((sequence_length, sequence_length), dtype=torch.bool, device=device)
    dense_mask[:block_size, :block_size] = True
    dense_mask[block_size:, block_size:] = True
    workspace_bytes = 128 * 1024 * 1024
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BlockSparseAttentionWrapper(workspace)
    scale = 1.0 / math.sqrt(head_dimension)
    wrapper.plan(
        indptr,
        indices,
        sequence_length,
        sequence_length,
        block_size,
        block_size,
        heads,
        heads,
        head_dimension,
        sm_scale=scale,
    )
    with torch.inference_mode():
        output = wrapper.run(q, k, v)
        reference = (
            functional.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                attn_mask=dense_mask,
                is_causal=False,
                scale=scale,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
    torch.cuda.synchronize(device)
    if tuple(output.shape) != tuple(reference.shape):
        raise RuntimeError(
            f"FlashInfer output shape mismatch: {tuple(output.shape)!r} versus "
            f"{tuple(reference.shape)!r}"
        )
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("FlashInfer block-sparse attention produced non-finite output")
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
    max_absolute_error = float((output.float() - reference.float()).abs().max().item())

    return {
        "flashinfer": flashinfer_version,
        "operation": "FlashInfer BlockSparseAttentionWrapper plan and run",
        "shape": [sequence_length, heads, head_dimension],
        "block_size": block_size,
        "nonzero_blocks": int(indices.numel()),
        "block_sparsity_ratio": 0.5,
        "workspace_bytes": workspace_bytes,
        "finite_output": True,
        "correctness": "dense torch SDPA comparison with rtol=0.02 and atol=0.02",
        "max_absolute_error": max_absolute_error,
    }


def _check_nccl_dtensor() -> dict[str, Any]:
    """Run a one-rank NCCL collective and the chapter 13 DTensor benchmark path."""

    torch = import_module("torch")
    distributed = import_module("torch.distributed")
    dtensor_module = import_module("ch13.dtensor_mesh_tool")
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL and DTensor execution require an available CUDA device")
    if not distributed.is_nccl_available():
        raise RuntimeError("torch.distributed reports that NCCL is unavailable")
    if os.environ.get("RANK") != "0" or os.environ.get("WORLD_SIZE") != "1":
        raise RuntimeError("NCCL and DTensor smoke requires a one-rank torchrun worker")

    benchmark = dtensor_module.DTensorMeshBenchmark()
    try:
        benchmark.setup()
        if distributed.get_world_size() != 1 or distributed.get_rank() != 0:
            raise RuntimeError("initialized process group is not the required one-rank group")
        backend = str(distributed.get_backend()).lower()
        if backend != "nccl":
            raise RuntimeError(f"expected NCCL process group, got {backend!r}")

        collective = torch.tensor([7.0], dtype=torch.float32, device=benchmark.device)
        work = distributed.all_reduce(
            collective,
            op=distributed.ReduceOp.SUM,
            async_op=True,
        )
        wait = getattr(work, "wait", None)
        if not callable(wait):
            raise RuntimeError("asynchronous NCCL all_reduce did not return a waitable Work")
        wait()
        torch.cuda.synchronize(benchmark.device)
        torch.testing.assert_close(
            collective,
            torch.tensor([7.0], dtype=torch.float32, device=benchmark.device),
            rtol=0.0,
            atol=0.0,
        )

        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        input_local = benchmark.get_verify_inputs()["input"]
        output_local = benchmark.get_verify_output()
        if not bool(torch.isfinite(output_local).all().item()):
            raise RuntimeError("chapter 13 DTensor benchmark produced non-finite output")
        torch.testing.assert_close(output_local, input_local * 2, rtol=0.0, atol=0.0)
        torch.cuda.synchronize(benchmark.device)
        output_shape = list(output_local.shape)
    finally:
        benchmark.teardown()

    return {
        "backend": "nccl",
        "world_size": 1,
        "collective": "one-element asynchronous all_reduce SUM followed by Work.wait",
        "collective_correctness": "exact identity for one rank",
        "dtensor_path": "ch13.dtensor_mesh_tool.DTensorMeshBenchmark",
        "dtensor_operation": "Replicate DTensor multiply and redistribute",
        "dtensor_output_shape": output_shape,
        "dtensor_finite_output": True,
        "dtensor_correctness": "exact local output equals input multiplied by two",
    }


def _check_import(
    distribution: str,
    module_names: Sequence[str],
) -> dict[str, Any]:
    version = _require_distribution_version(distribution)
    imported = []
    for module_name in module_names:
        module = import_module(module_name)
        imported.append(
            {
                "module": module_name,
                "module_version": _module_version(module),
            }
        )
    return {"distributions": {distribution: version}, "imports": imported}


def _check_cuda_python() -> dict[str, Any]:
    version = _require_distribution_version("cuda-python")
    runtime = import_module("cuda.bindings.runtime")

    def checked_call(name: str) -> tuple[Any, ...]:
        result = getattr(runtime, name)()
        if not isinstance(result, tuple) or not result:
            raise RuntimeError(f"{name} returned an unexpected result: {result!r}")
        status = result[0]
        if status != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{name} failed with {status!r}")
        return result[1:]

    (device_count,) = checked_call("cudaGetDeviceCount")
    (current_device,) = checked_call("cudaGetDevice")
    (driver_version,) = checked_call("cudaDriverGetVersion")
    (runtime_version,) = checked_call("cudaRuntimeGetVersion")
    if int(device_count) < 1:
        raise RuntimeError("CUDA Python reported no visible devices")

    return {
        "distributions": {"cuda-python": version},
        "imports": [{"module": "cuda.bindings.runtime", "module_version": None}],
        "device_count": int(device_count),
        "selected_device": int(current_device),
        "driver_version_raw": int(driver_version),
        "runtime_version_raw": int(runtime_version),
    }


def _check_transformer_engine() -> dict[str, Any]:
    versions = {
        "transformer-engine": _require_distribution_version("transformer-engine"),
        "transformer-engine-cu13": _require_distribution_version("transformer-engine-cu13"),
        "transformer-engine-torch": _require_distribution_version("transformer-engine-torch"),
    }
    modules = (
        "transformer_engine",
        "transformer_engine.pytorch",
        "transformer_engine_torch",
    )
    imported = []
    for module_name in modules:
        module = import_module(module_name)
        imported.append({"module": module_name, "module_version": _module_version(module)})
    return {"distributions": versions, "imports": imported}


def _check_torchtitan() -> dict[str, Any]:
    version = _require_distribution_version("torchtitan")
    module = import_module("torchtitan")
    config = import_module("torchtitan.config")
    compile_config = config.CompileConfig()
    if not hasattr(compile_config, "enable_async_tensor_parallel"):
        raise RuntimeError("TorchTitan CompileConfig lacks enable_async_tensor_parallel")
    if not hasattr(config, "ParallelismConfig"):
        raise RuntimeError("TorchTitan config lacks ParallelismConfig")
    return {
        "distributions": {"torchtitan": version},
        "imports": [
            {"module": "torchtitan", "module_version": _module_version(module)},
            {"module": "torchtitan.config", "module_version": _module_version(config)},
        ],
        "compile_config": type(compile_config).__qualname__,
        "parallelism_config": config.ParallelismConfig.__qualname__,
    }


def _check_vllm() -> dict[str, Any]:
    version = _require_distribution_version("vllm")
    modules = (
        "vllm",
        "vllm._C_stable_libtorch",
        "vllm._moe_C_stable_libtorch",
    )
    imported = []
    for module_name in modules:
        module = import_module(module_name)
        imported.append({"module": module_name, "module_version": _module_version(module)})
    return {"distributions": {"vllm": version}, "imports": imported}


def _check_repository_identity(code_root: Path) -> dict[str, Any]:
    revision_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    revision = revision_result.stdout.strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RuntimeError(f"git rev-parse returned an invalid revision: {revision!r}")
    tree_result = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=code_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    tree = tree_result.stdout.strip()
    status_result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=code_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if status_result.stdout.strip():
        raise RuntimeError("repository files differ from the recorded Git revision")
    return {"git_revision": revision, "git_tree": tree}


def _check_runtime_platform() -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine().lower()
    implementation = platform.python_implementation()
    python_major_minor = tuple(sys.version_info[:2])
    failures = []
    if system != "Linux":
        failures.append(f"operating system is {system!r}")
    if machine not in {"x86_64", "amd64"}:
        failures.append(f"machine architecture is {machine!r}")
    if implementation != "CPython":
        failures.append(f"Python implementation is {implementation!r}")
    if python_major_minor != (3, 12):
        failures.append(f"Python version is {platform.python_version()!r}")
    if failures:
        raise RuntimeError(
            "the pinned B200 stack requires Linux x86-64 CPython 3.12: " + ", ".join(failures)
        )
    return {
        "system": system,
        "machine": machine,
        "python_implementation": implementation,
        "python": platform.python_version(),
    }


def load_repository_entrypoints(code_root: Path) -> list[EntrypointSpec]:
    """Load and validate the bounded real-entrypoint manifest."""

    manifest_path = code_root / ENTRYPOINT_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(f"repository entrypoint manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"repository entrypoint manifest is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("repository entrypoint manifest must be a JSON object")
    if payload.get("schema_version") != ENTRYPOINT_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            "repository entrypoint manifest schema mismatch: "
            f"expected {ENTRYPOINT_MANIFEST_SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}"
        )
    raw_entrypoints = payload.get("entrypoints")
    if not isinstance(raw_entrypoints, list) or not raw_entrypoints:
        raise RuntimeError("repository entrypoint manifest must declare entrypoints")

    required_keys = {
        "name",
        "demo",
        "args",
        "timeout_seconds",
        "required_stdout",
        "coverage",
    }
    specs: list[EntrypointSpec] = []
    seen_names: set[str] = set()
    for index, raw_spec in enumerate(raw_entrypoints):
        if not isinstance(raw_spec, dict) or set(raw_spec) != required_keys:
            raise RuntimeError(f"entrypoint {index} must contain exactly {sorted(required_keys)!r}")
        name = raw_spec["name"]
        demo = raw_spec["demo"]
        args = raw_spec["args"]
        timeout_seconds = raw_spec["timeout_seconds"]
        required_stdout = raw_spec["required_stdout"]
        coverage = raw_spec["coverage"]
        if not isinstance(name, str) or re.fullmatch(r"[a-z0-9_]+", name) is None:
            raise RuntimeError(f"entrypoint {index} has an invalid name: {name!r}")
        if name in seen_names:
            raise RuntimeError(f"duplicate repository entrypoint name: {name!r}")
        seen_names.add(name)
        if not isinstance(demo, str) or re.fullmatch(r"[a-z0-9-]+", demo) is None:
            raise RuntimeError(f"entrypoint {name!r} has an invalid demo name: {demo!r}")
        if not isinstance(args, list) or not all(
            isinstance(value, str) and value and "\0" not in value for value in args
        ):
            raise RuntimeError(f"entrypoint {name!r} args must contain only nonempty strings")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int | float)
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
        ):
            raise RuntimeError(f"entrypoint {name!r} timeout_seconds must be greater than zero")
        if (
            not isinstance(required_stdout, list)
            or not required_stdout
            or not all(isinstance(value, str) and value for value in required_stdout)
        ):
            raise RuntimeError(f"entrypoint {name!r} required_stdout must contain nonempty strings")
        if not isinstance(coverage, str) or not coverage.strip():
            raise RuntimeError(f"entrypoint {name!r} coverage must be nonempty")
        specs.append(
            EntrypointSpec(
                name=name,
                demo=demo,
                args=tuple(args),
                timeout_seconds=float(timeout_seconds),
                required_stdout=tuple(required_stdout),
                coverage=coverage.strip(),
            )
        )
    return specs


def _bounded_output(value: str) -> str:
    if len(value) <= MAX_CAPTURED_OUTPUT_CHARS:
        return value
    half = MAX_CAPTURED_OUTPUT_CHARS // 2
    omitted = len(value) - (half * 2)
    return value[:half] + f"\n... {omitted} output characters omitted ...\n" + value[-half:]


def _register_child_process(process: subprocess.Popen[Any]) -> None:
    """Track an isolated child process group until it has been reaped."""

    _PROCESS_GROUPS.register(process)


def _unregister_child_process(process: subprocess.Popen[Any]) -> None:
    _PROCESS_GROUPS.unregister(process)


def _kill_active_child_process_groups() -> None:
    """Kill and reap every tracked child group during abnormal parent exit."""

    _PROCESS_GROUPS.kill_all()


def _cleanup_children_on_parent_exit():
    """Ensure outer timeout or cancellation cannot orphan GPU workers."""

    return _PROCESS_GROUPS.cleanup_on_parent_exit()


def _terminate_process_group(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Terminate a bounded entrypoint and any child processes it started."""

    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        return process.communicate(timeout=PROCESS_GROUP_TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        return process.communicate(timeout=PROCESS_GROUP_KILL_WAIT_SECONDS)


def _gpu_check_worker_main(check_name: str, code_root: Path) -> int:
    """Execute one GPU check and emit one prefixed JSON report."""

    root = code_root.resolve()
    root_text = str(root)
    if not sys.path or sys.path[0] != root_text:
        sys.path.insert(0, root_text)
    os.chdir(root)
    invalidate_caches()
    checks: dict[str, Callable[[], dict[str, Any]]] = {
        "flashinfer_block_sparse": _check_flashinfer_block_sparse,
        "nccl_dtensor": _check_nccl_dtensor,
        "transformer_engine_imports": _check_transformer_engine,
    }
    try:
        check = checks[check_name]
    except KeyError:
        report = {
            "status": STATUS_FAIL,
            "diagnostic": f"unknown GPU check: {check_name!r}",
        }
    else:
        try:
            with redirect_stdout(sys.stderr):
                details = check()
        except BaseException as exc:
            report = {
                "status": STATUS_FAIL,
                "diagnostic": _exception_diagnostic(exc),
            }
        else:
            report = {"status": STATUS_PASS, "details": details}
    print(_GPU_CHECK_REPORT_PREFIX + json.dumps(report, sort_keys=True), flush=True)
    return 0


def _gpu_check_command(check_name: str, code_root: Path) -> list[str]:
    worker_args = [
        _GPU_CHECK_WORKER_FLAG,
        check_name,
        str(code_root.resolve()),
    ]
    if check_name == "nccl_dtensor":
        return [
            sys.executable,
            "-u",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=1",
            "--max-restarts=0",
            "--module",
            "core.scripts.b200_torch2141_smoke",
            *worker_args,
        ]
    return [
        sys.executable,
        "-u",
        "-m",
        "core.scripts.b200_torch2141_smoke",
        *worker_args,
    ]


def _parse_gpu_check_report(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if not line.startswith(_GPU_CHECK_REPORT_PREFIX):
            continue
        payload = line.removeprefix(_GPU_CHECK_REPORT_PREFIX)
        try:
            report = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid GPU check worker report: {_exception_diagnostic(exc)}"
            ) from exc
        if not isinstance(report, dict):
            raise RuntimeError(f"invalid GPU check worker report: {report!r}")
        return report
    raise RuntimeError("GPU check worker did not emit a result report")


def _run_isolated_gpu_check(
    check_name: str,
    *,
    code_root: Path,
    deadline: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run a native GPU check in a killable process group."""

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SmokeDeadlineExceededError(f"deadline reached before {check_name}")
    effective_timeout = min(timeout_seconds, remaining)
    command = _gpu_check_command(check_name, code_root)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    module_root = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        module_root if not existing_pythonpath else module_root + os.pathsep + existing_pythonpath
    )
    process = _PROCESS_GROUPS.spawn(
        command,
        cwd=code_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        try:
            stdout, stderr = process.communicate(timeout=effective_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout, stderr = _terminate_process_group(process)
    except BaseException:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        with suppress(subprocess.TimeoutExpired):
            process.communicate(timeout=PROCESS_GROUP_KILL_WAIT_SECONDS)
        raise
    finally:
        if process.poll() is not None:
            _unregister_child_process(process)

    if timed_out:
        raise SmokeDeadlineExceededError(f"{check_name} timed out after {effective_timeout:.3f}s")
    try:
        report = _parse_gpu_check_report(stdout)
    except RuntimeError as exc:
        diagnostic_stderr = _bounded_output(stderr.strip())
        if process.returncode != 0:
            raise RuntimeError(
                f"{check_name} worker exited with status {process.returncode}: "
                f"{diagnostic_stderr or exc}"
            ) from exc
        raise
    if process.returncode != 0:
        raise RuntimeError(
            f"{check_name} worker exited with status {process.returncode}: "
            f"{_bounded_output(stderr.strip()) or 'no worker diagnostic'}"
        )
    diagnostic = report.get("diagnostic")
    if report.get("status") == STATUS_FAIL and isinstance(diagnostic, str):
        raise RuntimeError(diagnostic.removeprefix("RuntimeError: "))
    details = report.get("details")
    if report.get("status") != STATUS_PASS or not isinstance(details, dict):
        raise RuntimeError(f"invalid GPU check worker report: {report!r}")
    details["worker_timeout_seconds"] = effective_timeout
    return details


def _run_repository_entrypoint(
    spec: EntrypointSpec,
    *,
    code_root: Path,
    timeout_seconds: float,
) -> EntrypointResult:
    started = time.monotonic()
    command = spec.command
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        process = _PROCESS_GROUPS.spawn(
            command,
            cwd=code_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return EntrypointResult(
            name=spec.name,
            demo=spec.demo,
            coverage=spec.coverage,
            command=command,
            required_stdout=list(spec.required_stdout),
            status=STATUS_FAIL,
            duration_ms=(time.monotonic() - started) * 1000.0,
            timeout_seconds=timeout_seconds,
            returncode=None,
            stdout="",
            stderr="",
            diagnostic=_exception_diagnostic(exc),
        )

    timed_out = False
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout, stderr = _terminate_process_group(process)
    finally:
        _unregister_child_process(process)

    diagnostic = None
    if timed_out:
        diagnostic = f"timed out after {timeout_seconds:.3f}s"
    elif process.returncode != 0:
        diagnostic = f"entrypoint exited with status {process.returncode}"
    else:
        missing = [value for value in spec.required_stdout if value not in stdout]
        if missing:
            diagnostic = f"stdout lacked required evidence: {missing!r}"

    return EntrypointResult(
        name=spec.name,
        demo=spec.demo,
        coverage=spec.coverage,
        command=command,
        required_stdout=list(spec.required_stdout),
        status=STATUS_PASS if diagnostic is None else STATUS_FAIL,
        duration_ms=(time.monotonic() - started) * 1000.0,
        timeout_seconds=timeout_seconds,
        returncode=process.returncode,
        stdout=_bounded_output(stdout.strip()),
        stderr=_bounded_output(stderr.strip()),
        diagnostic=diagnostic,
    )


def run_repository_entrypoints(
    code_root: Path,
    *,
    deadline: float,
) -> EntrypointSweepResult:
    """Run every required manifest entrypoint under per-command and global bounds."""

    started = time.monotonic()
    specs = load_repository_entrypoints(code_root)
    results: list[EntrypointResult] = []
    passed = 0
    failed = 0
    not_run = 0
    deadline_exceeded = False
    for index, spec in enumerate(specs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            deadline_exceeded = True
            not_run = len(specs) - index
            break
        timeout = min(spec.timeout_seconds, remaining)
        deadline_limited = timeout < spec.timeout_seconds
        result = _run_repository_entrypoint(
            spec,
            code_root=code_root,
            timeout_seconds=timeout,
        )
        results.append(result)
        if result.status == STATUS_PASS:
            passed += 1
        else:
            failed += 1
        if deadline_limited and time.monotonic() >= deadline:
            deadline_exceeded = True
            not_run = len(specs) - index - 1
            break

    return EntrypointSweepResult(
        discovered=len(specs),
        passed=passed,
        failed=failed,
        not_run=not_run,
        duration_ms=(time.monotonic() - started) * 1000.0,
        deadline_exceeded=deadline_exceeded,
        results=results,
    )


def _run_check(
    name: str,
    policy: str,
    check: Callable[[], dict[str, Any]],
    deadline: float,
) -> CheckResult:
    started = time.monotonic()
    remaining = deadline - started
    try:
        with _wall_timeout(remaining, name):
            details = check()
    except (Exception, SystemExit) as exc:
        return CheckResult(
            name=name,
            policy=policy,
            status=STATUS_FAIL,
            duration_ms=(time.monotonic() - started) * 1000.0,
            diagnostic=_exception_diagnostic(exc),
        )
    return CheckResult(
        name=name,
        policy=policy,
        status=STATUS_PASS,
        duration_ms=(time.monotonic() - started) * 1000.0,
        details=details,
    )


def discover_repository_examples(code_root: Path) -> list[Path]:
    """Find baseline/optimized modules and checked compatibility-import roots."""

    patterns = (
        "ch*/**/baseline_*.py",
        "ch*/**/optimized_*.py",
        "labs/**/baseline_*.py",
        "labs/**/optimized_*.py",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in code_root.glob(pattern) if path.is_file())

    manifest_path = code_root / COMPATIBILITY_IMPORT_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(f"compatibility import manifest is missing: {manifest_path}")
    seen_manifest_entries: set[Path] = set()
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        relative = Path(line)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix != ".py"
            or not relative.parts
            or not (relative.parts[0].startswith("ch") or relative.parts[0] == "labs")
        ):
            raise RuntimeError(
                f"invalid compatibility import entry on line {line_number}: {line!r}"
            )
        if relative in seen_manifest_entries:
            raise RuntimeError(f"duplicate compatibility import entry: {line!r}")
        seen_manifest_entries.add(relative)
        path = code_root / relative
        if not path.is_file():
            raise RuntimeError(f"compatibility import path is missing: {line!r}")
        paths.add(path)
    return sorted(paths)


def repository_group(path: Path, code_root: Path) -> str:
    relative = path.relative_to(code_root)
    if relative.parts[0] == "labs":
        if len(relative.parts) < 3:
            raise ValueError(f"lab benchmark path lacks a lab directory: {relative}")
        return f"labs/{relative.parts[1]}"
    return relative.parts[0]


def repository_module_name(path: Path, code_root: Path) -> str:
    relative = path.relative_to(code_root).with_suffix("")
    return ".".join(relative.parts)


@dataclass
class _ImportWorker:
    """One isolated process that imports repository modules on request."""

    process: subprocess.Popen[bytes]
    commands: BinaryIO
    reports: BinaryIO


def _repository_import_worker_main(
    command_fd: int,
    report_fd: int,
    code_root: Path,
) -> int:
    """Import requested modules and report Python exceptions out of process."""

    root = code_root.resolve()
    root_text = str(root)
    if not sys.path or sys.path[0] != root_text:
        sys.path.insert(0, root_text)
    os.chdir(root)
    invalidate_caches()

    with (
        os.fdopen(command_fd, "r", encoding="utf-8") as commands,
        os.fdopen(report_fd, "w", encoding="utf-8", buffering=1) as reports,
    ):
        for raw_command in commands:
            command = json.loads(raw_command)
            module_name = command["module"]
            relative_path = command["path"]
            os.chdir(root)
            sys.argv = [str(root / relative_path)]
            try:
                import_module(module_name)
            except BaseException as exc:
                report = {
                    "status": STATUS_FAIL,
                    "diagnostic": _exception_diagnostic(exc),
                }
                sys.modules.pop(module_name, None)
            else:
                report = {"status": STATUS_PASS}
            reports.write(json.dumps(report, sort_keys=True) + "\n")
    return 0


def _start_import_worker(code_root: Path) -> _ImportWorker:
    command_read_fd, command_write_fd = os.pipe()
    report_read_fd, report_write_fd = os.pipe()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    module_root = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        module_root if not existing_pythonpath else module_root + os.pathsep + existing_pythonpath
    )
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        _IMPORT_WORKER_FLAG,
        str(command_read_fd),
        str(report_write_fd),
        str(code_root.resolve()),
    ]
    try:
        process = _PROCESS_GROUPS.spawn(
            command,
            cwd=code_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=(command_read_fd, report_write_fd),
        )
    except BaseException:
        os.close(command_read_fd)
        os.close(command_write_fd)
        os.close(report_read_fd)
        os.close(report_write_fd)
        raise

    os.close(command_read_fd)
    os.close(report_write_fd)
    return _ImportWorker(
        process=process,
        commands=os.fdopen(command_write_fd, "wb", buffering=0),
        reports=os.fdopen(report_read_fd, "rb", buffering=0),
    )


def _stop_import_worker(worker: _ImportWorker, *, immediate: bool = False) -> None:
    """Stop the worker and every process in its isolated process group."""

    try:
        worker.commands.close()
        initial_signal = signal.SIGKILL if immediate else signal.SIGTERM
        with suppress(ProcessLookupError):
            os.killpg(worker.process.pid, initial_signal)
        try:
            worker.process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(worker.process.pid, signal.SIGKILL)
            worker.process.wait(timeout=1.0)
        else:
            # The worker may have left descendants in its process group.
            with suppress(ProcessLookupError):
                os.killpg(worker.process.pid, signal.SIGKILL)
        worker.reports.close()
    finally:
        _unregister_child_process(worker.process)


def _signal_diagnostic(returncode: int) -> tuple[int, str]:
    signal_number = -returncode
    try:
        signal_name = signal.Signals(signal_number).name
    except ValueError:
        signal_name = "UNKNOWN"
    return (
        signal_number,
        f"import worker terminated by signal {signal_name} ({signal_number})",
    )


def _worker_exit_failure(
    worker: _ImportWorker,
    *,
    module_name: str,
    relative_path: str,
) -> ImportFailure:
    returncode = worker.process.poll()
    if returncode is None:
        try:
            returncode = worker.process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            _stop_import_worker(worker, immediate=True)
            return ImportFailure(
                module=module_name,
                path=relative_path,
                diagnostic="import worker closed its report channel without exiting",
                outcome=IMPORT_FAILURE_PROTOCOL,
                returncode=worker.process.returncode,
            )
    if returncode < 0:
        signal_number, diagnostic = _signal_diagnostic(returncode)
        return ImportFailure(
            module=module_name,
            path=relative_path,
            diagnostic=diagnostic,
            outcome=IMPORT_FAILURE_SIGNAL,
            returncode=returncode,
            signal_number=signal_number,
        )
    return ImportFailure(
        module=module_name,
        path=relative_path,
        diagnostic=f"import worker exited with status {returncode}",
        outcome=IMPORT_FAILURE_EXIT,
        returncode=returncode,
    )


def _run_worker_import(
    worker: _ImportWorker,
    *,
    module_name: str,
    relative_path: str,
    timeout_seconds: float,
) -> ImportFailure | None:
    command = (
        json.dumps({"module": module_name, "path": relative_path}, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    try:
        worker.commands.write(command)
    except BrokenPipeError:
        return _worker_exit_failure(
            worker,
            module_name=module_name,
            relative_path=relative_path,
        )

    deadline = time.monotonic() + timeout_seconds
    while True:
        ready, _, _ = select.select([worker.reports], [], [], 0.0)
        if ready:
            raw_report = worker.reports.readline()
            if not raw_report:
                return _worker_exit_failure(
                    worker,
                    module_name=module_name,
                    relative_path=relative_path,
                )
            try:
                report = json.loads(raw_report)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                return ImportFailure(
                    module=module_name,
                    path=relative_path,
                    diagnostic=f"invalid import worker report: {_exception_diagnostic(exc)}",
                    outcome=IMPORT_FAILURE_PROTOCOL,
                    returncode=worker.process.poll(),
                )
            if report.get("status") == STATUS_PASS:
                return None
            diagnostic = report.get("diagnostic")
            if report.get("status") == STATUS_FAIL and isinstance(diagnostic, str):
                return ImportFailure(
                    module=module_name,
                    path=relative_path,
                    diagnostic=diagnostic,
                )
            return ImportFailure(
                module=module_name,
                path=relative_path,
                diagnostic=f"invalid import worker report: {report!r}",
                outcome=IMPORT_FAILURE_PROTOCOL,
                returncode=worker.process.poll(),
            )

        if worker.process.poll() is not None:
            return _worker_exit_failure(
                worker,
                module_name=module_name,
                relative_path=relative_path,
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _stop_import_worker(worker, immediate=True)
            return ImportFailure(
                module=module_name,
                path=relative_path,
                diagnostic=f"timed out after {timeout_seconds:.3f}s",
                outcome=IMPORT_FAILURE_TIMEOUT,
                returncode=worker.process.returncode,
                signal_number=(
                    -worker.process.returncode
                    if worker.process.returncode is not None and worker.process.returncode < 0
                    else None
                ),
                timeout_seconds=timeout_seconds,
            )
        select.select(
            [worker.reports],
            [],
            [],
            min(remaining, _IMPORT_WORKER_POLL_SECONDS),
        )


def import_repository_examples(
    code_root: Path,
    *,
    deadline: float,
    per_import_timeout_seconds: float,
) -> ImportSweepResult:
    """Import examples in a bounded child process and aggregate hard failures."""

    started = time.monotonic()
    root = code_root.resolve()
    paths = discover_repository_examples(root)
    groups: dict[str, ImportGroupCounts] = {}
    for path in paths:
        group = repository_group(path, root)
        groups.setdefault(group, ImportGroupCounts()).discovered += 1

    failures: list[ImportFailure] = []
    passed = 0
    failed = 0
    not_run = 0
    deadline_exceeded = False
    worker: _ImportWorker | None = None

    try:
        for index, path in enumerate(paths):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                deadline_exceeded = True
                for remaining_path in paths[index:]:
                    groups[repository_group(remaining_path, root)].not_run += 1
                not_run += len(paths) - index
                break

            group = repository_group(path, root)
            module_name = repository_module_name(path, root)
            relative_path = str(path.relative_to(root))
            if worker is None:
                try:
                    worker = _start_import_worker(root)
                except OSError as exc:
                    failure = ImportFailure(
                        module=module_name,
                        path=relative_path,
                        diagnostic=_exception_diagnostic(exc),
                        outcome=IMPORT_FAILURE_LAUNCH,
                    )
                    failures.append(failure)
                    groups[group].failed += 1
                    failed += 1
                    continue

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                deadline_exceeded = True
                for remaining_path in paths[index:]:
                    groups[repository_group(remaining_path, root)].not_run += 1
                not_run += len(paths) - index
                break

            timeout = min(per_import_timeout_seconds, remaining)
            deadline_limited = timeout < per_import_timeout_seconds
            failure = _run_worker_import(
                worker,
                module_name=module_name,
                relative_path=relative_path,
                timeout_seconds=timeout,
            )
            if failure is None:
                groups[group].passed += 1
                passed += 1
            else:
                failures.append(failure)
                groups[group].failed += 1
                failed += 1
                if not worker.commands.closed:
                    if worker.process.poll() is None:
                        _stop_import_worker(worker)
                    else:
                        _stop_import_worker(worker, immediate=True)
                worker = None

            if deadline_limited and time.monotonic() >= deadline:
                deadline_exceeded = True
                for remaining_path in paths[index + 1 :]:
                    groups[repository_group(remaining_path, root)].not_run += 1
                not_run += len(paths) - index - 1
                break
    finally:
        if worker is not None:
            _stop_import_worker(worker)

    return ImportSweepResult(
        discovered=len(paths),
        passed=passed,
        failed=failed,
        skipped=0,
        not_run=not_run,
        duration_ms=(time.monotonic() - started) * 1000.0,
        deadline_exceeded=deadline_exceeded,
        groups=groups,
        failures=failures,
    )


def smoke_succeeded(
    checks: Sequence[CheckResult],
    entrypoints: EntrypointSweepResult,
    imports: ImportSweepResult,
    *,
    require_repository_imports: bool = True,
) -> bool:
    checks_succeeded = all(
        (check.policy == POLICY_REQUIRED and check.status == STATUS_PASS)
        or (check.policy == POLICY_NOT_REQUESTED and check.status == STATUS_NOT_RUN)
        for check in checks
    )
    imports_succeeded = (
        imports.succeeded
        if require_repository_imports
        else imports.coverage_status == IMPORT_COVERAGE_NOT_REQUESTED
    )
    return checks_succeeded and entrypoints.succeeded and imports_succeeded


def _not_requested_check(*, name: str, reason: str) -> CheckResult:
    """Record a mode-specific omission without presenting it as a passing check."""

    return CheckResult(
        name=name,
        policy=POLICY_NOT_REQUESTED,
        status=STATUS_NOT_RUN,
        duration_ms=0.0,
        diagnostic=reason,
    )


def _not_requested_repository_imports(*, reason: str) -> ImportSweepResult:
    """Record an intentionally omitted import sweep without claiming coverage."""

    return ImportSweepResult(
        discovered=0,
        passed=0,
        failed=0,
        skipped=0,
        not_run=0,
        duration_ms=0.0,
        deadline_exceeded=False,
        groups={},
        failures=[],
        requested=False,
        omission_reason=reason,
    )


def execute_smoke(
    code_root: Path,
    *,
    deadline_seconds: float,
    per_import_timeout_seconds: float,
    mode: str = SMOKE_MODE_FULL,
) -> dict[str, Any]:
    if mode not in SMOKE_MODES:
        raise ValueError(f"unsupported smoke mode: {mode!r}")
    if (
        not math.isfinite(deadline_seconds)
        or deadline_seconds <= 0
        or deadline_seconds > DEFAULT_DEADLINE_SECONDS
    ):
        raise ValueError(
            f"deadline_seconds must be greater than zero and at most {DEFAULT_DEADLINE_SECONDS:g}"
        )

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    cleanup_reserve_seconds = min(PROCESS_GROUP_CLEANUP_RESERVE_SECONDS, deadline_seconds)
    deadline = started + deadline_seconds - cleanup_reserve_seconds
    checks: list[CheckResult] = []

    check_specs: tuple[tuple[str, str, Callable[[], dict[str, Any]]], ...] = (
        (
            "repository_identity",
            POLICY_REQUIRED,
            lambda: _check_repository_identity(code_root),
        ),
        ("runtime_platform", POLICY_REQUIRED, _check_runtime_platform),
        ("torch_cuda_b200", POLICY_REQUIRED, _check_torch_cuda),
        ("triton_kernel", POLICY_REQUIRED, _check_triton),
        (
            "flashinfer_block_sparse",
            POLICY_REQUIRED,
            lambda: _run_isolated_gpu_check(
                "flashinfer_block_sparse",
                code_root=code_root,
                deadline=deadline,
                timeout_seconds=FLASHINFER_CHECK_TIMEOUT_SECONDS,
            ),
        ),
        (
            "nccl_dtensor",
            POLICY_REQUIRED,
            lambda: _run_isolated_gpu_check(
                "nccl_dtensor",
                code_root=code_root,
                deadline=deadline,
                timeout_seconds=NCCL_DTENSOR_CHECK_TIMEOUT_SECONDS,
            ),
        ),
        (
            "torchvision_import",
            POLICY_REQUIRED,
            lambda: _check_import("torchvision", ("torchvision",)),
        ),
        (
            "torchaudio_import",
            POLICY_REQUIRED,
            lambda: _check_import("torchaudio", ("torchaudio",)),
        ),
        (
            "torchao_import",
            POLICY_REQUIRED,
            lambda: _check_import("torchao", ("torchao",)),
        ),
        (
            "torchtitan_import",
            POLICY_REQUIRED,
            _check_torchtitan,
        ),
        (
            "torch_c_dlpack_ext_import",
            POLICY_REQUIRED,
            lambda: _check_import(
                "torch-c-dlpack-ext",
                ("torch_c_dlpack_ext", "torch_c_dlpack_ext.core"),
            ),
        ),
        ("cuda_python_runtime", POLICY_REQUIRED, _check_cuda_python),
        (
            "flashinfer_import",
            POLICY_REQUIRED,
            lambda: _check_import("flashinfer-python", ("flashinfer",)),
        ),
        (
            "transformer_engine_imports",
            POLICY_REQUIRED,
            lambda: _run_isolated_gpu_check(
                "transformer_engine_imports",
                code_root=code_root,
                deadline=deadline,
                timeout_seconds=TRANSFORMER_ENGINE_CHECK_TIMEOUT_SECONDS,
            ),
        ),
        ("vllm_abi_imports", POLICY_REQUIRED, _check_vllm),
    )
    for name, policy, check in check_specs:
        if mode == SMOKE_MODE_SHORT_GPU and name == "vllm_abi_imports":
            checks.append(
                _not_requested_check(
                    name=name,
                    reason="short-gpu mode excludes the optional vLLM ABI import",
                )
            )
            continue
        checks.append(_run_check(name, policy, check, deadline))

    repository_entrypoints = run_repository_entrypoints(code_root, deadline=deadline)
    require_repository_imports = mode == SMOKE_MODE_FULL
    if require_repository_imports:
        repository_imports = import_repository_examples(
            code_root,
            deadline=deadline,
            per_import_timeout_seconds=per_import_timeout_seconds,
        )
    else:
        repository_imports = _not_requested_repository_imports(
            reason="short-gpu mode excludes the repository import sweep"
        )
    succeeded = smoke_succeeded(
        checks,
        repository_entrypoints,
        repository_imports,
        require_repository_imports=require_repository_imports,
    )
    elapsed_ms = (time.monotonic() - started) * 1000.0

    return {
        "schema_version": 2,
        "status": STATUS_PASS if succeeded else STATUS_FAIL,
        "mode": mode,
        "started_at": started_at,
        "elapsed_ms": elapsed_ms,
        "deadline_seconds": deadline_seconds,
        "cleanup_reserve_seconds": cleanup_reserve_seconds,
        "per_import_timeout_seconds": per_import_timeout_seconds,
        "environment": {
            "code_root": str(code_root.resolve()),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "git_revision": checks[0].details.get("git_revision"),
            "git_tree": checks[0].details.get("git_tree"),
        },
        "checks": [check.to_dict() for check in checks],
        "repository_entrypoints": repository_entrypoints.to_dict(),
        "repository_imports": repository_imports.to_dict(),
        "summary": {
            "required_failures": sum(
                check.status != STATUS_PASS and check.policy == POLICY_REQUIRED for check in checks
            ),
            "checks_not_requested": [
                check.name for check in checks if check.policy == POLICY_NOT_REQUESTED
            ],
            "repository_entrypoint_failures": repository_entrypoints.failed,
            "repository_entrypoints_not_run": repository_entrypoints.not_run,
            "repository_import_coverage": repository_imports.coverage_status,
            "repository_import_failures": (
                repository_imports.failed if repository_imports.requested else None
            ),
            "repository_imports_not_run": (
                repository_imports.not_run if repository_imports.requested else None
            ),
        },
    }


def render_text(report: dict[str, Any]) -> str:
    lines = [
        "B200 PyTorch 2.14.1 compatibility smoke",
        (
            f"environment python={report['environment']['python']} "
            f"platform={report['environment']['platform']} "
            f"mode={report.get('mode', SMOKE_MODE_FULL)}"
        ),
        (
            f"source git_revision={report['environment']['git_revision']} "
            f"git_tree={report['environment']['git_tree']} "
            f"started_at={report['started_at']}"
        ),
    ]
    for check in report["checks"]:
        line = (
            f"CHECK {check['status'].upper()} policy={check['policy']} "
            f"name={check['name']} duration_ms={check['duration_ms']:.3f}"
        )
        if check["details"]:
            line += f" details={json.dumps(check['details'], sort_keys=True)}"
        if check["diagnostic"]:
            line += f" diagnostic={check['diagnostic']}"
        lines.append(line)

    entrypoints = report["repository_entrypoints"]
    lines.append("REPOSITORY ENTRYPOINT RESULTS")
    for result in entrypoints["results"]:
        line = (
            f"ENTRYPOINT {result['status'].upper()} name={result['name']} "
            f"demo={result['demo']} duration_ms={result['duration_ms']:.3f} "
            f"timeout_seconds={result['timeout_seconds']:.3f} "
            f"returncode={result['returncode']} "
            f"command={json.dumps(result['command'])}"
        )
        if result["diagnostic"]:
            line += f" diagnostic={result['diagnostic']}"
        lines.append(line)
    lines.append(
        f"ENTRYPOINT_SUMMARY discovered={entrypoints['discovered']} "
        f"pass={entrypoints['passed']} fail={entrypoints['failed']} "
        f"not_run={entrypoints['not_run']} complete={entrypoints['complete']} "
        f"deadline_exceeded={entrypoints['deadline_exceeded']} "
        f"duration_ms={entrypoints['duration_ms']:.3f}"
    )

    imports = report["repository_imports"]
    lines.append("REPOSITORY IMPORT COUNTS")
    imports_requested = imports.get("requested", True)
    if not imports_requested:
        lines.append(
            f"IMPORT_COVERAGE NOT_REQUESTED reason={json.dumps(imports.get('omission_reason'))}"
        )
    for group, counts in imports["groups"].items() if imports_requested else ():
        lines.append(
            f"IMPORT_GROUP {group} discovered={counts['discovered']} "
            f"pass={counts['passed']} fail={counts['failed']} "
            f"skip={counts['skipped']} not_run={counts['not_run']}"
        )
    if imports_requested:
        for failure in imports["failures"]:
            lines.append(
                f"IMPORT_FAILURE module={failure['module']} path={failure['path']} "
                f"outcome={failure['outcome']} returncode={failure['returncode']} "
                f"signal={failure['signal_number']} "
                f"timeout_seconds={failure['timeout_seconds']} "
                f"diagnostic={failure['diagnostic']}"
            )
        lines.append(
            f"IMPORT_SUMMARY requested=true "
            f"coverage_status={imports.get('coverage_status', IMPORT_COVERAGE_COMPLETE)} "
            f"discovered={imports['discovered']} pass={imports['passed']} "
            f"fail={imports['failed']} skip={imports['skipped']} not_run={imports['not_run']} "
            f"complete={imports['complete']} deadline_exceeded={imports['deadline_exceeded']} "
            f"duration_ms={imports['duration_ms']:.3f}"
        )
    else:
        lines.append(
            "IMPORT_SUMMARY requested=false coverage_status=not_requested "
            "counts_available=false complete=false"
        )
    lines.append(
        f"RESULT {report['status'].upper()} elapsed_ms={report['elapsed_ms']:.3f} "
        f"summary={json.dumps(report['summary'], sort_keys=True)}"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == _GPU_CHECK_WORKER_FLAG:
        if len(raw_argv) != 3:
            raise RuntimeError("internal GPU check worker requires two arguments")
        return _gpu_check_worker_main(raw_argv[1], Path(raw_argv[2]))
    if raw_argv and raw_argv[0] == _IMPORT_WORKER_FLAG:
        if len(raw_argv) != 4:
            raise RuntimeError("internal import worker requires three arguments")
        return _repository_import_worker_main(
            int(raw_argv[1]),
            int(raw_argv[2]),
            Path(raw_argv[3]),
        )

    args = build_parser().parse_args(raw_argv)
    code_root = Path(__file__).resolve().parents[2]
    if args.format == "json":
        # Keep stdout machine-readable even when imported dependencies print.
        with _cleanup_children_on_parent_exit(), redirect_stdout(sys.stderr):
            report = execute_smoke(
                code_root,
                deadline_seconds=args.deadline_seconds,
                per_import_timeout_seconds=args.per_import_timeout_seconds,
                mode=args.mode,
            )
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        with _cleanup_children_on_parent_exit():
            report = execute_smoke(
                code_root,
                deadline_seconds=args.deadline_seconds,
                per_import_timeout_seconds=args.per_import_timeout_seconds,
                mode=args.mode,
            )
        print(render_text(report))
    return 0 if report["status"] == STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
