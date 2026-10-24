#!/usr/bin/env python3
"""Run a bounded B200 compatibility smoke for the PyTorch 2.14.1 stack.

This is a diagnostic runner. It performs one tiny PyTorch CUDA operation, one
tiny Triton kernel launch, import and ABI checks for the declared companion
packages, and imports every owned chapter and lab baseline/optimized module
plus the checked runnable-example manifest. It does not run benchmark methods
or produce performance evidence.

Run from the ``code`` directory:

    python -m core.scripts.b200_torch2141_smoke

The default deadline is three minutes. A missed deadline, an incomplete
repository import sweep, or any attempted check failure returns a nonzero exit
code. Every declared companion package is a required part of this pinned stack.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
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
from typing import Any

EXPECTED_VERSIONS = {
    "cuda-python": "13.0.3",
    "flashinfer-python": "0.7.2",
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

DEFAULT_DEADLINE_SECONDS = 180.0
DEFAULT_IMPORT_TIMEOUT_SECONDS = 10.0
RUNNABLE_EXAMPLE_MANIFEST = Path("core/b200_torch2141_runnable_examples.txt")

STATUS_PASS = "pass"
STATUS_FAIL = "fail"

POLICY_REQUIRED = "required"

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

    def to_dict(self) -> dict[str, str]:
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
            "skipped": self.skipped,
            "not_run": self.not_run,
            "duration_ms": self.duration_ms,
            "deadline_exceeded": self.deadline_exceeded,
            "complete": self.complete,
            "groups": {name: counts.to_dict() for name, counts in sorted(self.groups.items())},
            "failures": [failure.to_dict() for failure in self.failures],
        }


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded PyTorch 2.14.1 compatibility smoke on a B200 GPU."
    )
    parser.add_argument(
        "--deadline-seconds",
        type=_positive_float,
        default=DEFAULT_DEADLINE_SECONDS,
        help="Overall wall-clock deadline. Default: 180 seconds.",
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
    x_ptr: Any,
    y_ptr: Any,
    output_ptr: Any,
    element_count: int,
    block_size: tl.constexpr,
) -> None:
    """One-block vector add used only by the real Triton smoke check."""

    offsets = tl.arange(0, block_size)
    mask = offsets < element_count
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def _check_torch_cuda() -> dict[str, Any]:
    torch_version = _require_distribution_version("torch")
    torch = import_module("torch")
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
    """Find baseline/optimized modules and checked runnable example roots."""

    patterns = (
        "ch*/**/baseline_*.py",
        "ch*/**/optimized_*.py",
        "labs/**/baseline_*.py",
        "labs/**/optimized_*.py",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in code_root.glob(pattern) if path.is_file())

    manifest_path = code_root / RUNNABLE_EXAMPLE_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(f"runnable example manifest is missing: {manifest_path}")
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
                f"invalid runnable example manifest entry on line {line_number}: {line!r}"
            )
        if relative in seen_manifest_entries:
            raise RuntimeError(f"duplicate runnable example manifest entry: {line!r}")
        seen_manifest_entries.add(relative)
        path = code_root / relative
        if not path.is_file():
            raise RuntimeError(f"runnable example manifest path is missing: {line!r}")
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


def import_repository_examples(
    code_root: Path,
    *,
    deadline: float,
    per_import_timeout_seconds: float,
) -> ImportSweepResult:
    """Import all discovered examples and aggregate every hard failure."""

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
    root_text = str(root)
    inserted_path = not sys.path or sys.path[0] != root_text
    original_argv = sys.argv[:]
    original_cwd = Path.cwd()
    if inserted_path:
        sys.path.insert(0, root_text)
    invalidate_caches()

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
            timeout = min(per_import_timeout_seconds, remaining)
            sys.argv = [str(path)]
            os.chdir(root)
            try:
                with _wall_timeout(timeout, f"importing {module_name}"):
                    import_module(module_name)
            except (Exception, SystemExit) as exc:
                diagnostic = _exception_diagnostic(exc)
                failures.append(
                    ImportFailure(
                        module=module_name,
                        path=str(path.relative_to(root)),
                        diagnostic=diagnostic,
                    )
                )
                groups[group].failed += 1
                failed += 1
                sys.modules.pop(module_name, None)
            else:
                groups[group].passed += 1
                passed += 1
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)
        if inserted_path:
            with suppress(ValueError):
                sys.path.remove(root_text)

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


def smoke_succeeded(checks: Sequence[CheckResult], imports: ImportSweepResult) -> bool:
    return all(check.status == STATUS_PASS for check in checks) and imports.succeeded


def execute_smoke(
    code_root: Path,
    *,
    deadline_seconds: float,
    per_import_timeout_seconds: float,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    deadline = started + deadline_seconds
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
            _check_transformer_engine,
        ),
        ("vllm_abi_imports", POLICY_REQUIRED, _check_vllm),
    )
    for name, policy, check in check_specs:
        checks.append(_run_check(name, policy, check, deadline))

    repository_imports = import_repository_examples(
        code_root,
        deadline=deadline,
        per_import_timeout_seconds=per_import_timeout_seconds,
    )
    succeeded = smoke_succeeded(checks, repository_imports)
    elapsed_ms = (time.monotonic() - started) * 1000.0

    return {
        "schema_version": 1,
        "status": STATUS_PASS if succeeded else STATUS_FAIL,
        "started_at": started_at,
        "elapsed_ms": elapsed_ms,
        "deadline_seconds": deadline_seconds,
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
        "repository_imports": repository_imports.to_dict(),
        "summary": {
            "required_failures": sum(
                check.status == STATUS_FAIL and check.policy == POLICY_REQUIRED for check in checks
            ),
            "repository_import_failures": repository_imports.failed,
            "repository_imports_not_run": repository_imports.not_run,
        },
    }


def render_text(report: dict[str, Any]) -> str:
    lines = [
        "B200 PyTorch 2.14.1 compatibility smoke",
        (
            f"environment python={report['environment']['python']} "
            f"platform={report['environment']['platform']}"
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

    imports = report["repository_imports"]
    lines.append("REPOSITORY IMPORT COUNTS")
    for group, counts in imports["groups"].items():
        lines.append(
            f"IMPORT_GROUP {group} discovered={counts['discovered']} "
            f"pass={counts['passed']} fail={counts['failed']} "
            f"skip={counts['skipped']} not_run={counts['not_run']}"
        )
    for failure in imports["failures"]:
        lines.append(
            f"IMPORT_FAILURE module={failure['module']} path={failure['path']} "
            f"diagnostic={failure['diagnostic']}"
        )
    lines.append(
        f"IMPORT_SUMMARY discovered={imports['discovered']} pass={imports['passed']} "
        f"fail={imports['failed']} skip={imports['skipped']} not_run={imports['not_run']} "
        f"complete={imports['complete']} deadline_exceeded={imports['deadline_exceeded']} "
        f"duration_ms={imports['duration_ms']:.3f}"
    )
    lines.append(
        f"RESULT {report['status'].upper()} elapsed_ms={report['elapsed_ms']:.3f} "
        f"summary={json.dumps(report['summary'], sort_keys=True)}"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code_root = Path(__file__).resolve().parents[2]
    if args.format == "json":
        # Keep stdout machine-readable even when imported dependencies print.
        with redirect_stdout(sys.stderr):
            report = execute_smoke(
                code_root,
                deadline_seconds=args.deadline_seconds,
                per_import_timeout_seconds=args.per_import_timeout_seconds,
            )
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        report = execute_smoke(
            code_root,
            deadline_seconds=args.deadline_seconds,
            per_import_timeout_seconds=args.per_import_timeout_seconds,
        )
        print(render_text(report))
    return 0 if report["status"] == STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
