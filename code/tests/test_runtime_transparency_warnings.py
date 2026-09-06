from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _run_fresh_python(code: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_ncu_set_discovery_records_structured_warning(monkeypatch, tmp_path) -> None:
    from core.profiling.nsight_automation import NsightAutomation

    automation = NsightAutomation(tmp_path)
    automation.ncu_available = True
    automation._ncu_sets_cache = None

    def _raise_list_sets(*_args: Any, **_kwargs: Any):
        raise RuntimeError("list-sets failed")

    monkeypatch.setattr(subprocess, "run", _raise_list_sets)

    assert automation._available_ncu_sets() == set()

    automation._begin_run({"tool": "ncu"})
    warnings_list = automation.last_run.get("warnings", [])
    assert any("Failed to enumerate available Nsight Compute metric sets" in item for item in warnings_list)
    assert any("list-sets failed" in item for item in warnings_list)


def test_finalize_timed_out_nsys_records_cleanup_warnings(monkeypatch, tmp_path) -> None:
    from core.profiling.nsight_automation import NsightAutomation

    automation = NsightAutomation(tmp_path)
    automation.last_run = {"tool": "nsys"}

    class _FakeProcess:
        pid = 4242

        def __init__(self) -> None:
            self._calls = 0

        def communicate(self, timeout=None):
            self._calls += 1
            if self._calls <= 2:
                raise subprocess.TimeoutExpired(cmd="nsys", timeout=timeout, output="", stderr="")
            raise RuntimeError("trailing stderr collection failed")

    monkeypatch.setattr(
        "core.profiling.nsight_automation.os.killpg",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ProcessLookupError("process gone")),
    )
    monkeypatch.setattr(automation, "_detect_nsys_defunct_launcher", lambda _pid: False)

    finalize = automation._finalize_timed_out_nsys(_FakeProcess(), grace_seconds=1.0)

    assert finalize["completed"] is False
    assert len(finalize["warnings"]) == 4
    assert any("SIGINT" in item for item in finalize["warnings"])
    assert any("SIGTERM" in item for item in finalize["warnings"])
    assert any("SIGKILL" in item for item in finalize["warnings"])
    assert any("trailing profiler output" in item for item in finalize["warnings"])
    assert automation.last_run.get("warnings") == finalize["warnings"]


def test_reset_cuda_memory_pool_emits_runtime_warning(monkeypatch) -> None:
    import core.harness.validity_checks as validity_checks

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda _device=None: None,
            empty_cache=lambda: None,
            ipc_collect=lambda: (_ for _ in ()).throw(RuntimeError("ipc reset failed")),
            reset_peak_memory_stats=lambda _device=None: None,
            reset_accumulated_memory_stats=lambda _device=None: None,
        ),
        _C=SimpleNamespace(),
    )
    monkeypatch.setattr(validity_checks, "torch", fake_torch)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        validity_checks.reset_cuda_memory_pool()

    assert any("Failed to collect CUDA IPC allocations" in str(item.message) for item in captured)
    assert any("ipc reset failed" in str(item.message) for item in captured)


def test_reset_cuda_memory_pool_emits_release_pool_limitation_once(monkeypatch) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks, "_EMITTED_VALIDITY_LIMITATIONS", set())
    monkeypatch.setattr(validity_checks, "_VALIDITY_LIMITATION_RECORDS", {})
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda _device=None: None,
            empty_cache=lambda: None,
            ipc_collect=lambda: None,
            reset_peak_memory_stats=lambda _device=None: None,
            reset_accumulated_memory_stats=lambda _device=None: None,
        ),
        _C=SimpleNamespace(
            _cuda_releasePool=lambda: (_ for _ in ()).throw(
                TypeError(
                    "_cuda_releasePool(): incompatible function arguments. "
                    "The following argument types are supported: "
                    "(arg0: typing.SupportsInt, arg1: tuple[typing.SupportsInt, typing.SupportsInt]) -> None"
                )
            )
        ),
    )
    monkeypatch.setattr(validity_checks, "torch", fake_torch)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        validity_checks.reset_cuda_memory_pool()
        validity_checks.reset_cuda_memory_pool()

    # reset_cuda_memory_pool() deliberately calls gc.collect(), which can emit
    # warnings for unrelated objects made unreachable by earlier tests. Count
    # this limitation precisely so those warnings cannot make the dedupe check
    # flaky, while a duplicate release-pool warning still fails the test.
    release_pool_warnings = [
        item
        for item in captured
        if item.category is RuntimeWarning
        and "without a zero-argument reset entrypoint" in str(item.message)
    ]
    assert len(release_pool_warnings) == 1
    assert "_cuda_releasePool(): incompatible function arguments" in str(
        release_pool_warnings[0].message
    )
    assert validity_checks.get_runtime_capability_limitations() == [
        {
            "key": "cuda_release_pool_signature",
            "category": "allocator_cleanup",
            "component": "cuda_memory_pool_reset",
            "summary": "This PyTorch runtime exposes torch._C._cuda_releasePool without a zero-argument reset entrypoint; memory-pool reset will fall back to cache/IPC/stat cleanup and allocator reuse checks are partially degraded",
            "detail": "_cuda_releasePool(): incompatible function arguments. The following argument types are supported: (arg0: typing.SupportsInt, arg1: tuple[typing.SupportsInt, typing.SupportsInt]) -> None",
            "first_observed_at": validity_checks.get_runtime_capability_limitations()[0]["first_observed_at"],
        }
    ]


def test_reset_cuda_memory_pool_emits_allocator_limitation_once(monkeypatch) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks, "_EMITTED_VALIDITY_LIMITATIONS", set())
    monkeypatch.setattr(validity_checks, "_VALIDITY_LIMITATION_RECORDS", {})
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda _device=None: None,
            empty_cache=lambda: None,
            ipc_collect=lambda: None,
            reset_peak_memory_stats=lambda _device=None: None,
            reset_accumulated_memory_stats=lambda _device=None: None,
        ),
        _C=SimpleNamespace(
            _accelerator_setAllocatorSettings=lambda _value: (_ for _ in ()).throw(
                RuntimeError("Unrecognized CachingAllocator option: reset_allocator")
            )
        ),
    )
    monkeypatch.setattr(validity_checks, "torch", fake_torch)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        validity_checks.reset_cuda_memory_pool()
        validity_checks.reset_cuda_memory_pool()

    assert len(captured) == 1
    assert "does not support the reset_allocator allocator setting" in str(captured[0].message)
    assert "Unrecognized CachingAllocator option: reset_allocator" in str(captured[0].message)
    assert validity_checks.get_runtime_capability_limitations() == [
        {
            "key": "cuda_allocator_reset_setting",
            "category": "allocator_cleanup",
            "component": "cuda_memory_pool_reset",
            "summary": "This PyTorch runtime does not support the reset_allocator allocator setting; memory-pool cleanup will proceed without forcing allocator-setting resets and allocator reuse checks are partially degraded",
            "detail": "Unrecognized CachingAllocator option: reset_allocator",
            "first_observed_at": validity_checks.get_runtime_capability_limitations()[0]["first_observed_at"],
        }
    ]


def test_clear_compile_cache_emits_runtime_warning(monkeypatch) -> None:
    import core.harness.validity_checks as validity_checks

    fake_torch = SimpleNamespace(
        _dynamo=SimpleNamespace(reset=lambda: (_ for _ in ()).throw(RuntimeError("dynamo reset failed"))),
        _inductor=SimpleNamespace(codecache=SimpleNamespace(clear=lambda: None)),
    )
    monkeypatch.setattr(validity_checks, "torch", fake_torch)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = validity_checks.clear_compile_cache()

    assert result is False
    assert any("Failed to clear torch.compile caches" in str(item.message) for item in captured)
    assert any("dynamo reset failed" in str(item.message) for item in captured)


def test_get_compile_state_emits_runtime_warning(monkeypatch) -> None:
    import core.harness.validity_checks as validity_checks

    class _ExplodingCounters:
        def get(self, *_args: Any, **_kwargs: Any):
            raise RuntimeError("counter read failed")

    fake_torch = SimpleNamespace(
        _dynamo=SimpleNamespace(
            utils=SimpleNamespace(counters=_ExplodingCounters())
        )
    )
    monkeypatch.setattr(validity_checks, "torch", fake_torch)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        state = validity_checks.get_compile_state()

    assert state["dynamo_available"] is True
    assert state["compile_count"] == 0
    assert any("Failed to inspect torch.compile state" in str(item.message) for item in captured)
    assert any("counter read failed" in str(item.message) for item in captured)


def test_stream_auditor_stop_emits_restore_warnings(monkeypatch) -> None:
    import core.harness.validity_checks as validity_checks

    class _RaisingStreamMeta(type):
        def __setattr__(cls, name: str, value: Any) -> None:
            if name in {"synchronize", "wait_stream"}:
                raise RuntimeError(f"{name} restore failed")
            super().__setattr__(name, value)

    class _FakeStream(metaclass=_RaisingStreamMeta):
        synchronize = object()
        wait_stream = object()

    class _FakeCuda:
        def __init__(self) -> None:
            object.__setattr__(self, "_raise_on_restore", False)
            object.__setattr__(self, "Stream", _FakeStream)
            object.__setattr__(self, "synchronize", object())
            object.__setattr__(self, "stream", object())
            object.__setattr__(self, "set_stream", object())

        def is_available(self) -> bool:
            return True

        def __setattr__(self, name: str, value: Any) -> None:
            if getattr(self, "_raise_on_restore", False) and name in {"synchronize", "stream", "set_stream"}:
                raise RuntimeError(f"{name} restore failed")
            object.__setattr__(self, name, value)

    fake_torch = SimpleNamespace(cuda=_FakeCuda())
    monkeypatch.setattr(validity_checks, "torch", fake_torch)

    auditor = validity_checks.StreamAuditor()
    auditor._orig_stream_cls = _FakeStream
    auditor._orig_stream_synchronize = object()
    auditor._orig_stream_wait_stream = object()
    auditor._orig_synchronize = object()
    auditor._orig_stream_fn = object()
    auditor._orig_set_stream_fn = object()
    fake_torch.cuda._raise_on_restore = True

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        auditor.stop()

    messages = [str(item.message) for item in captured]
    assert any("Failed to restore torch.cuda.Stream.synchronize" in item for item in messages)
    assert any("Failed to restore torch.cuda.Stream.wait_stream" in item for item in messages)
    assert any("Failed to restore torch.cuda.synchronize" in item for item in messages)
    assert any("Failed to restore torch.cuda.stream" in item for item in messages)
    assert any("Failed to restore torch.cuda.set_stream" in item for item in messages)


def test_direct_validity_checks_import_succeeds_in_fresh_python() -> None:
    result = _run_fresh_python("import core.harness.validity_checks as vc; print(vc.__name__)")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "core.harness.validity_checks"


def test_direct_hot_path_checks_import_succeeds_in_fresh_python() -> None:
    result = _run_fresh_python("import core.hot_path_checks as hp; print(hp.__name__)")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "core.hot_path_checks"


def test_direct_benchmark_contract_import_succeeds_in_fresh_python() -> None:
    result = _run_fresh_python("import core.benchmark.contract as contract; print(contract.__name__)")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "core.benchmark.contract"


def test_core_benchmark_package_import_stays_lazy_in_fresh_python() -> None:
    result = _run_fresh_python(
        "import sys\n"
        "import core.benchmark as benchmark\n"
        "assert 'core.benchmark.verify_runner' not in sys.modules\n"
        "assert 'core.benchmark.contract' not in sys.modules\n"
        "print(benchmark.__name__)\n"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "core.benchmark"


def test_core_benchmark_package_exports_resolve_in_fresh_python() -> None:
    result = _run_fresh_python(
        "from core.benchmark import BenchmarkContract, InputSignature, VerifyRunner\n"
        "print(BenchmarkContract.__module__)\n"
        "print(InputSignature.__module__)\n"
        "print(VerifyRunner.__module__)\n"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "core.benchmark.contract",
        "core.benchmark.verification",
        "core.benchmark.verify_runner",
    ]


def test_core_common_package_import_stays_lazy_in_fresh_python() -> None:
    result = _run_fresh_python(
        "import sys\n"
        "import core.common as common\n"
        "assert 'core.common.async_input_pipeline' not in sys.modules\n"
        "assert 'core.common.moe_parallelism_plan' not in sys.modules\n"
        "assert 'core.common.device_utils' not in sys.modules\n"
        "print(common.__name__)\n"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "core.common"


def test_core_common_package_exports_resolve_in_fresh_python() -> None:
    result = _run_fresh_python(
        "from core.common import ClusterSpec, PipelineConfig, get_preferred_device\n"
        "print(ClusterSpec.__module__)\n"
        "print(PipelineConfig.__module__)\n"
        "print(get_preferred_device.__module__)\n"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "core.common.moe_parallelism_plan",
        "core.common.async_input_pipeline",
        "core.common.device_utils",
    ]
