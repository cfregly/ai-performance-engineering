from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

from core.scripts import benchmark_coverage
from core.utils import extension_prewarm


def test_benchmark_coverage_records_metric_helper(tmp_path: Path) -> None:
    benchmark = tmp_path / "baseline_example.py"
    benchmark.write_text(
        "def get_custom_metrics():\n    return compute_matmul_metrics(measured_time_ms=1.0)\n",
        encoding="utf-8",
    )

    result = benchmark_coverage.analyze_file(benchmark)

    assert result["uses_helper"] is True
    assert result["helper_name"] == "compute_matmul_metrics"


def test_benchmark_coverage_main_scans_repository_root(monkeypatch) -> None:
    scanned_roots: list[Path] = []

    def _capture_root(root: Path) -> benchmark_coverage.CoverageReport:
        scanned_roots.append(root)
        return benchmark_coverage.CoverageReport()

    monkeypatch.setattr(benchmark_coverage, "generate_report", _capture_root)
    monkeypatch.setattr(benchmark_coverage, "print_text_report", lambda _report: None)
    monkeypatch.setattr("sys.argv", ["benchmark_coverage.py"])

    benchmark_coverage.main()

    expected_root = Path(benchmark_coverage.__file__).resolve().parents[2]
    assert scanned_roots == [expected_root]


def test_extension_health_check_uses_temporary_repo_import_path(monkeypatch) -> None:
    events: list[object] = []

    @contextmanager
    def _tracked_import_context():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def _check_with_imports(*, verbose: bool) -> bool:
        events.append(verbose)
        return True

    monkeypatch.setattr(extension_prewarm, "_repo_import_context", _tracked_import_context)
    monkeypatch.setattr(
        extension_prewarm,
        "_health_check_with_repo_imports",
        _check_with_imports,
    )

    assert extension_prewarm.health_check(verbose=False) is True
    assert events == ["enter", False, "exit"]


def test_extension_prewarm_invokes_an_explicit_registered_loader(monkeypatch) -> None:
    events: list[str] = []

    def load():
        events.append("load")
        return object()

    registration = extension_prewarm.ExtensionRegistration(
        name="application.example",
        loader=load,
        probe=lambda: (True, "ready"),
        owner="application",
    )
    monkeypatch.setattr(extension_prewarm, "_core_extensions", lambda: ())
    monkeypatch.setattr(extension_prewarm, "_repo_import_context", nullcontext)
    extension_prewarm.register_extension(registration)
    try:
        result = extension_prewarm._do_prewarm(verbose=False, parallel=False)
    finally:
        extension_prewarm.unregister_extension(registration.name)

    assert result == {"application.example": extension_prewarm.ExtensionResult("built", "OK")}
    assert events == ["load"]


def test_extension_registration_rejects_conflicting_duplicate() -> None:
    first = extension_prewarm.ExtensionRegistration(
        name="application.duplicate",
        loader=lambda: None,
        probe=lambda: (True, "ready"),
        owner="application",
    )
    second = extension_prewarm.ExtensionRegistration(
        name=first.name,
        loader=lambda: None,
        probe=lambda: (True, "different"),
        owner="application",
    )
    extension_prewarm.register_extension(first)
    try:
        with pytest.raises(ValueError, match="already registered"):
            extension_prewarm.register_extension(second)
    finally:
        extension_prewarm.unregister_extension(first.name)


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        ("core", "core.common.tcgen05.load_matmul_tcgen05_module"),
        ("core.common.tcgen05", "core.common.tcgen05.load_matmul_tcgen05_module"),
        ("core.application", "core.application.example"),
    ],
)
def test_extension_registration_rejects_reserved_core_namespace(owner, name) -> None:
    built_in_name = extension_prewarm._core_extensions()[0].name
    registration = extension_prewarm.ExtensionRegistration(
        name=built_in_name if "load_matmul" in name else name,
        loader=lambda: None,
        probe=lambda: (True, "ready"),
        owner=owner,
    )

    with pytest.raises(ValueError, match="reserved core namespace"):
        extension_prewarm.register_extension(registration)

    with pytest.raises(ValueError, match="reserved core namespace"):
        extension_prewarm.prewarm_extension_registrations((registration,))


def test_extension_prewarm_skips_unsupported_capability_without_loading(monkeypatch) -> None:
    events: list[str] = []
    registration = extension_prewarm.ExtensionRegistration(
        name="application.sm100_only",
        loader=lambda: events.append("load"),
        probe=lambda: (True, "ready"),
        supported_capabilities=frozenset({(10, 0), (10, 3)}),
        owner="application",
    )
    monkeypatch.setattr(extension_prewarm, "_core_extensions", lambda: ())
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    extension_prewarm.register_extension(registration)
    try:
        result = extension_prewarm._do_prewarm(verbose=False, parallel=False)
    finally:
        extension_prewarm.unregister_extension(registration.name)

    assert result[registration.name].status == "skipped"
    assert result[registration.name].message.startswith("requires SM100")
    assert events == []


def test_extension_inventory_is_owner_scoped() -> None:
    registration = extension_prewarm.ExtensionRegistration(
        name="application.example",
        loader=lambda: None,
        probe=lambda: (True, "ready"),
        owner="different-owner",
    )
    with pytest.raises(ValueError, match="scoped to their owner"):
        extension_prewarm.register_extension(registration)


@pytest.mark.parametrize("parallel", [False, True])
def test_extension_prewarm_preserves_success_and_failure_results(parallel) -> None:
    events: list[str] = []

    def succeed():
        events.append("successful-loader")
        return object()

    def fail() -> None:
        events.append("failed-loader")
        raise RuntimeError("compile failed")

    registrations = (
        extension_prewarm.ExtensionRegistration(
            name="application.success",
            loader=succeed,
            probe=lambda: (True, "ready"),
            owner="application",
        ),
        extension_prewarm.ExtensionRegistration(
            name="application.failure",
            loader=fail,
            probe=lambda: (True, "ready"),
            owner="application",
        ),
    )

    results = extension_prewarm.prewarm_extension_registrations(
        registrations,
        parallel=parallel,
    )

    assert results["application.success"].status == "built"
    assert results["application.failure"] == extension_prewarm.ExtensionResult(
        "failed", "compile failed"
    )
    assert sorted(events) == ["failed-loader", "successful-loader"]


def test_extension_loader_requires_a_build_artifact() -> None:
    registration = extension_prewarm.ExtensionRegistration(
        name="application.no_artifact",
        loader=lambda: None,
        probe=lambda: (True, "ready"),
        owner="application",
    )

    results = extension_prewarm.prewarm_extension_registrations(
        (registration,),
        parallel=False,
    )

    assert results == {
        registration.name: extension_prewarm.ExtensionResult(
            "failed", "extension loader returned no build artifact"
        )
    }


def test_explicit_extension_inventory_does_not_overwrite_default_lifecycle(
    monkeypatch,
) -> None:
    sentinel_results = {"core.sentinel": extension_prewarm.ExtensionResult("built", "existing")}
    sentinel_times = {"core.sentinel": 1.25}
    monkeypatch.setattr(extension_prewarm, "_PREWARM_STATE", "idle")
    monkeypatch.setattr(extension_prewarm, "_PREWARM_THREAD", None)
    monkeypatch.setattr(extension_prewarm, "_PREWARM_RESULTS", sentinel_results)
    monkeypatch.setattr(extension_prewarm, "_PREWARM_TIMES", sentinel_times)
    monkeypatch.setattr(extension_prewarm, "_repo_import_context", nullcontext)
    registration = extension_prewarm.ExtensionRegistration(
        name="application.local",
        loader=lambda: object(),
        probe=lambda: (True, "ready"),
        owner="application",
    )

    result = extension_prewarm.prewarm_extension_registrations(
        (registration,),
        parallel=False,
    )

    assert result == {"application.local": extension_prewarm.ExtensionResult("built", "OK")}
    assert extension_prewarm._PREWARM_STATE == "idle"
    assert extension_prewarm._PREWARM_THREAD is None
    assert extension_prewarm._PREWARM_RESULTS is sentinel_results
    assert extension_prewarm._PREWARM_TIMES is sentinel_times


def test_extension_success_requires_every_requested_target_to_build() -> None:
    built = extension_prewarm.ExtensionResult("built", "OK")
    skipped = extension_prewarm.ExtensionResult("skipped", "unsupported")
    failed = extension_prewarm.ExtensionResult("failed", "compile failed")

    assert extension_prewarm.all_extensions_built({}) is False
    assert extension_prewarm.all_extensions_built({"a": skipped}) is False
    assert extension_prewarm.all_extensions_built({"a": built, "b": skipped}) is False
    assert extension_prewarm.all_extensions_built({"a": built, "b": failed}) is False
    assert extension_prewarm.all_extensions_built({"a": built, "b": built}) is True
    assert (
        extension_prewarm.all_extensions_built(
            {"a": built},
            expected_names=("a", "b"),
        )
        is False
    )


def test_default_core_extension_inventory_never_imports_chapter_packages(monkeypatch) -> None:
    import builtins

    imported_chapters: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", maxsplit=1)[0] in {"ch06", "ch12"}:
            imported_chapters.append(name)
            raise AssertionError(f"unexpected chapter import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    results = extension_prewarm._do_prewarm(verbose=False, parallel=False)

    assert results
    assert all(result.status == "skipped" for result in results.values())
    assert imported_chapters == []


def test_operational_extension_catalog_preserves_chapter_ownership(monkeypatch) -> None:
    import sys
    import types

    from scripts.utilities import precompile_cuda_extensions as command

    ch06 = types.ModuleType("ch06.cuda_extensions")
    ch12 = types.ModuleType("ch12.cuda_extensions")
    for name in (
        "load_bank_conflicts_extension",
        "load_coalescing_extension",
        "load_ilp_extension",
        "load_launch_bounds_extension",
    ):
        setattr(ch06, name, lambda: None)
    for name in (
        "load_bias_relu_residual_extension",
        "load_cuda_graphs_extension",
        "load_graph_bandwidth_extension",
        "load_kernel_fusion_extension",
        "load_work_queue_extension",
    ):
        setattr(ch12, name, lambda: None)
    monkeypatch.setitem(sys.modules, "ch06.cuda_extensions", ch06)
    monkeypatch.setitem(sys.modules, "ch12.cuda_extensions", ch12)

    registrations = command.chapter_extension_registrations()

    assert {registration.name for registration in registrations} == {
        "ch06.cuda_extensions.bank_conflicts",
        "ch06.cuda_extensions.coalescing",
        "ch06.cuda_extensions.ilp",
        "ch06.cuda_extensions.launch_bounds",
        "ch12.cuda_extensions.bias_relu_residual",
        "ch12.cuda_extensions.cuda_graphs",
        "ch12.cuda_extensions.graph_bandwidth",
        "ch12.cuda_extensions.kernel_fusion",
        "ch12.cuda_extensions.work_queue",
    }
    assert {registration.owner for registration in registrations} == {"ch06", "ch12"}
    assert all(
        registration.name.startswith(f"{registration.owner}.") for registration in registrations
    )
    assert all(registration.probe() == (True, "ready") for registration in registrations)


def test_operational_extension_catalog_has_no_hidden_nvtx_build(monkeypatch) -> None:
    from ch06 import cuda_extensions as ch06_extensions
    from ch12 import cuda_extensions as ch12_extensions
    from scripts.utilities import precompile_cuda_extensions as command

    calls: list[str] = []

    def fail_if_built() -> None:
        calls.append("nvtx")
        raise RuntimeError("fixture NVTX toolchain unavailable")

    monkeypatch.setattr(ch06_extensions, "ensure_nvtx_stub", fail_if_built)
    monkeypatch.setattr(ch12_extensions, "ensure_nvtx_stub", fail_if_built)

    registrations = command.chapter_extension_registrations()
    assert calls == []

    results = extension_prewarm.prewarm_extension_registrations(
        registrations,
        parallel=False,
    )
    assert len(calls) == len(registrations)
    assert all(result.status == "failed" for result in results.values())
    assert all(
        result.message == "fixture NVTX toolchain unavailable" for result in results.values()
    )


def test_operational_extension_command_rejects_partial_build(monkeypatch) -> None:
    from scripts.utilities import precompile_cuda_extensions as command

    registrations = (
        extension_prewarm.ExtensionRegistration(
            name="chapter.first",
            owner="chapter",
            loader=lambda: None,
            probe=lambda: (True, "ready"),
        ),
        extension_prewarm.ExtensionRegistration(
            name="chapter.second",
            owner="chapter",
            loader=lambda: None,
            probe=lambda: (True, "ready"),
        ),
    )
    monkeypatch.setattr(command.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(command.torch.cuda, "get_device_name", lambda _index: "fixture")
    monkeypatch.setattr(command, "chapter_extension_registrations", lambda: registrations)
    monkeypatch.setattr(
        command,
        "prewarm_extension_registrations",
        lambda *_args, **_kwargs: {
            "chapter.first": extension_prewarm.ExtensionResult("built", "OK"),
            "chapter.second": extension_prewarm.ExtensionResult("skipped", "unsupported"),
        },
    )

    assert command.precompile_extensions(parallel=False) is False


def test_background_prewarm_records_setup_failure_and_retries(monkeypatch) -> None:
    attempts = 0
    registration = extension_prewarm.ExtensionRegistration(
        name="application.retry",
        owner="application",
        loader=lambda: object(),
        probe=lambda: (True, "ready"),
    )

    def flaky_inventory():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("inventory exploded")
        return (registration,)

    monkeypatch.setattr(extension_prewarm, "_PREWARM_STATE", "idle")
    monkeypatch.setattr(extension_prewarm, "_PREWARM_THREAD", None)
    monkeypatch.setattr(extension_prewarm, "_PREWARM_RESULTS", {})
    monkeypatch.setattr(extension_prewarm, "_PREWARM_TIMES", {})
    monkeypatch.setattr(extension_prewarm, "_extension_inventory", flaky_inventory)
    monkeypatch.setattr(extension_prewarm, "_repo_import_context", nullcontext)

    failed = extension_prewarm.prewarm_extensions(
        background=True,
        wait=True,
        parallel=False,
    )

    assert failed == {
        extension_prewarm._PREWARM_SETUP_RESULT_NAME: extension_prewarm.ExtensionResult(
            "failed", "inventory exploded"
        )
    }
    assert extension_prewarm._PREWARM_STATE == "failed"
    assert extension_prewarm.is_prewarm_complete() is True

    retried = extension_prewarm.prewarm_extensions(background=False, parallel=False)

    assert retried == {registration.name: extension_prewarm.ExtensionResult("built", "OK")}
    assert attempts == 2
    assert extension_prewarm._PREWARM_STATE == "complete"


def test_background_thread_handle_is_published_before_waiters_run(monkeypatch) -> None:
    real_thread = threading.Thread
    constructor_entered = threading.Event()
    release_constructor = threading.Event()
    waiter_entered = threading.Event()
    first_results: list[object] = []
    waiter_results: list[object] = []

    class BlockingThread:
        def __init__(self, *args, **kwargs):
            constructor_entered.set()
            if not release_constructor.wait(timeout=5):
                raise RuntimeError("thread constructor barrier timed out")
            self._thread = real_thread(*args, **kwargs)

        def start(self):
            return self._thread.start()

    monkeypatch.setattr(extension_prewarm, "_PREWARM_STATE", "idle")
    monkeypatch.setattr(extension_prewarm, "_PREWARM_THREAD", None)
    monkeypatch.setattr(extension_prewarm, "_PREWARM_RESULTS", {})
    monkeypatch.setattr(extension_prewarm, "_PREWARM_TIMES", {})
    monkeypatch.setattr(extension_prewarm, "_extension_inventory", lambda: ())
    monkeypatch.setattr(extension_prewarm, "_repo_import_context", nullcontext)
    monkeypatch.setattr(extension_prewarm.threading, "Thread", BlockingThread)

    first = real_thread(
        target=lambda: first_results.append(extension_prewarm.prewarm_extensions(background=True))
    )
    first.start()
    assert constructor_entered.wait(timeout=5)

    def wait_for_result() -> None:
        waiter_entered.set()
        waiter_results.append(extension_prewarm.prewarm_extensions(background=True, wait=True))

    waiter = real_thread(target=wait_for_result)
    waiter.start()
    assert waiter_entered.wait(timeout=5)
    try:
        waiter.join(timeout=0.1)
        assert waiter.is_alive(), "waiter returned before the thread handle was published"
    finally:
        release_constructor.set()

    first.join(timeout=5)
    waiter.join(timeout=5)
    assert not first.is_alive()
    assert not waiter.is_alive()
    assert first_results == [None]
    assert waiter_results == [{}]
    assert extension_prewarm._PREWARM_STATE == "complete"


def test_registry_mutation_invalidates_cached_state_and_rejects_running_mutation(
    monkeypatch,
) -> None:
    sentinel_results = {"core.sentinel": extension_prewarm.ExtensionResult("built", "cached")}
    registration = extension_prewarm.ExtensionRegistration(
        name="application.lifecycle",
        owner="application",
        loader=lambda: object(),
        probe=lambda: (True, "ready"),
    )
    monkeypatch.setattr(extension_prewarm, "_PREWARM_STATE", "complete")
    monkeypatch.setattr(extension_prewarm, "_PREWARM_THREAD", None)
    monkeypatch.setattr(extension_prewarm, "_PREWARM_RESULTS", sentinel_results)
    monkeypatch.setattr(extension_prewarm, "_PREWARM_TIMES", {"core.sentinel": 1.0})

    extension_prewarm.register_extension(registration)
    try:
        assert extension_prewarm._PREWARM_STATE == "idle"
        assert extension_prewarm.get_prewarm_results() == {}

        monkeypatch.setattr(extension_prewarm, "_PREWARM_STATE", "running")
        with pytest.raises(RuntimeError, match="cannot unregister"):
            extension_prewarm.unregister_extension(registration.name)
        assert registration.name in extension_prewarm._REGISTERED_EXTENSIONS
    finally:
        monkeypatch.setattr(extension_prewarm, "_PREWARM_STATE", "idle")
        extension_prewarm.unregister_extension(registration.name)

    monkeypatch.setattr(extension_prewarm, "_PREWARM_STATE", "running")
    with pytest.raises(RuntimeError, match="cannot register"):
        extension_prewarm.register_extension(registration)
    assert registration.name not in extension_prewarm._REGISTERED_EXTENSIONS


def test_wait_for_prewarm_timeout_is_structured(monkeypatch) -> None:
    loader_started = threading.Event()
    release_loader = threading.Event()

    def slow_loader():
        loader_started.set()
        if not release_loader.wait(timeout=5):
            raise RuntimeError("slow loader release timed out")
        return object()

    registration = extension_prewarm.ExtensionRegistration(
        name="application.slow",
        owner="application",
        loader=slow_loader,
        probe=lambda: (True, "ready"),
    )
    monkeypatch.setattr(extension_prewarm, "_PREWARM_STATE", "idle")
    monkeypatch.setattr(extension_prewarm, "_PREWARM_THREAD", None)
    monkeypatch.setattr(extension_prewarm, "_PREWARM_RESULTS", {})
    monkeypatch.setattr(extension_prewarm, "_PREWARM_TIMES", {})
    monkeypatch.setattr(extension_prewarm, "_extension_inventory", lambda: (registration,))
    monkeypatch.setattr(extension_prewarm, "_repo_import_context", nullcontext)

    extension_prewarm.prewarm_extensions(background=True, parallel=False)
    assert loader_started.wait(timeout=5)
    timed_out = extension_prewarm.wait_for_prewarm(timeout=0.01)

    assert timed_out == {
        extension_prewarm._PREWARM_WAIT_RESULT_NAME: extension_prewarm.ExtensionResult(
            "failed", "timed out waiting for extension prewarm after 0.010s"
        )
    }
    assert extension_prewarm._PREWARM_STATE == "running"

    release_loader.set()
    completed = extension_prewarm.wait_for_prewarm(timeout=5)
    assert completed == {registration.name: extension_prewarm.ExtensionResult("built", "OK")}


def test_default_prewarm_results_are_not_caller_mutable(monkeypatch) -> None:
    registration = extension_prewarm.ExtensionRegistration(
        name="application.immutable_result",
        owner="application",
        loader=lambda: object(),
        probe=lambda: (True, "ready"),
    )
    expected = {registration.name: extension_prewarm.ExtensionResult("built", "OK")}
    monkeypatch.setattr(extension_prewarm, "_PREWARM_STATE", "idle")
    monkeypatch.setattr(extension_prewarm, "_PREWARM_THREAD", None)
    monkeypatch.setattr(extension_prewarm, "_PREWARM_RESULTS", {})
    monkeypatch.setattr(extension_prewarm, "_PREWARM_TIMES", {})
    monkeypatch.setattr(extension_prewarm, "_extension_inventory", lambda: (registration,))
    monkeypatch.setattr(extension_prewarm, "_repo_import_context", nullcontext)

    first = extension_prewarm.prewarm_extensions(background=False, parallel=False)
    assert first == expected
    first.clear()
    assert extension_prewarm.get_prewarm_results() == expected

    cached = extension_prewarm.prewarm_extensions(background=False, parallel=False)
    assert cached == expected
    cached[registration.name] = extension_prewarm.ExtensionResult("failed", "tampered")
    assert extension_prewarm.get_prewarm_results() == expected
