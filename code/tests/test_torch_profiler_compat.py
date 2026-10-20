"""Controls for the Torch 2.9 and CUDA 13 CUPTI initialization."""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from core.profiling import profiling_utils
from core.profiling import torch_profiler_compat as compat
from core.profiling.profiler_wrapper import render_torch_python_profile_wrapper


class _FakeCFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class _FakeCupti:
    def __init__(
        self,
        *,
        initial_value: int = 1,
        get_results: list[int] | None = None,
        set_result: int = 0,
        apply_set: bool = True,
    ) -> None:
        self.value = initial_value
        self.get_results = list(get_results or [0, 0])
        self.set_result = set_result
        self.apply_set = apply_set
        self.calls: list[tuple[str, int]] = []
        self.cuptiActivityGetAttribute = _FakeCFunction(self._get)
        self.cuptiActivitySetAttribute = _FakeCFunction(self._set)

    def _get(self, attribute, value_size, value) -> int:
        self.calls.append(("get", int(attribute)))
        ctypes.cast(value_size, ctypes.POINTER(ctypes.c_size_t))[0] = ctypes.sizeof(ctypes.c_uint8)
        ctypes.cast(value, ctypes.POINTER(ctypes.c_uint8))[0] = self.value
        return self.get_results.pop(0)

    def _set(self, attribute, value_size, value) -> int:
        requested = int(ctypes.cast(value, ctypes.POINTER(ctypes.c_uint8))[0])
        self.calls.append(("set", int(attribute)))
        if self.apply_set and self.set_result == 0:
            self.value = requested
        return self.set_result


def _set_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    torch_version: str = "2.9.1+cu130",
    cuda_version: str | None = "13.0",
) -> None:
    monkeypatch.setattr(
        compat,
        "torch",
        SimpleNamespace(
            __version__=torch_version,
            version=SimpleNamespace(cuda=cuda_version),
            autograd=SimpleNamespace(_profiler_enabled=lambda: False),
            cuda=SimpleNamespace(is_available=lambda: True),
        ),
    )
    monkeypatch.setattr(compat, "_prepared_pid", None)
    monkeypatch.setattr(compat.os, "getpid", lambda: 31415)


def test_prepare_sets_and_verifies_attribute_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_runtime(monkeypatch)
    cupti = _FakeCupti()
    loaded: list[str] = []

    def load_library(name: str) -> _FakeCupti:
        loaded.append(name)
        return cupti

    monkeypatch.setattr(compat.ctypes, "CDLL", load_library)

    compat.prepare_torch_cuda_profiler(cuda_profile_requested=True)
    compat.prepare_torch_cuda_profiler(cuda_profile_requested=True)

    assert loaded == ["libcupti.so.13"]
    assert cupti.calls == [("get", 9), ("set", 9), ("get", 9)]
    assert cupti.value == 0


def test_prepare_rechecks_state_after_process_id_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_runtime(monkeypatch)
    cupti = _FakeCupti(get_results=[0, 0, 0])
    process_ids = iter([10, 20])
    monkeypatch.setattr(compat.os, "getpid", lambda: next(process_ids))
    loads: list[str] = []

    def load_library(name: str) -> _FakeCupti:
        loads.append(name)
        return cupti

    monkeypatch.setattr(compat.ctypes, "CDLL", load_library)

    compat.prepare_torch_cuda_profiler(cuda_profile_requested=True)
    compat.prepare_torch_cuda_profiler(cuda_profile_requested=True)

    assert loads == ["libcupti.so.13", "libcupti.so.13"]
    assert cupti.calls == [
        ("get", 9),
        ("set", 9),
        ("get", 9),
        ("get", 9),
    ]


def test_prepare_does_not_write_attribute_when_already_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_runtime(monkeypatch)
    cupti = _FakeCupti(initial_value=0)
    monkeypatch.setattr(compat.ctypes, "CDLL", lambda _name: cupti)

    compat.prepare_torch_cuda_profiler(cuda_profile_requested=True)

    assert cupti.calls == [("get", 9)]


def test_prepare_rejects_active_profiler_before_loading_cupti(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_runtime(monkeypatch)
    compat.torch.autograd._profiler_enabled = lambda: True

    def unexpected_load(_name: str) -> Any:
        raise AssertionError("CUPTI must not load while a profiler is active")

    monkeypatch.setattr(compat.ctypes, "CDLL", unexpected_load)

    with pytest.raises(RuntimeError, match="profiler is already active"):
        compat.prepare_torch_cuda_profiler(cuda_profile_requested=True)


@pytest.mark.parametrize(
    ("torch_version", "cuda_version", "cuda_profile_requested"),
    [
        ("2.9.1+cu130", "13.0", False),
        ("2.9.1+cu128", "12.8", True),
        ("2.10.0+cu130", "13.0", True),
        ("2.9.1+cpu", None, True),
    ],
)
def test_prepare_is_noop_outside_affected_cuda_profile(
    monkeypatch: pytest.MonkeyPatch,
    torch_version: str,
    cuda_version: str | None,
    cuda_profile_requested: bool,
) -> None:
    _set_runtime(
        monkeypatch,
        torch_version=torch_version,
        cuda_version=cuda_version,
    )

    def unexpected_load(_name: str) -> Any:
        raise AssertionError("CUPTI must not load for this runtime or profile")

    monkeypatch.setattr(compat.ctypes, "CDLL", unexpected_load)

    compat.prepare_torch_cuda_profiler(cuda_profile_requested=cuda_profile_requested)


def test_prepare_is_noop_when_cuda_wheel_has_no_visible_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_runtime(monkeypatch)
    compat.torch.cuda.is_available = lambda: False

    def unexpected_load(_name: str) -> Any:
        raise AssertionError("CUPTI must not load without an available CUDA device")

    monkeypatch.setattr(compat.ctypes, "CDLL", unexpected_load)

    compat.prepare_torch_cuda_profiler(cuda_profile_requested=True)


@pytest.mark.parametrize(
    ("cupti", "message"),
    [
        (
            _FakeCupti(get_results=[7]),
            "cuptiActivityGetAttribute before initialization.*status 7",
        ),
        (
            _FakeCupti(set_result=8),
            "cuptiActivitySetAttribute.*status 8",
        ),
        (
            _FakeCupti(get_results=[0, 9]),
            "cuptiActivityGetAttribute after initialization.*status 9",
        ),
        (
            _FakeCupti(apply_set=False),
            "did not disable per-thread activity buffers",
        ),
    ],
)
def test_prepare_fails_for_cupti_errors(
    monkeypatch: pytest.MonkeyPatch,
    cupti: _FakeCupti,
    message: str,
) -> None:
    _set_runtime(monkeypatch)
    monkeypatch.setattr(compat.ctypes, "CDLL", lambda _name: cupti)

    with pytest.raises(RuntimeError, match=message):
        compat.prepare_torch_cuda_profiler(cuda_profile_requested=True)

    assert compat._prepared_pid is None


def test_prepare_fails_when_cupti_library_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_runtime(monkeypatch)

    def missing_library(_name: str) -> Any:
        raise OSError("missing")

    monkeypatch.setattr(compat.ctypes, "CDLL", missing_library)

    with pytest.raises(RuntimeError, match="libcupti.so.13 could not be loaded"):
        compat.prepare_torch_cuda_profiler(cuda_profile_requested=True)


def test_chrome_trace_default_does_not_prepare_cupti_without_visible_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[bool] = []

    @contextmanager
    def fake_profile(**_kwargs):
        yield SimpleNamespace(export_chrome_trace=lambda _path: None)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        profiling_utils,
        "prepare_torch_cuda_profiler",
        lambda *, cuda_profile_requested: requested.append(cuda_profile_requested),
    )
    monkeypatch.setattr(profiling_utils, "profile", fake_profile)

    profiling_utils.profile_with_chrome_trace(
        lambda: None,
        trace_path=str(tmp_path / "trace.json"),
    )

    assert requested == [False]


def test_rendered_torch_wrapper_prepares_cupti_before_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_path = tmp_path / "benchmark.py"
    benchmark_path.write_text(
        """
class _Benchmark:
    def setup(self):
        pass

    def benchmark_fn(self):
        pass

    def teardown(self):
        pass


def get_benchmark():
    return _Benchmark()
""",
        encoding="utf-8",
    )
    events: list[str] = []

    @contextmanager
    def fake_profile(**_kwargs):
        events.append("profile_enter")
        yield SimpleNamespace(
            step=lambda: None,
            export_chrome_trace=lambda _path: None,
        )

    monkeypatch.setattr(
        compat,
        "prepare_torch_cuda_profiler",
        lambda **_kwargs: events.append("prepare"),
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(torch.profiler, "profile", fake_profile)

    wrapper = render_torch_python_profile_wrapper(
        benchmark_path=benchmark_path,
        torch_output=tmp_path / "trace.json",
        target_label=None,
        target_override_argv=None,
        validity_profile="portable",
        lock_gpu_clocks_flag=False,
        gpu_sm_clock_mhz=None,
        gpu_mem_clock_mhz=None,
    )
    scope: dict[str, Any] = {"__name__": "rendered_torch_wrapper_test"}
    exec(compile(wrapper, str(tmp_path / "wrapper.py"), "exec"), scope)
    scope["ramp_gpu_clocks"] = lambda **_kwargs: None

    scope["_run_profile"]()

    assert events == ["prepare", "profile_enter"]
