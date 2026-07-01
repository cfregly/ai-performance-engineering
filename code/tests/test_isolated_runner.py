from __future__ import annotations

import io
import os
from contextlib import redirect_stderr
import sys
from types import SimpleNamespace
from pathlib import Path

from core.harness import isolated_runner


def test_run_benchmark_emits_warning_when_reap_fails(monkeypatch, tmp_path) -> None:
    reaped: list[bool] = []
    monkeypatch.setattr(
        isolated_runner,
        "_reap_descendant_processes",
        lambda *args, **kwargs: reaped.append(True),
    )

    stderr = io.StringIO()
    with redirect_stderr(stderr):
        result = isolated_runner.run_benchmark(
            {
                "benchmark_module_path": str(tmp_path / "missing_benchmark.py"),
                "benchmark_class_name": "MissingBenchmark",
                "config_dict": {},
                "device": None,
                "initial_state": None,
            }
        )

    assert result["success"] is False
    assert reaped == []
    assert "isolated_runner_warning" not in stderr.getvalue()


def test_reset_cuda_state_emits_warnings_for_cleanup_failures(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: None,
            empty_cache=lambda: None,
            graph_pool_trim=lambda: (_ for _ in ()).throw(RuntimeError("graph reset failed")),
            current_device=lambda: 0,
            default_generators=[SimpleNamespace(set_offset=lambda _value: (_ for _ in ()).throw(RuntimeError("rng reset failed")))],
        ),
        _dynamo=SimpleNamespace(reset=lambda: (_ for _ in ()).throw(RuntimeError("dynamo reset failed"))),
        _inductor=SimpleNamespace(
            cudagraph_trees=SimpleNamespace(
                reset_cudagraph_trees=lambda: (_ for _ in ()).throw(RuntimeError("cudagraph reset failed"))
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    stderr = io.StringIO()
    with redirect_stderr(stderr):
        isolated_runner.reset_cuda_state()

    output = stderr.getvalue()
    assert "Failed to reset CUDA graph pool" in output
    assert "graph reset failed" in output
    assert "Failed to reset CUDA RNG state" in output
    assert "Failed to reset torch._dynamo state" in output
    assert "Failed to reset torch._inductor cudagraph trees" in output


def test_reset_cuda_state_skips_cuda_probe_for_fresh_cuda_worker(monkeypatch) -> None:
    triton_resets: list[bool] = []

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: False,
            is_available=lambda: (_ for _ in ()).throw(
                AssertionError("fresh worker should not probe CUDA availability")
            ),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        isolated_runner,
        "reset_triton_runtime_cache",
        lambda _emit_warning: triton_resets.append(True),
    )

    isolated_runner.reset_cuda_state("cuda:0")

    assert triton_resets == [True]


def test_reset_cuda_state_skips_gpu_cleanup_for_explicit_cpu_worker(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: (_ for _ in ()).throw(
                AssertionError("explicit CPU worker should not inspect CUDA")
            ),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        isolated_runner,
        "reset_triton_runtime_cache",
        lambda _emit_warning: (_ for _ in ()).throw(
            AssertionError("explicit CPU worker should not reset Triton CUDA cache")
        ),
    )

    isolated_runner.reset_cuda_state("cpu")


def test_reset_cuda_state_rotates_private_triton_cache_when_clear_api_missing(monkeypatch) -> None:
    class _FakeCacheKnobs:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1
            return self

        @property
        def dir(self) -> str:
            return os.environ["TRITON_CACHE_DIR"]

        @property
        def override_dir(self) -> str:
            return os.environ["TRITON_OVERRIDE_DIR"]

        @property
        def dump_dir(self) -> str:
            return os.environ["TRITON_DUMP_DIR"]

    fake_cache_knobs = _FakeCacheKnobs()
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    fake_triton = SimpleNamespace(
        runtime=SimpleNamespace(cache=SimpleNamespace()),
        knobs=SimpleNamespace(cache=fake_cache_knobs),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_OVERRIDE_DIR", raising=False)
    monkeypatch.delenv("TRITON_DUMP_DIR", raising=False)

    stderr = io.StringIO()
    with redirect_stderr(stderr):
        isolated_runner.reset_cuda_state()

    assert fake_cache_knobs.reset_calls == 1
    assert Path(os.environ["TRITON_CACHE_DIR"]).exists()
    assert Path(os.environ["TRITON_OVERRIDE_DIR"]).exists()
    assert Path(os.environ["TRITON_DUMP_DIR"]).exists()
    assert "Triton runtime cache" not in stderr.getvalue()


def test_apply_owner_markers_from_argv_sets_environment(monkeypatch) -> None:
    monkeypatch.delenv("AISP_BENCHMARK_OWNER_RUN_ID", raising=False)
    monkeypatch.delenv("AISP_BENCHMARK_OWNER_PID", raising=False)

    isolated_runner._apply_owner_markers_from_argv(
        [
            "--aisp-owner-run-id",
            "run-123",
            "--aisp-owner-pid",
            "4242",
        ]
    )

    assert isolated_runner.os.environ["AISP_BENCHMARK_OWNER_RUN_ID"] == "run-123"
    assert isolated_runner.os.environ["AISP_BENCHMARK_OWNER_PID"] == "4242"
