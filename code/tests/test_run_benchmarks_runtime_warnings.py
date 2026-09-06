from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.harness import run_benchmarks


class _ExplodingCloseHandle:
    def close(self) -> None:
        raise RuntimeError("close boom")


def test_benchmark_event_logger_close_warns_when_close_fails(tmp_path, caplog) -> None:
    logger = logging.getLogger("test.run_benchmarks_event_logger")
    logger.handlers = []
    logger.propagate = True
    event_logger = run_benchmarks.BenchmarkEventLogger(
        tmp_path / "events.jsonl",
        run_id="run-123",
        logger=logger,
    )
    event_logger._fh.close()
    event_logger._fh = _ExplodingCloseHandle()

    caplog.set_level(logging.WARNING)

    event_logger.close()

    assert any("Failed to close benchmark event log" in rec.message for rec in caplog.records)
    assert any("close boom" in rec.message for rec in caplog.records)


def test_reset_cuda_state_logs_cleanup_failures(caplog) -> None:
    class _ExplodingGenerator:
        def set_offset(self, _value: int) -> None:
            raise RuntimeError("rng reset failed")

        def manual_seed(self, _value: int) -> None:
            return None

    class _ExplodingStream:
        def synchronize(self) -> None:
            raise RuntimeError("stream sync failed")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: None,
            empty_cache=lambda: None,
            reset_peak_memory_stats=lambda: None,
            reset_accumulated_memory_stats=lambda: None,
            graph_pool_trim=lambda: (_ for _ in ()).throw(RuntimeError("graph reset failed")),
            current_device=lambda: 0,
            default_generators=[_ExplodingGenerator()],
            current_stream=lambda: _ExplodingStream(),
        ),
        initial_seed=lambda: 1234,
        _dynamo=SimpleNamespace(reset=lambda: (_ for _ in ()).throw(RuntimeError("dynamo reset failed"))),
        _inductor=SimpleNamespace(
            cudagraph_trees=SimpleNamespace(
                reset_cudagraph_trees=lambda: (_ for _ in ()).throw(RuntimeError("cudagraph reset failed"))
            )
        ),
    )
    fake_triton = SimpleNamespace(
        runtime=SimpleNamespace(
            cache=SimpleNamespace(clear=lambda: (_ for _ in ()).throw(RuntimeError("triton cache failed")))
        )
    )

    caplog.set_level(logging.WARNING)

    with patch.object(run_benchmarks, "torch", fake_torch), patch.dict(sys.modules, {"triton": fake_triton}):
        run_benchmarks.reset_cuda_state()

    messages = [record.message for record in caplog.records]
    assert any("Failed to reset CUDA graph pool: graph reset failed" in message for message in messages)
    assert any("Failed to reset CUDA RNG state: rng reset failed" in message for message in messages)
    assert any("Failed to synchronize default CUDA stream: stream sync failed" in message for message in messages)
    assert any("Failed to reset torch._dynamo state: dynamo reset failed" in message for message in messages)
    assert any(
        "Failed to reset torch._inductor cudagraph trees: cudagraph reset failed" in message
        for message in messages
    )
    assert any("Failed to clear Triton runtime cache: triton cache failed" in message for message in messages)


def test_reset_cuda_state_rotates_private_triton_cache_when_clear_api_missing(monkeypatch, caplog) -> None:
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

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    fake_cache_knobs = _FakeCacheKnobs()
    fake_triton = SimpleNamespace(
        runtime=SimpleNamespace(cache=SimpleNamespace()),
        knobs=SimpleNamespace(cache=fake_cache_knobs),
    )

    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_OVERRIDE_DIR", raising=False)
    monkeypatch.delenv("TRITON_DUMP_DIR", raising=False)
    caplog.set_level(logging.WARNING)

    with patch.object(run_benchmarks, "torch", fake_torch), patch.dict(sys.modules, {"triton": fake_triton}):
        run_benchmarks.reset_cuda_state()

    assert fake_cache_knobs.reset_calls == 1
    assert Path(os.environ["TRITON_CACHE_DIR"]).exists()
    assert Path(os.environ["TRITON_OVERRIDE_DIR"]).exists()
    assert Path(os.environ["TRITON_DUMP_DIR"]).exists()
    assert not any("Triton runtime cache" in rec.message for rec in caplog.records)


def test_reset_cuda_state_skips_cuda_calls_when_allow_cuda_context_is_false(monkeypatch) -> None:
    calls: list[str] = []

    class _ForbiddenCuda:
        def __getattr__(self, name: str):
            raise AssertionError(f"CUDA method {name} should not be touched when allow_cuda_context=False")

    fake_torch = SimpleNamespace(
        cuda=_ForbiddenCuda(),
        _dynamo=SimpleNamespace(reset=lambda: calls.append("dynamo")),
        _inductor=SimpleNamespace(
            cudagraph_trees=SimpleNamespace(reset_cudagraph_trees=lambda: calls.append("inductor"))
        ),
    )
    fake_triton = SimpleNamespace(runtime=SimpleNamespace(cache=SimpleNamespace(clear=lambda: calls.append("triton-clear"))))

    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_OVERRIDE_DIR", raising=False)
    monkeypatch.delenv("TRITON_DUMP_DIR", raising=False)

    with patch.object(run_benchmarks, "torch", fake_torch), patch.dict(sys.modules, {"triton": fake_triton}):
        run_benchmarks.reset_cuda_state(allow_cuda_context=False)

    assert "dynamo" in calls
    assert "inductor" in calls
    assert "triton-clear" in calls


def test_clean_build_directories_warns_when_build_root_listing_fails(tmp_path, caplog) -> None:
    chapter_dir = tmp_path / "chapter"
    build_root = chapter_dir / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    original_iterdir = Path.iterdir

    def _iterdir_side_effect(path_obj: Path):
        if path_obj == build_root:
            raise OSError("iterdir boom")
        return original_iterdir(path_obj)

    caplog.set_level(logging.WARNING)

    with patch.object(Path, "iterdir", autospec=True, side_effect=_iterdir_side_effect):
        run_benchmarks.clean_build_directories(chapter_dir)

    assert any("Failed to inspect build directory children under" in rec.message for rec in caplog.records)
    assert any("iterdir boom" in rec.message for rec in caplog.records)


def test_harden_profile_env_warns_when_site_discovery_fails(tmp_path, caplog) -> None:
    import site

    repo_root = tmp_path / "repo"
    chapter_dir = repo_root / "ch03"
    repo_root.mkdir()

    caplog.set_level(logging.WARNING)

    with patch.object(site, "getusersitepackages", side_effect=RuntimeError("user site boom")):
        env = run_benchmarks._harden_profile_env(
            {"AISP_PROFILE_NO_USER_SITE": "1"},
            repo_root=repo_root,
            chapter_dir=chapter_dir,
        )

    pythonpath_entries = [entry for entry in env["PYTHONPATH"].split(os.pathsep) if entry]

    assert pythonpath_entries
    assert pythonpath_entries[0].endswith("aisp_profile_python_startup")
    assert str(repo_root) in pythonpath_entries
    assert str(chapter_dir) in pythonpath_entries
    assert any(
        "Failed to discover Python site-packages while hardening profiler environment: user site boom"
        in rec.message
        for rec in caplog.records
    )


def test_harden_profile_env_keeps_site_packages_after_stdlib(tmp_path) -> None:
    """A selected runtime must keep stdlib precedence and its native packages."""

    user_base = tmp_path / "user-base"
    driver_env = os.environ.copy()
    driver_env["PYTHONUSERBASE"] = str(user_base)
    driver_env.pop("PYTHONNOUSERSITE", None)
    site_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import site; print(site.getusersitepackages())",
        ],
        env=driver_env,
        check=True,
        capture_output=True,
        text=True,
    )
    hostile_site = Path(site_probe.stdout.strip())
    hostile_site.mkdir(parents=True, exist_ok=True)
    (hostile_site / "pathlib.py").write_text(
        "raise RuntimeError('third-party pathlib shadowed the standard library')\n",
        encoding="utf-8",
    )

    explicit_dep = tmp_path / "explicit-dependency"
    explicit_dep.mkdir()
    (explicit_dep / "profile_dependency.py").write_text(
        "VALUE = 'explicit-import-ok'\n",
        encoding="utf-8",
    )
    repo_root = Path(run_benchmarks.__file__).resolve().parents[2]
    chapter_dir = tmp_path / "chapter"
    chapter_dir.mkdir()
    driver_env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """\
                import json, os, subprocess, sys
                from pathlib import Path
                from core.harness import run_benchmarks

                repo_root, chapter_dir, explicit_dep, hostile_site = map(Path, sys.argv[1:])
                base_env = os.environ.copy()
                base_env["PYTHONPATH"] = str(explicit_dep)
                env = run_benchmarks._harden_profile_env(
                    base_env,
                    repo_root=repo_root,
                    chapter_dir=chapter_dir,
                )
                child = subprocess.run(
                    [sys.executable, "-c", (
                        "from pathlib import Path; import profile_dependency, torch; "
                        "print(Path.__module__, profile_dependency.VALUE, torch.__version__)"
                    )],
                    env=env, capture_output=True, text=True, check=False,
                )
                entries = env["PYTHONPATH"].split(os.pathsep)
                print(json.dumps({
                    "child_returncode": child.returncode,
                    "child_stdout": child.stdout.strip(),
                    "child_stderr": child.stderr.strip(),
                    "explicit_dependency_retained": str(explicit_dep) in entries,
                    "hostile_site_promoted": str(hostile_site) in entries,
                }, sort_keys=True))
                """
            ),
            str(repo_root),
            str(chapter_dir),
            str(explicit_dep),
            str(hostile_site),
        ],
        env=driver_env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert payload["child_returncode"] == 0, payload["child_stderr"]
    assert payload["child_stdout"].startswith("pathlib explicit-import-ok ")
    assert payload["explicit_dependency_retained"] is True
    assert payload["hostile_site_promoted"] is False
