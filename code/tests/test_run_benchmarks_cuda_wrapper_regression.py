import importlib.util
import sys
from pathlib import Path

from core.benchmark.cuda_binary_benchmark import _run_subprocess_capture
from core.harness import run_benchmarks


class _DummyExpectationsStore:
    def __init__(self, chapter_dir: Path, *_args, **_kwargs) -> None:
        self.path = chapter_dir / "expectations_test.json"

    def save(self) -> None:
        return None

    def get_entry(self, _key: str) -> None:
        return None


def test_test_chapter_impl_uses_cuda_wrapper_detector_without_nameerror(tmp_path, monkeypatch):
    chapter_dir = tmp_path / "ch03"
    chapter_dir.mkdir()
    baseline_path = chapter_dir / "baseline_pageable_copy.py"
    baseline_path.write_text(
        "from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark\n"
        "class DemoCudaWrapper(CudaBinaryBenchmark):\n"
        "    cuda_binary_path = 'demo'\n",
        encoding="utf-8",
    )

    detector_calls: list[Path] = []
    original_detector = run_benchmarks.is_cuda_binary_benchmark_file

    def _tracked_detector(path: Path) -> bool:
        detector_calls.append(path)
        return original_detector(path)

    monkeypatch.setattr(run_benchmarks, "is_cuda_binary_benchmark_file", _tracked_detector)
    monkeypatch.setattr(run_benchmarks, "dump_environment_and_capabilities", lambda: None)
    monkeypatch.setattr(run_benchmarks, "detect_expectation_key", lambda: "test")
    monkeypatch.setattr(run_benchmarks, "detect_execution_environment", lambda: {"kind": "test"})
    monkeypatch.setattr(run_benchmarks, "get_git_info", lambda: {"commit": "deadbeef"})
    monkeypatch.setattr(run_benchmarks, "clean_build_directories", lambda _chapter_dir: None)
    monkeypatch.setattr(run_benchmarks, "reset_cuda_state", lambda **_kwargs: None)
    monkeypatch.setattr(run_benchmarks, "reset_gpu_state", lambda: None)
    monkeypatch.setattr(run_benchmarks, "emit_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_benchmarks, "start_progress_watchdog", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(run_benchmarks, "ExpectationsStore", _DummyExpectationsStore)
    monkeypatch.setattr(run_benchmarks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run_benchmarks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        run_benchmarks,
        "_discover_chapter_benchmark_pairs",
        lambda *args, **kwargs: ([(baseline_path, [], "pageable_copy")], [], None, 0, 0),
    )

    result = run_benchmarks._test_chapter_impl(chapter_dir, enable_profiling=False)

    assert detector_calls == [baseline_path]
    assert result["status"] == "completed"
    assert result["summary"]["informational"] == 0
    assert len(result["benchmarks"]) == 1
    assert result["benchmarks"][0]["example"] == "pageable_copy"
    assert result["benchmarks"][0]["baseline_file"] == "baseline_pageable_copy.py"


def test_discover_chapter_benchmark_pairs_recognizes_real_cuda_wrapper_files(tmp_path):
    chapter_dir = tmp_path / "ch03"
    chapter_dir.mkdir()
    benchmark_source = (
        "from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark\n"
        "from core.harness.benchmark_harness import BaseBenchmark\n"
        "class DemoCudaWrapper(CudaBinaryBenchmark):\n"
        "    cuda_binary_path = 'demo'\n"
        "def get_benchmark() -> BaseBenchmark:\n"
        "    raise RuntimeError('not executed in discovery test')\n"
    )
    baseline_path = chapter_dir / "baseline_pageable_copy.py"
    optimized_path = chapter_dir / "optimized_pageable_copy.py"
    baseline_path.write_text(benchmark_source, encoding="utf-8")
    optimized_path.write_text(benchmark_source, encoding="utf-8")

    python_pairs, cuda_pairs, example_filters, suppressed_alias_pairs, suppressed_variant_opts = (
        run_benchmarks._discover_chapter_benchmark_pairs(chapter_dir, only_cuda=True)
    )

    assert example_filters is None
    assert suppressed_alias_pairs == 0
    assert suppressed_variant_opts == 0
    assert cuda_pairs == []
    assert len(python_pairs) == 1
    assert python_pairs[0][2] == "pageable_copy"
    assert python_pairs[0][0] == baseline_path
    assert python_pairs[0][1] == [optimized_path]
    assert run_benchmarks._is_cuda_wrapper(baseline_path) is True


def test_benchmark_cuda_executable_treats_skip_exit_code_as_skip(tmp_path):
    executable = tmp_path / "skip_cuda_binary.sh"
    executable.write_text(
        "#!/bin/sh\n"
        "echo 'SKIPPED: cuBLASLt NVFP4 algorithm unavailable on this driver/toolchain.' >&2\n"
        "exit 3\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    result = run_benchmarks.benchmark_cuda_executable(executable, iterations=1, warmup=0, timeout=5)

    assert result is not None
    assert result.skip_reason == "SKIPPED: cuBLASLt NVFP4 algorithm unavailable on this driver/toolchain."


def test_is_distributed_benchmark_ignores_local_gpu_reduction_named_distributed(tmp_path):
    benchmark_path = tmp_path / "optimized_distributed.py"
    benchmark_path.write_text(
        "import torch\n"
        "\n"
        "class DemoBenchmark:\n"
        "    def benchmark_fn(self):\n"
        "        data = torch.randn(1024, device='cuda')\n"
        "        return data.sum()\n",
        encoding="utf-8",
    )

    assert run_benchmarks.is_distributed_benchmark(benchmark_path) is False


def test_is_distributed_benchmark_respects_explicit_single_gpu_torchrun_override(tmp_path):
    benchmark_path = tmp_path / "baseline_tensor_parallel.py"
    benchmark_path.write_text(
        "import torch.distributed as dist\n"
        "from core.harness.benchmark_harness import BenchmarkConfig, LaunchVia, TorchrunLaunchSpec\n"
        "\n"
        "class DemoBenchmark:\n"
        "    def get_config(self):\n"
        "        return BenchmarkConfig(launch_via=LaunchVia.TORCHRUN, nproc_per_node=1, multi_gpu_required=False)\n"
        "\n"
        "    def get_torchrun_spec(self, config=None):\n"
        "        return TorchrunLaunchSpec(script_path=__file__, script_args=[], multi_gpu_required=False, name='demo')\n"
        "\n"
        "def run():\n"
        "    dist.init_process_group(backend='nccl')\n",
        encoding="utf-8",
    )

    assert run_benchmarks.is_distributed_benchmark(benchmark_path) is False


def test_ch04_pipeline_and_tensor_parallel_benchmarks_are_classified_as_distributed():
    repo_root = Path(__file__).resolve().parents[1]
    cases = [
        repo_root / "ch04" / "baseline_pipeline_parallel.py",
        repo_root / "ch04" / "optimized_pipeline_parallel_1f1b.py",
        repo_root / "ch04" / "baseline_tensor_parallel.py",
        repo_root / "ch04" / "optimized_tensor_parallel_async.py",
    ]

    for benchmark_path in cases:
        assert run_benchmarks.is_distributed_benchmark(benchmark_path) is True


def test_ch04_torchcomms_benchmarks_keep_single_gpu_torchrun_override():
    repo_root = Path(__file__).resolve().parents[1]
    cases = [
        repo_root / "ch04" / "baseline_torchcomms.py",
        repo_root / "ch04" / "optimized_torchcomms.py",
    ]

    for benchmark_path in cases:
        assert run_benchmarks.is_distributed_benchmark(benchmark_path) is False


def test_ch04_nixl_tier_handoff_benchmarks_are_single_gpu():
    repo_root = Path(__file__).resolve().parents[1]
    cases = [
        repo_root / "ch04" / "baseline_nixl_tier_handoff.py",
        repo_root / "ch04" / "optimized_nixl_tier_handoff.py",
    ]
    for benchmark_path in cases:
        assert run_benchmarks.is_distributed_benchmark(benchmark_path) is False


def test_ch04_pipeline_and_tensor_parallel_benchmarks_require_multi_gpu_in_config():
    repo_root = Path(__file__).resolve().parents[1]
    cases = [
        repo_root / "ch04" / "baseline_pipeline_parallel.py",
        repo_root / "ch04" / "optimized_pipeline_parallel_1f1b.py",
        repo_root / "ch04" / "baseline_tensor_parallel.py",
        repo_root / "ch04" / "optimized_tensor_parallel_async.py",
    ]

    for benchmark_path in cases:
        spec = importlib.util.spec_from_file_location(benchmark_path.stem, benchmark_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        benchmark = module.get_benchmark()
        config = benchmark.get_config()
        assert config.launch_via.name == "TORCHRUN"
        assert config.multi_gpu_required is True


def test_append_profile_warning_persists_message_to_stderr_log(tmp_path, monkeypatch):
    log_path = tmp_path / "profile.stderr.log"
    logger_messages: list[str] = []

    monkeypatch.setattr(run_benchmarks, "LOGGER_AVAILABLE", True)
    monkeypatch.setattr(
        run_benchmarks.logger,
        "warning",
        lambda message, *args, **_kwargs: logger_messages.append(message % args if args else message),
    )

    run_benchmarks._append_profile_warning(log_path, "profiler cleanup detail")

    assert "profiler cleanup detail" in log_path.read_text(encoding="utf-8")
    assert any("profiler cleanup detail" in message for message in logger_messages)


def test_cuda_binary_subprocess_capture_propagates_owner_markers(tmp_path, monkeypatch):
    monkeypatch.setenv("AISP_BENCHMARK_OWNER_RUN_ID", "owner-run-123")
    monkeypatch.setenv("AISP_BENCHMARK_OWNER_PID", "4242")

    result = _run_subprocess_capture(
        [
            sys.executable,
            "-c",
            (
                "import os\n"
                "print(os.environ.get('AISP_BENCHMARK_OWNER_RUN_ID', ''))\n"
                "print(os.environ.get('AISP_BENCHMARK_OWNER_PID', ''))\n"
            ),
        ],
        cwd=tmp_path,
        timeout=5,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["owner-run-123", "4242"]
