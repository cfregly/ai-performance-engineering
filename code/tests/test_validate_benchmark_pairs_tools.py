from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from core.benchmark.verification import InputSignature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark
from core.scripts.ci.check_verification_compliance import check_file_compliance
from core.scripts.validate_benchmark_pairs import (
    discover_benchmark_pairs,
    get_input_signature_safe,
    validate_all_pairs,
    validate_pair,
)

_DISTRIBUTED_ENV_KEYS = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")


def _subprocess_env_without_distributed() -> dict[str, str]:
    env = os.environ.copy()
    for key in _DISTRIBUTED_ENV_KEYS:
        env.pop(key, None)
    return env


class _DummyPayloadBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None

    def setup(self) -> None:
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, device=self.device)
        self.y = None

    def benchmark_fn(self) -> None:
        if self.x is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        self.y = self.x * 2

    def capture_verification_payload(self) -> None:
        if self.x is None or self.y is None:
            raise RuntimeError(
                "setup() and benchmark_fn() must run before capture_verification_payload()"
            )
        self._set_verification_payload(
            inputs={"x": self.x},
            output=self.y,
            batch_size=int(self.x.shape[0]),
            parameter_count=0,
            output_tolerance=(0.0, 0.0),
        )


def test_get_input_signature_safe_executes_payload_path() -> None:
    bench = _DummyPayloadBenchmark()
    sig, err = get_input_signature_safe(bench)
    assert err is None
    assert isinstance(sig, InputSignature)


def test_signature_capture_requires_distributed_worker_payload() -> None:
    from core.harness.benchmark_harness import BenchmarkConfig, LaunchVia

    class DistributedPayload(_DummyPayloadBenchmark):
        def get_config(self):
            return BenchmarkConfig(launch_via=LaunchVia.TORCHRUN, nproc_per_node=2)

        def benchmark_fn(self):
            raise AssertionError("standalone validation must not run a parent proxy")

    signature, error = get_input_signature_safe(DistributedPayload())
    assert signature is None
    assert error is not None and "SKIPPED:" in error
    assert "requires a torchrun worker result" in error
    assert "aisp bench run" in error


def test_discover_benchmark_pairs_matches_optimized_variants(tmp_path: Path) -> None:
    ch_dir = tmp_path / "ch01"
    ch_dir.mkdir(parents=True)
    (ch_dir / "baseline_foo.py").write_text("# baseline\n")
    (ch_dir / "optimized_foo.py").write_text("# optimized\n")
    (ch_dir / "optimized_foo_bar.py").write_text("# optimized variant\n")

    pairs = discover_benchmark_pairs(tmp_path, chapter="ch01")
    assert "ch01:foo" in pairs
    assert "ch01:foo_bar" in pairs
    assert pairs["ch01:foo_bar"]["baseline"] == ch_dir / "baseline_foo.py"
    assert pairs["ch01:foo_bar"]["optimized"] == ch_dir / "optimized_foo_bar.py"


def test_ci_compliance_checker_accepts_payload_mixin_inheritance(tmp_path: Path) -> None:
    file_path = tmp_path / "baseline_dummy.py"
    file_path.write_text(
        "from core.benchmark.verification_mixin import VerificationPayloadMixin\n"
        "from core.harness.benchmark_harness import BaseBenchmark\n"
        "class Dummy(VerificationPayloadMixin, BaseBenchmark):\n"
        "    def benchmark_fn(self):\n"
        "        return None\n"
    )
    issues = check_file_compliance(file_path)
    assert not any(
        "get_input_signature" in issue.message and issue.severity == "error" for issue in issues
    )


def test_ci_compliance_checker_accepts_local_benchmark_base_inheritance(tmp_path: Path) -> None:
    file_path = tmp_path / "baseline_dummy.py"
    file_path.write_text(
        "from core.benchmark.verification_mixin import VerificationPayloadMixin\n"
        "from core.harness.benchmark_harness import BaseBenchmark\n"
        "class ParentBenchmark(VerificationPayloadMixin, BaseBenchmark):\n"
        "    def benchmark_fn(self):\n"
        "        return None\n"
        "class ChildBenchmark(ParentBenchmark):\n"
        "    def benchmark_fn(self):\n"
        "        return None\n"
    )
    issues = check_file_compliance(file_path)
    assert not any("get_input_signature" in issue.message for issue in issues)


def test_ci_compliance_checker_accepts_imported_benchmark_base_inheritance(tmp_path: Path) -> None:
    (tmp_path / "base_module.py").write_text(
        "from core.benchmark.verification_mixin import VerificationPayloadMixin\n"
        "from core.harness.benchmark_harness import BaseBenchmark\n"
        "class ImportedParentBenchmark(VerificationPayloadMixin, BaseBenchmark):\n"
        "    allow_cpu = True\n"
        "    def benchmark_fn(self):\n"
        "        return None\n"
    )
    file_path = tmp_path / "baseline_dummy.py"
    file_path.write_text(
        "from base_module import ImportedParentBenchmark\n"
        "class ChildBenchmark(ImportedParentBenchmark):\n"
        "    def benchmark_fn(self):\n"
        "        return None\n"
    )
    issues = check_file_compliance(file_path)
    assert not any("get_input_signature" in issue.message for issue in issues)


def test_ch04_torchrun_multigpu_pairs_skip_cleanly_on_single_gpu_hosts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cases = [
        (
            "pipeline_parallel_multigpu_1f1b",
            "baseline_pipeline_parallel_multigpu.py",
            "optimized_pipeline_parallel_multigpu_1f1b.py",
        ),
        (
            "tensor_parallel_allgather_multigpu",
            "baseline_tensor_parallel_allgather_multigpu.py",
            "optimized_tensor_parallel_allgather_multigpu.py",
        ),
        (
            "tensor_parallel_multigpu",
            "baseline_tensor_parallel_multigpu.py",
            "optimized_tensor_parallel_multigpu.py",
        ),
        (
            "torchcomms_multigpu",
            "baseline_torchcomms_multigpu.py",
            "optimized_torchcomms_multigpu.py",
        ),
    ]

    for example_name, baseline_name, optimized_name in cases:
        result = validate_pair(
            "ch04",
            example_name,
            repo_root / "ch04" / baseline_name,
            repo_root / "ch04" / optimized_name,
        )
        if torch.cuda.device_count() < 2:
            assert result.skipped is True
            assert result.error is not None and "SKIPPED" in result.error
        elif result.skipped:
            assert result.error is not None and "SKIPPED" in result.error
        else:
            assert result.valid is True
            assert result.error is None


def test_ch04_no_overlap_pair_requires_distributed_launch_context() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = validate_pair(
        "ch04",
        "no_overlap",
        repo_root / "ch04" / "baseline_no_overlap.py",
        repo_root / "ch04" / "optimized_no_overlap.py",
    )

    if torch.cuda.device_count() < 2:
        assert result.skipped is True
        assert result.error is not None and "SKIPPED" in result.error
    else:
        assert result.skipped is True
        assert result.error is not None
        assert "torchrun/distributed launch context" in result.error


def test_ch04_no_overlap_torchrun_specs_use_public_target_names() -> None:
    from ch04.ddp_no_overlap import BaselineNoOverlapBenchmark
    from ch04.ddp_overlap import OptimizedOverlapDdpBenchmark

    baseline_spec = BaselineNoOverlapBenchmark().get_torchrun_spec()
    optimized_spec = OptimizedOverlapDdpBenchmark().get_torchrun_spec()

    worker_path = (Path(__file__).resolve().parents[1] / "ch04" / "ddp_worker.py").resolve()
    assert baseline_spec.script_path is not None
    assert Path(baseline_spec.script_path).resolve() == worker_path
    assert baseline_spec.script_args == ["--variant", "no-overlap"]
    assert baseline_spec.name == "baseline_no_overlap"
    assert baseline_spec.config_arg_map == {"iterations": "--iterations", "warmup": "--warmup"}
    assert optimized_spec.script_path is not None
    assert Path(optimized_spec.script_path).resolve() == worker_path
    assert optimized_spec.script_args == ["--variant", "overlap"]
    assert optimized_spec.name == "optimized_overlap"
    assert optimized_spec.config_arg_map == {"iterations": "--iterations", "warmup": "--warmup"}


def test_ch04_additional_torchrun_specs_build_without_side_effects() -> None:
    cases = [
        (
            "ch04.baseline_nvshmem_training_example_multigpu",
            "NVSHMEMTrainingExampleMultiGPU",
        ),
        (
            "ch04.optimized_nvshmem_training_example_multigpu",
            "OptimizedNVSHMEMTrainingExampleMultiGPU",
        ),
        (
            "ch04.baseline_nvshmem_training_patterns_multigpu",
            "NVSHMEMTrainingPatternsMultiGPU",
        ),
        (
            "ch04.optimized_nvshmem_training_patterns_multigpu",
            "OptimizedNVSHMEMTrainingPatternsMultiGPU",
        ),
        (
            "ch04.baseline_nvshmem_pipeline_parallel_multigpu",
            "NVSHMEMPipelineParallelMultiGPU",
        ),
        (
            "ch04.optimized_nvshmem_pipeline_parallel_multigpu",
            "OptimizedNVSHMEMPipelineParallelMultiGPU",
        ),
        (
            "ch04.baseline_nvshmem_vs_nccl_benchmark_multigpu",
            "NVSHMEMVsNCCLBenchmarkMultiGPU",
        ),
        (
            "ch04.optimized_nvshmem_vs_nccl_benchmark_multigpu",
            "OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU",
        ),
        (
            "ch04.baseline_symmetric_memory_multigpu",
            "SymmetricMemoryMultiGPU",
        ),
        (
            "ch04.optimized_symmetric_memory_multigpu",
            "OptimizedSymmetricMemoryMultiGPU",
        ),
    ]

    for module_name, class_name in cases:
        module = __import__(module_name, fromlist=[class_name])
        bench_cls = getattr(module, class_name)
        bench = bench_cls()
        config = bench.get_config()
        config.nproc_per_node = 2
        config.nnodes = 1
        spec = bench.get_torchrun_spec(config)
        try:
            assert spec.script_path is None
            assert spec.module_name == "core.harness.benchmark_worker"
            assert spec.script_args[:5] == [
                "--module",
                "ch04.nvshmem_worker",
                "--callable",
                "main",
                "--",
            ]
            assert spec.config_arg_map == {}
        finally:
            for name, value in spec.env.items():
                if name.endswith("_RESULT_DIR"):
                    shutil.rmtree(value, ignore_errors=True)


def test_ch04_nvshmem_torchrun_specs_preserve_variant_contracts() -> None:
    from ch04.baseline_nvshmem_pipeline_parallel_multigpu import (
        NVSHMEMPipelineParallelMultiGPU,
    )
    from ch04.baseline_nvshmem_training_example_multigpu import (
        NVSHMEMTrainingExampleMultiGPU,
    )
    from ch04.baseline_nvshmem_training_patterns_multigpu import (
        NVSHMEMTrainingPatternsMultiGPU,
    )
    from ch04.baseline_nvshmem_vs_nccl_benchmark_multigpu import (
        NVSHMEMVsNCCLBenchmarkMultiGPU,
    )
    from ch04.optimized_nvshmem_pipeline_parallel_multigpu import (
        OptimizedNVSHMEMPipelineParallelMultiGPU,
    )
    from ch04.optimized_nvshmem_training_example_multigpu import (
        OptimizedNVSHMEMTrainingExampleMultiGPU,
    )
    from ch04.optimized_nvshmem_training_patterns_multigpu import (
        OptimizedNVSHMEMTrainingPatternsMultiGPU,
    )
    from ch04.optimized_nvshmem_vs_nccl_benchmark_multigpu import (
        OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU,
    )

    benchmarks = [
        NVSHMEMPipelineParallelMultiGPU(),
        OptimizedNVSHMEMPipelineParallelMultiGPU(),
        NVSHMEMTrainingExampleMultiGPU(),
        OptimizedNVSHMEMTrainingExampleMultiGPU(),
        NVSHMEMTrainingPatternsMultiGPU(),
        OptimizedNVSHMEMTrainingPatternsMultiGPU(),
        NVSHMEMVsNCCLBenchmarkMultiGPU(),
        OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU(),
    ]
    specs = []
    for benchmark in benchmarks:
        config = benchmark.get_config()
        config.nproc_per_node = 2
        config.nnodes = 1
        specs.append(benchmark.get_torchrun_spec(config))

    try:
        (
            baseline_pipeline,
            optimized_pipeline,
            baseline_training,
            optimized_training,
            baseline_patterns,
            optimized_patterns,
            baseline_collective,
            optimized_collective,
        ) = specs
        adapter_prefix = [
            "--module",
            "ch04.nvshmem_worker",
            "--callable",
            "main",
            "--",
        ]
        assert all(spec.script_args[:5] == adapter_prefix for spec in specs)
        assert baseline_pipeline.script_args[5:9] == [
            "--workload",
            "pipeline",
            "--variant",
            "baseline",
        ]
        assert optimized_pipeline.script_args[5:9] == [
            "--workload",
            "pipeline",
            "--variant",
            "optimized",
        ]
        assert baseline_pipeline.script_args[9:-2] == optimized_pipeline.script_args[9:-2]
        assert baseline_pipeline.script_args[-2:] == ["--transport", "nccl"]
        assert optimized_pipeline.script_args[-2:] == ["--transport", "nvshmem"]

        assert baseline_training.script_args[9:] == optimized_training.script_args[9:]
        assert baseline_training.env["AISP_NVSHMEM_PIPELINE_REUSE_BUFFERS"] == "0"
        assert optimized_training.env["AISP_NVSHMEM_PIPELINE_REUSE_BUFFERS"] == "1"

        assert baseline_patterns.script_args[9:] == optimized_patterns.script_args[9:]
        assert baseline_patterns.env["AISP_GRAD_SYNC_NAIVE"] == "1"
        assert optimized_patterns.env["AISP_GRAD_SYNC_NAIVE"] == "0"

        assert baseline_collective.script_args[-1] == "nccl"
        assert optimized_collective.script_args[-1] == "nvshmem"
        assert baseline_collective.env["AISP_BROADCAST_OVERLAP"] == "0"
        assert optimized_collective.env["AISP_BROADCAST_OVERLAP"] == "1"
    finally:
        for spec in specs:
            for name, value in spec.env.items():
                if name.endswith("_RESULT_DIR"):
                    shutil.rmtree(value, ignore_errors=True)


def test_ch04_ddp_worker_emits_explicit_skips_without_torchrun_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cases = ["no-overlap", "overlap"]

    for variant in cases:
        completed = subprocess.run(
            [sys.executable, "-m", "ch04.ddp_worker", "--variant", variant],
            cwd=repo_root,
            capture_output=True,
            text=True,
            env=_subprocess_env_without_distributed(),
            check=False,
        )
        assert completed.returncode == 3
        combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        assert "SKIPPED:" in combined
        assert "torchrun/distributed launch context" in combined or "Requires >= 2 GPUs" in combined


def test_ch04_nvshmem_workers_emit_explicit_skips_on_unsupported_hosts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cases = [
        ["-m", "ch04.nvshmem_training_example", "--demo", "pipeline"],
        ["-m", "ch04.nvshmem_training_patterns", "--pattern", "gradient"],
        ["-m", "ch04.nvshmem_pipeline_parallel_multigpu", "--schedule", "1f1b"],
        ["-m", "ch04.nvshmem_vs_nccl_benchmark", "--mode", "nccl"],
        [
            "-m",
            "ch04.nvshmem_worker",
            "--workload",
            "pipeline",
            "--variant",
            "baseline",
            "--schedule",
            "1f1b",
        ],
        [
            "-m",
            "ch04.nvshmem_worker",
            "--workload",
            "collective",
            "--variant",
            "optimized",
            "--mode",
            "nvshmem",
        ],
    ]

    for command in cases:
        completed = subprocess.run(
            [sys.executable, *command],
            cwd=repo_root,
            capture_output=True,
            text=True,
            env=_subprocess_env_without_distributed(),
            check=False,
        )
        assert completed.returncode == 3
        combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        assert "SKIPPED:" in combined
        assert (
            "requires >=2 GPUs" in combined
            or "requires >= 2 GPUs" in combined
            or "Requires >= 2 GPUs" in combined
            or "torchrun/distributed launch context" in combined
            or "requires NVSHMEM or SymmetricMemory support" in combined
            or "requires CUDA" in combined
        )


def test_ch04_nvshmem_pairs_skip_cleanly_on_single_gpu_or_missing_symmem() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cases = [
        (
            "nvshmem_training_example",
            "baseline_nvshmem_training_example.py",
            "optimized_nvshmem_training_example.py",
        ),
        (
            "nvshmem_training_patterns",
            "baseline_nvshmem_training_patterns.py",
            "optimized_nvshmem_training_patterns.py",
        ),
    ]

    for example_name, baseline_name, optimized_name in cases:
        result = validate_pair(
            "ch04",
            example_name,
            repo_root / "ch04" / baseline_name,
            repo_root / "ch04" / optimized_name,
        )
        if torch.cuda.device_count() < 2:
            assert result.skipped is True
            assert result.error is not None and "SKIPPED" in result.error
        elif result.skipped:
            assert result.error is not None and "SKIPPED" in result.error
            assert (
                "requires NVSHMEM or SymmetricMemory support" in result.error
                or "requires a torchrun worker result" in result.error
            )
        else:
            assert result.valid, result.to_dict()
            assert result.baseline_has_signature and result.optimized_has_signature
            assert result.signatures_match
            assert not result.mismatches
            assert result.error is None


def test_validate_all_pairs_timeout_kills_isolated_pair_descendants(tmp_path: Path) -> None:
    ch_dir = tmp_path / "ch01"
    ch_dir.mkdir(parents=True)
    pid_file = tmp_path / "spawned_child.pid"
    benchmark_source = f"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark


class HangingPairBenchmark(BaseBenchmark):
    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self._ready = False

    def setup(self) -> None:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        Path(r"{pid_file}").write_text(str(child.pid), encoding="utf-8")

    def benchmark_fn(self) -> None:
        time.sleep(60)
        self._ready = True

    def capture_verification_payload(self) -> None:
        if not self._ready:
            raise RuntimeError("payload not ready")

    def get_input_signature(self):
        raise RuntimeError("force execution path")
"""
    (ch_dir / "baseline_hanging.py").write_text(benchmark_source, encoding="utf-8")
    (ch_dir / "optimized_hanging.py").write_text(benchmark_source, encoding="utf-8")

    # Isolated pair startup includes interpreter + module import overhead on real hosts.
    # Give the validator enough time to reach setup() and spawn the descendant process
    # before the intentional benchmark_fn() hang triggers timeout cleanup.
    report = validate_all_pairs(tmp_path, chapter="ch01", pair_timeout_seconds=8)
    assert report.total_pairs == 1
    assert len(report.results) == 1
    result = report.results[0]
    assert result.valid is False
    assert result.error == "Validation timeout after 8 seconds"

    deadline = time.time() + 2.0
    while not pid_file.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert pid_file.exists()
    child_pid = int(pid_file.read_text(encoding="utf-8").strip())
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(child_pid, signal.SIGKILL)
        raise AssertionError(f"isolated pair child {child_pid} survived timeout cleanup")
