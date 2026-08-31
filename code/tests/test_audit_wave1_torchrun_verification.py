"""LOCAL-019: genuine launch control flow, never synthetic training verification."""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, LaunchVia
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark
from labs.train_distributed.training_utils.zero2_torchrun_benchmark import Zero2TorchrunBenchmark
from tests.protection_test_utils import preserve_rng_state


REASON = "actual child-training verification is unsupported"


@pytest.fixture(autouse=True)
def restore_rng():
    with preserve_rng_state():
        yield


def cpu_harness():
    config = BenchmarkConfig(
        device=torch.device("cpu"), iterations=1, warmup=5,
        launch_via=LaunchVia.TORCHRUN, nproc_per_node=1, multi_gpu_required=False,
        use_subprocess=False, enable_profiling=False, lock_gpu_clocks=False,
        enforce_environment_validation=False, measurement_timeout_seconds=60,
        nnodes="1", rdzv_backend="c10d", rdzv_endpoint="127.0.0.1:0",
    )
    return BenchmarkHarness(config=config), config


@pytest.mark.parametrize("method", [
    "setup", "benchmark_fn", "capture_verification_payload", "_prepare_verification_payload",
    "get_verify_inputs", "get_verify_output", "get_input_signature", "get_output_tolerance",
    "get_torchrun_spec",
])
@pytest.mark.parametrize("stale_payload", [False, True])
def test_generic_training_verification_is_unsupported(tmp_path, method, stale_payload):
    benchmark = TorchrunScriptBenchmark(
        script_path=tmp_path / "training.py", multi_gpu_required=False, default_nproc_per_node=1,
    )
    # Execute only the old/new wrapper's CPU host mechanisms, never fake CUDA.
    benchmark.device = torch.device("cpu")
    if stale_payload:
        benchmark._subprocess_verify_output = torch.ones(1)
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match=REASON):
        getattr(benchmark, method)()
    assert torch.equal(torch.get_rng_state(), before)
    assert benchmark._verification_payload is None


def test_generic_validate_result_never_accepts_a_surrogate(tmp_path):
    benchmark = TorchrunScriptBenchmark(script_path=tmp_path / "training.py", multi_gpu_required=False)
    assert REASON in benchmark.validate_result()
    benchmark._output = torch.ones(1)
    benchmark._subprocess_verify_output = benchmark._output
    assert REASON in benchmark.validate_result()


def test_launch_configuration_remains_discoverable(tmp_path):
    benchmark = TorchrunScriptBenchmark(
        script_path=tmp_path / "training.py", base_args=["--mode", "baseline"],
        target_label="training:example", config_arg_map={"iterations": "--steps"},
        multi_gpu_required=False, default_nproc_per_node=1, default_iterations=7,
        measurement_timeout_seconds=123, env={"TRAINING_TEST": "1"}, name="training-example",
    )
    config = benchmark.get_config()
    assert config.launch_via == LaunchVia.TORCHRUN
    assert config.nproc_per_node == 1 and not config.multi_gpu_required
    assert config.iterations == 7 and config.measurement_timeout_seconds == 123
    assert config.target_label == "training:example"
    assert benchmark.name == "training-example" and benchmark._device is None
    benchmark.teardown()
    assert benchmark._verification_payload is None


@pytest.mark.parametrize("name", ["baseline_ddp", "optimized_ddp"])
def test_real_training_factories_remain_discoverable_but_unverified(name):
    module = importlib.import_module(f"labs.train_distributed.{name}")
    benchmark = module.get_benchmark()
    assert isinstance(benchmark, TorchrunScriptBenchmark)
    assert benchmark._script_path.is_file()
    assert benchmark._base_args and benchmark._config_arg_map
    with pytest.raises(RuntimeError, match=REASON):
        benchmark.get_torchrun_spec()


@pytest.mark.parametrize(
    ("name", "mode", "variant"),
    [
        ("baseline_zero2", "baseline", "single"),
        ("optimized_zero2", "optimized", "single"),
        ("baseline_zero2_multigpu", "baseline", "multigpu"),
        ("optimized_zero2_multigpu", "optimized", "multigpu"),
    ],
)
def test_only_zero2_factories_select_the_child_result_adapter(name, mode, variant):
    module = importlib.import_module(f"labs.train_distributed.{name}")
    benchmark = module.get_benchmark()
    assert isinstance(benchmark, Zero2TorchrunBenchmark)
    assert isinstance(benchmark, TorchrunScriptBenchmark)
    assert benchmark._zero2_mode == mode
    assert benchmark._zero2_variant == variant
    assert benchmark._script_path.name == "zero2.py"


def test_zero2_single_spec_opts_into_fresh_result_callback(capsys):
    module = importlib.import_module("labs.train_distributed.baseline_zero2")
    benchmark = module.get_benchmark()
    config = benchmark.get_config()
    spec = benchmark.get_torchrun_spec(config)
    assert spec.result_callback == "consume_zero2_child_results"
    assert spec.env["AISP_ZERO2_RESULT_MODE"] == "baseline"
    assert spec.env["AISP_ZERO2_RESULT_VARIANT"] == "single"
    assert spec.env["AISP_ZERO2_PROFILE_KIND"] == "post-timing-correctness"
    result_dir = Path(spec.env["AISP_ZERO2_RESULT_DIR"])
    assert result_dir.is_dir() and not list(result_dir.iterdir())
    assert benchmark._zero2_result_context["retention"] == {
        "policy": "delete-after-success-retain-failure",
        "status": "pending-child-result",
        "path": str(result_dir),
    }
    assert str(result_dir) in benchmark.validate_result()
    benchmark.teardown()
    assert str(result_dir) in capsys.readouterr().out
    result_dir.rmdir()


def test_zero2_harness_rejects_verification_only_override_before_spawn(monkeypatch):
    module = importlib.import_module("labs.train_distributed.baseline_zero2")
    benchmark = module.get_benchmark()
    config = benchmark.get_config()
    config.target_extra_args = {
        "labs/train_distributed:zero2": [
            "--verification-only",
            "--verification-backend",
            "gloo",
        ]
    }
    launches = []

    def forbidden_spawn(*args, **kwargs):
        launches.append(args)
        raise AssertionError("reserved override reached subprocess launch")

    monkeypatch.setattr("core.harness.benchmark_harness.subprocess.Popen", forbidden_spawn)
    with pytest.raises(RuntimeError, match="reserved control '--verification-only'"):
        BenchmarkHarness(config=config)._benchmark_with_torchrun(benchmark, config)
    assert not launches
    assert benchmark._zero2_result_context is None


@pytest.mark.parametrize("override", ["--compile", "--comp"])
def test_zero2_harness_rejects_compile_override_before_artifact_creation(override):
    module = importlib.import_module("labs.train_distributed.baseline_zero2")
    benchmark = module.get_benchmark()
    config = benchmark.get_config()
    config.target_extra_args = {
        "labs/train_distributed:zero2": [override],
    }
    with pytest.raises(RuntimeError, match="reserved control '--compile'"):
        benchmark.get_torchrun_spec(config)
    assert benchmark._zero2_result_context is None


def test_zero2_rejects_a_second_unconsumed_result_context():
    module = importlib.import_module("labs.train_distributed.baseline_zero2")
    benchmark = module.get_benchmark()
    config = benchmark.get_config()
    first_spec = benchmark.get_torchrun_spec(config)
    first_dir = Path(first_spec.env["AISP_ZERO2_RESULT_DIR"])
    first_context = benchmark._zero2_result_context
    with pytest.raises(RuntimeError, match="unconsumed ZeRO child-result context"):
        benchmark.get_torchrun_spec(config)
    assert benchmark._zero2_result_context is first_context
    assert first_dir.is_dir() and not list(first_dir.iterdir())
    first_dir.rmdir()


def test_zero2_single_rejects_multi_rank_configuration_before_artifact_creation():
    module = importlib.import_module("labs.train_distributed.baseline_zero2")
    benchmark = module.get_benchmark()
    config = benchmark.get_config()
    config.nproc_per_node = 2
    with pytest.raises(RuntimeError, match="single child verification requires world_size == 1"):
        benchmark.get_torchrun_spec(config)
    assert benchmark._zero2_result_context is None


def test_zero2_local_transport_rejects_multinode_configuration():
    module = importlib.import_module("labs.train_distributed.baseline_zero2")
    benchmark = module.get_benchmark()
    config = benchmark.get_config()
    config.nnodes = "2"
    with pytest.raises(RuntimeError, match="requires nnodes == 1"):
        benchmark.get_torchrun_spec(config)
    assert benchmark._zero2_result_context is None


@pytest.mark.parametrize("error", [ValueError("invalid declared launch spec"), RuntimeError("SKIPPED: unverified child")])
def test_declared_spec_error_propagates_before_spawn(monkeypatch, error):
    class BrokenSpec(BaseBenchmark):
        def get_torchrun_spec(self, config=None):
            raise error

    harness, config = cpu_harness()
    launches = []

    def forbidden_spawn(*args, **kwargs):
        launches.append(args)
        raise AssertionError("unexpected spawn after declared spec error")

    monkeypatch.setattr("core.harness.benchmark_harness.subprocess.Popen", forbidden_spawn)
    with pytest.raises(type(error)) as caught:
        try:
            harness._benchmark_with_torchrun(BrokenSpec(), config)
        finally:
            assert not launches, "declared spec error reached subprocess launch"
    assert caught.value is error


def test_generic_torchrun_harness_rejects_before_spawn(tmp_path, monkeypatch):
    benchmark = TorchrunScriptBenchmark(
        script_path=tmp_path / "training.py", multi_gpu_required=False, default_nproc_per_node=1,
    )
    benchmark.device = torch.device("cpu")
    harness, config = cpu_harness()
    launches = []

    def forbidden_spawn(*args, **kwargs):
        launches.append(args)
        raise AssertionError("unverified generic training must not launch")

    monkeypatch.setattr("core.harness.benchmark_harness.subprocess.Popen", forbidden_spawn)
    with pytest.raises(RuntimeError, match=REASON):
        harness._benchmark_with_torchrun(benchmark, config)
    assert not launches


@pytest.mark.parametrize("value", [None, 7])
def test_noncallable_declared_spec_cannot_select_fallback(value, monkeypatch):
    benchmark = BaseBenchmark()
    benchmark.get_torchrun_spec = value
    harness, config = cpu_harness()
    launches = []

    def forbidden_spawn(*args, **kwargs):
        launches.append(args)
        raise AssertionError("noncallable spec must not launch")

    monkeypatch.setattr("core.harness.benchmark_harness.subprocess.Popen", forbidden_spawn)
    with pytest.raises(TypeError, match="callable get_torchrun_spec"):
        harness._benchmark_with_torchrun(benchmark, config)
    assert not launches


def test_explicit_none_spec_selects_fallback_script_before_launcher_error(monkeypatch):
    """Observe a real command construction and an explicit failed spawn, not fake success."""
    class DefaultSpec(BaseBenchmark):
        def get_torchrun_spec(self, config=None):
            return None

    harness, config = cpu_harness()
    launches = []

    def unavailable_launcher(command, **kwargs):
        launches.append(command)
        raise OSError("LOCAL019 launcher unavailable control")

    monkeypatch.setattr("core.harness.benchmark_harness.subprocess.Popen", unavailable_launcher)
    with pytest.raises(OSError, match="LOCAL019 launcher unavailable control"):
        harness._benchmark_with_torchrun(DefaultSpec(), config)
    assert len(launches) == 1
    command = launches[0]
    assert command[command.index("--aisp-target-script") + 1] == str(Path(__file__).resolve())


def test_real_direct_child_wrapper_still_executes_cpu_script(tmp_path):
    """Run the real child wrapper directly; this is launcher evidence, not training evidence."""
    target = tmp_path / "cpu_child.py"
    marker = tmp_path / "executed.json"
    target.write_text(
        "import json, os\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(json.dumps({{'pid':os.getpid(),'sum':sum(range(10))}}))\n"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # Explicit CPU child workload, no GPU evidence.
    result = subprocess.run(
        [sys.executable, "-m", "core.harness.torchrun_wrapper", "--aisp-expected-torch-seed", "42",
         "--aisp-target-script", str(target)],
        cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    child = json.loads(marker.read_text())
    assert child["pid"] != os.getpid() and child["sum"] == 45


def test_explicit_none_spec_retains_real_cpu_module_launch(tmp_path, monkeypatch):
    """Actual child process receipt proves launcher fallback only, not training correctness."""
    marker = tmp_path / "child.json"
    source = tmp_path / "local019_launch_control.py"
    source.write_text(
        "from core.harness.benchmark_harness import BaseBenchmark\n"
        "class DefaultSpec(BaseBenchmark):\n"
        "    def get_torchrun_spec(self, config=None): return None\n"
        "if __name__ == '__main__':\n"
        "    import json, os, torch\n"
        "    from pathlib import Path\n"
        "    value = (torch.arange(4) * 3).tolist()\n"
        "    Path(os.environ['LOCAL019_CHILD_RECEIPT']).write_text(json.dumps({'pid': os.getpid(), 'value': value}))\n"
        "    print('LOCAL019_CPU_CHILD_EXECUTED')\n"
    )
    spec = importlib.util.spec_from_file_location("local019_launch_control", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setenv("LOCAL019_CHILD_RECEIPT", str(marker))
    harness, config = cpu_harness()
    # Static loopback avoids this host's reverse-DNS elastic rendezvous issue.
    # This still launches the actual torchrun executable and actual CPU child.
    config.rdzv_backend = "static"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        config.rdzv_endpoint = f"127.0.0.1:{listener.getsockname()[1]}"
    result = harness._benchmark_with_torchrun(module.DefaultSpec(), config)
    assert not result.errors, result.errors
    child = json.loads(marker.read_text())
    assert child["pid"] != os.getpid()
    assert child["value"] == [0, 3, 6, 9]
    assert "LOCAL019_CPU_CHILD_EXECUTED" in result.validation_message
    assert result.timing.iterations == 1
