"""LOCAL-019: genuine launch control flow, never synthetic training verification."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional

from core.benchmark.verification import PrecisionFlags
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    LaunchVia,
)
from labs.train_distributed.training_utils.child_result import (
    CONTRACT_ENV,
    ITERATIONS_ENV,
    LAUNCH_MONOTONIC_NS_ENV,
    LAUNCH_WALL_NS_ENV,
    RESULT_DIR_ENV,
    RUN_ID_ENV,
    WORLD_SIZE_ENV,
    TorchrunChildResultContract,
    validate_training_child_result_bundle,
    write_training_child_result,
)
from labs.train_distributed.training_utils.ddp_child_result import (
    DDP_ADAMW_BETAS,
    DDP_ADAMW_WEIGHT_DECAY,
    DDP_TRAINING_OUTPUT_TOLERANCE,
    bind_distributed_sampler_seed,
    initialize_ddp_seed,
    make_ddp_adamw,
    make_ddp_child_result_contract,
    publish_ddp_child_result,
)
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
        device=torch.device("cpu"),
        iterations=1,
        warmup=5,
        launch_via=LaunchVia.TORCHRUN,
        nproc_per_node=1,
        multi_gpu_required=False,
        use_subprocess=False,
        enable_profiling=False,
        lock_gpu_clocks=False,
        enforce_environment_validation=False,
        measurement_timeout_seconds=60,
        nnodes="1",
        rdzv_backend="c10d",
        rdzv_endpoint="127.0.0.1:0",
    )
    return BenchmarkHarness(config=config), config


def linear_child_contract(world_size=1):
    return TorchrunChildResultContract(
        profile="tests:linear-training-step",
        input_names=("features",),
        output_names=("prediction",),
        per_rank_batch_size=2,
        parameter_count=8,
        precision_flags=PrecisionFlags(tf32=False),
        output_tolerance=(0.0, 0.0),
        independent_reference="torch.nn.functional.linear",
        collective_type="none" if world_size > 1 else None,
        collective_algorithm="independent-rank-publication" if world_size > 1 else None,
    )


@pytest.mark.parametrize(
    "method",
    [
        "setup",
        "benchmark_fn",
        "capture_verification_payload",
        "_prepare_verification_payload",
        "get_verify_inputs",
        "get_verify_output",
        "get_input_signature",
        "get_output_tolerance",
        "get_torchrun_spec",
    ],
)
@pytest.mark.parametrize("stale_payload", [False, True])
def test_generic_training_verification_is_unsupported(tmp_path, method, stale_payload):
    benchmark = TorchrunScriptBenchmark(
        script_path=tmp_path / "training.py",
        multi_gpu_required=False,
        default_nproc_per_node=1,
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


@pytest.mark.parametrize("world_size", [1, 2])
def test_opted_in_training_child_result_runs_real_torchrun_and_exposes_full_outputs(
    tmp_path, world_size
):
    target = tmp_path / "verified_training_child.py"
    target.write_text(
        "import os\n"
        "import torch\n"
        "import torch.nn.functional as F\n"
        "from labs.train_distributed.training_utils.child_result import write_training_child_result\n"
        "rank = int(os.environ.get('RANK', '0'))\n"
        "requested_steps = int(os.environ['AISP_TRAINING_RESULT_ITERATIONS'])\n"
        "completed_steps = max(1, requested_steps - 1)\n"
        "x = torch.arange(6, dtype=torch.float32).reshape(2, 3) + rank\n"
        "x_changed = x + 0.25\n"
        "weight = torch.tensor([[1., 2., 3.], [-2., 1., 0.5]])\n"
        "bias = torch.tensor([0.5, -1.])\n"
        "actual = x @ weight.t() + bias\n"
        "changed = x_changed @ weight.t() + bias\n"
        "write_training_child_result(\n"
        "    inputs={'features': x}, outputs={'prediction': actual},\n"
        "    reference_outputs={'prediction': F.linear(x, weight, bias)},\n"
        "    sensitivity_inputs={'features': x_changed},\n"
        "    sensitivity_outputs={'prediction': changed},\n"
        "    sensitivity_reference_outputs={'prediction': F.linear(x_changed, weight, bias)},\n"
        "    completed_iterations=completed_steps,\n"
        ")\n"
        "if rank == 0: print('rank0 time_per_iter_ms: 1.25', flush=True)\n"
    )
    benchmark = TorchrunScriptBenchmark(
        script_path=target,
        multi_gpu_required=False,
        default_nproc_per_node=world_size,
        child_result_contract=linear_child_contract(world_size),
        name="verified-linear-training-child",
    )
    harness, config = cpu_harness()
    config.iterations = 3
    config.seed = 1042
    config.nproc_per_node = world_size
    config.rdzv_backend = "static"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        config.rdzv_endpoint = f"127.0.0.1:{listener.getsockname()[1]}"

    result = harness._benchmark_with_torchrun(benchmark, config)

    assert not result.errors, result.errors
    assert result.timing.mean_ms == pytest.approx(1.25)
    assert result.timing.sample_scope == "rank0_iteration_mean"
    assert result.timing.iterations_per_sample == 2
    inputs = benchmark.get_verify_inputs()
    outputs = benchmark.get_verify_output()
    expected_input_names = {f"rank-{rank}:features" for rank in range(world_size)}
    expected_output_names = {f"rank-{rank}:prediction" for rank in range(world_size)} | {
        "completed_iterations"
    }
    assert inputs.keys() == expected_input_names
    assert outputs.keys() == expected_output_names
    assert outputs["completed_iterations"].shape == (2,)
    expected_weight = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]])
    for rank in range(world_size):
        expected_input = torch.arange(6, dtype=torch.float32).reshape(2, 3) + rank
        assert inputs[f"rank-{rank}:features"].shape == (2, 3)
        torch.testing.assert_close(
            outputs[f"rank-{rank}:prediction"],
            expected_input @ expected_weight.t() + torch.tensor([0.5, -1.0]),
            rtol=0,
            atol=0,
        )
    signature = benchmark.get_input_signature()
    assert signature.world_size == world_size and signature.batch_size == 2 * world_size
    assert signature.parameter_count == 8
    assert signature.shapes == {name: tuple(tensor.shape) for name, tensor in inputs.items()}
    assert signature.dtypes == {name: str(tensor.dtype) for name, tensor in inputs.items()}
    assert benchmark.get_output_tolerance() == (0.0, 0.0)
    assert benchmark.validate_result() is None
    child_pids = {payload["pid"] for payload in benchmark._child_result_bundle["payloads"]}
    assert len(child_pids) == world_size and os.getpid() not in child_pids
    assert benchmark._child_result_bundle["torch_seed"] == 1042
    assert not Path(benchmark._child_result_context["result_dir"]).exists()


def test_child_writer_rejects_iteration_overrun_and_input_insensitive_outputs(
    tmp_path, monkeypatch
):
    contract = linear_child_contract()
    now_wall = 1_000_000
    now_monotonic = 2_000_000
    environment = {
        RESULT_DIR_ENV: str(tmp_path),
        RUN_ID_ENV: "real-child-control",
        CONTRACT_ENV: json.dumps(contract.to_dict()),
        WORLD_SIZE_ENV: "1",
        ITERATIONS_ENV: "3",
        LAUNCH_WALL_NS_ENV: str(now_wall),
        LAUNCH_MONOTONIC_NS_ENV: str(now_monotonic),
        "RANK": "0",
        "WORLD_SIZE": "1",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    changed = x + 1
    output = torch.ones(2, 2)

    with pytest.raises(RuntimeError, match="completed iteration count"):
        write_training_child_result(
            inputs={"features": x},
            outputs={"prediction": output},
            reference_outputs={"prediction": output.clone()},
            sensitivity_inputs={"features": changed},
            sensitivity_outputs={"prediction": output.clone()},
            sensitivity_reference_outputs={"prediction": output.clone()},
            completed_iterations=4,
        )
    with pytest.raises(RuntimeError, match="independent reference"):
        write_training_child_result(
            inputs={"features": x},
            outputs={"prediction": output},
            reference_outputs={"prediction": torch.zeros_like(output)},
            sensitivity_inputs={"features": changed},
            sensitivity_outputs={"prediction": output + 1},
            sensitivity_reference_outputs={"prediction": output + 1},
            completed_iterations=3,
        )
    with pytest.raises(RuntimeError, match="did not respond"):
        write_training_child_result(
            inputs={"features": x},
            outputs={"prediction": output},
            reference_outputs={"prediction": output.clone()},
            sensitivity_inputs={"features": changed},
            sensitivity_outputs={"prediction": output.clone()},
            sensitivity_reference_outputs={"prediction": output.clone()},
            completed_iterations=3,
        )
    assert not list(tmp_path.glob("rank-*.pt"))


def test_ddp_post_training_result_uses_actual_batch_full_outputs_and_step_count(
    tmp_path, monkeypatch
):
    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(vocab_size=13)
            self.embedding = torch.nn.Embedding(13, 4)
            self.projection = torch.nn.Linear(4, 13)

        def forward(self, input_ids, attention_mask, labels):
            del attention_mask
            logits = self.projection(self.embedding(input_ids))
            loss = functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            return SimpleNamespace(logits=logits, loss=loss)

    class TransparentCandidate(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, **inputs):
            return self.module(**inputs)

    torch.manual_seed(7)
    reference = TinyCausalLM()
    candidate = TransparentCandidate(reference)
    input_ids = (torch.arange(16 * 10).reshape(16, 10) % 12) + 1
    attention_mask = torch.ones_like(input_ids)
    final_batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.masked_fill(attention_mask == 0, -100),
    }
    contract = make_ddp_child_result_contract(multigpu=False)
    result_dir = tmp_path / "ddp-result"
    result_dir.mkdir()
    launch_wall_ns = time.time_ns()
    launch_monotonic_ns = time.monotonic_ns()
    environment = {
        RESULT_DIR_ENV: str(result_dir),
        RUN_ID_ENV: "ddp-real-output-control",
        CONTRACT_ENV: json.dumps(contract.to_dict()),
        WORLD_SIZE_ENV: "1",
        ITERATIONS_ENV: "5",
        LAUNCH_WALL_NS_ENV: str(launch_wall_ns),
        LAUNCH_MONOTONIC_NS_ENV: str(launch_monotonic_ns),
        "RANK": "0",
        "WORLD_SIZE": "1",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    path = publish_ddp_child_result(
        candidate_model=candidate,
        reference_model=reference,
        final_batch=final_batch,
        completed_iterations=3,
    )
    finish_monotonic_ns = time.monotonic_ns()
    finish_wall_ns = time.time_ns()
    assert path == result_dir / "rank-0.pt"
    assert candidate.training and reference.training

    bundle = validate_training_child_result_bundle(
        result_dir,
        contract=contract,
        run_id="ddp-real-output-control",
        world_size=1,
        requested_iterations=5,
        launch_wall_ns=launch_wall_ns,
        launch_monotonic_ns=launch_monotonic_ns,
        finish_wall_ns=finish_wall_ns,
        finish_monotonic_ns=finish_monotonic_ns,
    )
    assert bundle["completed_iterations"] == 3
    assert bundle["torch_seed"] == 7
    assert bundle["verify_inputs"]["rank-0:input_ids"].shape == (16, 10)
    assert torch.equal(bundle["verify_inputs"]["rank-0:input_ids"], input_ids)
    assert bundle["verify_output"]["rank-0:logits"].shape == (16, 10, 13)
    assert bundle["verify_output"]["completed_iterations"].shape == (3,)
    assert bundle["input_signature"].shapes == {
        name: tuple(tensor.shape) for name, tensor in bundle["verify_inputs"].items()
    }
    assert not (set(bundle["input_signature"].shapes) & set(bundle["verify_output"]))
    payload = bundle["payloads"][0]
    assert not torch.equal(
        payload["inputs"]["input_ids"], payload["sensitivity_inputs"]["input_ids"]
    )
    assert not torch.equal(payload["outputs"]["logits"], payload["sensitivity_outputs"]["logits"])
    assert bundle["input_signature"].parameter_count == sum(
        parameter.numel() for parameter in reference.parameters()
    )
    assert bundle["output_tolerance"] == DDP_TRAINING_OUTPUT_TOLERANCE


def test_ddp_seed_preserves_harness_seed_and_changes_distributed_order(tmp_path, monkeypatch):
    monkeypatch.setenv(RESULT_DIR_ENV, str(tmp_path))
    torch.manual_seed(1042)
    assert initialize_ddp_seed() == 1042
    assert torch.initial_seed() == 1042

    sampler_42 = torch.utils.data.DistributedSampler(
        range(64), num_replicas=2, rank=0, shuffle=True
    )
    sampler_1042 = torch.utils.data.DistributedSampler(
        range(64), num_replicas=2, rank=0, shuffle=True
    )
    bind_distributed_sampler_seed(SimpleNamespace(sampler=sampler_42), 42)
    bind_distributed_sampler_seed(SimpleNamespace(sampler=sampler_1042), 1042)
    assert list(sampler_42) != list(sampler_1042)

    monkeypatch.delenv(RESULT_DIR_ENV)
    torch.manual_seed(9)
    assert initialize_ddp_seed() == 42
    assert torch.initial_seed() == 42


def test_ddp_child_result_rejects_an_unrelated_reference_model(tmp_path, monkeypatch):
    monkeypatch.setenv(RESULT_DIR_ENV, str(tmp_path))
    candidate = torch.nn.Sequential(torch.nn.Linear(2, 2))
    unrelated = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with pytest.raises(RuntimeError, match="exact unwrapped trained model"):
        publish_ddp_child_result(
            candidate_model=candidate,
            reference_model=unrelated,
            final_batch={},
            completed_iterations=1,
        )


def test_ddp_adamw_arms_share_optimizer_math_configuration():
    baseline_parameter = torch.nn.Parameter(torch.ones(2))
    optimized_parameter = torch.nn.Parameter(torch.ones(2))
    baseline = make_ddp_adamw([baseline_parameter], 2e-4, prefer_fused=False)
    optimized = make_ddp_adamw([optimized_parameter], 2e-4, prefer_fused=True)
    for optimizer in (baseline, optimized):
        group = optimizer.param_groups[0]
        assert group["lr"] == 2e-4
        assert group["betas"] == DDP_ADAMW_BETAS
        assert group["weight_decay"] == DDP_ADAMW_WEIGHT_DECAY


def test_generic_validate_result_never_accepts_a_surrogate(tmp_path):
    benchmark = TorchrunScriptBenchmark(
        script_path=tmp_path / "training.py", multi_gpu_required=False
    )
    assert REASON in benchmark.validate_result()
    benchmark._output = torch.ones(1)
    benchmark._subprocess_verify_output = benchmark._output
    benchmark._child_result_bundle = {"forged": True}
    assert REASON in benchmark.validate_result()
    with pytest.raises(RuntimeError, match=REASON):
        benchmark.get_verify_output()


def test_launch_configuration_remains_discoverable(tmp_path):
    benchmark = TorchrunScriptBenchmark(
        script_path=tmp_path / "training.py",
        base_args=["--mode", "baseline"],
        target_label="training:example",
        config_arg_map={"iterations": "--steps"},
        multi_gpu_required=False,
        default_nproc_per_node=1,
        default_iterations=7,
        measurement_timeout_seconds=123,
        env={"TRAINING_TEST": "1"},
        name="training-example",
    )
    config = benchmark.get_config()
    assert config.launch_via == LaunchVia.TORCHRUN
    assert config.nproc_per_node == 1 and not config.multi_gpu_required
    assert config.iterations == 7 and config.measurement_timeout_seconds == 123
    assert config.target_label == "training:example"
    assert benchmark.name == "training-example" and benchmark._device is None
    benchmark.teardown()
    assert benchmark._verification_payload is None


@pytest.mark.parametrize(
    ("name", "multigpu"),
    [
        ("baseline_ddp", False),
        ("optimized_ddp", False),
        ("baseline_ddp_multigpu", True),
        ("optimized_ddp_multigpu", True),
    ],
)
def test_plain_ddp_factories_opt_into_full_child_result_contract(name, multigpu):
    module = importlib.import_module(f"labs.train_distributed.{name}")
    benchmark = module.get_benchmark()
    assert isinstance(benchmark, TorchrunScriptBenchmark)
    assert benchmark._script_path.is_file()
    assert benchmark._base_args and benchmark._config_arg_map
    assert benchmark._child_result_contract == make_ddp_child_result_contract(multigpu=multigpu)
    config = BenchmarkConfig(
        iterations=5,
        nproc_per_node=2 if multigpu else 1,
        nnodes="1",
    )
    spec = benchmark.get_torchrun_spec(config)
    assert spec.result_callback == "consume_training_child_results"
    assert spec.timing_source == "rank0_time_per_iter_ms"
    assert spec.timing_iterations_per_sample == 5
    assert spec.script_path.name == "ddp.py"
    result_dir = Path(spec.env[RESULT_DIR_ENV])
    assert result_dir.is_dir() and not list(result_dir.iterdir())
    result_dir.rmdir()


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


@pytest.mark.parametrize(
    "error", [ValueError("invalid declared launch spec"), RuntimeError("SKIPPED: unverified child")]
)
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
        script_path=tmp_path / "training.py",
        multi_gpu_required=False,
        default_nproc_per_node=1,
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
        [
            sys.executable,
            "-m",
            "core.harness.torchrun_wrapper",
            "--aisp-expected-torch-seed",
            "42",
            "--aisp-target-script",
            str(target),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
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
