import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

from cli.commands import inference
from core.optimization.parallelism_planner.inference_optimization import (
    InferenceEngineRecommender,
    QuantizationType,
)


def test_vllm_launch_producer_preserves_literal_model_in_argv_and_display() -> None:
    model = "org/model with spaces; $(literal) 'quoted' #tag"
    recommendation = InferenceEngineRecommender()._create_vllm_recommendation(
        model,
        model_params_b=7,
        num_gpus=2,
        gpu_memory_gb=80,
        quantization=QuantizationType.FP8,
    )

    payload = recommendation.to_dict()
    launch_argv = payload["launch_argv"]
    assert launch_argv is not None
    model_index = launch_argv.index("--model") + 1
    assert launch_argv[model_index] == model
    assert shlex.split(payload["launch_command"]) == launch_argv
    assert launch_argv[-4:] == ["--quantization", "fp8", "--kv-cache-dtype", "fp8"]


def test_composite_launch_plans_are_display_only_and_quote_the_model() -> None:
    model = "org/model; $(literal) with spaces"
    recommender = InferenceEngineRecommender()

    tensorrt = recommender._create_tensorrt_recommendation(
        model,
        model_params_b=7,
        num_gpus=2,
        gpu_memory_gb=80,
        quantization=QuantizationType.NONE,
    )
    llama_cpp = recommender._create_llama_cpp_recommendation(model, model_params_b=7)

    for recommendation in (tensorrt, llama_cpp):
        assert recommendation.launch_argv is None
        assert shlex.quote(model) in recommendation.launch_command


def test_serve_executes_structured_argv_and_preserves_literal_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    marker = tmp_path / "shell-injection.marker"
    observed = tmp_path / "observed-argv.json"
    injected_command = shlex.join(
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('injected')",
        ]
    )
    model = f"safe-model; {injected_command} #"
    capture_script = (
        "import json, sys; from pathlib import Path; "
        "Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))"
    )
    launch_argv = [sys.executable, "-c", capture_script, str(observed), model]

    class FakeInference:
        def deploy(self, params):
            unsafe_display = f"{shlex.join(launch_argv[:-1])} {params['model']}"
            return {
                "success": True,
                "launch_command": unsafe_display,
                "engine": {"engine": "vllm", "launch_argv": launch_argv},
            }

    class FakeEngine:
        inference = FakeInference()

    monkeypatch.setattr(inference, "get_engine", lambda: FakeEngine())
    rc = inference.serve(SimpleNamespace(model=model, model_size=1, run=True, json=False))

    assert rc == 0
    assert not marker.exists()
    assert json.loads(observed.read_text(encoding="utf-8")) == [model]


def test_serve_refuses_composite_plan_execution(monkeypatch, tmp_path: Path, capsys) -> None:
    marker = tmp_path / "composite.marker"
    launch_command = shlex.join(
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
        ]
    )

    class FakeInference:
        def deploy(self, params):
            return {
                "success": True,
                "launch_command": launch_command,
                "engine": {"engine": "tensorrt_llm", "launch_argv": None},
            }

    class FakeEngine:
        inference = FakeInference()

    monkeypatch.setattr(inference, "get_engine", lambda: FakeEngine())
    rc = inference.serve(SimpleNamespace(model="model", model_size=1, run=True, json=False))

    assert rc == 1
    assert not marker.exists()
    assert "contains multiple commands" in capsys.readouterr().out
