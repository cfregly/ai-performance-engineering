"""Evidence-mode controls for the dynamic-router cheap eval tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labs.dynamic_router import eval_stack


def test_default_mode_fails_when_vllm_import_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_error = ModuleNotFoundError("No module named 'vllm'")
    monkeypatch.setattr(eval_stack, "LLM", None)
    monkeypatch.setattr(eval_stack, "SamplingParams", None)
    monkeypatch.setattr(eval_stack, "_VLLM_IMPORT_ERROR", import_error)

    with pytest.raises(eval_stack.VLLMRequiredError, match="could not be imported") as error:
        eval_stack.CheapEvalStack(eval_stack.EvalConfig())

    assert "--no-vllm" in str(error.value)
    assert error.value.__cause__ is import_error


def test_default_mode_fails_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eval_stack, "LLM", object())
    monkeypatch.setattr(eval_stack, "SamplingParams", object())
    monkeypatch.setattr(eval_stack.torch.cuda, "is_available", lambda: False)

    with pytest.raises(eval_stack.VLLMRequiredError, match="CUDA is unavailable"):
        eval_stack.CheapEvalStack(eval_stack.EvalConfig())


def test_vllm_initialization_failure_is_non_synthetic_and_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}\n")

    class FailingLLM:
        def __init__(self, **_: object) -> None:
            print("engine setup started")
            raise RuntimeError("engine initialization exploded")

    monkeypatch.setattr(eval_stack, "LLM", FailingLLM)
    monkeypatch.setattr(eval_stack, "SamplingParams", object())
    monkeypatch.setattr(eval_stack.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(eval_stack.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(eval_stack, "ensure_gpt_oss_20b", lambda _: model_path)

    with pytest.raises(eval_stack.VLLMRequiredError, match="vLLM initialization failed"):
        eval_stack.CheapEvalStack(eval_stack.EvalConfig(model_path=str(model_path)))

    diagnostic = json.loads(capsys.readouterr().err)
    assert diagnostic["event"] == "vllm_llm_init_error"
    assert diagnostic["lines"] == [
        "engine setup started",
        "llm_init_error: engine initialization exploded",
    ]


def test_explicit_no_vllm_mode_records_synthetic_provenance(tmp_path: Path) -> None:
    cfg = eval_stack.EvalConfig(
        run_root=tmp_path / "runs",
        request_count=4,
        use_vllm=False,
    )

    summary = eval_stack.CheapEvalStack(cfg).run()
    sys_meta = json.loads(Path(summary["sys_meta_path"]).read_text())
    quality_rows = [
        json.loads(line)
        for line in (Path(summary["run_dir"]) / "quality.jsonl").read_text().splitlines()
    ]

    assert summary["execution_mode"] == "synthetic_no_vllm"
    assert summary["evidence_mode"] == "synthetic"
    assert summary["used_vllm"] is False
    assert summary["synthetic_components"] == ["quality", "latency", "moe_router"]
    assert sys_meta["requested_vllm"] is False
    assert sys_meta["component_sources"] == {
        "quality": "synthetic_no_vllm",
        "latency": "synthetic_model",
        "moe_router": "synthetic_model",
        "throughput": "derived_from_synthetic_latency",
    }
    assert quality_rows
    assert {row["source"] for row in quality_rows} == {"synthetic"}


def test_allow_missing_metrics_remains_an_explicit_synthetic_replay_mode(
    tmp_path: Path,
) -> None:
    metrics_dir = tmp_path / "empty-metrics"
    metrics_dir.mkdir()
    cfg = eval_stack.EvalConfig(
        run_root=tmp_path / "runs",
        metrics_dir=metrics_dir,
        allow_missing_metrics=True,
        request_count=2,
    )

    summary = eval_stack.CheapEvalStack(cfg).run()
    sys_meta = json.loads(Path(summary["sys_meta_path"]).read_text())

    assert summary["execution_mode"] == "metrics_replay_with_synthetic_fallback"
    assert summary["evidence_mode"] == "synthetic"
    assert summary["used_vllm"] is False
    assert sys_meta["allow_missing_metrics"] is True
    assert sys_meta["requested_vllm"] is True
    assert sys_meta["synthetic_components"] == ["quality", "latency", "moe_router"]


def test_no_vllm_flag_selects_the_explicit_synthetic_mode() -> None:
    cfg = eval_stack.EvalConfig.from_flags(["--no-vllm", "--request-count", "3"])

    assert cfg.use_vllm is False
    assert cfg.request_count == 3
