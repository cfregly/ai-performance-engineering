from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ch18.nvfp4_trtllm_tool import (
    NVFP4TRTLLMBenchmark,
    _inspect_nvfp4_engine_assets,
    _load_model_runner,
)


def _write_engine_assets(
    root: Path,
    *,
    quant_algo: object = "NVFP4",
    world_size: object = 1,
    vocab_size: object = 32_000,
    engine_bytes: bytes | None = b"not-a-real-engine",
) -> Path:
    root.mkdir()
    config = {
        "version": "test-only",
        "pretrained_config": {
            "mapping": {"world_size": world_size, "tp_size": 1, "pp_size": 1},
            "quantization": {"quant_algo": quant_algo},
            "vocab_size": vocab_size,
        },
        "build_config": {},
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if engine_bytes is not None:
        (root / "rank0.engine").write_bytes(engine_bytes)
    return root


@pytest.mark.parametrize(
    "quant_algo",
    ["NVFP4", "W4A8_NVFP4_FP8"],
)
def test_engine_asset_inspection_accepts_declared_nvfp4_engine(
    tmp_path: Path,
    quant_algo: str,
) -> None:
    engine_dir = _write_engine_assets(tmp_path / "engine", quant_algo=quant_algo)

    assets = _inspect_nvfp4_engine_assets(engine_dir)

    assert assets.engine_dir == engine_dir.resolve()
    assert assets.config_path == engine_dir.resolve() / "config.json"
    assert assets.engine_path == engine_dir.resolve() / "rank0.engine"
    assert assets.quant_algo == quant_algo
    assert assets.vocab_size == 32_000


def test_engine_asset_inspection_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"^SKIPPED: .*engine directory is unavailable"):
        _inspect_nvfp4_engine_assets(tmp_path / "missing")


def test_engine_asset_inspection_rejects_invalid_json(tmp_path: Path) -> None:
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    (engine_dir / "config.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"^SKIPPED: .*invalid JSON"):
        _inspect_nvfp4_engine_assets(engine_dir)


@pytest.mark.parametrize("quant_algo", [None, "FP8", "W4A16_AWQ", "NVFP4_AWQ"])
def test_engine_asset_inspection_rejects_non_nvfp4_quantization(
    tmp_path: Path,
    quant_algo: object,
) -> None:
    engine_dir = _write_engine_assets(tmp_path / "engine", quant_algo=quant_algo)

    with pytest.raises(RuntimeError, match=r"^SKIPPED: .*must declare an NVFP4"):
        _inspect_nvfp4_engine_assets(engine_dir)


@pytest.mark.parametrize("world_size", [None, True, 0, 2])
def test_engine_asset_inspection_rejects_non_single_rank_engines(
    tmp_path: Path,
    world_size: object,
) -> None:
    engine_dir = _write_engine_assets(tmp_path / "engine", world_size=world_size)

    with pytest.raises(RuntimeError, match=r"^SKIPPED: .*requires a single-rank engine"):
        _inspect_nvfp4_engine_assets(engine_dir)


@pytest.mark.parametrize("engine_bytes", [None, b""])
def test_engine_asset_inspection_requires_nonempty_rank0_engine(
    tmp_path: Path,
    engine_bytes: bytes | None,
) -> None:
    engine_dir = _write_engine_assets(tmp_path / "engine", engine_bytes=engine_bytes)

    with pytest.raises(RuntimeError, match=r"^SKIPPED: .*rank-0 engine"):
        _inspect_nvfp4_engine_assets(engine_dir)


def test_benchmark_requires_explicit_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRT_LLM_ENGINE", raising=False)

    with pytest.raises(RuntimeError, match=r"^SKIPPED: .*set TRT_LLM_ENGINE"):
        NVFP4TRTLLMBenchmark()._configured_engine_dir()


def test_default_setup_exits_nonzero_without_engine() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("TRT_LLM_ENGINE", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONNOUSERSITE"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from ch18.nvfp4_trtllm_tool import NVFP4TRTLLMBenchmark; "
            "NVFP4TRTLLMBenchmark().setup()",
        ],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode != 0
    assert "SKIPPED: an explicit NVFP4 TensorRT-LLM engine directory is required" in completed.stderr
    assert "Transformer Engine" not in completed.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("tensorrt_llm") is not None,
    reason="exercises the actual missing-dependency path",
)
def test_runtime_loader_reports_actual_missing_trtllm_dependency() -> None:
    with pytest.raises(RuntimeError, match=r"^SKIPPED: TensorRT-LLM runtime dependency"):
        _load_model_runner()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (torch.ones(1, 2, dtype=torch.int32), "dict containing output_ids"),
        ({}, "dict containing output_ids"),
        ({"output_ids": [1, 2]}, "non-Tensor output_ids"),
        ({"output_ids": torch.ones(2, 2, dtype=torch.int32)}, "must be non-empty"),
        ({"output_ids": torch.ones(1, 2)}, "integer token dtype"),
    ],
)
def test_output_validation_rejects_unsupported_generate_payloads(
    payload: object,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        NVFP4TRTLLMBenchmark._require_output_ids(payload)


def test_output_validation_preserves_real_token_tensor() -> None:
    output_ids = torch.arange(8, dtype=torch.int32).reshape(1, 1, 8)

    observed = NVFP4TRTLLMBenchmark._require_output_ids({"output_ids": output_ids})

    assert observed is output_ids


def test_verification_payload_records_engine_quantization_without_false_fp8_flag() -> None:
    benchmark = NVFP4TRTLLMBenchmark("unused-for-payload-test")
    benchmark.input_ids = torch.arange(4, dtype=torch.int32)
    benchmark.output = torch.arange(5, dtype=torch.int32).reshape(1, 1, 5)
    benchmark._quant_algo = "NVFP4"

    benchmark.capture_verification_payload()
    signature = benchmark.get_input_signature()

    assert signature.quantization_mode == "NVFP4"
    assert signature.precision_flags.fp8 is False
    assert signature.parameter_count == 0
    assert benchmark.get_output_tolerance() == (0.0, 0.0)


_REAL_ENGINE = os.getenv("AISP_TEST_NVFP4_TRTLLM_ENGINE", "").strip()


@pytest.mark.skipif(not _REAL_ENGINE, reason="set AISP_TEST_NVFP4_TRTLLM_ENGINE to a real engine")
def test_real_nvfp4_engine_loads_and_generates() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the opt-in real TensorRT-LLM engine test")

    benchmark = NVFP4TRTLLMBenchmark(_REAL_ENGINE)
    try:
        benchmark.setup()
        result = benchmark.benchmark_fn()
        benchmark.capture_verification_payload()

        assert result == {}
        assert benchmark.output is not None
        assert benchmark.get_input_signature().quantization_mode in {
            "NVFP4",
            "W4A8_NVFP4_FP8",
        }
        assert benchmark.get_output_tolerance() == (0.0, 0.0)
    finally:
        benchmark.teardown()
