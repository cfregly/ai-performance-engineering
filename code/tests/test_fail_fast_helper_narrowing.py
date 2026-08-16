from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import pytest
import torch

from core.book import BookIndex
from core.harness import benchmark_harness, validity_checks
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode
from core.llm import _check_ollama
from core.perf_core_base import _package_root, _safe_package_version
from mcp.tool_generator import generate_tool_schema


def test_check_ollama_returns_false_for_unreachable_host() -> None:
    previous_host = os.environ.get("OLLAMA_HOST")
    os.environ["OLLAMA_HOST"] = "http://127.0.0.1:1"
    try:
        assert _check_ollama() is False
    finally:
        if previous_host is None:
            os.environ.pop("OLLAMA_HOST", None)
        else:
            os.environ["OLLAMA_HOST"] = previous_host


def test_generate_tool_schema_falls_back_when_type_hints_cannot_resolve() -> None:
    def sample_tool(
        missing: "MissingType",  # noqa: F821 - unresolved type exercises the fallback
        count: int,
        enabled: bool = False,
    ) -> None:
        return None

    schema = generate_tool_schema(sample_tool)

    assert schema["properties"]["missing"]["type"] == "string"
    assert schema["properties"]["count"]["type"] == "string"
    assert schema["properties"]["enabled"]["type"] == "string"
    assert schema["required"] == ["missing", "count"]


def test_safe_package_version_and_package_root_return_none_for_missing_modules() -> None:
    assert _safe_package_version("definitely-not-a-real-package-name") is None
    assert _package_root("definitely_not_a_real_module_name") is None


def test_book_index_search_reads_real_chapter_content(tmp_path: Path) -> None:
    chapter = tmp_path / "ch07.md"
    chapter.write_text(
        "# Chapter 7: Memory Optimization\n\n## Shared Memory\nShared memory improves coalescing.\n",
        encoding="utf-8",
    )

    index = BookIndex(book_dir=tmp_path)
    results = index.search("shared memory", max_results=1)

    assert results
    assert results[0].chapter == "ch07"
    assert results[0].chapter_title == "Chapter 7: Memory Optimization"


def test_book_index_raises_on_invalid_unicode_content(tmp_path: Path) -> None:
    chapter = tmp_path / "ch07.md"
    chapter.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(UnicodeDecodeError):
        BookIndex(book_dir=tmp_path)


class _CpuWarningBenchmark(BaseBenchmark):
    allow_cpu = True

    def __init__(self, *, broken_marker: bool = False, return_mapping: bool = False) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self._broken_marker = broken_marker
        self._return_mapping = return_mapping
        self.input_tensor: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        self.input_tensor = torch.ones(4, device=self.device)

    def benchmark_fn(self):
        if self.input_tensor is None:
            raise RuntimeError("setup() must initialize input_tensor")
        self.output = self.input_tensor + 1
        if self._return_mapping:
            return {"output": self.output}
        return self.output

    def mark_execution_complete(self) -> None:
        if self._broken_marker:
            raise RuntimeError("marker boom")
        super().mark_execution_complete()

    def validate_result(self) -> Optional[str]:
        return None

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        if self.input_tensor is None:
            raise RuntimeError("setup() must initialize input_tensor")
        return {"input": self.input_tensor}

    def get_verify_output(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def get_input_signature(self) -> dict[str, object]:
        if self.input_tensor is None:
            raise RuntimeError("setup() must initialize input_tensor")
        return {"shape": tuple(self.input_tensor.shape), "dtype": str(self.input_tensor.dtype)}


def _make_cpu_custom_harness() -> BenchmarkHarness:
    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=2,
        warmup=0,
        enable_profiling=False,
        enable_memory_tracking=False,
        use_subprocess=False,
        detect_setup_precomputation=False,
        monitor_gpu_state=False,
        monitor_backend_policy=False,
        track_memory_allocations=False,
        enforce_environment_validation=False,
    )
    return BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)


def test_resolve_physical_device_index_warns_when_nvml_shutdown_fails() -> None:
    fake_pynvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda idx: "handle",
        nvmlDeviceGetUUID=lambda handle: "GPU-test",
        nvmlShutdown=lambda: (_ for _ in ()).throw(RuntimeError("shutdown boom")),
    )

    with patch.dict(sys.modules, {"pynvml": fake_pynvml}), patch.dict(
        os.environ,
        {"CUDA_VISIBLE_DEVICES": "GPU-test"},
        clear=False,
    ):
        with pytest.warns(RuntimeWarning, match="Failed to shut down NVML after resolving physical device index"):
            assert benchmark_harness._resolve_physical_device_index(0) == 0


def test_force_tensor_evaluation_warns_when_item_materialization_fails() -> None:
    validity_checks._EMITTED_VALIDITY_LIMITATIONS.discard("force_tensor_evaluation_item_failure")
    validity_checks._VALIDITY_LIMITATION_RECORDS.pop("force_tensor_evaluation_item_failure", None)

    with patch.object(torch.Tensor, "item", side_effect=RuntimeError("item boom")):
        with pytest.warns(RuntimeWarning, match="force_tensor_evaluation could not materialize"):
            validity_checks.force_tensor_evaluation({"x": torch.tensor([1.0])})


def test_benchmark_harness_warns_when_mark_execution_complete_fails_on_cpu() -> None:
    harness = _make_cpu_custom_harness()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = harness.benchmark(_CpuWarningBenchmark(broken_marker=True))

    messages = [str(item.message) for item in captured]
    assert any("mark_execution_complete() failed during benchmark run" in message for message in messages)
    assert not any("marker boom" in error for error in result.errors)


def test_benchmark_harness_warns_once_when_force_tensor_evaluation_fails_on_cpu() -> None:
    harness = _make_cpu_custom_harness()

    with patch("core.harness.validity_checks.force_tensor_evaluation", side_effect=RuntimeError("force eval boom")):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = harness.benchmark(_CpuWarningBenchmark(return_mapping=True))

    matches = [
        str(item.message)
        for item in captured
        if "Failed to force eager tensor evaluation after timing" in str(item.message)
    ]
    assert len(matches) == 1
    assert not any("force eval boom" in error for error in result.errors)
