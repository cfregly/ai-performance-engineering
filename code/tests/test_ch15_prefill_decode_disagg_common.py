"""Smoke tests for the shared Chapter 15 prefill/decode disaggregation wrappers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from core.hot_path_checks import (
    check_benchmark_fn_antipatterns,
    check_benchmark_fn_sync_calls,
)
from core.harness.benchmark_harness import BaseBenchmark


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("relative_path", "expected_multi_gpu", "expected_allowed"),
    [
        ("ch15/baseline_prefill_decode_disagg.py", False, ("host_transfer",)),
        ("ch15/optimized_prefill_decode_disagg.py", False, ()),
        ("ch15/baseline_prefill_decode_disagg_multigpu.py", True, ("host_transfer",)),
        ("ch15/optimized_prefill_decode_disagg_multigpu.py", True, ()),
    ],
)
def test_prefill_decode_disagg_wrappers_attach_metadata(
    relative_path: str,
    expected_multi_gpu: bool,
    expected_allowed: tuple[str, ...],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    module = _load_module(module_path)

    bench = module.get_benchmark()

    assert isinstance(bench, BaseBenchmark)
    assert getattr(bench, "_module_file_override", None) == str(module_path)
    assert getattr(bench, "_factory_name_override", None) == "get_benchmark"
    assert bool(getattr(bench, "multi_gpu_required", False)) is expected_multi_gpu
    assert tuple(getattr(bench, "allowed_benchmark_fn_antipatterns", ())) == expected_allowed
    assert bool(bench.get_config().multi_gpu_required) is expected_multi_gpu


@pytest.mark.parametrize(
    ("relative_path", "expected_allowed"),
    [
        ("ch15/baseline_prefill_decode_disagg.py", ("host_transfer",)),
        ("ch15/optimized_prefill_decode_disagg.py", ()),
        ("ch15/baseline_prefill_decode_disagg_multigpu.py", ("host_transfer",)),
        ("ch15/optimized_prefill_decode_disagg_multigpu.py", ()),
    ],
)
def test_prefill_decode_disagg_common_hot_path_checks_stay_clean(
    relative_path: str,
    expected_allowed: tuple[str, ...],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / relative_path)
    bench = module.get_benchmark()

    sync_ok, sync_warnings = check_benchmark_fn_sync_calls(bench.benchmark_fn)
    antipattern_ok, antipattern_warnings = check_benchmark_fn_antipatterns(
        bench.benchmark_fn,
        allowed_codes=expected_allowed,
    )

    assert sync_ok, sync_warnings
    assert antipattern_ok, antipattern_warnings


def test_prefill_decode_disagg_handoff_reuses_staging_buffers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "ch15" / "prefill_decode_disagg_common.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def _handoff_kv", maxsplit=1
    )[0]
    handoff_section = source.split("def _handoff_kv", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]
    teardown_section = source.split("def teardown", maxsplit=1)[1].split(
        "def get_config", maxsplit=1
    )[0]

    assert "self._host_staging = {}" in setup_section
    assert "self._handoff_staging = {}" in setup_section
    assert "self._request_groups = []" in setup_section
    assert "self._request_output_groups = []" in setup_section
    assert "self._request_groups.extend(" in setup_section
    assert "self._request_output_groups = [" in setup_section
    assert "batch_slice[idx : idx + 1]" in setup_section
    assert "self._output_shards = [torch.empty(0) for _ in range(self.batch_size)]" in setup_section
    assert "probe_width = min(256, self.hidden_size)" in setup_section
    assert "probe_shape = torch.Size((1, 1, probe_width))" in setup_section
    assert "self._verify_probe = self._empty_cpu_staging(probe_shape, torch.bfloat16)" in setup_section
    assert "self._verify_probe.copy_(" in setup_section
    assert "self.prefill_inputs[0][:1, :1, :probe_width]" in setup_section
    assert "self._verify_output_stack: Optional[torch.Tensor] = None" in source
    assert "self._verify_output_buffer: Optional[torch.Tensor] = None" in source
    assert "verify_shape = torch.Size((min(2, self.batch_size), min(256, self.hidden_size)))" in setup_section
    assert "self._verify_output_stack = self._empty_cpu_staging(verify_shape, torch.bfloat16)" in setup_section
    assert "self._verify_output_buffer = self._empty_cpu_staging(verify_shape, torch.float32)" in setup_section
    assert "self._handoff_staging[staging_key] = torch.empty(" in setup_section
    assert "prefill_out.cpu()" not in handoff_section
    assert "kv_cpu.to(decode_device)" not in handoff_section
    assert "host_buf.copy_(prefill_out, non_blocking=False)" in handoff_section
    assert "decode_buf.copy_(host_buf, non_blocking=False)" in handoff_section
    assert "decode_buf.copy_(prefill_out, non_blocking=True)" in handoff_section
    assert "with torch.inference_mode():" in benchmark_section
    assert "outputs = self._output_shards" in benchmark_section
    assert "enumerate(" not in benchmark_section
    assert "self._request_groups" in benchmark_section
    assert "self._request_output_groups" in benchmark_section
    assert "output_idx," in benchmark_section
    assert "decode_device," in benchmark_section
    assert "prefill_model," in benchmark_section
    assert "decode_model," in benchmark_section
    assert ") in self._request_output_groups:" in benchmark_section
    assert "prefill_out = prefill_model(request)" in benchmark_section
    assert "outputs[output_idx] = token_state.squeeze(0).squeeze(0)" in benchmark_section
    assert "output_idx += 1" not in benchmark_section
    assert "zip(" not in benchmark_section
    assert "for idx in range(batch.shape[0])" not in benchmark_section
    assert "batch[idx : idx + 1]" not in benchmark_section
    assert "outputs: list[torch.Tensor] = []" not in benchmark_section
    assert "outputs.append(" not in benchmark_section
    assert "for output_idx in range(selected_count):" in capture_section
    assert "self._verify_output_stack[output_idx].copy_(" in capture_section
    assert "self._output_shards[output_idx][:verify_width]" in capture_section
    assert "verify_output = self._verify_output_buffer[:selected_count]" in capture_section
    assert "verify_output.copy_(self._verify_output_stack[:selected_count], non_blocking=False)" in capture_section
    assert 'inputs={"probe": self._verify_probe}' in capture_section
    assert "output=verify_output" in capture_section
    assert "self._verify_probe.detach().cpu()" not in capture_section
    assert ".float().clone()" not in capture_section
    assert "torch.stack([tensor.detach().cpu() for tensor in selected], dim=0)" not in capture_section
    assert "output_cpu[:, :256]" not in capture_section
    assert "self._verify_probe = None" in teardown_section
    assert "self._request_output_groups = []" in teardown_section
    assert "self._verify_output_stack = None" in teardown_section
    assert "self._verify_output_buffer = None" in teardown_section
