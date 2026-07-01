"""Smoke tests for the shared Chapter 15 prefill/decode disaggregation wrappers."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

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
        "def _prefill_into_decode_kv", maxsplit=1
    )[0]
    direct_prefill_section = source.split("def _prefill_into_decode_kv", maxsplit=1)[
        1
    ].split(
        "def _decode_into_output_shard", maxsplit=1
    )[0]
    decode_section = source.split("def _decode_into_output_shard", maxsplit=1)[
        1
    ].split(
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
    assert "self._prefill_weight_t: dict[int, torch.Tensor] = {}" in source
    assert "self._decode_weight_t: dict[int, torch.Tensor] = {}" in source
    assert "self._decode_token_staging: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}" in source
    assert "self._prefill_weight_t = {}" in setup_section
    assert "self._decode_weight_t = {}" in setup_section
    assert "self._decode_token_staging = {}" in setup_section
    assert "self._request_groups = []" in setup_section
    assert "self._request_output_groups = []" in setup_section
    assert "self._request_groups.extend(" in setup_section
    assert "self._request_output_groups = [" in setup_section
    assert "batch_slice[idx : idx + 1]" in setup_section
    assert "self._output_shards = []" in setup_section
    assert "self._output_shards = [" in setup_section
    assert "torch.empty(self.hidden_size, device=decode_device, dtype=torch.bfloat16)" in setup_section
    assert "torch.empty(0)" not in setup_section
    assert "self._output_shard_count = 0" in source
    assert "self._output_shard_count = len(self._output_shards)" in setup_section
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
    assert "first_decode_token = torch.empty(" in setup_section
    assert "self._decode_token_staging[staging_key] = (" in setup_section
    assert "self._prefill_weight_t[id(prefill_model)] = prefill_model.weight.detach().t()" in setup_section
    assert "self._decode_weight_t[id(decode_model)] = decode_model.weight.detach().t()" in setup_section
    assert "def _staging_numel(" in source
    assert "def _decode_token_buffer_pair(" in source
    assert "or buffer.numel() < numel" in source
    assert "return buffer[:numel].view(shape)" in source
    assert "if self.use_host_staging or not self._device_matches(" in direct_prefill_section
    assert "weight_t = self._prefill_weight_t.get(id(prefill_model))" in direct_prefill_section
    assert "output_shape = request.shape[:-1] + torch.Size((int(weight_t.shape[1]),))" in direct_prefill_section
    assert "decode_buf = self._decode_staging_view(" in direct_prefill_section
    assert "torch.matmul(request, weight_t, out=decode_buf)" in direct_prefill_section
    assert "decode_buf.add_(bias.detach())" in direct_prefill_section
    assert "prefill_model.weight.t()" not in direct_prefill_section
    assert "decode_weight_t = self._decode_weight_t.get(id(decode_model))" in decode_section
    assert "token_buffers = self._decode_token_buffer_pair(" in decode_section
    assert "if self.decode_length <= 0:" in decode_section
    assert "last_step_idx = self.decode_length - 1" in decode_section
    assert "output_state = output_shard.view(1, 1, self.hidden_size)" in decode_section
    assert "for step_idx in self._decode_step_range:" in decode_section
    assert "if step_idx == last_step_idx" in decode_section
    assert "else token_buffers[step_idx & 1]" in decode_section
    assert "torch.matmul(token_state, decode_weight_t, out=next_state)" in decode_section
    assert "next_state.add_(bias.detach())" in decode_section
    assert "output_shard.copy_(token_state.reshape(-1), non_blocking=True)" in decode_section
    assert "decode_buf.shape != prefill_out.shape" not in handoff_section
    assert "host_buf.shape != prefill_out.shape" not in handoff_section
    assert (
        "if not self.use_host_staging and self._device_matches("
        in handoff_section
    )
    assert "prefill_out.cpu()" not in handoff_section
    assert "kv_cpu.to(decode_device)" not in handoff_section
    assert "host_buf.copy_(prefill_out, non_blocking=False)" in handoff_section
    assert "decode_buf.copy_(host_buf, non_blocking=False)" in handoff_section
    assert "decode_buf.copy_(prefill_out, non_blocking=True)" in handoff_section
    assert "with torch.inference_mode():" in benchmark_section
    assert "outputs = self._output_shards" in benchmark_section
    assert "self._output_shard_count != self.batch_size" in benchmark_section
    assert "len(outputs)" not in benchmark_section
    assert "enumerate(" not in benchmark_section
    assert "self._request_groups" in benchmark_section
    assert "self._request_output_groups" in benchmark_section
    assert "output_idx," in benchmark_section
    assert "decode_device," in benchmark_section
    assert "prefill_model," in benchmark_section
    assert "decode_model," in benchmark_section
    assert ") in self._request_output_groups:" in benchmark_section
    assert "kv_decode = self._prefill_into_decode_kv(" in benchmark_section
    assert "if kv_decode is None:" in benchmark_section
    assert "prefill_out = prefill_model(request)" in benchmark_section
    assert "self._decode_into_output_shard(" in benchmark_section
    assert "outputs[output_idx]," in benchmark_section
    assert "outputs[output_idx] = token_state.squeeze(0).squeeze(0)" not in benchmark_section
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
    assert "self._output_shard_count = 0" in teardown_section
    assert ".float().clone()" not in capture_section
    assert "torch.stack([tensor.detach().cpu() for tensor in selected], dim=0)" not in capture_section
    assert "output_cpu[:, :256]" not in capture_section
    assert "self._verify_probe = None" in teardown_section
    assert "self._prefill_weight_t = {}" in teardown_section
    assert "self._decode_weight_t = {}" in teardown_section
    assert "self._decode_token_staging = {}" in teardown_section
    assert "self._request_output_groups = []" in teardown_section
    assert "self._verify_output_stack = None" in teardown_section
    assert "self._verify_output_buffer = None" in teardown_section


def test_prefill_decode_disagg_handoff_reuses_larger_capacity() -> None:
    module = importlib.import_module("ch15.prefill_decode_disagg_common")
    bench = module.PrefillDecodeDisaggBenchmark(
        use_host_staging=True,
        multi_gpu=False,
        label="test",
    )
    decode_device = torch.device("cpu")

    large = torch.randn((1, 5, 4), dtype=torch.bfloat16)
    bench._handoff_kv(large, decode_device)
    decode_ptr = bench._handoff_staging["cpu"].data_ptr()
    host_ptr = bench._host_staging["cpu"].data_ptr()

    small = torch.randn((1, 2, 4), dtype=torch.bfloat16)
    out_small = bench._handoff_kv(small, decode_device)

    assert out_small.shape == small.shape
    torch.testing.assert_close(out_small, small)
    assert bench._handoff_staging["cpu"].data_ptr() == decode_ptr
    assert bench._host_staging["cpu"].data_ptr() == host_ptr
    assert bench._handoff_staging["cpu"].numel() == large.numel()
    assert bench._host_staging["cpu"].numel() == large.numel()

    grown = torch.randn((1, 6, 4), dtype=torch.bfloat16)
    out_grown = bench._handoff_kv(grown, decode_device)

    assert out_grown.shape == grown.shape
    torch.testing.assert_close(out_grown, grown)
    assert bench._handoff_staging["cpu"].numel() == grown.numel()
    assert bench._host_staging["cpu"].numel() == grown.numel()


def test_prefill_decode_disagg_direct_prefill_writes_into_reusable_decode_buffer() -> None:
    module = importlib.import_module("ch15.prefill_decode_disagg_common")
    bench = module.PrefillDecodeDisaggBenchmark(
        use_host_staging=False,
        multi_gpu=False,
        label="test",
    )
    model = torch.nn.Linear(4, 5, bias=False).eval()
    request = torch.randn((1, 3, 4), dtype=torch.float32)
    bench._prefill_weight_t[id(model)] = model.weight.detach().t()

    out_large = bench._prefill_into_decode_kv(model, request, torch.device("cpu"))

    assert out_large is not None
    torch.testing.assert_close(out_large, model(request))
    decode_ptr = bench._handoff_staging["cpu"].data_ptr()

    out_small = bench._prefill_into_decode_kv(
        model,
        request[:, :2],
        torch.device("cpu"),
    )

    assert out_small is not None
    torch.testing.assert_close(out_small, model(request[:, :2]))
    assert bench._handoff_staging["cpu"].data_ptr() == decode_ptr


def test_prefill_decode_disagg_decode_reuses_token_buffers_and_output_shard() -> None:
    module = importlib.import_module("ch15.prefill_decode_disagg_common")
    cfg = module.PrefillDecodeDisaggConfig(
        batch_size=1,
        prefill_length=3,
        decode_length=3,
        hidden_size=5,
    )
    bench = module.PrefillDecodeDisaggBenchmark(
        use_host_staging=False,
        multi_gpu=False,
        label="test",
        cfg=cfg,
    )
    model = torch.nn.Linear(5, 5, bias=False).eval()
    bench._decode_weight_t[id(model)] = model.weight.detach().t()
    kv_decode = torch.randn((1, 3, 5), dtype=torch.float32)
    output = torch.empty(5, dtype=torch.float32)

    bench._decode_into_output_shard(model, kv_decode, torch.device("cpu"), output)

    expected = kv_decode[:, -1:, :]
    for _ in range(cfg.decode_length):
        expected = model(expected)
    torch.testing.assert_close(output, expected.reshape(-1))
    first_ptrs = tuple(buf.data_ptr() for buf in bench._decode_token_staging["cpu"])

    bench._decode_into_output_shard(model, kv_decode[:, :2], torch.device("cpu"), output)

    second_ptrs = tuple(buf.data_ptr() for buf in bench._decode_token_staging["cpu"])
    assert second_ptrs == first_ptrs


def test_prefill_decode_disagg_zero_decode_length_copies_last_prefill_token() -> None:
    module = importlib.import_module("ch15.prefill_decode_disagg_common")
    cfg = module.PrefillDecodeDisaggConfig(
        batch_size=1,
        prefill_length=3,
        decode_length=0,
        hidden_size=5,
    )
    bench = module.PrefillDecodeDisaggBenchmark(
        use_host_staging=False,
        multi_gpu=False,
        label="test",
        cfg=cfg,
    )
    model = torch.nn.Linear(5, 5, bias=False).eval()
    bench._decode_weight_t[id(model)] = model.weight.detach().t()
    kv_decode = torch.randn((1, 3, 5), dtype=torch.float32)
    output = torch.empty(5, dtype=torch.float32)

    bench._decode_into_output_shard(model, kv_decode, torch.device("cpu"), output)

    torch.testing.assert_close(output, kv_decode[:, -1:, :].reshape(-1))
