from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch


def test_w2_055_sparse_roofline_counts_only_attended_query_key_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ch14.optimized_flex_attention_sparse import (
        FlexAttentionSparseBenchmark,
        _causal_window_key_pairs,
    )
    from core.benchmark import metrics as benchmark_metrics

    captured: dict[str, Any] = {}

    def capture_metrics(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(benchmark_metrics, "compute_roofline_metrics", capture_metrics)
    benchmark = FlexAttentionSparseBenchmark()

    result = benchmark.get_custom_metrics()

    assert result == captured
    attended_pairs = _causal_window_key_pairs(
        benchmark.seq_len,
        benchmark.window_size,
    )
    assert attended_pairs == sum(
        min(query_position + 1, benchmark.window_size)
        for query_position in range(benchmark.seq_len)
    )
    assert captured["total_flops"] == (
        4
        * benchmark.batch_size
        * benchmark.num_heads
        * benchmark.head_dim
        * attended_pairs
    )
    dense_flops = (
        4
        * benchmark.batch_size
        * benchmark.num_heads
        * benchmark.head_dim
        * benchmark.seq_len
        * benchmark.seq_len
    )
    assert captured["total_flops"] < dense_flops / 16


def test_w2_058_ptq_tolerance_rejects_zero_and_unrelated_outputs() -> None:
    from ch16.awq_gptq_smoothquant_benchmarks import PTQ_OUTPUT_TOLERANCE

    reference = torch.tensor([[-0.75, -0.25, 0.25, 0.75]])
    rtol, atol = PTQ_OUTPUT_TOLERANCE

    assert PTQ_OUTPUT_TOLERANCE == (0.25, 0.15)
    assert not torch.allclose(torch.zeros_like(reference), reference, rtol=rtol, atol=atol)
    assert not torch.allclose(-reference, reference, rtol=rtol, atol=atol)


def test_w2_059_dense_attention_family_uses_bounded_shared_tolerance() -> None:
    from ch16.dense_attention_accuracy import DENSE_ATTENTION_OUTPUT_TOLERANCE

    reference = torch.tensor([[-0.25, -0.10, 0.10, 0.25]])
    rtol, atol = DENSE_ATTENTION_OUTPUT_TOLERANCE

    assert DENSE_ATTENTION_OUTPUT_TOLERANCE == (5e-2, 5e-2)
    assert not torch.allclose(torch.zeros_like(reference), reference, rtol=rtol, atol=atol)

    ch16 = Path(__file__).resolve().parents[1] / "ch16"
    for filename in (
        "baseline_dense_attention_flash.py",
        "optimized_dense_attention_flash.py",
        "optimized_dense_attention_flash_blackwell_variant.py",
    ):
        source = (ch16 / filename).read_text(encoding="utf-8")
        assert "output_tolerance=DENSE_ATTENTION_OUTPUT_TOLERANCE" in source
        assert "output_tolerance=(0.1, 1.0)" not in source


class _FakePrometheusMetric:
    def __init__(self) -> None:
        self.labels_value: dict[str, object] = {}
        self.events: list[tuple[str, dict[str, object], object]] = []

    def labels(self, **labels: object) -> _FakePrometheusMetric:
        self.labels_value = labels
        return self

    def set(self, value: object) -> None:
        self.events.append(("set", dict(self.labels_value), value))

    def inc(self, value: object) -> None:
        self.events.append(("inc", dict(self.labels_value), value))

    def info(self, value: object) -> None:
        self.events.append(("info", dict(self.labels_value), value))


def test_w2_060_dcgm_updates_nvlink_error_delta_for_every_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ch16.dcgm_prometheus_exporter as exporter_module

    exporter = object.__new__(exporter_module.DCGMPrometheusExporter)
    exporter.hostname = "test-host"
    exporter.dcgm_ready = False
    exporter.prev_nvlink_errors = {0: 1, 1: 2}
    exporter._gpu_info_cache = {"hostname": exporter.hostname}
    metric_names = (
        "gpu_utilization",
        "gpu_memory_used",
        "gpu_memory_total",
        "gpu_memory_utilization",
        "gpu_temperature",
        "gpu_power",
        "gpu_sm_clock",
        "gpu_memory_clock",
        "pcie_tx_throughput",
        "pcie_rx_throughput",
        "nvlink_tx_throughput",
        "nvlink_rx_throughput",
        "encoder_utilization",
        "decoder_utilization",
        "pcie_gen",
        "pcie_width",
        "retired_pages_sbe",
        "retired_pages_dbe",
        "sm_active",
        "sm_occupancy",
        "tensor_active",
        "dram_active",
        "fp64_active",
        "fp32_active",
        "fp16_active",
        "nvlink_errors",
        "gpu_info",
    )
    for name in metric_names:
        setattr(exporter, name, _FakePrometheusMetric())

    def gpu(gpu_id: int, errors: int) -> exporter_module.GPUMetrics:
        return exporter_module.GPUMetrics(
            gpu_id=gpu_id,
            gpu_util=0.0,
            mem_util=0.0,
            mem_used=1,
            mem_total=2,
            temperature=0.0,
            power_usage=0.0,
            pcie_tx=0,
            pcie_rx=0,
            nvlink_tx=0,
            nvlink_rx=0,
            sm_clock=0,
            memory_clock=0,
            encoder_util=0.0,
            decoder_util=0.0,
            nvlink_errors=errors,
        )

    exporter.collect_nvidia_smi_metrics = lambda: [gpu(0, 4), gpu(1, 7)]
    monkeypatch.setattr(exporter_module, "NVML_AVAILABLE", False)

    exporter.update_metrics()

    assert exporter.prev_nvlink_errors == {0: 4, 1: 7}
    increments = [
        (event[1]["gpu"], event[2])
        for event in exporter.nvlink_errors.events
        if event[0] == "inc"
    ]
    assert increments == [("gpu0", 3), ("gpu1", 5)]


def test_w2_061_blackwell_generate_passes_cache_to_prefill_and_decode() -> None:
    import inspect

    from ch16.inference_optimizations_blackwell import BlackwellInferencePipeline

    class FakeCache:
        def __init__(self) -> None:
            self.clear_count = 0

        def clear(self) -> None:
            self.clear_count += 1

    class CacheAwareModel:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, object]] = []

        def __call__(self, tokens: torch.Tensor, *, kv_cache: object) -> torch.Tensor:
            self.calls.append((tokens.clone(), kv_cache))
            logits = torch.zeros(tokens.size(0), tokens.size(1), 4)
            logits[..., len(self.calls) % 4] = 1.0
            return logits

    pipeline = BlackwellInferencePipeline.__new__(BlackwellInferencePipeline)
    pipeline.model = CacheAwareModel()
    pipeline.kv_cache = FakeCache()
    pipeline._next_token_buffer = None
    pipeline._next_token_values = None
    pipeline._generated_token_buffer = None
    prompt = torch.tensor([[3, 2, 1]])

    output = pipeline.generate(prompt, max_new_tokens=3)

    assert pipeline.kv_cache.clear_count == 1
    assert [tuple(tokens.shape) for tokens, _ in pipeline.model.calls] == [
        (1, 3),
        (1, 1),
        (1, 1),
    ]
    assert all(cache is pipeline.kv_cache for _, cache in pipeline.model.calls)
    assert output.shape == (1, 6)

    benchmark_source = inspect.getsource(BlackwellInferencePipeline.benchmark)
    assert "self._forward_with_kv_cache(input_ids)" in benchmark_source
    assert "self.model(input_ids)" not in benchmark_source


def test_w2_062_radix_cache_extensions_use_copy_on_write() -> None:
    from ch16.radix_attention_example import KVCache

    keys = torch.zeros(4, 1, 1)
    values = torch.zeros_like(keys)
    keys[:2, 0, 0] = torch.tensor([1.0, 2.0])
    values[:2, 0, 0] = torch.tensor([10.0, 20.0])
    prefix = KVCache(keys=keys, values=values, seq_len=2, capacity=4)
    left_ref = prefix.clone_ref()
    right_ref = prefix.clone_ref()

    left = left_ref.append(torch.tensor([[[3.0]]]), torch.tensor([[[30.0]]]))
    right = right_ref.append(torch.tensor([[[4.0]]]), torch.tensor([[[40.0]]]))

    torch.testing.assert_close(prefix.key_view[:, 0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(left.key_view[:, 0, 0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(right.key_view[:, 0, 0], torch.tensor([1.0, 2.0, 4.0]))
    assert left.keys.data_ptr() != prefix.keys.data_ptr()
    assert right.keys.data_ptr() != prefix.keys.data_ptr()
    assert left.keys.data_ptr() != right.keys.data_ptr()


@pytest.mark.parametrize(
    ("major", "minor", "name", "expected"),
    (
        (12, 0, "NVIDIA RTX PRO 6000 Blackwell", "blackwell_sm12x"),
        (12, 1, "NVIDIA GB10", "blackwell_sm12x"),
        (10, 0, "NVIDIA B200", "blackwell_sm100"),
        (10, 3, "NVIDIA B300", "blackwell_sm100"),
        (12, 0, "NVIDIA B200 mislabeled", "blackwell_sm12x"),
        (10, 0, "NVIDIA GB10 mislabeled", "blackwell_sm100"),
    ),
)
def test_w2_063_sm12_is_never_classified_as_b200(
    monkeypatch: pytest.MonkeyPatch,
    major: int,
    minor: int,
    name: str,
    expected: str,
) -> None:
    from types import SimpleNamespace

    import ch16.synthetic_moe_inference_benchmark as moe

    monkeypatch.setattr(moe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        moe.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(major=major, minor=minor, name=name),
    )

    assert moe.detect_device_flavor() == expected


def test_w2_064_065_vllm_monitoring_preserves_templates_and_valid_promql() -> None:
    import yaml

    from ch16.monitoring_config import AlertThresholds, MetricNames
    from ch16.vllm_monitoring import build_vllm_monitoring_bundle

    bundle = build_vllm_monitoring_bundle(MetricNames(), AlertThresholds())
    alerts = yaml.safe_load(bundle.alerting_rules)
    first_annotations = alerts["groups"][0]["rules"][0]["annotations"]

    assert "{{ $labels.model_name }}" in first_annotations["summary"]
    assert "{{ $labels.instance }}" in first_annotations["summary"]
    assert "{{ $value }}" in first_annotations["description"]

    panels = {panel.get("id"): panel for panel in bundle.grafana_dashboard["panels"]}
    for panel_id in (9, 10):
        expression = panels[panel_id]["targets"][0]["expr"]
        assert '\\"' not in expression
        assert 'model_name=~"$model"' in expression
        assert 'instance=~"$instance"' in expression
