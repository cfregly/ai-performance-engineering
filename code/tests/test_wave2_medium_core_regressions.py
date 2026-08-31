from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

from ch20.optimized_pipeline_sequential import OptimizedPipelineOverlapBenchmark
from core.analysis.report_generator import generate_report_from_metrics
from core.api.registry import ApiRoute
from core.benchmark.expectations import METRIC_DIRECTIONS, ExpectationsStore
from core.scripts.utilities import probe_hardware_capabilities
from dashboard.api import server

CODE_ROOT = Path(__file__).resolve().parents[1]


def test_w2_074_b200_utilization_uses_published_dense_per_gpu_peaks() -> None:
    from ch19 import fp8_compiled_matmul

    assert fp8_compiled_matmul._dense_tensor_core_peaks(10, 0) == (4_500.0, 2_250.0)
    assert fp8_compiled_matmul._dense_tensor_core_peaks(10, 3) is None
    assert fp8_compiled_matmul._dense_tensor_core_peaks(12, 1) is None
    assert "not B200" in fp8_compiled_matmul._architecture_name(12, 1)

    source = Path(fp8_compiled_matmul.__file__).read_text(encoding="utf-8")
    assert "same GPU as B200" not in source
    assert "tflops_fp8 / 450.0" not in source


def test_w2_075_pipeline_records_cross_stream_tensor_consumers(
    monkeypatch,
) -> None:
    log: list[tuple[str, str, str]] = []

    class FakeTensor:
        def __init__(self, name: str) -> None:
            self.name = name

        def record_stream(self, stream) -> None:
            log.append(("record_stream", self.name, stream.name))

    class FakeStream:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_event(self, event) -> None:
            log.append(("wait_event", self.name, event.name))

        def wait_stream(self, stream) -> None:
            log.append(("wait_stream", self.name, stream.name))

    class FakeEvent:
        def __init__(self, name: str) -> None:
            self.name = name

        def record(self, stream) -> None:
            log.append(("event_record", self.name, stream.name))

    class FakeStage:
        def __init__(self, output: FakeTensor) -> None:
            self.output = output

        def __call__(self, _stage_input) -> FakeTensor:
            return self.output

    stage0_output = FakeTensor("stage0-output")
    stage1_output = FakeTensor("stage1-output")
    stage0_stream = FakeStream("stage0")
    stage1_stream = FakeStream("stage1")
    event0 = FakeEvent("event0")
    event1 = FakeEvent("event1")
    current_stream = FakeStream("current")

    benchmark = object.__new__(OptimizedPipelineOverlapBenchmark)
    benchmark.stages = [FakeStage(stage0_output), FakeStage(stage1_output)]
    benchmark.microbatches = [FakeTensor("input")]
    benchmark.stage_events = [[event0], [event1]]
    benchmark.stage_streams = [stage0_stream, stage1_stream]
    benchmark._stage_outputs = [[None], [None]]
    benchmark._last_outputs = [FakeTensor("unused")]
    benchmark._microbatch_groups = [(0, benchmark.microbatches[0])]
    benchmark._stage_schedule = [
        (0, benchmark.stages[0], stage0_stream, [event0], benchmark._stage_outputs[0]),
        (1, benchmark.stages[1], stage1_stream, [event1], benchmark._stage_outputs[1]),
    ]
    benchmark.num_stages = 2
    benchmark.num_microbatches = 1
    benchmark.device = "cuda:0"
    benchmark._last_output_count = 0

    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: current_stream)

    outputs = benchmark._run_pipelined_once()

    assert outputs == [stage1_output]
    wait_index = log.index(("wait_event", "stage1", "event0"))
    record_index = log.index(("record_stream", "stage0-output", "stage1"))
    assert wait_index < record_index


def test_w2_076_unpadded_chapter_prefix_maps_to_zero_padded_targets() -> None:
    report = generate_report_from_metrics({"ch2_nvlink_bandwidth_gbs": 725.0})

    assert "### CH02: Hardware Overview" in report
    assert "| Nvlink Bandwidth Gbs | 725.00 GB/s | 725.00 GB/s | [OK] PASS |" in report


def test_w2_077_bytes_per_second_metrics_participate_in_regression_checks(
    tmp_path: Path,
) -> None:
    assert METRIC_DIRECTIONS["baseline_throughput.bytes_per_s"] == "higher"
    assert METRIC_DIRECTIONS["best_optimized_throughput.bytes_per_s"] == "higher"

    store = ExpectationsStore(tmp_path, "test")
    store._data["examples"]["bandwidth"] = {
        "example": "bandwidth",
        "type": "python",
        "metrics": {
            "baseline_throughput.bytes_per_s": 1_000.0,
            "best_optimized_throughput.bytes_per_s": 2_000.0,
        },
        "metadata": {},
    }

    evaluation = store.evaluate(
        "bandwidth",
        {
            "baseline_throughput.bytes_per_s": 600.0,
            "best_optimized_throughput.bytes_per_s": 1_000.0,
        },
    )

    assert evaluation is not None
    assert {item["metric"] for item in evaluation.regressions} == {
        "baseline_throughput.bytes_per_s",
        "best_optimized_throughput.bytes_per_s",
    }


def test_w2_078_w2_079_tma_box_limits_follow_driver_contract() -> None:
    from core.scripts.utilities.generate_arch_detection_header import (
        emit_capability_entries,
    )

    for major, minor in ((9, 0), (10, 0), (10, 3), (12, 0), (12, 1)):
        props = SimpleNamespace(major=major, minor=minor)
        assert probe_hardware_capabilities._derive_tma_limits(props) == (256, 256, 256)

    assert probe_hardware_capabilities._derive_tma_limits(
        SimpleNamespace(major=8, minor=9)
    ) == (0, 0, 0)

    header = (CODE_ROOT / "core/common/headers/arch_detection.cuh").read_text(
        encoding="utf-8"
    )
    assert "props.major >= 9" in header
    assert "cached_limits = {256, 256, 256};" in header
    assert "entry->tma" not in header
    assert "{1024, 128, 128}" not in header

    generated_entry = emit_capability_entries(
        {
            "sm_90": {
                "architecture": "hopper",
                "tma": {"max_1d": 1024, "max_2d_width": 128, "max_2d_height": 128},
                "cluster": {
                    "supports_clusters": True,
                    "has_dsmem": True,
                    "max_cluster_size": 4,
                },
            }
        }
    )
    assert "9, 0" in generated_entry
    assert "1024" not in generated_entry


def test_w2_080_sync_dashboard_handler_runs_via_worker_thread(monkeypatch) -> None:
    calls: list[tuple[object, tuple[object, ...]]] = []

    async def fake_to_thread(function, *args):
        calls.append((function, args))
        return function(*args)

    monkeypatch.setattr(server.asyncio, "to_thread", fake_to_thread)
    route = ApiRoute("GET", "/test", "test.route", lambda params: {"params": params})
    endpoint = server._make_endpoint(route)
    request = SimpleNamespace(method="GET", query_params={"value": "1"})

    response = asyncio.run(endpoint(request))

    assert response.status_code == 200
    assert calls == [(route.handler, ({"value": "1"},))]
    server_source = Path(server.__file__).read_text(encoding="utf-8")
    assert "await asyncio.to_thread(get_engine().gpu.info)" in server_source


def test_w2_081_unattributable_sdpa_speedup_is_not_a_b200_expectation() -> None:
    expectation_path = CODE_ROOT / "labs/cudnn_sdpa_bench/expectations_b200.json"
    payload = json.loads(expectation_path.read_text(encoding="utf-8"))
    readme = (expectation_path.parent / "README.md").read_text(encoding="utf-8")

    assert payload["examples"] == {}
    assert "backend attribution unverified" in readme
    assert "Fresh GPU verification and profiler evidence are required" in readme


def test_w2_091_software_pipeline_pair_controls_copy_and_compute_width() -> None:
    source = (
        CODE_ROOT / "labs/software_pipelining/software_pipelining_kernels.cu"
    ).read_text(encoding="utf-8")
    readme = (CODE_ROOT / "labs/software_pipelining/README.md").read_text(
        encoding="utf-8"
    )
    baseline = source.split("__global__ void baseline_tile_pipeline_kernel", 1)[1].split(
        "__global__ void optimized_tile_pipeline_kernel", 1
    )[0]
    launcher = source.split("torch::Tensor launch_common", 1)[1].split(
        "}  // namespace", 1
    )[0]

    assert "warp.meta_group_rank() == 0" not in baseline
    assert "for (int v = threadIdx.x; v < kVecPerTile; v += blockDim.x)" in baseline
    assert "lhs_shared[v] = lhs_global[v]" in baseline
    assert "rhs_shared[v] = rhs_global[v]" in baseline
    assert "if (use_pipeline)" in launcher
    assert "optimized_tile_pipeline_kernel<<<grid, block" in launcher
    assert "optimized_tile_pipeline_tma_kernel<<<" not in launcher
    assert "2.22x" not in readme
    assert "No current speedup is claimed" in readme
    assert "differ only in serialized versus two-stage scheduling" in readme


def test_w2_092_mcp_client_drains_large_server_stderr(tmp_path: Path) -> None:
    from mcp.mcp_client import RobustMCPClient

    server_script = tmp_path / "stderr_server.py"
    server_script.write_text(
        """
import json
import sys

for index in range(4096):
    print(f"diagnostic-{index}-" + "x" * 240, file=sys.stderr)
sys.stderr.flush()

for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}}), flush=True)
""".lstrip(),
        encoding="utf-8",
    )
    client = RobustMCPClient([sys.executable, str(server_script)], timeout=5.0)

    try:
        client.start()
        assert client._stderr_thread is not None
        assert client._stderr_thread.is_alive()
        assert client._stderr_tail
        assert client._stderr_tail[-1].startswith("diagnostic-4095-")
    finally:
        client.stop()


def test_w2_093_cuda_dtype_inference_uses_word_boundaries(tmp_path: Path) -> None:
    from mcp.mcp_server import _infer_dtype_from_cu

    cases = {
        "bf16.cu": ("__global__ void k(__nv_bfloat16* x) {}", "bfloat16"),
        "fp16.cu": ("__global__ void k(half* x) {}", "float16"),
        "fp64.cu": ("__global__ void k(double* x) {}", "float64"),
        "boundary.cu": ("int half_width = 2;", "float32"),
    }
    for filename, (text, expected) in cases.items():
        path = tmp_path / filename
        path.write_text(text, encoding="utf-8")
        assert _infer_dtype_from_cu(path) == expected


def test_w2_094_active_ib_link_without_rate_sample_is_unavailable(monkeypatch) -> None:
    from monitoring.cluster_monitor import ClusterAggregator, NetworkCollector, NodeMetrics

    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="State: Active\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metrics = NetworkCollector.collect()

    assert commands == [["ibstat"]]
    assert metrics["ib_rx_gbps"] is None
    assert metrics["ib_tx_gbps"] is None
    assert "no sampled byte-counter delta" in metrics["ib_telemetry_status"]

    aggregator = ClusterAggregator()
    for node_id in range(5):
        aggregator.report_node_metrics(
            NodeMetrics(hostname=f"node-{node_id}", node_id=node_id, timestamp=1.0)
        )
    cluster = aggregator.aggregate()
    assert "Consider enabling InfiniBand for better scaling" not in cluster.recommendations


def test_w2_095_cpu_utilization_uses_sample_delta(monkeypatch) -> None:
    from monitoring.cluster_monitor import SystemCollector

    samples = iter([(100, 1_000), (140, 1_200)])
    monkeypatch.setattr(SystemCollector, "_previous_cpu_sample", None)
    monkeypatch.setattr(
        SystemCollector,
        "_read_cpu_sample",
        staticmethod(lambda: next(samples)),
    )

    first = SystemCollector.collect()
    second = SystemCollector.collect()

    assert first["cpu_utilization_pct"] is None
    assert first["cpu_telemetry_status"] == "warming_up"
    assert second["cpu_utilization_pct"] == 80.0
    assert second["cpu_telemetry_status"] == "ok"


def test_w2_096_exporter_uses_device_wide_memory_for_utilization(monkeypatch) -> None:
    from monitoring import prometheus_exporter

    gib = 1024**3
    fake_cuda = SimpleNamespace(
        mem_get_info=lambda _gpu_id: (60 * gib, 80 * gib),
        memory_allocated=lambda _gpu_id: 2 * gib,
        memory_reserved=lambda _gpu_id: 3 * gib,
        get_device_properties=lambda _gpu_id: SimpleNamespace(multi_processor_count=100),
    )
    monkeypatch.setattr(prometheus_exporter, "torch", SimpleNamespace(cuda=fake_cuda))
    collector = object.__new__(prometheus_exporter.MetricsCollector)
    collector.cuda_available = True
    collector.device_count = 1

    metrics = collector.collect_gpu_metrics()

    assert metrics['gpu_memory_used_gb{gpu="0"}'] == 20.0
    assert metrics['gpu_memory_total_gb{gpu="0"}'] == 80.0
    assert metrics['gpu_memory_utilization{gpu="0"}'] == 0.25
    assert metrics['gpu_process_memory_allocated_gb{gpu="0"}'] == 2.0
    assert metrics['gpu_process_memory_reserved_gb{gpu="0"}'] == 3.0
    assert 'gpu_memory_allocated_gb{gpu="0"}' not in metrics
    assert 'gpu_memory_reserved_gb{gpu="0"}' not in metrics


def test_w2_124_custom_matmul_lab_makes_only_measured_comparisons() -> None:
    source = (CODE_ROOT / "labs/custom_vs_cublas/run_lab.py").read_text(
        encoding="utf-8"
    )

    assert "MMA hardware handles dependencies internally" not in source
    assert "+43%" not in source
    assert "Measured comparisons for this run only:" in source
    assert "bottleneck attribution require device profiling" in source


def test_w2_127_router_verification_uses_generated_token_ids() -> None:
    from labs.dynamic_router.baseline_dynamic_router_vllm import (
        BaselineDynamicRouterVllmBenchmark,
    )
    from labs.dynamic_router.optimized_dynamic_router_vllm import (
        OptimizedDynamicRouterVllmBenchmark,
    )
    from labs.dynamic_router.verification import VERIFICATION_OUTPUT_KEY
    from labs.dynamic_router.vllm_runner import _collect_verification_output_token_ids

    engines = {
        "gpu0": SimpleNamespace(_completed_output_token_ids={"req-1": (7,)}),
        "gpu1": SimpleNamespace(_completed_output_token_ids={"req-0": (11, 12)}),
    }
    framed_tokens = _collect_verification_output_token_ids(
        engines, ["req-0", "req-1"]
    )
    assert framed_tokens == [2, 11, 12, 1, 7]

    for benchmark_type in (
        BaselineDynamicRouterVllmBenchmark,
        OptimizedDynamicRouterVllmBenchmark,
    ):
        benchmark = benchmark_type()
        benchmark._summary = {
            "ttft_ms_p95": 999.0,
            "tpot_tok_per_step_gpu0": 0.01,
            VERIFICATION_OUTPUT_KEY: framed_tokens,
        }
        benchmark._summary_ready = True

        benchmark.capture_verification_payload()

        assert benchmark.get_verify_output().tolist() == [[2.0, 11.0, 12.0, 1.0, 7.0]]
        assert benchmark.get_output_tolerance() == (0.0, 0.0)


def test_w2_129_fp8_kv_tolerance_matches_e4m3_error_scale() -> None:
    from labs.kv_optimization.verification import FP8_KV_OUTPUT_TOLERANCE

    rtol, atol = FP8_KV_OUTPUT_TOLERANCE
    reference = torch.tensor([[-1.0, -0.25, 0.25, 1.0]], dtype=torch.float32)
    assert not torch.allclose(torch.zeros_like(reference), reference, rtol=rtol, atol=atol)
    assert not torch.allclose(reference * 0.5, reference, rtol=rtol, atol=atol)

    if hasattr(torch, "float8_e4m3fn"):
        torch.manual_seed(42)
        cache_values = torch.randn(8, 4, 32, dtype=torch.bfloat16)
        scale = 448.0 / cache_values.abs().amax().float()
        restored = (cache_values * scale).to(torch.float8_e4m3fn).float() / scale
        torch.testing.assert_close(
            restored,
            cache_values.float(),
            rtol=rtol,
            atol=atol,
        )

    for name in ("baseline_kv_standard.py", "optimized_kv_standard.py"):
        source = (CODE_ROOT / "labs/kv_optimization" / name).read_text(
            encoding="utf-8"
        )
        assert "output_tolerance=FP8_KV_OUTPUT_TOLERANCE" in source
        assert "output_tolerance=(0.1, 1.0)" not in source
