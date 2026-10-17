"""Regression tests for audit work unit r00 (residuals of W2-007, W2-033, W2-036, W2-037, W2-043, W2-086, W2-093)."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import ch04.ddp_nvlink_overlap as ddp_nvlink_overlap
import core.engine as engine_module
import core.harness.hardware_capabilities as hardware_capabilities
import core.llm as llm_module
import core.scripts.utilities.probe_hardware_capabilities as probe_hardware_capabilities
import mcp.mcp_server as mcp_server
from core.perf_core import PerfCore

CODE_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# W2-007: the overlap DDP module has no single-GPU mode
# ---------------------------------------------------------------------------


def test_w2_007_docstring_matches_the_multi_gpu_skip_contract() -> None:
    doc = ddp_nvlink_overlap.__doc__ or ""
    assert "Falls back to single-GPU" not in doc
    assert "There is no single-GPU mode" in doc
    assert "SKIPPED:" in doc

    if torch.cuda.device_count() < 2:
        with pytest.raises(RuntimeError, match=r"^SKIPPED: "):
            ddp_nvlink_overlap._require_root_peer_access()
    else:
        try:
            ddp_nvlink_overlap._require_root_peer_access()
        except RuntimeError as exc:
            assert str(exc).startswith("SKIPPED: Peer access unavailable")


def _expected_replica_averaged_delta(bench: Any) -> torch.Tensor:
    """Recompute the overlap benchmark's weight update in float64 on the CPU."""
    weights = [model.weight.detach().double().cpu() for model in bench.models]
    biases = [model.bias.detach().double().cpu() for model in bench.models]
    summed_grad = torch.zeros_like(weights[0])
    for micro_inputs in bench._inputs:
        for idx, x in enumerate(micro_inputs):
            weight = weights[idx].clone().requires_grad_(True)
            y = x.detach().double().cpu() @ weight.T + biases[idx]
            y.square().mean().backward()
            summed_grad += weight.grad
    return -1e-3 * summed_grad / len(bench.models)


def test_w2_007_get_benchmark_constructs_on_any_host_and_setup_skips() -> None:
    # The CUDA stream used to be created in __init__, so get_benchmark() raised a
    # plain CUDA error on a host without GPUs before setup() could report SKIPPED:.
    bench = ddp_nvlink_overlap.get_benchmark()
    assert bench.comm_stream is None
    try:
        if torch.cuda.device_count() < 2:
            with pytest.raises(RuntimeError, match=r"^SKIPPED: "):
                bench.setup()
            assert bench.comm_stream is None
            return

        try:
            bench.setup()
        except RuntimeError as exc:
            if str(exc).startswith("SKIPPED: Peer access unavailable"):
                pytest.skip(str(exc))
            raise
        assert bench.comm_stream is not None

        # W2-043 on real GPUs: one iteration applies the same replica-averaged
        # update, summed over every microbatch, to every model.
        before = [model.weight.detach().double().cpu() for model in bench.models]
        expected_delta = _expected_replica_averaged_delta(bench)
        bench.benchmark_fn()
        for idx, model in enumerate(bench.models):
            delta = model.weight.detach().double().cpu() - before[idx]
            torch.testing.assert_close(delta, expected_delta, rtol=1e-3, atol=2e-8)
        bench.capture_verification_payload()
    finally:
        bench.teardown()
    assert bench.comm_stream is None


# ---------------------------------------------------------------------------
# W2-036 and W2-037: ch02/hardware_info.py
# ---------------------------------------------------------------------------


@pytest.fixture
def hardware_info(monkeypatch: pytest.MonkeyPatch):
    """Import hardware_info without requiring its optional GPUtil dependency."""
    module_name = "ch02.hardware_info"
    sys.modules.pop(module_name, None)
    monkeypatch.setitem(sys.modules, "GPUtil", SimpleNamespace(getGPUs=lambda: []))
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)


def _memory_info(**overrides: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "max_shared_memory_per_block": 48 * 1024,
        "max_shared_memory_per_sm": 228 * 1024,
        "l2_cache_size": 126 * 1024 * 1024,
        "total_memory_gb": 178.4,
        "hbm3e_memory": True,
        "memory_bandwidth_tbps": 8.0,
        "max_unified_memory_tb": None,
    }
    info.update(overrides)
    return info


def test_w2_036_memory_hierarchy_prints_unified_memory_only_when_detected(hardware_info) -> None:
    # An HBM3e GPU without a reported unified-memory budget, such as a B200 in an
    # x86 HGX host, must not inherit a hard-coded Grace budget.
    hgx_lines = hardware_info._memory_specification_lines(_memory_info())
    assert "• HBM3e Bandwidth: 8.0 TB/s" in hgx_lines
    assert not any("Unified Memory" in line for line in hgx_lines)
    assert not any("30 TB" in line for line in hgx_lines)

    grace_lines = hardware_info._memory_specification_lines(
        _memory_info(max_unified_memory_tb=30)
    )
    assert "• Unified Memory Budget: 30 TB" in grace_lines

    no_bandwidth = hardware_info._memory_specification_lines(
        _memory_info(memory_bandwidth_tbps=None)
    )
    assert "• HBM3e Bandwidth: Unknown" in no_bandwidth


@pytest.fixture
def probe_record(monkeypatch: pytest.MonkeyPatch):
    """Return a builder that runs the real capability probe for a described GPU.

    Only the device and toolchain seams are replaced: the device properties, the
    ptxas TMA check, the cluster and DSMEM checks, and the CUDA driver host-link
    query. _probe_device itself sets the label, the features and the TMA flags,
    so each record follows the probe's own rules.
    """
    probe = probe_hardware_capabilities

    def _build(
        major: int,
        minor: int,
        *,
        ptxas_accepts_tma: bool = True,
        host_link: tuple[bool, bool] = (False, False),
    ) -> dict[str, Any]:
        props = SimpleNamespace(
            name=f"Test GPU sm_{major}{minor}",
            major=major,
            minor=minor,
            total_memory=96 * 1024**3,
            multi_processor_count=48,
            warp_size=32,
            max_threads_per_block=1024,
            max_threads_per_multi_processor=1536,
            shared_memory_per_block=48 * 1024,
            shared_memory_per_multiprocessor=100 * 1024,
            L2_cache_size=24 * 1024 * 1024,
        )
        compiler_result = (
            (True, None) if ptxas_accepts_tma else (False, "ptxas rejected tensormap.replace")
        )
        monkeypatch.setattr(probe.torch.cuda, "set_device", lambda index: None)
        monkeypatch.setattr(probe.torch.cuda, "get_device_properties", lambda index: props)
        monkeypatch.setattr(probe, "_probe_tma_compiler_support", lambda sm_tag: compiler_result)
        monkeypatch.setattr(probe, "_query_cluster_launch", lambda index: 1)
        monkeypatch.setattr(probe, "_probe_dsmem_support", lambda sm_tag: (True, None))
        monkeypatch.setattr(probe, "_host_link_flags", lambda device_props, index: host_link)
        return probe._probe_device(0)

    return _build


@pytest.mark.parametrize(
    ("major", "minor", "expected_first_line"),
    [
        (10, 0, " This is a Test GPU sm_100 GPU"),
        (10, 3, " This is a Test GPU sm_103 GPU"),
        # SM 12.0 is GeForce RTX 50 and RTX PRO Blackwell, which the probe
        # labels "blackwell_consumer".
        (12, 0, " This is a Test GPU sm_120 GPU"),
        (12, 1, " This is a Grace-Blackwell GPU (Test GPU sm_121)"),
        (9, 0, "This is not a Blackwell/Grace-Blackwell GPU"),
    ],
)
def test_w2_036_blackwell_features_cover_every_probe_blackwell_label(
    hardware_info,
    probe_record,
    major: int,
    minor: int,
    expected_first_line: str,
) -> None:
    cap = hardware_capabilities._entry_to_capabilities(probe_record(major, minor))

    lines = hardware_info._blackwell_feature_lines(cap)

    assert lines[0] == expected_first_line
    if major >= 10:
        assert not any("not a Blackwell" in line for line in lines)
        assert " TMA (Tensor Memory Accelerator)" in lines


_TMA_COMPILER_WARNING = " ⚠  TMA hardware present but compiler support is disabled"


@pytest.mark.parametrize(
    ("major", "minor", "host_link"),
    [
        # GB10 reaches its Grace CPU over NVLink-C2C.
        (12, 1, (True, True)),
        (10, 0, (False, False)),
        (12, 0, (False, False)),
    ],
)
def test_w2_036_tma_compiler_warning_uses_the_record_the_probe_emits(
    hardware_info,
    probe_record,
    major: int,
    minor: int,
    host_link: tuple[bool, bool],
) -> None:
    record = probe_record(major, minor, ptxas_accepts_tma=False, host_link=host_link)
    # The probe clears tma.supported whenever its ptxas check fails, so the
    # record cannot say that the TMA hardware exists.
    assert record["tma"]["supported"] is False
    assert record["tma"]["compiler_support"] is False

    lines = hardware_info._blackwell_feature_lines(hardware_capabilities._entry_to_capabilities(record))

    warnings = [line for line in lines if line.startswith(_TMA_COMPILER_WARNING)]
    assert len(warnings) == 1
    assert "    TMA probe: ptxas rejected tensormap.replace" in lines
    assert " TMA (Tensor Memory Accelerator)" not in lines

    supported = probe_record(major, minor, ptxas_accepts_tma=True, host_link=host_link)
    assert supported["tma"]["supported"] is True
    assert supported["tma"]["compiler_support"] is True
    supported_lines = hardware_info._blackwell_feature_lines(
        hardware_capabilities._entry_to_capabilities(supported)
    )
    assert " TMA (Tensor Memory Accelerator)" in supported_lines
    assert not any("compiler support is disabled" in line for line in supported_lines)


def test_w2_036_tma_lines_stay_empty_without_tma_hardware(hardware_info, probe_record) -> None:
    # SM 8.9 has no TMA hardware, so a missing compiler check is not a warning.
    record = probe_record(8, 9, ptxas_accepts_tma=False)
    cap = hardware_capabilities._entry_to_capabilities(record)

    assert record["tma"]["supported"] is False
    assert hardware_info._tma_feature_lines(cap) == []


@pytest.mark.parametrize(
    ("major", "minor", "expected_level"),
    [
        (10, 0, "5. HBM3e Memory (global)"),
        (10, 3, "5. HBM3e Memory (global)"),
        (12, 0, "5. Device Memory (global)"),
        (12, 1, "5. Device Memory (global)"),
        (9, 0, "5. Device Memory (global)"),
    ],
)
def test_w2_036_memory_hierarchy_names_hbm3e_only_when_the_probe_reports_it(
    hardware_info,
    probe_record,
    major: int,
    minor: int,
    expected_level: str,
) -> None:
    features = probe_record(major, minor)["features"]
    # get_gpu_info() derives hbm3e_memory from the probe's feature list the same way.
    lines = hardware_info._memory_hierarchy_lines({"hbm3e_memory": "HBM3e" in features})

    assert lines[4] == expected_level
    assert len(lines) == 6


def test_w2_037_copy_timer_requires_warmup_and_cuda_tensors(hardware_info) -> None:
    src = torch.zeros(16)
    dst = torch.empty_like(src)

    with pytest.raises(ValueError, match="warmup copy counts must be positive"):
        hardware_info._measure_copy_bandwidth_gbps(src, dst, iterations=10, warmup_iterations=0)
    with pytest.raises(ValueError, match="same CUDA device"):
        hardware_info._measure_copy_bandwidth_gbps(src, dst, iterations=10, warmup_iterations=2)


def test_w2_037_copy_timer_measures_a_real_cuda_copy(hardware_info) -> None:
    if not torch.cuda.is_available():
        pytest.skip("SKIPPED: CUDA device required for the real copy timing path")
    src = torch.arange(1 << 22, device="cuda", dtype=torch.float16)
    dst = torch.zeros_like(src)

    bandwidth_gbps = hardware_info._measure_copy_bandwidth_gbps(
        src,
        dst,
        iterations=5,
        warmup_iterations=2,
    )

    assert bandwidth_gbps > 0
    torch.testing.assert_close(dst, src, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# W2-093: CUDA source scanning ignores comments and literals
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_w2_093_dtype_inference_ignores_comments_and_literals(tmp_path: Path) -> None:
    infer = mcp_server._infer_dtype_from_cu
    cases = {
        "line_comment.cu": (
            "// Uses half the shared memory of the double-buffered version.\n"
            "__global__ void k(float* x) { x[0] = 1.0f; }\n",
            "float32",
        ),
        "block_comment.cu": (
            "/* half precision would also work.\n   double check the bounds. */\n"
            "__global__ void k(float* x) {}\n",
            "float32",
        ),
        "string_literal.cu": (
            '__global__ void k(float* x) {}\n'
            'int main() { printf("double buffering, half tile\\n"); }\n',
            "float32",
        ),
        "raw_string.cu": (
            'const char* src = R"kernel(\n__global__ void k(double* x) {}\n)kernel";\n'
            "__global__ void k(float* x) {}\n",
            "float32",
        ),
        "code_after_comment.cu": (
            "// float32 reference comment\n__global__ void k(half* x) {}\n",
            "float16",
        ),
        "digit_separator.cu": (
            "constexpr int N = 1'000'000;\n__global__ void k(__nv_bfloat16* x) {}\n",
            "bfloat16",
        ),
        "char_literal.cu": (
            "char quote = '\"';\n__global__ void k(double* x) {}\n",
            "float64",
        ),
    }
    for filename, (text, expected) in cases.items():
        assert infer(_write(tmp_path, filename, text)) == expected, filename

    with pytest.raises(ValueError, match="Unterminated block comment"):
        infer(_write(tmp_path, "broken.cu", "/* never closed\n__global__ void k(half* x) {}\n"))


def test_w2_093_workload_constants_survive_comment_markers_inside_strings(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "baseline_markers.cu",
        'const char* banner = "/* not a comment";\n'
        "constexpr int TILE_SIZE = 64;\n"
        "/* a real comment\n   const int FAKE_SIZE = 99; */\n"
        "const int NUM_ELEMENTS = 1 << 20; // const int OTHER_SIZE = 7;\n",
    )

    params = mcp_server._extract_workload_params_from_cu(path)

    assert params == {"TILE_SIZE": 64, "NUM_ELEMENTS": 1 << 20}


# ---------------------------------------------------------------------------
# W2-033 residual: analyze_bottlenecks reaches real analysis code
# ---------------------------------------------------------------------------


_TRACE_MTIME_NS = 1_790_000_000 * 1_000_000_000


def _write_json(path: Path, payload: Any, *, seconds_after_trace: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    mtime_ns = _TRACE_MTIME_NS + seconds_after_trace * 1_000_000_000
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def _write_pair_manifest(bench_root: Path, *, seconds_after_trace: int) -> Path:
    """Write a manifest in another run, newer than any trace, as mcp-fixtures does."""
    return _write_json(
        bench_root / "artifacts" / "runs" / "mcp-fixtures" / "profiles" / "bench" / "ch04"
        / "pair_manifest.json",
        {"baseline": "baseline_trace.json", "optimized": "optimized_trace.json"},
        seconds_after_trace=seconds_after_trace,
    )


def _write_torch_capture(bench_root: Path) -> Path:
    """Lay out a capture the way profile_torch does, with newer non-trace JSON beside the trace.

    pytorch_profiler_runner writes trace.json before key_averages_full.json (a
    list of operator rows) and torch_profile_summary.json (an object without
    traceEvents). This layout also makes metadata.json newer than the trace, so
    every non-trace file outranks the trace by modification time.
    """
    capture = (
        bench_root / "artifacts" / "runs" / "r00_trace_run" / "profiles" / "tools" / "torch"
        / "target" / "target_20261013_000000"
    )
    events = [
        {"ph": "X", "name": "train_step", "cat": "python_function", "dur": 1000},
        {"ph": "X", "name": "ampere_sgemm_128x64_nn", "cat": "kernel", "dur": 600},
        {"ph": "X", "name": "Memcpy HtoD (Pageable -> Device)", "cat": "gpu_memcpy", "dur": 300},
        {"ph": "X", "name": "elementwise_kernel", "cat": "kernel", "dur": 100},
        {"ph": "i", "name": "marker", "cat": "kernel"},
    ]
    trace_path = _write_json(capture / "trace.json", {"traceEvents": events}, seconds_after_trace=0)
    _write_json(capture / "metadata.json", {"mode": "full", "error": None}, seconds_after_trace=1)
    _write_json(
        capture / "key_averages_full.json",
        [{"name": "aten::mm", "count": 2, "cpu_time_total_us": 50.0, "cuda_time_total_us": 600.0}],
        seconds_after_trace=2,
    )
    _write_json(
        capture / "torch_profile_summary.json",
        {"mode": "full", "total_cuda_time_us": 1000.0, "top_ops": []},
        seconds_after_trace=3,
    )
    _write_pair_manifest(bench_root, seconds_after_trace=10)
    return trace_path


@pytest.fixture
def trace_core(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    trace_path = _write_torch_capture(tmp_path)
    core = PerfCore(bench_root=tmp_path)
    monkeypatch.setattr(engine_module, "_get_handler", lambda: core)
    return trace_path


def test_w2_033_profile_mode_skips_newer_manifest_and_summary_json(trace_core: Path) -> None:
    result = engine_module.get_engine().analyze.bottlenecks(mode="profile")

    assert result["source"] == "profile"
    assert result["llm"] is None
    profile = result["profile"]
    assert "error" not in profile
    assert profile["trace_path"] == str(trace_core)
    assert profile["trace_path"].endswith("trace.json")
    assert profile["trace_event_count"] == 5
    assert profile["total_time_us"] == 2000
    assert profile["duration_basis"] == (
        "Sum of complete trace-event durations. Overlapping and nested events can be counted "
        "more than once"
    )
    assert profile["bottlenecks"]
    types = {(entry["type"], entry.get("kernel_name")) for entry in profile["bottlenecks"]}
    assert ("memory_transfer", None) in types
    memory_transfer = next(
        entry for entry in profile["bottlenecks"] if entry["type"] == "memory_transfer"
    )
    assert memory_transfer["representative_events"] == ["Memcpy HtoD (Pageable -> Device)"]
    assert ("hot_kernel", "ampere_sgemm_128x64_nn") in types
    assert ("hot_kernel", "train_step") not in types
    assert ("hot_kernel", "Memcpy HtoD (Pageable -> Device)") not in types
    assert all("execution time" not in entry["description"] for entry in profile["bottlenecks"])
    assert all("GPU time" not in entry["description"] for entry in profile["bottlenecks"])
    # The trace does not say which GPU recorded it.
    assert profile["hardware"]["name"] == "Unknown"


def _write_text(path: Path, text: str, *, seconds_after_trace: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    mtime_ns = _TRACE_MTIME_NS + seconds_after_trace * 1_000_000_000
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


# A JSON Lines file standing in for `nsys export --type=json` output. This host
# has no nsys, so the real export layout is not checked here. The search skips
# such a file whether it is JSON Lines or one JSON document without traceEvents.
_NSYS_JSON_LINES = (
    '{"id":0,"table":"StringIds","value":"[Unknown]"}\n'
    '{"globalPid":284057963331584,"name":"python","pid":153958,"table":"PROCESSES"}\n'
    '{"name":"COLLECT_GPU_CTX_SW_TRACE","table":"META_DATA_CAPTURE","value":"false"}\n'
)


def test_w2_033_profile_mode_skips_a_newer_nsys_json_lines_export(
    trace_core: Path,
    tmp_path: Path,
) -> None:
    # HTA capture saves `nsys export --type=json` output as <name>.json in a profiles directory.
    export = _write_text(
        tmp_path / "artifacts" / "runs" / "r00_hta_run" / "profiles" / "tools" / "hta"
        / "hta_capture_20261013_000100.json",
        _NSYS_JSON_LINES,
        seconds_after_trace=20,
    )

    profile = engine_module.get_engine().analyze.bottlenecks(mode="profile")["profile"]

    assert "error" not in profile
    assert profile["trace_path"] == str(trace_core)
    assert profile["skipped_json_lines"] == [str(export)]

    # A file whose later line is broken is not JSON Lines, so it still stops the search.
    _write_text(export, _NSYS_JSON_LINES + '{"table": "TRUNCATED"\n', seconds_after_trace=20)
    broken = engine_module.get_engine().analyze.bottlenecks(mode="profile")["profile"]
    assert broken["error_type"] == "trace_read_error"
    assert broken["unreadable_path"] == str(export)


def test_w2_033_profile_mode_reports_a_missing_chrome_trace_without_naming_another_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_module, "_get_handler", lambda: PerfCore(bench_root=tmp_path))
    analyze = engine_module.get_engine().analyze

    empty_root = analyze.bottlenecks(mode="profile")["profile"]
    assert empty_root["error_type"] == "no_chrome_trace"
    assert "0 JSON file(s)" in empty_root["error"]

    capture = tmp_path / "artifacts" / "runs" / "r00_trace_run" / "profiles" / "capture"
    _write_json(capture / "torch_profile_summary.json", {"top_ops": []}, seconds_after_trace=1)
    _write_json(capture / "key_averages_full.json", [{"name": "aten::mm"}], seconds_after_trace=2)
    _write_pair_manifest(tmp_path, seconds_after_trace=3)

    no_trace = analyze.bottlenecks(mode="profile")["profile"]
    assert no_trace["error_type"] == "no_chrome_trace"
    assert "3 JSON file(s)" in no_trace["error"]
    assert "trace_path" not in no_trace

    trace_file = _write_json(capture / "trace.json", {"traceEvents": []}, seconds_after_trace=0)
    empty = analyze.bottlenecks(mode="profile")["profile"]
    assert empty["error_type"] == "empty_chrome_trace"
    assert empty["trace_path"] == str(trace_file)

    _write_json(trace_file, {"traceEvents": {"ph": "X"}}, seconds_after_trace=0)
    malformed = analyze.bottlenecks(mode="profile")["profile"]
    assert malformed["error_type"] == "trace_read_error"
    assert malformed["unreadable_path"] == str(trace_file)
    assert "not a list" in malformed["error"]


def test_w2_033_profile_mode_reads_a_real_torch_profiler_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = tmp_path / "artifacts" / "runs" / "r00_real_capture" / "profiles" / "capture"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    command = [
        sys.executable,
        "-m",
        "core.scripts.profiling.pytorch_profiler_runner",
        str(CODE_ROOT / "tests" / "fixtures" / "mcp_torch_profile_target.py"),
        "--output-dir",
        str(capture),
        "--profile-mode",
        "full",
    ]
    completed = subprocess.run(
        command, cwd=CODE_ROOT, env=environment, capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    trace_path = capture / "trace.json"
    newest_capture_mtime = max(path.stat().st_mtime_ns for path in capture.iterdir())
    manifest = _write_pair_manifest(tmp_path, seconds_after_trace=0)
    os.utime(manifest, ns=(newest_capture_mtime + 10**9, newest_capture_mtime + 10**9))
    monkeypatch.setattr(engine_module, "_get_handler", lambda: PerfCore(bench_root=tmp_path))

    profile = engine_module.get_engine().analyze.bottlenecks(mode="profile")["profile"]

    assert "error" not in profile
    assert profile["trace_path"] == str(trace_path)
    assert profile["trace_event_count"] == len(
        json.loads(trace_path.read_text(encoding="utf-8"))["traceEvents"]
    )
    assert profile["total_time_us"] > 0
    assert {entry["type"] for entry in profile["bottlenecks"]} <= {
        "memory_transfer",
        "cpu_overhead",
        "hot_kernel",
        "profiler_overhead",
    }


def test_w2_033_llm_mode_explains_trace_findings(
    trace_core: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_llm_call(prompt: str, system: str | None = None, **kwargs: Any) -> str:
        calls.append({"prompt": prompt, "system": system, **kwargs})
        return "Host-to-device copies use pageable memory."

    monkeypatch.setattr(llm_module, "is_available", lambda: True)
    monkeypatch.setattr(llm_module, "llm_call", _fake_llm_call)

    result = engine_module.get_engine().analyze.bottlenecks(mode="llm", focus="memory")

    assert result["profile"] is None
    assert result["focus"] == "memory"
    llm = result["llm"]
    assert llm["success"] is True
    assert llm["llm_response"] == "Host-to-device copies use pageable memory."
    assert llm["trace_path"] == str(trace_core)
    assert len(calls) == 1
    assert calls[0]["json_mode"] is False
    assert calls[0]["system"] == llm_module.PERF_EXPERT_SYSTEM
    assert "Focus area: memory" in calls[0]["prompt"]
    assert "Memcpy HtoD" in calls[0]["prompt"]


def test_w2_033_llm_mode_reports_missing_backend_and_missing_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_module, "_get_handler", lambda: PerfCore(bench_root=tmp_path))
    _write_pair_manifest(tmp_path, seconds_after_trace=0)
    monkeypatch.setattr(llm_module, "is_available", lambda: False)
    unavailable = engine_module.get_engine().analyze.bottlenecks(mode="both")
    assert unavailable["llm"]["error_type"] == "llm_unavailable"
    assert unavailable["profile"]["error_type"] == "no_chrome_trace"
    assert "trace_path" not in unavailable["profile"]

    monkeypatch.setattr(llm_module, "is_available", lambda: True)
    monkeypatch.setattr(
        llm_module,
        "llm_call",
        lambda *args, **kwargs: pytest.fail("LLM must not be called without trace findings"),
    )
    no_trace = engine_module.get_engine().analyze.bottlenecks(mode="llm")
    assert no_trace["profile"] is None
    assert no_trace["llm"]["error_type"] == "no_profile_bottlenecks"
    assert no_trace["llm"]["profile_error_type"] == "no_chrome_trace"
    assert "No Chrome trace found" in no_trace["llm"]["error"]


def test_w2_033_bottlenecks_rejects_unknown_mode_and_profile_focus(trace_core: Path) -> None:
    analyze = engine_module.get_engine().analyze
    with pytest.raises(ValueError, match="mode must be one of"):
        analyze.bottlenecks(mode="bottleneck")
    with pytest.raises(ValueError, match="focus must be one of"):
        analyze.bottlenecks(mode="both", focus="network")
    with pytest.raises(ValueError, match="focus applies only to LLM analysis"):
        analyze.bottlenecks(mode="profile", focus="memory")


def _payload(result: mcp_server.ToolResult) -> dict[str, Any]:
    entry = result.content[0]
    assert entry.get("type") == "text"
    return json.loads(entry["text"])


def test_w2_033_mcp_default_call_runs_both_analyses(
    trace_core: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_module, "is_available", lambda: False)
    server = mcp_server.MCPServer()

    payload = _payload(server.call_tool("analyze_bottlenecks", {"include_context": False}))

    assert payload["status"] == "ok"
    result = payload["result"]
    assert result["source"] == "both"
    assert result["focus"] == "bottleneck"
    assert result["profile"]["trace_path"] == str(trace_core)
    assert result["availability"] == {"profile": True, "llm": False}
    assert result["llm"]["error_type"] == "llm_unavailable"

    profile_only = _payload(
        server.call_tool("analyze_bottlenecks", {"mode": "profile", "include_context": False})
    )
    assert profile_only["result"]["source"] == "profile"
    assert profile_only["result"]["llm"] is None


# ---------------------------------------------------------------------------
# W2-086: occupancy lab note matches what the sweep script does
# ---------------------------------------------------------------------------


def test_w2_086_readme_does_not_claim_the_sweep_regenerates_targets() -> None:
    readme = (CODE_ROOT / "labs" / "occupancy_tuning" / "README.md").read_text(encoding="utf-8")
    sweep = (CODE_ROOT / "labs" / "occupancy_tuning" / "sweep_schedules.py").read_text(
        encoding="utf-8"
    )

    assert "regenerate the harness targets by rerunning the sweep script" not in readme
    assert "The sweep script only measures schedules and writes a CSV." in readme
    assert "optimized_proton_matmul_<schedule>.py" in readme
    # The sweep writes one CSV and creates no benchmark modules.
    assert "_write_csv(results, args.csv)" in sweep
    assert "optimized_proton_matmul" not in sweep


def test_w2_086_readme_generator_carries_the_corrected_note() -> None:
    # The README is generated, so the next regeneration must not restore the old note.
    from core.scripts.refresh_readmes import ENTRIES, _format_markdown

    readme = (CODE_ROOT / "labs" / "occupancy_tuning" / "README.md").read_text(encoding="utf-8")
    generated = _format_markdown(ENTRIES["labs/occupancy_tuning"]).rstrip() + "\n"

    assert "regenerate the harness targets by rerunning the sweep script" not in generated
    assert generated == readme
