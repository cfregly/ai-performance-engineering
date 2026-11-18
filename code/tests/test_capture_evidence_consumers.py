from __future__ import annotations

import base64
import os
import signal
import socket
import sys
from contextlib import suppress
from pathlib import Path

import pytest
import torch

from core.benchmark.cuda_binary_benchmark import _run_subprocess_capture
from core.diagnostics.process import run_owned_process
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    SubprocessCaptureEvidenceError,
    TorchrunLaunchSpec,
)
from core.tools.tools_commands import (
    TOOLS,
    ToolCaptureEvidenceError,
    ToolSpec,
    _require_exact_tool_capture,
    _run_tool,
)

COOPERATIVE_CAPTURE_SUPPORTED = all(
    hasattr(os, name) for name in ("waitid", "P_PID", "WEXITED", "WNOWAIT", "WNOHANG")
)


def _raw_stream(receipt: dict, name: str) -> bytes:
    return base64.b64decode(receipt["raw_streams_base64"][name])


def _cpu_config(**overrides) -> BenchmarkConfig:
    return BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=1,
        warmup=5,
        use_subprocess=False,
        enable_profiling=False,
        enable_memory_tracking=False,
        enforce_environment_validation=False,
        measurement_timeout_seconds=30,
        **overrides,
    )


class _InvalidByteCpuBenchmark(BaseBenchmark):
    """Real CPU benchmark whose worker emits one byte that UTF-8 cannot decode."""

    allow_cpu = True

    def setup(self) -> None:
        os.write(1, b"isolated-worker-prefix\xff-tail\n")
        self.input = torch.arange(8, dtype=torch.float32)
        self.output = None

    def benchmark_fn(self) -> None:
        self.output = self.input.square()

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        return {"input": self.input}

    def get_verify_output(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("CPU benchmark did not execute")
        return self.output

    def get_input_signature(self) -> dict[str, object]:
        return {"shape": tuple(self.input.shape), "dtype": str(self.input.dtype)}

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def validate_result(self) -> None:
        torch.testing.assert_close(self.output, self.input.square())


class _InvalidByteTorchrunTarget:
    name = "invalid_byte_torchrun_target"

    def __init__(self, script_path: Path) -> None:
        self.script_path = script_path

    def get_torchrun_spec(self, _config: BenchmarkConfig) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(
            script_path=self.script_path,
            name=self.name,
        )


def _free_rendezvous_endpoint() -> str:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{listener.getsockname()[1]}"


def _register_test_tool(name: str, script_path: Path) -> None:
    if name in TOOLS:
        raise AssertionError(f"Test tool name already registered: {name}")
    TOOLS[name] = ToolSpec(
        name=name,
        script_path=script_path,
        description="Exercise exact subprocess capture evidence.",
        cooperative_process=True,
    )


def test_capture_evidence_cuda_binary_rejects_invalid_bytes(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import os; os.write(1, b'cuda-prefix\\xff-tail')",
    ]

    with pytest.raises(SubprocessCaptureEvidenceError) as caught:
        _run_subprocess_capture(command, cwd=tmp_path, timeout=10)

    receipt = caught.value.cooperative_process_result
    assert caught.value.result is receipt
    assert _raw_stream(receipt, "stdout") == b"cuda-prefix\xff-tail"
    assert "undecodable_streams_base64" in receipt["execution"]


def test_capture_evidence_isolated_worker_fails_before_json_parsing() -> None:
    config = _cpu_config()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    harness._ensure_runtime_initialized()

    with pytest.raises(SubprocessCaptureEvidenceError) as caught:
        harness._benchmark_with_subprocess(_InvalidByteCpuBenchmark(), config)

    receipt = caught.value.cooperative_process_result
    assert b"isolated-worker-prefix\xff-tail" in _raw_stream(receipt, "stdout")
    assert "Failed to parse subprocess output" not in str(caught.value)


def test_capture_evidence_torchrun_fails_before_protocol_parsing(tmp_path: Path) -> None:
    script_path = tmp_path / "invalid_byte_torchrun_worker.py"
    script_path.write_text(
        "import os\nos.write(1, b'torchrun-prefix\\xff-tail\\n')\n",
        encoding="utf-8",
    )
    config = _cpu_config(
        nproc_per_node=1,
        rdzv_endpoint=_free_rendezvous_endpoint(),
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)

    with pytest.raises(SubprocessCaptureEvidenceError) as caught:
        harness._benchmark_with_torchrun(
            _InvalidByteTorchrunTarget(script_path),
            config,
        )

    receipt = caught.value.cooperative_process_result
    assert b"torchrun-prefix\xff-tail" in _raw_stream(receipt, "stdout")


def test_capture_evidence_tool_gate_retains_real_invalid_byte_receipt() -> None:
    receipt = run_owned_process(
        [
            sys.executable,
            "-c",
            "import os; os.write(1, b'tool-gate-prefix\\xff-tail')",
        ],
        timeout_seconds=10,
    )

    with pytest.raises(ToolCaptureEvidenceError) as caught:
        _require_exact_tool_capture(receipt)

    assert caught.value.cooperative_process_result is receipt
    assert _raw_stream(receipt, "stdout") == b"tool-gate-prefix\xff-tail"


def test_capture_evidence_tool_gate_retains_cutoff_metadata() -> None:
    receipt = {
        "stdout": "partial output",
        "stderr": "",
        "execution": {
            "capture_limit": {
                "reason": "stream_closed_before_eof",
                "streams_closed_before_eof": ["stdout"],
            }
        },
    }

    with pytest.raises(ToolCaptureEvidenceError) as caught:
        _require_exact_tool_capture(receipt)

    assert caught.value.cooperative_process_result is receipt
    assert (
        caught.value.result["execution"]["capture_limit"] == receipt["execution"]["capture_limit"]
    )


@pytest.mark.skipif(
    not COOPERATIVE_CAPTURE_SUPPORTED,
    reason="cooperative process ownership requires waitid with WNOWAIT",
)
def test_capture_evidence_tool_rejects_invalid_bytes_before_replay(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    name = "capture-evidence-invalid-byte-test"
    script_path = tmp_path / "invalid_byte_tool.py"
    script_path.write_text(
        "import os\nos.write(1, b'tool-prefix\\xff-tail')\n",
        encoding="utf-8",
    )
    _register_test_tool(name, script_path)
    try:
        with pytest.raises(ToolCaptureEvidenceError) as caught:
            _run_tool(name, [])
    finally:
        TOOLS.pop(name, None)

    assert capsys.readouterr() == ("", "")
    receipt = caught.value.cooperative_process_result
    assert _raw_stream(receipt, "stdout") == b"tool-prefix\xff-tail"
    assert receipt["execution"]["capture_errors"]


@pytest.mark.skipif(
    not COOPERATIVE_CAPTURE_SUPPORTED,
    reason="cooperative process ownership requires waitid with WNOWAIT",
)
def test_capture_evidence_tool_retains_stream_cutoff_receipt(tmp_path: Path) -> None:
    name = "capture-evidence-cutoff-test"
    pid_path = tmp_path / "detached-child.pid"
    script_path = tmp_path / "cutoff_tool.py"
    script_path.write_text(
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(10)'],\n"
        "    start_new_session=True,\n"
        ")\n"
        "Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')\n"
        "print('parent complete', flush=True)\n",
        encoding="utf-8",
    )
    _register_test_tool(name, script_path)
    child_pid = None
    try:
        with pytest.raises(ToolCaptureEvidenceError) as caught:
            _run_tool(name, [str(pid_path)])
        child_pid = int(pid_path.read_text(encoding="utf-8"))
        receipt = caught.value.cooperative_process_result
        limit = receipt["execution"]["capture_limit"]
        assert limit["streams_closed_before_eof"] == ["stderr", "stdout"]
        assert limit["bytes_retained"]["stdout"] >= len("parent complete\n")
    finally:
        TOOLS.pop(name, None)
        if child_pid is None and pid_path.exists():
            child_pid = int(pid_path.read_text(encoding="utf-8"))
        if child_pid is not None:
            with suppress(ProcessLookupError):
                os.killpg(child_pid, signal.SIGKILL)
