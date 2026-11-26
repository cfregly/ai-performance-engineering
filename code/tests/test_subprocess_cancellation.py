"""Cancel real CPU child processes through the profiler and harness owners."""

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]

WAITING_CHILD = """
import json
import os
import signal
import socket
import sys

connection = socket.socket(socket.AF_UNIX)
connection.connect(os.environ["AISP_CANCEL_TEST_SOCKET"])

def shutdown(signum, frame):
    os.write(1, b"owned child stopped\\n")
    os.write(2, b"owned child stderr stopped\\n")
    connection.close()
    raise SystemExit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)
os.write(1, b"owned child ready\\n")
os.write(2, b"owned child stderr ready\\n")
connection.sendall(json.dumps({"pid": os.getpid(), "pgid": os.getpgrp()}).encode())
signal.pause()
"""


@pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(os, "waitid"),
    reason="requires Linux process-group and waitid semantics",
)
@pytest.mark.parametrize(
    "owner", ["captured_profiler", "streamed_profiler", "nsys", "ncu", "isolated"]
)
def test_keyboard_interrupt_reaps_owned_child_and_retains_logs(tmp_path: Path, owner: str) -> None:
    child = tmp_path / "waiting_child.py"
    child.write_text(textwrap.dedent(WAITING_CHILD), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(CODE_ROOT)

    if owner.endswith("profiler"):
        capture_output = owner == "captured_profiler"
        invocation = f"""
from core.harness.run_benchmarks import _run_profile_subprocess
_run_profile_subprocess(
    command=[sys.executable, {str(child)!r}],
    cwd=Path({str(tmp_path)!r}), env=dict(os.environ), timeout_seconds=None,
    log_base=artifacts / "capture", terminate_reason="owned-cancellation-test",
    capture_output={capture_output!r},
    timeout_collect_error_message="unused timeout path", wait_error_message="wait failed",
)
"""
        stdout_log = artifacts / "capture.stdout.log"
        stderr_log = artifacts / "capture.stderr.log"
        receipt = artifacts / "capture.command.json"
    elif owner in {"nsys", "ncu"}:
        executable = tmp_path / owner
        executable.write_text(
            f"#!{sys.executable}\n"
            "import runpy, sys\n"
            "if '--version' in sys.argv:\n"
            "    print('CPU cancellation fixture')\n"
            "    raise SystemExit(0)\n"
            f"runpy.run_path({str(child)!r}, run_name='__main__')\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        environment["PATH"] = str(tmp_path) + os.pathsep + environment.get("PATH", "")
        if owner == "nsys":
            invocation = """
from core.profiling.nsight_automation import NsightAutomation
automation = NsightAutomation(artifacts)
automation.profile_nsys(
    command=[sys.executable, "-c", "raise AssertionError('capture must be cancelled')"],
    output_name="capture", trace_cuda=False, trace_nvtx=False, trace_osrt=False,
    timeout_seconds=None, sanitize_python_startup=False,
)
"""
        else:
            invocation = """
from core.profiling.nsight_automation import NsightAutomation
automation = NsightAutomation(artifacts)
automation.profile_ncu(
    command=[sys.executable, "-c", "raise AssertionError('capture must be cancelled')"],
    output_name="capture", metric_set="minimal", timeout_seconds=None,
    sanitize_python_startup=False,
)
"""
        stdout_log = artifacts / "capture.stdout.log"
        stderr_log = artifacts / "capture.stderr.log"
        receipt = artifacts / "capture.command.json"
    else:
        fixture = tmp_path / "cancellation_benchmark.py"
        fixture.write_text(
            "import os\n"
            "if str(os.getpid()) != os.environ['AISP_CANCEL_TEST_OWNER_PID']:\n"
            "    import runpy\n"
            f"    runpy.run_path({str(child)!r}, run_name='__main__')\n"
            "from core.harness.benchmark_harness import BaseBenchmark\n"
            "class CancellationBenchmark(BaseBenchmark):\n"
            "    allow_cpu = True\n"
            "    def benchmark_fn(self):\n"
            "        raise AssertionError('benchmark must be cancelled before execution')\n",
            encoding="utf-8",
        )
        invocation = f"""
import importlib.util
import torch
from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness
os.environ["AISP_CANCEL_TEST_OWNER_PID"] = str(os.getpid())
spec = importlib.util.spec_from_file_location("cancellation_benchmark", {str(fixture)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
config = BenchmarkConfig(
    device=torch.device("cpu"),
    execution_timeout_seconds=0,
    subprocess_stderr_dir=str(artifacts),
)
harness = BenchmarkHarness(config=config)
harness._benchmark_with_subprocess(module.CancellationBenchmark(), config)
"""
        stdout_log = artifacts / "CancellationBenchmark_subprocess.stdout.log"
        stderr_log = artifacts / "CancellationBenchmark_subprocess.stderr.log"
        receipt = None

    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os, sys\nfrom pathlib import Path\n"
        f"artifacts = Path({str(artifacts)!r})\n"
        "try:\n"
        + textwrap.indent(textwrap.dedent(invocation).strip(), "    ")
        + "\nexcept KeyboardInterrupt:\n"
        "    print('owner propagated cancellation', flush=True)\n"
        "    raise SystemExit(130)\n"
        "raise AssertionError('owner did not propagate cancellation')\n",
        encoding="utf-8",
    )

    with tempfile.TemporaryDirectory(prefix="aisp-cancel-", dir="/tmp") as socket_directory:
        socket_path = str(Path(socket_directory) / "ready.sock")
        environment["AISP_CANCEL_TEST_SOCKET"] = socket_path
        with socket.socket(socket.AF_UNIX) as server:
            server.bind(socket_path)
            server.listen(1)
            process = subprocess.Popen(
                [sys.executable, str(worker)], cwd=CODE_ROOT, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
            )
            connection = None
            child_pid = None
            try:
                connection, _ = server.accept()
                ready = json.loads(connection.recv(4096))
                child_pid = ready["pid"]
                assert ready["pgid"] == child_pid
                os.kill(process.pid, signal.SIGINT)
                stdout, stderr = process.communicate()

                assert process.returncode == 130, stdout + stderr
                assert "owner propagated cancellation" in stdout
                with pytest.raises(ProcessLookupError):
                    os.kill(child_pid, 0)
                assert "owned child ready" in stdout_log.read_text()
                assert "owned child stopped" in stdout_log.read_text()
                assert "owned child stderr ready" in stderr_log.read_text()
                assert "owned child stderr stopped" in stderr_log.read_text()
                if receipt is not None:
                    record = json.loads(receipt.read_text())
                    assert record.get("command") or record.get("cmd")
                    if owner == "captured_profiler":
                        assert record["cancelled"] is True
                        assert record["timeout_seconds"] is None
                        assert record["cancellation"]["signal_forwarded"] is True
                        assert record["cancellation"]["status"] == "cleanup_completed"
                    else:
                        assert record["status"] == "cancelled"
                    if owner in {"nsys", "ncu"}:
                        cooperative = record["cooperative_process"]
                        assert cooperative["timeout_seconds"] is None
                        assert cooperative["cancellation"]["signal_forwarded"] is True
                        assert cooperative["cancellation"]["status"] == "cleanup_completed"
                        assert record["finalize_signals"] == ["SIGINT"]
                        assert record["graceful_finalize_completed"] is True
                assert not list(artifacts.glob("*.nsys-rep"))
                assert not list(artifacts.glob("*.ncu-rep"))
            finally:
                if child_pid is not None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(child_pid, signal.SIGTERM)
                if connection is not None:
                    connection.recv(4096)
                    connection.close()
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.communicate()
