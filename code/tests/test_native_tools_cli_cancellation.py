"""Cancel CPU fixtures through the documented native tool CLI paths."""

from __future__ import annotations

import contextlib
import json
import os
import selectors
import signal
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]


def _ready_or_exit(server, process, stdout_path, stderr_path):
    read_fd, write_fd = os.pipe()

    def observe_exit() -> None:
        process.wait()
        with contextlib.suppress(BrokenPipeError):
            os.write(write_fd, b"exited")
        os.close(write_fd)

    observer = threading.Thread(target=observe_exit, daemon=True)
    observer.start()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(server, selectors.EVENT_READ, "ready")
            selector.register(read_fd, selectors.EVENT_READ, "exit")
            events = selector.select()
        if any(key.data == "ready" for key, _ in events):
            connection, _ = server.accept()
            with connection, connection.makefile("r") as stream:
                return json.loads(stream.readline())
        process.wait()
        raise AssertionError(
            f"launcher exited before fixture readiness with status {process.returncode}. "
            f"Full output is retained in {stdout_path} and {stderr_path}"
        )
    finally:
        os.close(read_fd)


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "waitid"),
    reason="Owned launchers require POSIX waitid with WNOWAIT",
)
@pytest.mark.parametrize("tool_name", ["multi-gpu-communication", "nvshmem-device"])
def test_native_tool_cli_waits_for_owned_cleanup_and_retains_output(
    tmp_path: Path,
    tool_name: str,
) -> None:
    marker = tmp_path / "cleanup-completed.txt"
    fixture = tmp_path / "cpu_native_tool_fixture.py"
    fixture.write_text(
        "import json, os, signal, socket, sys\n"
        "from pathlib import Path\n"
        "connection = socket.socket(socket.AF_UNIX)\n"
        "connection.connect(os.environ['AISP_NATIVE_TOOL_TEST_SOCKET'])\n"
        "def cleanup(signum, frame):\n"
        "    Path(os.environ['AISP_NATIVE_TOOL_TEST_MARKER']).write_text('cleanup completed')\n"
        "    print('native tool stdout cleanup completed', flush=True)\n"
        "    print('native tool stderr cleanup completed', file=sys.stderr, flush=True)\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGINT, cleanup)\n"
        "print('native tool stdout ready', flush=True)\n"
        "print('native tool stderr ready', file=sys.stderr, flush=True)\n"
        "connection.sendall((json.dumps({'pid': os.getpid(), 'pgid': os.getpgrp()}) + '\\n').encode())\n"
        "signal.pause()\n",
        encoding="utf-8",
    )
    (tmp_path / "sitecustomize.py").write_text(
        "import os\n"
        "from dataclasses import replace\n"
        "from pathlib import Path\n"
        "import core.tools.tools_commands as tools\n"
        "name = os.environ['AISP_NATIVE_TOOL_TEST_NAME']\n"
        "tools.TOOLS[name] = replace(tools.TOOLS[name], "
        "script_path=Path(os.environ['AISP_NATIVE_TOOL_TEST_FIXTURE']), module_name=None)\n",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path) + os.pathsep + str(CODE_ROOT)
    environment["AISP_NATIVE_TOOL_TEST_NAME"] = tool_name
    environment["AISP_NATIVE_TOOL_TEST_FIXTURE"] = str(fixture)
    environment["AISP_NATIVE_TOOL_TEST_MARKER"] = str(marker)
    stdout_path = tmp_path / "launcher-stdout.log"
    stderr_path = tmp_path / "launcher-stderr.log"
    child_pid = None
    with tempfile.TemporaryDirectory(prefix="aisp-native-tool-", dir="/tmp") as socket_directory:
        environment["AISP_NATIVE_TOOL_TEST_SOCKET"] = str(Path(socket_directory) / "ready.sock")
        with (
            socket.socket(socket.AF_UNIX) as server,
            stdout_path.open("wb") as stdout_capture,
            stderr_path.open("wb") as stderr_capture,
        ):
            server.bind(environment["AISP_NATIVE_TOOL_TEST_SOCKET"])
            server.listen(1)
            process = subprocess.Popen(
                [sys.executable, "-m", "cli.aisp", "tools", tool_name, "--"],
                cwd=CODE_ROOT,
                env=environment,
                stdout=stdout_capture,
                stderr=stderr_capture,
                text=True,
                start_new_session=True,
            )
            try:
                ready = _ready_or_exit(server, process, stdout_path, stderr_path)
                child_pid = ready["pid"]
                assert ready["pgid"] == child_pid
                os.kill(process.pid, signal.SIGINT)
                process.wait()
                stdout = stdout_path.read_text(encoding="utf-8")
                stderr = stderr_path.read_text(encoding="utf-8")
                assert process.returncode != 0
                assert marker.read_text(encoding="utf-8") == "cleanup completed"
                assert "native tool stdout ready" in stdout
                assert "native tool stdout cleanup completed" in stdout
                assert "native tool stderr ready" in stderr
                assert "native tool stderr cleanup completed" in stderr
                with pytest.raises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if child_pid is not None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(child_pid, signal.SIGTERM)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.communicate()
