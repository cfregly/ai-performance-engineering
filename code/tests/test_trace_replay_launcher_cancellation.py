"""Cancel real CPU fixtures through the public replay CLI and MCP launchers."""

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


def _ready_or_exit(server, process):
    read_fd, write_fd = os.pipe()

    def observe_exit():
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
        stdout, stderr = process.communicate()
        raise AssertionError(f"launcher exited before fixture readiness: {stdout}\n{stderr}")
    finally:
        os.close(read_fd)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
@pytest.mark.parametrize("surface", ["cli", "mcp"])
def test_public_replay_launchers_wait_for_owned_cleanup_and_retain_output(tmp_path, surface):
    marker = tmp_path / "cleanup-completed.txt"
    fixture = tmp_path / "cpu_replay_fixture.py"
    fixture.write_text(
        "import json, os, signal, socket, sys\n"
        "from pathlib import Path\n"
        "connection = socket.socket(socket.AF_UNIX)\n"
        "connection.connect(os.environ['AISP_REPLAY_TEST_SOCKET'])\n"
        "def cleanup(signum, frame):\n"
        "    Path(os.environ['AISP_REPLAY_TEST_MARKER']).write_text('cleanup completed')\n"
        "    print('CPU replay cleanup completed', flush=True)\n"
        "    print('CPU replay stderr cleanup completed', file=sys.stderr, flush=True)\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGINT, cleanup)\n"
        "print('CPU replay ready', flush=True)\n"
        "print('CPU replay stderr ready', file=sys.stderr, flush=True)\n"
        "connection.sendall((json.dumps({'pid': os.getpid(), 'pgid': os.getpgrp()}) + '\\n').encode())\n"
        "signal.pause()\n",
        encoding="utf-8",
    )
    # Replace only the tool's entrypoint with a waiting CPU fixture. Both public
    # launch chains and their subprocess ownership code execute unchanged.
    (tmp_path / "sitecustomize.py").write_text(
        "import os\n"
        "from dataclasses import replace\n"
        "from pathlib import Path\n"
        "import core.tools.tools_commands as tools\n"
        "tools.TOOLS['trace-replay'] = replace(tools.TOOLS['trace-replay'], "
        "script_path=Path(os.environ['AISP_REPLAY_TEST_FIXTURE']), module_name=None)\n",
        encoding="utf-8",
    )
    result_path = tmp_path / "mcp-result.json"
    if surface == "cli":
        command = [sys.executable, "-m", "cli.aisp", "tools", "trace-replay", "--", "replay"]
    else:
        worker = tmp_path / "mcp_worker.py"
        worker.write_text(
            "import json, os\nfrom pathlib import Path\n"
            "import mcp.mcp_server as server\n"
            "server._subprocess_env = lambda: dict(os.environ)\n"
            "result = server.tool_tools_trace_replay({'args': ['replay'], 'include_context': False})\n"
            f"Path({str(result_path)!r}).write_text(json.dumps(result))\n",
            encoding="utf-8",
        )
        command = [sys.executable, str(worker)]

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path) + os.pathsep + str(CODE_ROOT)
    environment["AISP_REPLAY_TEST_FIXTURE"] = str(fixture)
    environment["AISP_REPLAY_TEST_MARKER"] = str(marker)
    child_pid = None
    with tempfile.TemporaryDirectory(prefix="aisp-launch-", dir="/tmp") as socket_directory:
        environment["AISP_REPLAY_TEST_SOCKET"] = str(Path(socket_directory) / "ready.sock")
        with socket.socket(socket.AF_UNIX) as server:
            server.bind(environment["AISP_REPLAY_TEST_SOCKET"])
            server.listen(1)
            process = subprocess.Popen(
                command, cwd=CODE_ROOT, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
            )
            try:
                ready = _ready_or_exit(server, process)
                child_pid = ready["pid"]
                assert ready["pgid"] == child_pid
                os.kill(process.pid, signal.SIGINT)
                stdout, stderr = process.communicate()
                assert marker.read_text() == "cleanup completed"
                with pytest.raises(ProcessLookupError):
                    os.kill(child_pid, 0)
                if surface == "cli":
                    assert process.returncode != 0
                else:
                    assert process.returncode == 0, stdout + stderr
                    result = json.loads(result_path.read_text())
                    assert result["cancelled"] is True
                    assert result["success"] is False
                    assert result["timeout_seconds"] is None
                    assert result["cancellation"]["status"] == "cleanup_completed"
                    stdout, stderr = result["stdout"], result["stderr"]
                assert "CPU replay ready" in stdout
                assert "CPU replay cleanup completed" in stdout
                assert "CPU replay stderr ready" in stderr
                assert "CPU replay stderr cleanup completed" in stderr
            finally:
                if child_pid is not None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(child_pid, signal.SIGTERM)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.communicate()
