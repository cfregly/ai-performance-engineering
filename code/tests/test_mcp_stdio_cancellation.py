"""Forward the first SIGINT through real MCP stdio to an owned CPU fixture."""

from __future__ import annotations

from contextlib import suppress
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import sys
import tempfile
import threading

import pytest
import psutil


CODE_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "waitid"),
    reason="owned POSIX subprocess support requires waitid",
)


def _fixture_connection(listener: socket.socket, process: subprocess.Popen[str]) -> socket.socket:
    read_fd, write_fd = os.pipe()

    def observe_exit() -> None:
        process.wait()
        with suppress(BrokenPipeError):
            os.write(write_fd, b"exited")
        os.close(write_fd)

    threading.Thread(target=observe_exit, daemon=True).start()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(listener, selectors.EVENT_READ, "fixture")
            selector.register(read_fd, selectors.EVENT_READ, "exit")
            selector.register(process.stdout, selectors.EVENT_READ, "response")
            events = selector.select()
        if any(key.data == "fixture" for key, _ in events):
            connection, _ = listener.accept()
            return connection
        if any(key.data == "response" for key, _ in events):
            response = process.stdout.readline()
            raise AssertionError(f"MCP returned before fixture readiness: {response}")
        raise AssertionError(f"MCP server exited before fixture readiness: {process.returncode}")
    finally:
        os.close(read_fd)


@pytest.mark.parametrize("surface", ["server_main", "cli"])
def test_first_sigint_reaches_owned_fixture_through_real_stdio(tmp_path: Path, surface: str) -> None:
    marker = tmp_path / "sigint-count.txt"
    fixture = tmp_path / "stdio_cpu_fixture.py"
    fixture.write_text(
        "import json, os, signal, socket, sys\n"
        "from pathlib import Path\n"
        "connection = socket.socket(socket.AF_UNIX)\n"
        "def cleanup(signum, frame):\n"
        "    Path(os.environ['AISP_STDIO_TEST_MARKER']).write_text('1')\n"
        "    print('stdio CPU fixture cleaned', flush=True)\n"
        "    print('stdio CPU stderr cleaned', file=sys.stderr, flush=True)\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGINT, cleanup)\n"
        "print('stdio CPU fixture ready', flush=True)\n"
        "print('stdio CPU stderr ready', file=sys.stderr, flush=True)\n"
        "connection.connect(os.environ['AISP_STDIO_TEST_SOCKET'])\n"
        "connection.sendall((json.dumps({'pid': os.getpid(), 'pgid': os.getpgrp()}) + '\\n').encode())\n"
        "connection.recv(1)\n",
        encoding="utf-8",
    )
    # Replace only the replay script with a labeled CPU fixture. The public MCP
    # entrypoint, JSON-RPC dispatch, CLI launcher and ownership helper stay real.
    # Keep the fixture startup hook first in the child environment so the
    # repository's sitecustomize module cannot hide the script replacement.
    (tmp_path / "sitecustomize.py").write_text(
        "import os\nfrom pathlib import Path\nfrom dataclasses import replace\n"
        "import core.tools.tools_commands as tools\n"
        "tools.TOOLS['trace-replay'] = replace(tools.TOOLS['trace-replay'], "
        "script_path=Path(os.environ['AISP_STDIO_TEST_FIXTURE']), module_name=None)\n"
        "import mcp.mcp_server as server\n"
        "server._subprocess_env = lambda: dict(os.environ)\n",
        encoding="utf-8",
    )
    command = (
        [sys.executable, "-c", "from mcp.mcp_server import main\nmain()\n", "--serve"]
        if surface == "server_main"
        else [sys.executable, "-m", "cli.aisp", "mcp", "--serve"]
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path) + os.pathsep + str(CODE_ROOT)
    environment["AISP_STDIO_TEST_FIXTURE"] = str(fixture)
    environment["AISP_STDIO_TEST_MARKER"] = str(marker)
    stderr_path = tmp_path / "stdio-server-stderr.log"
    with tempfile.TemporaryDirectory(prefix="aisp-stdio-", dir="/tmp") as socket_directory:
        environment["AISP_STDIO_TEST_SOCKET"] = str(Path(socket_directory) / "ready.sock")
        with socket.socket(socket.AF_UNIX) as listener, stderr_path.open("wb") as stderr:
            listener.bind(environment["AISP_STDIO_TEST_SOCKET"])
            listener.listen(1)
            process = subprocess.Popen(
                command, cwd=CODE_ROOT, env=environment,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                text=True, start_new_session=True,
            )
            connection = None
            try:
                process.stdin.write(json.dumps({
                    "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "tools_trace_replay", "arguments": {
                        "args": ["replay"], "include_context": False,
                    }},
                }) + "\n")
                process.stdin.flush()
                connection = _fixture_connection(listener, process)
                with connection.makefile("r") as ready_stream:
                    ready = json.loads(ready_stream.readline())
                assert ready["pid"] == ready["pgid"]
                os.kill(process.pid, signal.SIGINT)
                line = process.stdout.readline()
                assert line, stderr_path.read_text(encoding="utf-8")
                response = json.loads(line)
                assert response["id"] == 1
                assert response["result"]["isError"] is True
                content = response["result"]["content"][0]
                payload = json.loads(content["text"]) if content["type"] == "text" else content["json"]
                result = payload["result"]
                assert marker.read_text() == "1"
                assert result["cancelled"] is True
                assert result["success"] is False
                assert result["cancellation"]["status"] == "cleanup_completed"
                assert result["cancellation"]["repeated_interrupts"] == 0
                assert "stdio CPU fixture ready" in result["stdout"]
                assert "stdio CPU fixture cleaned" in result["stdout"]
                assert "stdio CPU stderr ready" in result["stderr"]
                assert "stdio CPU stderr cleaned" in result["stderr"]
                with pytest.raises(ProcessLookupError):
                    os.kill(ready["pid"], 0)
            finally:
                if connection is not None:
                    with suppress(BrokenPipeError, ConnectionResetError):
                        connection.sendall(b"release")
                    connection.close()
                process.stdin.close()
                process.wait()
                process.stdout.close()
            assert process.returncode == 0, stderr_path.read_text(encoding="utf-8")


def test_script_cli_stdio_keeps_the_original_process_id(tmp_path: Path) -> None:
    stderr_path = tmp_path / "script-stdio-stderr.log"
    command = [sys.executable, str(CODE_ROOT / "cli" / "aisp.py"), "mcp", "--serve"]
    with stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command, cwd=CODE_ROOT, env=dict(os.environ),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            text=True, start_new_session=True,
        )
        try:
            for request_id, method in ((1, "initialize"), (2, "tools/list")):
                process.stdin.write(json.dumps({
                    "jsonrpc": "2.0", "id": request_id, "method": method, "params": {},
                }) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
                assert line, stderr_path.read_text(encoding="utf-8")
                response = json.loads(line)
                assert response["id"] == request_id
                assert "result" in response
            serving_process = psutil.Process(process.pid)
            assert serving_process.cmdline()[1:] == ["-m", "cli.aisp", "mcp", "--serve"]
            assert serving_process.children() == []
        finally:
            process.stdin.close()
            process.wait()
            process.stdout.close()
        assert process.returncode == 0, stderr_path.read_text(encoding="utf-8")
