from __future__ import annotations

import datetime as dt
import argparse
import hashlib
import importlib.util
import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
PREPARE_SPEC = importlib.util.spec_from_file_location(
    "prepare_capture", DIRECTORY / "prepare_capture.py"
)
assert PREPARE_SPEC is not None and PREPARE_SPEC.loader is not None
prepare_capture = importlib.util.module_from_spec(PREPARE_SPEC)
sys.modules["prepare_capture"] = prepare_capture
PREPARE_SPEC.loader.exec_module(prepare_capture)

RUN_SPEC = importlib.util.spec_from_file_location(
    "run_capture", DIRECTORY / "run_capture.py"
)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
run_capture = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(run_capture)


GPU = "GPU-12345678-1234-1234-1234-123456789abc"
OTHER_GPU = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def response(value: object, status: int = 200) -> tuple[int, bytes]:
    return status, json.dumps(value, separators=(",", ":")).encode()


class RequestOps:
    def __init__(self, responses: list[tuple[int, bytes]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, object]] = []

    def request(self, method: str, path: str, payload: object) -> tuple[int, bytes]:
        self.calls.append((method, path, payload))
        return self.responses.pop(0)


class RuntimeOps:
    def __init__(self, result: dict[str, str], returncode: int = 0) -> None:
        self.result = result
        self.returncode = returncode
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def run(
        self, argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            self.returncode,
            stdout=json.dumps(self.result),
            stderr="",
        )


class Process:
    def __init__(self, waits: list[object]) -> None:
        self.pid = 4321
        self.returncode = None
        self.waits = list(waits)

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: int) -> int:
        action = self.waits.pop(0)
        if isinstance(action, BaseException):
            raise action
        self.returncode = int(action)
        return self.returncode


class SignalOps:
    def __init__(self) -> None:
        self.signals: list[tuple[int, signal.Signals]] = []

    def killpg(self, pgid: int, sig: signal.Signals) -> None:
        self.signals.append((pgid, sig))


class WorkerOps(SignalOps):
    def __init__(self, process: Process) -> None:
        super().__init__()
        self.process = process
        self.popen_kwargs: dict[str, object] = {}
        self.group_alive = True

    def popen(self, argv: list[str], **kwargs: object) -> Process:
        self.popen_kwargs = kwargs
        return self.process

    def process_group_alive(self, pgid: int) -> bool:
        return self.group_alive

    def killpg(self, pgid: int, sig: signal.Signals) -> None:
        super().killpg(pgid, sig)
        self.group_alive = False

    def sleep(self, seconds: float) -> None:
        raise AssertionError("cleared worker group should not require polling sleep")


class RunCaptureTest(unittest.TestCase):
    def test_empty_config_set_success_still_requires_readback(self) -> None:
        ops = RequestOps([(200, b"")])
        raw, value = run_capture._request_json(
            ops, "POST", run_capture.CONFIG_SET_PATH, {"config": {}}
        )
        self.assertEqual(raw, b"")
        self.assertIsNone(value)
        with self.assertRaisesRegex(run_capture.CaptureError, "invalid JSON"):
            run_capture._request_json(
                RequestOps([(200, b"")]), "GET", run_capture.CONFIG_GET_PATH, {}
            )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _runtime_result(self, python: Path, **overrides: str) -> dict[str, str]:
        result = {
            "torch_version": "2.14.1+cu130",
            "cuda_version": "13.0",
            "sys_executable": str(python),
            "prefix": str(python.parent.parent),
            "base_prefix": "/usr",
        }
        result.update(overrides)
        return result

    def test_cpu_runtime_preflight_keeps_chosen_python_path(self) -> None:
        python = self.root / "venv/bin/python"
        ops = RuntimeOps(self._runtime_result(python))

        receipt = run_capture._runtime_preflight(ops, python, self.root)

        argv, kwargs = ops.calls[0]
        self.assertEqual(argv[:3], [str(python), "-B", "-c"])
        self.assertNotIn("torch.cuda", argv[3])
        self.assertEqual(kwargs["timeout"], 15)
        self.assertEqual(receipt["result"]["sys_executable"], str(python))
        self.assertEqual(len(receipt["record"]["sha256"]), 64)

    def test_cpu_runtime_preflight_rejects_wrong_torch_before_capture(self) -> None:
        python = self.root / "venv/bin/python"
        ops = RuntimeOps(
            self._runtime_result(python, torch_version="2.14.0+cu130")
        )

        with self.assertRaisesRegex(run_capture.CaptureError, "Torch is not"):
            run_capture._runtime_preflight(ops, python, self.root)

        self.assertEqual(len(ops.calls), 1)

    def test_cpu_runtime_preflight_rejects_changed_executable_path(self) -> None:
        python = self.root / "venv/bin/python"
        ops = RuntimeOps(
            self._runtime_result(python, sys_executable="/usr/bin/python3.12")
        )

        with self.assertRaisesRegex(run_capture.CaptureError, "invocation path"):
            run_capture._runtime_preflight(ops, python, self.root)

    def test_restore_removes_only_owned_rule_from_fresh_version(self) -> None:
        original = {"expression": 'host.name == "other"', "settings": []}
        concurrent = {"expression": 'host.name == "new"', "settings": []}
        owned = {
            "expression": 'host.name == "capture"',
            "project": "project-id",
            "expires_at": "2026-10-26T20:00:00Z",
            "settings": [{"setting": {"HighResProfiling": True}}],
        }
        before = {
            "config": {
                "version": 19,
                "config": {"entries": [original, concurrent, owned]},
            }
        }
        after = {
            "config": {
                "version": 20,
                "config": {"entries": [original, concurrent]},
            }
        }
        ops = RequestOps([response(before), response({}), response(after)])

        receipt = run_capture._restore_rule(ops, self.root, owned)

        self.assertEqual(receipt["preserved_entries"], 2)
        post = ops.calls[1][2]
        self.assertEqual(post["config"]["version"], 19)
        self.assertEqual(
            post["config"]["config"]["entries"], [original, concurrent]
        )

    def test_restore_version_drift_does_not_retry_or_overwrite(self) -> None:
        owned = {"expression": "owned"}
        before = {
            "config": {"version": 7, "config": {"entries": [owned]}}
        }
        ops = RequestOps([response(before), response({"conflict": True}, 409)])

        with self.assertRaisesRegex(run_capture.CaptureError, "HTTP 409"):
            run_capture._restore_rule(ops, self.root, owned)

        self.assertEqual([call[0] for call in ops.calls], ["GET", "POST"])

    def test_owned_process_group_cleanup_escalates_safely(self) -> None:
        process = Process(
            [
                subprocess.TimeoutExpired("agent", 3),
                subprocess.TimeoutExpired("agent", 2),
                0,
            ]
        )
        ops = SignalOps()
        log: list[dict[str, object]] = []

        run_capture._stop_agent(ops, process, log)

        self.assertEqual(
            ops.signals,
            [
                (4321, signal.SIGINT),
                (4321, signal.SIGTERM),
                (4321, signal.SIGKILL),
            ],
        )
        self.assertTrue(all(item["owned_pgid"] == 4321 for item in log))

    def test_worker_timeout_terminates_only_owned_session(self) -> None:
        process = Process([subprocess.TimeoutExpired("worker", 10), 0])
        ops = WorkerOps(process)
        log: list[dict[str, object]] = []

        with self.assertRaisesRegex(run_capture.CaptureError, "controller timeout"):
            run_capture._run_owned_worker(
                ops, ["timeout", "10s", "worker"], None, None, 10, log
            )

        self.assertTrue(ops.popen_kwargs["start_new_session"])
        self.assertEqual(ops.signals, [(4321, signal.SIGTERM)])
        self.assertEqual(log[0]["owned_pgid"], 4321)

    def test_competing_agent_ignores_ancestor_argument_false_positive(self) -> None:
        profiler = Path("/opt/zymtrace/profiler/zymtrace-profiler")
        ssh = Path("/usr/bin/ssh")
        arguments = f"ssh host python run_capture.py --profiler-binary {profiler}"

        self.assertFalse(run_capture._is_competing_agent(ssh, profiler, arguments))
        self.assertTrue(run_capture._is_competing_agent(profiler, profiler, ""))

    def test_archive_manifest_rejects_tampered_source_file(self) -> None:
        repo = self.root / "archive"
        records = []
        for relative in sorted(run_capture.REQUIRED_SOURCE_PATHS):
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = (relative + "\n").encode()
            path.write_bytes(raw)
            records.append(
                {
                    "path": relative,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        manifest = self.root / "source-manifest.json"
        raw_manifest = json.dumps(
            {"repo_commit": "a" * 40, "files": records},
            separators=(",", ":"),
        ).encode()
        manifest.write_bytes(raw_manifest)
        (repo / sorted(run_capture.REQUIRED_SOURCE_PATHS)[0]).write_text(
            "tampered\n", encoding="utf-8"
        )
        args = argparse.Namespace(
            repo_root=repo,
            repo_commit="a" * 40,
            source_manifest=manifest,
            source_manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        )

        with self.assertRaisesRegex(run_capture.CaptureError, "file changed"):
            run_capture._verify_source_closure(args, RequestOps([]))

    def _events(self, *, pid: int = 41, gpu: str = GPU) -> dict[str, object]:
        return {
            "strings": ["cuda", gpu.removeprefix("GPU-")],
            "frames": [],
            "events": [
                {"c": 2.5, "pid": pid, "sid": 9, "u": 1, "k": 0, "f": []}
            ],
        }

    def test_event_audit_rejects_wrong_gpu(self) -> None:
        with self.assertRaisesRegex(run_capture.CaptureError, "no positive finite"):
            run_capture._validate_event_export(
                self._events(gpu=OTHER_GPU), expected_pid=41, expected_gpu_uuid=GPU
            )

    def test_event_audit_rejects_wrong_pid(self) -> None:
        with self.assertRaisesRegex(run_capture.CaptureError, "no positive finite"):
            run_capture._validate_event_export(
                self._events(pid=40), expected_pid=41, expected_gpu_uuid=GPU
            )

    def test_event_audit_rejects_empty_capture(self) -> None:
        document = {"strings": [], "frames": [], "events": []}
        with self.assertRaisesRegex(run_capture.CaptureError, "no positive finite"):
            run_capture._validate_event_export(document, 41, GPU)

    def test_event_audit_accepts_uuid_without_gpu_prefix(self) -> None:
        audit = run_capture._validate_event_export(self._events(), 41, GPU)
        self.assertEqual(audit["matching_cuda_rows"], 1)
        self.assertEqual(audit["source_pid"], 41)

    def test_deadline_clamps_worker_and_preserves_cleanup_allowance(self) -> None:
        now = dt.datetime(2026, 10, 26, 19, 0, tzinfo=dt.timezone.utc)
        deadline = now + dt.timedelta(seconds=200)
        self.assertEqual(run_capture._worker_timeout(now, deadline, 240), 110)
        with self.assertRaisesRegex(run_capture.CaptureError, "no safe worker"):
            run_capture._worker_timeout(
                now, now + dt.timedelta(seconds=149), 240
            )

    def test_timeout_rewrite_changes_only_builder_duration(self) -> None:
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=30s",
            "240s",
            "env",
            "A=B",
        ]
        self.assertEqual(run_capture._replace_timeout(command, 91)[3], "91s")
        self.assertEqual(command[3], "240s")


if __name__ == "__main__":
    unittest.main()
