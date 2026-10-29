"""Control-flow checks use explicitly synthetic outputs, never codec measurements."""
import importlib.util
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("run_codec_pair", Path(__file__).with_name("run_codec_pair.py"))
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def synthetic_outputs(directory: Path, correct: bool = True, index: Path | None = None) -> None:
    value = {"input_sha256": "a" * 64, "roundtrip_sha256": ("a" if correct else "b") * 64,
             "SYNTHETIC_NONEXECUTED": True}
    if index is not None:
        value.update(input_manifest_sha256=driver.digest(index), input_bytes=1, chunk_count=1,
                     source_file_count=1, corpus_ledger_sha256="c" * 64)
    driver.retain(directory / "result.json", value)
    driver.retain(directory / "native-evidence.json", {
        "observation_mode": "standalone_workload", "qualification": "not_assessed",
        "SYNTHETIC_NONEXECUTED": True})
    def ref(name):
        path = directory / name
        return {"artifactName": name, "sha256": driver.digest(path), "sizeBytes": path.stat().st_size}
    driver.retain(directory / "workload-result.json", {
        "observation_mode": "standalone_workload", "qualification": "not_assessed",
        "timing_scope": "adapter_local_workload_window", "result": ref("result.json"),
        "observations": ref("native-evidence.json"), "native_evidence": {},
        "SYNTHETIC_NONEXECUTED": True})


class PairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.index = self.root / "input-index.json"
        self.index.write_text(json.dumps({"corpusBytes": 1, "corpusSha256": "a" * 64,
                                         "chunks": [{}], "sourceFileCount": 1,
                                         "ledger": {"sha256": "c" * 64}}))
        self.source = self.root / "synthetic_workload.py"
        self.source.write_text('# Synthetic fixture, never a compression workload.\n')
        paths = []
        for name in ("pigz", "zstd", "loader"):
            path = self.root / name
            path.write_text("SYNTHETIC NONEXECUTED " + name)
            paths.append(path)
        def ref(path):
            return {"path": str(path), "sha256": driver.digest(path)}
        value = {"kind": "piq.codec-toolchain.v1", "threads": 2,
                 "tools": {"pigz": {**ref(paths[0]), "version": "2.5"},
                           "zstd": {**ref(paths[1]), "version": "1.5.5"}},
                 "libraries": [ref(paths[2])]}
        self.toolchain = self.root / "toolchain.json"
        self.toolchain.write_bytes(driver.canonical(value))

    def tearDown(self):
        self.temporary.cleanup()

    def run_fixture(self, executor, order="AB"):
        with patch.object(driver, "require_linux"), patch.object(driver, "parse_cpus", return_value=(0, 1)), patch.object(driver, "fresh_process", side_effect=executor):
            return driver.run(self.index, self.toolchain, self.root / "output", "0,1", order, workload=self.source)

    def test_pair_order_request_and_complete_handback(self):
        calls = []
        def execute(argv, directory, cpus, timeout):
            self.assertEqual(argv[:5], [sys.executable, "-I", "-B", "-X", "pycache_prefix=/var/empty/piq-python-cache"])
            request = json.loads(Path(argv[-1]).read_text())
            self.assertEqual(set(request), {"input_index", "output_directory", "codec", "toolchain_json", "timeout_seconds"})
            self.assertEqual((cpus, timeout), ((0, 1), 15))
            calls.append(request["codec"])
            synthetic_outputs(directory, index=self.index)
            return {"state": "COMPLETE", "pid": 12345, "exit_code": 0,
                    "complete_process_wall_ns": 123, "SYNTHETIC_NONEXECUTED": True}
        result = self.run_fixture(execute, "BA")
        self.assertEqual(result["state"], "COMPLETE")
        self.assertEqual(calls, ["zstd", "pigz-gzip"])
        self.assertEqual(driver.FILE_CAP, 32 * 1024 * 1024)
        self.assertEqual(result["limits"]["file_bytes"], driver.FILE_CAP)
        self.assertEqual(result["qualification"], "not_assessed")
        self.assertEqual(json.loads((self.root / "output/handback.json").read_text()), result)

    def test_failed_arm_preserved_no_retry_or_second_arm(self):
        calls = []
        def execute(*args):
            calls.append(args)
            return {"state": "FAILED", "pid": 12345, "exit_code": 7, "timed_out": False}
        result = self.run_fixture(execute)
        self.assertEqual(len(calls), 1)
        self.assertEqual([x["state"] for x in result["arms"]], ["FAILED", "NOT_STARTED"])
        self.assertEqual(result["state"], "PARTIAL")
        self.assertTrue((self.root / "output/A/process.json").exists())

    def test_bad_roundtrip_stops_pair(self):
        def execute(argv, directory, *args):
            synthetic_outputs(directory, correct=False, index=self.index)
            return {"state": "COMPLETE", "pid": 12345, "exit_code": 0}
        result = self.run_fixture(execute)
        self.assertEqual(result["state"], "PARTIAL")
        self.assertEqual([x["state"] for x in result["arms"]], ["FAILED_OUTPUT_VALIDATION", "NOT_STARTED"])

    def test_reference_drift_and_escape_refused(self):
        directory = self.root / "results"
        directory.mkdir()
        synthetic_outputs(directory)
        (directory / "result.json").write_bytes(b'{}')
        with self.assertRaises(ValueError):
            driver.validate_outputs(directory)
        with self.assertRaises(ValueError):
            driver.verify_reference(directory, {"artifactName": "../toolchain.json", "sha256": "a" * 64})

    def test_exclusive_output(self):
        output = self.root / "exists"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            driver.run(self.index, self.toolchain, output, "0,1", workload=self.source)

    def test_unknown_toolchain_and_pin_drift_refused(self):
        value = driver.document(self.toolchain)
        value["unknown"] = True
        with self.assertRaises(ValueError):
            driver.validate_toolchain(value)
        value.pop("unknown")
        Path(value["tools"]["pigz"]["path"]).write_text("changed")
        with self.assertRaises(ValueError):
            driver.validate_toolchain(value)

    def test_ldd_missing_unknown_and_loader_absent_refused(self):
        for raw in (b'libx.so => not found\n', b'not a dynamic executable\n', b'unknown\n', b''):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                driver.parse_ldd(raw)

    def test_pin_retains_synthetic_native_metadata(self):
        loader = self.root / "ld-linux-synthetic.so.2"
        loader.write_text("SYNTHETIC NONEXECUTED loader")
        ldd = self.root / "ldd-synthetic"
        ldd.write_text("SYNTHETIC NONEXECUTED metadata tool")
        actual_regular = driver.regular
        def resolve(path):
            return ldd if path == Path("/usr/bin/ldd") else actual_regular(path)
        def execute(argv, directory, *args):
            if "--version" in argv:
                raw = b"pigz 2.5\n" if Path(argv[0]).name == "pigz" else b"Zstandard CLI v1.5.5\n"
            else:
                raw = (str(loader) + " (0x1234)\n").encode()
            (directory / "stdout.txt").write_bytes(raw)
            (directory / "stderr.txt").write_bytes(b"")
            return {"state": "COMPLETE", "SYNTHETIC_NONEXECUTED": True}
        with patch.object(driver, "require_linux"), patch.object(driver, "regular", side_effect=resolve), patch.object(driver, "fresh_process", side_effect=execute):
            result = driver.pin(self.root / "pigz", self.root / "zstd", self.root / "pins")
        self.assertEqual(result["state"], "COMPLETE")
        self.assertEqual(driver.document(self.root / "pins/toolchain.json")["libraries"][0]["path"], str(loader))
        self.assertEqual((self.root / "pins/pigz-version/stdout.txt").read_bytes(), b"pigz 2.5\n")
        self.assertTrue((self.root / "pins/zstd-ldd/process.json").is_file())

    def test_cpu_selection_requires_two_allowed_ids(self):
        with patch.object(driver.os, "sched_getaffinity", return_value={2, 3}, create=True):
            self.assertEqual(driver.parse_cpus("2,3"), (2, 3))
            for value in ("2,2", "0,1", "2", "2,3,4", "-1,2"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    driver.parse_cpus(value)

    def test_existing_pycache_prefix_is_refused_without_creation(self):
        with patch.object(driver.sys, "platform", "linux"), patch.object(driver.os, "waitid", create=True), patch.object(driver.os, "WNOWAIT", 0x1000000, create=True), patch.object(driver.os.path, "lexists", return_value=True) as exists:
            with self.assertRaisesRegex(ValueError, "pycache prefix must be absent"):
                driver.require_linux()
        exists.assert_called_once_with("/var/empty/piq-python-cache")

    def test_cli_help_real_repository_entrypoint(self):
        completed = subprocess.run([sys.executable, *driver.PYTHON_FLAGS, str(Path(driver.__file__)), "--help"],
                                   capture_output=True, timeout=5)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(b"pin", completed.stdout)
        self.assertIn(b"run", completed.stdout)


class ProcessOwnershipTests(unittest.TestCase):
    def test_kill_group_precedes_reap_without_poll(self):
        events = []
        class Child:
            pid = 12345
            def wait(self, timeout):
                events.append("reap")
                return 0
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(driver.subprocess, "Popen", return_value=Child()), patch.object(driver.os, "waitid", return_value=object(), create=True), patch.object(driver.os, "P_PID", 1, create=True), patch.object(driver.os, "WEXITED", 4, create=True), patch.object(driver.os, "WNOHANG", 1, create=True), patch.object(driver.os, "WNOWAIT", 0x1000000, create=True), patch.object(driver.os, "killpg", side_effect=lambda pid, sig: events.append((pid, sig))), patch.object(driver.time, "monotonic_ns", side_effect=itertools.count(100, 10)):
                result = driver.fresh_process(["synthetic-executor"], Path(temporary), None, 1)
        self.assertEqual(events, [(12345, signal.SIGKILL), "reap"])
        self.assertTrue(result["cleanup"]["leader_reaped"])
        self.assertLess(result["process_exit_observed_ns"], result["finished_monotonic_ns"])
        self.assertEqual(result["complete_process_wall_ns"], result["finished_monotonic_ns"] - result["started_monotonic_ns"])

    @unittest.skipUnless(sys.platform == "linux", "Linux process-limit path")
    def test_tiny_real_executor_exit_and_timeout(self):
        driver.require_linux()
        cpus = tuple(sorted(os.sched_getaffinity(0))[:2])
        if len(cpus) != 2:
            self.skipTest("two allowed CPUs required")
        with tempfile.TemporaryDirectory() as temporary:
            success_fixture = Path(temporary) / "success_fixture.py"
            success_fixture.write_text("print('tiny executor')\n")
            timeout_fixture = Path(temporary) / "timeout_fixture.py"
            timeout_fixture.write_text("import time\ntime.sleep(5)\n")
            first = Path(temporary) / "success"
            first.mkdir()
            result = driver.fresh_process([sys.executable, *driver.PYTHON_FLAGS, str(success_fixture)], first, cpus, 2)
            self.assertEqual(result["state"], "COMPLETE")
            self.assertGreater(result["complete_process_wall_ns"], 0)
            second = Path(temporary) / "timeout"
            second.mkdir()
            result = driver.fresh_process([sys.executable, *driver.PYTHON_FLAGS, str(timeout_fixture)], second, cpus, 0.05)
            self.assertTrue(result["timed_out"])
            self.assertEqual(result["exit_code"], -signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
