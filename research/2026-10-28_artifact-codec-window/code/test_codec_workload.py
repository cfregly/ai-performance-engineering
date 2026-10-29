"""Synthetic stand-in programs test custody and correctness, not codec performance."""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

MODULE = Path(__file__).with_name("codec_workload.py")
spec = importlib.util.spec_from_file_location("codec_workload", MODULE)
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


class CodecWorkloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="codec-workload-synthetic-")
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.output = self.root / "output"
        self.output.mkdir()
        self.parts = [
            b"synthetic alpha bytes\x00\xff\n",
            b"different synthetic beta bytes\n",
            b"synthetic gamma bytes\n",
            b"synthetic delta bytes\n",
        ]
        refs = []
        for number, raw in enumerate(self.parts):
            name = f"part-{number:03d}.bin"
            (self.inputs / name).write_bytes(raw)
            refs.append({"artifactName": "inputs/" + name, "sha256": workload._sha(raw), "sizeBytes": len(raw)})
        ledger = workload._bytes({"fixture": "synthetic bytes, not a measured codec corpus", "files": ["alpha", "beta"]})
        (self.inputs / "corpus-ledger.json").write_bytes(ledger)
        self.index = self.root / "input-index.json"
        self.index.write_bytes(workload._bytes({
            "schemaVersion": "piq.artifact-compression-chunks.v1",
            "corpusSha256": workload._sha(b"".join(self.parts)),
            "corpusBytes": sum(map(len, self.parts)), "sourceFileCount": 4,
            "ledger": {"artifactName": "inputs/corpus-ledger.json", "sha256": workload._sha(ledger), "sizeBytes": len(ledger)},
            "chunks": refs,
        }))
        tools = {}
        for name in ("pigz", "zstd"):
            program = self.root / (name + "-standin")
            program.write_text("#!/bin/sh\n# Synthetic identity transform for " + name + ".\n/bin/cat\n")
            program.chmod(0o700)
            tools[name] = {"path": str(program), "sha256": workload._file_hash(program), "version": "synthetic stand-in, not a native codec"}
        library = self.root / "standin-library.bin"
        library.write_bytes(b"synthetic toolchain validation fixture only")
        self.toolchain = {"kind": "piq.codec-toolchain.v1", "threads": 2, "tools": tools,
                          "libraries": [{"path": str(library), "sha256": workload._file_hash(library)}]}
        self.toolchain_json = json.dumps(self.toolchain)
        self.host_patch = mock.patch.object(workload, "_linux_host_context", return_value={
            "fixture": "synthetic host context, not native telemetry", "cpu_affinity": [0, 1]})
        self.host_patch.start()
        self.addCleanup(self.host_patch.stop)
        self.usage_patch = mock.patch.object(workload, "_usage", return_value={
            "fixture": "synthetic usage, not native telemetry", "user_cpu_seconds": 0,
            "system_cpu_seconds": 0, "minor_faults": 0, "major_faults": 0,
            "block_input_operations": 0, "block_output_operations": 0,
            "high_water_rss_kib_linux": 0})
        self.usage_mock = self.usage_patch.start()
        self.addCleanup(self.usage_patch.stop)

    def request(self):
        return {"input_index": str(self.index), "output_directory": str(self.output),
                "codec": "pigz-gzip", "toolchain_json": self.toolchain_json, "timeout_seconds": 2}

    def test_standin_roundtrip_executes_core_and_retains_exact_refs(self):
        payload = workload.run_workload(self.index, self.output, "pigz-gzip", self.toolchain_json, 2)
        result = json.loads((self.output / "result.json").read_bytes())
        evidence = json.loads((self.output / "native-evidence.json").read_bytes())
        self.assertEqual(result["input_sha256"], workload._sha(b"".join(self.parts)))
        self.assertEqual(result["input_sha256"], result["roundtrip_sha256"])
        self.assertEqual(result["chunk_count"], 4)
        self.assertEqual(payload["observation_mode"], "standalone_workload")
        self.assertEqual(payload["qualification"], "not_assessed")
        self.assertEqual(payload["timing_scope"], "adapter_local_workload_window")
        self.assertEqual(evidence["qualification"], "not_assessed")
        self.assertEqual(evidence["host"]["fixture"], "synthetic host context, not native telemetry")
        self.assertEqual(evidence["rusage"]["before"]["self"]["fixture"], "synthetic usage, not native telemetry")
        self.assertNotIn("primary_process_wall_ns", payload)
        for ref in (payload["result"], payload["observations"], *payload["native_evidence"].values()):
            raw = (self.output / ref["artifactName"]).read_bytes()
            self.assertEqual(ref["sha256"], workload._sha(raw))
            self.assertEqual(ref["sizeBytes"], len(raw))
        self.assertEqual(json.loads((self.output / "workload-result.json").read_bytes()), payload)
        self.assertFalse(any(self.output.glob("roundtrip-*.bin")))

    def test_zstd_standin_uses_the_preserved_native_argv(self):
        workload.run_workload(self.index, self.output, "zstd", self.toolchain_json, 2)
        evidence = json.loads((self.output / "native-evidence.json").read_bytes())
        self.assertEqual(evidence["parts"][0]["compression"]["argv"][1:], ["-3", "-T2", "--no-progress", "-q", "-c"])
        self.assertEqual(evidence["parts"][0]["decompression"]["argv"][1:], ["-d", "--no-progress", "-q", "-c"])

    def test_input_or_toolchain_drift_refuses_without_a_success_receipt(self):
        (self.inputs / "part-000.bin").write_bytes(b"changed input")
        with self.assertRaisesRegex(ValueError, "retained bytes"):
            workload.run_workload(self.index, self.output, "pigz-gzip", self.toolchain_json, 2)
        self.assertFalse((self.output / "workload-result.json").exists())
        tool = Path(self.toolchain["tools"]["pigz"]["path"])
        tool.write_text("#!/bin/sh\nexit 1\n")
        with self.assertRaisesRegex(ValueError, "identity drift"):
            workload._toolchain(self.toolchain_json)

    def test_stderr_is_retained_and_refuses_success(self):
        program = Path(self.toolchain["tools"]["pigz"]["path"])
        program.write_text("#!/bin/sh\n/bin/cat\nprintf diagnostic >&2\n")
        self.toolchain["tools"]["pigz"]["sha256"] = workload._file_hash(program)
        with self.assertRaisesRegex(ValueError, "emitted diagnostics"):
            workload.run_workload(self.index, self.output, "pigz-gzip", json.dumps(self.toolchain), 2)
        retained = json.loads((self.output / "archive-000.gz.stderr.json").read_bytes())
        self.assertEqual(base64.b64decode(retained["data"]), b"diagnostic")
        self.assertEqual(retained["original_sha256"], workload._sha(b"diagnostic"))
        self.assertFalse((self.output / "workload-result.json").exists())

    def test_request_schema_is_closed_and_bounded(self):
        path = self.root / "request.json"
        value = self.request()
        path.write_bytes(workload._bytes(value))
        self.assertEqual(workload._load_request(path), value)
        for mutation in ({**value, "extra": True}, {key: item for key, item in value.items() if key != "codec"},
                         {**value, "timeout_seconds": True}, {**value, "codec": "other"}):
            path.write_bytes(workload._bytes(mutation))
            with self.assertRaises(ValueError):
                workload._load_request(path)

    def test_sealed_input_caps_are_narrow_and_fail_closed(self):
        self.assertEqual(workload.MAX_CORPUS_BYTES, 128 * 1024 * 1024)
        self.assertEqual(workload.MAX_PARTS, 4)
        self.assertEqual(workload.MAX_PART_BYTES, 30 * 1024 * 1024)
        self.assertEqual(workload.MAX_LEDGER_BYTES, 2 * 1024 * 1024)
        self.assertEqual(workload.MAX_OUTPUT_BYTES, 32 * 1024 * 1024)

        def encoded(part_sizes, *, ledger_size=1, corpus_bytes=None):
            parts = [
                {"artifactName": f"inputs/part-{number:03d}.bin",
                 "sha256": f"{number + 1:064x}", "sizeBytes": size}
                for number, size in enumerate(part_sizes)
            ]
            total = sum(part_sizes) if corpus_bytes is None else corpus_bytes
            return workload._bytes({
                "schemaVersion": "piq.artifact-compression-chunks.v1",
                "corpusSha256": "f" * 64,
                "corpusBytes": total,
                "sourceFileCount": 4096,
                "ledger": {"artifactName": "inputs/corpus-ledger.json",
                           "sha256": "e" * 64, "sizeBytes": ledger_size},
                "chunks": parts,
            })

        valid = [workload.MAX_PART_BYTES] * 4
        self.assertEqual(len(workload._index(encoded(valid))["chunks"]), 4)
        for sealed in (
            encoded(valid[:3]),
            encoded([1] * 5),
            encoded([workload.MAX_PART_BYTES + 1, 1, 1, 1]),
            encoded([1] * 4, ledger_size=workload.MAX_LEDGER_BYTES + 1),
            encoded([1] * 4, corpus_bytes=workload.MAX_CORPUS_BYTES + 1),
        ):
            with self.assertRaises(ValueError):
                workload._index(sealed)

        with mock.patch.object(workload, "MAX_OUTPUT_BYTES", 4):
            with self.assertRaisesRegex(ValueError, "output exceeds"):
                workload._write(self.output / "too-large.bin", b"12345")
        self.assertFalse((self.output / "too-large.bin").exists())

    def test_existing_output_is_preserved(self):
        path = self.output / "native-evidence.json"
        path.write_bytes(b"existing owner bytes")
        with self.assertRaisesRegex(ValueError, "already exists"):
            workload.run_workload(self.index, self.output, "pigz-gzip", self.toolchain_json, 2)
        self.assertEqual(path.read_bytes(), b"existing owner bytes")
        self.assertFalse((self.output / "corpus-ledger.json").exists())

    def test_library_refuses_non_linux_before_outputs(self):
        self.host_patch.stop()
        with mock.patch.object(workload.sys, "platform", "darwin"):
            with self.assertRaisesRegex(ValueError, "requires Linux"):
                workload.run_workload(self.index, self.output, "pigz-gzip", self.toolchain_json, 2)
        self.assertEqual(list(self.output.iterdir()), [])
        self.usage_mock.assert_not_called()

    def test_real_cli_help_and_platform_refusal(self):
        help_result = subprocess.run([sys.executable, "-I", "-B", "-X", "pycache_prefix=/var/empty/piq-python-cache", str(MODULE), "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--request", help_result.stdout)
        if sys.platform != "linux":
            result = subprocess.run([sys.executable, "-I", "-B", "-X", "pycache_prefix=/var/empty/piq-python-cache", str(MODULE), "--request", str(self.root / "missing.json")], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires Linux", result.stderr)

    @unittest.skipUnless(sys.platform == "linux" and hasattr(os, "sched_getaffinity"), "Linux CLI resource path requires Linux")
    def test_linux_cli_applies_real_process_bounds_with_standins(self):
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < 2:
            self.skipTest("two CPUs are required")
        path = self.root / "request.json"
        path.write_bytes(workload._bytes(self.request()))
        def bind_affinity():
            os.sched_setaffinity(0, cpus[:2])
        result = subprocess.run([sys.executable, "-I", "-B", "-X", "pycache_prefix=/var/empty/piq-python-cache", str(MODULE), "--request", str(path)],
                                capture_output=True, text=True, preexec_fn=bind_affinity)
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = json.loads((self.output / "native-evidence.json").read_bytes())
        self.assertEqual(len(evidence["host"]["cpu_affinity"]), 2)
        self.assertTrue(all(0 < value <= 512 * 1024**2 for value in evidence["runtime"]["rlimits"]["address_space"]))
        self.assertTrue(all(0 < value <= 32 * 1024**2 for value in evidence["runtime"]["rlimits"]["file_size"]))


if __name__ == "__main__":
    unittest.main()
