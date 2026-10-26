#!/usr/bin/env python3
"""Synthetic regression tests for retained-trace reader integration."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
DIAGNOSTIC_SOURCE = HERE / "nested_graphs_diagnostic.py"
ADJACENT_READER_SOURCE = HERE / "retained_trace_reader.py"
READER_SOURCE = (
    ADJACENT_READER_SOURCE
    if ADJACENT_READER_SOURCE.is_file()
    else HERE.parent / "reader-fix-proposal" / "retained_trace_reader.py"
)


def _load_diagnostic() -> Any:
    spec = importlib.util.spec_from_file_location(
        "nested_graphs_diagnostic_under_test", DIAGNOSTIC_SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load diagnostic from {DIAGNOSTIC_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TRACE_READER_SOURCE = READER_SOURCE
    return module


def _cpu_marker(timestamp: float, duration: float) -> dict[str, Any]:
    return {
        "name": "nested_graphs_compiled_invocation",
        "cat": "user_annotation",
        "ph": "X",
        "pid": 101,
        "tid": 202,
        "ts": timestamp,
        "dur": duration,
    }


def _gpu_duplicate(timestamp: float, duration: float) -> dict[str, Any]:
    return {
        "name": "nested_graphs_compiled_invocation",
        "cat": "gpu_user_annotation",
        "ph": "X",
        "pid": 0,
        "tid": 7,
        "ts": timestamp,
        "dur": duration,
    }


def _cuda_api(
    timestamp: float, duration: float, correlation: int
) -> dict[str, Any]:
    return {
        "name": "cudaGraphLaunch",
        "cat": "cuda_runtime",
        "ph": "X",
        "pid": 101,
        "tid": 202,
        "ts": timestamp,
        "dur": duration,
        "args": {"correlation": correlation},
    }


def _kernel(timestamp: float, duration: float, correlation: int) -> dict[str, Any]:
    return {
        "name": "synthetic_kernel",
        "cat": "kernel",
        "ph": "X",
        "pid": 0,
        "tid": 7,
        "ts": timestamp,
        "dur": duration,
        "args": {"correlation": correlation},
    }


def _write_trace(root: Path, events: list[Mapping[str, Any]]) -> Path:
    path = root / "synthetic.chrome.json"
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return path


def _all_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_all_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_all_keys(item))
    return keys


class DiagnosticTraceReaderIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.diagnostic = _load_diagnostic()

    def test_profile_decomposition_delegates_and_retains_signed_exclusions(self) -> None:
        events = [
            _cpu_marker(0.0, 100.0),
            _gpu_duplicate(0.0, 100.0),
            _cuda_api(10.0, 2.0, 1),
            _kernel(20.0, 5.0, 1),
            _cpu_marker(200.0, 100.0),
            _gpu_duplicate(200.0, 100.0),
            _cuda_api(210.0, 2.0, 2),
            _kernel(280.0, 30.0, 2),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = _write_trace(Path(temporary), events)
            result = self.diagnostic._profile_decomposition(
                trace_path, expected_invocations=2
            )

        validation = result["marker_validation"]
        self.assertTrue(validation["pass"])
        self.assertEqual(validation["same_name_event_count"], 4)
        self.assertEqual(validation["selected_unique_cpu_invocations"], 2)
        self.assertEqual(result["exclusion_counts"]["excluded_same_name_marker_events"], 2)
        self.assertEqual(result["exclusion_counts"]["kernel_boundary_violations"], 1)
        self.assertEqual(
            result["exclusion_counts"]["excluded_invalid_geometry_invocations"], 1
        )
        self.assertEqual(result["status"], "PASS_WITH_BOUNDARY_EXCLUSIONS")
        self.assertEqual(
            result["invocations"][1]["signed_cpu_tail_after_last_kernel_us"],
            -10.0,
        )
        self.assertEqual(
            result["aggregate_valid_geometry_only"]["included_invocation_indices"],
            [1],
        )
        self.assertFalse(
            any("fraction" in key.lower() or "percentage" in key.lower() for key in _all_keys(result))
        )

    def test_profile_decomposition_rejects_invalid_cpu_markers(self) -> None:
        events = [_gpu_duplicate(0.0, 100.0), _kernel(20.0, 5.0, 1)]
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = _write_trace(Path(temporary), events)
            with self.assertRaisesRegex(RuntimeError, "rejected the CPU invocation"):
                self.diagnostic._profile_decomposition(
                    trace_path, expected_invocations=1
                )

    def test_profile_decomposition_fails_closed_on_malformed_kernel(self) -> None:
        events = [
            _cpu_marker(0.0, 100.0),
            _cuda_api(10.0, 2.0, 1),
            _kernel(20.0, 5.0, 1),
            {
                "name": "malformed_kernel",
                "cat": "kernel",
                "ph": "X",
                "pid": 0,
                "tid": 7,
                "ts": 30.0,
                "dur": "not-a-duration",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = _write_trace(Path(temporary), events)
            with self.assertRaisesRegex(RuntimeError, "non-attributable event geometry"):
                self.diagnostic._profile_decomposition(
                    trace_path, expected_invocations=1
                )

    def test_manifest_records_distinct_diagnostic_and_reader_sources(self) -> None:
        manifest = self.diagnostic._diagnostic_source_manifest()
        diagnostic = manifest["diagnostic"]
        reader = manifest["retained_trace_reader"]
        self.assertEqual(Path(diagnostic["path"]), DIAGNOSTIC_SOURCE)
        self.assertEqual(Path(reader["path"]), READER_SOURCE)
        self.assertNotEqual(diagnostic["sha256"], reader["sha256"])
        self.assertEqual(diagnostic["sha256"], self.diagnostic._sha256_file(DIAGNOSTIC_SOURCE))
        self.assertEqual(reader["sha256"], self.diagnostic._sha256_file(READER_SOURCE))


if __name__ == "__main__":
    unittest.main()
