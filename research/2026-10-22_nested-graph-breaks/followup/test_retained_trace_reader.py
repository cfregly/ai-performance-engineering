"""Non-performance synthetic tests for retained_trace_reader.

These fixtures contain invented timestamps and counts. They must never be used
as benchmark evidence or compared with retained performance results.
"""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import retained_trace_reader as reader


CPU_PID = 101
CPU_TID = 202
GPU_PID = 0
GPU_TID = 7


def cpu_marker(start: float, duration: float) -> dict:
    return {
        "ph": "X",
        "name": reader.MARKER_NAME,
        "cat": "user_annotation",
        "pid": CPU_PID,
        "tid": CPU_TID,
        "ts": start,
        "dur": duration,
        "args": {"Device Type": "CPU"},
    }


def gpu_annotation_duplicate(start: float, duration: float) -> dict:
    return {
        "ph": "X",
        "name": reader.MARKER_NAME,
        "cat": "gpu_user_annotation",
        "pid": GPU_PID,
        "tid": GPU_TID,
        "ts": start,
        "dur": duration,
        "args": {"Device Type": "CUDA"},
    }


def cuda_api(
    start: float,
    duration: float,
    correlation: int,
    *,
    pid: int = CPU_PID,
    tid: int = CPU_TID,
    name: str = "cudaGraphLaunch",
) -> dict:
    return {
        "ph": "X",
        "name": name,
        "cat": "cuda_runtime",
        "pid": pid,
        "tid": tid,
        "ts": start,
        "dur": duration,
        "args": {"correlation": correlation},
    }


def kernel(
    start: float, duration: float, correlation: int | None = None
) -> dict:
    args = {} if correlation is None else {"correlation": correlation}
    return {
        "ph": "X",
        "name": "synthetic_kernel",
        "cat": "kernel",
        "pid": GPU_PID,
        "tid": GPU_TID,
        "ts": start,
        "dur": duration,
        "args": args,
    }


class RetainedTraceReaderSyntheticTests(unittest.TestCase):
    def test_malformed_kernel_is_not_a_successful_read(self) -> None:
        broken = kernel(20.0, 5.0)
        broken["dur"] = -1
        result = reader.read_events([cpu_marker(0.0, 100.0), broken], expected_invocations=1)
        self.assertEqual(result["status"], "INVALID_EVENT_ATTRIBUTION")
        self.assertEqual(len(result["assignment_diagnostics"]["malformed_kernel_events"]), 1)

    def test_cli_returns_failure_but_retains_invalid_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "trace.json"
            output = root / "output.json"
            source.write_text(json.dumps({"traceEvents": [cpu_marker(0.0, 100.0), kernel(200.0, 5.0)]}))
            self.assertEqual(reader.main(["parse", "--trace", str(source), "--out", str(output), "--expected-invocations", "1"]), 2)
            self.assertEqual(json.loads(output.read_text())["status"], "INVALID_EVENT_ATTRIBUTION")

    def test_exact_cpu_markers_exclude_gpu_annotation_duplicates(self) -> None:
        events = [
            cpu_marker(0.0, 100.0),
            gpu_annotation_duplicate(0.0, 100.0),
            cuda_api(10.0, 2.0, 1),
            kernel(15.0, 5.0, 1),
            cpu_marker(200.0, 100.0),
            gpu_annotation_duplicate(200.0, 100.0),
            cuda_api(210.0, 2.0, 2),
            kernel(215.0, 5.0, 2),
        ]
        result = reader.read_events(events, expected_invocations=2)
        validation = result["marker_validation"]
        self.assertTrue(validation["pass"])
        self.assertEqual(validation["selected_unique_cpu_invocations"], 2)
        self.assertEqual(
            len(validation["excluded_non_cpu_or_non_complete_same_name_events"]),
            2,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["aggregate_valid_geometry_only"]["included_invocation_indices"],
            [1, 2],
        )

    def test_duplicate_cpu_marker_fails_unique_validation(self) -> None:
        marker = cpu_marker(0.0, 100.0)
        events = [marker, dict(marker), cuda_api(10.0, 2.0, 1), kernel(15.0, 5.0, 1)]
        result = reader.read_events(events, expected_invocations=1)
        self.assertFalse(result["marker_validation"]["pass"])
        self.assertEqual(
            len(result["marker_validation"]["duplicate_cpu_marker_events"]), 1
        )
        self.assertEqual(result["status"], "INVALID_CPU_MARKERS")

    def test_overlapping_cpu_markers_fail_validation(self) -> None:
        events = [cpu_marker(0.0, 100.0), cpu_marker(90.0, 100.0)]
        result = reader.read_events(events, expected_invocations=2)
        self.assertFalse(result["marker_validation"]["pass"])
        self.assertEqual(
            len(result["marker_validation"]["overlapping_cpu_intervals"]), 1
        )

    def test_cuda_api_requires_matching_cpu_thread_and_full_interval(self) -> None:
        events = [
            cpu_marker(0.0, 100.0),
            cuda_api(10.0, 2.0, 1, tid=CPU_TID + 1),
            kernel(20.0, 5.0),
        ]
        result = reader.read_events(events, expected_invocations=1)
        diagnostics = result["assignment_diagnostics"]
        self.assertEqual(len(diagnostics["unassigned_cuda_api_events"]), 1)
        self.assertEqual(result["invocations"][0]["cuda_api_count"], 0)
        self.assertEqual(result["invocations"][0]["kernel_count"], 1)

    def test_kernel_correlation_retains_negative_tail_as_boundary_violation(self) -> None:
        events = [
            cpu_marker(0.0, 100.0),
            cuda_api(10.0, 2.0, 1),
            kernel(80.0, 30.0, 1),
        ]
        result = reader.read_events(events, expected_invocations=1)
        invocation = result["invocations"][0]
        self.assertEqual(invocation["kernel_count"], 1)
        self.assertEqual(invocation["signed_cpu_tail_after_last_kernel_us"], -10.0)
        self.assertFalse(invocation["geometry_valid_for_aggregate"])
        self.assertEqual(
            result["aggregate_valid_geometry_only"]["included_invocation_indices"],
            [],
        )
        self.assertEqual(result["status"], "PASS_WITH_BOUNDARY_EXCLUSIONS")

    def test_uncorrelated_kernel_uses_unique_full_timestamp_containment(self) -> None:
        events = [cpu_marker(0.0, 100.0), kernel(20.0, 5.0)]
        result = reader.read_events(events, expected_invocations=1)
        invocation = result["invocations"][0]
        self.assertTrue(invocation["geometry_valid_for_aggregate"])
        self.assertEqual(invocation["kernel_count"], 1)
        self.assertEqual(result["status"], "PASS")

    def test_partial_cuda_api_boundary_is_reported_not_clamped(self) -> None:
        events = [
            cpu_marker(0.0, 100.0),
            cuda_api(90.0, 20.0, 1),
            kernel(20.0, 5.0),
        ]
        result = reader.read_events(events, expected_invocations=1)
        violations = result["assignment_diagnostics"][
            "cuda_api_boundary_violations"
        ]
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["end_from_cpu_marker_end_us"], 10.0)
        self.assertFalse(result["invocations"][0]["geometry_valid_for_aggregate"])


if __name__ == "__main__":
    unittest.main()
