#!/usr/bin/env python3
"""Synthetic parser and selection tests for retained profile views."""

from __future__ import annotations

import csv
import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


SOURCE = Path(__file__).resolve().with_name("render_profile_views.py")


def _load_renderer() -> Any:
    spec = importlib.util.spec_from_file_location("profile_view_renderer", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load renderer from {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


renderer = _load_renderer()


def _build_sqlite(path: Path, per_invocation: int) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                correlationId INTEGER,
                globalTid INTEGER,
                nameId INTEGER NOT NULL
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                correlationId INTEGER,
                streamId INTEGER NOT NULL,
                shortName INTEGER NOT NULL
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                correlationId INTEGER,
                streamId INTEGER NOT NULL,
                graphId INTEGER NOT NULL,
                graphExecId INTEGER NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO StringIds(id, value) VALUES (?, ?)",
            [
                (1, "cudaGraphLaunch_v10000"),
                (2, "cudaDeviceSynchronize_v3020"),
                (3, "cudaLaunchKernel_v7000"),
                (4, "cuKernelGetName"),
                (5, "synthetic_gpu_kernel"),
            ],
        )
        target_group = renderer.PRIOR_REPLAY_INVOCATIONS
        tid = 91
        graph_correlation = 1000
        kernel_correlation = 100000
        for group in range(renderer.TOTAL_REPLAY_INVOCATIONS):
            base = 1_000_000 + group * 100_000
            if group == target_group:
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
                    (base + 1000, base + 2000, 90, tid, 2),
                )
            for index in range(per_invocation):
                graph_start = base + 10_000 + index * 3000
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
                    (
                        graph_start,
                        graph_start + 500,
                        graph_correlation,
                        tid,
                        1,
                    ),
                )
                if group == target_group:
                    connection.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
                        (
                            graph_start - 900,
                            graph_start - 600,
                            kernel_correlation,
                            tid,
                            3,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
                        (
                            graph_start + 100,
                            graph_start + 800,
                            kernel_correlation,
                            7,
                            5,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            graph_start + 900,
                            graph_start + 1400,
                            graph_correlation,
                            7,
                            index + 1,
                            index + 101,
                        ),
                    )
                    kernel_correlation += 1
                graph_correlation += 1
            if group == target_group:
                last_end = base + 10_000 + (per_invocation - 1) * 3000 + 800
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
                    (last_end + 1000, last_end + 2000, 99, tid, 2),
                )
        connection.commit()
    finally:
        connection.close()


def _write_ncu_page(path: Path, *, extra_row: bool = False) -> None:
    header = [
        "Kernel Name",
        "device__attribute_display_name",
        *renderer.NCU_METRICS,
    ]
    units = [
        "",
        "",
        "us",
        "%",
        "%",
        "%",
        "%",
        "",
        "",
        "register/thread",
        "Kbyte/block",
        "pass",
    ]
    values = [
        "synthetic_kernel",
        "Synthetic GPU",
        "006.400000",
        "17.261867",
        "30.132462",
        "75.796713",
        "11.632164",
        "2048",
        "256",
        "18",
        "1.024000",
        "9.000000",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(units)
        writer.writerow(values)
        if extra_row:
            writer.writerow(values)


class NsysSelectionTests(unittest.TestCase):
    def test_source_ordinal_and_boundaries_select_false_and_true(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm, count in (("false", 16), ("true", 2)):
                path = root / f"{arm}.sqlite"
                _build_sqlite(path, count)
                selection = renderer.select_first_post_warmup_invocation(
                    path, arm=arm
                )
                self.assertEqual(selection.graph_launch_ordinal_start, 34 * count + 1)
                self.assertEqual(selection.graph_launch_ordinal_end, 35 * count)
                self.assertEqual(selection.target_compiled_call_ordinal, 36)
                self.assertEqual(len(selection.graph_launches), count)
                self.assertEqual(len(selection.graph_executions), count)
                self.assertEqual(len(selection.kernels), count)
                self.assertLess(selection.pre_sync.end, selection.graph_launches[0].start)
                self.assertLess(selection.graph_launches[-1].end, selection.post_sync.start)

    def test_missing_graph_launch_fails_protocol_total(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "false.sqlite"
            _build_sqlite(path, 16)
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                    "WHERE rowid = (SELECT max(rowid) FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                    "WHERE nameId = 1)"
                )
                connection.commit()
            finally:
                connection.close()
            with self.assertRaisesRegex(renderer.RenderError, "graph-launch total"):
                renderer.select_first_post_warmup_invocation(path, arm="false")

    def test_missing_graph_execution_fails_graph_trace_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "true.sqlite"
            _build_sqlite(path, 2)
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "DELETE FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
                    "WHERE rowid = (SELECT max(rowid) "
                    "FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE)"
                )
                connection.commit()
            finally:
                connection.close()
            with self.assertRaisesRegex(renderer.RenderError, "execution count"):
                renderer.select_first_post_warmup_invocation(path, arm="true")

    def test_extra_launch_inside_sync_window_fails_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "true.sqlite"
            _build_sqlite(path, 2)
            connection = sqlite3.connect(path)
            try:
                target_offset = renderer.PRIOR_REPLAY_INVOCATIONS * 2
                graph_rows = connection.execute(
                    "SELECT rowid, start FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                    "WHERE nameId = 1 ORDER BY start"
                ).fetchall()
                post_start = connection.execute(
                    "SELECT start FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                    "WHERE nameId = 2 ORDER BY start DESC LIMIT 1"
                ).fetchone()[0]
                rowid, _ = graph_rows[target_offset + 2]
                connection.execute(
                    "UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start=?, end=? WHERE rowid=?",
                    (post_start - 800, post_start - 500, rowid),
                )
                connection.commit()
            finally:
                connection.close()
            with self.assertRaisesRegex(renderer.RenderError, "does not isolate"):
                renderer.select_first_post_warmup_invocation(path, arm="true")


class NcuPageTests(unittest.TestCase):
    def test_values_and_units_remain_verbatim_strings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "page.csv"
            _write_ncu_page(path)
            page = renderer.read_ncu_page(path)
            self.assertEqual(page.values["gpu__time_duration.avg"], "006.400000")
            self.assertEqual(page.units["gpu__time_duration.avg"], "us")
            self.assertEqual(page.values["profiler__replayer_passes"], "9.000000")

    def test_second_data_row_fails_single_kernel_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "page.csv"
            _write_ncu_page(path, extra_row=True)
            with self.assertRaisesRegex(renderer.RenderError, "one data row"):
                renderer.read_ncu_page(path)


class PublicManifestTests(unittest.TestCase):
    def test_repo_manifest_relative_paths_identify_the_same_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.sqlite"
            path.write_bytes(b"synthetic input")
            manifest = {"files": [{"path": "exports/baseline.sqlite", "bytes": path.stat().st_size,
                                    "sha256": renderer._sha256_file(path)}]}
            result = renderer.verify_manifest_input(path, manifest)
            self.assertEqual(result["sha256"], manifest["files"][0]["sha256"])
            manifest["files"].append(dict(manifest["files"][0], path="other/baseline.sqlite"))
            with self.assertRaisesRegex(renderer.RenderError, "one member"):
                renderer.verify_manifest_input(path, manifest)


if __name__ == "__main__":
    unittest.main()
