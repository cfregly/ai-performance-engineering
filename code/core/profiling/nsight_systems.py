#!/usr/bin/env python3
"""Stdlib-only parsing and ranking for offline Nsight Systems reports."""

from __future__ import annotations

import csv
import hashlib
import heapq
import math
import re
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

__all__ = ["NsightSystemsReportError", "NsightSystemsReportParser"]

_TIME_PERCENT_COLUMNS = ("Time (%)", "Time (%) [sum]")


def _parse_time_percentage(row: dict[str, str]) -> float | None:
    """Return one unambiguous, finite percentage in the closed range 0..100."""
    parsed_values: list[float] = []
    for column in _TIME_PERCENT_COLUMNS:
        raw_value = row.get(column)
        if raw_value is None or not str(raw_value).strip():
            continue
        text = str(raw_value).strip()
        if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
        if text.endswith("%"):
            text = text[:-1].strip()
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text) is None:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            return None
        parsed_values.append(value)
    if not parsed_values or any(value != parsed_values[0] for value in parsed_values[1:]):
        return None
    return parsed_values[0]


class NsightSystemsReportError(RuntimeError):
    """Nsight parsing failure that preserves the source-artifact receipt."""

    def __init__(self, message: str, source: dict[str, Any]) -> None:
        super().__init__(message)
        self.source = source


class NsightSystemsReportParser:
    """Parse and rank data exported by the ``nsys stats`` command."""

    STATS_TIMEOUT_SECONDS = 60.0

    @classmethod
    def summarize_report(
        cls,
        report_path: str,
        *,
        kernel_regex: str | None = None,
        top_k: int = 5,
        print_summary: bool = True,
    ) -> dict[str, Any]:
        """Parse an Nsight Systems report via `nsys stats`."""
        report = Path(report_path).expanduser()
        source: dict[str, Any] = {
            "requested_path": report_path,
            "resolved_path": str(report.resolve()),
            "sha256": None,
            "size_bytes": None,
            "status": "pending",
            "error": None,
        }
        if not report.exists():
            source["status"] = "missing"
            source["error"] = "file not found"
            raise NsightSystemsReportError(
                f"Nsight Systems report not found: {report_path}", source
            )
        if not report.is_file():
            source["status"] = "error"
            source["error"] = "input is not a regular file"
            raise NsightSystemsReportError(
                f"Nsight Systems report is not a regular file: {report_path}", source
            )
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        # Some nsys versions create or refresh a sibling SQLite export while
        # reading a report. Work only from a temporary copy so offline analysis
        # cannot mutate the evidence artifact supplied by the caller.
        with tempfile.TemporaryDirectory(prefix="aisp-nsys-stats-") as temp_dir:
            temp_report = Path(temp_dir) / report.name
            try:
                digest = hashlib.sha256()
                size_bytes = 0
                with report.open("rb") as input_stream, temp_report.open("wb") as snapshot:
                    for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size_bytes += len(chunk)
                        snapshot.write(chunk)
                source["sha256"] = digest.hexdigest()
                source["size_bytes"] = size_bytes
                source["status"] = "snapshotted"
                summary_rows = cls._run_nsys_stats(temp_report, "cuda_api_sum")
                kernel_rows = cls._run_nsys_stats(temp_report, "cuda_gpu_kern_sum")
            except Exception as exc:
                source["status"] = "error"
                source["error"] = str(exc)
                raise NsightSystemsReportError(str(exc), source) from exc
        try:
            kernels = cls._filter_and_rank_kernels(kernel_rows, kernel_regex, top_k)
        except Exception as exc:
            source["status"] = "error"
            source["error"] = str(exc)
            raise NsightSystemsReportError(str(exc), source) from exc
        source["status"] = "parsed"

        summary: dict[str, Any] = {
            "report": str(report.resolve()),
            "source": source,
            "summary": summary_rows,
            "kernels": kernels,
        }
        if print_summary:
            cls._print_nsys_summary(summary, kernel_regex)
        return summary

    @staticmethod
    def _run_nsys_stats(report: Path, section: str) -> list[dict[str, str]]:
        cmd = [
            "nsys",
            "stats",
            "--format",
            "csv",
            "--report",
            section,
            str(report),
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=NsightSystemsReportParser.STATS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - CLI
            raise RuntimeError(
                "nsys stats timed out after "
                f"{NsightSystemsReportParser.STATS_TIMEOUT_SECONDS:g}s for {report}"
            ) from exc
        except subprocess.CalledProcessError as exc:  # pragma: no cover - CLI
            raise RuntimeError(f"Failed to run {' '.join(cmd)}\n{exc.stderr.strip()}") from exc

        rows: list[dict[str, str]] = []
        headers: list[str] | None = None
        reader = csv.reader(proc.stdout.splitlines())
        for raw_row in reader:
            if not raw_row:
                continue
            if raw_row[0].startswith("#") or raw_row[0].startswith("NOTICE"):
                continue
            if headers is None:
                candidate = [column.strip().lstrip("\ufeff") for column in raw_row]
                if "Name" in candidate and any(
                    column in candidate for column in _TIME_PERCENT_COLUMNS
                ):
                    headers = candidate
                continue
            if len(raw_row) != len(headers):
                raise RuntimeError(
                    f"nsys stats {section} emitted a malformed row with "
                    f"{len(raw_row)} fields for {len(headers)} headers"
                )
            row = {headers[i]: raw_row[i] for i in range(len(headers))}
            name = row.get("Name")
            if not isinstance(name, str) or not name.strip():
                raise RuntimeError(f"nsys stats {section} emitted a row without a kernel/API name")
            if _parse_time_percentage(row) is None:
                raise RuntimeError(
                    f"nsys stats {section} emitted an invalid or ambiguous Time (%) value"
                )
            rows.append(row)
        if headers is None:
            raise RuntimeError(
                f"nsys stats {section} did not emit the recognized Name/Time (%) CSV schema"
            )
        return rows

    @staticmethod
    def _filter_and_rank_kernels(
        kernel_rows: list[dict[str, str]],
        kernel_regex: str | None,
        top_k: int,
    ) -> list[dict[str, str]]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        rows: Iterable[dict[str, str]] = kernel_rows
        if kernel_regex:
            pattern = re.compile(kernel_regex)
            rows = (row for row in kernel_rows if pattern.search(row.get("Name", "")))

        ranked = (
            (row, value)
            for row in rows
            if isinstance(row.get("Name"), str)
            and row["Name"].strip()
            and (value := _parse_time_percentage(row)) is not None
        )
        return [
            row
            for row, _value in heapq.nlargest(
                top_k, ranked, key=lambda item: item[1]
            )
        ]

    @staticmethod
    def _print_nsys_summary(summary: dict[str, Any], kernel_regex: str | None) -> None:
        print(f"\n=== Nsight Systems Summary: {summary['report']} ===")
        kernels = summary["kernels"]
        if kernel_regex:
            print(f"Filter: {kernel_regex}")
        if not kernels:
            print("No kernel entries found.")
            return
        for idx, row in enumerate(kernels, 1):
            name = row.get("Name", "Unknown")
            pct = row.get("Time (%)") or row.get("Time (%) [sum]", "0")
            ns = (
                row.get("Total Time (ns)")
                or row.get("Time (ns)")
                or row.get("Total Time (ns) [sum]")
                or "0"
            )
            try:
                ms = float(ns.replace('"', "")) / 1e6
            except ValueError:
                ms = 0.0
            print(f"{idx:>2}. {name}  Time: {ms:.3f} ms  Share: {pct}")
