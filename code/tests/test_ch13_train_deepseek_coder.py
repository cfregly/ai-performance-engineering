from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ch13.train_deepseek_coder import (
    _build_top_op_summary,
    _external_torch_profiler_active,
    _format_top_ops_report,
)


class _FakeEvent:
    def __init__(
        self,
        key: str,
        *,
        count: int,
        self_device_time_total: float = 0.0,
        self_cpu_time_total: float = 0.0,
        device_time_total: float = 0.0,
        cpu_time_total: float = 0.0,
    ) -> None:
        self.key = key
        self.count = count
        self.self_device_time_total = self_device_time_total
        self.self_cpu_time_total = self_cpu_time_total
        self.device_time_total = device_time_total
        self.cpu_time_total = cpu_time_total


class _FakeProfiler:
    def __init__(self, events: list[_FakeEvent]) -> None:
        self._events = events

    def key_averages(self) -> list[_FakeEvent]:
        return self._events


def test_build_top_op_summary_prefers_device_times() -> None:
    prof = _FakeProfiler(
        [
            _FakeEvent(
                "aten::linear",
                count=2,
                self_device_time_total=25_000.0,
                self_cpu_time_total=100.0,
                device_time_total=30_000.0,
                cpu_time_total=200.0,
            ),
            _FakeEvent(
                "aten::gelu",
                count=1,
                self_device_time_total=10_000.0,
                self_cpu_time_total=500.0,
                device_time_total=12_000.0,
                cpu_time_total=600.0,
            ),
        ]
    )

    summary = _build_top_op_summary(prof, row_limit=10)

    assert summary["device_times_available"] is True
    assert summary["metric_key"] == "self_device_time_total_us"
    assert summary["top_ops"][0]["name"] == "aten::linear"
    report = _format_top_ops_report(summary)
    assert "Top operations by self CUDA/device time" in report
    assert "aten::linear" in report


def test_build_top_op_summary_ignores_profiler_step_rows() -> None:
    prof = _FakeProfiler(
        [
            _FakeEvent(
                "ProfilerStep*",
                count=1,
                self_device_time_total=999_000.0,
                device_time_total=999_000.0,
            ),
            _FakeEvent(
                "aten::linear",
                count=2,
                self_device_time_total=25_000.0,
                device_time_total=30_000.0,
            ),
        ]
    )

    summary = _build_top_op_summary(prof, row_limit=10)

    assert summary["top_ops"][0]["name"] == "aten::linear"
    assert all(row["name"] != "ProfilerStep*" for row in summary["top_ops"])


def test_build_top_op_summary_selects_top_rows_without_full_sort() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "ch13" / "train_deepseek_coder.py"
    ).read_text(encoding="utf-8")
    summary_section = source.split("def _build_top_op_summary", maxsplit=1)[1].split(
        "def _format_top_ops_report",
        maxsplit=1,
    )[0]

    assert "import heapq" in source
    assert "top_ops = heapq.nlargest(row_limit, events, key=_sort_value)" in summary_section
    assert "sorted(events, key=_sort_value" not in summary_section


def test_build_top_op_summary_fails_when_device_times_are_missing() -> None:
    prof = _FakeProfiler(
        [
            _FakeEvent(
                "aten::linear",
                count=2,
                self_device_time_total=0.0,
                self_cpu_time_total=25_000.0,
            ),
        ]
    )

    with pytest.raises(RuntimeError, match="did not expose per-op self device totals"):
        _build_top_op_summary(prof, row_limit=10)


def test_external_torch_profiler_active_reflects_profiler_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.autograd, "_profiler_enabled", lambda: True)
    assert _external_torch_profiler_active() is True

    monkeypatch.setattr(torch.autograd, "_profiler_enabled", lambda: False)
    assert _external_torch_profiler_active() is False
