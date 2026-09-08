"""Timing text regressions; GPU qualification is separate from parsing."""

import pytest

from core.benchmark.timing_parser import parse_kernel_time_ms


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("TIME_MS: 8.84707164764404252e-01", 0.8847071647644043),
        ("TIME_MS: 6.61862421035766646e-01", 0.6618624210357666),
        ("TIME_MS: 5.12830734252929688e+00", 5.128307342529297),
        ("TIME_MS: 1.25E+02", 125.0),
        ("TIME_MS: .25", 0.25),
        ("kernel: 7.5e-02 ms", 0.075),
        ("kernel: 7.5e-02ms", 0.075),
        ("kernel: 5e+02 us", 0.5),
        ("kernel: 5e+02 μs", 0.5),
        ("kernel: 1.2e-6 s", 0.0012),
        ("TIME_MS: 12", 12.0),
        ("kernel: 2.3074 ms", 2.3074),
        ("Warmup: 10 ms\nTIME_MS: 8e-2", 0.08),
        ("TIME_MS: 10\nResult: 8e-2 ms", 0.08),
    ],
)
def test_complete_timing_token_and_units(text: str, expected: float) -> None:
    assert parse_kernel_time_ms(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    ["TIME_MS: 3.2e-foo", "3.2e-foo ms", "-1.2 ms", "TIME_MS: 1e309", "1e308 s"],
)
def test_invalid_token_is_not_truncated_into_a_valid_timing(text: str) -> None:
    assert parse_kernel_time_ms(text) is None


def test_explicit_custom_format_still_works() -> None:
    assert parse_kernel_time_ms("duration=2.5e-1", r"duration=([\deE.+-]+)") == 0.25
