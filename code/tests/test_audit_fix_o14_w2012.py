"""Regression tests for W2-012: published ch07 lookup results must fit the current kernel.

The ch07 README and its generator published `0.009 ms` (`45.41x`) for `lookup`. That time
came from a removed kernel that read one precomputed sum per output. The current optimized
kernel reads all 64 path values per output, so a B200 cannot finish a launch faster than
its HBM bandwidth allows.
"""

from __future__ import annotations

import json
import re

import pytest

from ch07.optimized_lookup import get_benchmark as get_optimized_lookup
from core.analysis.gpu_bottleneck_analyzer import GPU_SPECS
from core.benchmark.metrics import BLACKWELL_B200
from core.scripts.refresh_readmes import ENTRIES, REPO_ROOT, _format_markdown, check_readmes

CH07_DIR = REPO_ROOT / "ch07"
_MS_CELL = re.compile(r"^`([0-9]+(?:\.[0-9]+)?) ms`$")
_SPEEDUP_CELL = re.compile(r"`[0-9]+(?:\.[0-9]+)?x`")


def _lookup_bytes_per_launch() -> float:
    metadata = get_optimized_lookup().get_workload_metadata()
    assert metadata is not None
    assert metadata.bytes_per_iteration is not None
    return float(metadata.bytes_per_iteration)


def _b200_hbm_floor_ms(bytes_per_launch: float) -> float:
    return bytes_per_launch / (BLACKWELL_B200.hbm_bandwidth_gbps * 1e9) * 1e3


def _lookup_row(markdown: str) -> list[str]:
    rows = [line for line in markdown.splitlines() if line.startswith("| `lookup` |")]
    assert len(rows) == 1, rows
    return [cell.strip() for cell in rows[0].strip().strip("|").split("|")]


def _assert_lookup_row_respects_floor(markdown: str) -> None:
    floor_ms = _b200_hbm_floor_ms(_lookup_bytes_per_launch())
    _target, _baseline, optimized, delta, _what_changed = _lookup_row(markdown)
    match = _MS_CELL.match(optimized)
    if match is None:
        assert optimized == "pending B200 rerun"
        assert not _SPEEDUP_CELL.search(delta), delta
        return
    assert float(match.group(1)) >= floor_ms, (
        f"lookup optimized time {optimized} is below the B200 HBM floor of {floor_ms:.4f} ms"
    )


def test_lookup_bytes_match_the_current_kernel_shape() -> None:
    source = (CH07_DIR / "optimized_lookup.cu").read_text(encoding="utf-8")
    assert "constexpr int N = 1 << 20;" in source
    assert "constexpr int RANDOM_STEPS = 64;" in source
    assert "for (int step = 0; step < RANDOM_STEPS; ++step) {" in source
    assert "val += paths[step * n + idx];" in source
    # 64 path reads plus one output write per element, in float32.
    assert _lookup_bytes_per_launch() == float((1 << 20) * (64 + 1) * 4)


def test_lookup_working_set_exceeds_b200_l2() -> None:
    l2_bytes = GPU_SPECS["B200"].l2_cache_mb * 2**20
    assert _lookup_bytes_per_launch() > l2_bytes


def test_generated_ch07_lookup_row_respects_b200_hbm_floor() -> None:
    _assert_lookup_row_respects_floor(_format_markdown(ENTRIES["ch07"]))


def test_ch07_readme_on_disk_respects_b200_hbm_floor_and_matches_generator() -> None:
    readme = (CH07_DIR / "README.md").read_text(encoding="utf-8")
    _assert_lookup_row_respects_floor(readme)
    assert check_readmes(targets=["ch07"]) == []


def test_ch07_readme_states_the_floor_computed_from_the_benchmark() -> None:
    bytes_per_launch = _lookup_bytes_per_launch()
    markdown = _format_markdown(ENTRIES["ch07"])
    assert f"{bytes_per_launch / 2**20:.0f} MiB" in markdown
    assert f"`{_b200_hbm_floor_ms(bytes_per_launch):.3f} ms`" in markdown
    assert f"{BLACKWELL_B200.hbm_bandwidth_gbps / 1000:.0f} TB/s" in markdown
    assert "shared-memory staging for lookup" not in markdown


def test_b200_lookup_expectations_respect_hbm_floor() -> None:
    floor_ms = _b200_hbm_floor_ms(_lookup_bytes_per_launch())
    peak_bytes_per_s = BLACKWELL_B200.hbm_bandwidth_gbps * 1e9
    payload = json.loads((CH07_DIR / "expectations_b200.json").read_text(encoding="utf-8"))
    # A fresh B200 recording may re-add lookup_cuda, but only above the floor.
    for key, entry in payload["examples"].items():
        if entry.get("example") != "lookup":
            continue
        metrics = entry["metrics"]
        assert metrics["best_optimized_time_ms"] >= floor_ms, key
        assert metrics["best_optimized_throughput.bytes_per_s"] <= peak_bytes_per_s, key


def test_floor_rejects_the_withdrawn_result() -> None:
    stale = "| `lookup` | `0.397 ms` | `0.009 ms` | `45.41x` | locality-aware lookup path |"
    with pytest.raises(AssertionError, match="below the B200 HBM floor"):
        _assert_lookup_row_respects_floor(stale)
