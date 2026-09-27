"""Control-plane tests for the pinned SGLang runtime overlay."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from labs.serving_comparison import prepare_sglang_runtime_tool as runtime_tool

COLLECTOR_FIXTURE = b"""class SchedulerMetricsCollector:
    def __init__(self, labels, Counter):
        self.labels = labels
        self.num_bootstrap_failed_reqs = Counter(
            name="sglang:num_bootstrap_failed_reqs_total",
            documentation="The number of bootstrap failed requests.",
            labelnames=labels.keys(),
        )
        self.num_transfer_failed_reqs = Counter(
            name="sglang:num_transfer_failed_reqs_total",
            documentation="The number of transfer failed requests.",
            labelnames=labels.keys(),
        )
        self.num_prefill_retries_total = Counter(
            name="sglang:num_prefill_retries_total",
            documentation="Total number of prefill retries.",
            labelnames=labels.keys(),
        )

    def increment_bootstrap_failed_reqs(self):
        self.num_bootstrap_failed_reqs.labels(**self.labels).inc(1)

    def increment_transfer_failed_reqs(self):
        self.num_transfer_failed_reqs.labels(**self.labels).inc(1)
"""


def _write_package(
    root: Path,
    *,
    collector: bytes = COLLECTOR_FIXTURE,
    version: str = "0.5.20",
) -> Path:
    package = root / "sglang"
    collector_path = package / runtime_tool.COLLECTOR_RELATIVE_PATH
    collector_path.parent.mkdir(parents=True)
    collector_path.write_bytes(collector)
    (package / runtime_tool.VERSION_RELATIVE_PATH).write_text(
        f"__version__ = version = {version!r}\nraise RuntimeError('must not import')\n",
        encoding="utf-8",
    )
    (package / "untouched.py").write_text("VALUE = 7\n", encoding="utf-8")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "untouched.cpython-310.pyc").write_bytes(b"cache bytes")
    (package / "loose.pyc").write_bytes(b"loose cache bytes")
    return package


def _authorize_fixture(monkeypatch: pytest.MonkeyPatch, collector: bytes) -> None:
    monkeypatch.setattr(
        runtime_tool,
        "AUTHORIZED_COLLECTOR_SHA256",
        hashlib.sha256(collector).hexdigest(),
    )


def test_prepare_runtime_overlay_changes_only_collector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _write_package(tmp_path)
    original_collector = (package / runtime_tool.COLLECTOR_RELATIVE_PATH).read_bytes()
    _authorize_fixture(monkeypatch, original_collector)
    output = tmp_path / "runtime-overlay"

    receipt = runtime_tool.prepare_runtime_overlay(package, output)

    output_package = output / "sglang"
    patched_collector = (output_package / runtime_tool.COLLECTOR_RELATIVE_PATH).read_bytes()
    expected_collector = original_collector.replace(
        runtime_tool._COUNTER_DEFINITIONS,
        runtime_tool._COUNTER_DEFINITIONS + runtime_tool._COUNTER_INITIALIZATION,
        1,
    )
    assert patched_collector == expected_collector
    assert (package / runtime_tool.COLLECTOR_RELATIVE_PATH).read_bytes() == original_collector
    assert (
        runtime_tool._COUNTER_DEFINITIONS + runtime_tool._COUNTER_INITIALIZATION
        in patched_collector
    )
    assert b".inc(0)" not in patched_collector
    assert patched_collector.count(b"increment_bootstrap_failed_reqs") == 1
    assert patched_collector.count(b"increment_transfer_failed_reqs") == 1

    assert (output_package / "untouched.py").read_text(encoding="utf-8") == "VALUE = 7\n"
    assert not (output_package / "__pycache__").exists()
    assert not (output_package / "loose.pyc").exists()
    assert (package / "__pycache__/untouched.cpython-310.pyc").exists()
    assert (package / "loose.pyc").exists()

    collector_key = runtime_tool.COLLECTOR_RELATIVE_PATH.as_posix()
    assert receipt["schema_version"] == "serving-comparison.sglang-runtime-receipt.v1"
    assert receipt["sglang_version"] == "0.5.20"
    assert receipt["patch_identity"] == runtime_tool.PATCH_IDENTITY
    assert receipt["collector_relative_path"] == collector_key
    assert receipt["changed_files"] == [collector_key]
    assert receipt["runtime_build_id"] == receipt["output_package_digest"]
    assert receipt["original_package_digest"] != receipt["output_package_digest"]
    assert receipt["collector_original_sha256"] == (
        "sha256:" + hashlib.sha256(original_collector).hexdigest()
    )
    assert receipt["collector_patched_sha256"] == (
        "sha256:" + hashlib.sha256(patched_collector).hexdigest()
    )
    manifest_paths = [item["path"] for item in receipt["relative_file_manifest"]]
    assert manifest_paths == [
        "_version.py",
        "srt/observability/metrics_collector.py",
        "untouched.py",
    ]
    saved_receipt = json.loads((output / "runtime-receipt.json").read_text(encoding="utf-8"))
    assert saved_receipt == receipt


def test_cli_prints_bounded_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _write_package(tmp_path)
    collector = (package / runtime_tool.COLLECTOR_RELATIVE_PATH).read_bytes()
    _authorize_fixture(monkeypatch, collector)
    output = tmp_path / "runtime-overlay"

    assert runtime_tool.main(["--package-dir", str(package), "--output-dir", str(output)]) == 0

    summary = json.loads(capsys.readouterr().out)
    receipt = json.loads((output / runtime_tool.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert summary == {
        "changed_files": [runtime_tool.COLLECTOR_RELATIVE_PATH.as_posix()],
        "receipt": str(output / runtime_tool.RECEIPT_NAME),
        "runtime_build_id": receipt["runtime_build_id"],
    }


def test_unknown_collector_bytes_are_rejected_before_output(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    output = tmp_path / "runtime-overlay"

    with pytest.raises(runtime_tool.PreparationError, match="not the authorized"):
        runtime_tool.prepare_runtime_overlay(package, output)

    assert not output.exists()


def test_wrong_sglang_version_is_rejected_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _write_package(tmp_path, version="0.5.21")
    collector = (package / runtime_tool.COLLECTOR_RELATIVE_PATH).read_bytes()
    _authorize_fixture(monkeypatch, collector)
    output = tmp_path / "runtime-overlay"

    with pytest.raises(runtime_tool.PreparationError, match="exactly as 0.5.20"):
        runtime_tool.prepare_runtime_overlay(package, output)

    assert not output.exists()


def test_existing_output_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _write_package(tmp_path)
    collector = (package / runtime_tool.COLLECTOR_RELATIVE_PATH).read_bytes()
    _authorize_fixture(monkeypatch, collector)
    output = tmp_path / "runtime-overlay"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")

    with pytest.raises(runtime_tool.PreparationError, match="already exists"):
        runtime_tool.prepare_runtime_overlay(package, output)

    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_nested_output_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _write_package(tmp_path)
    collector = (package / runtime_tool.COLLECTOR_RELATIVE_PATH).read_bytes()
    _authorize_fixture(monkeypatch, collector)
    output = package / "runtime-overlay"

    with pytest.raises(runtime_tool.PreparationError, match="must not be nested"):
        runtime_tool.prepare_runtime_overlay(package, output)

    assert not output.exists()


def test_source_tree_symlink_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _write_package(tmp_path)
    collector = (package / runtime_tool.COLLECTOR_RELATIVE_PATH).read_bytes()
    _authorize_fixture(monkeypatch, collector)
    external = tmp_path / "external.py"
    external.write_text("VALUE = 9\n", encoding="utf-8")
    (package / "linked.py").symlink_to(external)
    output = tmp_path / "runtime-overlay"

    with pytest.raises(runtime_tool.PreparationError, match="contains a symlink"):
        runtime_tool.prepare_runtime_overlay(package, output)

    assert not output.exists()


def test_nonunique_patch_anchor_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ambiguous_collector = COLLECTOR_FIXTURE + runtime_tool._COUNTER_DEFINITIONS
    package = _write_package(tmp_path, collector=ambiguous_collector)
    _authorize_fixture(monkeypatch, ambiguous_collector)
    output = tmp_path / "runtime-overlay"

    with pytest.raises(runtime_tool.PreparationError, match="unique patch anchor"):
        runtime_tool.prepare_runtime_overlay(package, output)

    assert not output.exists()


def test_post_copy_failure_retains_output_without_success_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _write_package(tmp_path)
    collector_path = package / runtime_tool.COLLECTOR_RELATIVE_PATH
    original_collector = collector_path.read_bytes()
    _authorize_fixture(monkeypatch, original_collector)
    output = tmp_path / "runtime-overlay"
    real_scan = runtime_tool._scan_package
    scan_count = 0

    def changed_source_scan(path: Path) -> list[dict[str, object]]:
        nonlocal scan_count
        scan_count += 1
        manifest = real_scan(path)
        if scan_count == 2:
            manifest = [dict(item) for item in manifest]
            manifest[0]["sha256"] = "sha256:" + "f" * 64
        return manifest

    monkeypatch.setattr(runtime_tool, "_scan_package", changed_source_scan)

    with pytest.raises(runtime_tool.PreparationError, match="source package changed"):
        runtime_tool.prepare_runtime_overlay(package, output)

    assert output.is_dir()
    assert (output / "sglang").is_dir()
    assert not (output / runtime_tool.RECEIPT_NAME).exists()
    assert not (output / f".{runtime_tool.RECEIPT_NAME}.tmp").exists()
    assert collector_path.read_bytes() == original_collector


def test_labeled_counter_initialization_exports_zero_then_positive_in_multiprocess_mode(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("prometheus_client") is None:
        pytest.skip("prometheus_client is not installed in this test environment")
    multiprocess_dir = tmp_path / "prometheus-multiprocess"
    multiprocess_dir.mkdir()
    script = r"""
import json
from prometheus_client import CollectorRegistry, Counter, generate_latest, multiprocess

labels = {"model_name": "fixture", "engine_type": "prefill"}
bootstrap = Counter(
    name="sglang:num_bootstrap_failed_reqs_total",
    documentation="bootstrap failures",
    labelnames=labels.keys(),
)
transfer = Counter(
    name="sglang:num_transfer_failed_reqs_total",
    documentation="transfer failures",
    labelnames=labels.keys(),
)
bootstrap.labels(**labels)
transfer.labels(**labels)
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)
before = generate_latest(registry).decode("utf-8")
bootstrap.labels(**labels).inc(1)
transfer.labels(**labels).inc(1)
after = generate_latest(registry).decode("utf-8")
print(json.dumps({"before": before, "after": after}))
"""
    environment = os.environ.copy()
    environment["PROMETHEUS_MULTIPROC_DIR"] = str(multiprocess_dir)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    payload = json.loads(completed.stdout)

    def metric_value(text: str, name: str) -> float:
        matches = [
            line
            for line in text.splitlines()
            if line.startswith(name + "{") or line.startswith(name + " ")
        ]
        assert len(matches) == 1
        return float(matches[0].split()[-1])

    for metric_name in (
        "sglang:num_bootstrap_failed_reqs_total",
        "sglang:num_transfer_failed_reqs_total",
    ):
        assert f"# TYPE {metric_name} counter" in payload["before"]
        assert metric_value(payload["before"], metric_name) == 0
        assert metric_value(payload["after"], metric_name) == 1
