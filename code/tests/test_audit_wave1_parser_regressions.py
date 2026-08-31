"""File-backed parser fixtures; these tests do not qualify real GPU hardware."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from core import perf_core_base
from core.analysis.distributed_analysis import ClusterDiscovery


@pytest.fixture
def nvidia_smi_fixture(monkeypatch, tmp_path):
    """Run a real subprocess that serves only the declared text fixtures."""
    responses_path = tmp_path / "responses.json"
    executable = tmp_path / "nvidia-smi"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"responses = json.loads(open({str(responses_path)!r}).read())\n"
        "key = ' '.join(sys.argv[1:])\n"
        "if key not in responses:\n"
        "    print('unexpected fixture command: ' + key, file=sys.stderr)\n"
        "    raise SystemExit(2)\n"
        "result = responses[key]\n"
        "print(result.get('stdout', ''), end='')\n"
        "print(result.get('stderr', ''), end='', file=sys.stderr)\n"
        "raise SystemExit(result.get('returncode', 0))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    # No real GPU, cluster, or InfiniBand command can be reached by these tests.
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)

    def configure(responses):
        responses_path.write_text(json.dumps(responses), encoding="utf-8")

    return configure


def test_nvlink_status_preserves_gpu_link_ids_and_fractional_rates(nvidia_smi_fixture, tmp_path):
    nvidia_smi_fixture({"nvlink --status": {"stdout": (
        "GPU 2: Fixture GPU\n"
        "    Link 0: 26.562 GB/s\n"
        "    Link 1: <inactive>\n"
        "    Link 4: 50 GB/s\n"
        "GPU 10: Fixture GPU\n"
        "    Link 3: 26.562 GB/s\n"
    )}})
    result = perf_core_base.PerformanceCoreBase(bench_root=tmp_path).get_nvlink_status()
    assert result["available"] is True
    assert result["links_per_gpu"] == {2: 2, 10: 1}
    assert result["link_details"] == [
        {"gpu": 2, "link": 0, "bandwidth_gbs": 26.562},
        {"gpu": 2, "link": 4, "bandwidth_gbs": 50.0},
        {"gpu": 10, "link": 3, "bandwidth_gbs": 26.562},
    ]
    assert result["total_bandwidth_gbs"] == pytest.approx(103.124)


def test_nvlink_inactive_links_do_not_report_available(nvidia_smi_fixture, tmp_path):
    nvidia_smi_fixture({"nvlink --status": {"stdout": "GPU 0: Fixture\nLink 0: <inactive>\n"}})
    result = perf_core_base.PerformanceCoreBase(bench_root=tmp_path).get_nvlink_status()
    assert result["available"] is False
    assert result["links_per_gpu"] == {0: 0}
    assert result["total_bandwidth_gbs"] == 0


def test_nvlink_unknown_rate_is_not_invented(nvidia_smi_fixture, tmp_path):
    nvidia_smi_fixture({"nvlink --status": {"stdout": "GPU 0: Fixture\nLink 0: Active\n"}})
    result = perf_core_base.PerformanceCoreBase(bench_root=tmp_path).get_nvlink_status()
    assert result["available"] is True
    assert result["links_per_gpu"] == {0: 1}
    assert result["total_bandwidth_gbs"] is None
    assert result["link_details"] == [{"gpu": 0, "link": 0, "bandwidth_gbs": None}]
    assert any("bandwidth" in warning for warning in result["warnings"])


def test_nvlink_malformed_gpu_header_does_not_reuse_previous_gpu(nvidia_smi_fixture, tmp_path):
    nvidia_smi_fixture({"nvlink --status": {"stdout": (
        "GPU 2: Fixture\nLink 0: 25 GB/s\nGPU invalid: Fixture\nLink 1: 50 GB/s\n"
    )}})
    result = perf_core_base.PerformanceCoreBase(bench_root=tmp_path).get_nvlink_status()
    assert result["links_per_gpu"] == {2: 1}
    assert result["total_bandwidth_gbs"] == 25
    assert result["warnings"]


def test_nvlink_query_failure_remains_visible(nvidia_smi_fixture, tmp_path):
    nvidia_smi_fixture({"nvlink --status": {"returncode": 1, "stderr": "fixture query failed"}})
    result = perf_core_base.PerformanceCoreBase(bench_root=tmp_path).get_nvlink_status()
    assert result["available"] is False
    assert any("fixture query failed" in warning for warning in result["warnings"])


def test_topology_returns_peer_gpu_ids_not_local_link_indexes(nvidia_smi_fixture):
    nvidia_smi_fixture({"topo -m": {"stdout": (
        "       GPU2 NIC0 GPU10 GPU17 CPU Affinity NUMA Affinity\n"
        "GPU2   X    PIX  NV2   SYS   0-31 0\n"
        "GPU10  NV2  PIX  X     PHB   0-31 0\n"
        "GPU17  SYS  PIX  PHB   X     32-63 1\n"
        "NIC0   PIX  X    PIX   PIX\n"
        "Legend:\n  NV# = Connection traversing a bonded set of # NVLinks\n"
    )}})
    assert ClusterDiscovery().discover_nvlink_topology() == {2: [10], 10: [2], 17: []}


def test_full_topology_does_not_infer_nvlink_from_gpu_headers(nvidia_smi_fixture):
    nvidia_smi_fixture({
        "topo -m": {"stdout": "GPU2 GPU10 CPU Affinity\nGPU2 X PHB 0-31\nGPU10 PHB X 0-31\n"},
        "--query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits": {
            "stdout": "2, Fixture GPU 2, 1024, 4, 16\n10, Fixture GPU 10, 1024, 4, 16\n"
        },
    })
    result = ClusterDiscovery().get_full_topology()
    assert result["nvlink_topology"] == {2: [], 10: []}
    assert result["interconnect"] not in {"nvlink", "nvswitch"}


def test_eight_gpu_nvlink_matrix_does_not_prove_nvswitch(nvidia_smi_fixture):
    matrix = " ".join(f"GPU{i}" for i in range(8)) + " CPU Affinity\n"
    for row in range(8):
        matrix += f"GPU{row} " + " ".join("X" if row == col else "NV18" for col in range(8)) + " 0-31\n"
    nvidia_smi_fixture({
        "topo -m": {"stdout": matrix},
        "--query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits": {
            "stdout": "".join(f"{i}, Fixture GPU {i}, 1024, 4, 16\n" for i in range(8))
        },
    })
    result = ClusterDiscovery().get_full_topology()
    assert result["gpus_per_node"] == 8
    assert result["interconnect"] == "nvlink"
    assert result["nvlink_topology"][0] == [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize("output", [
    "GPU2 GPU10 CPU Affinity\nGPU2 X NV2 0-31\n",
    "GPU 0: Fixture\nLink 0: Active\n",
    "GPU2 GPU10 CPU Affinity\nGPU2 X NVX 0-31\nGPU10 NVX X 0-31\n",
])
def test_topology_rejects_incomplete_or_status_only_output(nvidia_smi_fixture, capsys, output):
    nvidia_smi_fixture({"topo -m": {"stdout": output}})
    assert ClusterDiscovery().discover_nvlink_topology() == {}
    assert "NVLink topology" in capsys.readouterr().err


def test_cutlass_version_reads_only_macro_definitions(monkeypatch, tmp_path):
    include_dir = tmp_path / "third_party" / "cutlass" / "include" / "cutlass"
    include_dir.mkdir(parents=True)
    (include_dir / "version.h").write_text(
        "// Copyright 2026. CUTLASS_VERSION_MAJOR should not be read here.\n"
        "#define CUTLASS_VERSION_MAJOR 4\n"
        "#define CUTLASS_VERSION_MINOR 12\n"
        "#define CUTLASS_VERSION_PATCH 3\n"
        "#define CUTLASS_VERSION (CUTLASS_VERSION_MAJOR * 100 + CUTLASS_VERSION_MINOR)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(perf_core_base, "CODE_ROOT", tmp_path)
    result = perf_core_base.PerformanceCoreBase(bench_root=tmp_path).get_dependency_health()
    assert result["cutlass"]["version"] == "4.12.3"
    assert not any("Failed to parse CUTLASS version" in warning for warning in result["warnings"])


@pytest.mark.parametrize("header", [
    "#define CUTLASS_VERSION_MAJOR 4\n#define CUTLASS_VERSION_MINOR 1\n",
    "#define CUTLASS_VERSION_MAJOR invalid\n#define CUTLASS_VERSION_MINOR 1\n#define CUTLASS_VERSION_PATCH 0\n",
])
def test_cutlass_incomplete_or_invalid_versions_remain_unknown(monkeypatch, tmp_path, header):
    include_dir = tmp_path / "third_party" / "cutlass" / "include" / "cutlass"
    include_dir.mkdir(parents=True)
    (include_dir / "version.h").write_text(header, encoding="utf-8")
    monkeypatch.setattr(perf_core_base, "CODE_ROOT", tmp_path)
    result = perf_core_base.PerformanceCoreBase(bench_root=tmp_path).get_dependency_health()
    assert result["cutlass"]["version"] is None
    assert any("Failed to parse CUTLASS version" in warning for warning in result["warnings"])
