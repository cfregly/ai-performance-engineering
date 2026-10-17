"""Follow-up regressions for the platform-detection audit fixes.

Covers W2-004, W2-006, W2-040, W2-078, W2-098 and W2-120 after the Grace and
NVLink-C2C helpers moved into core.common.grace_platform. The NVIDIA SoC ID in
sysfs answers whether the host CPU is Grace. The CUDA driver answers whether a
GPU reaches host memory over NVLink-C2C, and nvidia-smi c2c -s must agree with
it wherever a tool lists the links. These tests run the real detection,
labeling and reporting code. They replace only host seams: the machine name,
the sysfs SoC tree, /proc/cpuinfo, subprocess.run, the CUDA driver attribute
query and the CUDA device query.
"""

from __future__ import annotations

import ast
import builtins
import io
import json
import os
import shutil
import subprocess
import sys
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ch02.baseline_grace_coherent_memory as baseline_grace
import ch02.cpu_gpu_topology_aware as topo
import ch02.optimized_grace_coherent_memory as optimized_grace
import ch04.multi_node_blackwell as multi_node
import ch04.nccl_blackwell_config as nccl_config
import ch04.symmetric_memory_example as symmetric_memory
import core.optimization.parallelism_planner.topology_detector as topology_detector
from core.common import grace_platform
from core.harness import cuda_capabilities, hardware_capabilities, l2_cache_utils
from core.harness.arch_config import ArchitectureConfig
from core.optimization.parallelism_planner import cluster_config
from core.optimization.parallelism_planner.model_analyzer import ModelArchitecture, ModelType
from core.optimization.parallelism_planner.strategy_optimizer import (
    ParallelismStrategy,
    StrategyOptimizer,
)
from core.optimization.parallelism_planner.topology_detector import GPUInfo, TopologyInfo
from core.scripts.utilities import probe_hardware_capabilities as probe
from labs.decode_optimization import decode_common

CODE_ROOT = Path(__file__).resolve().parents[1]
NVIDIA_SOC_ID = "jep106:036b:0241"
OTHER_VENDOR_SOC_ID = "jep106:043b:0000"
# The removed ch04 checks treated any "ARM" or "Neoverse" text here as Grace.
NEOVERSE_CPUINFO = "processor\t: 0\nmodel name\t: ARM Neoverse V2\nCPU part\t: 0xd4f\n"
NVLINK_DEMO = CODE_ROOT / "ch02" / "memory_transfer_nvlink_demo.cu"


# ---------------------------------------------------------------------------
# Host seams
# ---------------------------------------------------------------------------


def _machine(monkeypatch: pytest.MonkeyPatch, machine: str) -> None:
    monkeypatch.setattr(grace_platform.platform, "machine", lambda: machine)


def _soc_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, soc_id: str) -> None:
    """Point core.common.grace_platform at a temporary /sys/bus/soc/devices tree."""
    bus = tmp_path / "soc_devices"
    (bus / "soc0").mkdir(parents=True)
    (bus / "soc0" / "soc_id").write_text(soc_id + "\n", encoding="utf-8")
    monkeypatch.setattr(grace_platform, "_SOC_BUS_DIR", bus)


def _grace_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    soc_id: str = NVIDIA_SOC_ID,
    machine: str = "aarch64",
    cpuinfo: str = NEOVERSE_CPUINFO,
) -> None:
    _soc_tree(tmp_path, monkeypatch, soc_id)
    _machine(monkeypatch, machine)
    real_open = builtins.open

    def _open(file: object, *args: object, **kwargs: object) -> object:
        if str(file) == "/proc/cpuinfo":
            return io.StringIO(cpuinfo)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open)


def _driver(monkeypatch: pytest.MonkeyPatch, c2c_by_device: dict[int, bool]) -> None:
    """Make the CUDA driver report native host atomics only for C2C devices."""

    def _attribute(device_index: int, attribute: int) -> int:
        return 1 if c2c_by_device[device_index] else 0

    monkeypatch.setattr(grace_platform, "_cuda_driver_attribute", _attribute)


def _driver_fails(device_index: int, attribute: int) -> int:
    raise RuntimeError("cuInit failed with CUresult 100")


def _refuse_driver(device_index: int, attribute: int) -> int:
    raise AssertionError(f"driver attribute {attribute} must not be queried for device {device_index}")


def _no_tools(*args: object, **kwargs: object) -> object:
    raise FileNotFoundError(args[0][0] if args and args[0] else "tool")


def _c2c_status(link_count: int) -> str:
    lines = ["GPU 0: NVIDIA GB200 (UUID: GPU-00000000-0000-0000-0000-000000000000)"]
    lines += [f"\t C2C Link {link}: 44.712 GB/s" for link in range(link_count)]
    return "\n".join(lines) + "\n"


def _smi(monkeypatch: pytest.MonkeyPatch, responses: dict[int, tuple[int, str] | None]) -> None:
    """Answer ``nvidia-smi c2c -i N -s``. Every other tool is missing."""

    def _run(cmd: list[str], *args: object, **kwargs: object) -> SimpleNamespace:
        cmd = [str(part) for part in cmd]
        if cmd[:2] == ["nvidia-smi", "c2c"]:
            response = responses[int(cmd[cmd.index("-i") + 1])]
            if response is None:
                raise FileNotFoundError(cmd[0])
            return SimpleNamespace(returncode=response[0], stdout=response[1], stderr="")
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(grace_platform.subprocess, "run", _run)


def _props(major: int, minor: int, name: str = "Test GPU") -> SimpleNamespace:
    return SimpleNamespace(
        major=major,
        minor=minor,
        name=name,
        total_memory=96 * 1024**3,
        multi_processor_count=148,
        warp_size=32,
        max_threads_per_block=1024,
        max_threads_per_multi_processor=1536,
        shared_memory_per_block=48 * 1024,
        shared_memory_per_multiprocessor=100 * 1024,
        L2_cache_size=24 * 1024 * 1024,
    )


def _fake_cuda(monkeypatch: pytest.MonkeyPatch, *, count: int, major: int, minor: int, name: str) -> None:
    props = _props(major, minor, name)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props)


def _probe_record(monkeypatch: pytest.MonkeyPatch, major: int, minor: int) -> dict:
    """Run the real _probe_device on an x86 host with the toolchain probes replaced."""
    props = _props(major, minor, f"Test GPU sm_{major}{minor}")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(set_device=lambda _index: None, get_device_properties=lambda _index: props),
        version=SimpleNamespace(cuda="13.0"),
    )
    monkeypatch.setattr(probe, "torch", fake_torch)
    _machine(monkeypatch, "x86_64")
    monkeypatch.setattr(grace_platform, "_cuda_driver_attribute", _refuse_driver)
    monkeypatch.setattr(probe, "_probe_tma_compiler_support", lambda _sm: (True, None))
    monkeypatch.setattr(probe, "_query_cluster_launch", lambda _index: 1)
    monkeypatch.setattr(probe, "_probe_dsmem_support", lambda _sm: (True, None))
    return probe._probe_device(0)


# ---------------------------------------------------------------------------
# core.common.grace_platform (OOF-047, OOF-077, OOF-094)
# ---------------------------------------------------------------------------


def test_grace_identity_reads_sysfs_only_on_arm_hosts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _soc_tree(tmp_path, monkeypatch, NVIDIA_SOC_ID)

    _machine(monkeypatch, "x86_64")
    assert grace_platform.host_grace_soc_id() is None
    _machine(monkeypatch, "aarch64")
    assert grace_platform.host_grace_soc_id() == NVIDIA_SOC_ID


def test_grace_identity_rejects_other_arm_soc_vendors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _soc_tree(tmp_path, monkeypatch, OTHER_VENDOR_SOC_ID)
    _machine(monkeypatch, "aarch64")

    assert grace_platform.host_grace_soc_id() is None


@pytest.mark.parametrize(
    ("smi", "driver_c2c", "expected"),
    [
        pytest.param((0, _c2c_status(10)), True, {link: 44.712 for link in range(10)}, id="links-up"),
        pytest.param((0, _c2c_status(0)), False, {}, id="listed-without-links"),
        pytest.param((3, "No Devices support C2C.\n"), False, {}, id="smi-fails-driver-pcie"),
        pytest.param(None, False, {}, id="smi-missing-driver-pcie"),
        pytest.param((6, ""), True, None, id="smi-fails-driver-c2c"),
    ],
)
def test_gpu_c2c_links_follow_the_driver(
    monkeypatch: pytest.MonkeyPatch,
    smi: tuple[int, str] | None,
    driver_c2c: bool,
    expected: dict[int, float] | None,
) -> None:
    _machine(monkeypatch, "aarch64")
    _driver(monkeypatch, {0: driver_c2c})
    _smi(monkeypatch, {0: smi})

    assert grace_platform.gpu_c2c_links(_props(10, 0), 0) == expected


@pytest.mark.parametrize(
    ("smi", "driver_c2c", "message"),
    [
        (
            (0, _c2c_status(10)),
            False,
            "lists 10 active NVLink-C2C links, but the CUDA driver reports no NVLink-C2C host link",
        ),
        (
            (0, _c2c_status(0)),
            True,
            "lists no active NVLink-C2C link, but the CUDA driver reports an NVLink-C2C host link",
        ),
    ],
)
def test_gpu_c2c_links_stop_when_nvidia_smi_contradicts_the_driver(
    monkeypatch: pytest.MonkeyPatch, smi: tuple[int, str], driver_c2c: bool, message: str
) -> None:
    _machine(monkeypatch, "aarch64")
    _driver(monkeypatch, {0: driver_c2c})
    _smi(monkeypatch, {0: smi})

    with pytest.raises(RuntimeError, match=message):
        grace_platform.gpu_c2c_links(_props(10, 0), 0)


def test_grace_helpers_live_in_core_and_every_call_site_imports_them() -> None:
    assert probe._host_link_flags is grace_platform.host_link_flags
    for moved in (
        "_cuda_driver_attribute",
        "CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED",
        "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES",
    ):
        assert not hasattr(probe, moved), moved
    for moved in ("_SOC_BUS_DIR", "_NVIDIA_SOC_ID_RE", "_C2C_LINK_RE", "query_gpu_c2c_links"):
        assert not hasattr(topo, moved), moved
    assert topo.read_soc_ids is grace_platform.read_soc_ids
    assert topo.nvidia_grace_soc_id is grace_platform.nvidia_grace_soc_id
    assert topo.gpu_c2c_links is grace_platform.gpu_c2c_links

    gb200_source = (CODE_ROOT / "ch04" / "gb200_grace_numa_optimization.py").read_text(encoding="utf-8")
    assert "from ch02" not in gb200_source
    for relpath in (
        "ch04/multi_node_blackwell.py",
        "ch04/nccl_blackwell_config.py",
        "ch04/symmetric_memory_example.py",
        "core/optimization/parallelism_planner/topology_detector.py",
    ):
        text = (CODE_ROOT / relpath).read_text(encoding="utf-8")
        assert "from core.common.grace_platform import" in text, relpath
        for substring_check in ("'ARM' in cpuinfo", '"ARM" in cpuinfo', '"grace" in cpuinfo'):
            assert substring_check not in text, relpath


# ---------------------------------------------------------------------------
# Hardware probe and arch_detection header (OOF-027, OOF-047, OOF-125..127)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="SKIPPED: on a CUDA host the probe would rewrite artifacts/hardware_capabilities.json",
)
def test_probe_script_still_runs_as_a_standalone_script(tmp_path: Path) -> None:
    # The harness and setup.sh run the probe as a script outside the repo package.
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [
            sys.executable,
            str(CODE_ROOT / "core" / "scripts" / "utilities" / "probe_hardware_capabilities.py"),
            "--allow-unavailable",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("SKIPPED: ")


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((9, 0), 8), ((10, 0), 8), ((12, 0), 8), ((8, 0), 1)],
)
def test_probe_reports_the_portable_cluster_limit(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], expected: int
) -> None:
    entry = _probe_record(monkeypatch, *capability)

    assert entry["cluster"]["max_cluster_size"] == expected


def _hopper_fallback(text: str) -> str:
    return text.split("} else if (props.major == 9) {", 1)[1].split("} else if (props.major >= 8) {", 1)[0]


def test_hopper_fallback_uses_the_portable_cluster_limit_in_header_and_generator() -> None:
    header = (CODE_ROOT / "core" / "common" / "headers" / "arch_detection.cuh").read_text(encoding="utf-8")
    generator = (
        (CODE_ROOT / "core" / "scripts" / "utilities" / "generate_arch_detection_header.py")
        .read_text(encoding="utf-8")
        .replace("{{", "{")
        .replace("}}", "}")
    )

    assert "cached.max_cluster_size = 8;" in _hopper_fallback(header)
    assert _hopper_fallback(header) == _hopper_fallback(generator)
    for text in (header, generator):
        assert "cached.max_cluster_size = 4;" not in text


# ---------------------------------------------------------------------------
# SM 12.x labels (OOF-026, OOF-029, OOF-030, OOF-091, OOF-093, OOF-095, OOF-096)
# ---------------------------------------------------------------------------


def _verifier_labels() -> tuple[str, dict]:
    """Load the verifier's label table without importing triton or needing CUDA."""
    path = CODE_ROOT / "core" / "verification" / "verify_triton_blackwell_features.py"
    source = path.read_text(encoding="utf-8")
    wanted = {"_BLACKWELL_FAMILIES", "_HBM3E_ARCHITECTURES", "_blackwell_family"}
    nodes = [
        node
        for node in ast.parse(source).body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in wanted for target in node.targets)
        )
    ]
    namespace: dict = {"Tuple": typing.Tuple}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return source, namespace


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (12, 0), (12, 1), (11, 0), (9, 0), (13, 0)])
def test_triton_verifier_labels_match_arch_config(capability: tuple[int, int]) -> None:
    source, namespace = _verifier_labels()
    metadata = ArchitectureConfig.metadata_for_capability(capability)

    assert namespace["_blackwell_family"](*capability) == (metadata["architecture"], metadata["family"])
    assert namespace["_HBM3E_ARCHITECTURES"] == frozenset({"blackwell", "blackwell_ultra"})
    assert "props.major >= 12" not in source
    assert "is_grace_blackwell" not in source


@pytest.mark.parametrize(
    ("capability", "supported"),
    [((12, 0), True), ((12, 1), True), ((10, 0), True), ((10, 3), True), ((9, 0), False)],
)
def test_blackwell_tma_gate_accepts_every_probe_blackwell_label(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], supported: bool
) -> None:
    cap = hardware_capabilities._entry_to_capabilities(_probe_record(monkeypatch, *capability))
    monkeypatch.setattr(cuda_capabilities, "detect_capabilities", lambda: cap)

    ok, reason = cuda_capabilities.blackwell_tma_support_status()

    assert ok is supported, reason
    if not supported:
        assert reason == "requires a Blackwell-family GPU (SM 10.x, 11.x or 12.x), found hopper"


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((12, 0), True), ((12, 1), True), ((10, 3), True), ((9, 0), False)],
)
def test_fp4_decode_treats_every_probe_blackwell_label_as_blackwell(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], expected: bool
) -> None:
    cap = hardware_capabilities._entry_to_capabilities(_probe_record(monkeypatch, *capability))
    monkeypatch.setattr(decode_common, "detect_capabilities", lambda: cap)

    assert decode_common._is_blackwell_family() is expected


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((10, 0), "blackwell"),
        ((10, 3), "blackwell"),
        ((12, 0), "blackwell_consumer"),
        ((12, 1), "grace_blackwell"),
        ((13, 0), "unknown"),
        ((9, 0), "hopper"),
    ],
)
def test_l2_cache_architecture_labels_follow_the_probe(capability: tuple[int, int], expected: str) -> None:
    assert l2_cache_utils._get_architecture_from_compute_capability(*capability) == expected


def test_l2_cache_fallback_size_is_unchanged_for_sm12x_and_newer() -> None:
    unknown = l2_cache_utils.L2_CACHE_DEFAULTS_MB["unknown"]
    for label in ("blackwell_consumer", "grace_blackwell", "unknown"):
        assert l2_cache_utils._get_fallback_l2_size_mb(label) == unknown


def _single_gpu_topology(architecture: str, compute_capability: str, *, is_grace: bool) -> TopologyInfo:
    gpu = GPUInfo(0, "Test GPU", compute_capability, 96.0, 148, architecture, True)
    return TopologyInfo(
        num_gpus=1,
        gpus=[gpu],
        total_memory_gb=96.0,
        interconnects=[],
        p2p_matrix=[[True]],
        bandwidth_matrix=[[0.0]],
        has_nvlink=False,
        has_nvswitch=False,
        nvlink_version=None,
        max_nvlink_bandwidth_gbps=0.0,
        numa_nodes=1,
        gpu_numa_mapping={0: 0},
        numa_distance_matrix=[[10]],
        cpu_type="aarch64",
        is_grace_cpu=is_grace,
        has_nvlink_c2c=is_grace,
    )


@pytest.mark.parametrize(
    ("architecture", "capability", "expected"),
    [("blackwell", "10.0", True), ("blackwell_sm12x", "12.0", False)],
)
def test_cluster_detector_names_grace_blackwell_only_for_sm10_gpus(
    monkeypatch: pytest.MonkeyPatch, architecture: str, capability: str, expected: bool
) -> None:
    for name in ("SLURM_NNODES", "PBS_NUM_NODES", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cluster_config.subprocess, "run", _no_tools)

    cluster = cluster_config.ClusterDetector().detect(
        single_node_topology=_single_gpu_topology(architecture, capability, is_grace=True),
        force_refresh=True,
    )

    assert cluster.is_grace_blackwell is expected


def test_strategy_optimizer_gives_the_b200_estimate_only_to_sm10_blackwell() -> None:
    model = ModelArchitecture(
        name="tiny",
        model_type=ModelType.DENSE,
        hidden_size=1024,
        num_layers=4,
        num_attention_heads=8,
        vocab_size=32000,
    )
    latency = {}
    for architecture, capability in (("blackwell", "10.0"), ("blackwell_sm12x", "12.0")):
        analysis = StrategyOptimizer()._analyze_strategy(
            ParallelismStrategy(),
            _single_gpu_topology(architecture, capability, is_grace=False),
            model,
            batch_size=1,
            seq_length=128,
            is_training=False,
        )
        latency[architecture] = analysis.estimated_latency_ms

    # One GPU has no communication time, so latency scales with 1 / TFLOPS (2500 vs 1000).
    assert latency["blackwell_sm12x"] == pytest.approx(latency["blackwell"] * 2.5)


def test_ch02_sm12_gpu_labels_name_the_real_families() -> None:
    assert topo.get_gpu_arch_info(12, 0) == {
        "name": "Blackwell (GeForce RTX 50 / RTX PRO)",
        "family": "Blackwell",
        "sm": "sm_120",
    }
    assert topo.get_gpu_arch_info(12, 1) == {"name": "Blackwell (GB10)", "family": "Blackwell", "sm": "sm_121"}
    assert "Ultra" in topo.get_gpu_arch_info(10, 3)["name"]


# ---------------------------------------------------------------------------
# Planner CPU detection (OOF-077, OOF-094)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("soc_id", "c2c", "is_grace", "has_c2c"),
    [
        (NVIDIA_SOC_ID, {0: True, 1: True}, True, True),
        (NVIDIA_SOC_ID, {0: True, 1: False}, True, False),
        (OTHER_VENDOR_SOC_ID, {0: False, 1: False}, False, False),
    ],
)
def test_planner_takes_grace_from_the_soc_id_and_c2c_from_the_driver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    soc_id: str,
    c2c: dict[int, bool],
    is_grace: bool,
    has_c2c: bool,
) -> None:
    # The removed check read "grace" or "nvidia" anywhere in /proc/cpuinfo.
    _grace_host(monkeypatch, tmp_path, soc_id=soc_id, cpuinfo="processor\t: 0\nvendor\t: NVIDIA\n")
    _driver(monkeypatch, c2c)
    props = _props(10, 0, "NVIDIA GB200")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_properties=lambda index: props,
            can_device_access_peer=lambda a, b: False,
        )
    )
    monkeypatch.setattr(topology_detector, "torch", fake_torch)
    monkeypatch.setattr(topology_detector.subprocess, "run", _no_tools)

    topology = topology_detector.TopologyDetector().detect(force_refresh=True)

    assert topology.is_grace_cpu is is_grace
    assert topology.has_nvlink_c2c is has_c2c


# ---------------------------------------------------------------------------
# ch02 topology tool (OOF-077)
# ---------------------------------------------------------------------------


def test_ch02_topology_stops_when_nvidia_smi_contradicts_the_driver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _grace_host(monkeypatch, tmp_path)
    _fake_cuda(monkeypatch, count=1, major=10, minor=0, name="NVIDIA GB200")
    _smi(monkeypatch, {0: (0, _c2c_status(10))})
    _driver(monkeypatch, {0: False})

    with pytest.raises(RuntimeError, match="the CUDA driver reports no NVLink-C2C host link"):
        topo.detect_system_topology()


def test_ch02_topology_takes_a_driver_reported_pcie_link_as_no_c2c(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _grace_host(monkeypatch, tmp_path)
    _fake_cuda(monkeypatch, count=1, major=12, minor=0, name="NVIDIA RTX PRO 6000 Blackwell")
    _smi(monkeypatch, {0: (3, "No Devices support C2C.\n")})
    _driver(monkeypatch, {0: False})

    topology = topo.detect_system_topology()

    assert topology["gpus"][0]["c2c_links_gbps"] == {}
    assert topology["has_nvlink_c2c"] is False
    assert "NVLink-C2C" not in topology["interconnect"]
    assert "unknown" not in topology["interconnect"]
    assert topology["system_type"] == "NVIDIA Grace + Blackwell"


# ---------------------------------------------------------------------------
# ch02 grace_coherent_memory pair (OOF-034, OOF-035, OOF-037, OOF-048, OOF-049)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "impl_class",
    [baseline_grace.BaselineGraceCoherentMemory, optimized_grace.OptimizedGraceCoherentMemory],
    ids=["baseline", "optimized"],
)
def test_grace_pair_runs_only_on_a_driver_reported_nvlink_c2c_link(
    monkeypatch: pytest.MonkeyPatch, impl_class: type
) -> None:
    _fake_cuda(monkeypatch, count=1, major=10, minor=0, name="NVIDIA GB200")
    _machine(monkeypatch, "aarch64")
    skipped = "^SKIPPED: grace_coherent_memory requires Grace-Blackwell coherent memory support"

    _driver(monkeypatch, {0: True})
    assert impl_class(size_mb=1, iterations=1).is_grace_blackwell is True

    # An SM 10.0 GPU on PCIe in an Arm server passed the removed Arm-plus-SM-10 check.
    _driver(monkeypatch, {0: False})
    with pytest.raises(RuntimeError, match=skipped):
        impl_class(size_mb=1, iterations=1)

    # The removed check logged driver errors at debug level and reported SKIPPED.
    monkeypatch.setattr(grace_platform, "_cuda_driver_attribute", _driver_fails)
    with pytest.raises(RuntimeError, match="cuInit failed"):
        impl_class(size_mb=1, iterations=1)

    _machine(monkeypatch, "x86_64")
    monkeypatch.setattr(grace_platform, "_cuda_driver_attribute", _refuse_driver)
    with pytest.raises(RuntimeError, match=skipped):
        impl_class(size_mb=1, iterations=1)


def test_grace_pair_reports_the_same_transfer_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch, count=1, major=10, minor=0, name="NVIDIA GB200")
    _machine(monkeypatch, "aarch64")
    _driver(monkeypatch, {0: True})
    baseline = baseline_grace.GraceCoherentMemoryBenchmark(size_mb=8, iterations=1)
    optimized = optimized_grace.OptimizedGraceCoherentMemoryBenchmark(size_mb=8, iterations=1)
    expected_bytes = float(8 * 1024 * 1024 * 2)

    # Without a timed step there is no elapsed time, so no bandwidth is invented.
    assert baseline.get_custom_metrics() == {"transfer.bytes": expected_bytes}

    # run_step returns the step's elapsed seconds. Fixing it at 4 ms checks the
    # unit conversion, which needs no GPU.
    for benchmark in (baseline, optimized):
        monkeypatch.setattr(benchmark._impl, "run_step", lambda: 0.004)
        benchmark.benchmark_fn()

    assert baseline.bandwidth_gb_s == pytest.approx(expected_bytes / 0.004 / 1e9)
    assert baseline.bandwidth_gb_s == pytest.approx(optimized.bandwidth_gb_s)
    assert baseline.get_custom_metrics() == optimized.get_custom_metrics()
    source = (CODE_ROOT / "ch02" / "baseline_grace_coherent_memory.py").read_text(encoding="utf-8")
    assert "Seed FIRST" not in source
    assert "compute_memory_transfer_metrics" not in source


@pytest.mark.parametrize("name", ["expectations_b200.json", "expectations_4x_gb200.json"])
def test_ch02_expectations_drop_the_grace_entry_from_the_removed_optimized_path(name: str) -> None:
    payload = json.loads((CODE_ROOT / "ch02" / name).read_text(encoding="utf-8"))

    assert "grace_coherent_memory" not in payload["examples"]
    assert "memory_transfer" in payload["examples"]


# ---------------------------------------------------------------------------
# ch04 siblings (OOF-073, OOF-074, OOF-075)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "detect",
    [multi_node.detect_gb200_gb300, symmetric_memory.detect_gb200_gb300],
    ids=["multi_node_blackwell", "symmetric_memory_example"],
)
@pytest.mark.parametrize(
    ("machine", "soc_id", "expected"),
    [("aarch64", OTHER_VENDOR_SOC_ID, False), ("aarch64", NVIDIA_SOC_ID, True), ("x86_64", NVIDIA_SOC_ID, False)],
)
def test_ch04_grace_checks_need_the_nvidia_soc_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    detect: typing.Callable[[], bool],
    machine: str,
    soc_id: str,
    expected: bool,
) -> None:
    _grace_host(monkeypatch, tmp_path, soc_id=soc_id, machine=machine)

    assert detect() is expected


@pytest.mark.parametrize(
    ("machine", "soc_id", "expected"),
    [("aarch64", OTHER_VENDOR_SOC_ID, False), ("aarch64", NVIDIA_SOC_ID, True)],
)
def test_nccl_topology_grace_flag_needs_the_nvidia_soc_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, machine: str, soc_id: str, expected: bool
) -> None:
    _grace_host(monkeypatch, tmp_path, soc_id=soc_id, machine=machine)
    _fake_cuda(monkeypatch, count=2, major=10, minor=0, name="NVIDIA GB200")
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: True)

    info = nccl_config.detect_b200_multigpu_topology()

    assert info["has_grace_cpu"] is expected
    assert info["grace_soc_id"] == (soc_id if expected else None)


@pytest.mark.parametrize(
    ("name", "major", "minor", "is_b200"),
    [
        ("NVIDIA B200", 10, 0, True),
        ("NVIDIA GB200", 10, 0, False),
        ("NVIDIA B300", 10, 3, False),
        ("NVIDIA RTX PRO 6000 Blackwell", 12, 0, False),
        ("NVIDIA GB10", 12, 1, False),
    ],
)
def test_topology_reports_detected_resources_without_inventing_fabric(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    major: int,
    minor: int,
    is_b200: bool,
) -> None:
    # Enumeration is a metadata seam. This test does not run or qualify a benchmark.
    _fake_cuda(monkeypatch, count=4, major=major, minor=minor, name=name)
    monkeypatch.setattr(nccl_config, "host_grace_soc_id", lambda: None)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: True)

    info = nccl_config.detect_b200_multigpu_topology()
    assert info["is_b200_multigpu"] is is_b200
    assert info["all_to_all_peer_access"] is True
    assert info["has_nvswitch"] is None
    assert info["nvlink_topology"] == "unknown"

    nccl_config.print_multigpu_topology()
    printed = capsys.readouterr().out
    assert name in printed
    assert f"Memory: {4 * 96 * 1024**3 / 1e9:.2f} GB" in printed
    assert "NVSwitch detected" not in printed
    assert "TB/s aggregate" not in printed
    assert "HBM3e" not in printed


def test_peer_access_summary_uses_actual_four_gpu_resources(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fake_cuda(monkeypatch, count=4, major=12, minor=0, name="NVIDIA RTX PRO 6000 Blackwell")
    monkeypatch.setattr(symmetric_memory, "detect_gb200_gb300", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: True)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction: None)

    symmetric_memory.enable_nvlink_c2c_optimizations()
    printed = capsys.readouterr().out
    assert "NVIDIA RTX PRO 6000 Blackwell" in printed
    assert f"Total memory: {4 * 96 * 1024**3 / 1e9:.2f} GB" in printed
    assert "4x B200" not in printed
    assert "0.72 TB HBM3e" not in printed
    assert "31.2 TB/s" not in printed
    assert "900 GB/s per pair" not in printed
    assert "GPU Peer Access Features:" in printed


def test_single_gpu_topology_printout_stops_at_the_multigpu_warning(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fake_cuda(monkeypatch, count=1, major=10, minor=0, name="NVIDIA B200")
    nccl_config.print_multigpu_topology()
    printed = capsys.readouterr().out
    assert "Expected >=2 GPUs, found 1" in printed
    assert "Detected resources:" not in printed
    assert "Memory: 0.00 GB" not in printed


@pytest.mark.parametrize(
    ("soc_id", "c2c", "present", "absent"),
    [
        (
            NVIDIA_SOC_ID,
            True,
            ("Architecture: ARM Neoverse V2", "CPU-GPU: NVLink-C2C, 900 GB/s bidirectional spec peak"),
            ("144 threads",),
        ),
        (NVIDIA_SOC_ID, False, ("Architecture: ARM Neoverse V2",), ("NVLink-C2C", "900 GB/s")),
        (OTHER_VENDOR_SOC_ID, True, ("its CPU is not NVIDIA Grace",), ("ARM Neoverse V2", "NVLink-C2C")),
    ],
)
def test_nccl_gb200_printout_follows_grace_and_driver_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    soc_id: str,
    c2c: bool,
    present: tuple[str, ...],
    absent: tuple[str, ...],
) -> None:
    _grace_host(monkeypatch, tmp_path, soc_id=soc_id)
    _fake_cuda(monkeypatch, count=2, major=10, minor=0, name="NVIDIA GB200")
    _driver(monkeypatch, {0: c2c, 1: c2c})
    monkeypatch.setattr(nccl_config.os, "environ", {})

    nccl_config.configure_nccl_for_gb200_gb300(verbose=True)
    printed = capsys.readouterr().out

    for text in present:
        assert text in printed
    for text in absent:
        assert text not in printed


@pytest.mark.parametrize("c2c", [True, False])
def test_multi_node_setup_prints_the_c2c_peak_only_with_driver_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], c2c: bool
) -> None:
    _grace_host(monkeypatch, tmp_path)
    _fake_cuda(monkeypatch, count=2, major=10, minor=0, name="NVIDIA GB200")
    _driver(monkeypatch, {0: c2c})
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(multi_node.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(multi_node.os, "environ", {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"})

    multi_node.setup_blackwell_distributed(backend="nccl")
    printed = capsys.readouterr().out

    assert "NVIDIA Grace CPU detected" in printed
    assert "GB200/GB300 Grace-Blackwell detected" not in printed
    assert "NVLS (NVLink SHARP)" in printed
    assert "NVLS (NVLink-C2C)" not in printed
    assert ("CPU-GPU: NVLink-C2C, 900 GB/s bidirectional spec peak" in printed) is c2c


@pytest.mark.parametrize(
    ("c2c", "present", "absent"),
    [
        (
            True,
            ("NVLink-C2C Features:", "CPU-GPU: NVLink-C2C, 900 GB/s bidirectional spec peak"),
            ("no NVLink-C2C host link reported",),
        ),
        (
            False,
            ("GPU Peer Access Configuration for Blackwell (no NVLink-C2C host link reported)", "GPU Peer Access Features:"),
            ("CPU-GPU", "NVLink-C2C Features"),
        ),
    ],
)
def test_peer_access_report_names_c2c_only_with_driver_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    c2c: bool,
    present: tuple[str, ...],
    absent: tuple[str, ...],
) -> None:
    _grace_host(monkeypatch, tmp_path)
    _fake_cuda(monkeypatch, count=2, major=10, minor=0, name="NVIDIA GB200")
    _driver(monkeypatch, {0: c2c, 1: c2c})
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: True)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction: None)

    symmetric_memory.enable_nvlink_c2c_optimizations()
    printed = capsys.readouterr().out

    for text in present:
        assert text in printed
    for text in absent:
        assert text not in printed


def test_unified_memory_demo_needs_a_driver_reported_c2c_link_on_gpu0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _grace_host(monkeypatch, tmp_path)
    _fake_cuda(monkeypatch, count=1, major=12, minor=0, name="NVIDIA RTX PRO 6000 Blackwell")
    _driver(monkeypatch, {0: False})

    symmetric_memory.demonstrate_gb200_unified_memory()
    printed = capsys.readouterr().out

    assert "GPU 0 has no NVLink-C2C host link to the Grace CPU" in printed
    assert "GB200/GB300 detected" not in printed
    assert "900 GB/s" not in printed


def test_unified_memory_demo_compares_a_one_way_copy_with_the_per_direction_peak() -> None:
    source = (CODE_ROOT / "ch04" / "symmetric_memory_example.py").read_text(encoding="utf-8")

    assert symmetric_memory._C2C_PEAK_PER_DIRECTION_GBPS == 450.0
    assert "~800 GB/s" not in source
    assert "bandwidth / 800" not in source
    assert "(bandwidth / _C2C_PEAK_PER_DIRECTION_GBPS)" in source


# ---------------------------------------------------------------------------
# ch02 CUDA demo (OOF-076). The .cu cannot build without nvcc, so these tests
# check the source and run the SoC ID reader through the host C++ compiler.
# ---------------------------------------------------------------------------


def _demo_grace_reader() -> str:
    source = NVLINK_DEMO.read_text(encoding="utf-8")
    signature = "bool detect_grace_cpu(char* soc_id, size_t soc_id_size) {"
    return signature + source.split(signature, 1)[1].split("SystemInfo detect_system_capabilities() {", 1)[0]


def test_nvlink_demo_takes_grace_from_the_soc_id_and_c2c_from_the_driver() -> None:
    source = NVLINK_DEMO.read_text(encoding="utf-8")
    reader = _demo_grace_reader()
    detection = source.split("SystemInfo detect_system_capabilities() {", 1)[1].split(
        "double benchmark_host_to_device", 1
    )[0]
    superchip_branch = detection.split("if (has_grace_cpu && info.has_nvlink_c2c) {", 1)[1].split("}", 1)[0]

    assert '"/sys/bus/soc/devices/soc*/soc_id"' in reader
    assert 'strncmp(line, "jep106:036b:", 12) == 0' in reader
    for stale in ("/proc/cpuinfo", "Neoverse", "0xd40", "At least ARM"):
        assert stale not in reader
    assert "cudaDevAttrHostNativeAtomicSupported" in detection
    assert "info.is_grace_blackwell = true;" in superchip_branch
    assert "/proc/driver/nvidia/gpus/0000:01:00.0/information" not in source
    assert "Grace CPU detected (ARM Neoverse)" not in source


def test_nvlink_demo_soc_id_reader_accepts_only_the_nvidia_vendor_prefix(tmp_path: Path) -> None:
    compiler = next((path for name in ("c++", "clang++", "g++") if (path := shutil.which(name))), None)
    if compiler is None:
        pytest.skip("SKIPPED: no host C++ compiler for the SoC ID reader check")
    tree = tmp_path / "devices"
    harness = tmp_path / "reader.cpp"
    harness.write_text(
        "#include <cstdio>\n#include <cstring>\n#include <glob.h>\n"
        # The reader body is Linux-only. Other hosts compile the same body here.
        "#ifndef __linux__\n#define __linux__ 1\n#endif\n"
        + _demo_grace_reader().replace(
            '"/sys/bus/soc/devices/soc*/soc_id"', json.dumps(f"{tree}/soc*/soc_id")
        )
        + "int main() {\n"
        "    char id[64];\n"
        "    bool found = detect_grace_cpu(id, sizeof(id));\n"
        '    std::printf("%d %s\\n", found ? 1 : 0, id);\n'
        "    return 0;\n"
        "}\n",
        encoding="utf-8",
    )
    binary = tmp_path / "reader"
    subprocess.run(
        [compiler, "-std=c++17", str(harness), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    def _read() -> str:
        return subprocess.run(
            [str(binary)], check=True, capture_output=True, text=True, timeout=30
        ).stdout.strip()

    assert _read() == "0"
    (tree / "soc0").mkdir(parents=True)
    (tree / "soc0" / "soc_id").write_text(OTHER_VENDOR_SOC_ID + "\n", encoding="utf-8")
    assert _read() == "0"
    (tree / "soc1").mkdir()
    (tree / "soc1" / "soc_id").write_text(NVIDIA_SOC_ID + "\n", encoding="utf-8")
    assert _read() == f"1 {NVIDIA_SOC_ID}"


def test_cluster_launch_query_requests_cuda_dev_attr_cluster_launch(monkeypatch) -> None:
    # The probe used attribute 164, which is not cudaDevAttrClusterLaunch (120), so the
    # query failed and the probe fell back to assuming cluster support on every GPU.
    import ctypes

    from core.scripts.utilities import probe_hardware_capabilities as probe

    requested: list[tuple[int, int]] = []

    class _GetAttribute:
        restype = None
        argtypes = None

        def __call__(self, value_ref, attribute, device) -> int:
            requested.append((attribute.value, device.value))
            value_ref._obj.value = 1
            return 0

    class _Runtime:
        cudaDeviceGetAttribute = _GetAttribute()

    monkeypatch.setattr(ctypes, "CDLL", lambda _name: _Runtime())

    assert probe._query_cluster_launch(3) == 1
    assert requested == [(120, 3)]


def test_ch12_enhanced_conditional_graph_labels_gb10_only_for_sm121() -> None:
    # Same CC-only GB10 claim as W2-004. SM 12.0 is RTX 50 / RTX PRO, not Grace.
    source = (
        Path(__file__).resolve().parents[1] / "ch12" / "optimized_cuda_graphs_conditional_enhanced.cu"
    ).read_text(encoding="utf-8")
    assert "bool is_gb10 = (prop.major == 12 && prop.minor == 1);" in source
    assert "is_gb10 ? \"Grace-Blackwell GB10\"" in source
    assert "is_sm12x ? \"Grace-Blackwell GB10\"" not in source
    assert "prop.major >= 12" not in source
    assert 'std::printf("SKIPPED: CUDA Graphs require compute capability 7.5 or newer\\n");' in source
