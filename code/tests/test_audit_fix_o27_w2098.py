"""W2-098 regressions: Grace identity and NVLink-C2C claims need explicit evidence.

Grace identity must come from the NVIDIA SoC vendor ID in sysfs, not from Arm
core names. NVLink-C2C claims must come from links that ``nvidia-smi c2c -s``
reports as up, and the CUDA driver must agree that the GPU sits on NVLink-C2C.
Both checks live in core.common.grace_platform. These tests replace only the
host seams (``/proc/cpuinfo``, sysfs, ``subprocess.run``, the CUDA driver
attribute query and the CUDA device query) and run the real detection and
reporting code in both chapter modules.
"""

from __future__ import annotations

import builtins
import io
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from ch02 import cpu_gpu_topology_aware as topo
from ch04 import distributed_helper
from ch04 import gb200_grace_numa_optimization as grace_numa
from core.common import grace_platform

# What a 64-bit task reads from /proc/cpuinfo on a Neoverse V2 core. arm64
# kernels print no model-name line here, so this text is the same on Grace and
# on other vendors' Neoverse V2 CPUs.
NEOVERSE_V2_CPUINFO = (
    "processor\t: 0\n"
    "BogoMIPS\t: 2000.00\n"
    "Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid "
    "asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat "
    "ilrcpc flagm ssbs sb dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 "
    "frint svei8mm svebf16 i8mm bf16 dgh bti\n"
    "CPU implementer\t: 0x41\n"
    "CPU architecture: 8\n"
    "CPU variant\t: 0x0\n"
    "CPU part\t: 0xd4f\n"
    "CPU revision\t: 0\n"
)
# A 32-bit compat task on any arm64 host also sees an "ARMv8" model-name line.
COMPAT_ARM_CPUINFO = "model name\t: ARMv8 Processor rev 0 (v8l)\n" + NEOVERSE_V2_CPUINFO

NVIDIA_SOC_ID = "jep106:036b:0241"
OTHER_VENDOR_SOC_ID = "jep106:043b:0000"


def c2c_status(gpu_name: str, link_count: int) -> str:
    lines = [f"GPU 0: {gpu_name} (UUID: GPU-00000000-0000-0000-0000-000000000000)"]
    lines += [f"\t C2C Link {link}: 44.712 GB/s" for link in range(link_count)]
    return "\n".join(lines) + "\n"


class FakeHostTools:
    """Replaces subprocess.run on a host with nvidia-smi but no numactl or lspci.

    ``c2c`` maps a GPU index to ``(returncode, stdout)`` for ``nvidia-smi c2c``,
    or to None when nvidia-smi is not installed.
    """

    def __init__(self, c2c: dict[int, tuple[int, str] | None] | None = None) -> None:
        self.c2c = c2c or {}
        self.c2c_queries: list[list[str]] = []

    def run(self, cmd: list[str], *args: object, **kwargs: object) -> SimpleNamespace:
        cmd = [str(part) for part in cmd]
        if cmd[:2] == ["nvidia-smi", "c2c"]:
            self.c2c_queries.append(cmd)
            selector = cmd[cmd.index("-i") + 1]
            if ":" in selector:
                device_index = next(
                    index for index in self.c2c if self.driver_pci_bus_id(index) == selector
                )
            else:
                device_index = int(selector)
            response = self.c2c[device_index]
            if response is None:
                raise FileNotFoundError(cmd[0])
            returncode, stdout = response
            return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
        raise FileNotFoundError(cmd[0])

    def driver_attribute(self, device_index: int, attribute: int) -> int:
        """Answer like a CUDA driver that agrees with the nvidia-smi output above.

        A GPU that nvidia-smi lists with links up, or that it cannot report,
        sits on NVLink-C2C. A GPU that it lists with no active link sits on PCIe.
        """
        assert attribute in (
            grace_platform.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,
            grace_platform.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
        ), attribute
        response = self.c2c[device_index]
        if response is None or response[0] != 0:
            return 1
        return 1 if "C2C Link 0:" in response[1] else 0

    def driver_pci_bus_id(self, device_index: int) -> str:
        return f"0000:{device_index + 1:02x}:00.0"


@pytest.fixture
def soc_bus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Mirror /sys/bus/soc/devices, where each socN entry links to /sys/devices/socN."""
    bus = tmp_path / "sys" / "bus" / "soc" / "devices"
    bus.mkdir(parents=True)
    monkeypatch.setattr(grace_platform, "_SOC_BUS_DIR", bus)
    next_index = [0]

    def add(soc_id: str) -> None:
        index = next_index[0]
        next_index[0] += 1
        device = tmp_path / "sys" / "devices" / f"soc{index}"
        device.mkdir(parents=True)
        (device / "soc_id").write_text(soc_id + "\n", encoding="utf-8")
        (bus / f"soc{index}").symlink_to(device, target_is_directory=True)

    return add


def use_host(monkeypatch: pytest.MonkeyPatch, machine: str, cpuinfo: str, tools: FakeHostTools) -> None:
    real_open = builtins.open

    def open_with_cpuinfo(file: object, *args: object, **kwargs: object) -> object:
        if str(file) == "/proc/cpuinfo":
            return io.StringIO(cpuinfo)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(topo.platform, "machine", lambda: machine)
    monkeypatch.setattr(builtins, "open", open_with_cpuinfo)
    monkeypatch.setattr(topo.subprocess, "run", tools.run)
    monkeypatch.setattr(grace_platform, "_cuda_driver_attribute", tools.driver_attribute)
    monkeypatch.setattr(grace_platform, "_cuda_driver_pci_bus_id", tools.driver_pci_bus_id)


def use_gpus(
    monkeypatch: pytest.MonkeyPatch, *, count: int, major: int, minor: int, name: str, sms: int = 148
) -> None:
    props = SimpleNamespace(
        name=name, major=major, minor=minor, total_memory=96 * 1024**3, multi_processor_count=sms
    )
    monkeypatch.setattr(topo.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(topo.torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(topo.torch.cuda, "get_device_properties", lambda index: props)


def test_soc_id_match_requires_the_nvidia_jep106_vendor_code() -> None:
    assert grace_platform.nvidia_grace_soc_id([OTHER_VENDOR_SOC_ID, NVIDIA_SOC_ID]) == NVIDIA_SOC_ID
    assert grace_platform.nvidia_grace_soc_id([OTHER_VENDOR_SOC_ID]) is None
    assert grace_platform.nvidia_grace_soc_id(["jep106:036b"]) is None
    assert grace_platform.nvidia_grace_soc_id([]) is None


def test_read_soc_ids_follows_sysfs_soc_links(soc_bus: Callable[[str], None]) -> None:
    soc_bus(OTHER_VENDOR_SOC_ID)
    soc_bus(NVIDIA_SOC_ID)

    assert grace_platform.read_soc_ids() == [OTHER_VENDOR_SOC_ID, NVIDIA_SOC_ID]


def test_query_gpu_c2c_links_keeps_only_links_with_a_reported_speed(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = c2c_status("NVIDIA GH200 480GB", 2) + "\t C2C Link 2: <inactive>\n"
    tools = FakeHostTools({0: (0, stdout), 1: (6, ""), 2: None})
    monkeypatch.setattr(grace_platform.subprocess, "run", tools.run)

    assert grace_platform.query_gpu_c2c_links(0) == {0: 44.712, 1: 44.712}
    assert grace_platform.query_gpu_c2c_links(1) is None
    assert grace_platform.query_gpu_c2c_links(2) is None
    assert tools.c2c_queries[0] == ["nvidia-smi", "c2c", "-i", "0", "-s"]


def test_ch02_neoverse_v2_without_nvidia_soc_id_is_not_grace(
    monkeypatch: pytest.MonkeyPatch,
    soc_bus: Callable[[str], None],
) -> None:
    soc_bus(OTHER_VENDOR_SOC_ID)
    use_host(monkeypatch, "aarch64", COMPAT_ARM_CPUINFO, FakeHostTools())

    info = topo.detect_cpu_info()

    assert info["is_grace"] is False
    assert info["soc_id"] is None
    assert info["cpu_type"] == "ARM"


def test_ch02_grace_is_identified_from_the_nvidia_soc_id(
    monkeypatch: pytest.MonkeyPatch,
    soc_bus: Callable[[str], None],
) -> None:
    soc_bus(NVIDIA_SOC_ID)
    use_host(monkeypatch, "aarch64", NEOVERSE_V2_CPUINFO, FakeHostTools())

    info = topo.detect_cpu_info()

    assert info["is_grace"] is True
    assert info["soc_id"] == NVIDIA_SOC_ID
    assert info["cpu_type"] == "NVIDIA Grace"


@pytest.mark.parametrize(
    ("major", "minor", "name", "system_type"),
    [
        (9, 0, "NVIDIA GH200 480GB", "GH200 (Grace + Hopper)"),
        (10, 0, "NVIDIA GB200", "GB200/GB300 (Grace + Blackwell)"),
    ],
)
def test_ch02_grace_superchip_needs_driver_reported_c2c_links(
    monkeypatch: pytest.MonkeyPatch,
    soc_bus: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
    major: int,
    minor: int,
    name: str,
    system_type: str,
) -> None:
    soc_bus(NVIDIA_SOC_ID)
    tools = FakeHostTools({0: (0, c2c_status(name, 10))})
    use_host(monkeypatch, "aarch64", NEOVERSE_V2_CPUINFO, tools)
    use_gpus(monkeypatch, count=1, major=major, minor=minor, name=name)

    topology = topo.detect_system_topology()
    topo.print_topology_info(topology)
    printed = capsys.readouterr().out

    assert topology["has_nvlink_c2c"] is True
    assert topology["gpus"][0]["c2c_links_gbps"] == {link: 44.712 for link in range(10)}
    assert topology["interconnect"] == "NVLink-C2C (10 links up, bandwidth unmeasured)"
    assert topology["system_type"] == system_type
    assert f"SoC ID: {NVIDIA_SOC_ID}" in printed
    assert "NVLink-C2C: 10 links up (nvidia-smi c2c -s)" in printed
    assert "NVLink-C2C enables CPU memory as extended GPU memory" in printed or "NVLink-C2C with Hopper" in printed


def test_ch02_grace_cpu_with_a_gpu_that_reports_no_c2c_link_claims_no_c2c(
    monkeypatch: pytest.MonkeyPatch,
    soc_bus: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    soc_bus(NVIDIA_SOC_ID)
    tools = FakeHostTools({0: (0, c2c_status("NVIDIA H100 PCIe", 0))})
    use_host(monkeypatch, "aarch64", NEOVERSE_V2_CPUINFO, tools)
    use_gpus(monkeypatch, count=1, major=9, minor=0, name="NVIDIA H100 PCIe")

    topology = topo.detect_system_topology()
    topo.print_topology_info(topology)
    printed = capsys.readouterr().out

    assert topology["has_nvlink_c2c"] is False
    assert topology["system_type"] == "NVIDIA Grace + Hopper"
    assert "NVLink-C2C" not in topology["interconnect"]
    assert "NVLink-C2C" not in printed
    assert "GH200" not in printed


def test_ch02_failed_c2c_query_is_reported_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
    soc_bus: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    soc_bus(NVIDIA_SOC_ID)
    tools = FakeHostTools({0: (6, "")})
    use_host(monkeypatch, "aarch64", NEOVERSE_V2_CPUINFO, tools)
    use_gpus(monkeypatch, count=1, major=10, minor=0, name="NVIDIA GB200")

    topology = topo.detect_system_topology()
    topo.print_topology_info(topology)
    printed = capsys.readouterr().out

    assert topology["gpus"][0]["c2c_links_gbps"] is None
    assert topology["has_nvlink_c2c"] is False
    assert topology["interconnect"] == "CPU-GPU link unknown (nvidia-smi c2c -s failed)"
    assert topology["system_type"] == "NVIDIA Grace + Blackwell"
    assert "NVLink-C2C: status unavailable from nvidia-smi c2c -s" in printed
    assert "GB200" not in topology["system_type"]


@pytest.mark.parametrize(
    ("gpu1_links", "has_c2c", "interconnect"),
    [
        (0, False, "Mixed per GPU (C2C links up on GPUs [0], no C2C link on GPUs [1])"),
        (None, False, "Mixed per GPU (C2C links up on GPUs [0], C2C status unknown on GPUs [1])"),
        (9, True, "NVLink-C2C (9-10 links up per GPU, bandwidth unmeasured)"),
    ],
)
def test_ch02_system_interconnect_claims_c2c_only_when_every_gpu_reports_links(
    monkeypatch: pytest.MonkeyPatch,
    soc_bus: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
    gpu1_links: int | None,
    has_c2c: bool,
    interconnect: str,
) -> None:
    soc_bus(NVIDIA_SOC_ID)
    gpu1 = (6, "") if gpu1_links is None else (0, c2c_status("NVIDIA GB200", gpu1_links))
    tools = FakeHostTools({0: (0, c2c_status("NVIDIA GB200", 10)), 1: gpu1})
    use_host(monkeypatch, "aarch64", NEOVERSE_V2_CPUINFO, tools)
    use_gpus(monkeypatch, count=2, major=10, minor=0, name="NVIDIA GB200")

    topology = topo.detect_system_topology()
    topo.print_topology_info(topology)
    printed = capsys.readouterr().out

    assert topology["has_nvlink_c2c"] is has_c2c
    assert topology["interconnect"] == interconnect
    assert f"Interconnect: {interconnect}" in printed
    if not has_c2c:
        assert not topology["interconnect"].startswith("NVLink-C2C")
        assert topology["system_type"] == "NVIDIA Grace + Blackwell"
        assert "NVLink-C2C enables CPU memory as extended GPU memory" not in printed


@pytest.mark.parametrize(
    ("machine", "cpuinfo", "system_type"),
    [
        ("x86_64", "vendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Xeon(R) Platinum\n", "Intel x86_64 + Blackwell"),
        ("aarch64", COMPAT_ARM_CPUINFO, "ARM + Blackwell"),
    ],
)
def test_ch02_non_grace_host_never_claims_or_queries_c2c(
    monkeypatch: pytest.MonkeyPatch,
    soc_bus: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
    machine: str,
    cpuinfo: str,
    system_type: str,
) -> None:
    soc_bus(OTHER_VENDOR_SOC_ID)
    tools = FakeHostTools()
    use_host(monkeypatch, machine, cpuinfo, tools)
    use_gpus(monkeypatch, count=2, major=10, minor=0, name="NVIDIA B200")

    topology = topo.detect_system_topology()
    topo.print_topology_info(topology)
    printed = capsys.readouterr().out

    assert tools.c2c_queries == []
    assert topology["has_nvlink_c2c"] is False
    assert topology["system_type"] == system_type
    assert "NVLink-C2C" not in topology["interconnect"]
    assert "NVLink-C2C" not in printed


def test_ch04_arm_model_name_without_nvidia_soc_id_is_not_grace(
    monkeypatch: pytest.MonkeyPatch,
    soc_bus: Callable[[str], None],
) -> None:
    soc_bus(OTHER_VENDOR_SOC_ID)
    use_host(monkeypatch, "aarch64", COMPAT_ARM_CPUINFO, FakeHostTools())

    info = grace_numa.detect_grace_cpu()

    assert info["is_grace"] is False
    assert info["soc_id"] is None
    assert "cpu_model" not in info


def test_ch04_main_skips_on_a_non_grace_arm_host(
    monkeypatch: pytest.MonkeyPatch,
    soc_bus: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    soc_bus(OTHER_VENDOR_SOC_ID)
    use_host(monkeypatch, "aarch64", COMPAT_ARM_CPUINFO, FakeHostTools())

    status = distributed_helper.run_main_with_skip_status(grace_numa.main)
    printed = capsys.readouterr().out

    assert status == 3
    assert printed.startswith("SKIPPED: gb200_grace_numa_optimization requires an NVIDIA Grace CPU")
    assert "900 GB/s" not in printed


def test_ch04_grace_is_identified_from_the_nvidia_soc_id(
    monkeypatch: pytest.MonkeyPatch,
    soc_bus: Callable[[str], None],
) -> None:
    soc_bus(NVIDIA_SOC_ID)
    use_host(monkeypatch, "aarch64", NEOVERSE_V2_CPUINFO, FakeHostTools())

    info = grace_numa.detect_grace_cpu()

    assert info["is_grace"] is True
    assert info["soc_id"] == NVIDIA_SOC_ID
    assert info["cpu_model"] == "NVIDIA Grace"


def test_ch04_system_info_claims_c2c_only_when_every_gpu_reports_links(
    monkeypatch: pytest.MonkeyPatch,
    soc_bus: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    soc_bus(NVIDIA_SOC_ID)
    tools = FakeHostTools({0: (0, c2c_status("NVIDIA GB200", 10)), 1: (0, c2c_status("NVIDIA GB200", 0))})
    use_host(monkeypatch, "aarch64", NEOVERSE_V2_CPUINFO, tools)
    use_gpus(monkeypatch, count=2, major=10, minor=0, name="NVIDIA GB200")

    grace_numa.print_grace_system_info()
    partial = capsys.readouterr().out

    assert "NVIDIA Grace System Information" in partial
    assert "Grace Superchip" not in partial
    assert f"NVIDIA Grace CPU detected (SoC ID {NVIDIA_SOC_ID})" in partial
    assert "Physical cores:" in partial
    assert "NVLink-C2C: not reported up on every GPU, so no C2C features are claimed" in partial
    assert "GPU 1: 0 links up" in partial
    assert "900 GB/s" not in partial
    assert "NVLink 5" not in partial

    tools.c2c[1] = (0, c2c_status("NVIDIA GB200", 10))
    grace_numa.print_grace_system_info()
    complete = capsys.readouterr().out

    assert "GB200 Grace Superchip System Information" in complete
    assert "GPU 1: 10 NVLink-C2C links up (nvidia-smi c2c -s)" in complete
    assert "NVLink-C2C spec peak: 900 GB/s bidirectional, 450 GB/s per direction (not measured here)" in complete
    assert "2-GPU totals from device properties:" in complete
    assert "  - SMs: 296" in complete
    assert "GPU-GPU NVLink 5 spec peak: 1800 GB/s bidirectional per GPU (not measured here)" in complete


def test_ch04_system_info_reports_a_pcie_blackwell_gpu_on_grace_from_its_own_properties(
    monkeypatch: pytest.MonkeyPatch,
    soc_bus: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    soc_bus(NVIDIA_SOC_ID)
    tools = FakeHostTools({0: (0, c2c_status("NVIDIA RTX PRO 6000 Blackwell", 0))})
    use_host(monkeypatch, "aarch64", NEOVERSE_V2_CPUINFO, tools)
    use_gpus(monkeypatch, count=1, major=12, minor=0, name="NVIDIA RTX PRO 6000 Blackwell", sms=188)

    grace_numa.print_grace_system_info()
    printed = capsys.readouterr().out

    assert "NVIDIA Grace System Information" in printed
    assert "GPU 0: NVIDIA RTX PRO 6000 Blackwell" in printed
    assert "Compute capability: 12.0" in printed
    assert "SMs: 188" in printed
    assert "Memory: 103 GB\n" in printed
    for claim in ("GB200", "GB300", "B200/B300", "SMs: 148", "HBM3e", "Grace NVLink-C2C Features", "900 GB/s"):
        assert claim not in printed


def test_ch04_system_info_on_a_non_grace_host_names_no_grace_or_superchip(
    monkeypatch: pytest.MonkeyPatch,
    soc_bus: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    soc_bus(OTHER_VENDOR_SOC_ID)
    tools = FakeHostTools()
    use_host(monkeypatch, "x86_64", "vendor_id\t: GenuineIntel\n", tools)
    use_gpus(monkeypatch, count=8, major=10, minor=0, name="NVIDIA B200")

    grace_numa.print_grace_system_info()
    printed = capsys.readouterr().out

    assert tools.c2c_queries == []
    assert "CPU-GPU System Information" in printed
    assert "Not NVIDIA Grace (no NVIDIA SoC ID under /sys/bus/soc/devices)" in printed
    assert "Memory: 103 GB HBM3e" in printed
    for claim in ("GB200", "Grace CPU detected", "NVLink-C2C", "1184 SMs", "NVLink 5"):
        assert claim not in printed


INTEL_CPUINFO = "vendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Xeon(R) Platinum\n"
NO_GRACE_CLAIMS = ("Grace CPU", "GB200", "GH200", "NVLink-C2C spec peak", "800 GB/s")


@pytest.mark.parametrize(
    ("machine", "cpuinfo", "soc_id", "c2c", "gpu", "expected", "unexpected"),
    [
        pytest.param(
            "aarch64",
            NEOVERSE_V2_CPUINFO,
            NVIDIA_SOC_ID,
            {0: (0, c2c_status("NVIDIA GB200", 10))},
            (10, 0, "NVIDIA GB200"),
            (
                "DataLoader Configuration for GB200 (GPU 0)",
                "Grace CPU: ",
                "CPU→GPU bandwidth: up to 450 GB/s per direction (NVLink-C2C spec peak, not measured here)",
            ),
            ("800 GB/s",),
            id="grace-gb200-c2c-up",
        ),
        pytest.param(
            "aarch64",
            NEOVERSE_V2_CPUINFO,
            NVIDIA_SOC_ID,
            {0: (0, c2c_status("NVIDIA GH200 480GB", 10))},
            (9, 0, "NVIDIA GH200 480GB"),
            ("DataLoader Configuration for GH200 (GPU 0)",),
            ("GB200",),
            id="grace-gh200-c2c-up",
        ),
        pytest.param(
            "aarch64",
            NEOVERSE_V2_CPUINFO,
            NVIDIA_SOC_ID,
            {0: (0, c2c_status("NVIDIA RTX PRO 6000 Blackwell", 0))},
            (12, 0, "NVIDIA RTX PRO 6000 Blackwell"),
            (
                "DataLoader Configuration (GPU 0)",
                "Grace CPU: ",
                "CPU→GPU bandwidth: no NVLink-C2C link reported for this GPU",
            ),
            ("GB200", "GH200", "NVLink-C2C spec peak"),
            id="grace-pcie-gpu-no-c2c",
        ),
        pytest.param(
            "aarch64",
            NEOVERSE_V2_CPUINFO,
            NVIDIA_SOC_ID,
            {0: (6, "")},
            (10, 0, "NVIDIA GB200"),
            (
                "DataLoader Configuration (GPU 0)",
                "CPU→GPU bandwidth: NVLink-C2C status unavailable from nvidia-smi c2c -s",
            ),
            ("GB200 (GPU", "NVLink-C2C spec peak"),
            id="grace-c2c-query-failed",
        ),
        pytest.param(
            "aarch64",
            COMPAT_ARM_CPUINFO,
            OTHER_VENDOR_SOC_ID,
            {},
            (10, 0, "NVIDIA B200"),
            (
                "DataLoader Configuration (GPU 0)",
                "(not NVIDIA Grace)",
                "CPU→GPU bandwidth: no NVLink-C2C path because the host is not NVIDIA Grace",
            ),
            NO_GRACE_CLAIMS,
            id="other-vendor-arm",
        ),
        pytest.param(
            "x86_64",
            INTEL_CPUINFO,
            OTHER_VENDOR_SOC_ID,
            {},
            (10, 0, "NVIDIA B200"),
            (
                "DataLoader Configuration (GPU 0)",
                "CPU: x86_64, ",
                "NUMA node: not bound, existing CPU affinity kept",
                "CPU→GPU bandwidth: no NVLink-C2C path because the host is not NVIDIA Grace",
            ),
            NO_GRACE_CLAIMS,
            id="x86-b200",
        ),
    ],
)
def test_ch04_dataloader_text_follows_grace_and_c2c_evidence(
    monkeypatch: pytest.MonkeyPatch,
    soc_bus: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
    machine: str,
    cpuinfo: str,
    soc_id: str,
    c2c: dict[int, tuple[int, str] | None],
    gpu: tuple[int, int, str],
    expected: tuple[str, ...],
    unexpected: tuple[str, ...],
) -> None:
    soc_bus(soc_id)
    tools = FakeHostTools(c2c)
    use_host(monkeypatch, machine, cpuinfo, tools)
    major, minor, name = gpu
    use_gpus(monkeypatch, count=1, major=major, minor=minor, name=name)

    grace_numa.optimize_data_loading_for_grace(gpu_id=0, num_workers=2, verbose=True)
    printed = capsys.readouterr().out

    for text in expected:
        assert text in printed
    for text in unexpected:
        assert text not in printed
    if soc_id != NVIDIA_SOC_ID:
        assert tools.c2c_queries == []
