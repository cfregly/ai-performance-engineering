"""Utilities for Grace-Blackwell NIC/GPU/NUMA topology planning.

These helpers stay read-only by default. They collect NIC metadata, derive CPU
masks that stay local to each adapter, and emit shell snippets for IRQ/RPS/XPS
pinning so the rack-prep scripts remain copy/paste-friendly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


_NIC_PATTERN = re.compile(r"(enp|ens|eth|mlx|ib)")


@dataclass
class NICInfo:
    """Minimal NIC metadata used for affinity planning."""

    name: str
    numa_node: Optional[int]
    local_cpus: List[int]
    irq_ids: List[int]
    rx_queues: int
    tx_queues: int


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _read_int(path: Path) -> Optional[int]:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def parse_cpulist(cpulist: Optional[str]) -> List[int]:
    """Parse cpulist strings such as '0-3,8,10-12' into sorted CPU ids."""
    if not cpulist or cpulist.strip() in {"", "-1"}:
        return []
    cpus: List[int] = []
    for part in cpulist.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            try:
                a, b = int(low), int(high)
            except ValueError:
                continue
            cpus.extend(range(min(a, b), max(a, b) + 1))
        else:
            try:
                cpus.append(int(part))
            except ValueError:
                continue
    return sorted(set(cpus))


def cpulist_to_mask(cpus: Sequence[int]) -> Optional[str]:
    """Return a hex CPU mask (little endian) for sysfs affinity files."""
    if not cpus:
        return None
    mask = 0
    for cpu in cpus:
        if cpu < 0:
            continue
        mask |= 1 << cpu
    return hex(mask)


def format_cpulist(cpus: Sequence[int]) -> str:
    if not cpus:
        return ""
    ranges: List[str] = []
    start = cpus[0]
    prev = start
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = cpu
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def discover_nics(target_names: Optional[Sequence[str]] = None) -> List[NICInfo]:
    """Return NIC metadata for the requested interfaces (or best-effort auto-discovery)."""
    sys_class_net = Path("/sys/class/net")
    names: List[str] = []

    if target_names:
        names.extend([n.strip() for n in target_names if n.strip()])

    if not names and sys_class_net.is_dir():
        names = [
            p.name
            for p in sys_class_net.iterdir()
            if p.is_dir() and p.name != "lo" and _NIC_PATTERN.match(p.name)
        ]

    seen = set()
    unique_names = [n for n in names if not (n in seen or seen.add(n))]

    nics: List[NICInfo] = []
    for name in unique_names:
        nic_dir = sys_class_net / name
        if not nic_dir.exists():
            continue

        numa_node = _read_int(nic_dir / "device" / "numa_node")
        local_cpus = parse_cpulist(_read_text(nic_dir / "device" / "local_cpulist"))

        irq_ids: List[int] = []
        try:
            with Path("/proc/interrupts").open(encoding="utf-8") as interrupts:
                for line in interrupts:
                    if name not in line:
                        continue
                    prefix, _sep, _rest = line.partition(":")
                    try:
                        irq_ids.append(int(prefix.strip()))
                    except ValueError:
                        continue
        except (FileNotFoundError, PermissionError, OSError):
            pass

        queues_dir = nic_dir / "queues"
        if queues_dir.exists():
            rx_queues = sum(1 for _ in queues_dir.glob("rx-*"))
            tx_queues = sum(1 for _ in queues_dir.glob("tx-*"))
        else:
            rx_queues = 0
            tx_queues = 0

        nics.append(
            NICInfo(
                name=name,
                numa_node=numa_node,
                local_cpus=local_cpus,
                irq_ids=irq_ids,
                rx_queues=rx_queues,
                tx_queues=tx_queues,
            )
        )

    return nics


def recommended_cpuset(cpus: Sequence[int], reserve: int = 2) -> List[int]:
    """Keep a small reserve for system daemons and return the worker set."""
    if not cpus:
        return []
    if len(cpus) <= reserve:
        return list(cpus)
    return list(cpus[reserve:])


def render_affinity_block(nic: NICInfo, cpus: Sequence[int]) -> List[str]:
    """Return a shell snippet (no side effects) to pin IRQ/RPS/XPS for a NIC."""
    mask = cpulist_to_mask(cpus)
    if not mask:
        return []
    mask_no_prefix = mask[2:] if mask.startswith("0x") else mask
    cpu_range = format_cpulist(cpus)
    return [
        f"# {nic.name} NUMA {nic.numa_node} cpus {cpu_range} mask {mask}",
        f"for IRQ in $(grep {nic.name} /proc/interrupts | awk -F: '{{print $1}}' | xargs); do",
        f"  printf \"%s\\n\" \"{mask_no_prefix}\" > /proc/irq/$IRQ/smp_affinity",
        "done",
        f"for q in /sys/class/net/{nic.name}/queues/rx-*; do echo {mask_no_prefix} > $q/rps_cpus; echo 32768 > $q/rps_flow_cnt; done",
        f"for q in /sys/class/net/{nic.name}/queues/tx-*; do echo {mask_no_prefix} > $q/xps_cpus; done",
    ]
