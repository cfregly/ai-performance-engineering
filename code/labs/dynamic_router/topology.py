"""Lightweight GPU↔NUMA topology helpers for the dynamic router lab."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TypedDict


class _NvmlGpuInfo(TypedDict):
    bus_id: str | None
    numa_node: int | None


def _read_int(path: Path) -> Optional[int]:
    try:
        value = int(path.read_text().strip())
    except Exception:
        return None
    return value if value >= 0 else None


def _available_numa_nodes() -> List[int]:
    node_root = Path("/sys/devices/system/node")
    nodes = []
    if not node_root.exists():
        return nodes
    for child in node_root.iterdir():
        if child.name.startswith("node") and child.name[4:].isdigit():
            nodes.append(int(child.name[4:]))
    return sorted(nodes)


def _distance_matrix() -> Dict[int, List[int]]:
    matrix: Dict[int, List[int]] = {}
    for node in _available_numa_nodes():
        path = Path(f"/sys/devices/system/node/node{node}/distance")
        if not path.exists():
            continue
        try:
            parts = [int(x) for x in path.read_text().split()]
        except Exception:
            continue
        matrix[node] = parts
    return matrix


def _normalized_bus_id(bus_id: str) -> str:
    # NVML can emit 00000000:17:00.0; sysfs expects 0000:17:00.0
    bus = bus_id.strip().replace("\x00", "").lower()
    if bus.count(":") >= 2:
        domain, bus_number, device = bus.rsplit(":", 2)
        try:
            domain_number = int(domain, 16)
        except ValueError:
            return bus
        if domain_number <= 0xFFFF:
            domain = f"{domain_number:04x}"
        return f"{domain}:{bus_number}:{device}"
    return bus


def _sysfs_numa_for_bus(bus_id: str) -> Optional[int]:
    bus_norm = _normalized_bus_id(bus_id)
    path = Path(f"/sys/bus/pci/devices/{bus_norm}/numa_node")
    if not path.exists():
        return None
    return _read_int(path)


def _cuda_visible_tokens() -> list[str] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return None

    tokens: list[str] = []
    for part in visible.split(","):
        token = part.strip()
        # CUDA stops parsing at an empty or invalid device token. Treat -1 as
        # the conventional explicit no-device mask.
        if not token or token == "-1":
            break
        if not token.isdigit() and not token.startswith(("GPU-", "MIG-")):
            break
        tokens.append(token)
    return tokens


def _cuda_logical_device_uuids(max_gpus: int | None) -> list[object | None] | None:
    """Return CUDA-visible logical device UUIDs, or None without a CUDA runtime."""
    try:
        import torch
    except Exception:
        return None

    try:
        if not torch.cuda.is_available():
            return None
        count = int(torch.cuda.device_count())
    except Exception:
        return None

    limit = count if max_gpus is None else min(count, max(0, max_gpus))
    uuids: list[object | None] = []
    for logical_idx in range(limit):
        try:
            properties = torch.cuda.get_device_properties(logical_idx)
            uuids.append(getattr(properties, "uuid", None))
        except Exception:
            uuids.append(None)
    return uuids


def _nvml_handle_for_logical_device(
    pynvml: object,
    logical_idx: int,
    cuda_uuid: object | None,
    visible_tokens: list[str] | None,
    *,
    allow_unmasked_index: bool,
) -> object | None:
    if cuda_uuid is not None:
        from core.profiling.gpu_telemetry import normalize_gpu_uuid, resolve_nvml_device_handle

        normalized_uuid = normalize_gpu_uuid(cuda_uuid)
        if normalized_uuid is not None:
            return resolve_nvml_device_handle(
                pynvml,
                logical_idx,
                cuda_device_uuid=normalized_uuid,
            )

    if visible_tokens is None:
        if allow_unmasked_index:
            return pynvml.nvmlDeviceGetHandleByIndex(logical_idx)  # type: ignore[attr-defined]
        return None
    if logical_idx >= len(visible_tokens):
        return None

    token = visible_tokens[logical_idx]
    if token.isdigit():
        # Numeric visibility entries are CUDA ordinals. CUDA_DEVICE_ORDER can
        # make those differ from NVML indices, so they are not exact identity.
        return None

    from core.profiling.gpu_telemetry import resolve_nvml_device_handle

    return resolve_nvml_device_handle(
        pynvml,
        logical_idx,
        cuda_device_uuid=token,
    )


def _nvml_info_for_handle(pynvml: object, handle: object) -> _NvmlGpuInfo:
    bus_id: str | None = None
    try:
        pci = pynvml.nvmlDeviceGetPciInfo(handle)  # type: ignore[attr-defined]
        raw_bus_id = getattr(pci, "busId", None)
        if raw_bus_id is not None:
            bus_id = raw_bus_id.decode() if hasattr(raw_bus_id, "decode") else str(raw_bus_id)
    except Exception:
        pass

    numa_id: int | None = None
    try:
        raw_numa_id = pynvml.nvmlDeviceGetNumaNodeId(handle)  # type: ignore[attr-defined]
        if raw_numa_id is not None and int(raw_numa_id) >= 0:
            numa_id = int(raw_numa_id)
    except Exception:
        pass
    return {"bus_id": bus_id, "numa_node": numa_id}


def _nvml_gpu_bus_and_numa(max_gpus: int | None = None) -> dict[int, _NvmlGpuInfo]:
    mapping: dict[int, _NvmlGpuInfo] = {}
    try:
        import pynvml  # type: ignore
    except Exception:
        return mapping
    try:
        pynvml.nvmlInit()
    except Exception:
        return mapping

    try:
        visible_tokens = _cuda_visible_tokens()
        logical_uuids = _cuda_logical_device_uuids(max_gpus)
        if logical_uuids is not None and visible_tokens is not None:
            logical_uuids = logical_uuids[: len(visible_tokens)]

        if logical_uuids is None:
            if visible_tokens is None:
                try:
                    count = int(pynvml.nvmlDeviceGetCount())
                except Exception:
                    count = 0
                limit = count if max_gpus is None else min(count, max(0, max_gpus))
            else:
                limit = len(visible_tokens)
                if max_gpus is not None:
                    limit = min(limit, max(0, max_gpus))
            logical_uuids = [None] * limit
            allow_unmasked_index = visible_tokens is None
        else:
            # CUDA's logical device count is authoritative. Without a UUID or
            # an explicit visibility token, NVML index order is not a safe
            # substitute for logical CUDA identity.
            allow_unmasked_index = False

        for logical_idx, cuda_uuid in enumerate(logical_uuids):
            mapping[logical_idx] = {"bus_id": None, "numa_node": None}
            try:
                handle = _nvml_handle_for_logical_device(
                    pynvml,
                    logical_idx,
                    cuda_uuid,
                    visible_tokens,
                    allow_unmasked_index=allow_unmasked_index,
                )
            except Exception:
                continue
            if handle is not None:
                mapping[logical_idx] = _nvml_info_for_handle(pynvml, handle)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return mapping


@dataclass
class TopologySnapshot:
    gpu_numa: Dict[int, Optional[int]]
    distance: Dict[int, List[int]]
    timestamp: float
    gpu_numa_status: str = "unknown"

    def to_json(self) -> Dict[str, object]:
        return {
            "gpu_numa": self.gpu_numa,
            "distance": self.distance,
            "timestamp": self.timestamp,
            "gpu_numa_status": self.gpu_numa_status,
        }


def _classify_gpu_numa_status(gpu_map: Dict[int, Optional[int]]) -> str:
    if not gpu_map:
        return "unknown"
    known = sum(1 for numa_node in gpu_map.values() if numa_node is not None)
    if known == 0:
        return "unknown"
    if known == len(gpu_map):
        return "complete"
    return "partial"


def detect_topology(max_gpus: Optional[int] = None) -> TopologySnapshot:
    """Best-effort GPU→NUMA map plus NUMA distance matrix."""
    gpu_map: Dict[int, Optional[int]] = {}

    nvml_info = _nvml_gpu_bus_and_numa(max_gpus=max_gpus)
    for idx, info in nvml_info.items():
        numa_guess = info.get("numa_node")
        if numa_guess is None:
            bus = info.get("bus_id")
            if bus:
                numa_guess = _sysfs_numa_for_bus(bus)
        gpu_map[idx] = numa_guess

    # If the caller already knows how many GPUs are in play, preserve that shape
    # even when locality is unavailable.
    if not gpu_map:
        if max_gpus is not None:
            visible_tokens = _cuda_visible_tokens()
            slot_count = max(0, max_gpus)
            if visible_tokens is not None:
                slot_count = min(slot_count, len(visible_tokens))
            for idx in range(slot_count):
                gpu_map[idx] = None
            distance = _distance_matrix()
            return TopologySnapshot(
                gpu_numa=gpu_map,
                distance=distance,
                timestamp=time.time(),
                gpu_numa_status=_classify_gpu_numa_status(gpu_map),
            )
        visible_tokens = _cuda_visible_tokens()
        if visible_tokens is not None:
            for idx in range(len(visible_tokens)):
                gpu_map[idx] = None

    distance = _distance_matrix()
    return TopologySnapshot(
        gpu_numa=gpu_map,
        distance=distance,
        timestamp=time.time(),
        gpu_numa_status=_classify_gpu_numa_status(gpu_map),
    )


def default_topology_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "artifacts" / "topology" / "topology.json"


def write_topology(snapshot: TopologySnapshot, path: Optional[Path] = None) -> Path:
    target = path or default_topology_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(snapshot.to_json(), f, indent=2)
    return target
