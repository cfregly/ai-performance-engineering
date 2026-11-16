"""Store complete contiguous tensors without pickle or dtype conversion."""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import torch

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int32": torch.int32,
    "uint8": torch.uint8,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_file(root: Path, relative: str) -> Path:
    """Resolve a regular bundle member without traversing symlinks."""
    if not isinstance(relative, str):
        raise ValueError("bundle paths must be strings")
    relative_path = Path(relative)
    if (
        not isinstance(relative, str)
        or not relative
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative == "."
    ):
        raise ValueError("bundle paths must be nonempty relative paths without traversal")
    root = Path(root)
    current = root
    if root.is_symlink():
        raise ValueError("bundle roots must not be symlinks")
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"bundle member must not be a symlink: {relative}")
    if not current.is_file():
        raise ValueError(f"bundle member is not a regular file: {relative}")
    return current


def write_tensor(path: Path, tensor: torch.Tensor) -> dict:
    """Write every tensor byte once and return its content descriptor."""
    if not isinstance(tensor, torch.Tensor) or tensor.dtype not in DTYPES.values():
        raise ValueError("unsupported retained tensor dtype")
    if tensor.layout != torch.strided or not tensor.is_contiguous() or tensor.numel() == 0:
        raise ValueError("retained tensors must be nonempty and contiguous")
    cpu = tensor.detach().cpu().contiguous()
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
    return {
        "path": path.name,
        "dtype": str(cpu.dtype).removeprefix("torch."),
        "shape": list(cpu.shape),
        "nbytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "layout": "contiguous",
        "byteorder": sys.byteorder,
    }


def validate_descriptor(descriptor: dict) -> None:
    fields = {"path", "dtype", "shape", "nbytes", "sha256", "layout", "byteorder"}
    if not isinstance(descriptor, dict) or set(descriptor) != fields:
        raise ValueError("tensor descriptor fields do not match the schema")
    if descriptor["dtype"] not in DTYPES or descriptor["layout"] != "contiguous":
        raise ValueError("unsupported tensor dtype or layout")
    if descriptor["byteorder"] != sys.byteorder:
        raise ValueError("tensor byte order does not match this host")
    shape = descriptor["shape"]
    if not isinstance(shape, list) or not shape or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("tensor shapes must contain positive integer dimensions")
    expected = math.prod(shape) * torch.empty((), dtype=DTYPES[descriptor["dtype"]]).element_size()
    if type(descriptor["nbytes"]) is not int or descriptor["nbytes"] != expected:
        raise ValueError("tensor byte count does not match its shape and dtype")
    digest = descriptor["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("tensor SHA-256 must contain 64 lowercase hex digits")
    if not isinstance(descriptor["path"], str):
        raise ValueError("tensor path must be a string")


def read_tensor(root: Path, descriptor: dict, device="cpu") -> torch.Tensor:
    validate_descriptor(descriptor)
    path = safe_file(Path(root), descriptor["path"])
    raw = path.read_bytes()
    if len(raw) != descriptor["nbytes"] or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
        raise ValueError(f"retained tensor hash or byte count mismatch: {descriptor['path']}")
    return (
        torch.frombuffer(bytearray(raw), dtype=DTYPES[descriptor["dtype"]])
        .clone()
        .reshape(descriptor["shape"])
        .to(device)
    )


def load_event_inputs(bundle, event_id: str, device="cpu") -> dict[str, torch.Tensor]:
    event = bundle.event(event_id)
    if event["status"] != "completed":
        raise ValueError(f"source event {event_id} has status {event['status']}")
    return {name: read_tensor(bundle.root, desc, device) for name, desc in event["inputs"].items()}
